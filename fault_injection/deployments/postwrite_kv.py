"""Fault41's bounded live-cache slot boundary, separate from Fault33 tiling.

The proxy changes legal BF16 tail columns only after an actual cache writer and
before its actual attention consumer. It does not claim an out-of-bounds write.
"""
from dataclasses import dataclass
import threading

import torch

from .postwrite import PostwriteRefused, _storage, _validate_span
from .postwrite_boundaries import tensor_geometry


@dataclass(frozen=True)
class KVSlotPlan:
    cache: torch.Tensor
    element_offset: int
    elements: int
    indices: tuple[int, ...]
    proof: dict


def live_value_tail_plan(*, operator, actual, impl, metadata, scheduler, writer,
                         tail_elements=2, allow_cpu_test=False):
    from demoncat.runtime.vllm_ascend.worker_extension.paged33_audit import audit_paged_call
    if operator != 'npu_fused_infer_attention_score' or actual.get('input_layout') != 'TND':
        raise PostwriteRefused('TAIL_COLUMN_REQUIRES_ACTUAL_TND_FIA_CONSUMER')
    if type(tail_elements) is not int or not 1 <= tail_elements <= 128:
        raise PostwriteRefused('TAIL_COLUMN_SIZE_NOT_REVIEWED')
    proof = audit_paged_call(operator, actual, impl=impl, metadata=metadata,
                            scheduler=scheduler, writer=writer, allow_cpu_test=allow_cpu_test)
    cache = impl.value_cache
    if _storage(actual['query'])[:2] == _storage(cache)[:2]:
        raise PostwriteRefused('TAIL_COLUMN_QUERY_ALIASES_MUTATED_CACHE')
    if cache.dtype != torch.bfloat16 or len(proof['slots']) != 1:
        raise PostwriteRefused('TAIL_COLUMN_BF16_SINGLE_ACTUAL_SLOT_REQUIRED')
    blocks, page, heads, columns = map(int, cache.shape)
    if tail_elements > columns:
        raise PostwriteRefused('TAIL_COLUMN_OUTSIDE_ACTUAL_HEAD')
    elements = heads * columns
    offset = proof['slots'][0] * elements
    _validate_span(cache, offset, elements)
    indices = tuple(head * columns + column for head in range(heads)
                    for column in range(columns - tail_elements, columns))
    return KVSlotPlan(cache, offset, elements, indices,
        dict(boundary='actual_kv_writer_to_live_value_slot_to_tnd_fia',
             controlled_legal_address_proxy=True, native_out_of_bounds_claim=False,
             observational_metadata=proof, cache=tensor_geometry(cache),
             storage=_storage(cache), slot=proof['slots'][0], cache_kind='value',
             span_element_offset=offset, span_elements=elements,
             relative_indices=list(indices),
             allocation_element_indices=[cache.storage_offset() + offset + i for i in indices]))


class LiveKVSlotLease:
    """Hold the real allocation until its actual consumer has drained/restored."""
    def __init__(self, transport, plan, owner, generation):
        self.transport, self.plan = transport, plan
        self.owner, self.generation = owner, generation
        self.thread = threading.get_ident()
        self.native = None
        self.consumer_calls = 0
        self.consuming = False
        self.arm_attempted = False
        self.stopped = False
        self.report = dict(owner=owner, generation=generation, boundary=plan.proof,
            native_applied=False, original_calls=0, consumer_returned=False,
            restored=False, released=False)
        self.native = transport.open_span(plan.cache, plan.element_offset, plan.elements,
                                          owner, generation)

    def arm(self, *, fault):
        # The caller retains this object before apply, including on failure.
        if threading.get_ident() != self.thread:
            raise PostwriteRefused('TAIL_COLUMN_LEASE_OWNER_REQUIRED')
        if self.stopped or self.native is None or self.arm_attempted:
            raise PostwriteRefused('TAIL_COLUMN_ARM_STATE_MISMATCH')
        self.arm_attempted = True
        changed = self.native.apply(4 if fault else 0, self.plan.indices)
        self.report.update(native_applied=fault, changed_elements=changed,
                           selected_elements=len(self.plan.indices), readback_exact=True)
        if fault and changed != len(self.plan.indices):
            raise PostwriteRefused('TAIL_COLUMN_NATIVE_EFFECT_COUNT_MISMATCH')

    def consume(self, original, args, kwargs):
        if (self.stopped or self.consumer_calls or 'readback_exact' not in self.report
                or threading.get_ident() != self.thread):
            raise PostwriteRefused('TAIL_COLUMN_CONSUMER_ONCE_OWNER_REQUIRED')
        self.consumer_calls += 1
        self.consuming = True
        self.report['original_calls'] = 1
        failure = None
        try:
            result = original(*args, **kwargs)
            self.report['consumer_returned'] = True
            return result
        except BaseException as error:
            failure = error
            self.report['consumer_error'] = repr(error)
            raise
        finally:
            self.consuming = False
            self.stopped = True
            # Native restore synchronizes the actual device before checking or
            # restoring bytes. The consumer has already been called exactly once.
            try:self.recover()
            except BaseException as error:
                self.report['recovery_error'] = repr(error)
                if failure is None:raise

    def recover(self):
        if threading.get_ident() != self.thread:
            raise PostwriteRefused('TAIL_COLUMN_LEASE_OWNER_REQUIRED')
        if self.consuming:
            raise PostwriteRefused('TAIL_COLUMN_CONSUMER_WINDOW_ACTIVE')
        self.stopped = True
        if self.native is not None:
            self.native.recover()
            if self.native.live():
                raise PostwriteRefused('TAIL_COLUMN_NATIVE_LEASE_REMAINS')
            self.native = None
            self.report.update(restored=True, released=True)

    @property
    def live(self):
        return self.native is not None
