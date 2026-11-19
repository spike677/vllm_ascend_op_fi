"""One live projection per independent task, without experimental replay/dump.

The caller supplies *actual* scheduled request boundaries. This module never
calls the model. One normal request binds the first supported weight-owned
return and a real consumer; one subsequent matched request may inject. Native
lossless/apply/restore are the shared postwrite implementation, not a new ABI.
"""
from copy import deepcopy
import hashlib
import os
import struct
import threading

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

from .postwrite import (Fault39OutputProxyProvider, PostwriteRefused, _ObserveDispatch,
                        _storage, _tensors, _hash_inputs)
from .postwrite_boundaries import (PROJECTION_OPERATORS, VALUE_CONSUMERS,
    projection_return, tensor_geometry, reads_selected_elements)


def require(value, reason):
    if not value:
        raise PostwriteRefused(reason)


def _shape(tensor):
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset(),
            str(tensor.dtype), str(tensor.device), tensor.untyped_storage().nbytes())


def _bf16_effect(before, after, indices):
    """Use existing bounded Native readback bytes, not Tensor diagnostics."""
    require(len(before) == len(after) and len(before) % 2 == 0, 'OUTPUT39_READBACK_SIZE_CHANGED')
    left = struct.unpack('<' + 'H' * (len(before)//2), before)
    right = struct.unpack('<' + 'H' * (len(after)//2), after)
    changed = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
    def kind(bits):
        if bits & 0x7f80 != 0x7f80: return 'finite'
        if bits & 0x7f: return 'nan'
        return 'negative_inf' if bits & 0x8000 else 'positive_inf'
    counts = {name: sum(kind(left[i]) == 'finite' and kind(right[i]) == name for i in changed)
              for name in ('positive_inf', 'negative_inf', 'nan')}
    return dict(schema='demoncat.effect.native_bf16_return.v1',
        comparison='SAME_ACTUAL_RETURN_BEFORE_AFTER', elements=len(left),
        selected_elements=len(indices), changed_elements=len(changed),
        changed_only_selected=all(i in indices for i in changed),
        finite_to_positive_inf=counts['positive_inf'], finite_to_negative_inf=counts['negative_inf'],
        finite_to_nan=counts['nan'], before_sha256=hashlib.sha256(before).hexdigest(),
        after_sha256=hashlib.sha256(after).hexdigest(), samples=[dict(flat_index=i,
            before_bf16_hex=f'{left[i]:04x}', after_bf16_hex=f'{right[i]:04x}',
            after_class=kind(right[i])) for i in changed[:4]])


def _consumer_semantics(row):
    # Current addresses must differ across requests. Logical read geometry,
    # the real call ordinal and API/schema/scalars must not drift.
    return dict(operator=row['operator'], schema=row['schema'],
        compute_boundary=row['compute_boundary'], dispatch_ordinal=row['dispatch_ordinal'],
        selected_elements_read=row['selected_elements_read'],
        operand_views=[{key: value[key] for key in ('shape', 'stride', 'storage_offset', 'dtype', 'device')}
                       for value in row['actual_operands']],
        scalars=(row.get('reviewed_consumer') or {}).get('scalars'))


class Output39SingleExecution(Fault39OutputProxyProvider):
    """A spent instance cannot be reset or used for another task.

    Normal examines at most 16 real projections, retaining only one pending
    candidate. The first proven value consumer fixes the target. A displaced
    pending candidate is not a failure or a claim about the full model. After
    binding, neither fault nor OFF may switch targets.
    """
    def __init__(self, model, transport, *, guard, prefix_elements=1, proxy='pos_inf'):
        super().__init__(model, transport, prefix_elements=prefix_elements, proxy=proxy)
        require(callable(guard), 'OUTPUT39_LIVE_OWNER_GUARD_REQUIRED')
        self.guard, self.pid = guard, os.getpid()
        self.normal = self.window = None
        self.normal_output = None
        self.requests = set()
        self.reports = []
        self.failure = None
        self._held_identity = self._identity()
        self._held_forwards = self._forwards()

    def _forwards(self):
        return tuple((path, id(module), name, getattr(getattr(module, name), '__func__', getattr(module, name)),
                      getattr(getattr(getattr(module, name), '__func__', getattr(module, name)), '__code__', None))
            for path, module in self.model.named_modules() for name in ('forward', '_forward_method')
            if hasattr(module, name))

    def context(self, *, full=True):
        require(os.getpid() == self.pid and threading.get_ident() == self._thread,
                'OUTPUT39_EXECUTION_PROCESS_OR_THREAD_CHANGED')
        self.guard()
        require(not self.model.training, 'OUTPUT39_EXECUTION_MODEL_OR_STORAGE_CHANGED')
        if full:
            require(self._identity() == self._held_identity and self._forwards() == self._held_forwards,
                    'OUTPUT39_EXECUTION_MODEL_OR_STORAGE_CHANGED')
        if full and self.normal is not None:
            self.normal['callable'].verify()
            require(getattr(self.normal['module'], 'weight', None) is self.normal['weight'],
                    'OUTPUT39_SELECTED_WEIGHT_OBJECT_CHANGED')

    def begin(self, request_id, phase, args, kwargs):
        self.context()
        require(torch.is_inference_mode_enabled(), 'OUTPUT39_ACTUAL_INFERENCE_REQUIRED')
        require(self.window is None and not self.live_lease_count, 'OUTPUT39_WINDOW_OR_RECOVERY_PENDING')
        require(not _get_current_dispatch_mode_stack(), 'OUTPUT39_EXISTING_DISPATCH_MODE_UNADAPTED')
        require(isinstance(request_id, str) and 0 < len(request_id) <= 256 and request_id not in self.requests,
                'OUTPUT39_FRESH_REQUEST_REQUIRED')
        require(len(self.requests) < 8, 'OUTPUT39_BOUNDED_TASK_REQUESTS_EXHAUSTED')
        require(phase in ('normal', 'fault', 'off'), 'OUTPUT39_UNKNOWN_EXECUTION_PHASE')
        digest = _hash_inputs(args, kwargs)
        if phase == 'normal':
            require(self.normal is None and not self.requests, 'OUTPUT39_NORMAL_BINDING_ONCE_ONLY')
            self._input_hash = digest
        else:
            require(self.normal is not None and digest == self._input_hash,
                    'OUTPUT39_CURRENT_NORMAL_INPUT_BINDING_REQUIRED')
        if phase == 'fault':
            require(not self.failure and not self._fault_attempted, 'OUTPUT39_SINGLE_TASK_ALREADY_SPENT')
            self._fault_attempted = True  # Persist before entering any writer.
        self.requests.add(request_id)
        self.generation += 1
        self._last_native_lease_receipt = self._last_intervention = None
        self.window = _SingleReturnWindow(self, request_id, phase)
        return self.window

    def failed(self, error):
        if self.failure is None:
            self.failure = type(error).__name__ + ': ' + str(error)

    # Explicitly prohibit the inherited experimental entrypoints.
    def qualify(self, *args, **kwargs):
        raise PostwriteRefused('OUTPUT39_ACTUAL_REQUEST_BOUNDARY_ONLY')

    def run(self, *args, **kwargs):
        raise PostwriteRefused('OUTPUT39_ACTUAL_REQUEST_BOUNDARY_ONLY')


class _SingleReturnWindow(TorchDispatchMode):
    def __init__(self, execution, request_id, phase):
        super().__init__()
        self.p, self.request_id, self.phase = execution, request_id, phase
        self.active = False
        self.candidate = self.effect = self.before = None
        self.consumers = []
        self.counts = {}
        self.gemm_binding = None
        self.entered = False
        self.report = None
        self.mode = None
        self.in_compute = True
        self.inspected_projections = 0
        self.dispatch_ordinal = 0
        self.compute_boundary = 'root'

    def start(self):
        self.p.context()
        require(not self.entered, 'OUTPUT39_WINDOW_CANNOT_REENTER')
        self.entered = True
        try:
            self.__enter__()
        finally:
            self.active = any(mode is self for mode in _get_current_dispatch_mode_stack())
            self.mode = self if self.active else None

    def _owner(self, operands):
        # Automatic selection uses a real weight operand/storage. No user path
        # or global module census is accepted as a target receipt.
        matches = {}
        for path, module in self.p.model.named_modules():
            weight = getattr(module, 'weight', None)
            if isinstance(weight, torch.Tensor) and any(_storage(t) == _storage(weight) for t in operands):
                matches[id(module)] = (path, module, weight)
        return next(iter(matches.values())) if len(matches) == 1 else None

    def _read(self, tensor):
        read = getattr(self.p.transport, 'snapshot', None)
        require(callable(read), 'OUTPUT39_NATIVE_READ_SPAN_REQUIRED')
        result = read(tensor)
        require(type(result) is bytes and len(result) == tensor.numel()*2,
                'OUTPUT39_COMPLETE_BF16_RETURN_READ_REQUIRED')
        return result

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        p = self.p
        require(self.active and p.window is self, 'OUTPUT39_LATE_OR_FOREIGN_WINDOW')
        p.context(full=False)
        operands = _tensors((args, kwargs))
        candidate = self.candidate
        if candidate is not None and _ObserveDispatch._writes(func, args, kwargs, candidate['storage']):
            if self.phase == 'normal' and not self.consumers and p._lease is None:
                self.candidate = candidate = self.before = None
            else:
                raise PostwriteRefused('WRITE_AFTER_SELECTED_WRITER_REFUSED')
        result = func(*args, **kwargs)  # Exactly one actual original dispatch.
        if not self.in_compute:
            return result
        self.dispatch_ordinal += 1
        operation = str(func)
        if candidate is not None:
            from .postwrite_npu_consumers import NPU_VALUE_CONSUMERS, npu_value_consumer
            proof, reads = None, operands
            if operation in NPU_VALUE_CONSUMERS:
                proof, reads, _ = npu_value_consumer(func, args, kwargs, result)
            if operation in VALUE_CONSUMERS or proof is not None:
                used = [t for t in reads if reads_selected_elements(t, candidate['indices'], candidate['tensor'])]
                if used and not self.consumers:
                    self.consumers.append(dict(operator=operation, schema=str(func._schema),
                        compute_boundary=self.compute_boundary, dispatch_ordinal=self.dispatch_ordinal,
                        selected_elements_read=list(candidate['indices']), original_returned=True,
                        actual_operands=[tensor_geometry(t) for t in used[:4]], reviewed_consumer=proof))
                    if self.phase == 'normal':
                        require(self._read(candidate['tensor']) == self.before,
                                'OUTPUT39_NORMAL_VALUE_CHANGED_BEFORE_BINDING')
                        changed = p._apply_single_return(candidate['tensor'], candidate['indices'], 0)
                        self.effect = _bf16_effect(self.before, self._read(candidate['tensor']), candidate['indices'])
                        require(changed == self.effect['changed_elements'] == 0,
                                'OUTPUT39_NATIVE_LOSSLESS_RETURN_CHANGED')
            if self.phase != 'normal' or self.consumers:
                return result
        if operation not in PROJECTION_OPERATORS:
            return result
        self.inspected_projections += 1
        require(self.inspected_projections <= 16, 'OUTPUT39_BOUNDED_PROJECTION_DISCOVERY_EXHAUSTED')
        values = dict(kwargs)
        values.update((parameter.name, value) for parameter, value in zip(func._schema.arguments, args))
        weight_key = ('weight' if operation in ('aten.linear.default', 'vllm.unquantized_gemm.default')
                      else 'mat2' if operation == 'aten.mm.default' else 'other')
        owner = self._owner(_tensors(values.get(weight_key)))
        if owner is None:
            return result
        path, module, weight = owner
        occurrence_key = (id(module), operation)
        occurrence = self.counts.get(occurrence_key, 0)
        self.counts[occurrence_key] = occurrence+1
        key = (id(module), operation, occurrence)
        if p.normal is not None and key != p.normal['key']:
            return result
        if operation == 'vllm.unquantized_gemm.default' and self.gemm_binding is None:
            from demoncat.runtime.vllm_ascend.worker_extension.gemm_lineage import ReviewedGemmBinding
            self.gemm_binding = ReviewedGemmBinding()
        proof, reason = projection_return(func, args, kwargs, result,
            gemm_binding=self.gemm_binding, expected_weight=weight)
        if proof is None:
            require(p.normal is None, 'OUTPUT39_SELECTED_PRODUCER_CHANGED:' + str(reason))
            return result
        try:
            p.transport.validate(result)
        except PostwriteRefused:
            if p.normal is not None: raise
            return result
        require(result.numel() >= p.prefix_elements, 'OUTPUT39_SELECTED_RETURN_TOO_SMALL')
        from fault_injection.tensor_callable import TensorCallableGuard
        guard = TensorCallableGuard(module) if p.normal is None else p.normal['callable']
        guard.verify()
        if p.normal is not None:
            require(_shape(result) == p.normal['geometry'] and weight is p.normal['weight'],
                    'OUTPUT39_SELECTED_RETURN_OR_WEIGHT_CHANGED')
            require((self.compute_boundary, self.dispatch_ordinal) == p.normal['producer_position'],
                    'OUTPUT39_SELECTED_NORMAL_POSITION_CHANGED')
            prior = p.normal['consumer'].get('reviewed_consumer')
            if prior is not None:
                from .postwrite_npu_consumers import npu_consumer_registration
                function = getattr(torch.ops.npu, prior['operator'].split('.')[1]).default
                current, reason = npu_consumer_registration(function)
                require(not reason and all(prior.get(name) == value for name, value in (current or {}).items()),
                        'OUTPUT39_CURRENT_NATIVE_CONSUMER_REGISTRATION_CHANGED')
        self.candidate = dict(key=key, module=module, weight=weight, path=path, callable=guard,
            geometry=_shape(result), tensor=result, storage=_storage(result),
            producer_position=(self.compute_boundary, self.dispatch_ordinal),
            indices=tuple(range(p.prefix_elements)), proof=proof)
        self.before = self._read(result)
        if p.normal is not None:
            require(hashlib.sha256(self.before).hexdigest() == p.normal['before_sha256'],
                    'OUTPUT39_SELECTED_NORMAL_VALUES_CHANGED')
        if self.phase == 'fault':
            changed = p._apply_single_return(result, self.candidate['indices'], p.mutation_mode)
            self.effect = _bf16_effect(self.before, self._read(result), self.candidate['indices'])
            expected = p.prefix_elements
            require(changed == expected == self.effect['changed_elements'] and self.effect['changed_only_selected'],
                    'OUTPUT39_NATIVE_SAME_RETURN_EFFECT_UNPROVEN')
        return result

    def close(self):
        self.p.context()
        if self.active:
            from fault_injection._scoped_dispatch import _close_owned_mode
            try:
                _close_owned_mode(self, 'mode', 'OUTPUT39_FOREIGN_DISPATCH_MODE_ON_TOP')
            finally:
                self.active = self.mode is not None
        self.p.recover()  # Failed restore keeps the original owned lease.
        self.p.window = None

    def finish(self, output_sha256):
        p = self.p
        try:
            require(self.candidate is not None and self.consumers,
                    'OUTPUT39_CURRENT_WRITER_OR_VALUE_CONSUMER_NOT_OBSERVED')
            p.context()
            self.close()
            candidate = self.candidate
            restored = self._read(candidate['tensor'])
            require(restored == self.before, 'OUTPUT39_EXACT_STORAGE_RESTORE_UNPROVEN')
            if self.phase == 'normal':
                p.normal = {key: value for key, value in candidate.items()
                            if key not in ('tensor', 'storage', 'proof', 'indices')}
                p.normal['before_sha256'] = hashlib.sha256(self.before).hexdigest()
                p.normal['consumer'] = deepcopy(self.consumers[0])
                p.normal_output = output_sha256
            else:
                require(_consumer_semantics(self.consumers[0]) == _consumer_semantics(p.normal['consumer']),
                        'OUTPUT39_CURRENT_CONSUMER_CHANGED')
            if self.phase == 'off':
                require(output_sha256 == p.normal_output, 'OUTPUT39_MATCHED_OFF_OUTPUT_CHANGED')
            self.report = dict(request_id=self.request_id, phase=self.phase,
                inspected_projections=self.inspected_projections, discovery_limit=16,
                target_binding='ACTUAL_WEIGHT_OPERAND_OWNER_AND_SELECTED_VALUE_CONSUMER',
                native_lossless_timing='AFTER_NORMAL_ACTUAL_CONSUMER_BEFORE_REQUEST_RETURN',
                target=dict(module_path=candidate['path'], module_object=id(candidate['module']),
                    operator=candidate['key'][1], ordinal=candidate['key'][2],
                    returned_storage=tensor_geometry(candidate['tensor']), producer=candidate['proof']),
                consumers=deepcopy(self.consumers), effect=deepcopy(self.effect),
                applied=int(self.phase == 'fault'), recovered_storage_exact=True,
                output_sha256=output_sha256, exact_normal_output=output_sha256 == p.normal_output,
                native_lease_receipt=deepcopy(p._last_native_lease_receipt),
                c1=False, c2=False, c3=False, E5=False)
            p.reports.append(self.report)
            return deepcopy(self.report)
        except BaseException as error:
            p.failed(error)
            raise
