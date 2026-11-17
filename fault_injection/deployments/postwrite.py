"""Bounded post-operator proxies with actual dispatch and storage observations.

This is a new public Tensor/operator boundary. It does not inherit the legacy
OpCache output-proxy certificate or claim a naturally occurring cache overwrite.
Only an independent normal model replay can discover a target. No layer name,
pointer, claimed writer receipt, or claimed consumer receipt is accepted.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import threading

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map


class PostwriteRefused(RuntimeError):
    pass


def _storage(tensor):
    return (str(tensor.device), tensor.untyped_storage()._cdata,
            tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())


def _tensors(value):
    return [item for item in tree_flatten(value)[0] if isinstance(item, torch.Tensor)]


def _cpu(value):
    return tree_map(lambda x: x.detach().cpu().clone() if isinstance(x, torch.Tensor) else x, value)


def _equal(left, right):
    a, sa = tree_flatten(left)
    b, sb = tree_flatten(right)
    return sa == sb and all(torch.equal(x, y) if isinstance(x, torch.Tensor)
                           and isinstance(y, torch.Tensor) else x == y for x, y in zip(a, b))


def _hash_inputs(args, kwargs):
    flattened, spec = tree_flatten((args, kwargs))
    digest = hashlib.sha256(str(spec).encode())
    for value in flattened:
        if isinstance(value, torch.Tensor):
            digest.update(str((tuple(value.shape), str(value.dtype), str(value.device))).encode())
            digest.update(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        elif isinstance(value, (str, int, float, bool, type(None))):
            digest.update(json.dumps(value, allow_nan=False).encode())
        else:
            raise PostwriteRefused('UNSUPPORTED_REPLAY_INPUT')
    return digest.hexdigest()


class NativePostwriteLibrary:
    """Deployment-owned public ACL ABI. Loading never confers model acceptance."""
    transport_kind = 'public_acl_native'

    def __init__(self, library, sha256):
        path = Path(library).resolve(strict=True)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != sha256 or len(sha256) != 64:
            raise PostwriteRefused('POSTWRITE_BUILD_HASH_MISMATCH')
        self.sha256 = actual
        self.path = str(path)
        self.lib = ctypes.CDLL(str(path))
        u64, u32, i32, ptr = ctypes.c_uint64, ctypes.c_uint32, ctypes.c_int32, ctypes.c_void_p
        signatures = {
            'dcat_pw_abi': ([], u32),
            'dcat_pw_open': ([ptr, u64, i32, u64, u64, ctypes.POINTER(u64)], ctypes.c_int),
            'dcat_pw_apply': ([u64, u64, u64, u32, ctypes.POINTER(u64), u64, ctypes.POINTER(u64)], ctypes.c_int),
            'dcat_pw_restore': ([u64, u64, u64], ctypes.c_int),
            'dcat_pw_release': ([u64, u64, u64], ctypes.c_int),
            'dcat_pw_live': ([u64, u64, u64], ctypes.c_int),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, result
        if self.lib.dcat_pw_abi() != 1:
            raise PostwriteRefused('POSTWRITE_ABI_MISMATCH')
        self._verify_loaded_file(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != actual:
            raise PostwriteRefused('POSTWRITE_LOADED_BYTES_CHANGED')

    @staticmethod
    def _verify_loaded_file(path):
        # A successful CDLL lookup can reuse an older mapping with the same
        # filename. Bind this process's mapped inode as well as on-disk bytes.
        maps = Path('/proc/self/maps')
        if not maps.is_file():
            raise PostwriteRefused('POSTWRITE_LINUX_LOADED_IDENTITY_REQUIRED')
        stat = path.stat()
        expected_device = f'{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}'
        matching = []
        for line in maps.read_text().splitlines():
            fields = line.split(None, 5)
            if len(fields) == 6 and fields[5] == str(path):
                matching.append(fields)
        if not matching or any(int(row[4]) != stat.st_ino or row[3] != expected_device for row in matching):
            raise PostwriteRefused('POSTWRITE_LOADED_INODE_MISMATCH')

    def validate(self, tensor):
        if tensor.device.type != 'npu' or tensor.dtype != torch.bfloat16:
            raise PostwriteRefused('POSTWRITE_REQUIRES_NPU_BF16')
        import torch_npu
        if int(torch_npu.get_npu_format(tensor)) not in (0, 2):
            raise PostwriteRefused('POSTWRITE_REQUIRES_BASE_FORMAT')
        _validate_tensor(tensor)

    def open(self, tensor, owner, generation):
        self.validate(tensor)
        return _NativeLease(self, tensor, owner, generation)


def _validate_tensor(tensor):
    if (tensor.requires_grad or not tensor.is_contiguous() or tensor.storage_offset() != 0
            or tensor.numel() <= 0 or tensor.numel() * tensor.element_size() > 1024 * 1024
            or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()):
        raise PostwriteRefused('POSTWRITE_REQUIRES_EXCLUSIVE_CONTIGUOUS_BOUNDED_STORAGE')


class _NativeLease:
    def __init__(self, library, tensor, owner, generation):
        self.library, self.tensor = library, tensor  # Strong reference until exact restoration.
        self.owner, self.generation = owner, generation
        self.storage = _storage(tensor)
        self.handle = ctypes.c_uint64()
        code = library.lib.dcat_pw_open(tensor.data_ptr(), tensor.numel() * tensor.element_size(),
            tensor.device.index, owner, generation, ctypes.byref(self.handle))
        self.closed = False
        if code:
            raise PostwriteRefused(f'POSTWRITE_OPEN_{code}')

    def _check(self):
        if self.closed or _storage(self.tensor) != self.storage:
            raise PostwriteRefused('POSTWRITE_LIVE_STORAGE_CHANGED')

    def apply(self, mode, indices):
        self._check()
        selected = (ctypes.c_uint64 * len(indices))(*indices)
        changed = ctypes.c_uint64()
        code = self.library.lib.dcat_pw_apply(self.handle, self.owner, self.generation,
            mode, selected, len(indices), ctypes.byref(changed))
        if code:
            raise PostwriteRefused(f'POSTWRITE_APPLY_{code}')
        return changed.value

    def recover(self):
        if self.closed:
            return
        self._check()
        code = self.library.lib.dcat_pw_restore(self.handle, self.owner, self.generation)
        if code:
            raise PostwriteRefused(f'POSTWRITE_RESTORE_{code}')
        code = self.library.lib.dcat_pw_release(self.handle, self.owner, self.generation)
        if code:
            raise PostwriteRefused(f'POSTWRITE_RELEASE_{code}')
        self.closed = True

    def live(self):
        return self.library.lib.dcat_pw_live(self.handle, self.owner, self.generation) == 1


@dataclass
class _Candidate:
    key: tuple
    tensor: torch.Tensor
    indices: tuple[int, ...]
    operator: str
    module_path: str
    storage: tuple
    consumers: list


class _ObserveDispatch(TorchDispatchMode):
    # Views and alias-only dispatches do not prove a value was consumed.
    VALUE_CONSUMERS = frozenset({'aten.add.Tensor', 'aten.mul.Tensor', 'aten.mm.default',
                                'aten.bmm.default', 'aten.addmm.default', 'aten.sum.dim_IntList'})
    def __init__(self, provider, phase):
        super().__init__()
        self.provider, self.phase = provider, phase
        self.candidates = []
        self.counts = {}
        self.original_calls = 0
        self.operator_counts = {}
        self.projection_observations = []
        self.rejections = []

    @staticmethod
    def _writes(func, args, kwargs, storage):
        for i, parameter in enumerate(func._schema.arguments):
            value = args[i] if i < len(args) else kwargs.get(parameter.name)
            if (parameter.alias_info and parameter.alias_info.is_write and
                    any(_storage(t) == storage for t in _tensors(value))):
                return True
        return False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        p = self.provider
        if not p._stack:
            return func(*args, **kwargs)
        operation = str(func)
        operands = _tensors((args, kwargs))
        read_storages = {_storage(t) for t in operands}
        previous = [c for c in self.candidates if c.storage in read_storages]
        if p._lease is not None and self._writes(func, args, kwargs, p._lease.storage):
            raise PostwriteRefused('WRITE_AFTER_SELECTED_WRITER_REFUSED')
        result = func(*args, **kwargs)  # Exactly one real original operation.
        self.original_calls += 1
        self.operator_counts[operation] = self.operator_counts.get(operation, 0) + 1
        if operation in ('aten.mm.default', 'aten.addmm.default', 'aten.linear.default',
                         'aten.matmul.default', 'npu.npu_linear.default') and len(self.projection_observations) < 32:
            outputs = _tensors(result)
            self.projection_observations.append(dict(operator=operation, module_path=p._stack[-1][0],
                admitted_producer_operator=operation == 'aten.mm.default',
                inputs=[dict(shape=list(t.shape), dtype=str(t.dtype)) for t in operands],
                outputs=[dict(shape=list(t.shape), dtype=str(t.dtype), contiguous=t.is_contiguous(),
                    offset=t.storage_offset(), storage_bytes=t.untyped_storage().nbytes(),
                    logical_bytes=t.numel() * t.element_size(), requires_grad=t.requires_grad,
                    aliases_input=_storage(t) in read_storages) for t in outputs]))
        for candidate in previous:
            if (operation in self.VALUE_CONSUMERS
                    and not self._writes(func, args, kwargs, candidate.storage)):
                candidate.consumers.append(operation)
        tensor, indices = None, ()
        if p.mechanism == '39.output_proxy' and operation == 'aten.mm.default':
            if (len(args) >= 2 and all(isinstance(t, torch.Tensor) and t.ndim == 2
                    and t.dtype == torch.bfloat16 for t in args[:2])
                    and isinstance(result, torch.Tensor) and result.dtype == torch.bfloat16
                    and result.ndim == 2 and result.numel() >= p.prefix_elements
                    and _storage(result) not in read_storages):
                tensor = result
                indices = tuple(range(p.prefix_elements))
        elif p.mechanism == '41.tail_column' and operation == 'aten.index_copy_.default':
            if len(args) >= 4:
                cache, dim, rows, source = args[:4]
                if (isinstance(cache, torch.Tensor) and id(cache) in p._buffers
                        and p._buffers[id(cache)] is cache and cache.ndim == 2 and dim == 0
                        and cache.dtype == torch.bfloat16 and cache.shape[1] >= p.tail_elements
                        and isinstance(rows, torch.Tensor) and rows.dtype == torch.int64
                        and rows.numel() == 1 and source.shape == (1, cache.shape[1])):
                    slot = int(rows.cpu().item())
                    if 0 <= slot < cache.shape[0]:
                        tensor = cache
                        indices = tuple(slot * cache.shape[1] + col
                            for col in range(cache.shape[1] - p.tail_elements, cache.shape[1]))
        if tensor is None:
            return result
        try:
            p.transport.validate(tensor)
        except PostwriteRefused as error:
            p._rejections.append(str(error))
            self.rejections.append(dict(reason=str(error), operator=operation, module_path=p._stack[-1][0],
                                        shape=list(tensor.shape), dtype=str(tensor.dtype)))
            return result
        module_path, module = p._stack[-1]
        signature = (id(module), operation, tuple(tensor.shape), tuple(tensor.stride()),
                     str(tensor.dtype), indices)
        occurrence = self.counts.get(signature, 0)
        self.counts[signature] = occurrence + 1
        candidate = _Candidate(signature + (occurrence,), tensor, indices, operation,
                               module_path, _storage(tensor), [])
        self.candidates.append(candidate)
        if self.phase in ('baseline', 'fault') and candidate.key == p._target:
            if p._lease is not None:
                raise PostwriteRefused('ONLY_ONE_TARGET_CALL_PER_REPLAY')
            p._lease = p.transport.open(tensor, p.owner, p.generation)
            mode = p.mutation_mode if self.phase == 'fault' else 0
            changed = p._lease.apply(mode, indices)
            p._last_intervention = dict(operator=operation, original_executed=True,
                selected_elements=len(indices), changed_elements=changed, readback_exact=True,
                boundary='post_operator_tensor_output' if p.mechanism.startswith('39.')
                         else 'post_cache_writer_live_slot', transport=p.transport.transport_kind,
                new_fault_hits=int(self.phase == 'fault'), module_path=module_path)
        return result


class PostwriteProvider:
    """Single automatically discovered, lossless-qualified inference target.

    ``qualify`` runs two real unchanged forwards. ``run`` always drives the
    actual bound model. Runtime hooks cannot manufacture a writer/consumer
    receipt. Full Worker integration and all-eligible modes are not granted.
    """
    def __init__(self, model, transport, mechanism, *, prefix_elements=1,
                 tail_elements=1, proxy='pos_inf'):
        if mechanism not in ('39.output_proxy', '41.tail_column'):
            raise ValueError('Separate output_proxy or tail_column mechanism required')
        if not isinstance(transport, NativePostwriteLibrary) and getattr(transport, 'transport_kind', None) != 'cpu_test':
            raise TypeError('A verified native transport is required')
        if type(prefix_elements) is not int or prefix_elements not in (1, 5, 51, 512):
            raise ValueError('Reviewed prefix sizes are 1, 5, 51 or 512')
        if type(tail_elements) is not int or not 1 <= tail_elements <= 128:
            raise ValueError('Tail must contain 1 through 128 logical columns')
        if proxy not in ('pos_inf', 'neg_inf', 'qnan'):
            raise ValueError('Unknown output proxy')
        self.model, self.transport, self.mechanism = model, transport, mechanism
        self.prefix_elements, self.tail_elements = prefix_elements, tail_elements
        self.mutation_mode = {'pos_inf': 1, 'neg_inf': 2, 'qnan': 3}[proxy] if mechanism.startswith('39.') else 4
        self.owner, self.generation = secrets.randbits(63) + 1, 0
        self._thread = threading.get_ident()
        self._stack, self._rejections, self._history = [], [], []
        self._lease = self._target = self._input_hash = self._model_identity = None
        self._last_intervention = None
        self._buffers = {id(t): t for t in model.buffers()}
        self._lock = threading.Lock()
        self._qualified = self._lossless = self._fault_attempted = False
        self._qualification_diagnostics = None

    def _identity(self):
        return (id(self.model), tuple((id(m), type(m).__module__, type(m).__qualname__)
                                     for m in self.model.modules()),
                tuple((id(p), _storage(p), p._version) for p in self.model.parameters()),
                tuple((id(b), _storage(b)) for b in self.model.buffers()))

    def _replay(self, phase, args, kwargs, model_observation=None):
        if threading.get_ident() != self._thread or not self._lock.acquire(False):
            raise PostwriteRefused('POSTWRITE_SYNCHRONOUS_OWNER_REQUIRED')
        hooks = []
        observer = _ObserveDispatch(self, phase)
        self._last_intervention = None
        try:
            if self._lease is not None:
                raise PostwriteRefused('POSTWRITE_RECOVERY_REQUIRED')
            for name, module in self.model.named_modules():
                def pre(current, inputs, name=name):
                    self._stack.append((name, current))
                def post(current, inputs, output):
                    if not self._stack or self._stack[-1][1] is not current:
                        raise PostwriteRefused('MODULE_CALL_STACK_MISMATCH')
                    self._stack.pop()
                hooks.append(module.register_forward_pre_hook(pre))
                hooks.append(module.register_forward_hook(post, always_call=True))
            # Start collection only after replay identity checks and hook
            # installation. Its context ends before CPU copies, comparisons
            # and native recovery, preserving the actual model call boundary.
            from contextlib import nullcontext
            window = nullcontext() if model_observation is None else model_observation(self.model)
            with window:
                with torch.inference_mode(), observer:
                    output = self.model(*args, **kwargs)
            copied = _cpu(output)
            candidates = [c for c in observer.candidates if c.consumers]
            selected = [c for c in candidates if c.key == self._target]
            if phase in ('baseline', 'fault') and (len(selected) != 1 or self._lease is None):
                raise PostwriteRefused('QUALIFIED_WRITER_OR_CONSUMER_NOT_OBSERVED')
            if phase == 'off' and (len(selected) != 1 or self._lease is not None):
                raise PostwriteRefused('OFF_TARGET_OR_DRAIN_NOT_OBSERVED')
            report = dict(phase=phase, mechanism=self.mechanism,
                transport=self.transport.transport_kind, original_operator_calls=observer.original_calls,
                observed_candidates=len(observer.candidates), eligible_with_consumers=len(candidates),
                intervention=self._last_intervention,
                consumers=selected[0].consumers if selected else [],
                rejections=list(self._rejections), scope='controlled single-target inference replay',
                native_acceptance_inherited=False, public_worker_qualified=False,
                all_eligible_qualified=False, every_match_qualified=False)
            report['discovery_diagnostics'] = dict(operator_counts=dict(sorted(observer.operator_counts.items())),
                projection_observations=observer.projection_observations, projection_observation_limit=32,
                transport_rejections=observer.rejections,
                candidates=[dict(module_path=c.module_path, operator=c.operator,
                    shape=list(c.tensor.shape), dtype=str(c.tensor.dtype), consumers=list(c.consumers))
                    for c in observer.candidates[:64]], candidate_observation_limit=64)
            if selected:
                chosen = selected[0]
                report['actual_target'] = dict(pid=os.getpid(), model_instance=id(self.model),
                    owner=self.owner, generation=self.generation, operator=chosen.operator,
                    module_path=chosen.module_path, storage_identity=chosen.storage,
                    shape=list(chosen.tensor.shape), stride=list(chosen.tensor.stride()),
                    storage_offset=chosen.tensor.storage_offset(), dtype=str(chosen.tensor.dtype),
                    logical_indices=list(chosen.indices), consumer_calls=list(chosen.consumers))
            return copied, candidates, report
        finally:
            for hook in reversed(hooks):
                hook.remove()
            self._stack.clear()
            try:
                self.recover()
            finally:
                self._lock.release()

    def qualify(self, *args, **kwargs):
        if self._qualified or self._fault_attempted:
            raise PostwriteRefused('NEW_PROVIDER_REQUIRED_FOR_NEW_QUALIFICATION')
        identity, input_hash = self._identity(), _hash_inputs(args, kwargs)
        self._qualification_diagnostics = dict(status='IN_PROGRESS', observations=[], guards_relaxed=False)
        left, first, first_report = self._replay('observe', args, kwargs)
        self._qualification_diagnostics['observations'].append(first_report)
        right, second, second_report = self._replay('observe', args, kwargs)
        self._qualification_diagnostics['observations'].append(second_report)
        keys_equal = [c.key for c in first] == [c.key for c in second]
        outputs_equal = _equal(left, right)
        identity_equal = self._identity() == identity
        failures = []
        if not first:failures.append('NO_ELIGIBLE_WRITER_WITH_OBSERVED_VALUE_CONSUMER')
        if not keys_equal:failures.append('ACTUAL_CANDIDATE_SEQUENCE_CHANGED')
        if not outputs_equal:failures.append('UNCHANGED_REPLAYS_OUTPUT_NOT_EXACT')
        if not identity_equal:failures.append('MODEL_BINDING_CHANGED_DURING_REPLAY')
        self._qualification_diagnostics.update(status='REFUSED' if failures else 'QUALIFIED',
            failure_reasons=failures, candidate_sequences_equal=keys_equal,
            complete_original_outputs_equal=outputs_equal, model_identity_equal=identity_equal)
        if failures:
            raise PostwriteRefused('NO_STABLE_ORIGINAL_WRITER_CONSUMER_REPLAY:' + ','.join(failures))
        self._target = first[0].key
        self._input_hash, self._model_identity = input_hash, identity
        self._qualified = True
        self._baseline_output = left
        return dict(status='ACTUAL_CALL_CANDIDATE_SELECTED', count=len(first),
                    selection='first eligible in stable actual execution order',
                    operator=first[0].operator, module_path=first[0].module_path,
                    consumers=first[0].consumers, input_sha256=input_hash,
                    requires_lossless=True)

    def run(self, phase, *args, model_observation=None, **kwargs):
        if phase not in ('baseline', 'fault', 'off'):
            raise ValueError('phase must be baseline, fault or off')
        if not self._qualified or self._identity() != self._model_identity:
            raise PostwriteRefused('POSTWRITE_MODEL_IDENTITY_NOT_QUALIFIED')
        if _hash_inputs(args, kwargs) != self._input_hash:
            raise PostwriteRefused('POSTWRITE_REPLAY_INPUT_CHANGED')
        if phase == 'fault' and (not self._lossless or self._fault_attempted):
            raise PostwriteRefused('POSTWRITE_LOSSLESS_OR_ONCE_GATE')
        if phase == 'fault':
            self._fault_attempted = True
        self.generation += 1
        output, _, report = self._replay(phase, args, kwargs, model_observation=model_observation)
        if phase == 'baseline':
            if not _equal(output, self._baseline_output):
                raise PostwriteRefused('POSTWRITE_LOSSLESS_OUTPUT_MISMATCH')
            self._lossless = True
        if phase == 'off' and not _equal(output, self._baseline_output):
            raise PostwriteRefused('POSTWRITE_OFF_OUTPUT_MISMATCH')
        report['recovery_confirmed'] = self._lease is None
        report['exact_baseline_output'] = _equal(output, self._baseline_output)
        report['new_fault_hits'] = (report['intervention'] or {}).get('new_fault_hits', 0)
        self._history.append(report)
        return output, report

    def recover(self):
        if self._lease is not None:
            self._lease.recover()
            if self._lease.live():
                raise PostwriteRefused('POSTWRITE_NATIVE_LEASE_REMAINS')
            self._lease = None

    def query(self):
        return dict(mechanism=self.mechanism, owner=self.owner, generation=self.generation,
            execution_boundary='public_acl_postwrite', transport=self.transport.transport_kind,
            qualified=self._qualified, lossless=self._lossless, attempted=self._fault_attempted,
            leases=int(self._lease is not None), recovery_pending=self._lease is not None,
            history=list(self._history), qualification_diagnostics=self._qualification_diagnostics,
            public_worker_qualified=False)


class Fault39OutputProxyProvider(PostwriteProvider):
    """Finite actual BF16 MM output -> selected nonfinite prefix -> real consumer."""
    def __init__(self, model, transport, *, prefix_elements=1, proxy='pos_inf'):
        super().__init__(model, transport, '39.output_proxy',
                         prefix_elements=prefix_elements, proxy=proxy)


class Fault41TailColumnProvider(PostwriteProvider):
    """Actual model-owned cache IndexCopy writer -> finite logical slot tail sign flip."""
    def __init__(self, model, transport, *, tail_elements=1):
        super().__init__(model, transport, '41.tail_column', tail_elements=tail_elements)
