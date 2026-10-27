"""Bounded eager-only dynamic-quant producer/consumer binding.

Install in an isolated process. Only calls inside an explicit scope are changed;
all other calls use the original APIs. A scope must contain exactly one producer
and at least one consumer of its *same* Q/scale pair. This is an experimental
TP1 dense decode adapter, not a global change to the existing single-request UI.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import threading

from .batch_scale_core import compute_pair


class BindingError(RuntimeError):
    pass


_INSTALL_LOCK = threading.Lock()


def tensor_digest(tensor):
    value = tensor.detach().contiguous().cpu()
    raw = value.view(__import__('torch').uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


class NativePairBinding:
    def __init__(self, backend, *, max_calls=32, max_elements=4194304,
                 scale_rtol=1e-6, reference=None, observer=None):
        if type(max_calls) is not int or not 1 <= max_calls <= 256:
            raise ValueError('max_calls must be in [1,256]')
        if type(max_elements) is not int or not 1 <= max_elements <= 16777216:
            raise ValueError('max_elements must be in [1,16777216]')
        if not 0 <= scale_rtol <= 1e-6:
            raise ValueError('scale_rtol must be in [0,1e-6]')
        if observer is not None and not callable(observer):
            raise ValueError('observer must be callable or None')
        self.backend, self.max_calls = backend, max_calls
        self.max_elements, self.scale_rtol = max_elements, scale_rtol
        self.reference = reference
        self.observer = observer
        self.events = []
        self._active = None
        self._installed = False

    def __enter__(self):
        if self._installed or not _INSTALL_LOCK.acquire(blocking=False):
            raise BindingError('ANOTHER_PAIR_BINDING_IS_INSTALLED')
        try:
            self._producer = self.backend.npu_dynamic_quant
            self._consumer = self.backend.npu_quant_matmul
            self._producer_wrapper = self._guard(self._produce)
            self._consumer_wrapper = self._guard(self._consume)
            self.backend.npu_dynamic_quant = self._producer_wrapper
            self.backend.npu_quant_matmul = self._consumer_wrapper
            self._installed = True
        except BaseException:
            try:
                for name, wrapper_name, original_name in (
                    ('npu_dynamic_quant', '_producer_wrapper', '_producer'),
                    ('npu_quant_matmul', '_consumer_wrapper', '_consumer'),
                ):
                    wrapper = getattr(self, wrapper_name, None)
                    if wrapper is not None and getattr(self.backend, name, None) is wrapper:
                        setattr(self.backend, name, getattr(self, original_name))
            finally:
                _INSTALL_LOCK.release()
            raise
        return self

    def _guard(self, operation):
        def guarded(*args, **kwargs):
            try:
                return operation(*args, **kwargs)
            except BaseException as error:
                if self._active is not None:
                    self._active['event']['operator_error'] = type(error).__name__ + ':' + str(error)
                raise
        return guarded

    def __exit__(self, exc_type, exc, traceback):
        collisions = []
        try:
            for name, wrapper, original in (
                ('npu_dynamic_quant', self._producer_wrapper, self._producer),
                ('npu_quant_matmul', self._consumer_wrapper, self._consumer),
            ):
                if getattr(self.backend, name) is wrapper:
                    setattr(self.backend, name, original)
                else:
                    collisions.append(name)
        finally:
            self._installed = False
            self._active = None
            _INSTALL_LOCK.release()
        if collisions:
            raise BindingError('API_RESTORE_COLLISION:' + ','.join(collisions))

    @contextmanager
    def scope(self, rows, *, mode, module_name, target_request_id):
        if not self._installed or self._active is not None:
            raise BindingError('SCOPE_REQUIRES_EXCLUSIVE_INSTALLED_BINDING')
        if mode not in ('healthy', 'shared') or not module_name:
            raise BindingError('INVALID_SCOPE_MODE_OR_MODULE')
        if target_request_id not in rows.expected_request_ids:
            raise BindingError('TARGET_REQUEST_NOT_IN_BATCH')
        if mode == 'shared' and len(set(rows.valid_request_ids)) < 2:
            raise BindingError('SHARED_SCOPE_REQUIRES_MULTIPLE_REQUESTS')
        if len(self.events) >= self.max_calls:
            raise BindingError('CALL_BUDGET_EXHAUSTED')
        if any(e['call_id'] == rows.call_id for e in self.events):
            raise BindingError('DUPLICATE_CALL_ID')
        event = {'call_id': rows.call_id, 'module': module_name, 'mode': mode,
                 'target_request_id': target_request_id, 'rows': rows.to_dict(),
                 'status': 'ENTERED', 'consumers': []}
        state = {'thread': threading.get_ident(), 'rows': rows, 'event': event,
                 'pair': None, 'input': None}
        self.events.append(event)
        self._active = state
        try:
            yield event
            if 'operator_error' in event:
                raise BindingError('SCOPE_TAINTED_BY_OPERATOR_FAILURE')
            if state['pair'] is None or not event['consumers']:
                raise BindingError('PRODUCER_CONSUMER_CHAIN_INCOMPLETE')
            self._verify_pair(state)
            import torch
            if not torch.equal(state['input'], state['input_snapshot']):
                raise BindingError('UPSTREAM_ACTIVATION_CHANGED')
            event['input_unchanged'] = True
            event['status'] = 'COMPLETE'
        except BaseException as error:
            event.update(status='FAILED', error=type(error).__name__ + ':' + str(error))
            raise
        finally:
            self._active = None

    def _state_for_thread(self):
        state = self._active
        if state is not None and state['thread'] != threading.get_ident():
            # An isolated eager experiment must not have concurrent operators.
            raise BindingError('CONCURRENT_EXECUTION_DURING_BOUND_SCOPE')
        return state

    def _produce(self, x, *args, **kwargs):
        state = self._state_for_thread()
        if state is None:
            return self._producer(x, *args, **kwargs)
        import torch
        if state['pair'] is not None:
            raise BindingError('MULTIPLE_PRODUCERS_IN_ONE_SCOPE')
        if args or set(kwargs) - {'smooth_scales', 'group_index', 'dst_type'}:
            raise BindingError('UNREVIEWED_DYNAMIC_QUANT_SIGNATURE')
        if kwargs.get('smooth_scales') is not None or kwargs.get('group_index') is not None:
            raise BindingError('SMOOTH_OR_GROUPED_QUANT_UNSUPPORTED')
        if kwargs.get('dst_type') not in (None, torch.int8):
            raise BindingError('ONLY_INT8_SUPPORTED')
        if x.ndim != 2 or not x.numel() or x.numel() > self.max_elements:
            raise BindingError('INPUT_SHAPE_OR_ELEMENT_BUDGET')
        if x.dtype not in (torch.float16, torch.bfloat16):
            raise BindingError('NATIVE_INPUT_REQUIRES_FP16_OR_BF16')
        state['rows'].for_tensor_rows(x.shape[0])
        if not bool(torch.isfinite(x).all().item()):
            raise BindingError('NONFINITE_ACTIVATION')
        state['input'], state['input_snapshot'] = x, x.detach().clone()
        native_q, native_s = self._producer(x, **kwargs)
        if (native_q.shape != x.shape or native_q.dtype != torch.int8
                or native_s.shape != (x.shape[0],) or native_s.dtype != torch.float32
                or native_q.device != x.device or native_s.device != x.device):
            raise BindingError('NATIVE_PAIR_CONTRACT_MISMATCH')
        if not bool(torch.isfinite(native_s).all().item()) or bool((native_s < 0).any().item()):
            raise BindingError('INVALID_NATIVE_SCALE')
        valid = torch.tensor(state['rows'].valid_rows, device=x.device, dtype=torch.bool)
        healthy = compute_pair(x, shared=False, valid_rows=list(state['rows'].valid_rows),
                               reference=self.reference)
        zero = x.detach().abs().amax(dim=-1) == 0
        nonzero_valid = valid & ~zero
        q_equal = torch.equal(native_q[valid], healthy.q[valid])
        s_close = torch.allclose(native_s[nonzero_valid], healthy.scale[nonzero_valid],
                                 rtol=self.scale_rtol, atol=0)
        # Native zero-row scales may be zero; codes must still be exactly zero.
        if not q_equal or not s_close:
            raise BindingError('HEALTHY_REPLAY_DOES_NOT_MATCH_NATIVE')
        event = state['event']
        scale_delta = (native_s[nonzero_valid] - healthy.scale[nonzero_valid]).abs()
        event['calibration'] = {'q_exact': q_equal, 'scale_rtol': self.scale_rtol,
                                'scale_atol': 0, 'scale_close': s_close,
                                'max_scale_abs_delta': float(scale_delta.max().item()) if scale_delta.numel() else 0,
                                'zero_row_policy': 'native scale preserved in healthy mode; Q must equal zero'}
        if event['mode'] == 'shared' and bool(nonzero_valid.any().item()):
            shared = compute_pair(x, shared=True, valid_rows=list(state['rows'].valid_rows),
                                  reference=self.reference)
            q = torch.where(valid[:, None], shared.q, native_q)
            scale = torch.where(valid, shared.scale, native_s)
        else:
            q, scale = native_q, native_s
        state['pair'] = (q, scale)
        state['pair_snapshot'] = (q.detach().clone(), scale.detach().clone())
        self._record_effect(x, native_q, native_s, q, scale, valid, state)
        self._observe('pair', state, input_x=x, native_q=native_q,
                      native_scale=native_s, returned_q=q, returned_scale=scale)
        return q, scale

    def _observe(self, stage, state, **tensors):
        """Optional recorder receives independent copies, never live API payloads.

        Recording errors propagate and taint the active scope. This makes an
        incomplete capture a failed experiment instead of a successful injection.
        """
        if self.observer is None:
            return
        self.observer(stage, metadata=deepcopy(state['event']),
                      tensors={name: value.detach().clone() for name, value in tensors.items()})
        state['event'].setdefault('observed_stages', []).append(stage)

    def _record_effect(self, x, native_q, native_s, q, scale, valid, state):
        import torch
        row_max = x.detach().float().abs().amax(dim=-1)
        all_max = row_max[valid].max()
        rows, event = state['rows'], state['event']
        target = torch.tensor([v and r == event['target_request_id']
                               for v, r in zip(rows.valid_rows, rows.request_ids)],
                              device=x.device, dtype=torch.bool)
        restored = q.float() * scale[:, None]
        baseline = native_q.float() * native_s[:, None]
        mse_h = (baseline - x.float()).square().mean(dim=-1)
        mse_f = (restored - x.float()).square().mean(dim=-1)
        max_rows = (valid & (row_max == all_max)).nonzero().flatten().cpu().tolist()
        event.update(input_sha256=tensor_digest(x), q_sha256=tensor_digest(q),
                     scale_sha256=tensor_digest(scale),
                     native_q_sha256=tensor_digest(native_q), native_scale_sha256=tensor_digest(native_s),
                     shared_absmax=float(all_max.item()),
                     target_request_only_absmax=float(row_max[target].max().item()),
                     external_range_increase=bool((all_max > row_max[target].max()).item()),
                     extremum_rows=max_rows,
                     extremum_request_ids=[rows.request_ids[i] for i in max_rows],
                     native_scales=native_s.detach().cpu().tolist(),
                     returned_scales=scale.detach().cpu().tolist(),
                     row_absmax=row_max.cpu().tolist(),
                     row_healthy_mse=mse_h.detach().cpu().tolist(),
                     row_returned_mse=mse_f.detach().cpu().tolist(),
                     changed_codes=int((q[valid] != native_q[valid]).sum().item()),
                     changed_dequant_values=int((restored[valid] != baseline[valid]).sum().item()),
                     padding_pair_preserved=bool(torch.equal(q[~valid], native_q[~valid])
                                                  and torch.equal(scale[~valid], native_s[~valid])))

    def _consume(self, *args, **kwargs):
        state = self._state_for_thread()
        if state is None:
            return self._consumer(*args, **kwargs)
        import torch
        if state['pair'] is None:
            raise BindingError('CONSUMER_WITHOUT_PRODUCER')
        q, scale = state['pair']
        actual_q = args[0] if args else kwargs.get('x1')
        if actual_q is not q or kwargs.get('pertoken_scale') is not scale:
            raise BindingError('Q_SCALE_PAIR_IDENTITY_MISMATCH')
        self._verify_pair(state)
        if len(state['event']['consumers']) >= 4:
            raise BindingError('CONSUMER_BUDGET_EXHAUSTED')
        output = self._consumer(*args, **kwargs)
        self._verify_pair(state)
        state['event']['consumers'].append({
            'api': 'npu_quant_matmul', 'same_q_object': True, 'same_scale_object': True,
            'q_sha256': tensor_digest(q), 'scale_sha256': tensor_digest(scale),
            'output_shape': list(output.shape), 'output_sha256': tensor_digest(output),
        })
        self._observe('consumer', state, output=output)
        return output

    @staticmethod
    def _verify_pair(state):
        import torch
        if not all(torch.equal(a, b) for a, b in zip(state['pair'], state['pair_snapshot'])):
            raise BindingError('Q_SCALE_PAIR_MUTATED')

    @contextmanager
    def bind_module(self, module, *, module_name, context_provider):
        """Scope one actual module forward. Provider returns None or scope kwargs.

        The provider must read the active scheduler step; it must never infer
        request membership from client submission count or tensor shape alone.
        None is used for non-target phases such as prefill. Restores even on error.
        """
        had_own = 'forward' in module.__dict__
        own = module.__dict__.get('forward')
        original = module.forward

        def forward(*args, **kwargs):
            context = context_provider(*args, **kwargs)
            if context is None:
                return original(*args, **kwargs)
            with self.scope(module_name=module_name, **context):
                return original(*args, **kwargs)

        module.forward = forward
        try:
            yield module
        finally:
            if module.forward is not forward:
                raise BindingError('MODULE_RESTORE_COLLISION')
            if had_own:
                module.forward = own
            else:
                del module.forward
