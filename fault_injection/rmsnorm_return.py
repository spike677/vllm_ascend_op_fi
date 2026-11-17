"""Fault18 at actual RMSNorm module returns, including reviewed natural paths.

Native fused outputs are admitted only with live weight/operator/return lineage.
The canonical eager implementation is admitted by its complete normalized AST,
not a class name or a source keyword. This remains an output-half proxy.
"""
import ast
import hashlib
import inspect
import textwrap

from ._scoped_dispatch import ScopedDispatch
from .core import UnsupportedCall
from .module_faults import _original_standard_forward
from .tensor_observer import tensors


_CANONICAL = '''def forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)
'''


def _ast(source):
    node = ast.parse(textwrap.dedent(source)).body[0]
    if not isinstance(node, ast.FunctionDef):
        return None
    node.decorator_list = []
    node.returns = None
    for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
        arg.annotation = None
    if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
        if isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
    return ast.dump(node, include_attributes=False)


def eager_rms_contract(module):
    import torch
    function = _original_standard_forward(module)
    if function is None:
        return None
    implementation = getattr(function, '__func__', function)
    if implementation is torch.nn.RMSNorm.forward:
        return dict(kind='stock_torch_rmsnorm', source_sha256=hashlib.sha256(
            inspect.getsource(implementation).encode()).hexdigest())
    weight = getattr(module, 'weight', None)
    epsilon = getattr(module, 'variance_epsilon', None)
    if (not isinstance(weight, torch.Tensor) or weight.ndim != 1 or
        type(epsilon) not in (int, float) or not 0 < epsilon < float('inf')):
        return None
    try:
        source = inspect.getsource(implementation)
        if _ast(source) != _ast(_CANONICAL):
            return None
    except (OSError, TypeError, SyntaxError):
        return None
    return dict(kind='canonical_fp32_variance_eager_rmsnorm',
                source_sha256=hashlib.sha256(source.encode()).hexdigest())


def _same_view(left, right):
    return (left.untyped_storage()._cdata == right.untyped_storage()._cdata
        and left.storage_offset() == right.storage_offset() and left.shape == right.shape
        and left.stride() == right.stride() and left.dtype == right.dtype and left.device == right.device)


class RMSNormReturnSession(ScopedDispatch):
    mechanism = '18.api_half'

    def __init__(self, model, *, selected_calls=None, admission=None,
                 observe_post_root_consumers=False, **kwargs):
        super().__init__(model, **kwargs)
        self.selected_calls = None if selected_calls is None else frozenset(selected_calls)
        self.eager = {id(module): eager_rms_contract(module) for module in model.modules()}
        self.pending = {}
        self.returns = []
        self.admission = admission
        self.observe_post_root_consumers = observe_post_root_consumers

    def is_selected(self, event):
        if self.admission is not None:
            return bool(self.admission(event))
        if self.selected_calls is not None:
            return (event['path'], event['ordinal']) in self.selected_calls
        return super().is_selected(event)

    def dispatch(self, func, args, kwargs):
        import torch
        schema = getattr(func, '_schema', None)
        name = getattr(schema, 'name', '')
        native_rms = (name.split('::', 1)[0] in ('npu', '_C_ascend') and
            name.rsplit('::', 1)[-1] in ('npu_rms_norm', 'npu_add_rms_norm', 'npu_add_rms_norm_bias'))
        if not native_rms:
            return self._read_call(func, args, kwargs)
        actual_tensors = tuple(tensors((args, kwargs)))
        owners = [(path, module) for path, module in self.stack
            if isinstance(getattr(module, 'weight', None), torch.Tensor)
            and module.weight.ndim == 1 and any(t is module.weight for t in actual_tensors)]
        if len(owners) != 1:
            return func(*args, **kwargs)
        path, module = owners[0]
        event = self.event(str(func)); event['path'] = path
        if id(module) in self.pending:
            raise UnsupportedCall('MULTIPLE_RMS_NATIVE_RETURNS_REQUIRE_SEPARATE_ADAPTER')
        result = func(*args, **kwargs)
        outputs = tuple(tensors(result))
        if not outputs or not outputs[0].is_floating_point() or module.training:
            event.update(status='unsupported', reason='INFERENCE_NATIVE_RMS_TENSOR_RETURN_REQUIRED')
            return result
        event.update(original_execution_verified=True, native_calls=1,
            native_operator=str(func), schema=str(schema), weight_object_id=id(module.weight),
            native_output_storage=int(outputs[0].untyped_storage()._cdata),
            source_contract='ACTUAL_NATIVE_RMS_WEIGHT_AND_MODULE_RETURN_LINEAGE')
        self.pending[id(module)] = (event, outputs[0], tuple(outputs[1:]))
        return result

    def _read_call(self, func, args, kwargs):
        name = getattr(getattr(func, '_schema', None), 'name', '')
        input_storages = {int(t.untyped_storage()._cdata) for t in tensors((args, kwargs))}
        linked = [(tensor, event) for tensor, event in self.returns
            if event['forward'] == len(self.forwards)-1
            and int(tensor.untyped_storage()._cdata) in input_storages]
        unique = {id(event): event for tensor, event in linked}
        readers = list(unique.values()) if name in ('aten::add', 'aten::mul', 'aten::linear', 'aten::matmul', 'aten::mm',
                'npu::npu_dynamic_quant', 'npu::npu_quant_matmul', 'npu::npu_add_rms_norm',
                'vllm::unquantized_gemm', 'vllm::npu_rotary_embedding') else []
        result = func(*args, **kwargs)
        for event in readers:
            event['consumers'].append(dict(operator=str(func),
                path=self.stack[-1][0] if self.stack else '<post_root_consumer>',
                returned=True, actual_return_lineage=True, post_root=not bool(self.stack)))
        if name in ('aten::view', 'aten::reshape', 'aten::_unsafe_view', 'aten::slice', 'aten::select',
                    'aten::index', 'aten::index_select', 'aten::clone', 'aten::_to_copy',
                    'aten::squeeze', 'aten::unsqueeze', 'aten::flatten', 'aten::transpose', 'aten::permute'):
            for event in unique.values():
                self.returns.extend((value, event) for value in tensors(result))
                event.setdefault('lineage_operators', []).append(str(func))
        return result

    def dispatch_outside_root(self, func, args, kwargs):
        # A serving adapter may keep the owned mode until the actual logits
        # processor returns. Only already-returned storage is observed here;
        # no new RMSNorm target can be discovered or changed outside the root.
        if (self.observe_post_root_consumers and self.forwards
                and self.forwards[-1]['status'] == 'returned'):
            return self._read_call(func, args, kwargs)
        return func(*args, **kwargs)

    def _after(self, module):
        original = super()._after(module)
        def leave(actual, args, output):
            replacement = None
            try:
                if self.active and self.stack and self.stack[-1][1] is actual and output is not None:
                    replacement = self._return(actual, args, output)
            finally:
                original(actual, args, output)
            return replacement
        return leave

    def _return(self, module, args, output):
        import torch
        pending = self.pending.pop(id(module), None)
        contract = self.eager.get(id(module))
        if pending is None and contract is None:
            return None
        values = tuple(tensors(output))
        if (not values or not isinstance(output, (torch.Tensor, tuple)) or
            not values[0].is_floating_point() or module.training or
            not self.simulation and values[0].device.type != 'npu'):
            raise UnsupportedCall('RMSNORM_ACTUAL_INFERENCE_RETURN_REQUIRED')
        if pending is not None:
            event, native_output, auxiliaries = pending
            if (not _same_view(native_output, values[0]) or
                any(not any(_same_view(aux, value) for aux in auxiliaries) for value in values[1:])):
                raise UnsupportedCall('RMSNORM_MODULE_RETURN_NATIVE_LINEAGE_UNPROVEN')
        else:
            if not isinstance(output, torch.Tensor):
                raise UnsupportedCall('EAGER_RMSNORM_SINGLE_RETURN_REQUIRED')
            event = self.event('RMSNorm.forward')
            event.update(original_execution_verified=True, source_contract=contract,
                         native_calls=None, module_object_id=id(module))
        injecting = self.arm == 'fault' and self.is_selected(event)
        # One identical call site in lossless and fault arms makes the API
        # collector see the real return boundary in both arms. Healthy *1 is
        # checked against an entirely unhooked baseline by the replay provider.
        gain = 0.5 if injecting else 1.0
        with self.diagnostics():
            changed = values[0] * gain
            before, after = self.cpu(values[0]), self.cpu(changed)
            if not bool(torch.isfinite(before).all()) or not torch.equal(after, before * gain):
                raise UnsupportedCall('RMSNORM_COMPLETE_RETURN_GAIN_READBACK_FAILED')
            event.update(status='applied' if injecting else 'baseline', applied=injecting,
                output_shape=list(changed.shape), dtype=str(changed.dtype),
                boundary='module_return_gain_proxy', gain=gain, output_half_proxy=True,
                native_reduce_error_claimed=False, return_readback_exact=True, consumers=[],
                output_changed=not torch.equal(before, after),
                auxiliary_return_storage_unchanged=True)
            self.returns.append((changed, event))
        if isinstance(output, tuple):
            if not isinstance(output[0], torch.Tensor) or output[0] is not values[0]:
                raise UnsupportedCall('RMSNORM_TUPLE_FIRST_OUTPUT_REQUIRED')
            return (changed, *output[1:])
        return changed

    def close(self):
        super().close()
        self.pending.clear(); self.returns.clear()
