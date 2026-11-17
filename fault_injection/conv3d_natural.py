"""Original Fault17 padding argument at an observed stock patch Conv3d.

No output crop, synthetic layer or geometry change is permitted. Independent
effect comparison requires identical complete input/weight/bias bytes to the
healthy call; propagated-input cases refuse instead of claiming a comparison.
"""
import hashlib

from ._scoped_dispatch import ScopedDispatch
from .core import UnsupportedCall
from .module_faults import discover_modules
from .tensor_observer import tensors


def conv_arguments(func, args, kwargs):
    schema = getattr(func, '_schema', None)
    if (getattr(schema, 'name', '') not in ('aten::conv3d', 'aten::convolution') or
            getattr(schema, 'overload_name', '') != ''):
        return None
    values = {arg.name: arg.default_value for arg in schema.arguments}
    values.update(zip((arg.name for arg in schema.arguments), args)); values.update(kwargs)
    weight = values.get('weight')
    return values if getattr(weight, 'ndim', None) == 5 else None


def tensor_identity(cpu):
    import torch
    return dict(shape=list(cpu.shape), dtype=str(cpu.dtype), sha256=hashlib.sha256(
        cpu.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


class PatchConv3dSession(ScopedDispatch):
    mechanism = '17.spatial_padding'

    def __init__(self, model, *, admission=None, references=None, **kwargs):
        super().__init__(model, **kwargs)
        self.candidates = dict(discover_modules(model, self.mechanism))
        self.admission = admission
        self.references = {} if references is None else dict(references)
        self.healthy_calls, self.records = {}, []

    def _identity(self, value):
        return None if value is None else tensor_identity(self.cpu(value))

    def dispatch(self, func, args, kwargs):
        import torch
        values = conv_arguments(func, args, kwargs)
        if values is None:
            name = getattr(getattr(func, '_schema', None), 'name', '')
            inputs = tuple(tensors((args, kwargs)))
            linked = [record for record in self.records if record['forward'] == len(self.forwards)-1
                and any(x.untyped_storage()._cdata in record['storage_ids'] for x in inputs)]
            output = func(*args, **kwargs)
            for record in linked:
                if name in ('aten::view','aten::reshape','aten::_unsafe_view','aten::clone','aten::_to_copy',
                            'aten::slice','aten::select','aten::transpose','aten::permute'):
                    record['storage_ids'].update(x.untyped_storage()._cdata for x in tensors(output))
                if name in ('aten::add','aten::mul','aten::linear','aten::mm','aten::matmul','aten::addmm',
                            'aten::bmm','aten::convolution','aten::conv3d'):
                    record['event']['consumers'].append(dict(operator=str(func), returned=True,
                        module_path=self.stack[-1][0], actual_conv_output_lineage=True))
            return output
        owners = [(path,module) for path,module in self.stack
                  if getattr(module,'weight',None) is values['weight']]
        event = self.event(str(func))
        if len(owners) != 1:
            raise UnsupportedCall('CONV3D_ACTUAL_WEIGHT_OWNER_AMBIGUOUS')
        path,module = owners[0]
        if path != event['path'] or self.candidates.get(path) is not module:
            raise UnsupportedCall('ACTUAL_STOCK_CONV3D_FORWARD_REQUIRED')
        with self.diagnostics():
            x,w,b = values['input'],values['weight'],values['bias']
            kernel = tuple(w.shape[2:])
            stride,padding,dilation = (tuple(values[key]) for key in ('stride','padding','dilation'))
            if (module.training or x.ndim != 5 or tuple(x.shape[2:]) != kernel
                or kernel[1:] != (16,16) or stride != kernel or padding != (0,0,0)
                or dilation != (1,1,1) or values['groups'] != 1 or values.get('transposed',False)
                or any(values.get('output_padding',())) or b is not module.bias
                or module.padding_mode != 'zeros' or not x.is_floating_point()
                or x.dtype != w.dtype or x.device != w.device
                or not self.simulation and x.device.type != 'npu'):
                raise UnsupportedCall('ORIGINAL_ZERO_PADDED_NONOVERLAP_16X16_PATCH_CONV3D_REQUIRED')
            before = [self._identity(tensor) for tensor in (x,w,b)]
            key = (path,event['ordinal'])
            injecting = self.arm == 'fault' and bool(self.admission(event) if self.admission else self.is_selected(event))
            reference = self.references.get(key)
            if injecting and (reference is None or before != reference['inputs']):
                raise UnsupportedCall('SAME_ACTUAL_CONV_INPUT_WEIGHT_BIAS_REQUIRED_FOR_EFFECT_COMPARISON')
            chosen = (0,7,7) if injecting else padding
            native = dict(values,padding=list(chosen))
        # Original observed dispatcher executes once, outside diagnostic pause.
        output = func(**native)
        with self.diagnostics():
            after = self._identity(output)
            if (tuple(output.shape) != (x.shape[0],w.shape[0],1,1,1) or output.dtype != x.dtype
                    or output.device != x.device or before != [self._identity(t) for t in (x,w,b)]):
                raise UnsupportedCall('CONV3D_RETURN_GEOMETRY_OR_IMMUTABLE_INPUT_READBACK_CHANGED')
            if not injecting:
                self.healthy_calls[key] = dict(inputs=before,output=after)
            event.update(status='applied' if injecting else 'baseline', applied=injecting,
                input_shape=list(x.shape),weight_shape=list(w.shape),output_shape=list(output.shape),
                dtype=str(output.dtype),stride=list(stride),padding=list(chosen),dilation=list(dilation),
                groups=values['groups'],baseline_padding=list(padding),native_calls=1,
                original_execution_verified=True,return_readback_exact=True,inputs_unchanged=True,
                complete_input_weight_bias_identity=before,complete_output_identity=after,
                output_changed=bool(injecting and after != reference['output']),consumers=[],
                effect_comparison='Same complete actual inputs/weights/bias; full native output bytes',
                boundary=str(func)+'.arguments.padding',semantics='NATIVE_CONV3D_PADDING_0_7_7')
            self.records.append(dict(event=event,forward=len(self.forwards)-1,
                storage_ids={output.untyped_storage()._cdata}))
        return output

    def close(self):
        super().close()
        self.records.clear()
