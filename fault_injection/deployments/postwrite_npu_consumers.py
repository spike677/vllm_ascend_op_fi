"""Narrow read-only consumers from the inspected torch_npu 2.9 build.

This observes an already completed operation. It never calls a consumer,
copies its input, or accepts an opaque custom operator by its name alone.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys

import torch
from torch._ops import OpOverload

from .postwrite_boundaries import tensor_geometry


TORCH_VERSION = '2.9.0+cpu'
TORCH_NPU_VERSION = '2.9.0.post1+gitee7ba04'
META_MODULE = 'torch_npu.op_plugin.meta._meta_registrations'
META_SHA256 = '3e3997c2b99dcff7a4ee4d693bd7e2365f6cf7e906dee5e08585081c7ea13df4'
RMS_SCHEMA = ('npu::npu_rms_norm(Tensor input, Tensor gamma, '
              'float epsilon=9.9999999999999995e-07) -> (Tensor, Tensor)')
SWIGLU_SCHEMA = 'npu::npu_swiglu(Tensor input, int dim=-1) -> Tensor'
NPU_VALUE_CONSUMERS = frozenset({'npu.npu_rms_norm.default', 'npu.npu_swiglu.default'})
PRIVATE_REGISTRATION = {
    'npu.npu_rms_norm.default': ('PrivateUse1: registered at '
        'build/CMakeFiles/torch_npu.dir/compiler_depend.ts:8413 :: '
        '(Tensor _0, Tensor _1, float _2) -> (Tensor _0, Tensor _1) [ boxed unboxed ]'),
    'npu.npu_swiglu.default': ('PrivateUse1: registered at '
        'build/CMakeFiles/torch_npu.dir/compiler_depend.ts:8413 :: '
        '(Tensor _0, int _1) -> Tensor _0 [ boxed unboxed ]'),
}


def _contiguous_view(tensor, *, dtype, device):
    if (not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype or tensor.device != device
            or not tensor.is_contiguous() or tensor.storage_offset() < 0 or not tensor.numel()):
        return False
    size = tensor.element_size()
    storage = tensor.untyped_storage()
    return (tensor.data_ptr() == storage.data_ptr() + tensor.storage_offset() * size
            and (tensor.storage_offset() + tensor.numel()) * size <= storage.nbytes())


def npu_consumer_registration(func):
    """Read-only current schema/build/source check, also used before admission."""
    operation = str(func)
    if operation not in NPU_VALUE_CONSUMERS:
        return None, 'NPU_CONSUMER_NOT_ADAPTED'
    rms = operation == 'npu.npu_rms_norm.default'
    schema = RMS_SCHEMA if rms else SWIGLU_SCHEMA
    expected_op = torch.ops.npu.npu_rms_norm.default if rms else torch.ops.npu.npu_swiglu.default
    if (not isinstance(func, OpOverload) or func is not expected_op or str(func._schema) != schema
            or any(p.alias_info is not None for p in (*func._schema.arguments, *func._schema.returns))):
        return None, 'UNREVIEWED_NPU_CONSUMER_SCHEMA_OR_ALIAS'
    npu = sys.modules.get('torch_npu')
    if (str(torch.__version__) != TORCH_VERSION or getattr(npu, '__version__', None) != TORCH_NPU_VERSION
            or not callable(getattr(npu, 'get_npu_format', None))):
        return None, 'UNREVIEWED_NPU_CONSUMER_BUILD'
    try:
        meta = sys.modules[META_MODULE]
        meta_sha = hashlib.sha256(Path(meta.__file__).resolve().read_bytes()).hexdigest()
    except (KeyError, AttributeError, OSError, TypeError):
        return None, 'NPU_CONSUMER_META_SOURCE_UNAVAILABLE'
    if meta_sha != META_SHA256:
        return None, 'UNREVIEWED_NPU_CONSUMER_META_SOURCE'
    name = 'npu::npu_rms_norm' if rms else 'npu::npu_swiglu'
    dispatch = torch._C._dispatch_dump(name)
    if (('schema: ' + schema) not in dispatch.splitlines()
            or 'alias analysis kind: FROM_SCHEMA' not in dispatch.splitlines()
            or [line for line in dispatch.splitlines() if line.startswith('PrivateUse1:')]
                != [PRIVATE_REGISTRATION[operation]]
            or not torch._C._dispatch_has_kernel_for_dispatch_key(name, 'PrivateUse1')):
        return None, 'UNREVIEWED_NPU_CONSUMER_DISPATCH'
    return dict(operator=operation, schema=schema,
        schema_sha256=hashlib.sha256(schema.encode()).hexdigest(),
        torch_version=str(torch.__version__), torch_npu_version=npu.__version__,
        meta_shape_source=dict(module=META_MODULE, sha256=meta_sha,
            function='npu_rms_norm_meta' if rms else 'npu_swiglu_meta'),
        actual_dispatch_dump=dispatch, schema_has_no_alias_or_write=True), None


def npu_value_consumer(func, args, kwargs, result):
    """Return (proof, actual value-read operands, reason), with no replay.

    Only the ``input`` operand consumes a projection. A gamma alias is not a
    substitute for input lineage. Selected-index membership is checked separately
    against this exact returned operand object, storage, offset and stride.
    """
    proof, reason = npu_consumer_registration(func)
    if reason:
        return None, (), reason
    rms = proof['operator'] == 'npu.npu_rms_norm.default'
    npu = sys.modules['torch_npu']
    parameters = func._schema.arguments
    if (len(args) > len(parameters) or set(kwargs) - {p.name for p in parameters}
            or any(p.name in kwargs for p in parameters[:len(args)])):
        return None, (), 'NPU_CONSUMER_ARGUMENT_BINDING_CHANGED'
    bound = dict(zip((p.name for p in parameters), args))
    bound.update(kwargs)
    for parameter in parameters:
        if parameter.name not in bound:
            if not parameter.has_default_value():
                return None, (), 'NPU_CONSUMER_ARGUMENT_MISSING'
            bound[parameter.name] = parameter.default_value
    value = bound['input']
    if (not isinstance(value, torch.Tensor) or value.ndim != 2
            or not _contiguous_view(value, dtype=torch.bfloat16, device=value.device)):
        return None, (), 'NPU_CONSUMER_INPUT_VIEW_NOT_REVIEWED'
    operands = [value]
    if rms:
        gamma, epsilon = bound['gamma'], bound['epsilon']
        if (not _contiguous_view(gamma, dtype=value.dtype, device=value.device)
                or tuple(gamma.shape) != (value.shape[-1],)
                or type(epsilon) is not float or not math.isfinite(epsilon) or epsilon <= 0):
            return None, (), 'NPU_RMS_CONSUMER_GAMMA_OR_EPSILON_CHANGED'
        operands.append(gamma)
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            return None, (), 'NPU_RMS_CONSUMER_RETURN_CHANGED'
        output, rstd = result
        if (not _contiguous_view(output, dtype=value.dtype, device=value.device)
                or tuple(output.shape) != tuple(value.shape)
                or not _contiguous_view(rstd, dtype=torch.float32, device=value.device)
                or tuple(rstd.shape) != tuple(value.shape[:-1]) + (1,)):
            return None, (), 'NPU_RMS_CONSUMER_RETURN_GEOMETRY_CHANGED'
        outputs = [output, rstd]
    else:
        if type(bound['dim']) is not int or bound['dim'] != -1 or value.shape[-1] % 2:
            return None, (), 'NPU_SWIGLU_LAST_EVEN_DIM_REQUIRED'
        if (not _contiguous_view(result, dtype=value.dtype, device=value.device)
                or tuple(result.shape) != (value.shape[0], value.shape[1] // 2)):
            return None, (), 'NPU_SWIGLU_CONSUMER_RETURN_GEOMETRY_CHANGED'
        outputs = [result]
    # No returned alias is permitted by the reviewed schemas. Also exclude
    # parameter/input aliasing so a parameter read cannot masquerade as x.
    tensors = operands + outputs
    if len({t.untyped_storage()._cdata for t in tensors}) != len(tensors):
        return None, (), 'NPU_CONSUMER_OPERAND_OR_RETURN_ALIAS'
    try:
        formats = [int(npu.get_npu_format(t)) for t in tensors]
    except (RuntimeError, TypeError, ValueError):
        return None, (), 'NPU_CONSUMER_FORMAT_UNPROVEN'
    if any(value != 2 for value in formats):
        return None, (), 'NPU_CONSUMER_ND_VIEWS_REQUIRED'
    proof.update(
        original_returned_once=True, input=tensor_geometry(value),
        outputs=[tensor_geometry(t) for t in outputs], formats=formats,
        scalars={'epsilon': bound['epsilon']} if rms else {'dim': bound['dim']})
    if rms:
        proof['gamma'] = tensor_geometry(gamma)
    return proof, (value,), None
