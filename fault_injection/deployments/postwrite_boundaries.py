"""Observed producer geometry and selected-byte consumer lineage for postwrite.

These checks describe the return tensor of the actual operation. They do not
reinterpret linear/matmul as MM, lower an operator, or manufacture a consumer.
"""
from __future__ import annotations

import hashlib
from copy import deepcopy

import torch


PROJECTION_OPERATORS = frozenset({
    'aten.mm.default', 'aten.linear.default', 'aten.matmul.default',
    'vllm.unquantized_gemm.default',
})
VALUE_CONSUMERS = frozenset({
    'aten.add.Tensor', 'aten.mul.Tensor', 'aten.mm.default', 'aten.bmm.default',
    'aten.addmm.default', 'aten.sum.dim_IntList', 'aten.linear.default',
    'aten.matmul.default', 'aten._softmax.default', 'aten.softmax.int',
    'aten.silu.default', 'aten.gelu.default',
})


def tensor_geometry(tensor):
    return dict(shape=list(tensor.shape), stride=list(tensor.stride()),
                storage_offset=tensor.storage_offset(), dtype=str(tensor.dtype),
                device=str(tensor.device), data_ptr=tensor.data_ptr(),
                storage_identity=tensor.untyped_storage()._cdata,
                storage_bytes=tensor.untyped_storage().nbytes(),
                logical_bytes=tensor.numel() * tensor.element_size())


def _broadcast(left, right):
    out = []
    for i in range(1, max(len(left), len(right)) + 1):
        a = left[-i] if i <= len(left) else 1
        b = right[-i] if i <= len(right) else 1
        if a != b and a != 1 and b != 1:
            return None
        out.append(max(a, b))
    return tuple(reversed(out))


def projection_return(func, args, kwargs, result, *, gemm_binding=None, expected_weight=None):
    """Return an operation-specific proof, or a precise rejection reason.

    Only a fresh, exclusive BF16 result is considered. Storage bounds/format
    remain the transport's responsibility. No copy of ``result`` is targeted.
    """
    operation = str(func)
    if operation not in PROJECTION_OPERATORS:
        return None, 'OPERATOR_NOT_ADAPTED'
    custom = operation == 'vllm.unquantized_gemm.default'
    registered = None
    if custom:
        # This is the actual custom operator, never an inferred aten dispatch.
        # Reuse the audited file/code/Library/PrivateUse1 ownership checks; a
        # matching name/schema or a numerically equal weight is insufficient.
        if gemm_binding is None:
            return None, 'CUSTOM_GEMM_SOURCE_BINDING_REQUIRED'
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import ContextRefused
        try:
            pair = gemm_binding.arguments(func, args, kwargs, expected_weight)
        except (ContextRefused, OSError, AttributeError) as error:
            return None, 'CUSTOM_GEMM_BINDING_REFUSED:' + str(error)
        if pair is None:
            return None, 'CUSTOM_GEMM_OWNED_WEIGHT_AND_NO_BIAS_REQUIRED'
        registered = deepcopy(gemm_binding.last_proof)
        # The lowering source is reviewed, but this return-storage adapter
        # makes no claim about a native MatMul cache or its execution identity.
        registered.pop('native_self', None)
        registered.pop('native_mat2', None)
    actual = dict(kwargs)
    for parameter, value in zip(func._schema.arguments, args):
        if parameter.name in actual:
            return None, 'DUPLICATE_PROJECTION_ARGUMENT'
        actual[parameter.name] = value
    names = ('x', 'weight') if custom else (('input', 'weight') if operation == 'aten.linear.default'
        else ('self', 'mat2' if operation == 'aten.mm.default' else 'other'))
    left, right = (actual.get(name) for name in names)
    if not all(isinstance(t, torch.Tensor) and t.dtype == torch.bfloat16 for t in (left, right)):
        return None, 'PROJECTION_BF16_OPERANDS_REQUIRED'
    if left.device != right.device or not isinstance(result, torch.Tensor):
        return None, 'PROJECTION_RETURN_OR_DEVICE_MISMATCH'
    if result.dtype != left.dtype or result.device != left.device:
        return None, 'PROJECTION_RETURN_OR_DEVICE_MISMATCH'
    a, b = tuple(left.shape), tuple(right.shape)
    if operation == 'aten.linear.default' or custom:
        if not a or len(b) != 2 or a[-1] != b[1]:
            return None, 'LINEAR_INPUT_WEIGHT_GEOMETRY_MISMATCH'
        bias = actual.get('bias')
        if bias is not None and (not isinstance(bias, torch.Tensor) or tuple(bias.shape) != (b[0],)
                                 or bias.dtype != result.dtype or bias.device != result.device):
            return None, 'LINEAR_BIAS_GEOMETRY_MISMATCH'
        expected = a[:-1] + (b[0],)
        kind = 'reviewed_vllm_custom_gemm_return' if custom else 'linear_affine_return'
    elif operation == 'aten.mm.default':
        if len(a) != 2 or len(b) != 2 or a[1] != b[0]:
            return None, 'MM_OPERAND_GEOMETRY_MISMATCH'
        expected = (a[0], b[1])
        kind = 'matrix_matrix_return'
    else:
        if not a or not b or a[-1] != (b[-2] if len(b) > 1 else b[-1]):
            return None, 'MATMUL_CONTRACTION_GEOMETRY_MISMATCH'
        batches = _broadcast(a[:-2], b[:-2])
        if batches is None:
            return None, 'MATMUL_BATCH_BROADCAST_MISMATCH'
        expected = batches + ((a[-2],) if len(a) > 1 else ()) + ((b[-1],) if len(b) > 1 else ())
        kind = 'matmul_broadcast_return'
    if tuple(result.shape) != expected:
        return None, 'PROJECTION_RETURN_GEOMETRY_MISMATCH'
    operands = [value for value in actual.values() if isinstance(value, torch.Tensor)]
    if any(result.untyped_storage()._cdata == item.untyped_storage()._cdata for item in operands):
        return None, 'PROJECTION_RETURN_ALIASES_OPERAND'
    proof = dict(operator=operation, boundary=kind, schema=str(func._schema),
                schema_sha256=hashlib.sha256(str(func._schema).encode()).hexdigest(),
                expected_output_shape=list(expected), operands={name: tensor_geometry(value)
                    for name, value in actual.items() if isinstance(value, torch.Tensor)},
                returned_tensor=tensor_geometry(result), original_return_storage=True)
    if custom:
        proof.update(registered_implementation=registered, native_matmul_claim=False)
    return proof, None


def reads_selected_elements(tensor, indices, producer):
    """Prove selected producer elements occur in an actual read operand view.

    Greedy stride decoding yields only concrete in-bounds coordinates, so
    unusual layouts may be refused but cannot create false alias matches.
    This includes ordinary reshape/transpose chains without counting them as
    value consumers, and rejects a disjoint slice of the same allocation.
    """
    if (tensor.device != producer.device or tensor.dtype != producer.dtype
            or tensor.untyped_storage()._cdata != producer.untyped_storage()._cdata):
        return False
    dimensions = sorted(((int(stride), int(size)) for stride, size in zip(tensor.stride(), tensor.shape)
                         if stride), reverse=True)
    if any(stride < 0 for stride, _ in dimensions):
        return False
    for index in indices:
        remainder = producer.storage_offset() + index - tensor.storage_offset()
        if remainder < 0:
            return False
        for stride, size in dimensions:
            coordinate, remainder = divmod(remainder, stride)
            if coordinate >= size:
                return False
        if remainder:
            return False
    return bool(indices)
