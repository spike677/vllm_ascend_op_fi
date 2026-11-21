"""Batch-shared INT8 quantization arithmetic, without operator interception."""
import torch


def shared_dynamic_quant(rows, *, arithmetic_device='input'):
    """Return contiguous Q[N,D], FP32 scale[N]; sites validate native layout.

All real rows of this call participate. Request count and padding removal are
site responsibilities, not inferred from N. Zero input has zero scale and Q.
"""
    if rows.ndim != 2 or min(rows.shape) <= 0 or not rows.is_floating_point():
        raise ValueError('REAL_FLOATING_POINT_ROWS_REQUIRED')
    if arithmetic_device not in ('input','cpu'):
        raise ValueError('UNKNOWN_REFERENCE_ARITHMETIC_DEVICE')
    # Explicit portable reference path: do not assume device division/rounding
    # is bit-identical near half-integer quantization boundaries.
    x = rows.float() if arithmetic_device=='input' else rows.detach().float().cpu()
    if not torch.isfinite(x).all().item():
        raise ValueError('NONFINITE_DYNAMIC_QUANT_INPUT')
    scale = x.abs().amax() / 127
    divisor = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(x / divisor).clamp(-127, 127).to(torch.int8).contiguous()
    return q.to(rows.device), scale.expand(rows.shape[0]).contiguous().to(rows.device)
