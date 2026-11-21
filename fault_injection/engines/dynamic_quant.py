"""Batch-shared INT8 quantization arithmetic, without operator interception."""
import torch


def compensated_scale(maximum, initial=None):
    """Correct device approximate division using an exact nearby residual.

    127 = 128 - 1. For ordinary finite normal FP32 magnitudes, the two
    cancellation subtractions retain the division residual: m - 128*s + s.
    One correction is then rounded back to FP32. The site still verifies the
    resulting scalar exactly against independent CPU division; this formula
    is not an exemption from that check or a promise for underflow/overflow.
    """
    s=maximum/127 if initial is None else initial
    return s+((maximum-s*128)+s)/127


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
    maximum=x.abs().amax()
    scale = compensated_scale(maximum) if arithmetic_device=='input' else maximum/127
    if not torch.isfinite(scale).item() or (maximum>0).item() and not (scale>0).item():
        raise ValueError('SHARED_SCALE_OUTSIDE_FINITE_NORMAL_ARITHMETIC_RANGE')
    divisor = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(x / divisor).clamp(-127, 127).to(torch.int8).contiguous()
    return q.to(rows.device), scale.expand(rows.shape[0]).contiguous().to(rows.device)
