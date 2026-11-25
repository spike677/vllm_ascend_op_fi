"""Fixed numerical phenomenon variants; these do not execute native FP8 kernels."""
import torch


def fp8_like(x, variant):
    """Nearest-even, gradual underflow, finite saturation; preserve NaN/Inf.

    E4M3 uses the finite-only 448 range; E5M2 uses 57344. Arithmetic is
    FP32 and the result returns in the original dtype/layout.
    """
    fraction, minimum, maximum = {'e4m3': (3, -6, 448.),
                                   'e5m2': (2, -14, 57344.)}[variant]
    value = x.float()
    magnitude = value.abs().clamp(max=maximum)
    exponent = torch.floor(torch.log2(magnitude.clamp(min=2.**minimum)))
    quantum = torch.pow(2., exponent-fraction)
    rounded = (torch.round(magnitude/quantum)*quantum).clamp(max=maximum)
    result = torch.copysign(rounded, value)
    return torch.where(torch.isfinite(value), result, value).to(x.dtype)


def truncate_bf16(x, fraction_bits):
    """Truncate BF16 fraction toward zero, retaining sign/exponent and NaNs."""
    if x.dtype != torch.bfloat16 or fraction_bits not in (3, 5):
        raise ValueError('REFERENCE_BF16_FIXED_PRECISION_REQUIRED')
    bits = x.contiguous().view(torch.int16)
    mask = -(1 << (7-fraction_bits))
    result = torch.bitwise_and(bits, mask).view(torch.bfloat16)
    # Truncating a NaN payload must never silently convert it into infinity.
    return torch.where(torch.isfinite(x), result, x)
