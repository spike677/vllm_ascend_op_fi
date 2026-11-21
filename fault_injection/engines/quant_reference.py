"""Independent sampled CPU reference for device shared-scale quantization.

The full input maximum is reduced on device; only that scalar, all scales and
deterministically selected input/Q elements cross to CPU. This is a sample,
not a claim of full CPU elementwise verification. Differences of one code are
permitted only within two float32 ULPs of a half-integer division result.
"""
import numpy as np
import torch


def compare_samples(values, actual_q, actual_scales, absmax, *, total_elements):
    x=np.asarray(values,dtype=np.float32).reshape(-1)
    q=np.asarray(actual_q).reshape(-1)
    scales=np.asarray(actual_scales,dtype=np.float32).reshape(-1)
    maximum=np.float32(absmax)
    if (not x.size or q.shape!=x.shape or q.dtype!=np.int8 or not scales.size
            or not np.isfinite(x).all() or not np.isfinite(maximum) or maximum<0
            or np.max(np.abs(x))>maximum or type(total_elements) is not int
            or total_elements<x.size):
        raise ValueError('INVALID_QUANT_REFERENCE_SAMPLE')
    scale=np.float32(maximum/np.float32(127))
    if not np.array_equal(scales,np.full(scales.shape,scale,dtype=np.float32)):
        raise ValueError('SHARED_SCALE_NOT_CPU_REFERENCE_EXACT:'+str(dict(absmax=float(maximum),
            actual_first_scale=float(scales[0]),reference_scale=float(scale),
            differing_scale_rows=int(np.count_nonzero(scales!=scale)))))
    z=x/(scale if scale else np.float32(1))
    reference=np.clip(np.rint(z),-127,127).astype(np.int8)
    delta=q.astype(np.int16)-reference.astype(np.int16)
    # ULP bound is explicit, independent of observed mismatch magnitude.
    half=np.floor(z)+np.float32(.5)
    boundary=np.abs(z-half)<=2*np.abs(np.spacing(z))
    allowed=(delta==0)|((np.abs(delta)==1)&boundary)
    mismatches=int(np.count_nonzero(delta))
    report=dict(reference_passed=bool(allowed.all()),reference_exact=mismatches==0,
        scale_reference_exact=True,reference_scope='DETERMINISTIC_CPU_SAMPLE',
        total_elements=total_elements,sampled_elements=int(x.size),
        q_sample_differences=mismatches,q_sample_difference_fraction=mismatches/int(x.size),
        forbidden_sample_differences=int(np.count_nonzero(~allowed)),
        boundary_rule='ABS_CODE_DELTA_1_AND_WITHIN_2_FP32_ULP_OF_HALF_INTEGER',
        absmax_reference_source='DEVICE_FULL_REDUCTION_WITH_CPU_SAMPLE_LOWER_BOUND',
        reference_scale=float(scale))
    if not report['reference_passed']:
        raise ValueError('NON_BOUNDARY_QUANT_DIFFERENCE:'+str(report))
    return report


def sampled_reference(rows,q,scales,*,sample_limit=1024):
    if type(sample_limit) is not int or sample_limit<2:
        raise ValueError('INVALID_SAMPLE_BUDGET')
    x=rows.detach().float().reshape(-1);flatq=q.detach().reshape(-1)
    # Include extremal element as well as a uniform deterministic grid.
    n=x.numel();maximum,index=x.abs().max(dim=0)
    grid=torch.linspace(0,n-1,min(n,sample_limit-1),device=x.device).to(torch.int64)
    indexes=torch.unique(torch.cat((grid,index.reshape(1))))
    return compare_samples(x[indexes].cpu().numpy(),flatq[indexes].cpu().numpy(),
        scales.detach().cpu().numpy(),maximum.item(),total_elements=n)
