"""One real projection cell: omit the final quarter of its K accumulation.

CPU FP64 reference is bounded to two vectors, never a whole weight copy.
This is an explicitly incorrect accumulation-range proxy, not a native kernel.
"""
import torch


def recompute_cell(left, right, *, scale=1., bias=0.):
    if left.ndim!=1 or right.shape!=left.shape or left.numel()<4:
        raise ValueError('MATMUL_K_VECTORS_REQUIRED')
    x=left.detach().cpu().double();w=right.detach().cpu().double()
    if not torch.isfinite(x).all() or not torch.isfinite(w).all():
        raise ValueError('MATMUL_NONFINITE_OPERANDS')
    cut=3*len(x)//4
    full=(x*w).sum()*scale+bias
    faulty=(x[:cut]*w[:cut]).sum()*scale+bias
    return full,faulty,dict(k=len(x),included_k=cut,omitted_k=len(x)-cut,
                           rule='omit_final_quarter',arithmetic='CPU_FP64')
