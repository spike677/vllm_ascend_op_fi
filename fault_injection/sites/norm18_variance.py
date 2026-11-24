"""Whole-output gain proxy for RMS mean-square error, not kernel replay.

The fixed named variant models k=4 with gain=1/sqrt(k). It does not claim
the exact epsilon-dependent ratio or a centered-variance implementation.
Only dense live output regions with a proven full-region reader are supported.
"""
import torch
from torch.utils._python_dispatch import _disable_current_modes
from .norm18 import Norm18Site, NormConsumerMode
from .projection39 import tensors
from .mla28 import require

MECHANISM = '18.variance_over4'


def storage_span(t):
    if t.numel() == 0:
        return (0, 0)
    require(all(s >= 0 for s in t.stride()), 'NORM_NEGATIVE_STRIDE_UNSUPPORTED')
    start = t.storage_offset()
    return start, start + 1 + sum((n-1)*s for n, s in zip(t.shape, t.stride()))


class FullNormConsumerMode(NormConsumerMode):
    """Reuse lease completion but never infer a full read from one element."""
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        schema = func._schema
        view = any(r.alias_info is not None for r in schema.returns) and not schema.is_mutable
        metadata = schema.name in ('aten::empty_like', 'aten::zeros_like', 'aten::ones_like',
            'aten::full_like', 'aten::new_empty', 'aten::new_empty_strided', 'aten::new_zeros',
            'aten::new_ones', 'aten::new_full', 'aten::is_pinned')
        matched = []
        for lease in self.pending:
            region = lease['region']; rs, re = storage_span(region)
            full = False; overlap = False
            for i, a in enumerate(schema.arguments):
                value = args[i] if i < len(args) else kwargs.get(a.name)
                for t in tensors(value):
                    if t.device != region.device or t.untyped_storage()._cdata != region.untyped_storage()._cdata:
                        continue
                    require(t.dtype == region.dtype, 'NORM_READER_DTYPE_REINTERPRETATION')
                    ts, te = storage_span(t)
                    if max(rs, ts) >= min(re, te):
                        continue
                    overlap = True
                    require(a.alias_info is None or not a.alias_info.is_write,
                            'NORM_READER_WRITES_PROTECTED_REGION')
                    # A dense permutation is allowed; holes/overlapping strides are not.
                    stride = 1; dense = True
                    for s, n in sorted((s,n) for n,s in zip(t.shape,t.stride()) if n > 1):
                        dense = dense and s == stride; stride *= n
                    full = full or (dense and ts <= rs and te >= re)
            if overlap and not view and not metadata:
                require(full, 'NORM_PARTIAL_REGION_READER_UNSUPPORTED')
                matched.append(lease)
        result = func(*args, **kwargs)
        self.complete_consumers(matched, schema.name)
        if schema.name == 'vllm::triton_split_qkv_rmsnorm_mrope':
            require(len(self.site.parents) == 1, 'FUSED_NORM_PARENT_UNPROVEN')
            require(isinstance(result, (tuple,list)) and len(result) == 4, 'FUSED_NORM_OUTPUT_CHANGED')
            for suffix, t in zip(('q_norm','k_norm'), result[:2]):
                self.site.output(self.site.parents[0]+'.fused.'+suffix, t)
        return result


class Variance18Site(Norm18Site):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = FullNormConsumerMode(self)

    def output(self, name, t):
        require(self.active and self.forward_calls, 'NORM_OUTSIDE_MODEL')
        require(t.dtype == torch.bfloat16 and t.numel() > 0 and t.is_contiguous(),
                'VARIANCE_NORM_DENSE_BF16_OUTPUT_REQUIRED')
        calls = self.forward_calls[-1]; calls[name] = calls.get(name, 0)+1
        with _disable_current_modes():
            region = t.view(-1)
            e = dict(site=name, request_ids=list(self.runner.input_batch.req_ids), eligible=True,
                     applied=False, changed=0, changed_elements=0, consumed=False, restored=False,
                     failed=False, effect='RMS_MEAN_SQUARE_GAIN_PROXY', named_variant=MECHANISM,
                     mean_square_multiplier=4, gain=0.5, epsilon_exact=False, shape=list(t.shape),
                     covered_elements=t.numel())
            self.events.append(e)
            def replacement(x):
                value = x*0.5
                e['changed_elements'] = int((x.view(torch.uint8).reshape(-1, x.element_size()) !=
                    value.view(torch.uint8).reshape(-1, x.element_size())).any(dim=1).sum().item())
                return value
            self.mode.watch(region, e, inject=self.fault, replacement=replacement)
