"""Research control for the observed A2 BF16 paged split-fuse branch.

No injection or grant. BF16 probabilities, FP32 partial outputs/LSE, ordered
split merge and final BF16 round are modeled. Hardware instruction equivalence
must be measured; these torch operations alone do not prove it.
"""
import torch


def split_fuse_control(q,k,v,*,scale,split_spans,matmul_mode="fp32"):
    if q.ndim!=3 or q.shape[0]!=1 or k.ndim!=3 or k.shape!=v.shape or q.shape[-1]!=k.shape[-1]:raise ValueError('CONTROL_GEOMETRY')
    if any(t.dtype!=torch.bfloat16 for t in (q,k,v)):raise ValueError('CONTROL_BF16_REQUIRED')
    heads=q.shape[1];kv_heads=k.shape[1];group=heads//kv_heads
    if heads%kv_heads or not split_spans or split_spans[0][0]!=0 or split_spans[-1][1]!=k.shape[0]:raise ValueError('CONTROL_COMPLETE_KEY_SET')
    if any(a>=b for a,b in split_spans) or any(a[1]!=b[0] for a,b in zip(split_spans,split_spans[1:])):raise ValueError('CONTROL_KEY_GAP_OR_OVERLAP')
    if matmul_mode not in ("fp32","bf16_out_fp32"):raise ValueError("CONTROL_MATMUL_MODE")
    def mm(a,b):
        return torch.mm(a,b,out_dtype=torch.float32) if matmul_mode=="bf16_out_fp32" else a.float()@b.float()
    outputs=[]
    for h in range(kv_heads):
        query=q[0,h*group:(h+1)*group];partials=[];lses=[]
        for start,end in split_spans:
            score=mm(query,k[start:end,h].T)*scale
            maximum=score.amax(dim=-1,keepdim=True)
            exp=torch.exp(score-maximum)
            denom=exp.sum(dim=-1,keepdim=True)
            # Original per-core P goes through BF16 while denominator stays FP32.
            product=mm(exp.to(torch.bfloat16),v[start:end,h])
            partials.append(product/denom)
            lses.append((denom.log()+maximum).squeeze(-1))
        lse=torch.stack(lses);maximum=lse.amax(dim=0)
        total=torch.exp(lse-maximum).sum(dim=0).log()+maximum
        weights=torch.exp(lse-total)
        result=partials[0]*weights[0,:,None]
        for i in range(1,len(partials)):result=partials[i]*weights[i,:,None]+result
        outputs.append(result)
    return torch.cat(outputs).to(torch.bfloat16).unsqueeze(0)


def agreement(control,native):
    if control.shape!=native.shape or control.dtype!=native.dtype:raise ValueError('CONTROL_OUTPUT_LAYOUT')
    # Signed zero counts as distinct; compare representation, not torch equal.
    bits=(control.contiguous().view(torch.int16)==native.contiguous().view(torch.int16))
    return dict(elements=bits.numel(),bitwise_equal=int(bits.sum().item()),bitwise_ratio=float(bits.float().mean().item()),
        max_abs=float((control.float()-native.float()).abs().max().item()),
        checkpoint_99pct_passed=bool(bits.sum().item()/bits.numel()>=.99),qualification_granted=False)
