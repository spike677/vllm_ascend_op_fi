"""Private fixed-QKV control, no model output replacement or qualification.

The caller supplies S2 spans decoded from its own native launch, not A2 defaults.
This only tests the named split-fuse arithmetic hypothesis on the actual device.
"""
import argparse,hashlib,json
from pathlib import Path


def validate_spans(spans,length):
    if not isinstance(spans,list) or not spans or any(type(s) is not list or len(s)!=2 or any(type(v) is not int for v in s) for s in spans):
        raise ValueError('SPANS_SCHEMA')
    if spans[0][0]!=0 or spans[-1][1]!=length or any(a>=b for a,b in spans) or any(a[1]!=b[0] for a,b in zip(spans,spans[1:])):
        raise ValueError('KEY_GAP_OR_OVERLAP')


def main():
    p=argparse.ArgumentParser();p.add_argument('qkv');p.add_argument('--launch-review',required=True);p.add_argument('--device',type=int,required=True);a=p.parse_args()
    review_raw=Path(a.launch_review).read_bytes();r=json.loads(review_raw)
    raw=Path(a.qkv).read_bytes();sha=lambda b:hashlib.sha256(b).hexdigest()
    if r.get('qkv_sha256')!=sha(raw) or r.get('schema')!='dcat.paged33.observed_spans.v1' or r.get('operator')!='FIA_paged_decode' or r.get('arithmetic')!='BF16_MM_FP32_OUT_PBF16_LSE_FP32':
        raise ValueError('REVIEW_IDENTITY_OR_ARITHMETIC_UNPROVEN')
    for field in ('launch_sha256','serializer_sha256'):
        if not isinstance(r.get(field),str) or len(r[field])!=64 or any(c not in '0123456789abcdef' for c in r[field]):raise ValueError('REVIEW_HASH_MISSING:'+field)
    import torch,torch_npu
    from fault_injection.engines.paged33_native_control import split_fuse_control,agreement
    d=torch.load(a.qkv,map_location='cpu',weights_only=True)
    validate_spans(r['spans'],len(d['k']))
    torch.npu.set_device(a.device);torch.npu.matmul.allow_hf32=False
    q,k,v=[d[x].to('npu:'+str(a.device)) for x in ('q','k','v')];native=d['output'].to(q.device)
    if any(not torch.isfinite(t).all().item() for t in (q,k,v,native)):raise ValueError('NONFINITE_NATIVE_INPUT_OR_OUTPUT')
    values=[split_fuse_control(q,k,v,scale=d['metadata']['scale'],split_spans=r['spans'],matmul_mode='bf16_out_fp32') for _ in range(2)]
    torch.npu.synchronize()
    checks=[agreement(c,native) for c in values];repeat=agreement(*values)
    passed=all(c['checkpoint_99pct_passed'] for c in checks) and repeat['bitwise_equal']==repeat['elements']
    print(json.dumps(dict(schema='dcat.paged33.checkpoint.v1',passed=passed,qkv_sha256=sha(raw),launch_review_sha256=sha(review_raw),
        native_checks=checks,repeat=repeat,spans=r['spans'],device=torch.npu.get_device_name(a.device),qualification_granted=False),indent=2))
    return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
