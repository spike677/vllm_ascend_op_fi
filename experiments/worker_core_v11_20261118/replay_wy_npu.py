"""Private isolated operator replay. Allocation controls are not a released fix."""
import argparse,json,pathlib,hashlib,inspect
import torch,torch_npu
from diagnose_b8_nan import allocation_source,stats
from replay_wy_reference import reference,compare

def main():
    p=argparse.ArgumentParser();p.add_argument('capture');p.add_argument('--allocation',choices=['empty','zero','nan'],required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.npu.set_device(0)
    import vllm_ascend.ops.triton.fla.wy_fast as wy
    cap=torch.load(a.capture,map_location='cpu',weights_only=True);assert not cap['args']
    expected=reference(cap['kwargs'])
    inputs={k:v.npu() if isinstance(v,torch.Tensor) else v for k,v in cap['kwargs'].items()}
    original=wy.recompute_w_u_fwd;source=inspect.getsource(original);fn=original
    if a.allocation!='empty':
        source=allocation_source(source,a.allocation);ns=dict(original.__globals__);exec(compile(source,'<allocation-control>','exec'),ns);fn=ns[original.__name__]
    result=dict(allocation=a.allocation,source_sha256=hashlib.sha256(source.encode()).hexdigest(),input_stats=stats(cap['kwargs']),rounds=[])
    for i in range(4):
        outputs=fn(**inputs);torch.npu.synchronize()
        result['rounds'].append(dict(index=i,outputs={name:compare(value.cpu(),ref) for name,value,ref in zip(('w','u'),outputs,expected)}))
    pathlib.Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
