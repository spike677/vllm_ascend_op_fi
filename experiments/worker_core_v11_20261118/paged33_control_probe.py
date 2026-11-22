"""Fixed real QKV replay on one NPU; no model/admission qualification."""
import hashlib,json
from pathlib import Path
import torch,torch_npu
from fault_injection.engines.paged33_native_control import split_fuse_control,agreement

def main():
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32=False
    results=[]
    for p in sorted(Path('/private_qkv').glob('private-qkv-*.pt')):
        d=torch.load(p,map_location='cpu',weights_only=True)
        q,k,v=[d[x].npu() for x in ('q','k','v')]
        native=d['output'].npu();n=len(k)
        spans=[(i,min(i+512,n)) for i in range(0,n,512)]
        controls=[split_fuse_control(q,k,v,scale=d['metadata']['scale'],split_spans=spans) for _ in range(2)]
        torch.npu.synchronize()
        row=dict(file=p.name,input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),spans=spans,
            device='A2_NPU_CONTROL',control=agreement(controls[0],native),repeat=agreement(controls[1],controls[0]),alternatives=[])
        if row['control']['checkpoint_99pct_passed']:
            for block in (256,1024):
                other=[(i,min(i+block,n)) for i in range(0,n,block)]
                changed=split_fuse_control(q,k,v,scale=d['metadata']['scale'],split_spans=other)
                torch.npu.synchronize()
                row['alternatives'].append(dict(block=block,same_key_set=True,against_control=agreement(changed,controls[0]),against_native=agreement(changed,native)))
        results.append(row);print(json.dumps(row),flush=True)
    Path('/evidence/control.json').write_text(json.dumps(dict(results=results,qualification_granted=False),indent=2))
    assert len(results)==2 and all(r['control']['checkpoint_99pct_passed'] for r in results),'CONTROL_CHECKPOINT_NOT_PASSED'

if __name__=='__main__':main()
