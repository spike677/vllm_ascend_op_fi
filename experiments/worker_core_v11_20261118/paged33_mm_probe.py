"""NPU capability/control check for BF16-input FP32-output matrix products."""
from pathlib import Path
import torch,torch_npu,json
from fault_injection.engines.paged33_native_control import split_fuse_control,agreement

torch.npu.set_device(0);torch.npu.matmul.allow_hf32=False
rows=[]
for folder in ('/private_qkv','/failed_npu','/failed_cpu'):
    for p in sorted(Path(folder).glob('*.pt')):
        d=torch.load(p,map_location='cpu',weights_only=True)
        q,k,v=[d[x].npu() for x in ('q','k','v')];native=d['output'].npu();n=len(k)
        r={'group':folder,'file':p.name,'metadata':d['metadata']}
        try:
            c=split_fuse_control(q,k,v,scale=d['metadata']['scale'],split_spans=[(i,min(i+512,n)) for i in range(0,n,512)],matmul_mode='bf16_out_fp32')
            torch.npu.synchronize();r['agreement']=agreement(c.reshape_as(native),native)
        except Exception as error:r['error']=str(error)[:3000]
        rows.append(r);print(json.dumps(r),flush=True)
Path('/evidence/mm_capability.json').write_text(json.dumps({'results':rows,'qualification_granted':False},indent=2))
