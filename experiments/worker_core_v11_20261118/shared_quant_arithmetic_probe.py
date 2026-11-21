"""Small same-formula arithmetic diagnostic; no model, weights or injection."""
import json
from pathlib import Path
import numpy as np
import torch
import torch_npu
from fault_injection.engines.dynamic_quant import shared_dynamic_quant
from fault_injection.sites.batch_quant import cpu_reference

torch.npu.set_device(0)
rows=[]
generator=torch.Generator().manual_seed(42)
for i in range(24):
    maximum=(.001,.1,.3,.7,1.13,3.14,8.125,2.015625)[i%8]
    host=(torch.randn(88,1024,generator=generator)*maximum).to(torch.bfloat16)
    x=host.npu()
    q,s=shared_dynamic_quant(x);rq,rs=cpu_reference(x)
    aq,as_=q.cpu().numpy(),s.cpu().numpy()
    rows.append(dict(maximum=maximum,q_changed=int(np.count_nonzero(aq!=rq)),
        scale_changed=int(np.count_nonzero(as_!=rs)),actual_scale=float(as_[0]),reference_scale=float(rs[0]),
        scale_abs_error=float(np.max(np.abs(as_-rs)))))
Path('/evidence/arithmetic.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows))
