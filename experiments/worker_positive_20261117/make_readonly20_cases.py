import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',required=True);p.add_argument('--native',required=True)
p.add_argument('--math-proxy',action='store_true')
p.add_argument('--model-path',type=Path)
p.add_argument('--model-label',default='qwen')
p.add_argument('--operator',choices=['aclnnMatmulWeightNz','aclnnQuantMatmulWeightNz'],default='aclnnMatmulWeightNz')
p.add_argument('--memory',type=float,default=.35)
a=p.parse_args();root=Path('/home/worker-positive-20261117')
if not all(x.replace('-','').replace('.','').isalnum() for x in (a.base,a.out,a.native,a.model_label)):raise ValueError('Invalid label')
case=json.loads((root/a.base).read_text())[0]
case['id']='readonly20-controlled-pta'
if a.model_path:
    if not a.model_path.is_absolute() or not a.model_path.is_dir():raise ValueError('Existing model directory required')
    case['id']='readonly20-'+a.model_label
case['env'].update(VLLM_ASCEND_ENABLE_NZ='2',DEMONCAT_F20_REQUIRE_TYPED_GWS='1',
                   DEMONCAT_F20_DIAGNOSTICS_PATH='/evidence/current-call.jsonl')
for mount in case['mounts']:
    if mount[1]=='/native':mount[0]=str(root/a.native/'build20')
    elif mount[1].endswith('/libopapi_nn.so'):mount[0]=str(root/a.native/'build20/libopapi_nn.so')
    elif mount[1]=='/probe.py':mount[0]=str(root/'readonly20_probe.py')
    elif mount[1]=='/model' and a.model_path:mount[0]=str(a.model_path.resolve())
case['command']=case['command'].replace('libfault39_native_v1.so','libdcat20_qmm.so')
case['command']=case['command'].replace(' --statistics','')
case['command']=case['command'].replace(' --baseline-control','')
case['command']+=' --operator '+a.operator+' --memory '+str(a.memory)
if a.math_proxy:
    case['mounts'].append([str(root/a.native/'build20/libopapi_math.so'),
                          '/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64/libopapi_math.so','ro'])
with (root/a.out).open('x') as stream:json.dump([case],stream,indent=2)
