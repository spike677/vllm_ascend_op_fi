"""Immutable case file for one gated, wheel-installed standard Worker."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--packages',required=True);p.add_argument('--native',required=True)
p.add_argument('--out',required=True);p.add_argument('--statistics',action='store_true')
p.add_argument('--baseline-control',action='store_true');a=p.parse_args()
root=Path('/home/worker-positive-20261117')
for x in (a.packages,a.native,a.out):
    if not x.replace('-','').replace('.','').isalnum():raise ValueError('Invalid label')
native=root/a.native/'build39'
case=dict(id='fault39-serving-qwen'+('-statistics' if a.statistics else ''),
    mounts=[(str(root/a.packages/'installed'),'/installed','ro'),
            (str(native),'/native','ro'),
            (str(native/'libopapi_nn.so'),'/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64/libopapi_nn.so','ro'),
            (str(root/'serving_probe.py'),'/probe.py','ro'),('/home/gyl/models/Qwen3-0.6B','/model','ro')],
    timeout=1800,image='quay.io/ascend/vllm-ascend:v0.18.0',
    env=dict(VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_WORKER_MULTIPROC_METHOD='spawn',
             VLLM_ASCEND_ENABLE_NZ='0',OMP_NUM_THREADS='4',VLLM_HOST_IP='127.0.0.1',
             GLOO_SOCKET_IFNAME='lo',HCCL_CONNECT_TIMEOUT='120',HCCL_EXEC_TIMEOUT='120',
             VLLM_ALLOW_INSECURE_SERIALIZATION='1'),
    command='source /usr/local/Ascend/cann-8.5.1/set_env.sh; '
            'export LD_PRELOAD=/native/libfault39_native_v1.so:/native/libopapi.so; '
            'export LD_LIBRARY_PATH=/native:$LD_LIBRARY_PATH; '
            'export PYTHONPATH=/installed:/msprobe_packages:$PYTHONPATH; exec python3 /probe.py'+
            (' --statistics' if a.statistics else '')+(' --baseline-control' if a.baseline_control else ''))
with (root/a.out).open('x') as stream:json.dump([case],stream,indent=2)
