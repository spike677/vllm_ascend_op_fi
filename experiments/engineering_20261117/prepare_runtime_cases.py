"""Prepare immutable bounded natural Worker or actual collective cases."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--installed',type=Path)
    p.add_argument('--native',type=Path);p.add_argument('--kind',choices=['26','33','25'],required=True)
    p.add_argument('--model',default='/home/haoran/models/DeepSeek-V2-Lite-W8A8')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    env=dict(OMP_NUM_THREADS='4',VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_WORKER_MULTIPROC_METHOD='spawn',
        VLLM_ALLOW_INSECURE_SERIALIZATION='1',VLLM_HOST_IP='127.0.0.1',GLOO_SOCKET_IFNAME='lo',
        HCCL_CONNECT_TIMEOUT='120',HCCL_EXEC_TIMEOUT='120',XDG_CACHE_HOME='/evidence/cache',
        VLLM_CACHE_ROOT='/evidence/cache/vllm')
    mounts=[[str(a.source),'/source','ro']]
    command='source /usr/local/Ascend/cann-8.5.1/set_env.sh\n'
    if a.kind=='25':
        env['PYTHONPATH']='/source:/source/fault39/src:/source/src'
        command+='exec torchrun --nnodes=1 --master-addr=127.0.0.1 --master-port=29500 --nproc-per-node=2 /source/experiments/engineering_20261117/collective25_probe.py --out /evidence'
    else:
        if a.installed is None:raise ValueError('INSTALLED_PACKAGES_REQUIRED')
        mounts += [[str(a.installed),'/installed','ro'],[a.model,'/model','ro']]
        command+='export PYTHONPATH=/installed:/msprobe_packages:$PYTHONPATH\n'
        if a.kind=='33':
            command+='exec python3 /source/experiments/engineering_20261117/paged33_worker_audit.py --memory .55'
        else:
            if a.native is None:raise ValueError('NEW_NATIVE_REQUIRED')
            mounts += [[str(a.native),'/native','ro'],[str(a.native/'libopapi_nn.so'),
                '/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64/libopapi_nn.so','ro']]
            env.update(FAULT26_TRACE_ENABLED='1',FAULT26_FIA_WRITER_PROXY_ALLOWED='1',
                FAULT26_WRITER_TOPOLOGY_ENABLED='1',FAULT26_QMM_UNDERFLOW_LOG_PATH='/evidence/writer.jsonl',
                FAULT26_TRACE_CAPTURE_TENSORS='1',FAULT26_TRACE_MAX_EVENTS='100000',
                FAULT26_QMM_UNDERFLOW_ALLOWED='1',FAULT26_TRACE_LOG_PATH='/evidence/trace.jsonl',
                FAULT26_FIA_WRITER_LOG_PATH='/evidence/writer.jsonl')
            command+='export ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend\n'
            command+='export LD_PRELOAD=/native/libfault26_readonly_trace_abi0.so:/native/libfault26_fia_writer_proxy_abi0.so:/native/libfault26_qmm_output_underflow_abi0.so:/native/libfault26_fia_opapi_proxy.so:/native/libopapi.so:/native/libcust_opapi.so\n'
            command+='export LD_LIBRARY_PATH=/native:/native/real:$LD_LIBRARY_PATH\n'
            command+='exec python3 /source/experiments/engineering_20261117/writer26_worker_probe.py --mechanism 26.residual_stomp --statistics --memory .65'
    cases=[dict(id='fault'+a.kind+'-natural',image='quay.io/ascend/vllm-ascend:v0.18.0',
                timeout=1200,env=env,mounts=mounts,command=command)]
    with a.out.open('x') as stream:json.dump(cases,stream,indent=2)


if __name__=='__main__':main()
