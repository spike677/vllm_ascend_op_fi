"""Prepare two bounded natural Worker B/F cases; never execute NPU work."""
import argparse
import json
from pathlib import Path
import re


def cases(*,source,native,sha256,build_fingerprint,model,cards=(2,),statistics=True,max_seconds=600,expected_targets=28):
    if any(re.fullmatch('[0-9a-f]{64}',value) is None for value in (sha256,build_fingerprint)):
        raise ValueError('ACTUAL_BUILD_IDENTITIES_REQUIRED')
    if type(max_seconds) is not int or not 0<max_seconds<=1500:
        raise ValueError('BOUNDED_SESSION_REQUIRED')
    if type(expected_targets) is not int or not 0<expected_targets<=64:
        raise ValueError('BOUNDED_ACTUAL_UNIVERSE_REQUIRED')
    rows=[]
    for arm in ('baseline','fault'):
        name='postwrite41-c2-'+arm
        command=('source /usr/local/Ascend/cann-8.5.1/set_env.sh\n'
            'exec python3 /source/experiments/c2c3_postwrite_20261117/natural41_coverage_probe.py'
            ' --model /model --library /native/libdcat_postwrite.so --sha256 '+sha256+
            ' --build-fingerprint '+build_fingerprint+' --out /evidence --arm '+arm+
            ' --max-seconds '+str(max_seconds)+' --expected-targets '+str(expected_targets))
        if statistics:command+=' --statistics'
        if arm=='fault':command+=' --try-c3'
        rows.append(dict(id=name,image='quay.io/ascend/vllm-ascend:v0.18.0',timeout=1800,
            mounts=[[source,'/source','ro'],[native,'/native','ro'],[model,'/model','ro']],
            env={'PYTHONPATH':'/source/src:/source/fault39/src:/source:/msprobe_packages','OMP_NUM_THREADS':'4',
                 'VLLM_ENABLE_V1_MULTIPROCESSING':'0','VLLM_WORKER_MULTIPROC_METHOD':'spawn'},command=command,
            review=dict(experiment_id=name,provider='41.tail_column',model=model,cann='8.5.1',soc='Ascend910B4/A2',
                physical_cards=list(cards),ranks=1,worker='standard vLLM Worker TP1/PP1/DP1 serial eager first pure decode',
                build_fingerprint=build_fingerprint,
                expected_hits='fresh C1 1; measured C2 '+('0 controls' if arm=='baseline' else
                    str(expected_targets)+' once; then C3 '+str(2*expected_targets)+' every-match'),
                resource_budget=dict(timeout_seconds=1800,memory_gib=32,cpus=8),
                recovery_plan='per-FIA native drain/restore/release, all-target OFF, whole group barrier, remove owned hooks/APIs, exact separate normal-request health')))
    return rows


def main():
    p=argparse.ArgumentParser()
    for name in ('source','native','sha256','build-fingerprint'):p.add_argument('--'+name,required=True)
    p.add_argument('--model',default='/home/gyl/models/Qwen3-0.6B')
    p.add_argument('--cards',nargs='+',type=int,default=[2])
    p.add_argument('--max-seconds',type=int,default=600)
    p.add_argument('--expected-targets',type=int,default=28)
    p.add_argument('--without-statistics',action='store_true')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    rows=cases(source=a.source,native=a.native,sha256=a.sha256,build_fingerprint=a.build_fingerprint,
        model=a.model,cards=a.cards,statistics=not a.without_statistics,max_seconds=a.max_seconds,
        expected_targets=a.expected_targets)
    with a.out.open('x') as stream:json.dump(rows,stream,indent=2)


if __name__=='__main__':main()
