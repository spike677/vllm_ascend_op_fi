"""Generate reviewed root-queue cases; this script launches no NPU work."""
import argparse
import json
from pathlib import Path
import re


def cases(*, source, native, sha256, build_fingerprint, model, cards=(2,), statistics=True,
          max_seconds=1200, max_live_leases=1):
    if any(re.fullmatch('[0-9a-f]{64}', value) is None for value in (sha256, build_fingerprint)):
        raise ValueError('ACTUAL_BUILD_IDENTITIES_REQUIRED')
    if type(max_seconds) is not int or not 1 <= max_seconds <= 1500:
        raise ValueError('BOUNDED_SESSION_REQUIRED')
    if type(max_live_leases) is not int or not 1 <= max_live_leases <= 256:
        raise ValueError('BOUNDED_LIVE_LEASE_COUNT_REQUIRED')
    prepared = []
    for arm in ('baseline','fault'):
        name = 'postwrite39-c2-' + arm
        command = ('source /usr/local/Ascend/cann-8.5.1/set_env.sh\n'
            'exec python3 /source/experiments/c2c3_postwrite_20261117/natural39_coverage_probe.py'
            ' --model /model --library /native/libdcat_postwrite.so --sha256 ' + sha256 +
            ' --build-fingerprint ' + build_fingerprint + ' --out /evidence --arm ' + arm +
            ' --max-seconds ' + str(max_seconds))
        if statistics:command += ' --statistics'
        if arm == 'fault':
            command += ' --try-c3'
            if max_live_leases > 1:
                command += ' --max-live-leases ' + str(max_live_leases)
        prepared.append(dict(id=name,image='quay.io/ascend/vllm-ascend:v0.18.0',timeout=1800,
            mounts=[[source,'/source','ro'],[native,'/native','ro'],[model,'/model','ro']],
            env={'PYTHONPATH':'/source/src:/source/fault39/src:/source:/msprobe_packages','OMP_NUM_THREADS':'4'},
            command=command,review=dict(experiment_id=name,provider='39.output_proxy',model=model,
                cann='8.5.1',soc='Ascend910B4/A2',physical_cards=list(cards),ranks=1,
                worker='natural pretrained HF MODEL_LOCAL, no vLLM Worker',build_fingerprint=build_fingerprint,
                c3_max_live_leases=max_live_leases if arm == 'fault' else None,
                expected_hits=0 if arm=='baseline' else 'one per independently qualified target; previous C1 observed 140',
                resource_budget=dict(timeout_seconds=1800,memory_gib=32,cpus=8),
                recovery_plan='per-forward Native restore, per-target exact OFF, whole group drain/health; C3 refusal retains denominator')))
    return prepared


def main():
    p=argparse.ArgumentParser()
    for name in ('source','native','sha256','build-fingerprint'):p.add_argument('--'+name,required=True)
    p.add_argument('--model',default='/home/gyl/models/Qwen3-0.6B')
    p.add_argument('--cards',nargs='+',type=int,default=[2])
    p.add_argument('--max-seconds',type=int,default=1200)
    p.add_argument('--max-live-leases', type=int, default=1)
    p.add_argument('--without-statistics',action='store_true')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    result=cases(source=a.source,native=a.native,sha256=a.sha256,build_fingerprint=a.build_fingerprint,
        model=a.model,cards=a.cards,statistics=not a.without_statistics,max_seconds=a.max_seconds,
        max_live_leases=a.max_live_leases)
    with a.out.open('x') as stream:json.dump(result,stream,indent=2)


if __name__=='__main__':main()
