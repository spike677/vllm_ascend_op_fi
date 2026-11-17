"""Freeze one reviewed final-launch observation case from actual build receipts."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'registered', 'schema', 'observer', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--phase', choices=('baseline', 'fault', 'off_recovery'), required=True)
    parser.add_argument('--card', type=int, choices=(2, 5), default=2)
    args = parser.parse_args()
    completion = args.registered/'completion.json'
    build = json.loads(completion.read_bytes())
    variant = build['variant']
    vendor = {'legal_tiling': 'dcat_f28_legal_transformer',
              'denominator_replay': 'dcat_f28_replay_transformer'}[variant]
    schema = args.schema/'fia_abi_schema.json'
    observer = args.observer/'attention-preflight/result.json'
    observed = json.loads(observer.read_bytes())
    if not observed.get('observer_compiled') or observed.get('actual_npu_calls') != 0:
        raise ValueError('ACTUAL_DEVICE_FREE_OBSERVER_BUILD_REQUIRED')
    name = 'fia-' + ('legal' if variant == 'legal_tiling' else 'replay') + '-' + args.phase.replace('_', '-')
    command = ('source /usr/local/Ascend/cann-9.1.0/set_env.sh\n'
        'exec python3 /repo/experiments/c2c3_closure_20261117/attention28_launch_probe.py '
        '--completion /registered/completion.json --completion-sha256 '+sha(completion)+
        ' --vendor /registered/vendors/'+vendor+
        ' --public-schema /schema/fia_abi_schema.json --schema-sha256 '+sha(schema)+
        ' --observer-receipt /build/attention-preflight/result.json --observer-receipt-sha256 '+sha(observer)+
        ' --phase '+args.phase+' --batch 8 --out /evidence/run --experimental-qualification'+
        (' --statistics' if args.phase != 'off_recovery' else ''))
    case = dict(id=name, image='cann:9.1.0-910b-torch2.10.0_npu-atk26.7.8', timeout=900,
        mounts=[[str(args.source), '/repo', 'ro'], [str(args.registered), '/registered', 'ro'],
                [str(args.schema), '/schema', 'ro'], [str(args.observer), '/build', 'ro']],
        env=dict(PYTHONPATH='/repo:/repo/src:/repo/fault39/src:/msprobe_packages', OMP_NUM_THREADS='4', ACLNN_CACHE_LIMIT='0'),
        command=command, review=dict(experiment_id=name, provider='28.fia.'+variant,
            model='standalone fixed FIA input; no model qualification', cann='9.1.0', soc='Ascend910B4/A2',
            physical_cards=[args.card], ranks=1, worker='standalone synchronous API',
            build_fingerprint=sha(completion), expected_hits=2 if args.phase == 'fault' else 0,
            resource_budget=dict(timeout_seconds=900, memory_gib=32, cpus=8),
            recovery_plan='Synchronous bounded calls; restore owned API wrapper. off_recovery runs B/F/OFF in the same process; no public C2/C3 grant.'))
    with args.out.open('x') as stream:
        json.dump([case], stream, indent=2)


if __name__ == '__main__':
    main()
