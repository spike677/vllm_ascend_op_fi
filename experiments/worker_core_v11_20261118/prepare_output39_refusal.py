"""Emit one reviewed isolated case; this module never executes it."""
import argparse
import json
from pathlib import Path

from experiments.c2c3_postwrite_20261117.prepare39_worker_core import core_cases
from experiments.worker_core_v11_20261118.output39_refusal_diagnostic import DiagnosticPlan
from experiments.worker_core_v11_20261118.output39_refusal_probe import phase_budget, original


def cases(*, source, native, sha256, build_fingerprint, model, card=2):
    row, = core_cases(source=source,native=native,sha256=sha256,build_fingerprint=build_fingerprint,
                     model=model,card=card,stage='c2_c3',max_live_leases=128)
    row['id'] = 'output39-fixed-c3-refusal-diagnostic'
    row['timeout'] = 1800  # Retain the existing queue hard budget; soft window is 1500s.
    row['command'] = ('source /usr/local/Ascend/cann-8.5.1/set_env.sh\nexec '
        'python3 -u /source/experiments/worker_core_v11_20261118/output39_refusal_probe.py '
        '--model /model --library /native/libdcat_postwrite.so --sha256 '+sha256+
        ' --build-fingerprint '+build_fingerprint+' --out /evidence/probe --memory .35 --deadline-seconds 1500 '
        '> /evidence/output39-refusal.stdout.log 2> /evidence/output39-refusal.stderr.log')
    expected = list(original.PHASES)+['c2_lossless']+['c2_fault']*28+['c2_off','c2_health'] \
        + ['c3_lossless','c3_fault','c3_fault','c3_off','c3_health']
    schedule = phase_budget(expected,28)
    schedule['derivation'] = 'PREDECLARED_CURRENT_PUBLIC_PLAN; MUST_MATCH_ACTUAL_PROVIDER_PLAN_AT_INSTALL'
    row['review'].update(experiment_id=row['id'],
        acceptance='ONE_REFUSAL_DIAGNOSTIC_ONLY_C3_AND_E5_REMAIN_FALSE',
        expected_exit_code=2, expected_hits='Original fixed all-target protocol; Native apply4 stays failure',
        diagnostic_plan=DiagnosticPlan().__dict__, maximum_user_requests_including_health=43,
        phase_budget=schedule,
        maximum_additional_observations=5, statistics='NONE; Native BF16 diagnostic, not E5',
        recovery_plan='Same public driver abort/recover/OFF/health/unregister/detach. '
            'No F retry or conditional every-match. Preserve original apply error and uncertain owners.',
        resource_budget=dict(timeout_seconds=1800,soft_fault_window_seconds=1500,memory_gib=32,cpus=8),
        raw_evidence=['/evidence/probe/result.json','/evidence/probe/refusal-diagnostic.json',
            '/evidence/probe/host_receipt.json','/evidence/probe/worker_core_rpc','/evidence/output39-runtime',
            '/evidence/output39-refusal.stdout.log','/evidence/output39-refusal.stderr.log'])
    return [row]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source','native','sha256','build-fingerprint','model'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--card',type=int,choices=(2,5),default=2)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args(argv)
    rows = cases(source=args.source,native=args.native,sha256=args.sha256,
        build_fingerprint=args.build_fingerprint,model=args.model,card=args.card)
    with args.out.open('x',encoding='utf-8') as stream:
        json.dump(rows,stream,sort_keys=True,indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
