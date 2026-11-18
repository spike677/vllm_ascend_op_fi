"""Prepare one public Core41 case from the existing bounded Worker workload."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]
from experiments.c2c3_postwrite_20261117.prepare41_coverage_cases import cases


def core_cases(*, source, native, sha256, build_fingerprint, model='/home/gyl/models/Qwen3-0.6B',
               card=2, max_seconds=600):
    if card not in (2, 5):
        raise ValueError('ROOT_REVIEWED_CARD_REQUIRED')
    _, row = cases(source=source, native=native, sha256=sha256, build_fingerprint=build_fingerprint,
        model=model, cards=[card], statistics=False, max_seconds=max_seconds)
    row['id'] = 'natural-kv41-worker-core'
    row['command'] = ('source /usr/local/Ascend/cann-8.5.1/set_env.sh\nexec '
        'python3 -u /source/experiments/c2c3_postwrite_20261117/natural41_worker_core_probe.py '
        '--model /model --library /native/libdcat_postwrite.so --sha256 '+sha256+
        ' --build-fingerprint '+build_fingerprint+' --max-seconds '+str(max_seconds)+
        ' --out /evidence/probe --memory .35 '
        '> /evidence/worker-core41.stdout.log 2> /evidence/worker-core41.stderr.log')
    row['review'].update(experiment_id=row['id'],
        provider='WorkerFaultClient+FaultConfig(41,tail_column) reviewed live cache serving group',
        expected_hits='Fresh C1 effect/OFF, complete actual inventory C2 once, original live C2 proof C3 twice; no fixed layer count',
        recovery_plan='17 actual enqueue requests preserve the existing C1/group/OFF/health gates; '
            'public query/readback and unregister/detach followed by one ordinary same-Worker enqueue. '
            'Failure permits at most one explicit OFF plus health; no fault retry, restart or JSON proof. E5 is separate.',
        acceptance='NEW_FROZEN_SOURCE_AND_ACTUAL_WORKER_REVIEW_REQUIRED_NO_INHERITED_CORE41_QUALIFICATION',
        raw_evidence=['/evidence/probe/result.json', '/evidence/probe/host_receipt.json',
                      '/evidence/probe/worker_core_rpc', '/evidence/worker-core41.stdout.log',
                      '/evidence/worker-core41.stderr.log'])
    return [row]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'native', 'sha256', 'build-fingerprint'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--model', default='/home/gyl/models/Qwen3-0.6B')
    parser.add_argument('--card', type=int, choices=(2, 5), default=2)
    parser.add_argument('--max-seconds', type=int, default=600)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    rows = core_cases(source=args.source, native=args.native, sha256=args.sha256,
        build_fingerprint=args.build_fingerprint, model=args.model, card=args.card, max_seconds=args.max_seconds)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(rows, stream, indent=2)


if __name__ == '__main__':
    main()
