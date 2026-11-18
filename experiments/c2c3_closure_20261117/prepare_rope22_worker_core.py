"""Prepare one bounded public Worker Core22 case; never launch it."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from experiments.c2c3_closure_20261117.prepare_rope22_group import group_cases


def core_cases(source, card=2, image='quay.io/ascend/vllm-ascend:v0.18.0'):
    row, = group_cases(source, card, image)
    row['id'] = 'natural-rope22-worker-core'
    row['command'] = ('source /usr/local/Ascend/cann-8.5.1/set_env.sh\nexec '
        'python3 -u /source/experiments/c2c3_closure_20261117/natural_rope_worker_core_probe.py '
        '--model /model --out /evidence/probe --memory .35 '
        '> /evidence/worker-core22.stdout.log 2> /evidence/worker-core22.stderr.log')
    row['review'].update(experiment_id=row['id'],
        provider='WorkerFaultClient+FaultConfig(22,wrong_row) reviewed fused serving deployment',
        expected_hits='Actual independent all-site C2 once, original live-proof C3 twice; no fixed layer count',
        recovery_plan='10 actual enqueue requests: 4 healthy, C2+OFF, original live C2 proof C3 twice+OFF, '
            'cleanup then same-Worker healthy. Public RPC actions/query and complete reader/call/phase JSON '
            'remain in /evidence on server124. Failed attempts retain raw receipts; at most one explicit '
            'recovery OFF, no fault retry or Worker restart. E5 is separate.',
        acceptance='NEW_FROZEN_SOURCE_AND_ACTUAL_WORKER_REVIEW_REQUIRED_NOT_INHERITED_FROM_SOURCE46',
        raw_evidence=['/evidence/probe/result.json', '/evidence/probe/host_receipt.json',
                      '/evidence/probe/worker_core_rpc', '/evidence/worker-core22.stdout.log',
                      '/evidence/worker-core22.stderr.log'])
    return [row]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--card', type=int, choices=(2, 5), default=2)
    parser.add_argument('--image', default='quay.io/ascend/vllm-ascend:v0.18.0')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    with args.out.open('x') as stream:
        json.dump(core_cases(args.source, args.card, args.image), stream, indent=2)


if __name__ == '__main__':
    main()
