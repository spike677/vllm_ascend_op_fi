"""Prepare one bounded new-source Worker output39 C1 case; never run it."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]
from experiments.c2c3_postwrite_20261117.prepare41_coverage_cases import cases


def core_cases(*, source, native, sha256, build_fingerprint, model='/home/gyl/models/Qwen3-0.6B', card=2):
    if card not in (2, 5):
        raise ValueError('ROOT_REVIEWED_CARD_REQUIRED')
    _, row = cases(source=source, native=native, sha256=sha256, build_fingerprint=build_fingerprint,
        model=model, cards=[card], statistics=False)
    row['id'] = 'natural-output39-worker-core-c1'
    row['command'] = ('source /usr/local/Ascend/cann-8.5.1/set_env.sh\nexec '
        'python3 -u /source/experiments/c2c3_postwrite_20261117/natural39_worker_core_probe.py '
        '--model /model --library /native/libdcat_postwrite.so --sha256 '+sha256+
        ' --build-fingerprint '+build_fingerprint+' --out /evidence/probe --memory .35 '
        '> /evidence/worker-core39-output.stdout.log 2> /evidence/worker-core39-output.stderr.log')
    row['review'].update(experiment_id=row['id'], provider='WorkerFaultClient+FaultConfig(39,output_proxy) C1 only',
        worker='standard vLLM TP1 serial eager one fresh prefill per enqueue',
        expected_hits='One actual projection output after two healthy observations and fresh lossless; opaque custom GEMM refuses',
        recovery_plan='Six real requests: observe/repeat/lossless/F/OFF/health, then public cleanup and ordinary health. '
            'Existing finite input, Native owner/storage/bytes, consumer and exact full OFF gates retained. '
            'Unknown lowering refuses; no model-local qualification inheritance. Failed owners remain for explicit OFF/health retry.',
        acceptance='NEW_SOURCE_STANDARD_WORKER_C1_REVIEW_REQUIRED_C2_C3_FALSE',
        statistics='UNADAPTED: root/logits diagnostic value reads must not pollute mix API statistics',
        raw_evidence=['/evidence/probe/result.json', '/evidence/probe/worker_core_rpc',
            '/evidence/probe/host_receipt.json', '/evidence/output39-runtime',
            '/evidence/worker-core39-output.stdout.log', '/evidence/worker-core39-output.stderr.log'])
    return [row]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'native', 'sha256', 'build-fingerprint'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--model', default='/home/gyl/models/Qwen3-0.6B')
    parser.add_argument('--card', type=int, choices=(2,5), default=2)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    rows = core_cases(source=args.source, native=args.native, sha256=args.sha256,
        build_fingerprint=args.build_fingerprint, model=args.model, card=args.card)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(rows, stream, indent=2)


if __name__ == '__main__':
    main()
