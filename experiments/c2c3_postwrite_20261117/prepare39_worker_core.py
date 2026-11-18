"""Prepare one bounded new-source Worker output39 C1 case; never run it."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]
from experiments.c2c3_postwrite_20261117.prepare41_coverage_cases import cases


def core_cases(*, source, native, sha256, build_fingerprint, model='/home/gyl/models/Qwen3-0.6B', card=2,
               stage='c1', max_live_leases=128):
    if card not in (2, 5):
        raise ValueError('ROOT_REVIEWED_CARD_REQUIRED')
    if stage not in ('c1', 'c2_once', 'c2_c3') or type(max_live_leases) is not int or not 1 <= max_live_leases <= 256:
        raise ValueError('EXPLICIT_COMPLETE_GROUP_STAGE_AND_LEASE_BUDGET_REQUIRED')
    _, row = cases(source=source, native=native, sha256=sha256, build_fingerprint=build_fingerprint,
        model=model, cards=[card], statistics=False)
    row['id'] = 'natural-output39-worker-core-'+stage
    row['command'] = ('source /usr/local/Ascend/cann-8.5.1/set_env.sh\nexec '
        'python3 -u /source/experiments/c2c3_postwrite_20261117/natural39_worker_core_probe.py '
        '--model /model --library /native/libdcat_postwrite.so --sha256 '+sha256+
        ' --build-fingerprint '+build_fingerprint+' --out /evidence/probe --memory .35 '
        '--stage '+stage+' --max-live-leases '+str(max_live_leases)+' '
        '> /evidence/worker-core39-output.stdout.log 2> /evidence/worker-core39-output.stderr.log')
    row['review'].update(experiment_id=row['id'], provider='WorkerFaultClient+FaultConfig(39,output_proxy) C1 only',
        worker='standard vLLM TP1 serial eager one fresh prefill per enqueue',
        expected_hits='One actual projection output after two healthy observations and fresh lossless; '
            'reviewed source-bound vllm.unquantized_gemm may qualify only with an actual verified value consumer',
        recovery_plan='Six real requests: observe/repeat/lossless/F/OFF/health, then public cleanup and ordinary health. '
            'Existing finite input, Native owner/storage/bytes, consumer and exact full OFF gates retained. '
            'Only the inspected custom GEMM schema/source/owned weight and narrow ND RMSNorm/SwiGLU consumers are added; '
            'unknown lowering or mutating attention refuses. No model-local qualification inheritance. '
            'Failed owners remain for explicit OFF/health retry.',
        acceptance='NEW_SOURCE_STANDARD_WORKER_C1_REVIEW_REQUIRED_C2_C3_FALSE',
        statistics='UNADAPTED: root/logits diagnostic value reads must not pollute mix API statistics',
        raw_evidence=['/evidence/probe/result.json', '/evidence/probe/worker_core_rpc',
            '/evidence/probe/host_receipt.json', '/evidence/output39-runtime',
            '/evidence/worker-core39-output.stdout.log', '/evidence/worker-core39-output.stderr.log'])
    if stage != 'c1':
        row['review'].update(provider='WorkerFaultClient+FaultConfig(39,output_proxy) complete-set '+stage,
            expected_hits='C2: every stable eligible target once in separate real requests; '
                'C3 only if requested: every eligible call in two real requests after original live C2 proof',
            recovery_plan='Existing six-request C1 first. All-target lossless, N isolated target F requests, OFF, health '
                'and owned-hook removal precede original C2 issuance. Explicit c2_c3 adds new lossless, two every-match '
                'F requests, OFF and health; the original live C2 object must match owner/scope/targets. '
                'Every actual ordinal and module/alias remains; no target prefix or fallback. Current registered custom '
                'GEMM, weight object/no bias, actual value consumer, finite original and exact Native span gates stay. '
                'Unresolved discovery or propagated nonfinite downstream operands can refuse C3; refusal keeps owner '
                'until explicit actual OFF/health and Native cleanup. No RPC calls model.forward.',
            acceptance='NEW_SOURCE_PUBLIC_GROUP_C2_C3_ATTEMPT_REQUIRES_INDEPENDENT_REVIEW',
            max_live_leases=max_live_leases, max_complete_targets=256,
            execution_scope=dict(c2='independent target replays', c3='same-request every-match multiple live leases'),
            legacy_model_local_evidence_inherited=False)
    return [row]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'native', 'sha256', 'build-fingerprint'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--model', default='/home/gyl/models/Qwen3-0.6B')
    parser.add_argument('--card', type=int, choices=(2,5), default=2)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('c1', 'c2_once', 'c2_c3'), default='c1')
    parser.add_argument('--max-live-leases', type=int, default=128)
    args = parser.parse_args(argv)
    rows = core_cases(source=args.source, native=args.native, sha256=args.sha256,
        build_fingerprint=args.build_fingerprint, model=args.model, card=args.card,
        stage=args.stage, max_live_leases=args.max_live_leases)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(rows, stream, indent=2)


if __name__ == '__main__':
    main()
