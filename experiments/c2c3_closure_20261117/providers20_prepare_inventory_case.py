"""Prepare a no-injection natural inventory case for the root resource queue."""
import argparse
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser()
    for name in ('installed', 'probe', 'packages-sha', 'source-commit'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--model', default='/home/haoran/models/DeepSeek-V2-Lite-W8A8')
    parser.add_argument('--card', type=int, choices=(2, 5), default=2)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if (not re.fullmatch('[0-9a-f]{64}', args.packages_sha)
            or not re.fullmatch('[0-9a-f]{7,40}', args.source_commit)
            or any(not value.startswith('/') or any(c in value for c in ('\n', '\r', '\0', ','))
                   for value in (args.installed, args.probe, args.model))):
        parser.error('Actual package hash, source commit and immutable absolute inputs required')
    name = 'providers20-deepseek-public-inventory'
    case = dict(id=name, timeout=1800, image='quay.io/ascend/vllm-ascend:v0.18.0',
        mounts=[[args.installed, '/installed', 'ro'], [args.probe, '/probe.py', 'ro'], [args.model, '/model', 'ro']],
        env=dict(VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_WORKER_MULTIPROC_METHOD='spawn',
                 VLLM_ASCEND_ENABLE_NZ='2', VLLM_ALLOW_INSECURE_SERIALIZATION='1',
                 OMP_NUM_THREADS='4', VLLM_HOST_IP='127.0.0.1', GLOO_SOCKET_IFNAME='lo',
                 HCCL_CONNECT_TIMEOUT='120', HCCL_EXEC_TIMEOUT='120'),
        command='source /usr/local/Ascend/cann-8.5.1/set_env.sh\n'
                'export PYTHONPATH=/installed:/msprobe_packages:$PYTHONPATH\n'
                'exec python3 /probe.py',
        review=dict(experiment_id=name, provider='20/shared-quant independent public lineage inventory only',
            model=args.model, cann='8.5.1', soc='Ascend910B4/A2; recheck Worker', physical_cards=[args.card], ranks=[0],
            worker='new standard TP1 eager BF16 Worker; two identical healthy fresh-prefill requests',
            build_fingerprint=args.packages_sha, source_commit=args.source_commit, expected_hits=0,
            resource_budget=dict(timeout_seconds=1800, memory_gib=80, cpus=16, public_api_calls=4096),
            recovery_plan='No fault Native library or injection is loaded. Observe all actual dynamic projection chains and '
                'unowned public quant/QMM calls. Preserve every original exactly once. Remove only owned method/API/model '
                'hooks and verify two complete logits references and same-process generated control. Root removes its '
                'owned temporary container on any failure; report no native C1/C2/C3/E5 acceptance.'))
    with args.out.open('x') as stream:
        json.dump([case], stream, indent=2)


if __name__ == '__main__':
    main()
