"""Prepare two bounded public Worker Core cases; this tool never launches them."""
import argparse
import json
from pathlib import Path, PurePosixPath


def core_cases(source, card=2, image='quay.io/ascend/vllm-ascend:v0.18.0'):
    if (not isinstance(source, str) or not source.startswith('/') or '..' in PurePosixPath(source).parts
            or any(char in source for char in ('\n', '\r', '\0', ',')) or card not in (2, 5)):
        raise ValueError('ABSOLUTE_FROZEN_SOURCE_AND_REVIEWED_CARD_REQUIRED')
    rows = []
    for mechanism, model, memory in (('18.api_half', '/home/gyl/models/Qwen3-0.6B', '.35'),
                                     ('8.restore_offset', '/home/haoran/models/DeepSeek-V2-Lite-W8A8', '.55')):
        fault, _ = mechanism.split('.', 1)
        name = 'natural-tensor' + fault + '-worker-core'
        rows.append(dict(id=name, image=image, timeout=1800,
            mounts=[[source, '/source', 'ro'], [model, '/model', 'ro']],
            command='source /usr/local/Ascend/cann-8.5.1/set_env.sh\nexec python3 -u '
                '/source/experiments/c2c3_closure_20261117/natural_tensor_worker_core_probe.py '
                '--mechanism ' + mechanism + ' --model /model --out /evidence/probe --memory ' + memory
                + ' > /evidence/worker-core.stdout.log 2> /evidence/worker-core.stderr.log',
            env=dict(PYTHONPATH='/source:/source/src:/source/fault39/src', OMP_NUM_THREADS='4',
                     VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_WORKER_MULTIPROC_METHOD='spawn'),
            review=dict(experiment_id=name, provider='WorkerFaultClient+FaultConfig ' + mechanism,
                model=model, cann='8.5.1', soc='Ascend910B4/A2', physical_cards=[card], ranks=1,
                worker='Existing standard TP1 eager Worker; actual enqueue; no model/runner replacement',
                build_fingerprint='Fresh frozen source manifest and matching packages; no new Native DSO',
                expected_hits='Actual complete natural targets once, then twice with original live C2 object; historical counts confer no scope',
                resource_budget=dict(timeout_seconds=1800, memory_gib=64, cpus=8),
                recovery_plan='10 actual requests: four healthy/lossless, C2+OFF, C3 twice+OFF, owned cleanup then same-Worker healthy. '
                    'Failure retains owner until at most one real recovery OFF and one healthy request. No fault retry, restart or JSON proof.',
                acceptance='NEW_PUBLIC_ENTRYPOINT_ACTUAL_REVIEW_REQUIRED_C1_C2_C3_NOT_INHERITED_E5_SEPARATE',
                raw_evidence=['/evidence/probe/result.json', '/evidence/probe/host_receipt.json',
                              '/evidence/probe/worker_core_rpc', '/evidence/worker-core.stdout.log',
                              '/evidence/worker-core.stderr.log'])))
    return rows


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
