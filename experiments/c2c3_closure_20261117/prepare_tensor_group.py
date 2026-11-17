"""Prepare one bounded natural pretrained-model group case, not a Worker claim."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--model', default='/home/gyl/models/Qwen3-0.6B')
    p.add_argument('--card', type=int, choices=(2, 5), default=2)
    p.add_argument('--mechanism', choices=('18.api_half', '22.wrong_row'), required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    name = 'natural-group-' + a.mechanism.split('.')[0]
    case = dict(id=name, image='quay.io/ascend/vllm-ascend:v0.18.0', timeout=1500,
        mounts=[[a.source, '/source', 'ro'], [a.model, '/model', 'ro']],
        env=dict(PYTHONPATH='/source:/source/src:/source/fault39/src:/msprobe_packages', OMP_NUM_THREADS='4'),
        command='source /usr/local/Ascend/cann-8.5.1/set_env.sh\n'
            'exec python3 /source/experiments/c2c3_closure_20261117/natural_tensor_group_probe.py '
            '--mechanism '+a.mechanism+' --model /model --out /evidence/probe --c3-forwards 2 --statistics',
        review=dict(experiment_id=name, provider=a.mechanism, model=a.model, cann='8.5.1',
            soc='Ascend910B4/A2', physical_cards=[a.card], ranks=1,
            worker='pretrained HF model-local eager inference, not standard vLLM Worker',
            build_fingerprint='reviewed Python source_manifest; actual weights/model source bound at runtime',
            expected_hits='C2: each independently eligible target once; C3: each match in two bounded forwards only after live C2; separately recorded B/B/F statistics replay',
            resource_budget=dict(timeout_seconds=1500, memory_gib=48, cpus=8),
            recovery_plan='Stop/drain/release all targets; same-process exact full-model OFF and zero owned hooks/leases. Raw statistics never confer runtime qualification.'))
    with a.out.open('x') as stream:
        json.dump([case], stream, indent=2)


if __name__ == '__main__':
    main()
