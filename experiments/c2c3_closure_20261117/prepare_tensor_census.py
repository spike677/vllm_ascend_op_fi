"""Create two observation-only natural inference cases; never launch them."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for kind in ('worker', 'vae'):
        source = args.source
        model = '/home/gyl/models/Qwen3-0.6B' if kind == 'worker' else '/home/hunyuan-vae-cann91-ab-20261022/models/vae'
        mounts = [[source, '/source', 'ro']]
        if kind == 'worker':
            image = 'quay.io/ascend/vllm-ascend:v0.18.0'
            mounts.append([model, '/model', 'ro'])
            command = ('source /usr/local/Ascend/cann-8.5.1/set_env.sh\n'
                       'exec python3 /source/experiments/c2c3_closure_20261117/natural_tensor_census.py '
                       '--kind worker --model /model --out /evidence/probe')
            path = '/source:/source/src:/source/fault39/src'
        else:
            image = 'sha256:8988c3cd3a971a72ebb2e5baae24e0091cde55146f96d2b4c8229448210735b9'
            mounts += [['/home/hunyuan-vae-cann91-ab-20261022', '/audit', 'ro'],
                       ['/home/tik-constfold-cann91-overnight-20261022', '/work', 'ro']]
            command = ('source /usr/local/Ascend/cann-9.1.0/set_env.sh\n'
                       'source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1\n'
                       'exec /work/venv/bin/python /source/experiments/c2c3_closure_20261117/natural_tensor_census.py '
                       '--kind vae --model /audit/models/vae --vae-source /audit/sources/hv_vae '
                       '--latent /audit/inputs/latent.safetensors --out /evidence/probe')
            path = '/audit/deps:/audit/sources:/source:/source/src'
        name = 'natural-census-' + kind
        rows.append(dict(id=name, image=image, timeout=1500, mounts=mounts, command=command,
            env=dict(PYTHONPATH=path, OMP_NUM_THREADS='4', VLLM_ENABLE_V1_MULTIPROCESSING='0',
                     VLLM_WORKER_MULTIPROC_METHOD='spawn'),
            review=dict(experiment_id=name, provider='read-only natural tensor census', model=model,
                cann='8.5.1' if kind == 'worker' else '9.1.0', soc='Ascend910B4/A2',
                physical_cards=[2], ranks=1,
                worker='standard vLLM TP1 eager' if kind == 'worker' else 'pretrained VAE component, not a Worker',
                build_fingerprint='source-manifest verified; observer only, no Native injection',
                expected_hits=0, resource_budget=dict(timeout_seconds=1500, memory_gib=48, cpus=8),
                recovery_plan='No tensor writes or injection. Remove owned hooks/mode and require exact B repeat; root removes only owned temporary container.')))
    with args.out.open('x') as stream:
        json.dump(rows, stream, indent=2)


if __name__ == '__main__':
    main()
