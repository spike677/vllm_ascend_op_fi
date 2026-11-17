"""Prepare, but never launch, two fresh B/F cases per natural postwrite path."""
import argparse
import json
from pathlib import Path
import re
import shlex


def cases(*, source, native, library_sha256, build_fingerprint, model, cards=(2,), mechanisms=('39.output_proxy', '41.tail_column')):
    for value in (library_sha256, build_fingerprint):
        if re.fullmatch('[0-9a-f]{64}', value) is None:
            raise ValueError('ACTUAL_LIBRARY_AND_BUILD_FINGERPRINT_REQUIRED')
    result = []
    for mechanism in mechanisms:
        for arm in ('baseline', 'fault'):
            prefix = 'postwrite39hf' if mechanism == '39.output_proxy' else 'postwrite41worker'
            name = f'{prefix}-{arm}'
            probe = ('engineering_20261117/postwrite_probe.py --mechanism 39.output_proxy'
                     if mechanism == '39.output_proxy' else 'c2c3_postwrite_20261117/natural41_probe.py')
            command = ('source /usr/local/Ascend/cann-8.5.1/set_env.sh\n'
                       'exec python3 /source/experiments/' + probe + ' --arm ' + arm +
                       ' --library /native/libdcat_postwrite.so --sha256 ' + library_sha256 +
                       ' --model /model --out /evidence --statistics')
            if mechanism == '41.tail_column':command += ' --build-fingerprint ' + build_fingerprint
            result.append(dict(id=name, image='quay.io/ascend/vllm-ascend:v0.18.0', timeout=1200,
                mounts=[[source, '/source', 'ro'], [native, '/native', 'ro'], [model, '/model', 'ro']],
                env={'PYTHONPATH':'/source/src:/source/fault39/src:/source:/msprobe_packages',
                     'OMP_NUM_THREADS':'4', 'VLLM_ENABLE_V1_MULTIPROCESSING':'0',
                     'VLLM_WORKER_MULTIPROC_METHOD':'spawn'}, command=command,
                review=dict(experiment_id=name, provider=mechanism, model=model, cann='8.5.1', soc='Ascend910B4/A2',
                    physical_cards=list(cards), ranks=1, worker='fresh process, natural HF prefill' if mechanism.startswith('39.')
                        else 'standard vLLM Worker TP1/PP1/DP1 serial eager first pure decode',
                    build_fingerprint=build_fingerprint, expected_hits=1 if arm == 'fault' else 0,
                    resource_budget=dict(timeout_seconds=1200, memory_gib=32, cpus=8),
                    recovery_plan='native device sync, checked exact storage restore, release, same-process OFF, remove owned hooks')))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--native', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--build-fingerprint', required=True)
    parser.add_argument('--model', default='/home/gyl/models/Qwen3-0.6B')
    parser.add_argument('--cards', nargs='+', type=int, default=[2])
    parser.add_argument('--mechanism', action='append', choices=['39.output_proxy', '41.tail_column'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    prepared = cases(source=args.source, native=args.native, library_sha256=args.sha256,
        build_fingerprint=args.build_fingerprint, model=args.model, cards=args.cards,
        mechanisms=args.mechanism or ('39.output_proxy', '41.tail_column'))
    with args.out.open('x') as stream:json.dump(prepared, stream, indent=2)


if __name__ == '__main__':main()
