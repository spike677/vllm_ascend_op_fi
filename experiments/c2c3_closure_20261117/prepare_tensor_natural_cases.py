"""Prepare bounded natural Tensor jobs; never acquire devices or launch them."""
import argparse
import json
from pathlib import Path, PurePosixPath
import shlex


def cases(source, task, card=2, image='quay.io/ascend/vllm-ascend:v0.18.0'):
    if (not isinstance(source, str) or not source.startswith('/') or
            any(char in source for char in ('\n', '\r', '\x00', ',')) or
            '..' in PurePosixPath(source).parts or card not in (2, 5)):
        raise ValueError('ABSOLUTE_FROZEN_SOURCE_AND_REVIEWED_CARD_REQUIRED')
    if task in ('moe8-census', 'moe8-group'):
        phases = [('natural-moe8-census' if task == 'moe8-census' else 'natural-moe8-group', None)]
        model = '/home/haoran/models/DeepSeek-V2-Lite-W8A8'
    elif task == 'worker18-statistics':
        phases = [('natural-worker18-stats-b', 'baseline'), ('natural-worker18-stats-repeat', 'baseline'),
                  ('natural-worker18-stats-f', 'fault')]
        model = '/home/gyl/models/Qwen3-0.6B'
    elif task in ('vision17-census','vision17-group'):
        phases = [('natural-'+task, None)]
        model = '/home/17-fault-c2c3-20261117/models/Qwen3-VL-2B-Instruct'
    elif task == 'rope22-census':
        phases = [('natural-rope22-census',None)]
        model = '/home/gyl/models/Qwen3-0.6B'
    else:
        raise ValueError('EXPLICIT_TENSOR_TASK_REQUIRED')
    result = []
    for name, arm in phases:
        command = ('python3 /source/experiments/c2c3_closure_20261117/natural_tensor_census.py '
                   '--kind worker --model /model --out /evidence/probe --memory .55 --moe-trace') if arm is None else (
                   'python3 /source/experiments/c2c3_closure_20261117/natural_rms_worker_statistics.py '
                   '--model /model --out /evidence/probe --memory .35 --arm ' + shlex.quote(arm))
        if task == 'moe8-group':
            command = ('python3 /source/experiments/c2c3_closure_20261117/natural_moe_worker_probe.py '
                       '--model /model --out /evidence/probe --memory .55')
        result.append(dict(id=name, image=image, timeout=1800,
            mounts=[[source, '/source', 'ro'], [model, '/model', 'ro']],
            command='source /usr/local/Ascend/cann-8.5.1/set_env.sh\nexec '+command,
            env=dict(PYTHONPATH='/source:/source/src:/source/fault39/src:/msprobe_packages',
                OMP_NUM_THREADS='4', VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_WORKER_MULTIPROC_METHOD='spawn'),
            review=dict(experiment_id=name, provider='NaturalMoETrace' if arm is None else 'ServingTensorGroup RMS output-half proxy',
                model=model, cann='8.5.1', soc='Ascend910B4/A2', physical_cards=[card], ranks=1,
                worker='Fresh actual standard vLLM TP1 eager; original pretrained model and scheduler',
                build_fingerprint='Frozen source-manifest identity required; no native patch or route modification',
                expected_hits=0 if arm != 'fault' else 'All runtime-qualified RMS returns; prior observed113 is expectation only',
                resource_budget=dict(timeout_seconds=1800, memory_gib=64 if arm is None else 48, cpus=8),
                recovery_plan=('No injection; exact independent B repeat and post-cleanup healthy generation; remove owned observer/modes.'
                    if arm is None else 'Same request ordinal in fresh B/repeat/F Workers; full hidden/logits OFF and same-Worker healthy generation; public collector stop, owned hooks/modes drained; raw bytes retained.'))))
        if task == 'moe8-group':
            result[-1]['review'].update(provider='Source-bound actual MoE paired I/P input remap',
                expected_hits='Runtime-qualified all MoE targets once, then two every-match forwards; prior26 is expectation only',
                recovery_plan='Four actual healthy qualifications; C2+OFF then C3 two requests+OFF; exact full hidden/logits and post-cleanup healthy generation; owned source-bound profile/modes removed; no layer/router changes.')
        if task in ('vision17-census','vision17-group'):
            result[-1]['command'] = ('source /usr/local/Ascend/cann-8.5.1/set_env.sh\nexec '
                'python3 /source/experiments/c2c3_closure_20261117/natural_tensor_census.py '
                '--kind hf-vision --model /model --image /natural-input.JPG --out /evidence/probe')
            result[-1]['mounts'].append(['/home/jw/Wan2.2/examples/i2v_input.JPG','/natural-input.JPG','ro'])
            result[-1]['review'].update(provider='NaturalTensorObserver full pretrained image prefill census',
                worker='MODEL_LOCAL full pretrained Qwen3VL; not standard Worker evidence',
                expected_hits=0, recovery_plan='No injection; full exact B/B-repeat; original image processor and native Conv3d geometry; remove owned observer.')
            if task == 'vision17-group':
                result[-1]['command'] = ('source /usr/local/Ascend/cann-8.5.1/set_env.sh\nexec '
                    'python3 /source/experiments/c2c3_closure_20261117/natural_tensor_group_probe.py '
                    '--kind hf-vision --mechanism 17.spatial_padding --model /model '
                    '--image /natural-input.JPG --out /evidence/probe --statistics')
                result[-1]['review'].update(provider='Actual stock patch Conv3d padding argument group',
                    expected_hits='All actual qualified patch Conv3d calls once, then two every-match image forwards',
                    recovery_plan='Four exact healthy image forwards; C2+OFF then C3+OFF; full raw B/B-repeat/F+OFF statistics; immutable original weights/image/geometry, complete logits and owned cleanup.')
        if task == 'rope22-census':
            result[-1]['command'] = ('source /usr/local/Ascend/cann-8.5.1/set_env.sh\nexec '
                'python3 /source/experiments/c2c3_closure_20261117/natural_tensor_census.py '
                '--kind worker --model /model --out /evidence/probe --memory .35 --rope-trace')
            result[-1]['review'].update(provider='Source-bound real fused RoPE positions/cache/native/consumer census',
                expected_hits=0,recovery_plan='No injection or model-method replacement; original scheduler positions and cache unchanged, full exact B/B-repeat and post-cleanup healthy request; owned nested modes/profile/hooks removed.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--task', choices=('moe8-census', 'moe8-group', 'worker18-statistics',
                                         'vision17-census','vision17-group','rope22-census'), required=True)
    parser.add_argument('--card', type=int, choices=(2, 5), default=2)
    parser.add_argument('--image', default='quay.io/ascend/vllm-ascend:v0.18.0')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with args.out.open('x') as stream:
        json.dump(cases(args.source, args.task, args.card, args.image), stream, indent=2)


if __name__ == '__main__':
    main()
