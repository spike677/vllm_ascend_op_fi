"""Root-queued pretrained 18/22 all-target/every-match replay; no layer names.

This is model-local natural inference, not a standard vLLM Worker. Exact weight
and model implementation hashes are bound into the live C2 -> C3 proof.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from experiments.c2c3_closure_20261117.natural_tensor_census import load_local, sha_file


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('hf', 'vae'), default='hf')
    parser.add_argument('--mechanism', choices=('18.api_half', '22.wrong_row'), required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--vae-source', type=Path)
    parser.add_argument('--latent', type=Path)
    parser.add_argument('--c3-forwards', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.c3_forwards <= 4:
        parser.error('C3 is a bounded 1..4 real-forward session')
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='STARTED', mechanism=args.mechanism, public_qualification_granted=False,
                  standard_vllm_worker=False, pretrained_model=True)
    provider = None
    def save():
        (args.out / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    try:
        import torch
        import torch_npu
        from demoncat.runtime.vllm_ascend.worker_extension.providers.tensor_natural import NaturalTensorGroupProvider
        from demoncat.runtime.vllm_ascend.worker_extension.providers.tensor_modules import _capture, _identity
        torch.npu.set_device(0)
        model, inputs, provenance = load_local(args)
        weight_files = sorted({*args.model.glob('*.safetensors'), *args.model.glob('pytorch_model*.bin'),
                               *args.model.glob('pytorch_model.pt')})
        if not weight_files:
            raise RuntimeError('ACTUAL_PRETRAINED_WEIGHT_FILES_REQUIRED')
        weights = {path.name: dict(sha256=sha_file(path), bytes=path.stat().st_size) for path in weight_files}
        sources = {}
        for module in model.modules():
            try:
                source = Path(inspect.getsourcefile(type(module))).resolve()
            except (TypeError, OSError):
                continue
            if source.is_file() and source.name not in ('natural_tensor_census.py',):
                sources[str(source)] = sha_file(source)
        if not sources:
            raise RuntimeError('ACTUAL_MODEL_IMPLEMENTATION_SOURCE_REQUIRED')
        runtime = dict(torch=torch.__version__, torch_npu=torch_npu.__version__,
            device=torch.npu.get_device_name(0), cann_home=os.environ.get('ASCEND_HOME_PATH'),
            dtype=str(next(model.parameters()).dtype))
        workload = dict(model_weights=weights, model_sources=sources, inputs=_identity(_capture(inputs)))
        report.update(provenance=provenance, weights=weights, model_sources=sources, runtime=runtime,
            workload_sha256=digest(workload), model_weights_sha256=digest(weights),
            model_source_sha256=digest(sources), probe_sha256=sha_file(__file__))
        save()
        provider = NaturalTensorGroupProvider(model, args.mechanism, replay=lambda actual, phase: actual(*inputs),
            synchronize=torch.npu.synchronize, workload_sha256=report['workload_sha256'],
            environment_sha256=digest(runtime), model_weights_sha256=report['model_weights_sha256'],
            model_source_sha256=report['model_source_sha256'])
        report['discovery'] = provider.qualify(); save()
        report['c2'] = provider.run(); save()
        if not report['c2']['contract_satisfied']:
            raise RuntimeError('C2_CONTRACT_NOT_COMPLETE')
        proof = provider.group.c2_proof()
        report['c3'] = provider.run('every_match', forwards=args.c3_forwards, c2_proof=proof); save()
        if not report['c3']['contract_satisfied']:
            raise RuntimeError('C3_CONTRACT_NOT_COMPLETE')
        report.update(status='NATURAL_PRETRAINED_MODEL_GROUP_CONTRACT_COMPLETE',
            statistics_status='NOT_COLLECTED_SEPARATE_E5_REQUIRED')
    except BaseException as exc:
        report.update(status='REFUSED_OR_FAILED', error=repr(exc), traceback=traceback.format_exc())
    finally:
        if provider is not None:
            report['all_phases'] = provider.rows
            report['provider_error'] = provider.error
            if provider.group is not None:
                report['latest_group'] = provider.group.report()
        save()
    print(json.dumps({key: report[key] for key in ('status', 'mechanism', 'error') if key in report}))
    return 0 if report['status'] == 'NATURAL_PRETRAINED_MODEL_GROUP_CONTRACT_COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
