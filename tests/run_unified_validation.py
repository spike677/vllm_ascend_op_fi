"""Write a reproducible CPU verification receipt, never an NPU certificate."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True)
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    runs = []
    for label, directory in [('unified', 'tests'), ('native_strategy_regression', 'native_execution_strategy/tests')]:
        command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', directory, '-v']
        result = subprocess.run(command, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        text = result.stdout.decode('utf-8', errors='replace')
        (args.output / (label + '.log')).write_bytes(result.stdout)
        counts = re.findall(r'Ran (\d+) tests?', text)
        runs.append(dict(suite=label, command=command[1:], returncode=result.returncode,
                         tests=int(counts[-1]) if counts else 0,
                         skipped=bool(re.search(r'\bskipped\b', text))))
    sources = [p for d in ('fault_injection', 'tests') for p in (root/d).glob('*.py')]
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'numpy')}
    passed = all(r['returncode'] == 0 and r['tests'] > 0 and not r['skipped'] for r in runs)
    receipt = dict(schema='demoncat.model_interface.cpu_validation.v1', status='PASS_CPU_ONLY' if passed else 'NOT_ACCEPTED',
        versions=versions, runs=runs, actual_npu_calls=0, pretrained_weights_downloaded=False,
        models=['Qwen3', 'Glm', 'DeepseekV3', 'Llama'],
        scope='Small random CPU models, real CPU Conv3d/RMSNorm, and explicit CPU alias/quant/native fixtures: structural routing, per-forward coverage, padded prefill, cached decode, owner-scoped lifecycle, cleanup failure quarantine/retry, document-host registration rollback, isolation, restoration and rejection. Not NPU injection, inner-framework wiring or model-quality validation.',
        remaining_mechanism_count=15, inner_framework_integration='PENDING',
        fault39='Existing independent native package and acceptance unchanged; legacy bridge fixtures are not that acceptance',
        mechanism_tests={'15.alias_detach':'CPU explicit format simulation; no actual private NPU format',
            '17.spatial_padding':'real CPU Conv3d argument change; new reconstructed proxy',
            '18.api_half':'real CPU RMSNorm module-return gain; new boundary',
            'quant.batch_shared_scale':'CPU provider and scheduler contract only',
            '20/26/33/39':'CPU native-boundary fixtures; deployment providers still required',
            '28.fia':'CPU registration/mask/vendor-contract checks, not NPU launch'},
        sources={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
