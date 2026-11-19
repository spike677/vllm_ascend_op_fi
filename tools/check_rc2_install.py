"""Read installed origins and public contracts; no Worker or Native is created."""
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


DISTRIBUTIONS = {'demoncatpfi': '0.3.0', 'demoncat-worker-extension': '0.2.0',
                 'demoncat-fault39-core': '0.2.0'}
MODULES = ('fault_injection.rc2_profile','demoncat.runtime.vllm_ascend.worker_extension.deployment_authorization',
           'demoncat.runtime.vllm_ascend.worker_extension.deployment_summary','fault_injection', 'demoncat_fault39',
           'demoncat.runtime.vllm_ascend.controller',
           'demoncat.runtime.vllm_ascend.worker_extension.config',
           'demoncat.runtime.vllm_ascend.worker_extension.host',
           'demoncat.runtime.vllm_ascend.worker_extension.registry')


def check(root):
    root = Path(root).resolve(strict=True)
    result = dict(schema='demoncat.yellow_rc2.install_check.v1',
        status='IMPORT_AND_DEFAULT_CLOSED_CHECK_ONLY', python=sys.version,
        platform=platform.platform(), expected_root=str(root), modules={}, distributions={},
        issues=[], worker_created=False, native_loaded_by_check=False,
        inference_executed=False, yellow_integration_verified=False)
    for name, expected in DISTRIBUTIONS.items():
        try:
            dist = importlib.metadata.distribution(name)
            location = Path(dist.locate_file('')).resolve()
            result['distributions'][name] = dict(version=dist.version, location=str(location))
            if dist.version != expected or not location.is_relative_to(root):
                result['issues'].append('DISTRIBUTION_ORIGIN_OR_VERSION_MISMATCH:' + name)
        except importlib.metadata.PackageNotFoundError:
            result['issues'].append('DISTRIBUTION_NOT_INSTALLED:' + name)
    for name in MODULES:
        try:
            module = importlib.import_module(name)
            origin = Path(module.__file__).resolve()
            result['modules'][name] = str(origin)
            if not origin.is_relative_to(root):
                result['issues'].append('MODULE_ORIGIN_MISMATCH:' + name)
        except Exception as error:
            result['issues'].append('IMPORT_FAILED:' + name + ':' + repr(error))
    for name in ('demoncat', 'demoncat.runtime'):
        module = sys.modules.get(name)
        result[name] = dict(file=getattr(module, '__file__', None),
            search_paths=list(getattr(module, '__path__', [])))
        origin = getattr(module, '__file__', None)
        if origin and not Path(origin).resolve().is_relative_to(root):
            result['issues'].append('EXISTING_REGULAR_PACKAGE_REQUIRES_COMPOSITION_REVIEW:' + name)
    if not result['issues']:
        from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
        from demoncat.runtime.vllm_ascend.worker_extension.registry import (
            worker_core_manifest, create_worker_provider)
        manifest = worker_core_manifest()
        result['catalog'] = manifest
        for args in ((39, 'output_proxy'), ('quant', 'batch_shared_scale'), (18, 'api_half'), (8, 'restore_offset'), (17, 'spatial_padding'), (22, 'wrong_row')):
            FaultConfig(*args)
        if manifest['qualification_granted'] is not False:
            result['issues'].append('CATALOG_UNEXPECTED_QUALIFICATION')
        try:
            create_worker_provider('25.reduce_scatter', '1', None, None, None)
        except RuntimeError as error:
            if str(error) != 'FAULT25_UNSUPPORTED_NO_SUBSTITUTE':
                result['issues'].append('FAULT25_WRONG_REFUSAL:' + str(error))
        else:
            result['issues'].append('FAULT25_NOT_REFUSED')
        row = next(r for r in manifest['mechanisms'] if r['mechanism'] == '28.fia.denominator_replay')
        if row['route']['fault_execution_ready'] is not False:
            result['issues'].append('TND28_UNFINISHED_MARKED_READY')
    # Imports must stay independent of the model/device stack.
    loaded = sorted(n for n in ('torch', 'torch_npu', 'vllm', 'vllm_ascend') if n in sys.modules)
    result['model_stack_modules_present'] = loaded
    if loaded:
        result['issues'].append('RUN_CHECK_IN_FRESH_PROCESS_WITHOUT_MODEL_STACK')
    if result['issues']:
        result['status'] = 'REFUSED_INSTALL_OR_IMPORT_CONFLICT'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-root', required=True,
        help='fresh --target directory containing all three installed wheels')
    parser.add_argument('--output', help='new JSON receipt, never overwritten')
    args = parser.parse_args()
    result = check(args.expected_root)
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        with Path(args.output).open('x', encoding='utf-8') as stream:
            stream.write(text)
    print(text)
    return bool(result['issues'])


if __name__ == '__main__':
    raise SystemExit(main())
