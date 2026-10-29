"""Validate and select the explicitly approved task-local msprobe statistics package."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

STAGE = Path('/audit/collector_high_a1')
PROCESSOR = 'msprobe/core/dump/data_dump/data_processor/pytorch_processor.py'
VARIANT = 'task_local_fp32_high_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def select_collector(case, stage=STAGE):
    variant = case.get('statistics_variant', 'official_low')
    require(variant in ('official_low', VARIANT), 'Unknown statistics collector variant')
    if variant == 'official_low':
        require(not case.get('approved_collector_manifest_file') and not case.get('approved_collector_manifest_sha256'),
                'Official collector cannot carry patched-stage approval')
        return {'variant': variant, 'precision': 'low', 'modified_official_collector': False}
    require(case.get('capture_msprobe') is True, 'High collector requires statistics capture')
    require(not any(n == 'msprobe' or n.startswith('msprobe.') for n in sys.modules), 'msprobe imported before stage selection')
    manifest_path = stage / 'stage_manifest.json'
    require(case.get('approved_collector_manifest_file') == str(manifest_path), 'Unexpected approved collector stage')
    require(stage.resolve(strict=True) == stage and not stage.is_symlink(), 'Collector stage must be a direct directory')
    require(sha(manifest_path) == case.get('approved_collector_manifest_sha256'), 'Collector manifest SHA differs')
    manifest = json.loads(manifest_path.read_bytes())
    require(manifest.get('schema') == 'fault28.msprobe.collector_high.v1' and manifest.get('variant') == VARIANT
            and manifest.get('precision') == 'high' and manifest.get('stage_root') == str(STAGE), 'Unexpected collector manifest policy')
    names = set()
    for row in manifest['artifacts']:
        name = row['path']
        rel = PurePosixPath(name)
        require(name and not rel.is_absolute() and '..' not in rel.parts and name not in names, 'Unsafe/duplicate stage path')
        names.add(name)
        path = stage / name
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(stage), 'Missing/escaping collector artifact')
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Collector artifact differs: ' + name)
    actual = {p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_file()}
    require(actual == names | {'stage_manifest.json'}, 'Unexpected/missing collector files')
    require(not any(p.is_symlink() for p in stage.rglob('*')), 'Symlinks forbidden in collector stage')
    require(sha(stage / 'packages' / PROCESSOR) == manifest['patched_processor_sha256'], 'Patched processor SHA differs')
    sys.path.insert(0, str(stage / 'packages'))
    return {'variant': VARIANT, 'precision': 'high', 'modified_official_collector': True,
            'manifest_file': str(manifest_path), 'manifest_sha256': sha(manifest_path),
            'package_root': str(stage / 'packages'), 'processor_file': str(stage / 'packages' / PROCESSOR),
            'original_processor_sha256': manifest['original_processor_sha256'],
            'patched_processor_sha256': manifest['patched_processor_sha256'],
            'source_manifest_sha256': sha(stage / 'source_manifest.json'),
            'patch_sha256': sha(stage / 'collector.patch'), 'artifact_count': len(names),
            'provenance': manifest['provenance'], 'policy': manifest['policy']}


def verify_loaded(selection, module):
    path = Path(module.__file__).resolve(strict=True)
    method = module.PytorchDataProcessor.get_stat_info
    actual = {'module': module.__name__, 'path': str(path), 'sha256': sha(path),
              'method_code_file': str(Path(method.__code__.co_filename).resolve(strict=True))}
    if selection['modified_official_collector']:
        require(actual['path'] == selection['processor_file'] and actual['method_code_file'] == actual['path']
                and actual['sha256'] == selection['patched_processor_sha256'], 'Actual collector import differs from approved stage')
    else:
        require(actual['sha256'] == 'e981aa6fe9afa1e7a1f0654db44166d7c199674b0727d8d7ffa40626b1074011',
                'Official default collector changed')
    return actual
