"""Create a fresh task-local statistics collector from a RECORD-verified installation.

No torch import, device use, installation mutation, or output-tensor modification.
"""
import argparse
import base64
import csv
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ORIGINAL_SHA = 'e981aa6fe9afa1e7a1f0654db44166d7c199674b0727d8d7ffa40626b1074011'
PROCESSOR = 'msprobe/core/dump/data_dump/data_processor/pytorch_processor.py'
DIST = 'mindstudio_probe-26.1.0.dist-info'
WHEEL_SHA = 'c004abe1123629ecdfb9938eee25cc8b554694d545c5123ae801c1846e4cfa68'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def patch(source):
    first = '        if torch.is_complex(data_clone):\n'
    replacement = (
        '        # Task-local repair: high statistics use a detached FP32 temporary.\n'
        '        # Uniform for every FP16/BF16 node; original tensor and metadata stay unchanged.\n'
        '        if precision == Const.DUMP_PRECISION_HIGH and data_clone.dtype in (torch.float16, torch.bfloat16):\n'
        '            data_clone = data_clone.to(dtype=torch.float32)\n\n' + first)
    second = ('            if (\n'
              '                precision != Const.DUMP_PRECISION_HIGH\n'
              '                and data_clone.dtype != torch.float64\n'
              '                and data_clone.is_floating_point()\n'
              '            ):\n')
    require(source.count(first) == 1 and source.count(second) == 1, 'Exact reviewed patch anchors differ')
    return source.replace(first, replacement).replace(second,
        '            if data_clone.dtype != torch.float64 and data_clone.is_floating_point():\n')


def build(source, stage):
    source = source.resolve(strict=True)
    require(not stage.exists(), 'Stage must be new; never overwrite earlier evidence')
    require(sha(source / PROCESSOR) == ORIGINAL_SHA, 'Installed collector source changed')
    direct = json.loads((source / DIST / 'direct_url.json').read_bytes())
    require(direct.get('archive_info', {}).get('hashes', {}).get('sha256') == WHEEL_SHA,
            'Installed wheel provenance differs')
    record_path = source / DIST / 'RECORD'
    inventory = {}
    for name, digest, size in csv.reader(record_path.read_text(encoding='utf-8').splitlines()):
        if not (name.startswith('msprobe/') or name.startswith(DIST + '/')) or '__pycache__/' in name:
            continue
        require('..' not in Path(name).parts and not Path(name).is_absolute(), 'Unsafe RECORD path')
        path = source / name
        require(path.is_file() and not path.is_symlink(), 'Missing/symlinked installed package source: ' + name)
        actual = sha(path)
        if digest:
            algorithm, encoded = digest.split('=', 1)
            require(algorithm == 'sha256' and base64.urlsafe_b64encode(bytes.fromhex(actual)).decode().rstrip('=') == encoded,
                    'Installed file differs from wheel RECORD: ' + name)
            require(path.stat().st_size == int(size), 'RECORD size mismatch')
        else:
            require(name in (DIST + '/RECORD',), 'Unexpected unhashed RECORD entry')
        inventory[name] = {'path': name, 'bytes': path.stat().st_size, 'sha256': actual,
                           'record_verified': bool(digest)}
    actual_names = {p.relative_to(source).as_posix() for top in (source / 'msprobe', source / DIST)
                    for p in top.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    require(actual_names == set(inventory), 'Unrecorded package files present')
    stage.mkdir(parents=True)
    for row in inventory.values():
        dest = stage / 'packages' / row['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / row['path'], dest)
    original = (source / PROCESSOR).read_text(encoding='utf-8')
    modified = patch(original)
    (stage / 'packages' / PROCESSOR).write_text(modified, encoding='utf-8', newline='')
    shutil.copyfile(source / PROCESSOR, stage / 'original_pytorch_processor.py')
    (stage / 'collector.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), modified.splitlines(True), fromfile='original/' + PROCESSOR,
        tofile='task_local_high/' + PROCESSOR)), encoding='utf-8', newline='')
    source_manifest = {'schema': 'fault28.msprobe.installed_source.v1', 'source_root': str(source),
                       'wheel_sha256_from_direct_url': WHEEL_SHA, 'wheel_archive_available': False,
                       'record_sha256': sha(record_path), 'files': sorted(inventory.values(), key=lambda r: r['path'])}
    (stage / 'source_manifest.json').write_text(json.dumps(source_manifest, indent=2) + '\n', encoding='utf-8')
    artifacts = [{'path': p.relative_to(stage).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                 for p in sorted(stage.rglob('*')) if p.is_file()]
    manifest = {'schema': 'fault28.msprobe.collector_high.v1', 'variant': 'task_local_fp32_high_v1',
                'package_version': '26.1.0', 'stage_root': '/audit/collector_high_a1',
                'precision': 'high', 'original_processor_sha256': ORIGINAL_SHA,
                'patched_processor_sha256': sha(stage / 'packages' / PROCESSOR),
                'wheel_sha256_from_direct_url': WHEEL_SHA, 'wheel_archive_available': False,
                'provenance': 'Installed complete package verified against its wheel RECORD; one task-local source repair. Not an unmodified official collector.',
                'policy': 'high: detached FP16/BF16 temporary cast to FP32; floating non-FP64 Mean/Norm calculated. All nodes and arms use identical policy. Original tensor dtype/shape metadata unchanged.',
                'artifacts': artifacts}
    (stage / 'stage_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    for p in stage.rglob('*'):
        p.chmod(0o555 if p.is_dir() else 0o444)
    stage.chmod(0o555)
    require(sha(source / PROCESSOR) == ORIGINAL_SHA, 'Original installed source changed')
    print(json.dumps({'status': 'STAGED_NOT_EXECUTED', 'stage': str(stage), 'manifest_sha256': sha(stage / 'stage_manifest.json'),
                      'original_processor_sha256': ORIGINAL_SHA, 'patched_processor_sha256': manifest['patched_processor_sha256'],
                      'artifact_count': len(artifacts)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--stage', type=Path, required=True)
    a = p.parse_args()
    build(a.source, a.stage)
