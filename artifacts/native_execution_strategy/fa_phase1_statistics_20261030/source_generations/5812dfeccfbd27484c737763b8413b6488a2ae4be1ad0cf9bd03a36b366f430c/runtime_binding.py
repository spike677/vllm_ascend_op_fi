"""Verify the frozen, task-local Python runtime before importing its package."""
import hashlib
import json
from pathlib import Path
import sys


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_runtime(case):
    manifest_path = Path('/audit/runtime_manifest.json')
    root = Path('/audit/runtime')
    require(case['runtime_manifest_file'] == str(manifest_path)
            and sha(manifest_path) == case['runtime_manifest_sha256'], 'Runtime manifest differs from case')
    manifest = json.loads(manifest_path.read_bytes())
    require(manifest['schema'] == 'nes.runtime_snapshot.v1', 'Unexpected runtime schema')
    require(not any(name == 'native_execution_strategy' or name.startswith('native_execution_strategy.')
                    for name in sys.modules), 'Runtime package imported before source validation')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual == set(manifest['files']) and not any(p.is_symlink() for p in root.rglob('*')),
            'Runtime inventory differs or contains symlinks')
    for name, row in manifest['files'].items():
        path = root / name
        require(path.resolve(strict=True).is_relative_to(root.resolve(strict=True))
                and path.suffix in ('.py', '.json') and path.stat().st_size == row['bytes']
                and sha(path) == row['sha256'], 'Runtime source differs: ' + name)
    sys.path.insert(0, str(root))
    return manifest


def verify_imports(manifest):
    root = Path('/audit/runtime').resolve(strict=True)
    actual = {}
    for name, module in list(sys.modules.items()):
        if name != 'native_execution_strategy' and not name.startswith('native_execution_strategy.'):
            continue
        file = getattr(module, '__file__', None)
        if file is None:  # A namespace has no implementation; each actual module still must be pinned.
            continue
        path = Path(file).resolve(strict=True)
        require(path.is_relative_to(root), 'Runtime imported from an unapproved package root')
        relative = path.relative_to(root).as_posix()
        require(relative in manifest['files'] and sha(path) == manifest['files'][relative]['sha256'],
                'Imported module differs from frozen source: ' + name)
        actual[name] = {'path': str(path), 'sha256': sha(path)}
    require(actual, 'No actual runtime implementation was imported')
    return actual
