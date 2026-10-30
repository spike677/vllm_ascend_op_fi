"""Export exactly the Python runtime/assets, never tests, native libraries or tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(package, runtime, manifest, native_components):
    package = package.resolve(strict=True)
    if runtime.exists() or manifest.exists():
        raise ValueError('Runtime and manifest generation must be fresh')
    runtime.mkdir(parents=True)
    files = {}
    for path in sorted(package.rglob('*')):
        relative = path.relative_to(package)
        if not path.is_file() or any(x in relative.parts for x in ('tests', 'native', 'revisions', '__pycache__')):
            continue
        selected = path.suffix == '.py' or path.suffix == '.json' and (relative.parts[0] == 'data' or relative.as_posix() == 'adapters/fa_cann91_schema.json')
        if not selected:
            continue
        if path.is_symlink():
            raise ValueError('Runtime source symlinks are unsupported')
        name = 'native_execution_strategy/' + relative.as_posix()
        target = runtime / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha(target) != sha(path):
            raise ValueError('Runtime copy changed bytes')
        files[name] = {'sha256': sha(target), 'bytes': target.stat().st_size}
    value = {'schema': 'nes.runtime_snapshot.v1', 'files': files,
             'native_components_sha256': native_components,
             'scope': 'Exact Python runtime and data assets; native vendor inventory is independently verified by launcher'}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    return {'status': 'EXPORTED_NOT_EXECUTED', 'files': len(files), 'manifest_sha256': sha(manifest)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--native-components', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.package, args.runtime, args.manifest, json.loads(args.native_components.read_bytes()))))


if __name__ == '__main__':
    main()
