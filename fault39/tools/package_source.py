"""Create an immutable, source-only transfer; no CANN libraries or tensor data."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = root / 'fault39'
    files = []
    for path in sorted(source.rglob('*')):
        if not path.is_file() or any(x in {'.local', '__pycache__', 'build', 'dist', '.git', 'evidence', 'reports'} or x.endswith('.egg-info') for x in path.relative_to(source).parts):
            continue
        if path.suffix not in {'.py', '.cpp', '.h', '.inc', '.sh', '.md', '.json', '.yaml', '.toml'} and path.name != '.gitignore':
            continue
        data = path.read_bytes()
        if path.suffix == '.sh':
            data = data.replace(b'\r\n', b'\n')
        files.append((path.relative_to(root).as_posix(), data))
    manifest = {'schema': 'fault39.source_archive.v1', 'files': [
        {'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        for name, data in files]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('xb') as raw, tarfile.open(fileobj=raw, mode='w:gz') as archive:
        for name, data in files + [('source_manifest.json', json.dumps(manifest, indent=2).encode())]:
            item = tarfile.TarInfo(name); item.size = len(data); item.mode = 0o644
            archive.addfile(item, io.BytesIO(data))
    print(json.dumps({'archive': str(a.output), 'sha256': hashlib.sha256(a.output.read_bytes()).hexdigest(), 'files': len(files)}))


if __name__ == '__main__':
    main()
