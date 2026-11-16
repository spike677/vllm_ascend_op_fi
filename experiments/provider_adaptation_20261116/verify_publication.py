"""Verify the reviewed text-file manifest; raw captures are not publication inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = 'experiments/provider_adaptation_20261116/checksums.json'
ALLOWED = {'.py', '.cpp', '.h', '.inc', '.sh', '.md', '.json'}


def read_file(path, staged):
    if staged:
        return subprocess.check_output(['git', '-C', str(ROOT), 'show', ':' + path])
    return (ROOT / path).read_bytes()


def validate_path(path):
    p = Path(path)
    if p.is_absolute() or '..' in p.parts or 'private' in p.parts:
        raise ValueError('Unreviewable publication path: ' + path)
    if p.suffix not in ALLOWED and p.name != '.gitignore':
        raise ValueError('Non-source publication file: ' + path)
    if p.name in ('dump.json', 'stack.json', 'construct.json'):
        raise ValueError('Raw statistics cannot be published in this task')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', action='store_true')
    parser.add_argument('--write-from-index', action='store_true')
    args = parser.parse_args()
    if args.write_from_index:
        paths = subprocess.check_output(['git', '-C', str(ROOT), 'diff', '--cached',
                                         '--name-only', '--diff-filter=ACM'], text=True).splitlines()
        files = {}
        for path in sorted(set(paths) - {MANIFEST}):
            validate_path(path)
            raw = read_file(path, True).replace(b'\r\n', b'\n')
            if len(raw) > 1_000_000:
                raise ValueError('Unexpectedly large publication file: ' + path)
            files[path] = dict(canonical_lf_bytes=len(raw), canonical_lf_sha256=hashlib.sha256(raw).hexdigest())
        data = dict(schema='demoncat.provider.publication.v1',
                    baseline_commit='052db5c1db1ab9ea6b85eb9ce5020d8a95e3ac4d',
                    raw_statistics_published=False,
                    hashing='UTF-8/source bytes with CRLF replaced by LF; manifest excludes itself',
                    files=files)
        (ROOT / MANIFEST).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(dict(status='MANIFEST_CREATED', files=len(files))))
        return
    data = json.loads(read_file(MANIFEST, args.staged))
    for path, expected in data['files'].items():
        validate_path(path)
        raw = read_file(path, args.staged).replace(b'\r\n', b'\n')
        actual = dict(canonical_lf_bytes=len(raw), canonical_lf_sha256=hashlib.sha256(raw).hexdigest())
        if actual != expected:
            raise ValueError('Publication checksum mismatch: ' + path)
    if args.staged:
        changed = set(subprocess.check_output(['git', '-C', str(ROOT), 'diff', '--cached',
                         '--name-only'], text=True).splitlines())
        if changed != set(data['files']) | {MANIFEST}:
            raise ValueError('Staged paths differ from reviewed publication manifest')
    print(json.dumps(dict(status='PASS', files=len(data['files']), raw_statistics_published=False)))


if __name__ == '__main__':
    main()
