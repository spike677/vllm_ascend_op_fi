"""Audit the entire new Git history, then hash the reviewed publication tree.

No Git mutation or network calls. Raw captures, tensors and native binaries stay
on 124. This checks committed bytes, including files removed in later commits;
it does not turn source review or CPU contracts into mechanism qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

BASE = '91decf5efe48dfdf8c5494e52d258ccde788cc5d'
REPORT = 'experiments/c2c3_closure_20261117/publication_manifest.json'
FORBIDDEN_SUFFIXES = {'.pt', '.pth', '.npy', '.npz', '.safetensors', '.bin',
                      '.so', '.a', '.o', '.whl', '.zip', '.gz', '.tar', '.dll',
                      '.exe', '.pkl', '.pickle', '.pyc', '.key', '.pem'}
FORBIDDEN_NAMES = {'dump.json', 'stack.json', 'construct.json', '.env'}


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def check_path(name):
    path = Path(name)
    if path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError('PRIVATE_OR_BINARY_PUBLICATION_ENTRY:' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    subprocess.run(['git', 'merge-base', '--is-ancestor', BASE, 'HEAD'], cwd=root, check=True)
    status = git(root, 'status', '--porcelain', '-z').decode('utf-8').split('\0')
    dirty = [row for row in status if row and row[3:] != REPORT]
    if dirty:
        raise ValueError('COMMIT_REVIEWED_SOURCE_BEFORE_PUBLICATION_AUDIT:' + repr(dirty))
    head = git(root, 'rev-parse', 'HEAD').decode().strip()
    historical_paths = set(git(root, 'log', '--format=', '--name-only', BASE + '..' + head).decode().splitlines()) - {''}
    for name in historical_paths:
        check_path(name)
    objects = git(root, 'rev-list', '--objects', BASE + '..' + head).decode().splitlines()
    sizes = subprocess.run(['git', 'cat-file', '--batch-check=%(objectname) %(objecttype) %(objectsize)'],
        cwd=root, input='\n'.join(line.split(' ', 1)[0] for line in objects).encode(),
        stdout=subprocess.PIPE, check=True).stdout.decode().splitlines()
    history_blobs = []
    for item, metadata in zip(objects, sizes):
        blob, kind, size = metadata.split()
        if kind != 'blob':
            continue
        name = item.partition(' ')[2]
        check_path(name)
        raw = git(root, 'cat-file', 'blob', blob)
        raw.decode('utf-8')
        if b'\0' in raw or len(raw) != int(size) or len(raw) > 1024 * 1024:
            raise ValueError('UNREVIEWED_LARGE_OR_BINARY_HISTORY_BLOB:' + name)
        history_blobs.append(dict(blob=blob, first_path=name, bytes=len(raw),
                                  sha256=hashlib.sha256(raw).hexdigest()))
    current = []
    names = git(root, 'diff', '--name-only', '--diff-filter=ACMRT', '-z', BASE, head).decode().split('\0')
    for name in sorted(filter(None, names)):
        if name == REPORT:
            continue
        check_path(name)
        blob = git(root, 'rev-parse', head + ':' + name).decode().strip()
        raw = git(root, 'cat-file', 'blob', blob)
        actual = root / name
        # Check canonical Git content; CRLF checkout conversions are allowed only
        # when Git itself maps the working tree to this exact committed blob.
        checkout_blob = git(root, 'hash-object', '--path=' + name, str(actual)).decode().strip()
        if checkout_blob != blob:
            raise ValueError('WORKTREE_BYTES_NOT_COMMITTED_CONTENT:' + name)
        current.append(dict(path=name, blob=blob, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    result = dict(schema='demoncat.c2c3.publication.v1', base_commit=BASE, reviewed_head=head,
        branch=git(root, 'branch', '--show-current').decode().strip(),
        new_commits=int(git(root, 'rev-list', '--count', BASE + '..' + head)),
        historical_paths_checked=len(historical_paths), historical_blobs=history_blobs,
        files=current, excludes_self=REPORT, contains_new_raw_statistics=False,
        contains_private_tensor_payloads=False, contains_native_binaries=False,
        content_scope='Reviewed implementation, CPU contracts, probes, summaries and hashes only',
        mechanism_qualification_granted=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic JSON bytes on Windows as well as Linux; this manifest is
    # published as a hashable source-audit artifact, not a native text stream.
    args.out.write_bytes((json.dumps(result, indent=2, ensure_ascii=False) + '\n').encode('utf-8'))
    print(json.dumps(dict(head=head, files=len(current), history_blobs=len(history_blobs),
                          manifest_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
