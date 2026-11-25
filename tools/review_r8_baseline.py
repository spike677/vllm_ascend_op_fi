"""Compare committed runtime bytes with r7.4.1; differences never grant reuse."""
import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = '59f6aea5'
PREFIXES = ('fault_injection/', 'src/demoncat/', 'fault39/src/', 'native_execution_strategy/')


def snapshot(ref):
    commit = subprocess.check_output(['git', 'rev-parse', '--verify', ref + '^{commit}'], cwd=ROOT, text=True).strip()
    raw = subprocess.check_output(['git', 'archive', commit, '--', *PREFIXES], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        files = {m.name: archive.extractfile(m).read() for m in archive if m.isfile()}
    return commit, files


def compare(before, after):
    def identity(raw):
        return None if raw is None else dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    differences = [dict(path=p, before=identity(before[p]), after=identity(after.get(p)))
                   for p in sorted(before) if p not in after or before[p] != after[p]]
    added = [dict(path=p, **identity(after[p])) for p in sorted(after.keys() - before.keys())]
    return dict(original_files_checked=len(before), unchanged_count=len(before)-len(differences),
                differences=differences, additions=added,
                original_runtime_bytes_unchanged=bool(before) and not differences,
                qualification_granted=False, a3_results_automatically_reused=False,
                note='Conservative full-runtime diff: admission changes require explicit review; mechanism changes require owner approval. Additions require routing/incremental checks even when original bytes are equal.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    base, before = snapshot(BASELINE)
    candidate, after = snapshot(args.candidate)
    result = dict(schema='dcat.r8.baseline.review.v1', baseline_commit=base,
                  candidate_commit=candidate, prefixes=PREFIXES, **compare(before, after))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['original_runtime_bytes_unchanged'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
