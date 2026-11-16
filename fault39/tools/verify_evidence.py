"""Offline verification of explicitly listed evidence files under one root.

The JSON manifest is {"files": [{"path": "relative/posix/path", "bytes": N,
"sha256": "..."}]}. Unlisted files are not inspected. No network is used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re


def verify_manifest(manifest, root):
    report = {'status': 'FAIL', 'count': 0, 'checked_count': 0, 'verified_count': 0,
              'unlisted_files_checked': False, 'failures': []}
    if not isinstance(manifest, dict) or not isinstance(manifest.get('files'), list):
        report['failures'].append({'code': 'INVALID_MANIFEST_SCHEMA'})
        return report
    entries = manifest['files']
    report['count'] = len(entries)
    if not entries:
        report['failures'].append({'code': 'EMPTY_MANIFEST'})
        return report
    try:
        root = Path(root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError('Root is not a directory')
    except (OSError, RuntimeError, ValueError):
        report['failures'].append({'code': 'INVALID_ROOT'})
        return report
    seen_paths, seen_resolved = set(), set()
    for index, entry in enumerate(entries):
        failure = {'index': index}
        if not isinstance(entry, dict):
            report['failures'].append(dict(failure, code='INVALID_FILE_ENTRY'))
            continue
        relative = entry.get('path')
        failure['path'] = relative if isinstance(relative, str) else None
        # Reject platform-dependent forms before joining or resolving. Drive-
        # relative paths such as C:foo are unsafe even on a Linux verifier.
        if (not isinstance(relative, str) or not relative or '\\' in relative or '\0' in relative
                or relative.startswith('/') or PureWindowsPath(relative).drive
                or any(part in ('', '.', '..') or ':' in part for part in relative.split('/'))):
            report['failures'].append(dict(failure, code='UNSAFE_RELATIVE_PATH'))
            continue
        key = os.path.normcase(relative)
        if key in seen_paths:
            report['failures'].append(dict(failure, code='DUPLICATE_PATH'))
            continue
        seen_paths.add(key)
        expected_size, expected_hash = entry.get('bytes'), entry.get('sha256')
        if (type(expected_size) is not int or expected_size < 0
                or not isinstance(expected_hash, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', expected_hash)):
            report['failures'].append(dict(failure, code='INVALID_CHECKSUM_ENTRY'))
            continue
        try:
            resolved = (root / relative).resolve(strict=True)
            if not resolved.is_relative_to(root):
                report['failures'].append(dict(failure, code='RESOLVED_PATH_ESCAPE'))
                continue
            if resolved in seen_resolved:
                report['failures'].append(dict(failure, code='DUPLICATE_RESOLVED_PATH'))
                continue
            seen_resolved.add(resolved)
            if not resolved.is_file():
                report['failures'].append(dict(failure, code='NOT_A_REGULAR_FILE'))
                continue
            digest, size = hashlib.sha256(), 0
            with resolved.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    size += len(chunk)
                    digest.update(chunk)
            report['checked_count'] += 1
            actual_hash = digest.hexdigest()
            if size != expected_size or actual_hash != expected_hash.lower():
                report['failures'].append(dict(failure, code='CHECKSUM_MISMATCH',
                    expected_bytes=expected_size, actual_bytes=size,
                    expected_sha256=expected_hash.lower(), actual_sha256=actual_hash))
            else:
                report['verified_count'] += 1
        except (OSError, RuntimeError, ValueError) as error:
            report['failures'].append(dict(failure, code='FILE_UNAVAILABLE', error_type=type(error).__name__))
    if not report['failures'] and report['verified_count'] == len(entries):
        report['status'] = 'PASS'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
        report = verify_manifest(manifest, args.root)
    except (OSError, ValueError) as error:
        report = {'status': 'FAIL', 'count': 0, 'checked_count': 0, 'verified_count': 0,
                  'unlisted_files_checked': False,
                  'failures': [{'code': 'MANIFEST_UNREADABLE', 'error_type': type(error).__name__}]}
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(text)
    print(text, end='')
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
