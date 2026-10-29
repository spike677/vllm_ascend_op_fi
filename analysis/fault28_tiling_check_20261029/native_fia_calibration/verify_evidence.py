"""Offline integrity/source-link checks for the compact a5 calibration package."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = read(ROOT / 'manifest.json')
    entries = manifest['files']
    names = {row['path'] for row in entries}
    assert len(names) == len(entries)
    actual = {str(p.relative_to(ROOT)).replace('\\', '/') for p in ROOT.rglob('*') if p.is_file()}
    assert actual == names | {'manifest.json'}
    for row in entries:
        path = (ROOT / row['path']).resolve()
        assert ROOT in path.parents and path.is_file() and path.stat().st_size == row['bytes']
        assert sha(path) == row['sha256']
        assert path.suffix.lower() not in ('.bin', '.pt', '.pth', '.npy', '.npz', '.safetensors')
    for row in read(ROOT / 'source_mapping.json')['files']:
        path = ROOT / row['path']
        assert row['original_bytes'] and path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    parser_sha = sha(ROOT / 'analysis_source/calibration_log_parser_v4.py')
    line_count = 0
    for case in ('stock_b1', 'stock_b1_repeat', 'stock_b8', 'stock_b8_repeat'):
        profile = read(ROOT / 'raw' / (case + '.native_profile.v4.json'))
        excerpts = read(ROOT / 'log_excerpts' / (case + '.json'))
        assert profile['parser_sha256'] == parser_sha
        assert excerpts['source_log_sha256'] == profile['inputs']['log_sha256']
        assert excerpts['source_log_total_lines'] == profile['total_lines']
        expected = {row['line']: row for call in profile['calls']
                    for kind in ('selected_tiling_records', 'selected_launch_records') for row in call[kind]}
        assert len(expected) == len(excerpts['selected_and_launch_lines']) == 56
        for row in excerpts['selected_and_launch_lines']:
            raw = base64.b64decode(row['raw_line_base64'], validate=True)
            assert hashlib.sha256(raw).hexdigest() == row['raw_line_sha256']
            text = raw.decode('utf-8', 'surrogateescape').rstrip('\r\n')
            assert row['text'] == text
            assert re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text) == expected.pop(row['line'])['text']
            line_count += 1
        assert not expected
    spec = importlib.util.spec_from_file_location('fia_summary', ROOT / 'summarize.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.build() == read(ROOT / 'summary.json')
    print(json.dumps({'status': 'PASS_CALIBRATION_EVIDENCE_INTEGRITY_NOT_MECHANISM_ACCEPTANCE',
                      'files_including_manifest': len(actual), 'bytes_including_manifest': sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
                      'exact_log_lines': line_count, 'manifest_sha256': sha(ROOT / 'manifest.json'),
                      'profile_complete': False, 'mechanism_accepted': False}))


if __name__ == '__main__':
    main()
