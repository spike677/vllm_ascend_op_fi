"""Offline compact HF evidence verification; no native operators or tensors."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = read(ROOT/'manifest.json')
    names = {x['path'] for x in manifest['files']}
    actual = {str(p.relative_to(ROOT)).replace('\\','/') for p in ROOT.rglob('*') if p.is_file()}
    assert len(names) == len(manifest['files']) and actual == names | {'manifest.json'}
    binary_count = 0
    for row in manifest['files']:
        path = (ROOT/row['path']).resolve()
        assert ROOT in path.parents and path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
        assert 'private_replay' not in path.parts and path.suffix not in ('.pt','.pth','.npy','.npz','.safetensors')
        if path.suffix == '.bin':
            assert path.parent.name.endswith('.fa_log_profile')
            assert path.name.endswith(('.declared_prefix.bin','.aligned_log.bin'))
            binary_count += 1
    assert binary_count == 224
    for row in read(ROOT/'source_mapping.json')['files']:
        p = ROOT/row['path']
        assert row['original_bytes'] and sha(p) == row['sha256'] and p.stat().st_size == row['bytes']
    line_count = 0
    for case in ('stock_b1','stock_b1_repeat','stock_b8','stock_b8_repeat'):
        profile = read(ROOT/'raw'/(case+'.fa_log_profile')/'profile.json')
        excerpt = read(ROOT/'log_excerpts'/(case+'.json'))
        assert excerpt['source_log_sha256'] == profile['source_sha256']['log']
        expected = {r['line']: r for c in profile['calls'] for kind in ('launches','chunks','api_workspaces') for r in c[kind]}
        assert len(expected) == len(excerpt['native_lines']) == 112
        for r in excerpt['native_lines']:
            raw = base64.b64decode(r['raw_line_base64'],validate=True)
            assert hashlib.sha256(raw).hexdigest() == r['raw_line_sha256']
            text = raw.decode('utf-8','surrogateescape').rstrip('\r\n')
            assert text == r['text'] == expected.pop(r['line'])['text']
            line_count += 1
        assert not expected
    spec = importlib.util.spec_from_file_location('hf_summary',ROOT/'summarize.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.build() == read(ROOT/'summary.json')
    print(json.dumps({'status':'PASS_HF_CALIBRATION_EVIDENCE_INTEGRITY_NOT_MECHANISM_ACCEPTANCE',
                      'files_including_manifest':len(actual),'bytes_including_manifest':sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
                      'tiling_metadata_binary_files':binary_count,'exact_log_lines':line_count,
                      'manifest_sha256':sha(ROOT/'manifest.json'),'raw_framing_verified':False,'mechanism_accepted':False}))


if __name__ == '__main__':
    main()
