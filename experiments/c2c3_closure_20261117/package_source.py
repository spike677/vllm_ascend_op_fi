"""Freeze reviewed source bytes; no Native binary, model, or tensor payload."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


PREFIXES = ('engineering_closure/', 'fault_injection/', 'src/demoncat/', 'fault39/native/', 'fault39/tools/',
            'fault39/src/', 'native_execution_strategy/',
            'experiments/c2c3_closure_20261117/', 'experiments/c2c3_postwrite_20261117/',
            'experiments/engineering_20261117/',
            'experiments/worker_positive_20261117/',
            'experiments/provider_adaptation_20261116/', 'tests/',
            'fault20_fast_validation/source/l4_buffer_substitute_allhit_v1/',
            'fault20_fast_validation/source/l4_metadata_lifecycle_proxy/',
            'fault20_fast_validation/source/fault26_readonly_trace_v1/',
            'fault20_fast_validation/source/fault26_fia_writer_proxy_v1/',
            'fault20_fast_validation/source/fault26_qmm_writer_proxy_v1/',
            'fault20_fast_validation/source/fault26_add_rms_norm_bias_proxy_v1/')
SINGLES = {'pyproject.toml', 'src/pyproject.toml', 'fault39/pyproject.toml',
           'experiments/p3p4_closure_20261117/build_quant_v2.py'}
EXTENSIONS = {'.py', '.cpp', '.h', '.inc', '.sh', '.json', '.toml', '.cmake', '.txt'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    names = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others',
                                     '--exclude-standard'], cwd=root).decode().split('\0')
    snapshot = {}
    for name in sorted(set(names)):
        path = root / name
        if not name or (name not in SINGLES and not name.startswith(PREFIXES)):
            continue
        extensions = EXTENSIONS | {'.md', '.csv'} if name.startswith('engineering_closure/') else EXTENSIONS
        if not path.is_file() or path.suffix not in extensions or '__pycache__' in path.parts:
            continue
        if path.is_symlink() or path.stat().st_size > 1024 * 1024:
            raise ValueError('UNREVIEWED_SOURCE_ENTRY:' + name)
        if path.name in {'dump.json', 'stack.json', 'construct.json'}:
            raise ValueError('RAW_STATISTICS_NOT_SOURCE:' + name)
        raw = path.read_bytes()
        raw.decode('utf-8')
        snapshot[name] = raw
    files = [dict(path=name, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
             for name, raw in snapshot.items()]
    manifest = dict(schema='demoncat.c2c3.source.v1',
                    base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                        cwd=root, text=True).strip(),
                    files=files, contains_model_weights=False,
                    contains_native_binaries=False, contains_tensor_payloads=False)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as tar:
        for name, raw in snapshot.items():
            entry = tarfile.TarInfo(name)
            entry.size, entry.mode = len(raw), 0o644
            tar.addfile(entry, io.BytesIO(raw))
        raw = (json.dumps(manifest, indent=2) + '\n').encode()
        entry = tarfile.TarInfo('source_manifest.json')
        entry.size, entry.mode = len(raw), 0o644
        tar.addfile(entry, io.BytesIO(raw))
    print(json.dumps(dict(path=str(a.out), files=len(files), bytes=a.out.stat().st_size,
                          sha256=hashlib.sha256(a.out.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
