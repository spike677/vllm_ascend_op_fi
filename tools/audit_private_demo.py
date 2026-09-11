"""Read-only provenance snapshot after a completed private demo; no model calls."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--branch', required=True)
    p.add_argument('--head', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    d = json.loads((a.session / 'deployment.json').read_text())
    source = Path(d['source_root'])
    files = {f.relative_to(source).as_posix(): sha(f) for name in ('src', 'tools', 'profiles')
             for f in (source / name).rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    final = json.loads((a.session / '03_evaluation/final_state.json').read_text())
    ref = Path(d['reference_path'])
    weights = Path(d['model_path']) / 'model.safetensors'
    install = Path('/usr/local/Ascend/cann-8.5.1/aarch64-linux/ascend_toolkit_install.info')
    ops = install.with_name('ascend_ops_install.info')
    value = {'branch': a.branch, 'base_HEAD': a.head, 'uncommitted_demo_changes': True,
        'source_files': files, 'reference_sha256': sha(ref), 'weight_sha256_after_run': sha(weights),
        'weight_integrity_before_after': 'NOT_MEASURED; model volume mounted read-only',
        'CANN_toolkit_install_record': install.read_text() if install.exists() else None,
        'CANN_ops_install_record': ops.read_text() if ops.exists() else None,
        'physical_device_mapping': d['logical_rank_mapping'], 'image_id': d['image_id'],
        'final_query_before_shutdown': final,
        'raw_files': {f.relative_to(a.session).as_posix(): sha(f) for folder in
            ('01_inspect', '02_bound_plan', '03_evaluation') for f in (a.session / folder).rglob('*.json')},
        'credentials_excluded': True, 'no_download_required': True}
    with a.out.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
    print(json.dumps({'reference_sha256': value['reference_sha256'], 'source_files': len(files),
                      'healthy_before_shutdown': final['healthy']}))


if __name__ == '__main__':
    main()
