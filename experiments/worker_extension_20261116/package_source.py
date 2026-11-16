"""Package reviewable source, with clean Linux newlines and no evidence data."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = 'experiments/worker_extension_20261116'
SDK = (
    'fault20_fast_validation/source/fault26_readonly_trace_v1/fault26_trace_shared.h',
    'fault20_fast_validation/source/l4_buffer_substitute_allhit_v1/buffer_substitute_preload.cpp',
    'fault20_fast_validation/source/l4_buffer_substitute_allhit_v1/extreme_transform.h',
    'fault20_fast_validation/source/l4_buffer_substitute_allhit_v1/fault20_buffer_api.h',
    'fault20_fast_validation/source/l4_buffer_substitute_allhit_v1/int_equivalent_transform.h',
    'fault20_fast_validation/source/l4_buffer_substitute_allhit_v1/mantissa_truncate.h',
    'fault20_fast_validation/source/l4_metadata_lifecycle_proxy/libopapi_metadata_proxy.cpp',
    'fault_injection/native_adapters/build_qmm.sh',
    'fault_injection/native_adapters/fault20_abi_probe.cpp',
    'fault_injection/native_adapters/matmul_auto.h',
    'fault_injection/native_adapters/qmm_auto.inc',
    'fault39/native/runtime_identity.h',
    'fault39/native/test_identity.py',
    'fault39/tools/deployment_profile.py',
    'fault39/tools/version_profile.py',
)


def add(archive, name, data):
    member = tarfile.TarInfo(name)
    member.size = len(data)
    member.mode = 0o644
    archive.addfile(member, io.BytesIO(data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('--sdk-only', action='store_true')
    args = parser.parse_args()
    paths = set(SDK)
    if not args.sdk_only:
        paths.update(('src/pyproject.toml', 'fault39/tools/run_124.py',
                      f'{EXPERIMENT}/probe.py', f'{EXPERIMENT}/run_124.py'))
        for source in ('src/demoncat', 'fault_injection', 'fault39/src/demoncat_fault39'):
            paths.update(file.relative_to(ROOT).as_posix() for file in (ROOT / source).rglob('*.py'))
    files = {path: (ROOT / path).read_bytes().replace(b'\r\n', b'\n') for path in sorted(paths)}
    manifest = json.dumps({path: hashlib.sha256(data).hexdigest()
                           for path, data in files.items()}, indent=2).encode() + b'\n'
    with args.archive.open('xb') as output:
        with gzip.GzipFile(fileobj=output, mode='wb', filename='', mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode='w') as archive:
                for path, data in files.items():
                    add(archive, path, data)
                add(archive, 'worker_extension_source_manifest.json', manifest)
    print(hashlib.sha256(manifest).hexdigest(), len(files))


if __name__ == '__main__':
    main()
