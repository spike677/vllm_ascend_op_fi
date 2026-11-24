"""Build the read-only interceptor against the local SDK, without NPU access."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source, sdk, output, source_commit, compiler='g++'):
    source, sdk, output = (Path(p).resolve() for p in (source, sdk, output))
    output.mkdir(parents=True, exist_ok=False)
    library = output / 'capture-v2-r2.so'
    depfile = output / 'headers.d'
    flags = ['-std=c++17', '-shared', '-fPIC', '-Wl,--build-id=sha1',
             '-MMD', '-MF', str(depfile), '-I', str(sdk / 'include'),
             str(source), '-ldl', '-pthread', '-o', str(library)]
    subprocess.run([compiler, *flags], check=True)
    dependencies = depfile.read_text().replace('\\\n', ' ').split(':', 1)[1].split()
    headers = []
    for name in sorted(set(dependencies)):
        path = Path(name).resolve()
        if path == source:
            continue
        # Record content identity, not an assumption that a version tag is enough.
        headers.append(dict(path=path.relative_to(sdk).as_posix(), bytes=path.stat().st_size,
                            sha256=sha(path)))
    versions = []
    for name in ('version.cfg', 'version.info', 'compiler/version.info',
                 'opp/version.info', 'share/info/runtime/version.info',
                 'share/info/asc-devkit/version.info'):
        path = sdk / name
        if path.is_file():
            versions.append(dict(path=name, sha256=sha(path), text=path.read_text()))
    receipt = dict(schema='dcat.capture.build.r2', source_commit=source_commit,
                   source_file=source.name, source_bytes=source.stat().st_size,
                   source_sha256=sha(source), library=library.name,
                   library_bytes=library.stat().st_size, library_sha256=sha(library),
                   compiler=subprocess.check_output([compiler, '--version'], text=True),
                   architecture=platform.machine(), flags=flags, headers=headers,
                   sdk_version_files=versions,
                   elf_notes=subprocess.check_output(['readelf', '-n', str(library)], text=True),
                   npu_initialized=False, a3_validated=False, qualification_granted=False)
    (output / 'build-receipt.json').write_bytes((json.dumps(receipt, indent=2) + '\n').encode())
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.sdk, args.output, args.source_commit), indent=2))
