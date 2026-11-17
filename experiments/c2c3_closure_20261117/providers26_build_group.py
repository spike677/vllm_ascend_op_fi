"""Root-invoked device-free SDK build; copies only verified trace/proxy dependencies.

This command starts a CPU-only build container. Agents must hand it to the root
queue owner; it is never a native acceptance run or a model inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess


SOURCES = (
    'fault_injection/native_adapters/writer26_group_core.h',
    'fault_injection/native_adapters/writer26_group_runtime.cpp',
    'fault_injection/native_adapters/test_writer26_group_core.cpp',
    'fault_injection/native_adapters/test_writer26_group_runtime_control.cpp',
    'fault20_fast_validation/source/fault26_readonly_trace_v1/fault26_qmm_underflow_api.h',
    'fault20_fast_validation/source/fault26_readonly_trace_v1/fault26_inventory.h',
)
OMIT = {'libfault26_qmm_output_underflow_abi0.so', 'libfault26_fia_writer_proxy_abi0.so'}
REQUIRED = {'libfault26_readonly_trace_abi0.so', 'libfault26_fia_opapi_proxy.so',
            'libopapi.so', 'libopapi_nn.so', 'libcust_opapi.so', 'real/libopapi_real.so',
            'real/libopapi_nn_real.so', 'real/libcust_opapi_real.so'}
IMAGE = 'quay.io/ascend/vllm-ascend:v0.18.0'
TOOLKIT = '/usr/local/Ascend/cann-8.5.1/aarch64-linux'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'parent', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--parent-manifest-sha256', required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    if platform.machine() not in ('aarch64', 'arm64') or not re.fullmatch('[0-9a-f]{40}', args.source_commit):
        raise RuntimeError('ACTUAL_AARCH64_HOST_AND_COMMITTED_SOURCE_REQUIRED')
    source, parent, out = args.source.resolve(strict=True), args.parent.resolve(strict=True), args.out.resolve()
    manifest = parent / 'build.json'
    if sha(manifest) != args.parent_manifest_sha256:
        raise RuntimeError('PARENT_MANIFEST_CHANGED')
    before = json.loads(manifest.read_bytes())
    if before.get('schema') != 'demoncat.writer26.deployment.v1' or before.get('actual_npu_calls') != 0:
        raise RuntimeError('VERIFIED_FRESH_TRACE_BUILD_REQUIRED')
    files = before['files']
    if not REQUIRED.issubset(files) or not OMIT.issubset(files):
        raise RuntimeError('COMPLETE_PARENT_TRACE_PROXY_SET_REQUIRED')
    for name, expected in files.items():
        path = (parent / name).resolve(strict=True)
        if not path.is_relative_to(parent) or sha(path) != expected:
            raise RuntimeError('PARENT_FILE_CHANGED:' + name)
    if out.exists():
        raise RuntimeError('NEW_EMPTY_BUILD_PATH_REQUIRED')
    out.mkdir(parents=True)
    for name in sorted(REQUIRED):
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent / name, out / name)
    # Legacy DSOs are deliberately absent from the new bundle; source is immutable.
    assert not any((out / name).exists() for name in OMIT)
    sources = {name: sha(source / name) for name in SOURCES}
    root_args = ['docker', 'run', '--rm', '--network', 'none', '--runtime', 'runc', '--cpus', '4', '--memory', '4g',
        '-e', 'ASCEND_VISIBLE_DEVICES=void', '-v', str(source) + ':/source:ro', '-v', str(out) + ':/build',
        '--entrypoint', 'bash', IMAGE, '-lc']
    commands = r'''set -euo pipefail
test "$(uname -m)" = aarch64
grep -Eq '^version=8\.5\.1$' /usr/local/Ascend/cann-8.5.1/aarch64-linux/ascend_toolkit_install.info
flags=(-std=c++17 -O2 -Wall -Wextra -Werror -D_GLIBCXX_USE_CXX11_ABI=0 -I/usr/local/Ascend/cann-8.5.1/aarch64-linux/include -I/usr/local/Ascend/cann-8.5.1/aarch64-linux/include/aclnn -I/usr/local/Ascend/cann-8.5.1/aarch64-linux/pkg_inc)
native=/source/fault_injection/native_adapters
g++ "${flags[@]}" -fPIC -shared "$native/writer26_group_runtime.cpp" -Wl,-soname,libdcat26_group_v2.so -ldl -lpthread -o /build/libdcat26_group_v2.so
g++ "${flags[@]}" "$native/test_writer26_group_runtime_control.cpp" -ldl -lpthread -o /build/test_group_control
/build/test_group_control > /build/sdk-control.json
g++ "${flags[@]}" "$native/test_writer26_group_core.cpp" -o /build/test_group_core
/build/test_group_core > /build/cpu-contract.json
nm -D /build/libdcat26_group_v2.so > /build/exported-symbols.txt
readelf -d /build/libdcat26_group_v2.so > /build/elf-dynamic.txt
sha256sum /usr/local/Ascend/cann-8.5.1/aarch64-linux/include/acl/acl.h /usr/local/Ascend/cann-8.5.1/aarch64-linux/include/aclnn/opdev/common_types.h > /build/sdk-header-sha256.txt
'''
    command = root_args + [commands]
    with (out / 'build.log').open('wb') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
    receipt = dict(schema='demoncat.writer26.group_build.v2', status='FAILED_CPU_BUILD', command=command,
        actual_npu_calls=0, device_mounts=[], source_commit=args.source_commit, sources=sources,
        parent_manifest_sha256=args.parent_manifest_sha256, parent_files={name: files[name] for name in sorted(REQUIRED)},
        omitted_legacy_writers=sorted(OMIT), image=IMAGE, cann='8.5.1', build_returncode=result.returncode,
        build_log_sha256=sha(out / 'build.log'))
    (out / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if result.returncode:
        raise RuntimeError('SDK_BUILD_FAILED_LOG_PRESERVED')
    for name in REQUIRED:
        if sha(out / name) != files[name]:
            raise RuntimeError('TRACE_PROXY_DEPENDENCY_CHANGED')
    exports = (out / 'exported-symbols.txt').read_text()
    for name in ('state', 'install', 'pass', 'continue', 'receipt', 'acknowledge', 'seal', 'stop', 'drain', 'restore', 'remove'):
        if not re.search(r'\bT dcat26_group_' + name + r'_v2$', exports, re.M):
            raise RuntimeError('NEW_GROUP_ABI_EXPORT_MISSING:' + name)
    if re.search(r'\bdcat26[qf]_(install|continue)_v1$', exports, re.M):
        raise RuntimeError('LEGACY_WRITER_ABI_MUST_NOT_BE_PRESENT')
    receipt.update(status='SDK_BUILD_AND_CPU_CONTRACT_PASS_NOT_NATIVE_ACCEPTANCE',
        sdk_control=json.loads((out / 'sdk-control.json').read_text()),
        cpu_contract=json.loads((out / 'cpu-contract.json').read_text()))
    (out / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    output_files = {name: sha(out / name) for name in sorted(REQUIRED | {'libdcat26_group_v2.so'})}
    (out / 'build.json').write_text(json.dumps(dict(schema='demoncat.writer26.group_deployment.v2',
        files=output_files, build_receipt_sha256=sha(out / 'build-receipt.json'), sources=sources,
        actual_npu_calls=0, native_qualified_targets=0, c2=False, c3=False,
        legacy_writers_absent=True, independent_trace_required=True), indent=2) + '\n')
    print(json.dumps(dict(status=receipt['status'], bundle=str(out), manifest_sha256=sha(out / 'build.json'))))


if __name__ == '__main__':
    main()
