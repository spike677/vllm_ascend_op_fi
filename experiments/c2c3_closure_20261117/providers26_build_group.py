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
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fault_injection.native_build.metadata26 import verify_parent, RECEIPT, CPU_RECEIPT, SYMBOLS


SOURCES = (
    'fault_injection/native_adapters/writer26_group_core.h',
    'fault_injection/native_adapters/writer26_group_runtime.cpp',
    'fault_injection/native_adapters/writer26_fia_typed_api.h',
    'fault_injection/native_adapters/test_writer26_group_core.cpp',
    'fault_injection/native_adapters/test_writer26_group_runtime_control.cpp',
    'fault_injection/native_adapters/test_writer26_pending_runtime.cpp',
    'fault_injection/native_adapters/test_writer26_fia_typed_runtime.cpp',
    'fault20_fast_validation/source/fault26_readonly_trace_v1/fault26_qmm_underflow_api.h',
    'fault20_fast_validation/source/fault26_readonly_trace_v1/fault26_inventory.h',
    'fault20_fast_validation/source/fault26_readonly_trace_v1/fault26_fia_typed.h',
    'fault39/native/metadata_array_owner.h',
    'fault39/native/metadata_ownership_fixture.cpp',
    'fault39/native/test_metadata_ownership.py',
    'fault_injection/native_build/metadata26.py',
)
OMIT = {'libfault26_qmm_output_underflow_abi0.so', 'libfault26_fia_writer_proxy_abi0.so'}
REQUIRED = {'libfault26_readonly_trace_abi0.so', 'libfault26_fia_opapi_proxy.so',
            'libopapi.so', 'libopapi_nn.so', 'libcust_opapi.so', 'real/libopapi_real.so',
            'real/libopapi_nn_real.so', 'real/libcust_opapi_real.so'}
IMAGE = 'quay.io/ascend/vllm-ascend:v0.18.0'
TOOLKIT = '/usr/local/Ascend/cann-8.5.1/aarch64-linux'
TYPED_TRACE_EXPORTS = ('dcat26_inventory_fia_typed_state_v2', 'dcat26_inventory_fia_typed_read_v2',
    'dcat26_inventory_fia_before_v2', 'dcat26_inventory_fia_after_v2', 'dcat26_inventory_fia_owned_execute_v2',
    'dcat26_inventory_fia_end_v2')


def require_typed_trace(symbols):
    if any(not re.search(r'\bT ' + name + r'$', symbols, re.M) for name in TYPED_TRACE_EXPORTS):
        raise RuntimeError('FRESH_TYPED_FIA_TRACE_PARENT_REQUIRED')


def typed_profile(control, contract):
    expected = dict(version=3, inventory_version=2, descriptor=280, record=656, state=104, lease=912, snapshots=2048)
    if (control.get('fia_typed_abi') != expected or contract.get('status') != 'PASS_CPU_TYPED_FIA_RUNTIME'
            or contract.get('actual_npu_calls') != 0 or contract.get('cases') != 18):
        raise RuntimeError('TYPED_FIA_ACTUAL_SDK_AND_RUNTIME_CONTRACT_REQUIRED')
    return dict(schema='demoncat.writer26.fia_ncl_typed.v3', abi=expected, native_format=47, public_format=2,
        dtype='bfloat16', dimensions=3, storage_layouts=['MATCHED_VIEW', 'EXACT_FLAT_SPAN'], offset=0, contiguous=True,
        borrowed_original_read_scope='ACTUAL_GWS_STACK_ONLY', effective_descriptor='GROUP_OWNED_ALL_SELECTED_MODES',
        stock_to_owned_reference_full_byte_comparison_required=True, old_reference_reusable=False,
        before_execute_complete_typed_readback_required=True, native_qualified=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_image():
    result = subprocess.run(['docker', 'image', 'inspect', IMAGE], capture_output=True, text=True, timeout=30, check=True)
    rows = json.loads(result.stdout)
    if len(rows) != 1 or not re.fullmatch(r'sha256:[0-9a-f]{64}', rows[0].get('Id', '')):
        raise RuntimeError('ACTUAL_LOCAL_IMAGE_ID_REQUIRED')
    if rows[0].get('Architecture') != 'arm64' or rows[0].get('Os') != 'linux':
        raise RuntimeError('ACTUAL_IMAGE_PLATFORM_CHANGED')
    return rows[0]


def cleanup_owned(name, nonce, image_id):
    """Inspect exact label and image before removing only this build container."""
    result = subprocess.run(['docker', 'container', 'inspect', name], capture_output=True, text=True, timeout=30)
    if result.returncode:
        missing = 'No such container' in result.stderr or 'No such object' in result.stderr
        return dict(status='ALREADY_ABSENT' if missing else 'INSPECTION_FAILED', confirmed=missing,
                    returncode=result.returncode, stderr=result.stderr)
    rows = json.loads(result.stdout)
    if (len(rows) != 1 or rows[0].get('Name') != '/' + name or rows[0].get('Image') != image_id
            or rows[0].get('Config', {}).get('Labels', {}).get('demoncat.writer26.build-owner') != nonce):
        return dict(status='FOREIGN_CONTAINER_REFUSED', confirmed=False)
    result = subprocess.run(['docker', 'container', 'rm', '-f', name], capture_output=True, text=True, timeout=30)
    return dict(status='OWNED_CONTAINER_REMOVED' if result.returncode == 0 else 'REMOVAL_FAILED',
                confirmed=result.returncode == 0, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


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
    ownership = verify_parent(source, parent, files)
    trace_exports = subprocess.run(['nm', '-D', '--defined-only', str(parent / 'libfault26_readonly_trace_abi0.so')],
        capture_output=True, text=True, check=True, timeout=30).stdout
    require_typed_trace(trace_exports)
    required = REQUIRED | {RECEIPT, CPU_RECEIPT}
    if out.exists():
        raise RuntimeError('NEW_EMPTY_BUILD_PATH_REQUIRED')
    out.mkdir(parents=True)
    for name in sorted(required):
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent / name, out / name)
    # Legacy DSOs are deliberately absent from the new bundle; source is immutable.
    assert not any((out / name).exists() for name in OMIT)
    sources = {name: sha(source / name) for name in SOURCES}
    sources.update(ownership['sources'])
    commands = r'''set -euo pipefail
test "$(uname -m)" = aarch64
grep -Eq '^version=8\.5\.1$' /usr/local/Ascend/cann-8.5.1/aarch64-linux/ascend_toolkit_install.info
flags=(-std=c++17 -O2 -Wall -Wextra -Werror -D_GLIBCXX_USE_CXX11_ABI=0 -I/usr/local/Ascend/cann-8.5.1/aarch64-linux/include -I/usr/local/Ascend/cann-8.5.1/aarch64-linux/include/aclnn -I/usr/local/Ascend/cann-8.5.1/aarch64-linux/pkg_inc)
native=/source/fault_injection/native_adapters
python3 -B /source/fault39/native/test_metadata_ownership.py --writer26 --output /build/metadata-array-group-cpu.json
g++ "${flags[@]}" -fPIC -shared "$native/writer26_group_runtime.cpp" -Wl,-soname,libdcat26_group_v2.so -ldl -lpthread -o /build/libdcat26_group_v2.so
g++ "${flags[@]}" "$native/test_writer26_group_runtime_control.cpp" -ldl -lpthread -o /build/test_group_control
/build/test_group_control > /build/sdk-control.json
g++ "${flags[@]}" "$native/test_writer26_pending_runtime.cpp" -ldl -lpthread -o /build/test_pending_runtime
/build/test_pending_runtime > /build/pending-runtime-cpu.json
g++ "${flags[@]}" "$native/test_writer26_fia_typed_runtime.cpp" -ldl -lpthread -o /build/test_fia_typed_runtime
/build/test_fia_typed_runtime > /build/fia-typed-runtime-cpu.json
g++ "${flags[@]}" "$native/test_writer26_group_core.cpp" -o /build/test_group_core
/build/test_group_core > /build/cpu-contract.json
nm -D /build/libdcat26_group_v2.so > /build/exported-symbols.txt
readelf -d /build/libdcat26_group_v2.so > /build/elf-dynamic.txt
sha256sum /usr/local/Ascend/cann-8.5.1/aarch64-linux/include/acl/acl.h /usr/local/Ascend/cann-8.5.1/aarch64-linux/include/acl/acl_base.h /usr/local/Ascend/cann-8.5.1/aarch64-linux/include/aclnn/opdev/common_types.h > /build/sdk-header-sha256.txt
'''
    nonce = uuid.uuid4().hex
    name = 'dcat26-group-build-' + nonce
    receipt = dict(schema='demoncat.writer26.group_build.v2', status='FAILED_CPU_BUILD', command=None,
        actual_npu_calls=0, device_mounts=[], source_commit=args.source_commit, sources=sources,
        parent_manifest_sha256=args.parent_manifest_sha256, parent_files={name: files[name] for name in sorted(required)},
        omitted_legacy_writers=sorted(OMIT), requested_image=IMAGE, actual_image_id=None,
        cann='8.5.1', container_name=name, container_owner_nonce=nonce, build_returncode=None)
    launched = False
    try:
        actual = inspect_image()
        (out / 'image-inspect.json').write_text(json.dumps(actual, indent=2) + '\n')
        receipt.update(actual_image_id=actual['Id'], image_inspect_sha256=sha(out / 'image-inspect.json'))
        command = ['docker', 'run', '--rm', '--name', name, '--label', 'demoncat.writer26.build-owner=' + nonce,
            '--network', 'none', '--runtime', 'runc', '--cpus', '4', '--memory', '4g',
            '-e', 'ASCEND_VISIBLE_DEVICES=void', '-v', str(source) + ':/source:ro', '-v', str(out) + ':/build',
            '--entrypoint', 'bash', actual['Id'], '-lc', commands]
        receipt['command'] = command
        with (out / 'build.log').open('wb') as stream:
            launched = True
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
        receipt['build_returncode'] = result.returncode
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error), timed_out=isinstance(error, subprocess.TimeoutExpired))
    finally:
        try:
            receipt['cleanup'] = (cleanup_owned(name, nonce, receipt['actual_image_id']) if launched
                                  else dict(status='NOT_LAUNCHED', confirmed=True))
        except BaseException as error:
            receipt['cleanup'] = dict(status='CLEANUP_EXCEPTION', confirmed=False, error=repr(error))
        if (out / 'build.log').exists():
            receipt['build_log_sha256'] = sha(out / 'build.log')
        (out / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if receipt['build_returncode'] != 0 or not receipt['cleanup']['confirmed']:
        raise RuntimeError('SDK_BUILD_OR_CLEANUP_FAILED_RECEIPT_PRESERVED')
    for name in required:
        if sha(out / name) != files[name]:
            raise RuntimeError('TRACE_PROXY_DEPENDENCY_CHANGED')
    exports = (out / 'exported-symbols.txt').read_text()
    for name in ('state', 'install', 'pass', 'continue', 'receipt', 'acknowledge', 'seal', 'stop', 'drain', 'restore', 'remove'):
        if not re.search(r'\bT dcat26_group_' + name + r'_v2$', exports, re.M):
            raise RuntimeError('NEW_GROUP_ABI_EXPORT_MISSING:' + name)
    if re.search(r'\bdcat26[qf]_(install|continue)_v1$', exports, re.M):
        raise RuntimeError('LEGACY_WRITER_ABI_MUST_NOT_BE_PRESENT')
    if not re.search(r'\bT dcat26_group_metadata_arrays_v1$', exports, re.M):
        raise RuntimeError('ACTUAL_GROUP_METADATA_COUNTER_EXPORT_REQUIRED')
    for name in ('abi', 'snapshots'):
        if not re.search(r'\bT dcat26_group_fia_' + name + r'_v3$', exports, re.M):
            raise RuntimeError('TYPED_FIA_GROUP_ABI_REQUIRED:' + name)
    if not re.search(r'\bT dcat26_group_continue_fia_v3$', exports, re.M):
        raise RuntimeError('TYPED_FIA_ADMISSION_ABI_REQUIRED')
    receipt.update(status='SDK_BUILD_AND_CPU_CONTRACT_PASS_NOT_NATIVE_ACCEPTANCE',
        sdk_control=json.loads((out / 'sdk-control.json').read_text()),
        pending_runtime_contract=json.loads((out / 'pending-runtime-cpu.json').read_text()),
        fia_typed_runtime_contract=json.loads((out / 'fia-typed-runtime-cpu.json').read_text()),
        cpu_contract=json.loads((out / 'cpu-contract.json').read_text()))
    profile = typed_profile(receipt['sdk_control'], receipt['fia_typed_runtime_contract'])
    receipt['fia_typed_profile'] = profile
    (out / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    output_files = {name: sha(out / name) for name in sorted(required | {'libdcat26_group_v2.so', 'metadata-array-group-cpu.json'})}
    (out / 'build.json').write_text(json.dumps(dict(schema='demoncat.writer26.group_deployment.v2',
        files=output_files, build_receipt_sha256=sha(out / 'build-receipt.json'), sources=sources,
        actual_npu_calls=0, native_qualified_targets=0, c2=False, c3=False,
        actual_image_id=receipt['actual_image_id'],
        fia_typed_profile=profile,
        metadata_array_ownership=dict(schema='demoncat.metadata_arrays.v1', state_version=1,
            images=dict(SYMBOLS, **{'libdcat26_group_v2.so': 'dcat26_group_metadata_arrays_v1'}),
            reset_supported=False, runtime_quiescent_readback_required=True,
            parent_receipt_sha256=files[RECEIPT]),
        legacy_writers_absent=True, independent_trace_required=True), indent=2) + '\n')
    print(json.dumps(dict(status=receipt['status'], bundle=str(out), manifest_sha256=sha(out / 'build.json'))))


if __name__ == '__main__':
    main()
