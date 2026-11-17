"""Root-only, device-free SDK build for the independent quant group DSO.

This program starts a CPU build container. Provider agents prepare this file
and the exact command; only the root queue owner may execute it.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import uuid


IMAGE = 'quay.io/ascend/vllm-ascend:v0.18.0'
SOURCES = tuple('fault_injection/native_adapters/' + name for name in (
    'batch_scale_group_core.h', 'batch_scale_group_runtime.h', 'batch_scale_group_runtime.cpp',
    'test_batch_scale_group_core.cpp', 'test_batch_scale_group_runtime.cpp'))
EXPORTS = ('abi', 'state', 'raw', 'install', 'public_begin', 'public_end', 'pass', 'continue',
           'seal', 'stop', 'drain', 'recover', 'remove')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_image(expected):
    rows = json.loads(subprocess.run(['docker', 'image', 'inspect', IMAGE], capture_output=True,
                                    text=True, check=True, timeout=30).stdout)
    if (len(rows) != 1 or rows[0].get('Id') != expected
            or rows[0].get('Architecture') != 'arm64' or rows[0].get('Os') != 'linux'):
        raise RuntimeError('APPROVED_ACTUAL_IMAGE_ID_AND_PLATFORM_REQUIRED')
    return rows[0]


def cleanup_owned(name, nonce, image):
    result = subprocess.run(['docker', 'container', 'inspect', name], capture_output=True, text=True, timeout=30)
    if result.returncode:
        absent = 'No such container' in result.stderr or 'No such object' in result.stderr
        return dict(confirmed=absent, status='ALREADY_ABSENT' if absent else 'INSPECT_FAILED', stderr=result.stderr)
    rows = json.loads(result.stdout)
    if (len(rows) != 1 or rows[0].get('Name') != '/' + name or rows[0].get('Image') != image
            or rows[0].get('Config', {}).get('Labels', {}).get('demoncat.quant.group-build-owner') != nonce):
        return dict(confirmed=False, status='FOREIGN_CONTAINER_REFUSED')
    result = subprocess.run(['docker', 'container', 'rm', '-f', name], capture_output=True, text=True, timeout=30)
    return dict(confirmed=result.returncode == 0, status='OWNED_REMOVAL', returncode=result.returncode,
                stdout=result.stdout, stderr=result.stderr)


COMMANDS = r'''set -euo pipefail
sdk=/usr/local/Ascend/cann-8.5.1/aarch64-linux
grep -Eq '^version=8\.5\.1$' "$sdk/ascend_toolkit_install.info"
test "$(uname -m)" = aarch64
native=/source/fault_injection/native_adapters
flags=(-std=c++17 -O2 -Wall -Wextra -Werror -pedantic -D_GLIBCXX_USE_CXX11_ABI=0 -isystem "$sdk/include" -isystem "$sdk/include/aclnn" -isystem "$sdk/pkg_inc")
mkdir -p /build/runtime/vendor/op_api/lib
g++ "${flags[@]}" -fPIC -shared "$native/batch_scale_group_runtime.cpp" -ldl -lpthread -Wl,-z,defs -Wl,-soname,libdcat_dq_group_v1.so -o /build/runtime/vendor/op_api/lib/libcust_opapi.so
g++ "${flags[@]}" "$native/test_batch_scale_group_runtime.cpp" -ldl -lpthread -o /build/test_group_runtime
/build/test_group_runtime > /build/sdk-runtime-contract.log
g++ "${flags[@]}" "$native/test_batch_scale_group_core.cpp" -o /build/test_group_core
/build/test_group_core > /build/core-contract.log
nm -D --defined-only /build/runtime/vendor/op_api/lib/libcust_opapi.so > /build/exported-symbols.txt
readelf -d /build/runtime/vendor/op_api/lib/libcust_opapi.so > /build/elf-dynamic.txt
python3 - <<'PY'
import ctypes as C,hashlib,json
from pathlib import Path
sdk=Path('/usr/local/Ascend/cann-8.5.1/aarch64-linux')
real=sdk.parent/'opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/aarch64/libopapi.so'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
names='view call config lease event binding journal plan child receipt state public_input completion runtime raw'.split()
class Sizes(C.Structure):
    _fields_=[('version',C.c_uint32),('size',C.c_uint32)]+[(name,C.c_uint64) for name in names]
sizes=Sizes();sizes.version=1;sizes.size=C.sizeof(sizes)
lib=C.CDLL('/build/runtime/vendor/op_api/lib/libcust_opapi.so',mode=C.RTLD_LOCAL)
lib.dcat_dqg_abi_v1.argtypes=[C.POINTER(Sizes)];lib.dcat_dqg_abi_v1.restype=C.c_int
assert lib.dcat_dqg_abi_v1(C.byref(sizes))==0
record=dict(actual_npu_calls=0,library_loaded_only=True,abi_sizes={name:getattr(sizes,name) for name in names},
 real_opapi=str(real),real_opapi_sha256=sha(real),
 sdk_headers={str(path.relative_to(sdk)):sha(path) for path in (sdk/'include/acl/acl.h',sdk/'include/aclnnop/aclnn_dynamic_quant_v2.h')})
Path('/build/sdk-identity.json').write_text(json.dumps(record,indent=2)+'\n')
PY
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--image-id', required=True)
    args = parser.parse_args()
    if (platform.machine() not in ('aarch64', 'arm64') or not re.fullmatch('[0-9a-f]{40}', args.source_commit)
            or not re.fullmatch('sha256:[0-9a-f]{64}', args.image_id)):
        raise RuntimeError('ACTUAL_AARCH64_HOST_COMMIT_AND_IMAGE_REQUIRED')
    source, out = args.source.resolve(strict=True), args.out.resolve()
    if out.exists():
        raise RuntimeError('NEW_BUILD_DIRECTORY_REQUIRED')
    sources = {name: sha(source / name) for name in SOURCES}
    out.mkdir(parents=True)
    nonce = uuid.uuid4().hex
    name = 'dcat-quant-group-build-' + nonce
    receipt = dict(schema='demoncat.quant.group_build.v1', status='FAILED_CPU_BUILD', actual_npu_calls=0,
        source_commit=args.source_commit, sources=sources, requested_image=IMAGE, actual_image_id=None,
        approved_image_id=args.image_id, container_name=name, container_owner_nonce=nonce,
        device_mounts=[], command=None, build_returncode=None)
    launched = False
    try:
        actual = inspect_image(args.image_id)
        (out / 'image-inspect.json').write_text(json.dumps(actual, indent=2) + '\n')
        receipt.update(actual_image_id=actual['Id'], image_inspect_sha256=sha(out / 'image-inspect.json'))
        command = ['docker', 'run', '--rm', '--name', name, '--label', 'demoncat.quant.group-build-owner=' + nonce,
            '--network', 'none', '--runtime', 'runc', '--cpus', '4', '--memory', '4g',
            '-e', 'ASCEND_VISIBLE_DEVICES=void', '-v', str(source) + ':/source:ro', '-v', str(out) + ':/build',
            '--entrypoint', 'bash', actual['Id'], '-lc', COMMANDS]
        receipt['command'] = command
        with (out / 'build.log').open('wb') as stream:
            launched = True
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
        receipt['build_returncode'] = result.returncode
        if result.returncode:
            raise RuntimeError('SDK_COMPILATION_OR_CPU_CONTRACT_FAILED')
        if any(sha(source / name) != expected for name, expected in sources.items()):
            raise RuntimeError('SOURCE_CHANGED_DURING_BUILD')
        symbols = (out / 'exported-symbols.txt').read_text()
        for symbol in tuple('dcat_dqg_' + name + '_v1' for name in EXPORTS) + (
                'aclnnDynamicQuantV2GetWorkspaceSize', 'aclnnDynamicQuantV2', 'CanUsePTACache'):
            if not re.search(r'\bT ' + symbol + r'$', symbols, re.M):
                raise RuntimeError('MISSING_TYPED_SYMBOL:' + symbol)
        if re.search(r'\b(?:dcat_dq_arm|dcat_dq_finish|aclnnDynamicQuant(?:V[134])?GetWorkspaceSize)$', symbols, re.M):
            raise RuntimeError('LEGACY_OR_UNREVIEWED_ABI_PRESENT')
        for relative in SOURCES:
            shutil.copyfile(source / relative, out / Path(relative).name)
        receipt['sdk_identity'] = json.loads((out / 'sdk-identity.json').read_bytes())
        receipt['status'] = 'SDK_BUILD_AND_CPU_CONTRACT_PASS_NOT_NPU_ACCEPTANCE'
    except BaseException as error:
        receipt.update(status='FAILED_CPU_BUILD', error_type=type(error).__name__, error=str(error),
                       timed_out=isinstance(error, subprocess.TimeoutExpired))
    finally:
        try:
            receipt['cleanup'] = cleanup_owned(name, nonce, args.image_id) if launched else dict(confirmed=True, status='NOT_LAUNCHED')
        except BaseException as error:
            receipt['cleanup'] = dict(confirmed=False, status='CLEANUP_FAILED', error=repr(error))
        if (out / 'build.log').exists():
            receipt['build_log_sha256'] = sha(out / 'build.log')
        (out / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if receipt['status'] == 'FAILED_CPU_BUILD' or not receipt['cleanup']['confirmed']:
        raise RuntimeError('CPU_BUILD_OR_OWNED_CLEANUP_FAILED_RECEIPT_PRESERVED')
    names = ('runtime/vendor/op_api/lib/libcust_opapi.so',) + tuple(Path(name).name for name in SOURCES)
    manifest = dict(schema='demoncat.quant.group_deployment.v1', actual_npu_calls=0, c2=False, c3=False,
        native_qualified_targets=0, source_commit=args.source_commit, sources=sources,
        actual_image_id=args.image_id, cann='8.5.1', library='runtime/vendor/op_api/lib/libcust_opapi.so',
        files={name: sha(out / name) for name in names}, sdk_identity=receipt['sdk_identity'],
        build_receipt_sha256=sha(out / 'build-receipt.json'), legacy_c1_library_present=False)
    (out / 'build.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(status=receipt['status'], bundle=str(out), manifest_sha256=sha(out / 'build.json'))))


if __name__ == '__main__':
    main()
