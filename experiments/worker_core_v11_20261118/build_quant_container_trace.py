"""Root-owned device-free SDK build; agents prepare this but never run Docker."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.c2c3_closure_20261117.providers_quant_build_group import inspect_image, cleanup_owned

IMAGE = 'sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0'
SDK = Path('/usr/local/Ascend/cann-8.5.1')
PACKAGE = Path('/vllm-workspace/vllm-ascend/vllm_ascend')
DRIVER = PurePosixPath('/usr/local/Ascend/driver/lib64')
DRIVER_PINS = {
    'driver/libascend_hal.so': '3ccd859a4df96c13106ef6b950aee89d61cee50cf743172d288fc76591b5a9e6',
    'common/libc_sec.so': 'c61ed19a711a75b1ac5c3c6793d7d3ec2848846eb8113da828632881b8d91db8',
}
PINS = dict(nn='6dc3b4fcfcf57b10be0018e89f4edbc079f02b4bb0aaadb109df350939fd4c99',
    custom='de1633c2f1f41e505277697ee82f3ffba940f730bed9986c5aeff4c8514b713f',
    torch_npu='dd07e4afb4e31a5123d0b97a4a319acaa96158705a22d704bfc0a91f662a6041',
    custom_extension='e63806486e0311c2df8ce9b120ae98f55e8c8eb91ccaf204ef0cf7d6b469ed67')
SOURCES = ('fault_injection/native_adapters/quant_container_trace.h',
    'fault_injection/native_adapters/quant_container_trace.cpp',
    'fault_injection/native_adapters/test_quant_container_trace.cpp',
    'fault39/native/runtime_identity.h', 'fault39/native/metadata_array_owner.h',
    'fault39/native/metadata_array_state.h', 'fault39/tools/version_profile.py',
    'experiments/worker_core_v11_20261118/build_quant_container_trace.py',
    'experiments/c2c3_closure_20261117/providers_quant_build_group.py')
CASES = ('normal', 'payload', 'scope', 'foreign_thread', 'getter_status', 'getter_throw',
    'destroy_retry', 'unknown_destroy', 'null_destroy', 'failed_create', 'reused_address',
    'duplicate_live', 'call_budget', 'byte_budget', 'member_budget', 'owner', 'scope_retry', 'read_bounds', 'empty')
FUNCTIONS = ('aclCreateTensorList', 'aclDestroyTensorList', 'aclCreateIntArray', 'aclDestroyIntArray')
EXPORTS = tuple('dcat_qct_' + name + '_v1' for name in
                ('marker', 'profile', 'sizes', 'state', 'begin', 'scope_begin', 'scope_end', 'stop', 'event', 'payload', 'release'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(argv, **kwargs):
    return subprocess.run([str(x) for x in argv], check=True, **kwargs)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2) + '\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def driver_identity(root=DRIVER):
    root = Path(root).resolve(strict=True); files = {}
    for relative, expected in DRIVER_PINS.items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise RuntimeError('DRIVER_DEPENDENCY_ESCAPES_READONLY_MOUNT:' + relative)
        before = path.stat(); actual = sha(path); after = path.stat()
        if (actual != expected or (relative == 'driver/libascend_hal.so' and after.st_size != 1861080)
                or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
            raise RuntimeError('ACTUAL_DRIVER_DEPENDENCY_CHANGED:' + relative)
        files[relative] = dict(path=str(path), sha256=actual, bytes=after.st_size,
                               device=after.st_dev, inode=after.st_ino)
    return dict(schema='demoncat.quant.container_driver_dependencies.v1', root=str(root),
                files=files, device_mounts=[], explicit_driver_api_calls=0)


def loader_path(image):
    existing = [value.split('=', 1)[1] for value in image.get('Config', {}).get('Env', [])
                if value.startswith('LD_LIBRARY_PATH=')]
    if len(existing) > 1:
        raise RuntimeError('AMBIGUOUS_IMAGE_LOADER_PATH')
    paths = [str(DRIVER / 'driver'), str(DRIVER / 'common')]
    if existing:
        paths += [path for path in existing[0].split(':') if path]
    if any(not path.startswith('/') or any(c in path for c in ('\n', '\r', '\0', '$')) for path in paths):
        raise RuntimeError('EXPLICIT_ABSOLUTE_LOADER_PATH_REQUIRED')
    return ':'.join(dict.fromkeys(paths))


def loaded_driver_mappings(expected, maps=None):
    """Only the two actual read-only driver dependencies, never all system ELFs."""
    rows = []; seen = set(); allowed = {row['path']: row for row in expected['files'].values()}
    for line in (Path('/proc/self/maps').read_text() if maps is None else maps).splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) != 6 or not parts[5].startswith(expected['root'] + '/'):
            continue
        path = parts[5]
        if path.endswith(' (deleted)') or path not in allowed:
            raise RuntimeError('UNREVIEWED_OR_DELETED_DRIVER_MAPPING:' + path)
        item = allowed[path]
        if (int(parts[4]) != item['inode'] or tuple(int(x, 16) for x in parts[3].split(':'))
                != (os.major(item['device']), os.minor(item['device'])) or sha(path) != item['sha256']):
            raise RuntimeError('LOADED_DRIVER_MAPPING_IDENTITY_CHANGED:' + path)
        seen.add(path)
        rows.append(dict(path=path, address_range=parts[0], permissions=parts[1], offset=parts[2],
                         mapped_device=parts[3], mapped_inode=int(parts[4]), sha256=item['sha256']))
    if seen != set(allowed):
        raise RuntimeError('REAL_HAL_AND_LIBC_SEC_MAPPINGS_REQUIRED')
    return rows


def cpu_contracts(source, out):
    receipt = dict(schema='demoncat.quant.container_cpu_contract.v1', status='INCOMPLETE',
        actual_npu_calls=0, scope='SDK_TYPED_REAL_SHIM_WITH_CPU_FUNCTION_DOUBLES',
        sources={name: sha(source / name) for name in SOURCES}, executable_sha256=sha(out / 'cpu_contract'),
        expected_cases=list(CASES), cases=[])
    path = out / 'cpu-contract.json'; write_json(path, receipt)
    try:
        with (out / 'cpu-contract.log').open('wb') as combined:
            for case in CASES:
                log = out / ('cpu-contract-' + case + '.log')
                row = dict(case=case, status='STARTED', returncode=None, log=log.name)
                receipt['cases'].append(row); write_json(path, receipt)
                try:
                    with log.open('wb') as stream:
                        result = run([out / 'cpu_contract', case], stdout=stream, stderr=subprocess.STDOUT)
                    row['returncode'] = result.returncode
                    expected = ('PASS_CPU_SDK_TYPED_FUNCTION_DOUBLE_NOT_NPU ' + case + '\n').encode()
                    if log.read_bytes() != expected:
                        raise RuntimeError('EXACT_CPU_CONTRACT_OUTPUT_REQUIRED:' + case)
                    row['status'] = 'PASS'
                except BaseException as error:
                    row.update(status='FAILED', error=repr(error), returncode=getattr(error, 'returncode', row['returncode']))
                    raise
                finally:
                    if log.exists():
                        raw = log.read_bytes(); row.update(log_sha256=sha(log), log_bytes=len(raw))
                        combined.write(raw); combined.flush(); os.fsync(combined.fileno())
                    write_json(path, receipt)
        receipt['status'] = 'PASS_ALL_19_SDK_TYPED_CPU_CONTRACTS_NOT_NPU'
    finally:
        if (out / 'cpu-contract.log').exists():
            receipt['combined_log_sha256'] = sha(out / 'cpu-contract.log')
        write_json(path, receipt)


def marker_load(out, library, ident):
    """Persist this separate stage even if the real loader rejects a dependency."""
    import ctypes
    receipt = dict(status='FAILED_SDK_MARKER_LOAD', actual_npu_calls=0, creator_calls=0,
        install_calls=0, explicit_driver_api_calls=0, profile_sha256=ident,
        driver_identity_sha256=sha(out / 'driver-identity.json'))
    try:
        expected = json.loads((out / 'driver-identity.json').read_text())
        before = driver_identity()
        if before != expected:
            raise RuntimeError('DRIVER_BYTES_OR_MOUNT_CHANGED_BEFORE_LOAD')
        loaded = ctypes.CDLL(str(library), mode=os.RTLD_LOCAL | os.RTLD_NOW)
        loaded.dcat_qct_marker_v1.restype = ctypes.c_uint32
        loaded.dcat_qct_profile_v1.restype = ctypes.c_char_p
        if loaded.dcat_qct_marker_v1() != 1 or loaded.dcat_qct_profile_v1().decode() != ident:
            raise RuntimeError('REAL_SDK_DSO_MARKER_PROFILE_LOAD_FAILED')
        maps = Path('/proc/self/maps').read_text()
        receipt['driver_maps_raw'] = [line for line in maps.splitlines() if expected['root'] + '/' in line]
        receipt['loaded_driver_mappings'] = loaded_driver_mappings(expected, maps=maps)
        after = driver_identity()
        if after != before:
            raise RuntimeError('DRIVER_BYTES_CHANGED_DURING_LOAD')
        receipt['driver_identity_after'] = after
        receipt['status'] = 'SDK_DSO_LOADED_MARKERS_ONLY_NOT_RUNTIME_ACCEPTANCE'
    except BaseException as error:
        receipt['error'] = repr(error)
        raise
    finally:
        write_json(out / 'sdk-load.json', receipt)


def declaration(name, image):
    rows = ','.join('{%d,%d,"%s"}' % (x['vaddr'], x['size'], x['sha256']) for x in image['segments'])
    return ('static constexpr f39identity::Segment ' + name + '_segments[]={' + rows + '};\n'
            + 'static constexpr f39identity::Image ' + name + '={"' + image['file_sha256'] + '",'
            + str(image['file_bytes']) + ',' + name + '_segments,' + str(len(image['segments'])) + '};\n')


def flags(out):
    return ['-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic', '-D_GLIBCXX_USE_CXX11_ABI=0',
            '-isystem', str(SDK / 'include'), '-isystem', str(SDK / 'include/aclnn'), '-I', str(out)]


def absolute_needed_contract(out):
    """Actual owned CPU ELF fixture; unchanged original and equal SONAMEs."""
    directory = out / 'absolute-needed-contract'; (directory / 'original').mkdir(parents=True)
    original = directory / 'original/libcust_opapi.so'; proxy = directory / 'libcust_opapi.so'
    (directory / 'original.cpp').write_text('extern "C" int fixture_original_anchor(){return 37;}\n')
    (directory / 'proxy.cpp').write_text('extern "C" int fixture_proxy_marker(){return 91;}\n')
    for source, target in ((directory / 'original.cpp', original), (directory / 'proxy.cpp', proxy)):
        run(['g++', '-shared', '-fPIC', '-Wall', '-Wextra', '-Werror', source,
             '-Wl,-soname,libcust_opapi.so', '-o', target])
    original_sha = sha(original)
    run(['patchelf', '--add-needed', str(original), proxy])
    script = '''import ctypes as C,json,os,pathlib,sys
class Info(C.Structure):
 _fields_=[('fname',C.c_char_p),('base',C.c_void_p),('sname',C.c_char_p),('addr',C.c_void_p)]
locate=C.CDLL(None).dladdr;locate.argtypes=[C.c_void_p,C.POINTER(Info)];locate.restype=C.c_int
def origin(fn):
 info=Info();assert locate(C.cast(fn,C.c_void_p),C.byref(info));return str(pathlib.Path(os.fsdecode(info.fname)).resolve())
proxy,original=map(pathlib.Path,sys.argv[1:]);mode=os.RTLD_NOW|os.RTLD_NOLOAD
by_name=C.CDLL('libcust_opapi.so',mode=mode);real=C.CDLL(str(original),mode=mode)
assert origin(by_name.fixture_proxy_marker)==str(proxy)
assert origin(by_name.fixture_original_anchor)==origin(real.fixture_original_anchor)==str(original)
assert by_name.fixture_proxy_marker()==91 and real.fixture_original_anchor()==37
print(json.dumps(dict(status='PASS_REAL_ELF_ABSOLUTE_NEEDED_SAME_SONAME',proxy=str(proxy),original=str(original),
 actual_npu_calls=0,mocked_cpu_anchor_calls=2)))
'''
    environment = dict(os.environ, LD_PRELOAD=str(proxy))
    result = run([sys.executable, '-c', script, proxy, original], env=environment, capture_output=True, text=True)
    if sha(original) != original_sha:
        raise RuntimeError('ABSOLUTE_DEPENDENCY_ORIGINAL_BYTES_CHANGED')
    proof = json.loads(result.stdout); proof['original_sha256_before'] = proof['original_sha256_after'] = original_sha
    (out / 'absolute-needed-contract.json').write_text(json.dumps(proof, indent=2) + '\n')


def inside(source, out):
    from fault39.tools.version_profile import image
    if driver_identity() != json.loads((out / 'driver-identity.json').read_text()):
        raise RuntimeError('ACTUAL_READONLY_DRIVER_MOUNT_IDENTITY_REQUIRED')
    nn = SDK / 'lib64/libnnopbase.so'
    paths = dict(nn=nn, custom=PACKAGE / '_cann_ops_custom/vendors/vllm-ascend/op_api/lib/libcust_opapi.so',
        torch_npu=Path('/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/lib/libtorch_npu.so'),
        custom_extension=PACKAGE / 'vllm_ascend_C.cpython-311-aarch64-linux-gnu.so')
    if any(sha(path) != PINS[key] for key, path in paths.items()):
        raise RuntimeError('FIXED_PRODUCTION_ELF_BYTES_REQUIRED')
    header = SDK / 'include/aclnn/acl_meta.h'
    if sha(header) != 'be93460abfc7aa14113d7b9a3536fe2dd45d5148a357f55e15847655c9d5ffed':
        raise RuntimeError('EXACT_PUBLIC_OPAQUE_CONTAINER_SDK_ABI_REQUIRED')
    library = out / 'runtime/vendor/op_api/lib/libcust_opapi.so'
    library.parent.mkdir(parents=True)
    absolute_needed_contract(out)
    originals = {key: image(path) for key, path in paths.items()}
    images = {key: value for key, value in originals.items() if key != 'custom'}
    images['custom_real'] = originals['custom']
    profile = dict(schema='demoncat.quant.container_profile.v1', image=IMAGE,
        actual_npu_calls=0, originals=originals, images=images,
        sdk_header_sha256=sha(header), functions=list(FUNCTIONS),
        sources={name: sha(source / name) for name in SOURCES},
        scope='FOUR_CONTAINER_APIS_ONLY_NOT_NATIVE_ROW_AUTHORITY',
        original_loading='ABSOLUTE_DT_NEEDED_ORIGINAL_IMAGE_BYTES_UNCHANGED')
    ident = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    profile['profile_sha256'] = ident
    (out / 'profile.json').write_text(json.dumps(profile, sort_keys=True, indent=2) + '\n')
    (out / 'quant_container_profile.h').write_text('namespace qct_profile {\nstatic constexpr const char* id="'
        + ident + '";\nstatic constexpr const char* custom_path="' + str(paths['custom'].resolve())
        + '";\n' + ''.join(declaration(k, v) for k, v in images.items()) + '}\n')
    native = source / 'fault_injection/native_adapters'
    run(['g++', *flags(out), '-fPIC', '-shared', native / 'quant_container_trace.cpp',
        '-L', nn.parent, '-Wl,--no-as-needed', '-lnnopbase',
        '-ldl', '-lpthread', '-Wl,-z,defs', '-Wl,-soname,libcust_opapi.so',
        '-o', library])
    # Only our newly compiled wrapper is changed. The fixed image's original
    # retains its entire file and executable segments, including its SONAME.
    run(['patchelf', '--add-needed', str(paths['custom'].resolve()), library])
    if any(sha(path) != PINS[key] for key, path in paths.items()):
        raise RuntimeError('ORIGINAL_IMAGE_BYTES_CHANGED_DURING_BUILD')
    run(['g++', *flags(out), native / 'test_quant_container_trace.cpp', '-ldl', '-lpthread',
         '-o', out / 'cpu_contract'])
    cpu_contracts(source, out)
    symbols = run(['nm', '-D', '--defined-only', library], capture_output=True, text=True).stdout
    (out / 'exports.txt').write_text(symbols)
    for symbol in FUNCTIONS + EXPORTS:
        if not re.search(r'\bT ' + symbol + r'$', symbols, re.M):
            raise RuntimeError('MISSING_EXACT_TYPED_EXPORT:' + symbol)
    if re.search(r'\bT (?:aclnn\w+|CanUsePTACache|dcat_dq\w+)$', symbols, re.M):
        raise RuntimeError('UNREQUESTED_OPERATOR_OR_FAULT_EXPORT')
    dynamic = run(['readelf', '-d', library], capture_output=True, text=True).stdout
    (out / 'dynamic.txt').write_text(dynamic)
    if '[' + str(paths['custom'].resolve()) + ']' not in dynamic or '[libnnopbase.so]' not in dynamic:
        raise RuntimeError('EXACT_ORIGINAL_DEPENDENCIES_REQUIRED')
    marker_load(out, library, ident)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source-commit', required=True); p.add_argument('--image-id', default=IMAGE)
    p.add_argument('--inside', action='store_true'); a = p.parse_args()
    source, out = a.source.resolve(strict=True), a.out.resolve()
    if a.image_id != IMAGE or not re.fullmatch('[0-9a-f]{40}', a.source_commit):
        raise RuntimeError('FIXED_IMAGE_AND_ACTUAL_SOURCE_COMMIT_REQUIRED')
    if a.inside:
        inside(source, out); return
    if out.exists():
        raise RuntimeError('NEW_BUILD_DIRECTORY_REQUIRED')
    sources = {name: sha(source / name) for name in SOURCES}; out.mkdir(parents=True)
    nonce = uuid.uuid4().hex; name = 'dcat-quant-container-build-' + nonce
    receipt = dict(schema='demoncat.quant.container_build.v1', status='FAILED_CPU_BUILD', actual_npu_calls=0,
        source_commit=a.source_commit, sources=sources, image_id=IMAGE, container_name=name, container_owner_nonce=nonce)
    launched = False
    try:
        actual = inspect_image(IMAGE)
        (out / 'image-inspect.json').write_text(json.dumps(actual, indent=2) + '\n')
        write_json(out / 'driver-identity.json', driver_identity())
        receipt['driver_identity_sha256'] = sha(out / 'driver-identity.json')
        receipt['loader_path'] = loader_path(actual)
        command = ['docker', 'run', '--rm', '--name', name, '--label', 'demoncat.quant.group-build-owner=' + nonce,
            '--network', 'none', '--runtime', 'runc', '--cpus', '4', '--memory', '4g', '-e', 'ASCEND_VISIBLE_DEVICES=void',
            '-e', 'PYTHONPATH=/source', '-e', 'LD_LIBRARY_PATH=' + receipt['loader_path'],
            '-v', str(DRIVER) + ':' + str(DRIVER) + ':ro',
            '-v', str(source) + ':/source:ro', '-v', str(out) + ':/build',
            '--entrypoint', 'python3', actual['Id'], '/source/' + SOURCES[-2],
            '--source', '/source', '--out', '/build', '--source-commit', a.source_commit, '--inside']
        receipt['command'] = command; receipt['image_inspect_sha256'] = sha(out / 'image-inspect.json')
        with (out / 'build.log').open('wb') as stream:
            launched = True
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
        receipt['returncode'] = result.returncode
        if result.returncode or any(sha(source / key) != value for key, value in sources.items()):
            raise RuntimeError('SDK_COMPILE_CONTRACT_OR_FROZEN_SOURCE_FAILED')
        lines = (out / 'cpu-contract.log').read_text().splitlines()
        if lines != ['PASS_CPU_SDK_TYPED_FUNCTION_DOUBLE_NOT_NPU ' + case for case in CASES]:
            raise RuntimeError('COMPLETE_TYPED_SHIM_CPU_CONTRACT_REQUIRED')
        contracts = json.loads((out / 'cpu-contract.json').read_text())
        if (contracts['status'] != 'PASS_ALL_19_SDK_TYPED_CPU_CONTRACTS_NOT_NPU'
                or contracts['expected_cases'] != list(CASES)
                or [row['case'] for row in contracts['cases']] != list(CASES)
                or any(row['status'] != 'PASS' or row['returncode'] != 0
                       or sha(out / row['log']) != row['log_sha256'] for row in contracts['cases'])):
            raise RuntimeError('COMPLETE_PERSISTENT_TYPED_SHIM_CPU_RECEIPT_REQUIRED')
        receipt['status'] = 'SDK_AND_TYPED_CPU_CONTRACT_PASS_NOT_NPU'
    except BaseException as error:
        receipt.update(error=repr(error), timed_out=isinstance(error, subprocess.TimeoutExpired))
    finally:
        try:
            receipt['cleanup'] = cleanup_owned(name, nonce, IMAGE) if launched else dict(confirmed=True, status='NOT_LAUNCHED')
        except BaseException as error:
            receipt['cleanup'] = dict(confirmed=False, error=repr(error))
        receipt['stage_receipts'] = {}
        for name in ('cpu-contract.json', 'sdk-load.json'):
            if (out / name).exists():
                try:
                    receipt['stage_receipts'][name] = dict(status=json.loads((out / name).read_text())['status'],
                                                          sha256=sha(out / name))
                except BaseException as error:
                    receipt['stage_receipts'][name] = dict(error=repr(error), sha256=sha(out / name))
        receipt['artifacts'] = {path.name: sha(path) for path in out.iterdir() if path.is_file()}
        (out / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if receipt['status'] != 'SDK_AND_TYPED_CPU_CONTRACT_PASS_NOT_NPU' or not receipt['cleanup']['confirmed']:
        raise RuntimeError('CPU_BUILD_FAILED_RECEIPT_PRESERVED')
    files = {str(path.relative_to(out)): sha(path) for path in (out / 'runtime').rglob('*') if path.is_file()}
    files['profile.json'] = sha(out / 'profile.json')
    manifest = dict(schema='demoncat.quant.container_deployment.v1', actual_npu_calls=0,
        source_commit=a.source_commit, image_id=IMAGE, sources=sources, files=files,
        library='runtime/vendor/op_api/lib/libcust_opapi.so', profile_sha256=json.loads((out / 'profile.json').read_text())['profile_sha256'],
        build_receipt_sha256=sha(out / 'build-receipt.json'), qualified=False, c1=False, c2=False, c3=False)
    (out / 'build.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(status=receipt['status'], build_sha256=sha(out / 'build.json'), out=str(out))))


if __name__ == '__main__':
    main()
