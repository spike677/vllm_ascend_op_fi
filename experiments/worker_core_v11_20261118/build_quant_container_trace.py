"""Root-owned device-free SDK build; agents prepare this but never run Docker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.c2c3_closure_20261117.providers_quant_build_group import inspect_image, cleanup_owned

IMAGE = 'sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0'
SDK = Path('/usr/local/Ascend/cann-8.5.1')
PACKAGE = Path('/vllm-workspace/vllm-ascend/vllm_ascend')
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


def declaration(name, image):
    rows = ','.join('{%d,%d,"%s"}' % (x['vaddr'], x['size'], x['sha256']) for x in image['segments'])
    return ('static constexpr f39identity::Segment ' + name + '_segments[]={' + rows + '};\n'
            + 'static constexpr f39identity::Image ' + name + '={"' + image['file_sha256'] + '",'
            + str(image['file_bytes']) + ',' + name + '_segments,' + str(len(image['segments'])) + '};\n')


def flags(out):
    return ['-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic', '-D_GLIBCXX_USE_CXX11_ABI=0',
            '-isystem', str(SDK / 'include'), '-isystem', str(SDK / 'include/aclnn'), '-I', str(out)]


def inside(source, out):
    from fault39.tools.version_profile import image
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
    real = library.parent / 'real/libqct_custom_real.so'
    real.parent.mkdir(parents=True)
    shutil.copyfile(paths['custom'], real)
    run(['patchelf', '--set-soname', 'libqct_custom_real.so', real])
    originals = {key: image(path) for key, path in paths.items()}
    images = {key: value for key, value in originals.items() if key != 'custom'}
    images['custom_real'] = image(real)
    if originals['custom']['segments'] != images['custom_real']['segments']:
        raise RuntimeError('SONAME_REWRITE_CHANGED_EXECUTABLE_SEGMENTS')
    profile = dict(schema='demoncat.quant.container_profile.v1', image=IMAGE,
        actual_npu_calls=0, originals=originals, images=images,
        sdk_header_sha256=sha(header), functions=list(FUNCTIONS),
        sources={name: sha(source / name) for name in SOURCES},
        scope='FOUR_CONTAINER_APIS_ONLY_NOT_NATIVE_ROW_AUTHORITY')
    ident = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    profile['profile_sha256'] = ident
    (out / 'profile.json').write_text(json.dumps(profile, sort_keys=True, indent=2) + '\n')
    (out / 'quant_container_profile.h').write_text('namespace qct_profile {\nstatic constexpr const char* id="'
        + ident + '";\n' + ''.join(declaration(k, v) for k, v in images.items()) + '}\n')
    native = source / 'fault_injection/native_adapters'
    run(['g++', *flags(out), '-fPIC', '-shared', native / 'quant_container_trace.cpp',
        '-L', real.parent, '-L', nn.parent, '-Wl,--no-as-needed', '-l:libqct_custom_real.so', '-lnnopbase',
        '-ldl', '-lpthread', '-Wl,-z,defs', '-Wl,-soname,libcust_opapi.so',
        '-Wl,-rpath,$ORIGIN/real', '-Wl,-z,origin', '-o', library])
    run(['g++', *flags(out), native / 'test_quant_container_trace.cpp', '-ldl', '-lpthread',
         '-o', out / 'cpu_contract'])
    with (out / 'cpu-contract.log').open('wb') as stream:
        for case in CASES:
            run([out / 'cpu_contract', case], stdout=stream, stderr=subprocess.STDOUT)
    symbols = run(['nm', '-D', '--defined-only', library], capture_output=True, text=True).stdout
    (out / 'exports.txt').write_text(symbols)
    for symbol in FUNCTIONS + EXPORTS:
        if not re.search(r'\bT ' + symbol + r'$', symbols, re.M):
            raise RuntimeError('MISSING_EXACT_TYPED_EXPORT:' + symbol)
    if re.search(r'\bT (?:aclnn\w+|CanUsePTACache|dcat_dq\w+)$', symbols, re.M):
        raise RuntimeError('UNREQUESTED_OPERATOR_OR_FAULT_EXPORT')
    dynamic = run(['readelf', '-d', library], capture_output=True, text=True).stdout
    (out / 'dynamic.txt').write_text(dynamic)
    if '[libqct_custom_real.so]' not in dynamic or '[libnnopbase.so]' not in dynamic:
        raise RuntimeError('EXACT_ORIGINAL_DEPENDENCIES_REQUIRED')
    # Loading and these read-only marker calls do not invoke a creator or NPU.
    import ctypes
    loaded = ctypes.CDLL(str(library), mode=os.RTLD_LOCAL | os.RTLD_NOW)
    loaded.dcat_qct_marker_v1.restype = ctypes.c_uint32
    loaded.dcat_qct_profile_v1.restype = ctypes.c_char_p
    if loaded.dcat_qct_marker_v1() != 1 or loaded.dcat_qct_profile_v1().decode() != ident:
        raise RuntimeError('REAL_SDK_DSO_MARKER_PROFILE_LOAD_FAILED')
    (out / 'sdk-load.json').write_text(json.dumps(dict(actual_npu_calls=0, creator_calls=0,
        install_calls=0, profile_sha256=ident, status='SDK_DSO_LOADED_MARKERS_ONLY_NOT_RUNTIME_ACCEPTANCE')) + '\n')


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
        command = ['docker', 'run', '--rm', '--name', name, '--label', 'demoncat.quant.group-build-owner=' + nonce,
            '--network', 'none', '--runtime', 'runc', '--cpus', '4', '--memory', '4g', '-e', 'ASCEND_VISIBLE_DEVICES=void',
            '-e', 'PYTHONPATH=/source', '-v', str(source) + ':/source:ro', '-v', str(out) + ':/build',
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
        receipt['status'] = 'SDK_AND_TYPED_CPU_CONTRACT_PASS_NOT_NPU'
    except BaseException as error:
        receipt.update(error=repr(error), timed_out=isinstance(error, subprocess.TimeoutExpired))
    finally:
        try:
            receipt['cleanup'] = cleanup_owned(name, nonce, IMAGE) if launched else dict(confirmed=True, status='NOT_LAUNCHED')
        except BaseException as error:
            receipt['cleanup'] = dict(confirmed=False, error=repr(error))
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
