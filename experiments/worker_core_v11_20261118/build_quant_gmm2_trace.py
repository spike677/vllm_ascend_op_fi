"""Root-only device-free SDK build for the single WeightNz observation pair."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.worker_core_v11_20261118 import build_quant_container_trace as base

IMAGE, SDK, PACKAGE = base.IMAGE, base.SDK, base.PACKAGE
HEADER_SHA = 'f41308d5814aa82ba92ff77aa8e258b15ca3ae72d2c0c61fadc2cdd95de6d7fd'
RUNTIME_HEADER_SHA = '8cce153ef9eb5ed49b0f5bfb5608de706e8d466c836ad2eb50745ec2e8629bc9'
PINS = dict(base.PINS, opapi='b3890625d3e0cc11ef32460083698ecaa8f702b5edd0169ea617e2e6bb76270c',
            device_runtime='bdfa66949fc798ff21c0124a025b4227e119678cc819febed9fe1c39f2a87306')
SOURCES = base.SOURCES + ('fault_injection/native_adapters/quant_gmm2_trace.h',
    'fault_injection/native_adapters/quant_gmm2_trace.inc', 'fault_injection/native_adapters/test_quant_gmm2_trace.cpp',
    'experiments/worker_core_v11_20261118/build_quant_gmm2_trace.py')
PAIR = ('aclnnGroupedMatmulWeightNzGetWorkspaceSize', 'aclnnGroupedMatmulWeightNz')
CONTROLS = tuple('dcat_qct_gmm2_' + value + '_v1' for value in
                 ('marker', 'resolve', 'sizes', 'begin', 'state', 'event', 'stop', 'release'))
CASES = ('original_four', 'normal', 'size_mismatch', 'size_status', 'getter_throw',
    'unknown_container', 'wrong_caller', 'changed_owner', 'device_error', 'wrong_device', 'foreign_thread',
    'workspace_short', 'failed_prepare', 'failed_prepare_null', 'prepare_throw', 'no_execute',
    'execute_error', 'execute_retry', 'execute_throw', 'call_budget', 'pair_budget', 'byte_budget',
    'executor_swapped', 'executor_reused', 'scope_changed', 'null_optional', 'release_order', 'active_scope_guard')


def cpu_pair_contracts(source, out):
    receipt = dict(schema='demoncat.quant.gmm2_cpu_contract.v1', status='INCOMPLETE',
        actual_npu_calls=0, scope='SDK_TYPED_REAL_EXTENSION_WITH_CPU_FUNCTION_DOUBLES',
        sources={name: base.sha(source / name) for name in SOURCES},
        executable_sha256=base.sha(out / 'gmm2_cpu_contract'), expected_cases=list(CASES), cases=[])
    path = out / 'gmm2-cpu-contract.json'; base.write_json(path, receipt)
    try:
        with (out / 'gmm2-cpu-contract.log').open('wb') as combined:
            for case in CASES:
                log = out / ('gmm2-contract-' + case + '.log')
                row = dict(case=case, status='STARTED', returncode=None, log=log.name)
                receipt['cases'].append(row); base.write_json(path, receipt)
                try:
                    with log.open('wb') as stream:
                        result = base.run([out / 'gmm2_cpu_contract', case], stdout=stream, stderr=subprocess.STDOUT)
                    row['returncode'] = result.returncode
                    if log.read_bytes() != ('PASS_CPU_SDK_GMM2_TYPED_FUNCTION_DOUBLE_NOT_NPU ' + case + '\n').encode():
                        raise RuntimeError('EXACT_GMM2_CPU_CONTRACT_OUTPUT_REQUIRED:' + case)
                    row['status'] = 'PASS'
                except BaseException as error:
                    row.update(status='FAILED', error=repr(error), returncode=getattr(error, 'returncode', row['returncode']))
                    raise
                finally:
                    if log.exists():
                        raw = log.read_bytes(); row.update(log_sha256=base.sha(log), log_bytes=len(raw))
                        combined.write(raw); combined.flush(); os.fsync(combined.fileno())
                    base.write_json(path, receipt)
        receipt['status'] = 'PASS_ALL_GMM2_SDK_TYPED_CPU_CONTRACTS_NOT_NPU'
    finally:
        if (out / 'gmm2-cpu-contract.log').exists():
            receipt['combined_log_sha256'] = base.sha(out / 'gmm2-cpu-contract.log')
        base.write_json(path, receipt)


def marker_preflight(out, library, ident):
    # Existing marker load retains its own stage receipt if a dependency fails.
    base.marker_load(out, library, ident)
    receipt = dict(status='FAILED_GMM2_STARTUP_RESOLUTION', actual_npu_calls=0,
        operator_calls=0, device_query_calls=0, install_calls=0,
        loader_constructor_driver_calls='NOT_COUNTED_MAY_PROBE_DEVICE_INFORMATION')
    try:
        loaded = C.CDLL(str(library), mode=os.RTLD_NOW | os.RTLD_NOLOAD | os.RTLD_LOCAL)
        loaded.dcat_qct_gmm2_marker_v1.restype = C.c_uint32
        loaded.dcat_qct_gmm2_resolve_v1.restype = C.c_int
        if loaded.dcat_qct_gmm2_marker_v1() != 1 or loaded.dcat_qct_gmm2_resolve_v1():
            raise RuntimeError('EXACT_PRELOADED_GMM2_ORIGINALS_REQUIRED')
        receipt['status'] = 'GMM2_ORIGINALS_RESOLVED_WITHOUT_DEVICE_OR_OPERATOR_CALLS'
    except BaseException as error:
        receipt['error'] = repr(error)
        raise
    finally:
        base.write_json(out / 'gmm2-startup.json', receipt)


def inside(source, out):
    from fault39.tools.version_profile import image
    if base.driver_identity() != json.loads((out / 'driver-identity.json').read_text()):
        raise RuntimeError('ACTUAL_READONLY_DRIVER_MOUNT_IDENTITY_REQUIRED')
    paths = dict(nn=SDK / 'lib64/libnnopbase.so', opapi=SDK / 'lib64/libopapi.so',
        device_runtime=SDK / 'lib64/libascendcl.so',
        custom=PACKAGE / '_cann_ops_custom/vendors/vllm-ascend/op_api/lib/libcust_opapi.so',
        torch_npu=Path('/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/lib/libtorch_npu.so'),
        custom_extension=PACKAGE / 'vllm_ascend_C.cpython-311-aarch64-linux-gnu.so')
    if any(base.sha(path) != PINS[key] for key, path in paths.items()):
        raise RuntimeError('FIXED_PRODUCTION_ELF_BYTES_REQUIRED')
    headers = {'include/aclnn/acl_meta.h': 'be93460abfc7aa14113d7b9a3536fe2dd45d5148a357f55e15847655c9d5ffed',
               'include/aclnnop/aclnn_grouped_matmul_weight_nz.h': HEADER_SHA,
               'include/acl/acl_rt.h': RUNTIME_HEADER_SHA}
    if any(base.sha(SDK / name) != value for name, value in headers.items()):
        raise RuntimeError('EXACT_TYPED_GMM2_AND_DEVICE_HEADER_REQUIRED')
    library = out / 'runtime/vendor/op_api/lib/libcust_opapi.so'; library.parent.mkdir(parents=True)
    base.absolute_needed_contract(out)
    originals = {key: image(path) for key, path in paths.items()}
    images = {key: value for key, value in originals.items() if key != 'custom'}
    images['custom_real'] = originals['custom']
    profile = dict(schema='demoncat.quant.container_profile.v1', image=IMAGE, actual_npu_calls=0,
        originals=originals, images=images, sdk_header_sha256=headers['include/aclnn/acl_meta.h'],
        sdk_headers=headers, functions=list(base.FUNCTIONS + PAIR),
        sources={name: base.sha(source / name) for name in SOURCES},
        scope='FOUR_CONTAINER_APIS_AND_ONE_TYPED_WEIGHTNZ_PAIR_NOT_ROW_AUTHORITY',
        gmm2_extension=dict(version=1, typed_pair=list(PAIR), max_calls=128, max_pairs=32, max_bytes=1024*1024),
        original_loading='ABSOLUTE_DT_NEEDED_ORIGINAL_AND_DEVICE_RUNTIME_BYTES_UNCHANGED')
    ident = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    profile['profile_sha256'] = ident
    base.write_json(out / 'profile.json', profile)
    aliases = dict(custom_path='custom', opapi_path='opapi', nn_path='nn', device_path='device_runtime')
    (out / 'quant_container_profile.h').write_text('namespace qct_profile {\nstatic constexpr const char* id="'
        + ident + '";\n' + ''.join('static constexpr const char* ' + key + '="' + str(paths[value].resolve()) + '";\n'
                                    for key, value in aliases.items())
        + ''.join(base.declaration(key, value) for key, value in images.items()) + '}\n')
    native = source / 'fault_injection/native_adapters'
    flags = base.flags(out) + ['-DDCAT_QCT_GMM2_EXTENSION']
    base.run(['g++', *flags, '-fPIC', '-shared', native / 'quant_container_trace.cpp',
        '-L', paths['nn'].parent, '-Wl,--no-as-needed', '-lnnopbase', '-ldl', '-lpthread',
        '-Wl,-z,defs', '-Wl,-soname,libcust_opapi.so', '-o', library])
    for name in ('custom', 'device_runtime'):
        base.run(['patchelf', '--add-needed', str(paths[name].resolve()), library])
    if any(base.sha(path) != PINS[key] for key, path in paths.items()):
        raise RuntimeError('ORIGINAL_IMAGE_BYTES_CHANGED_DURING_BUILD')
    # The original nineteen contracts now compile the same extension TU too.
    base.run(['g++', *flags, native / 'test_quant_container_trace.cpp', '-ldl', '-lpthread', '-o', out / 'cpu_contract'])
    base.cpu_contracts(source, out, source_paths=SOURCES)
    base.run(['g++', *flags, native / 'test_quant_gmm2_trace.cpp', '-ldl', '-lpthread', '-o', out / 'gmm2_cpu_contract'])
    cpu_pair_contracts(source, out)
    symbols = base.run(['nm', '-D', '--defined-only', library], capture_output=True, text=True).stdout
    (out / 'exports.txt').write_text(symbols)
    for symbol in base.FUNCTIONS + base.EXPORTS + PAIR + CONTROLS:
        if not re.search(r'\bT ' + symbol + r'$', symbols, re.M):
            raise RuntimeError('MISSING_EXACT_TYPED_EXPORT:' + symbol)
    observed = set(re.findall(r'\bT (aclnn\w+|CanUsePTACache|dcat_dq\w+)$', symbols, re.M))
    if observed != set(PAIR):
        raise RuntimeError('UNREQUESTED_OPERATOR_OR_FAULT_EXPORT')
    dynamic = base.run(['readelf', '-d', library], capture_output=True, text=True).stdout
    (out / 'dynamic.txt').write_text(dynamic)
    if any('[' + str(paths[name].resolve()) + ']' not in dynamic for name in ('custom', 'device_runtime')):
        raise RuntimeError('STARTUP_ORIGINAL_AND_DEVICE_DEPENDENCIES_REQUIRED')
    marker_preflight(out, library, ident)


def validate_contracts(out):
    for name, cases, prefix, status in (
        ('cpu-contract', base.CASES, 'PASS_CPU_SDK_TYPED_FUNCTION_DOUBLE_NOT_NPU ', 'PASS_ALL_19_SDK_TYPED_CPU_CONTRACTS_NOT_NPU'),
        ('gmm2-cpu-contract', CASES, 'PASS_CPU_SDK_GMM2_TYPED_FUNCTION_DOUBLE_NOT_NPU ', 'PASS_ALL_GMM2_SDK_TYPED_CPU_CONTRACTS_NOT_NPU')):
        value = json.loads((out / (name + '.json')).read_text())
        if (value['status'] != status or value['expected_cases'] != list(cases)
                or [row['case'] for row in value['cases']] != list(cases)
                or (out / (name + '.log')).read_text().splitlines() != [prefix + case for case in cases]
                or any(row['status'] != 'PASS' or row['returncode'] != 0
                       or base.sha(out / row['log']) != row['log_sha256'] for row in value['cases'])):
            raise RuntimeError('COMPLETE_PERSISTENT_TYPED_SHIM_CPU_RECEIPT_REQUIRED:' + name)
    if json.loads((out / 'gmm2-startup.json').read_text())['status'] != 'GMM2_ORIGINALS_RESOLVED_WITHOUT_DEVICE_OR_OPERATOR_CALLS':
        raise RuntimeError('ACTUAL_STARTUP_RESOLUTION_REQUIRED')


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
    sources = {name: base.sha(source / name) for name in SOURCES}; out.mkdir(parents=True)
    nonce = uuid.uuid4().hex; name = 'dcat-quant-gmm2-build-' + nonce
    receipt = dict(schema='demoncat.quant.gmm2_build.v1', status='FAILED_CPU_BUILD', actual_npu_calls=0,
        explicit_driver_api_calls=0, loader_constructor_driver_calls='NOT_COUNTED_MAY_PROBE_DEVICE_INFORMATION',
        source_commit=a.source_commit, sources=sources, image_id=IMAGE, container_name=name, container_owner_nonce=nonce)
    launched = False
    try:
        image = base.inspect_image(IMAGE)
        base.write_json(out / 'image-inspect.json', image)
        base.write_json(out / 'driver-identity.json', base.driver_identity())
        receipt['driver_identity_sha256'] = base.sha(out / 'driver-identity.json')
        receipt['loader_path'] = base.loader_path(image)
        command = ['docker', 'run', '--rm', '--name', name, '--label', 'demoncat.quant.group-build-owner=' + nonce,
            '--network', 'none', '--runtime', 'runc', '--cpus', '4', '--memory', '4g', '-e', 'ASCEND_VISIBLE_DEVICES=void',
            '-e', 'PYTHONPATH=/source', '-e', 'LD_LIBRARY_PATH=' + receipt['loader_path'],
            '-v', str(base.DRIVER) + ':' + str(base.DRIVER) + ':ro', '-v', str(source) + ':/source:ro',
            '-v', str(out) + ':/build', '--entrypoint', 'python3', image['Id'],
            '/source/experiments/worker_core_v11_20261118/build_quant_gmm2_trace.py',
            '--source', '/source', '--out', '/build', '--source-commit', a.source_commit, '--inside']
        receipt['command'] = command; receipt['image_inspect_sha256'] = base.sha(out / 'image-inspect.json')
        with (out / 'build.log').open('wb') as stream:
            launched = True
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
        receipt['returncode'] = result.returncode
        if result.returncode or any(base.sha(source / key) != value for key, value in sources.items()):
            raise RuntimeError('SDK_COMPILE_CONTRACT_OR_FROZEN_SOURCE_FAILED')
        validate_contracts(out)
        receipt['status'] = 'SDK_AND_GMM2_TYPED_CPU_CONTRACT_PASS_NOT_NPU'
    except BaseException as error:
        receipt.update(error=repr(error), timed_out=isinstance(error, subprocess.TimeoutExpired))
    finally:
        try:
            receipt['cleanup'] = base.cleanup_owned(name, nonce, IMAGE) if launched else dict(confirmed=True, status='NOT_LAUNCHED')
        except BaseException as error:
            receipt['cleanup'] = dict(confirmed=False, error=repr(error))
        receipt['stage_receipts'] = {}
        for key in ('cpu-contract.json', 'gmm2-cpu-contract.json', 'sdk-load.json', 'gmm2-startup.json'):
            if (out / key).exists():
                try:
                    receipt['stage_receipts'][key] = dict(status=json.loads((out / key).read_text())['status'], sha256=base.sha(out / key))
                except BaseException as error:
                    receipt['stage_receipts'][key] = dict(error=repr(error), sha256=base.sha(out / key))
        receipt['artifacts'] = {path.name: base.sha(path) for path in out.iterdir() if path.is_file()}
        base.write_json(out / 'build-receipt.json', receipt)
    if receipt['status'] != 'SDK_AND_GMM2_TYPED_CPU_CONTRACT_PASS_NOT_NPU' or not receipt['cleanup']['confirmed']:
        raise RuntimeError('CPU_BUILD_FAILED_RECEIPT_PRESERVED')
    profile = json.loads((out / 'profile.json').read_text())
    files = {str(path.relative_to(out)): base.sha(path) for path in (out / 'runtime').rglob('*') if path.is_file()}
    files['profile.json'] = base.sha(out / 'profile.json')
    manifest = dict(schema='demoncat.quant.container_deployment.v1', actual_npu_calls=0,
        source_commit=a.source_commit, image_id=IMAGE, sources=sources, files=files,
        library='runtime/vendor/op_api/lib/libcust_opapi.so', profile_sha256=profile['profile_sha256'],
        gmm2_extension=profile['gmm2_extension'], build_receipt_sha256=base.sha(out / 'build-receipt.json'),
        qualified=False, c1=False, c2=False, c3=False)
    base.write_json(out / 'build.json', manifest)
    print(json.dumps(dict(status=receipt['status'], build_sha256=base.sha(out / 'build.json'), out=str(out))))


if __name__ == '__main__':
    main()
