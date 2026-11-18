"""Bounded read-only identities shared by the normal-only preparer/reviewer.

No framework, operator, shared library, process or device is loaded. Only the
pure ABI decoding functions are selected from the explicitly frozen observer
verifier; the injection adapter is never imported as an authority.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import sys
from types import SimpleNamespace

EXPERIMENT = 'experiments/c2c3_closure_20261117/'
WORKER = 'src/demoncat/runtime/vllm_ascend/worker_extension/'
NATIVE = 'fault_injection/native_adapters/attention28/'
VARIANT = 'tnd_page16_arch22_normal_bridge'
VENDOR = 'dcat_f33_arch22_normal_transformer'
S2_VARIANT = 'tnd_page16_arch22_s2_256'
S2_VENDOR = 'dcat_f33_arch22_s2_transformer'
IMAGE = 'sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0'
COMMIT = 'c77c0829f1f0f0b8f882db1d9b051a758b605472'
KEY = 103000000010300003
KEYS = {103000000010200003, KEY, 104000000010200003, 104000000010300003,
        5000000000000200100, 5000000000000200103, 5000000000010200100,
        5000000000010200103, 5100000000010200100, 5100000000010200103}
CANONICAL_TOKEN_SOURCE_SHA256 = {
    'attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_info_parser.cpp':
        '380248471fef863d9c7ae5f77f00aafb71185330d350388652c6f01557a9f271',
    'attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_info_parser.h':
        '54753b541981eb176636bc0243c728830f5076a39d4e69721816d0d4eb6d44f0',
    'attention/common/op_host/fia_tiling_info.h':
        'd201e4ed6ca93fb90a7a60c9da5fbfbb5727810768ffd6cd74e8c67115febda4',
}
REQUIRED = tuple(EXPERIMENT+name+'.py' for name in (
    'arch22_normal_bridge_evidence', 'prepare_arch22_normal_bridge_cases',
    'review_arch22_normal_bridge', 'attention33_worker_probe', 'attention33_contract',
    'attention33_bridge', 'review_attention33_first_boundary_source53',
    'diagnose_attention33_baselines', 'diagnose_attention33_native_routes')) + tuple(
    WORKER+name+'.py' for name in ('paged33_arch22_bridge', 'paged33_tnd', 'paged33_tnd_probe',
        'paged33_boundary_probe', 'paged33_audit', 'paged33_loading', 'attention_native',
        'serving_context', 'call_binding')) + (
    NATIVE+'launch.py', NATIVE+'launch_observer.cpp', NATIVE+'fia/build_contract.json',
    NATIVE+'fia/apply_tnd33_arch22_bridge.py', NATIVE+'fia/paged33_arch22_bridge.h',
    NATIVE+'fia/compile_public_schema.py', NATIVE+'fia/fia_abi_decoder.cpp')
S2_REQUIRED = (EXPERIMENT+'prepare_arch22_s2_normal_bridge_cases.py',EXPERIMENT+'build_tnd33_124.py',
    WORKER+'paged33_arch22_s2_normal.py',WORKER+'paged33_logical_binding.py') + tuple(NATIVE+'fia/'+name for name in (
    'apply_tnd33_arch22_s2.py','paged33_arch22_s2.h','arch22_s2_geometry.py','compile_arch22_s2_matrix.py',
    'arch22_s2_build_evidence.py','build_fresh.py','binary_registration.py','base_policy.py'))


def variant(source):return source.get('variant',VARIANT)
def vendor_name(source):return S2_VENDOR if variant(source)==S2_VARIANT else VENDOR


def need(ok, reason):
    if not ok: raise ValueError(reason)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024**2), b''): value.update(chunk)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'DUPLICATE_JSON_KEY:'+key); result[key] = value
    return result


def read(path, limit=128*1024**2):
    path = Path(path); before = path.stat()
    need(0 < before.st_size <= limit and not path.is_symlink(), 'JSON_FILE_BOUND_OR_LINK')
    raw = path.read_bytes(); after = path.stat()
    need((before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns), 'JSON_CHANGED_DURING_REVIEW')
    def invalid(value): raise ValueError('NONFINITE_JSON:'+value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def child(root, relative):
    name = PurePosixPath(relative)
    need(type(relative) is str and not name.is_absolute() and '..' not in name.parts
         and name.parts and '\\' not in relative, 'RELATIVE_ARTIFACT_PATH_INVALID')
    root = Path(root).resolve(strict=True); path = root/Path(*name.parts)
    need(path.resolve(strict=True).is_relative_to(root), 'ARTIFACT_ESCAPES_ROOT')
    return path


def frozen_source(root, expected_sha, *, build_variant=VARIANT):
    need(build_variant in (VARIANT,S2_VARIANT),'NORMAL_BUILD_VARIANT_REQUIRED')
    root = Path(root).resolve(strict=True); manifest = root/'source_manifest.json'
    need(re.fullmatch('[0-9a-f]{64}', expected_sha or '') and sha(manifest) == expected_sha,
         'FROZEN_SOURCE_MANIFEST_SHA_DIFFERS')
    data = read(manifest); rows = data.get('files', [])
    need(data.get('schema') == 'demoncat.c2c3.source.v1'
         and len({row['path'] for row in rows}) == len(rows), 'FROZEN_SOURCE_MANIFEST_INVALID')
    files = {row['path']: row for row in rows}; pins = {}
    for name in REQUIRED+(S2_REQUIRED if build_variant==S2_VARIANT else ()):
        path = child(root, name)
        need(name in files and not path.is_symlink() and sha(path) == files[name]['sha256'],
             'FROZEN_REQUIRED_SOURCE_DIFFERS:'+name)
        pins[name] = files[name]['sha256']
    need(sha(Path(__file__)) == pins[EXPERIMENT+Path(__file__).name], 'REVIEW_COMMON_SOURCE_DIFFERS')
    return dict(root=root, pins=pins, manifest=data, manifest_sha256=expected_sha,variant=build_variant)


def helpers(source):
    def module(name):
        path = source['root']/EXPERIMENT/(name+'.py')
        if name in sys.modules:
            need(Path(sys.modules[name].__file__).resolve() == path, 'AMBIENT_REVIEW_HELPER_REFUSED:'+name)
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec); sys.modules[name] = value; spec.loader.exec_module(value)
        return value
    contract = module('attention33_contract')
    result = SimpleNamespace(contract=contract, bridge=module('attention33_bridge'),
        boundary=module('review_attention33_first_boundary_source53'),
        differences=module('diagnose_attention33_baselines'), elf=module('diagnose_attention33_native_routes'),
        preparer=module('prepare_arch22_normal_bridge_cases'))
    # Reuse only pure source-derived public decoder functions. In particular
    # do not execute launch.py's ctypes import or LaunchObserver constructor.
    path = source['root']/NATIVE/'launch.py'; tree = ast.parse(path.read_bytes(), filename=str(path))
    names = {'decode_raw', 'verify_public_values', 'derive_fia_kernel_abi', 'bundle_kernel_abi'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    need({node.name for node in nodes} == names, 'FROZEN_PURE_ABI_FUNCTION_SET_DIFFERS')
    namespace = dict(require=need, sha=sha, Path=Path, re=re, struct=struct, json=json)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    result.abi = SimpleNamespace(**{name:namespace[name] for name in names})
    if variant(source)==S2_VARIANT:
        for name in ('arch22_s2_geometry','arch22_s2_build_evidence'):
            path=source['root']/NATIVE/'fia'/(name+'.py')
            if name in sys.modules:
                need(Path(sys.modules[name].__file__).resolve()==path,'AMBIENT_S2_HELPER_REFUSED')
                loaded=sys.modules[name]
            else:
                spec=importlib.util.spec_from_file_location(name,path)
                loaded=importlib.util.module_from_spec(spec);sys.modules[name]=loaded;spec.loader.exec_module(loaded)
            setattr(result,name,loaded)
    return result


def installed_sources(source, installed):
    pins = {}
    for name, expected in source['pins'].items():
        if name.startswith('src/demoncat/'):
            relative = name.removeprefix('src/')
        elif name.startswith('fault_injection/'):
            relative = name
        else: continue
        path = child(installed, relative)
        need(not path.is_symlink() and sha(path) == expected, 'INSTALLED_CAPTURE_SOURCE_DIFFERS:'+relative)
        pins[relative] = expected
    return pins


def build_bundle(root, source, helper, expected_completion=None):
    root = Path(root).resolve(strict=True); completion = root/'evidence/completion.json'
    completion_sha = sha(completion)
    need(expected_completion is None or completion_sha == expected_completion, 'COMPLETION_SHA_DIFFERS')
    data = read(completion); vendor = root/'stage/vendors'/vendor_name(source)
    selected_variant=variant(source);s2=selected_variant==S2_VARIANT
    need(data.get('schema') == 'demoncat.fia.fresh_build.v1' and data.get('source_commit') == COMMIT
        and data.get('variant') == selected_variant and data.get('status') == 'CPU_BUILD_ONLY_REQUIRES_NATIVE_QUALIFICATION'
        and all(data.get(k) is False for k in ('loading_allowed', 'npu_executed',
            'previous_experiment_objects_used', 'fault_execution_ready', 'stock_private_exact_proven'))
        and len(data.get('compiled_keys', [])) == len(KEYS) and set(data['compiled_keys']) == KEYS,
        'SEPARATE_NORMAL_ONLY_FRESH_BUILD_REQUIRED')
    need(data.get('contract_sha256') == source['pins'][NATIVE+'fia/build_contract.json'], 'FRESH_BUILD_SOURCE_BINDING_DIFFERS')
    matrix=None
    if s2:matrix=helper.arch22_s2_build_evidence.verify_inputs(root,data,source['pins'])
    else:
        need(data.get('normal_bridge_recipe_sha256') == source['pins'][NATIVE+'fia/apply_tnd33_arch22_bridge.py']
            and data.get('normal_bridge_header_sha256') == source['pins'][NATIVE+'fia/paged33_arch22_bridge.h'],
            'FRESH_BUILD_SOURCE_BINDING_DIFFERS')
    for relative, expected in CANONICAL_TOKEN_SOURCE_SHA256.items():
        path=child(root/'source',relative)
        need(not path.is_symlink() and path.resolve(strict=True) == path
             and path.stat().st_size <= 256*1024 and sha(path) == expected,
             'FRESH_NORMAL_CAUSAL_TOKEN_PARSER_DIFFERS:'+relative)
    patch_path=root.parents[1]/'patch/policy_source_manifest.json'; patch=read(patch_path)
    need(sha(patch_path) == data['patch_receipt_sha256'] and patch['source_commit'] == COMMIT
        and patch['variant'] == selected_variant and patch['recipe_sha256'] == data['arch22_s2_recipe_sha256' if s2 else 'normal_bridge_recipe_sha256']
        and patch['normal_route_preference_changed'] is True and all(patch[key] is False for key in
            ('tensor_payload_mutated','kernel_source_modified','fault_execution_ready','stock_private_exact_proven'))
        and (patch.get('selection_before_original_split') is True and patch.get('m_policy_changed') is False
             and patch.get('old_normal_certificate_reused') is False and patch.get('loading_allowed') is False
             if s2 else patch.get('tiling_arithmetic_modified') is False),
        'ACTUAL_NORMAL_PATCH_RECEIPT_DIFFERS')
    need(0 < len(patch['files']) <= 32 and len({row['path'] for row in patch['files']}) == len(patch['files']),
         'NORMAL_PATCH_SOURCE_CENSUS_DIFFERS')
    for row in patch['files']:
        path=child(root/'source',row['path'])
        need(not path.is_symlink() and path.stat().st_size == row['bytes'] <= 8*1024**2
            and sha(path) == row['after_sha256'], 'BUILT_NORMAL_PATCH_SOURCE_DIFFERS:'+row['path'])
    rows = data.get('artifacts', [])
    need(0 < len(rows) <= 8192 and len({r['path'] for r in rows}) == len(rows), 'ARTIFACT_CENSUS_INVALID')
    files = {}; total = 0
    for row in rows:
        path = child(vendor, row['path'])
        if path.is_symlink():
            need(row['path'] == 'op_impl/ai_core/tbe/op_tiling/liboptiling.so'
                and os.readlink(path) == 'lib/linux/aarch64/libcust_opmaster_rt2.0.so', 'UNREVIEWED_ARTIFACT_LINK')
        need(0 <= path.stat().st_size == row['bytes'] <= 512*1024**2 and sha(path) == row['sha256'],
             'FRESH_ARTIFACT_BYTES_DIFFER:'+row['path'])
        files[row['path']] = row; total += row['bytes']
    need(total <= 2*1024**3, 'FRESH_ARTIFACT_TOTAL_BOUND')
    actual = set()
    for path in vendor.rglob('*'):
        need(not(path.is_symlink() and path.is_dir()), 'ARTIFACT_DIRECTORY_LINK')
        if path.is_file(): actual.add(path.relative_to(vendor).as_posix())
    need(actual == set(files), 'FRESH_VENDOR_MISSING_OR_EXTRA_FILES')
    kernels = [row for name,row in files.items() if name.endswith('.o')]
    need(len(kernels) == 1 and kernels[0]['sha256'] == data['kernel_binary_sha256'], 'EXACT_ONE_FRESH_KERNEL_REQUIRED')
    metadata, abi = helper.abi.bundle_kernel_abi(vendor, files, data['kernel_binary_sha256'])
    need(metadata.get('binFileName') == 'FusedInferAttentionScore_8cd36e66e4ceb2d60ef96db29a147347'
        and metadata.get('sha256') == data['kernel_binary_sha256']
        and len(metadata.get('kernelList', [])) == len(KEYS)
        and {int(row['tilingKey']) for row in metadata['kernelList']} == KEYS
        and len(metadata.get('supportInfo', {}).get('tilingKey', [])) == len(KEYS)
        and {int(k) for k in metadata['supportInfo']['tilingKey']} == KEYS, 'FRESH_COMPILED_ENTRY_SET_DIFFERS')
    schema_path = root/'evidence/arch22_schema/fia_abi_schema.json'
    compile_path = schema_path.with_name('compile_receipt.json')
    need(sha(schema_path) == data['arch22_public_schema_sha256']
        and sha(compile_path) == data['arch22_schema_compile_sha256'], 'FRESH_PUBLIC_SCHEMA_BYTES_DIFFER')
    schema, compiled = read(schema_path), read(compile_path)
    need(schema.get('schema') == 'fault28.fia_graph.public_serializer_layout.v1'
        and schema.get('type') == 'FusedInferAttentionScoreTilingData' and schema.get('size') == 1576
        and compiled.get('schema') == 'demoncat.fia.fresh_abi_compile.v1'
        and compiled.get('schema_sha256') == data['arch22_public_schema_sha256']
        and compiled.get('npu_executed') is False and compiled.get('loading_allowed') is False,
        'ARCH22_FRESH_PUBLIC_SERIALIZER_REQUIRED')
    projection=schema_path.with_name('fia_schema_projection.h'); decoder=schema_path.with_name('fia_abi_decoder')
    public=child(root/'source','attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.h')
    need(compiled['compiler_source_sha256'] == source['pins'][NATIVE+'fia/fia_abi_decoder.cpp']
        and sha(projection) == compiled['projection_sha256'] == schema['projection_sha256']
        and sha(public) == compiled['public_source_sha256'] == schema['public_source_sha256']
        and sha(decoder) == compiled['decoder_sha256']
        and str(compiled['actual_new_fia_compile_entry']['file']).endswith('/arch22/fia_tiling_nonquant.cpp'),
        'ACTUAL_SCHEMA_COMPILER_HEADER_OR_DECODER_DIFFERS')
    sentinel=read(schema_path.with_name('fia_abi_selftest.json'))
    need(sentinel['status'] == 'PASS_CPU_DERIVED_LAYOUT_SENTINEL' and sentinel['serialized_bytes'] == 1576
        and sentinel['stock_host_abi_verified'] is False and sentinel['npu_devices_required'] is False
        and sentinel['invalid_lengths_rejected'] == 2
        and sentinel['fields'] == {row['name']:{k:v for k,v in row.items() if k != 'name'} for row in schema['fields']},
        'COMPLETE_PUBLIC_SCHEMA_SENTINEL_FIELDS_DIFFER')
    # Reject duplicate/overlapping/out-of-bounds fields even before any runtime.
    helper.abi.decode_raw(bytes(schema['size']), schema)
    controller_path = root.parents[1]/'controller.json'; controller = read(controller_path)
    need(controller.get('status') == 'BUILD_COMMAND_COMPLETED' and controller.get('exit_code') == 0
        and controller.get('actual_npu_calls') == 0 and controller.get('devices') == []
        and controller.get('source_commit') == COMMIT
        and re.fullmatch('[0-9a-f]{64}', data.get('sdk_interface_manifest_sha256', ''))
        and controller.get('sdk_interfaces_sha256') == data['sdk_interface_manifest_sha256']
        and controller.get('image_id') == IMAGE and controller.get('strategy') == ('arch22-s2' if s2 else 'arch22-normal-bridge')
        and controller.get('owned_container_removed') is True
        and controller.get('final_state', {}).get('Running') is False, 'EXACT_DEVICE_FREE_BUILD_CONTROLLER_REQUIRED')
    return dict(root=root, vendor=vendor, completion=data, completion_sha256=completion_sha,
        artifacts=files, metadata=metadata, abi=abi, schema=schema, controller=controller,s2_cpu_matrix=matrix,
        summary=dict(completion_path=str(completion), completion_sha256=completion_sha,
            controller_sha256=sha(controller_path), schema_sha256=sha(schema_path),
            compile_receipt_sha256=sha(compile_path), schema_sentinel_sha256=sha(schema_path.with_name('fia_abi_selftest.json')),
            patch_receipt_sha256=sha(patch_path), artifacts=len(files), artifact_bytes=total,
            kernel_sha256=data['kernel_binary_sha256'], variant=selected_variant, compiled_keys=sorted(KEYS),
            s2_cpu_matrix=None if matrix is None else {key:matrix[key] for key in
                ('receipt_sha256','matrix_sha256','cases','qualification_granted')},
            canonical_token_source_sha256=dict(CANONICAL_TOKEN_SOURCE_SHA256)))


def observer_bundle(root, source, expected_sha=None):
    folder = Path(root).resolve(strict=True)/'attention-preflight'; path = folder/'result.json'
    need(expected_sha is None or sha(path) == expected_sha, 'OBSERVER_RECEIPT_SHA_DIFFERS')
    receipt = read(path)
    need(receipt.get('observer_compiled') is True and receipt.get('actual_npu_calls') == 0
        and receipt.get('observer_memory_selftest', {}).get('exit_code') == 0
        and receipt.get('observer_source', {}).get('sha256') == source['pins'][NATIVE+'launch_observer.cpp'],
        'EXACT_FRESH_OBSERVER_REQUIRED')
    recorded = PurePosixPath(receipt['observer_library']['path'])
    need(recorded.is_absolute() and '..' not in recorded.parts and recorded.name,
         'OBSERVER_ORIGINAL_LIBRARY_PATH_INVALID')
    library = child(folder, recorded.name)
    need(sha(library) == receipt['observer_library']['sha256'], 'OBSERVER_LIBRARY_BYTES_DIFFER')
    alias = str(recorded.parent)
    need(alias == '/build/attention-preflight' or alias.startswith('/output/'), 'OBSERVER_READONLY_ALIAS_UNREVIEWED')
    return dict(receipt=receipt, folder=folder, alias=alias, sha256=sha(path), library=library)


def prohibit_runtime(event, args):
    if event in ('ctypes.dlopen', 'subprocess.Popen', 'os.system', 'socket.connect'):
        raise RuntimeError('READ_ONLY_CPU_ACTION_REFUSED:'+event)
    if event == 'import' and args[0].split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend'):
        raise RuntimeError('DEVICE_FRAMEWORK_IMPORT_REFUSED:'+args[0])
