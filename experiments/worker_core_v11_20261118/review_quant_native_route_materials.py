"""Recompute the narrow Native route blocker from fixed, device-free originals.

This checks disk/source evidence, never imports a framework, loads a DSO, reads
an activation, or issues Native/row/C2/C3 authority. A successful exit means the
technical blocker was reproduced, not that a proxy or routed row is qualified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


IMAGE = 'sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0'
MATERIALS = {
    'sdk_operators': '9d7495ed47e1906efdbe8d332cb10899d1877e797165573f78cec002d90d75cd',
    'native_disk': 'e5f43d690210cc81c2ef46899ef8156aff01314b758e5dc4241f64536f20dae3',
    'loading': 'd21218ba517132c91eeb45a9043105e03f99d9311a5ba6d6a9ce1e4e4fe83054',
    'getters': 'c8b8fb83c35b78f6920ebe86973a773bbc37b615bc9e3ae59f5da86c1c560b77',
}
ELFS = {
    'libopapi.so': 'b3890625d3e0cc11ef32460083698ecaa8f702b5edd0169ea617e2e6bb76270c',
    'libcust_opapi.so': 'de1633c2f1f41e505277697ee82f3ffba940f730bed9986c5aeff4c8514b713f',
    'vllm_ascend_C.cpython-311-aarch64-linux-gnu.so': 'e63806486e0311c2df8ce9b120ae98f55e8c8eb91ccaf204ef0cf7d6b469ed67',
    'libtorch_npu.so': 'dd07e4afb4e31a5123d0b97a4a319acaa96158705a22d704bfc0a91f662a6041',
    '_C.cpython-311-aarch64-linux-gnu.so': 'cc905f411e6f11ad4717cf269d591418207cf5be26b605b35c9677123af7257d',
}
HEADERS = {
    'aclnn_grouped_matmul_swiglu_quant.h': '8a88234b6da476725e448055053544470327c8f29d783dce620f768a705ed643',
    'aclnn_grouped_matmul_v4.h': '78c28a77f90900ff1acd823cec93900abaa28d4bdc0edb9e49218d2951fba85e',
    'aclnn_grouped_matmul_v5.h': '577c8692b5bf627601a2191a01a3665792a38269c1fc14199137f0495e983080',
    'aclnn_moe_token_unpermute.h': 'c9576934b72199fd944d599afbcc263c5c4e2a35a7028419cb85c3888489bc6f',
}
META_SHA = 'be93460abfc7aa14113d7b9a3536fe2dd45d5148a357f55e15847655c9d5ffed'
CUSTOM_RESOLVER = '142023f0f0e682b3b79aeb987044ecbd9b7db0b63b20de967447ba6f9e3d6b16'
CREATORS = ('aclCreateTensorList', 'aclCreateIntArray')
DESTROYERS = ('aclDestroyTensorList', 'aclDestroyIntArray')
SIZES = ('aclGetTensorListSize', 'aclGetIntArraySize')
STOCK_CANDIDATES = ('aclnnGroupedMatmulSwigluQuant', 'aclnnGroupedMatmulV4',
                    'aclnnGroupedMatmulV5', 'aclnnMoeTokenUnpermute')


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_material(path, kind):
    raw = Path(path).read_bytes()
    require(sha(raw) == MATERIALS[kind], 'FIXED_MATERIAL_BYTES_CHANGED:' + kind)
    value = json.loads(raw)
    validate_origin(value, kind)
    return value


def validate_origin(value, kind):
    require(value.get('actual_npu_calls') == 0 and type(value.get('actual_npu_calls')) is int
            and value.get('framework_imported') is False, 'DEVICE_FREE_ORIGIN_REQUIRED:' + kind)
    require(value.get('image', value.get('production_image')) == IMAGE, 'FIXED_IMAGE_REQUIRED:' + kind)
    if kind != 'loading':
        require(value.get('loaded_owner_proven') is False, 'DISK_IS_NOT_LOADED_OWNER:' + kind)
    if kind in ('sdk_operators', 'native_disk'):
        require(value.get('executed_api_proven') is False, 'DISK_IS_NOT_EXECUTED_API:' + kind)
    rows = value.get('sources', value.get('files'))
    require(isinstance(rows, list) and rows, 'SOURCE_ORIGINALS_REQUIRED:' + kind)
    seen = set()
    for row in rows:
        require(isinstance(row.get('path'), str) and row['path'].startswith('/')
                and row['path'] not in seen, 'UNIQUE_SOURCE_PATH_REQUIRED:' + kind)
        seen.add(row['path'])
        require(isinstance(row.get('text'), str)
                and sha(row['text'].encode()) == row.get('sha256'), 'SOURCE_BYTES_CHANGED:' + row['path'])


def source(value, suffix, expected):
    rows = [row for row in value.get('sources', value.get('files', [])) if row['path'].endswith(suffix)]
    require(rows and all(row['sha256'] == expected for row in rows), 'REVIEWED_SOURCE_REQUIRED:' + suffix)
    require(len({row['text'] for row in rows}) == 1, 'SOURCE_ALIAS_BYTES_DIFFER:' + suffix)
    return rows[0]


def declarations(text):
    # Only declarations in the checked public header; no private struct casts.
    code = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
    return {name: ' '.join(args.split()) for name, args in re.findall(
        r'\b(acl(?:Create|Destroy|Get)\w+)\s*\(([^;{}]*)\)\s*;', code)}


def container_contract(text):
    for name in ('aclTensorList', 'aclIntArray'):
        require(re.search(r'\btypedef\s+struct\s+' + name + r'\s+' + name + r'\s*;', text),
                'OPAQUE_CONTAINER_DECLARATION_REQUIRED:' + name)
        require(not re.search(r'\b(?:struct|class)\s+' + name + r'\s*\{', text),
                'CONTAINER_LAYOUT_IS_NOT_THE_REVIEWED_PUBLIC_ABI:' + name)
    functions = declarations(text)
    for name in CREATORS + DESTROYERS + SIZES:
        require(name in functions, 'PUBLIC_CONTAINER_FUNCTION_MISSING:' + name)
    require(functions['aclCreateTensorList'] == 'const aclTensor *const *value, uint64_t size'
            and functions['aclCreateIntArray'] == 'const int64_t *value, uint64_t size',
            'CONTAINER_CREATOR_SIGNATURE_CHANGED')
    require(functions['aclDestroyTensorList'] == 'const aclTensorList *array'
            and functions['aclDestroyIntArray'] == 'const aclIntArray *array',
            'CONTAINER_DESTROY_SIGNATURE_CHANGED')
    require(functions['aclGetTensorListSize'] == 'const aclTensorList *tensorList, uint64_t *size'
            and functions['aclGetIntArraySize'] == 'const aclIntArray *array, uint64_t *size',
            'CONTAINER_SIZE_SIGNATURE_CHANGED')
    getters = sorted(name for name, args in functions.items() if name.startswith('aclGet')
                     and re.search(r'\bacl(?:TensorList|IntArray)\b', args))
    require(getters == sorted(SIZES), 'CONTAINER_GETTER_SET_CHANGED_REQUIRES_NEW_REVIEW')
    return dict(opaque_types=['aclTensorList', 'aclIntArray'], getter_names=getters,
                readable_member_api_in_reviewed_header=False,
                creator_destroy_declarations={name: functions[name] for name in CREATORS + DESTROYERS},
                private_layout_read_authorized=False, python_argument_backfill_authorized=False)


def elf_fact(row, expected, required=()):
    require(row.get('sha256') == expected and type(row.get('bytes')) is int and row['bytes'] > 0,
            'EXACT_ELF_DISK_IDENTITY_REQUIRED:' + row.get('path', '?'))
    require(row.get('symbol_scan_returncode') == 0, 'ELF_SYMBOL_SCAN_FAILED')
    metadata = row.get('metadata', {})
    require(metadata.get('dynamic_returncode') == 0 and metadata.get('notes_returncode') == 0,
            'ELF_METADATA_SCAN_FAILED')
    exports = set()
    for line in row.get('symbols', []):
        fields = line.split()
        if len(fields) >= 8 and fields[3] == 'FUNC' and fields[6] != 'UND':
            exports.add(fields[7])
    require(set(required) <= exports, 'ACTUAL_EXACT_UNVERSIONED_EXPORT_MISSING')
    return dict(path=row['path'], resolved_path=row['resolved_path'], sha256=expected, bytes=row['bytes'],
                needed=re.findall(r'\(NEEDED\).*?\[(.*?)\]', metadata.get('dynamic', '')),
                runpath=re.findall(r'\((?:RUNPATH|RPATH)\).*?\[(.*?)\]', metadata.get('dynamic', '')),
                build_ids=re.findall(r'Build ID:\s*([a-f0-9]+)', metadata.get('notes', '')),
                checked_defined_symbols=list(required), loaded_owner_proven=False, executed_api_proven=False)


def review(materials):
    require(set(materials) == set(MATERIALS), 'COMPLETE_FOUR_MATERIAL_SET_REQUIRED')
    for kind, value in materials.items():
        validate_origin(value, kind)
    native, sdk, loading, getters = (materials[k] for k in ('native_disk', 'sdk_operators', 'loading', 'getters'))
    require({Path(row['path']).name for row in native['elfs']} == set(ELFS)
            and len(native['elfs']) == len(ELFS), 'COMPLETE_FIVE_ELF_DISK_INVENTORY_REQUIRED')
    facts = {}
    for name, expected in ELFS.items():
        row = next(row for row in native['elfs'] if Path(row['path']).name == name)
        apis = STOCK_CANDIDATES if name == 'libopapi.so' else (
            ('aclnnMoeInitRoutingCustom',) if name == 'libcust_opapi.so' else ())
        required = tuple(part for api in apis for part in (api, api + 'GetWorkspaceSize'))
        facts[name] = elf_fact(row, expected, required)
    require('libopapi.so' in facts['libcust_opapi.so']['needed'], 'CUSTOM_TO_STOCK_DEPENDENCY_CHANGED')
    require('libopapi.so' in facts['vllm_ascend_C.cpython-311-aarch64-linux-gnu.so']['needed'],
            'EXTENSION_TO_STOCK_DEPENDENCY_CHANGED')
    header = source(getters, '/aclnn/acl_meta.h', META_SHA)
    contract = container_contract(header['text'])
    require({row['include'] for row in getters['unresolved_direct_includes']} <= {'cstdint', 'cstdlib'},
            'UNREVIEWED_MISSING_VENDOR_HEADER_INCLUDE')
    operator_headers = {name: dict(path=source(sdk, '/' + name, expected)['path'], sha256=expected)
                        for name, expected in HEADERS.items()}
    resolver = source(native, '/aclnn_torch_adapter/op_api_common.h', CUSTOM_RESOLVER)
    code = resolver['text']
    for fragment in ('dlopen(libName, RTLD_LAZY)', 'GET_OP_API_FUNC(aclCreateTensorList)',
                     'GET_OP_API_FUNC(aclCreateIntArray)', 'GET_OP_API_FUNC(aclDestroyTensorList)',
                     'GET_OP_API_FUNC(aclDestroyIntArray)', 'aclCreateTensorList(tensor_list.data(), tensor_list.size())',
                     'aclCreateIntArray(at_array.data(), at_array.size())'):
        require(fragment in code, 'ACTUAL_CUSTOM_RESOLVER_CLOSURE_CHANGED:' + fragment)
    source(loading, '/vllm_ascend/utils.py', '45b842cb4ffd5ba9ca055757847615525884c12eebdfe3da53aaef53381570b9')
    require(not any('/op_plugin/' in row['path'] and row['path'].endswith('op_api_common.h')
                    for row in native['sources']), 'STOCK_RESOLVER_SOURCE_NOW_PRESENT_REQUIRES_REVIEW')
    return dict(schema='demoncat.quant.native_route_material_review.v1', status='BLOCKED_OPAQUE_CONTAINER_MEMBER_IDENTITY',
                material_sha256=dict(MATERIALS), production_image=IMAGE,
                scope='FIXED_PRODUCTION_DISK_AND_PUBLIC_SDK_SOURCE_ONLY',
                sdk_header=dict(path=header['path'], sha256=META_SHA, **contract),
                operator_headers=operator_headers, disk_images=facts,
                custom_resolver=dict(path=resolver['path'], sha256=CUSTOM_RESOLVER,
                    custom_handle_before_stock=True, function_addresses_cached=True,
                    creator_arguments_observable_during_actual_creator_only=True,
                    gws_and_execute_may_have_different_threads=True,
                    stock_torch_npu_implementation_source_proven=False),
                stock_resolver=dict(library_sha256=ELFS['libtorch_npu.so'], exact_upstream_commit=None,
                    existing_inspected_profile='fault_injection/deployments/pta_resolver.py',
                    creator_destroy_actual_resolution_and_calls_proven=False),
                blocking_edges=['GMM2_TENSOR_LIST_MEMBERS_HAVE_NO_REVIEWED_GWS_GETTER',
                    'ROUTING_ACTIVE_RANGE_INT_ARRAY_VALUES_HAVE_NO_REVIEWED_GWS_GETTER',
                    'STOCK_CREATOR_DESTROY_RESOLUTION_AND_CONTAINER_LIFETIMES_UNOBSERVED',
                    'ACTUAL_NATIVE_API_VARIANTS_AND_LOADED_OWNERS_UNOBSERVED'],
                next_hypothesis='BOUNDED_ACTUAL_STOCK_CREATOR_DESTROY_ENTRY_DIAGNOSTIC_NOT_IMPLEMENTED',
                proposed_four_stage_proxy_implemented=False, new_actual_npu_calls=0, new_fault_calls=0,
                current_previous_request_row_scope=dict(proven_per_forward=10, candidates_per_forward=135,
                    unknown_per_forward=125, source='PREVIOUS_SOURCE93_REVIEW_NOT_REMEASURED_HERE'),
                native_qualified=False, new_request_row_proof=False, c1=False, c2=False, c3=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for kind in MATERIALS:
        parser.add_argument('--' + kind.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = review({kind: load_material(getattr(args, kind), kind) for kind in MATERIALS})
    raw = (json.dumps(result, sort_keys=True, indent=2) + '\n').encode()
    with args.out.open('xb') as stream:
        stream.write(raw)
    print(json.dumps(dict(status=result['status'], summary_sha256=sha(raw), native_qualified=False,
                         new_request_row_proof=False, actual_npu_calls=0)))


if __name__ == '__main__':
    main()
