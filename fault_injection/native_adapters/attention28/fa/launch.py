"""Independent fresh FA argument ABI and public POD validation.

No historical certificate is loaded. Actual registered ELF/function bindings
select a launch before any host log is inspected. FA derives its own 25-pointer
entry and 1464-byte public POD, not FIA's entry or serializer.
"""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess

from ..launch import observe_launch, require, sha
from .binary_registration import BINARY, KEY, validate_metadata

SOURCE_SHA = '7f4adb241e3f09ad3204159bcd1b963800e174230f80c78b8126e29b7481ba64'
OPBASE = {
    'src/nnopbase/individual_op/executor/indv_executor.cpp': 'd2e20768a962d0eb8e6cafb4d704d778c8ac265355c3a3b0ae4baf85439bbd16',
    'src/nnopbase/individual_op/utils/indv_soc.cpp': '0e5b533dc126ca91b3e0c850c614f48569901262c8f9327a9b6fae2c3fe509ce',
}
NAMES = ('query', 'key', 'value', 'pse', 'dropMask', 'paddingMask', 'attenMask', 'prefix',
    'actualSeqLengths', 'actualSeqLengthsKv', 'qStartIdx', 'kvStartIdx', 'deqScaleQ', 'deqScaleK', 'deqScaleV',
    'queryRope', 'keyRope', 'sink', 'pScale', 'softmaxMax', 'softmaxSum', 'softmaxOut', 'attentionOut', 'workspace', 'tiling')
FORMATS = {'int64_t': 'q', 'uint64_t': 'Q', 'int32_t': 'i', 'uint32_t': 'I', 'int16_t': 'h',
           'uint16_t': 'H', 'int8_t': 'b', 'uint8_t': 'B', 'float': 'f'}
OPAQUE = {'softmaxFlashTilingData': (1248, 64), 'transposeTilingData': (1312, 64),
          'transposeTilingDataTailCore': (1376, 64)}
PADDING = {'inputParams.remainPH': (115, 1), 'inputParams.attenMaskS2SizePH': (118, 2),
           'inputParams.qStartIdxPH': (132, 4), 'inputParams.InputParamsPH': (154, 6),
           'coreParams.s1SparseValidSizePH': (700, 4)}


def validate_schema(schema):
    require(schema.get('schema') == 'nes.fa.public_pod_layout.v1'
            and schema.get('type') == 'FlashAttentionScoreGeneralTilingData'
            and schema.get('size') == 1464 and schema.get('alignment') == 8
            and len(schema.get('fields', [])) == 205, 'FA_FRESH_PUBLIC_POD_SCHEMA_REQUIRED')
    used, fields = set(), {}
    for field in schema['fields']:
        name, offset, width, count = (field[k] for k in ('name', 'offset', 'width', 'count'))
        require(name not in fields and field['type'] in FORMATS and type(offset) is int and type(count) is int
                and count > 0 and offset >= 0 and width == struct.calcsize('<' + FORMATS[field['type']])
                and offset + width * count <= 1464, 'FA_PUBLIC_FIELD_INVALID')
        region = set(range(offset, offset + width * count))
        require(not region & used, 'FA_PUBLIC_FIELD_OVERLAP');used |= region;fields[name] = field
    for name, (offset, size) in OPAQUE.items():
        require(dict(name=name, offset=offset, size=size) in schema['sections'], 'FA_PUBLIC_OPAQUE_SECTION_DIFFERS')
        region = set(range(offset, offset + size))
        require(not region & used, 'FA_PUBLIC_OPAQUE_OVERLAP');used |= region
    require(used == set(range(1464)), 'FA_PUBLIC_POD_HAS_UNBOUND_BYTES')
    for name, (offset, count) in PADDING.items():
        field = fields[name]
        require(field['type'] == 'uint8_t' and field['offset'] == offset and field['width'] == 1
                and field['count'] == count, 'FA_PUBLIC_PADDING_LAYOUT_DIFFERS')
    return fields


def decode_public_pod(raw, schema):
    fields = validate_schema(schema)
    require(type(raw) is bytes and len(raw) == 1464, 'FA_EXACT_PUBLIC_POD_BYTES_REQUIRED')
    result = {}
    for name, field in fields.items():
        values = struct.unpack_from('<' + FORMATS[field['type']] * field['count'], raw, field['offset'])
        require(all(type(value) is not float or math.isfinite(value) for value in values), 'FA_NONFINITE_PUBLIC_POD_VALUE')
        result[name] = values[0] if field['count'] == 1 else list(values)
    return result


def verify_compiled_decoder_document(document, pod):
    """Validate the real C++ values envelope and every public field exactly."""
    require(type(document) is dict and set(document) == {'schema', 'raw_size', 'fields'}
            and document['schema'] == 'nes.fa.public_pod_values.v1'
            and type(document['raw_size']) is int and document['raw_size'] == 1464
            and type(document['fields']) is dict and len(document['fields']) == len(pod) == 205
            and set(document['fields']) == set(pod), 'FA_NEW_COMPILED_DECODER_ENVELOPE_DIFFERS')
    require(document['fields'] == pod, 'FA_NEW_COMPILED_DECODER_DISAGREES')


def semantic_pod(raw, schema):
    values = decode_public_pod(raw, schema)
    return dict(fields={name: value for name, value in values.items() if name not in PADDING},
                opaque={name: raw[offset:offset + size].hex() for name, (offset, size) in OPAQUE.items()})


def derive_kernel_abi(source, metadata, schema, opbase):
    validate_schema(schema)
    require(sha(source) == SOURCE_SHA, 'FA_PINNED_ENTRY_SOURCE_DIFFERS')
    matches = re.findall(r'void\s+flash_attention_score\s*\((.*?)\)\s*\{', Path(source).read_text(), re.S)
    require(len(matches) == 1, 'FA_KERNEL_ENTRY_AMBIGUOUS')
    names = []
    for declaration in matches[0].split(','):
        match = re.fullmatch(r'\s*__gm__\s+uint8_t\s*\*\s*(\w+)\s*', declaration)
        require(match is not None, 'FA_KERNEL_ARGUMENT_UNADAPTED');names.append(match[1])
    require(tuple(names) == NAMES and metadata.get('parameters') == [None] * (len(names) - 1)
            and metadata.get('opParaSize') == 1472, 'FA_COMPILED_PARAMETER_ABI_DIFFERS')
    require(metadata.get('coreType') == 'MIX' and metadata.get('intercoreSync') == 1
            and metadata.get('taskRation') == 'tilingKey' and len(metadata.get('kernelList', [])) == 1,
            'FA_COMPILED_MIX_ABI_DIFFERS')
    entry = metadata['kernelList'][0]
    require(int(entry['tilingKey']) == KEY and entry.get('kernelType') == 'MIX_AIC'
            and entry.get('crossCoreSync') == 1 and entry.get('taskRation') == '1:2', 'FA_COMPILED_ENTRY_DIFFERS')
    for name, digest in OPBASE.items():
        require(sha(Path(opbase) / name) == digest, 'FA_FRESH_RUNTIME_ARGUMENT_CONSTRUCTION_DIFFERS')
    # Fresh opbase indv_executor.cpp637-644 prepends the hardware sync pointer;
    # indv_soc.cpp59-61 restricts this path to 910B/910_93. No FIA slot is used.
    return dict(operator='FlashAttentionScore', source_sha256=SOURCE_SHA, pointer_bytes=8, names=names,
        soc_family='ascend910b', hidden_hardware_sync_prefix_bytes=8,
        declared_tiling_pointer_offset=8 * (len(names) - 1), tiling_pointer_offset=8 * len(names),
        public_pod_bytes=schema['size'], compiler_op_para_size=metadata['opParaSize'],
        opbase_argument_source_sha256=OPBASE, workspace_pointer_offset=8 * (names.index('workspace') + 1))


def verify_bundle(build_root, expected_completion_sha):
    root = Path(build_root).resolve(strict=True);ev = root / 'evidence'
    completion = ev / 'completion.json'
    require(sha(completion) == expected_completion_sha, 'FA_NEW_COMPLETION_HASH_DIFFERS')
    data = json.loads(completion.read_bytes())
    require(data.get('schema') == 'demoncat.fa.fresh_build.v1'
            and data.get('status') == 'CPU_BUILD_ONLY_REQUIRES_NATIVE_QUALIFICATION'
            and data.get('mechanism') == '28.fa.legal_tiling' and data.get('variant') == 'fa_s1s2_m_nratio'
            and data.get('compiled_keys') == [KEY] and data.get('npu_executed') is False
            and data.get('loading_allowed') is False and data.get('historical_certificate_inherited') is False
            and data.get('previous_experiment_objects_used') is False
            and data.get('source_commit') == 'c77c0829f1f0f0b8f882db1d9b051a758b605472', 'FA_NEW_SOURCE_BUILD_REQUIRED')
    vendor = root / 'stage/vendors/dcat_f28_fa_transformer'
    files, total = {}, 0
    require(0 < len(data['artifacts']) <= 20000, 'FA_VENDOR_INVENTORY_BOUND')
    for row in data['artifacts']:
        relative = PurePosixPath(row['path']);path = vendor / row['path']
        require(not relative.is_absolute() and '..' not in relative.parts and '\\' not in row['path']
                and relative.as_posix() == row['path'] and row['path'] not in files
                and path.is_file() and path.resolve().is_relative_to(vendor), 'FA_ARTIFACT_PATH_INVALID')
        if path.is_symlink():
            require(row['path'] == 'op_impl/ai_core/tbe/op_tiling/liboptiling.so'
                    and os.readlink(path) == 'lib/linux/aarch64/libcust_opmaster_rt2.0.so', 'FA_UNREVIEWED_ARTIFACT_LINK')
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'FA_ARTIFACT_BYTES_DIFFER')
        total += row['bytes'];require(total <= 512 * 1024**2, 'FA_VENDOR_SIZE_BOUND');files[row['path']] = row
    require({p.relative_to(vendor).as_posix() for p in vendor.rglob('*') if p.is_file()} == set(files), 'FA_VENDOR_INVENTORY_DIFFERS')
    kernel_base = vendor / ('op_impl/ai_core/tbe/kernel/ascend910b/flash_attention_score/' + BINARY)
    metadata_path = kernel_base.with_suffix('.json');binary = kernel_base.with_suffix('.o')
    require(sha(binary) == data['kernel_binary_sha256'] and sha(metadata_path) == data['kernel_metadata_sha256'],
            'FA_COMPLETION_KERNEL_IDENTITY_DIFFERS')
    metadata = json.loads(metadata_path.read_bytes());validate_metadata(metadata, binary)
    schema_path, decoder = ev / 'fa_public_schema.json', ev / 'fa_pod_decoder'
    require(sha(schema_path) == data['public_pod_schema_sha256'] and sha(decoder) == data['public_pod_decoder_sha256'],
            'FA_FRESH_POD_DECODER_OR_SCHEMA_DIFFERS')
    schema = json.loads(schema_path.read_bytes());validate_schema(schema)
    require(sha(ev / 'binary_summary_registration.json') == data['binary_summary_registration_sha256'], 'FA_REGISTRATION_RECEIPT_DIFFERS')
    registration = json.loads((ev / 'binary_summary_registration.json').read_bytes())
    require(sha(vendor / registration['generated_index']['path']) == registration['generated_index']['sha256'], 'FA_SUMMARY_INDEX_DIFFERS')
    entry = vendor / 'op_impl/ai_core/tbe/dcat_f28_fa_transformer_impl/flash_attention_score/op_kernel/flash_attention_score.cpp'
    abi = derive_kernel_abi(entry, metadata, schema, root / 'dependencies/opbase')
    return dict(root=str(root), vendor=str(vendor), completion=data, completion_sha256=expected_completion_sha,
                files=files, metadata=metadata, schema=schema, abi=abi, decoder=str(decoder))


def verify_runtime_sdk(bundle, sdk_root):
    root = Path(bundle['root']);sdk = Path(sdk_root).resolve(strict=True)
    data = bundle['completion']
    require(data.get('sdk_interface_manifest_sha256') is None and len(data.get('sdk_interface_files', [])) == 5,
            'FA_MATCHED_CANN91_BUILD_INTERFACE_REQUIRED')
    for row in data['sdk_interface_files']:
        path = sdk / row['path']
        require(path.is_file() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'FA_RUNTIME_SDK_DIFFERS')
    path = root / 'evidence/fa_decoder_headers.json'
    require(sha(path) == data['decoder_headers_sha256'], 'FA_DECODER_HEADER_MANIFEST_DIFFERS')
    for row in json.loads(path.read_bytes()):
        path = Path(row['path'])
        require(path.is_file() and sha(path) == row['sha256'], 'FA_ACTUAL_DECODER_BUILD_HEADER_DIFFERS:' + str(path))
    return dict(sdk=str(sdk), exact_sdk_files_verified=True, decoder_compile_headers_verified=True)


def verify_host_projection(pod, host):
    typed = host['typed']
    mapping = {'m': 'coreParams.s1BaseSize', 's2_base': 'coreParams.s2BaseSize', 'n_ratio': 'coreParams.nRatio',
        's1_outer': 'coreParams.s1OuterSize', 's2_outer': 'coreParams.s2OuterSize',
        's1_tail': 'coreParams.s1BaseTailSize', 's2_tail': 'coreParams.s2BaseTailSize',
        'core_num': 'multiCoreParams.coreNum', 'total_query_tasks': 'multiCoreParams.totalSize',
        'split_factor': 'multiCoreParams.splitFactorSize', 'split_factor_tail': 'multiCoreParams.splitFactorTailSize',
        'sparse_start_idx': 'multiCoreParams.sparseStartIdx', 'sparse_type': 'inputParams.sparseType',
        's1_sparse_valid_size': 'coreParams.s1SparseValidSize', 's2_sparse_valid_size': 'coreParams.s2SparseValidSize',
        'b': 'inputParams.bSize', 'n2': 'inputParams.n2Size', 'g': 'inputParams.gSize',
        's1': 'inputParams.s1Size', 's2': 'inputParams.s2Size', 'd': 'inputParams.dSize', 'dv': 'coreParams.d2Size',
        'impl_mode': 'inputParams.implMode', 'keep_prob': 'inputParams.keepProb'}
    for name, field in mapping.items():require(typed.get(name) == pod[field], 'FA_HOST_PUBLIC_FIELD_DIFFERS:' + name)
    require(struct.pack('<f', typed['scale']) == struct.pack('<f', pod['inputParams.scaleValue'])
            and typed['n1'] == pod['inputParams.n2Size'] * pod['inputParams.gSize']
            and typed['s2_group'] == pod['coreParams.s2BaseSize'] * pod['coreParams.nRatio'], 'FA_HOST_DERIVED_FIELD_DIFFERS')
    for label, prefix in (('bmm1', 'bmm1TilingData.'), ('bmm2', 'bmm2TilingData.')):
        fields = {name.removeprefix(prefix): value for name, value in pod.items() if name.startswith(prefix)}
        require(len(fields) == 50 and typed[label] == fields, 'FA_HOST_MATMUL_PUBLIC_FIELDS_DIFFER')
    return dict(scalar_fields=list(mapping), complete_matmul_fields=100,
                unprojected_pod_fields='Still decoded and bound by complete host/final raw byte equality')


def verify_fa_launch(directory, call_id, bundle, host, raw_file, *, unscoped_off=False):
    observed = observe_launch(directory, call_id, kernel_sha256=bundle['completion']['kernel_binary_sha256'],
        tiling_size=bundle['schema']['size'], kernel_metadata=bundle['metadata'], kernel_abi=bundle['abi'])
    raw = bytes.fromhex(observed['raw_hex'])
    require(observed['tiling_key'] == KEY, 'FA_ACTUAL_KERNEL_ENTRY_DIFFERS')
    require((host is None) == (unscoped_off is True), 'FA_EXPLICIT_UNSCOPED_OFF_REQUIRED')
    if host is not None:
        require(host.get('schema') == 'nes.fa.host_observation.v1' and host.get('callback_status') == 0
                and host.get('eligible') is True and host.get('certificate_verified_by_host') is False
                and host.get('new_build_runtime_validation_inherited') is False, 'FA_HOST_OBSERVATION_CONTRACT_DIFFERS')
        require((host['pid'], host['tid'], host['token']) == (observed['pid'], observed['tid'], call_id), 'FA_HOST_SCOPE_OWNER_DIFFERS')
        require(host['raw_size'] == len(raw) and bytes.fromhex(host['raw_hex']) == raw, 'FA_HOST_FINAL_RAW_BYTES_DIFFER')
        require(int(host['tiling_key']) == observed['tiling_key'] and host['block_dim'] == observed['block_dim']
                and host['schedule_mode'] == observed['schedule_mode'], 'FA_HOST_FINAL_LAUNCH_DIFFERS')
        require(host['guard_reports'] and all(row['passed'] is True and not row['failed_guards']
                for row in host['guard_reports']), 'FA_ACTUAL_CONTEXT_GUARD_FAILED')
    pod = decode_public_pod(raw, bundle['schema'])
    with Path(raw_file).open('xb') as stream:stream.write(raw)
    decoded = subprocess.run([bundle['decoder'], str(raw_file)], check=True, capture_output=True, text=True, timeout=15)
    verify_compiled_decoder_document(json.loads(decoded.stdout), pod)
    projection = verify_host_projection(pod, host) if host is not None else None
    if host is not None:
        require(host['selected_factors'] == dict(m=pod['coreParams.s1BaseSize'], n_ratio=pod['coreParams.nRatio']),
                'FA_SELECTOR_AND_FINAL_PUBLIC_POD_DIFFER')
    observed.update(schema='demoncat.fa.independent_launch.v1', status='FA_FINAL_ARGUMENTS_AND_LAUNCH_VERIFIED',
        typed=pod, semantic_pod=semantic_pod(raw, bundle['schema']), host_projection=projection,
        public_pod_decoder_sha256=bundle['completion']['public_pod_decoder_sha256'],
        decoded_field_count=205, opaque_bytes_preserved=192, source_padding_bytes_preserved=17,
        host_workspace_sizes=host['workspace_sizes'] if host is not None else None,
        host_observation_present=host is not None, unscoped_off=unscoped_off,
        workspace_allocation_size_independently_observed=False,
        native_launch_verified=True, historical_qualification_inherited=False)
    return observed
