"""Verify final runtime arguments independently of host policy and statistics.

This verifier does not discover a statistical first difference and must never be
fed into the statistics comparator. It accepts raw rtArgsEx snapshots, registered
ELF bytes, public serializer layout and the *new* build's artifact identity.
"""
from contextlib import contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import threading


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode_raw(raw, schema):
    require(type(schema.get('size')) is int and len(raw) == schema['size'], 'LAUNCH_RAW_LENGTH_DIFFERS')
    result, used = {}, set()
    formats = {'bool': '?', 'uint8_t': 'B', 'int8_t': 'b', 'uint16_t': 'H', 'int16_t': 'h',
               'uint32_t': 'I', 'int32_t': 'i', 'uint64_t': 'Q', 'int64_t': 'q', 'float': 'f', 'double': 'd'}
    for field in schema['fields']:
        name, offset, width, count = (field[k] for k in ('name', 'offset', 'width', 'count'))
        require(name not in result and field['dtype'] in formats, 'PUBLIC_SCHEMA_FIELD_INVALID')
        fmt = formats[field['dtype']]
        require(type(offset) is int and type(count) is int and count > 0 and offset >= 0
                and width == struct.calcsize('<' + fmt) and offset + width * count <= len(raw),
                'PUBLIC_SCHEMA_FIELD_OUT_OF_BOUNDS')
        positions = set(range(offset, offset + width * count))
        require(not positions & used, 'PUBLIC_SCHEMA_FIELD_OVERLAP')
        used |= positions
        values = struct.unpack_from('<' + fmt * count, raw, offset)
        result[name] = values[0] if count == 1 else list(values)
    require(bool(result), 'EMPTY_PUBLIC_SCHEMA')
    return result


def verify_public_values(typed, schema, host_typed):
    # SDK GetFieldInfo exposes its automatically generated alignment arrays;
    # the source-declared public getters do not. These two exact fields were
    # independently derived by the fresh 1576-byte public serializer sentinel.
    # Preserve and check them explicitly rather than dropping unknown fields.
    known = {
        'baseParams.FusedInferAttentionBaseParamsPH': (81, 7),
        'prefixParams.FusedInferAttentionPrefixParamsPH': (1537, 7),
    } if schema.get('type') == 'FusedInferAttentionScoreTilingData' and schema['size'] == 1576 else {}
    fields = {field['name']: field for field in schema['fields']}
    padding = {}
    for name, value in typed.items():
        if name in known:
            offset, count = known[name]
            field = fields[name]
            require(field['dtype'] == 'uint8_t' and field['width'] == 1
                    and field['offset'] == offset and field['count'] == count
                    and value == [0]*count, 'PUBLIC_SERIALIZER_PADDING_DIFFERS')
            padding[name] = value
        else:
            require(name in host_typed and host_typed[name] == value, 'PUBLIC_SCHEMA_HOST_ARGUMENT_VALUES_DIFFER')
    return padding


def derive_fia_kernel_abi(source, metadata, *, soc_family=None):
    """Read the pinned source entry declaration, not expected host data bytes."""
    text = Path(source).read_text()
    match = re.findall(r'void\s+fused_infer_attention_score\((.*?)\)\s*\{', text, re.S)
    require(len(match) == 1, 'FIA_KERNEL_ENTRY_DECLARATION_AMBIGUOUS')
    declarations = [part.strip() for part in match[0].split(',')]
    names = []
    for declaration in declarations:
        found = re.fullmatch(r'__gm__\s+uint8_t\s*\*\s*(\w+)', declaration)
        require(found, 'FIA_KERNEL_ARGUMENT_ABI_UNADAPTED')
        names.append(found[1])
    require(names[-2:] == ['workspace', 'tiling'] and len(set(names)) == len(names)
            and len(metadata['parameters']) == len(names)-1
            and all(value is None for value in metadata['parameters']), 'FIA_KERNEL_METADATA_SIGNATURE_DIFFERS')
    hidden_prefix = 0
    if metadata.get('coreType') == 'MIX':
        require(soc_family == 'ascend910b' and metadata.get('intercoreSync') == 1
                and metadata.get('kernelList') and all(row.get('kernelType') == 'MIX_AIC'
                    and row.get('crossCoreSync') == 1 and row.get('taskRation') == '1:2'
                    for row in metadata['kernelList']), 'FIA_MIXED_CORE_SYNC_ABI_UNADAPTED')
        # Pinned opbase cd48: indv_soc.cpp59 limits UseCoreTypeMagic to
        # ascend910b/910_93; indv_executor.cpp637 prepends ctrlAddr for MIX.
        # The pointer is additional to the public kernel entry declaration.
        hidden_prefix = 8
    return dict(source_sha256=sha(source), pointer_bytes=8, names=names,
                soc_family=soc_family, hidden_hardware_sync_prefix_bytes=hidden_prefix,
                declared_tiling_pointer_offset=(len(names)-1)*8,
                tiling_pointer_offset=hidden_prefix+(len(names)-1)*8)


def bundle_kernel_abi(vendor, files, kernel_sha256):
    root = Path(vendor)
    binaries = [name for name, row in files.items() if name.endswith('.o') and row['sha256'] == kernel_sha256]
    require(len(binaries) == 1, 'ONE_MANIFEST_KERNEL_REQUIRED')
    meta_path = root/Path(binaries[0]).with_suffix('.json')
    require(meta_path.relative_to(root).as_posix() in files, 'KERNEL_METADATA_NOT_IN_MANIFEST')
    metadata = json.loads(meta_path.read_bytes())
    require(metadata['sha256'] == kernel_sha256, 'KERNEL_METADATA_BINARY_SHA_DIFFERS')
    sources = [root/name for name in files if name.endswith('/fused_infer_attention_score/op_kernel/fused_infer_attention_score.cpp')]
    require(sources and len({sha(path) for path in sources}) == 1, 'NEW_KERNEL_ENTRY_SOURCE_MISSING_OR_AMBIGUOUS')
    require('/kernel/ascend910b/' in binaries[0], 'FIA_KERNEL_SOC_ABI_UNADAPTED')
    return metadata, derive_fia_kernel_abi(sources[0], metadata, soc_family='ascend910b')


def verify_launch(directory, call_id, *, kernel_sha256, schema, host, kernel_metadata=None, kernel_abi=None):
    """One synchronous API scope, exactly one final matching kernel launch.

    No scanning for the expected host raw bytes or key is used: the runtime's
    own tilingDataOffset selects the raw bytes. Ambiguous launch scopes refuse.
    """
    root = Path(directory).resolve(strict=True)
    require(re.fullmatch('[0-9a-f]{32}', call_id) and re.fullmatch('[0-9a-f]{64}', kernel_sha256),
            'INVALID_LAUNCH_IDENTITY')
    records = [json.loads(line) for line in (root / 'runtime.jsonl').read_text().splitlines()]
    require(not any(row.get('event') == 'capture_overflow' for row in records), 'RUNTIME_OBSERVER_CAPTURE_BUDGET_EXCEEDED')
    rows = [row for row in records if row.get('call_id') == call_id]
    require(rows and all(row.get('schema') == 'demoncat.attention.runtime_observer.v1' for row in rows),
            'INDEPENDENT_RUNTIME_RECORDS_MISSING')
    begins = [row for row in rows if row['event'] == 'begin']
    ends = [row for row in rows if row['event'] == 'end']
    launches = [row for row in rows if row['event'] == 'launch']
    require(len(begins) == len(ends) == 1 and launches, 'AMBIGUOUS_OR_MISSING_RUNTIME_LAUNCH')
    candidates, excluded, hashes = [], [], {}
    for candidate in launches:
        # msprobe may launch its own statistics operators inside the real API
        # wrapper. Classify only from independently observed binary/function
        # bindings, never from expected block count, key, host bytes or order.
        registrations = [r for r in records if r['event'] == 'register'
            and r['pid'] == candidate['pid'] and r['handle'] == candidate['handle']
            and r['generation'] == candidate['generation']]
        require(len(registrations) == 1 and registrations[0]['result'] == 0
                and registrations[0]['monotonic_ns'] <= candidate['launch_begin_ns'],
                'REGISTERED_BINARY_BINDING_MISSING')
        registered = registrations[0]
        name = candidate['binary_file']
        require(registered['binary_file'] == name and re.fullmatch(r'registered-\d+-\d+\.bin', name),
                'REGISTERED_BINARY_PATH_INVALID')
        binary = root/name
        require(not binary.is_symlink() and binary.resolve(strict=True).parent == root
                and binary.stat().st_size == registered['binary_bytes'],
                'ACTUAL_REGISTERED_BINARY_DIFFERS_FROM_NEW_BUILD')
        if name not in hashes:
            hashes[name] = sha(binary)
        if candidate.get('argument_layout') == 'aclrt_host_placeholders':
            bindings = [r for r in records if r['event'] == 'function_bind'
                and r['pid'] == candidate['pid'] and r['function_handle'] == candidate['function_handle']
                and r['handle'] == candidate['handle'] and r['generation'] == candidate['generation']
                and r['monotonic_ns'] <= candidate['launch_begin_ns']]
            require(bindings and all(r['result'] == 0 and int(r['tiling_key']) == int(candidate['tiling_key'])
                                     for r in bindings), 'ACL_ACTUAL_FUNCTION_BINDING_DIFFERS')
        if hashes[name] == kernel_sha256:
            candidates.append(candidate)
        else:
            require(candidate['result'] == 0, 'AUXILIARY_RUNTIME_LAUNCH_FAILED')
            excluded.append(dict(api=candidate['api'], kernel_sha256=hashes[name],
                function_handle=candidate.get('function_handle'), tiling_key=candidate['tiling_key'],
                reason='INDEPENDENT_REGISTERED_BINARY_DIFFERS_FROM_FRESH_FIA'))
    require(len(candidates) == 1, 'AMBIGUOUS_OR_MISSING_RUNTIME_LAUNCH')
    begin, end, launch = begins[0], ends[0], candidates[0]
    owner = (begin['pid'], begin['tid'])
    require(all((row['pid'], row['tid']) == owner for row in rows), 'LAUNCH_SCOPE_OWNER_DIFFERS')
    require(owner == (host['pid'], host['tid']), 'HOST_RUNTIME_OWNER_DIFFERS')
    require(host.get('token', host.get('call_id')) == call_id, 'HOST_RUNTIME_CALL_ID_DIFFERS')
    require(begin['monotonic_ns'] <= launch['launch_begin_ns'] <= launch['launch_end_ns']
            <= launch['monotonic_ns'] <= end['monotonic_ns'], 'LAUNCH_SCOPE_TIME_DIFFERS')
    require(launch['result'] == 0 and launch['snapshot_valid'] is True
            and launch['observer_mutates_arguments'] is False, 'RUNTIME_LAUNCH_OR_CAPTURE_FAILED')
    syncs = [row for row in rows if row['event'] in ('device_sync', 'stream_sync')
             and launch['monotonic_ns'] <= row['monotonic_ns'] <= end['monotonic_ns']]
    require(syncs and all(row['result'] == 0 for row in syncs)
            and any(row['event'] == 'device_sync' or row['stream'] == launch['stream'] for row in syncs),
            'SUCCESSFUL_POST_LAUNCH_RUNTIME_SYNC_MISSING')
    registrations = [row for row in records if row['event'] == 'register'
                     and row['pid'] == launch['pid'] and row['handle'] == launch['handle']
                     and row['generation'] == launch['generation']]
    require(len(registrations) == 1 and registrations[0]['result'] == 0
            and registrations[0]['monotonic_ns'] <= launch['launch_begin_ns'], 'REGISTERED_BINARY_BINDING_MISSING')
    registered = registrations[0]
    name = launch['binary_file']
    require(registered['binary_file'] == name and re.fullmatch(r'registered-\d+-\d+\.bin', name),
            'REGISTERED_BINARY_PATH_INVALID')
    binary = root / name
    require(not binary.is_symlink() and binary.resolve(strict=True).parent == root
            and binary.stat().st_size == registered['binary_bytes'] and sha(binary) == kernel_sha256,
            'ACTUAL_REGISTERED_BINARY_DIFFERS_FROM_NEW_BUILD')
    raw_args = bytes.fromhex(launch['args_hex'])
    size = schema['size']
    if launch.get('argument_layout') == 'aclrt_host_placeholders':
        require(kernel_metadata and kernel_abi and kernel_metadata['sha256'] == kernel_sha256
                and kernel_abi['pointer_bytes'] == 8, 'ACL_FINAL_ARGUMENT_KERNEL_ABI_REQUIRED')
        bindings = [row for row in records if row['event'] == 'function_bind'
                    and row['pid'] == launch['pid'] and row['function_handle'] == launch['function_handle']
                    and row['handle'] == launch['handle'] and row['generation'] == launch['generation']
                    and row['monotonic_ns'] <= launch['launch_begin_ns']]
        require(bindings and all(row['result'] == 0 and int(row['tiling_key']) == int(launch['tiling_key'])
                                 for row in bindings), 'ACL_ACTUAL_FUNCTION_BINDING_DIFFERS')
        require(int(launch['tiling_key']) in {int(row['tilingKey']) for row in kernel_metadata['kernelList']},
                'ACL_FUNCTION_KEY_NOT_IN_NEW_BINARY')
        placeholders = bytes.fromhex(launch['placeholders_hex'])
        require(len(placeholders) > 0 and len(placeholders) % 8 == 0, 'ACL_PLACEHOLDER_DESCRIPTOR_INVALID')
        pairs = list(struct.iter_unpack('<II', placeholders))
        require(len({addr for addr, _ in pairs}) == len(pairs), 'ACL_DUPLICATE_PLACEHOLDER_ADDRESS')
        # The exact compiled entry declaration places `tiling` after workspace;
        # mixed-core910B also has the documented leading hardware-sync pointer.
        # Select its argument slot, never a matching byte pattern or host value.
        selected = [(addr, data) for addr, data in pairs if addr == kernel_abi['tiling_pointer_offset']]
        require(len(selected) == 1, 'ACL_KERNEL_TILING_PLACEHOLDER_MISSING')
        address_offset, offset = selected[0]
        host_inputs = b''.join(struct.pack('<II', addr, data) for addr, data in pairs if addr != address_offset)
    else:
        offset, address_offset = launch['tiling_data_offset'], launch['tiling_addr_offset']
        require(launch['has_tiling'] == 1 and launch['no_h2d_copy'] == 0, 'FINAL_INLINE_TILING_ARGUMENT_UNAVAILABLE')
        host_inputs = bytes.fromhex(launch['host_input_hex'])
        require(len(host_inputs) == launch['host_input_count'] * 8, 'HOST_INPUT_DESCRIPTOR_SIZE_DIFFERS')
    require(len(raw_args) == launch['args_size'] and type(offset) is int and 0 <= offset
            and offset + size <= len(raw_args), 'FINAL_INLINE_TILING_ARGUMENT_UNAVAILABLE')
    require(type(address_offset) is int and 0 <= address_offset and address_offset + 8 <= offset,
            'TILING_ADDRESS_OFFSET_INVALID')
    for addr, data in struct.iter_unpack('<II', host_inputs):
        require(addr + 8 <= offset and offset + size <= data < len(raw_args),
                'TILING_AND_HOST_INPUT_BYTES_OVERLAP_OR_UNADAPTED')
    raw = raw_args[offset:offset + size]
    require(raw == bytes.fromhex(host['raw_hex']) and len(raw) == host['raw_size'], 'HOST_FINAL_ARGUMENT_BYTES_DIFFER')
    require(int(launch['tiling_key']) == int(host['tiling_key'])
            and launch['block_dim'] == host['block_dim'], 'FINAL_LAUNCH_KEY_OR_BLOCKS_DIFFER')
    require(launch['schedule_mode'] >= 0 and launch['schedule_mode'] == host['schedule_mode'],
            'FINAL_LAUNCH_SCHEDULE_UNOBSERVED_OR_DIFFERENT')
    typed = decode_raw(raw, schema)
    padding = verify_public_values(typed, schema, host['typed'])
    return dict(schema='demoncat.attention.independent_launch.v1', status='FINAL_ARGUMENT_AND_LAUNCH_VERIFIED',
                call_id=call_id, pid=owner[0], tid=owner[1], kernel_sha256=kernel_sha256,
                raw_sha256=hashlib.sha256(raw).hexdigest(), runtime_log_sha256=sha(root / 'runtime.jsonl'),
                tiling_key=int(launch['tiling_key']), block_dim=launch['block_dim'],
                schedule_mode=launch['schedule_mode'], typed=typed,
                kernel_argument_abi=kernel_abi,
                verified_public_serializer_padding=padding,
                excluded_independently_bound_launches=excluded,
                kernel_argument_raw_observed=True, native_launch_verified=True,
                device_completion_observed=True, production_loading_allowed=False,
                public_worker_qualified=False, c2=False, c3=False, statistics='NOT_READ')


class LaunchObserver:
    """Owns a synchronous capture scope; library must already be preloaded."""
    def __init__(self, library, expected_sha256):
        self.path = Path(library).resolve(strict=True)
        require(sha(self.path) == expected_sha256, 'OBSERVER_LIBRARY_HASH_DIFFERS')
        preload = [Path(p).resolve() for p in os.environ.get('LD_PRELOAD', '').split(':') if p]
        require(self.path in preload, 'OBSERVER_MUST_BE_PRELOADED_BEFORE_FRAMEWORK')
        self.lib = ctypes.CDLL(str(self.path))
        self.lib.dcat_attention_observer_begin_v1.argtypes = [ctypes.c_char_p]
        self.lib.dcat_attention_observer_begin_v1.restype = ctypes.c_int
        self.lib.dcat_attention_observer_end_v1.argtypes = []
        self.lib.dcat_attention_observer_end_v1.restype = ctypes.c_int
        self.lib.dcat_attention_observer_synchronize_v1.argtypes = []
        self.lib.dcat_attention_observer_synchronize_v1.restype = ctypes.c_int
        self.lib.dcat_attention_observer_memory_selftest_v1.restype = ctypes.c_int
        require(self.lib.dcat_attention_observer_memory_selftest_v1() == 0,
                'OBSERVER_BOUNDED_SELF_MEMORY_READ_FAILED')
        self.owner, self.active = threading.get_ident(), False

    @contextmanager
    def scope(self, call_id):
        require(threading.get_ident() == self.owner and not self.active, 'OBSERVER_SCOPE_OWNER_OR_REENTRY')
        require(self.lib.dcat_attention_observer_begin_v1(call_id.encode()) == 0, 'OBSERVER_SCOPE_BEGIN_FAILED')
        self.active = True
        try:
            yield
        finally:
            self.active = False
            require(self.lib.dcat_attention_observer_end_v1() == 0, 'OBSERVER_SCOPE_END_FAILED')

    def synchronize(self):
        require(self.active and threading.get_ident() == self.owner, 'OBSERVER_SYNC_REQUIRES_OWNED_SCOPE')
        require(self.lib.dcat_attention_observer_synchronize_v1() == 0, 'OBSERVER_ACTUAL_ACL_SYNC_FAILED')
