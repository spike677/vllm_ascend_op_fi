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
    formats = {'uint32_t': 'I', 'int32_t': 'i', 'uint64_t': 'Q', 'int64_t': 'q', 'float': 'f'}
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


def verify_launch(directory, call_id, *, kernel_sha256, schema, host):
    """One synchronous API scope, exactly one final matching kernel launch.

    No scanning for the expected host raw bytes or key is used: the runtime's
    own tilingDataOffset selects the raw bytes. Ambiguous launch scopes refuse.
    """
    root = Path(directory).resolve(strict=True)
    require(re.fullmatch('[0-9a-f]{32}', call_id) and re.fullmatch('[0-9a-f]{64}', kernel_sha256),
            'INVALID_LAUNCH_IDENTITY')
    records = [json.loads(line) for line in (root / 'runtime.jsonl').read_text().splitlines()]
    rows = [row for row in records if row.get('call_id') == call_id]
    require(rows and all(row.get('schema') == 'demoncat.attention.runtime_observer.v1' for row in rows),
            'INDEPENDENT_RUNTIME_RECORDS_MISSING')
    begins = [row for row in rows if row['event'] == 'begin']
    ends = [row for row in rows if row['event'] == 'end']
    launches = [row for row in rows if row['event'] == 'launch']
    require(len(begins) == len(ends) == len(launches) == 1, 'AMBIGUOUS_OR_MISSING_RUNTIME_LAUNCH')
    begin, end, launch = begins[0], ends[0], launches[0]
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
    offset, size = launch['tiling_data_offset'], schema['size']
    require(len(raw_args) == launch['args_size'] and launch['has_tiling'] == 1
            and launch['no_h2d_copy'] == 0 and type(offset) is int and 0 <= offset
            and offset + size <= len(raw_args), 'FINAL_INLINE_TILING_ARGUMENT_UNAVAILABLE')
    address_offset = launch['tiling_addr_offset']
    require(type(address_offset) is int and 0 <= address_offset and address_offset + 8 <= offset,
            'TILING_ADDRESS_OFFSET_INVALID')
    host_inputs = bytes.fromhex(launch['host_input_hex'])
    require(len(host_inputs) == launch['host_input_count'] * 8, 'HOST_INPUT_DESCRIPTOR_SIZE_DIFFERS')
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
    require(all(name in host['typed'] and host['typed'][name] == value for name, value in typed.items()),
            'PUBLIC_SCHEMA_HOST_ARGUMENT_VALUES_DIFFER')
    return dict(schema='demoncat.attention.independent_launch.v1', status='FINAL_ARGUMENT_AND_LAUNCH_VERIFIED',
                call_id=call_id, pid=owner[0], tid=owner[1], kernel_sha256=kernel_sha256,
                raw_sha256=hashlib.sha256(raw).hexdigest(), runtime_log_sha256=sha(root / 'runtime.jsonl'),
                tiling_key=int(launch['tiling_key']), block_dim=launch['block_dim'],
                schedule_mode=launch['schedule_mode'], typed=typed,
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
