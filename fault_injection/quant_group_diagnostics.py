"""Read an already loaded quant group DSO after an initialization failure.

Only the reviewed ABI, state and raw-journal exports are callable here. This
module does not install a binding, resolve the SDK, synchronize a stream, or
load a missing DSO. A snapshot is diagnostic evidence, never qualification.
"""
from contextlib import contextmanager
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re
import threading

from .core import UnsupportedCall
from . import quant_native_isolation as isolation
from .quant_group_native import ABI_TYPES, AbiSizes, PRODUCTION_SOURCES, RawRecord, RuntimeState, snapshot


CAPACITY = 32768
LIBRARY = 'runtime/vendor/op_api/lib/libcust_opapi.so'
READ_EXPORTS = {
    'abi': ('dcat_dqg_abi_v1', C.CFUNCTYPE(C.c_int, C.POINTER(AbiSizes))),
    'state': ('dcat_dqg_state_v1', C.CFUNCTYPE(C.c_int, C.POINTER(RuntimeState))),
    'raw': ('dcat_dqg_raw_v1', C.CFUNCTYPE(C.c_int, C.c_uint64, C.POINTER(RawRecord))),
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reviewed_library(bundle, approved_build_sha256):
    root = Path(bundle).resolve(strict=True)
    manifest_path = root / 'build.json'
    if (not isinstance(approved_build_sha256, str) or not re.fullmatch('[0-9a-f]{64}', approved_build_sha256)
            or _sha(manifest_path) != approved_build_sha256):
        raise UnsupportedCall('QUANT_DIAGNOSTIC_REVIEWED_BUILD_CHANGED')
    manifest = json.loads(manifest_path.read_bytes())
    if (manifest.get('schema') != 'demoncat.quant.group_deployment.v1' or manifest.get('cann') != '8.5.1'
            or manifest.get('actual_npu_calls') != 0 or manifest.get('legacy_c1_library_present') is not False
            or manifest.get('library') != LIBRARY or LIBRARY not in manifest.get('files', {})):
        raise UnsupportedCall('QUANT_DIAGNOSTIC_REVIEWED_LIBRARY_REQUIRED')
    path = (root / LIBRARY).resolve(strict=True)
    if not path.is_relative_to(root) or _sha(path) != manifest['files'][LIBRARY]:
        raise UnsupportedCall('QUANT_DIAGNOSTIC_LIBRARY_FILE_CHANGED')
    for name in PRODUCTION_SOURCES:
        expected = _sha(Path(__file__).parent / 'native_adapters' / name)
        if (manifest.get('sources', {}).get('fault_injection/native_adapters/' + name) != expected
                or manifest['files'].get(name) != expected):
            raise UnsupportedCall('QUANT_DIAGNOSTIC_REVIEWED_READ_EXPORTS_CHANGED:' + name)
    return dict(path=str(path), sha256=manifest['files'][LIBRARY], build_sha256=approved_build_sha256,
                sdk_abi_sizes=manifest['sdk_identity']['abi_sizes'])


@contextmanager
def _read_functions(owner):
    """Hold one NOLOAD reference and verify every read export's actual owner."""
    runtime = C.CDLL(None)
    runtime.dlopen.argtypes, runtime.dlopen.restype = [C.c_char_p, C.c_int], C.c_void_p
    runtime.dlsym.argtypes, runtime.dlsym.restype = [C.c_void_p, C.c_char_p], C.c_void_p
    runtime.dladdr.argtypes, runtime.dladdr.restype = [C.c_void_p, C.POINTER(isolation._DlInfo)], C.c_int
    runtime.dlclose.argtypes, runtime.dlclose.restype = [C.c_void_p], C.c_int
    handle = runtime.dlopen(os.fsencode(owner['path']), os.RTLD_NOLOAD | os.RTLD_NOW | os.RTLD_LOCAL)
    if not handle:
        raise UnsupportedCall('QUANT_DIAGNOSTIC_DSO_NO_LONGER_LOADED')
    try:
        functions = {}
        for name, (symbol, prototype) in READ_EXPORTS.items():
            address = runtime.dlsym(handle, symbol.encode())
            info = isolation._DlInfo()
            if (not address or not runtime.dladdr(address, C.byref(info)) or not info.name
                    or info.base != owner['base'] or str(Path(os.fsdecode(info.name)).resolve(strict=True)) != owner['path']
                    or (symbol in owner['symbols'] and owner['symbols'][symbol] != address)):
                raise UnsupportedCall('QUANT_DIAGNOSTIC_READ_EXPORT_OWNER_CHANGED:' + symbol)
            functions[name] = prototype(address)
        yield functions
    finally:
        if runtime.dlclose(handle):
            raise UnsupportedCall('QUANT_DIAGNOSTIC_NOLOAD_REFERENCE_RELEASE_FAILED')


def _state(function):
    value = RuntimeState()
    code = int(function(C.byref(value)))
    if code or value.version != 1 or value.size != C.sizeof(RuntimeState):
        raise UnsupportedCall('QUANT_DIAGNOSTIC_STATE_READ_FAILED:' + str(code))
    return snapshot(value)


def capture_loaded_quant_state(bundle, *, approved_build_sha256):
    """Return a bounded snapshot, retaining read errors and every available row.

    Nonzero Native statuses, unclassified rows and pending registrations are
    preserved. No attempt is made to drain, repair, reset, or infer health.
    """
    report = dict(schema='demoncat.quant.loaded_dso_diagnostic.v1', status='READ_REFUSED',
        process_id=os.getpid(), native_thread_id=threading.get_native_id(), snapshot_complete=False,
        diagnostic_device_calls=0, qualification_authority=False, records=[], read_errors=[])
    try:
        report['reviewed_library'] = identity = _reviewed_library(bundle, approved_build_sha256)
        report['mapped_before'] = list(isolation.mapped_native_modes())
        groups = [row for row in report['mapped_before'] if row['mode'] == 'group']
        if not groups:
            report['status'] = 'REVIEWED_DSO_NOT_LOADED_NO_LOAD_ATTEMPTED'
            return report
        if (len(groups) != 1 or groups[0]['path'] != identity['path'] or groups[0]['sha256'] != identity['sha256']
                or not all(name in groups[0]['symbols'] for name in isolation.MARKERS['group'])):
            raise UnsupportedCall('QUANT_DIAGNOSTIC_EXACT_MAPPED_GROUP_OWNER_REQUIRED')
        with _read_functions(groups[0]) as functions:
            sizes = AbiSizes()
            report['abi_status'] = int(functions['abi'](C.byref(sizes)))
            report['abi_sizes'] = snapshot(sizes)
            actual = {name: getattr(sizes, name) for name in ABI_TYPES}
            if (report['abi_status'] or sizes.version != 1 or sizes.size != C.sizeof(AbiSizes)
                    or actual != {name: C.sizeof(kind) for name, kind in ABI_TYPES.items()}
                    or actual != identity['sdk_abi_sizes']):
                raise UnsupportedCall('QUANT_DIAGNOSTIC_ACTUAL_SDK_CTYPES_ABI_DIFFERS')
            report['state_before'] = before = _state(functions['state'])
            if before['census']['capacity'] != CAPACITY:
                raise UnsupportedCall('QUANT_DIAGNOSTIC_REVIEWED_CENSUS_CAPACITY_CHANGED')
            last = min(before['last_sequence'], CAPACITY)
            report['sequence_range'] = dict(first=1 if last else 0, last=last,
                reported_last=before['last_sequence'], omitted=max(0, before['last_sequence'] - last))
            for sequence in range(1, last + 1):
                value = RawRecord()
                code = int(functions['raw'](sequence, C.byref(value)))
                row = dict(sequence=sequence, status=code, record=snapshot(value))
                report['records'].append(row)
                if (code or value.version != 1 or value.size != C.sizeof(RawRecord)
                        or value.kind not in (1, 2)
                        or (value.parent.token if value.kind == 1 else value.native.sequence) != sequence):
                    report['read_errors'].append(dict(sequence=sequence, status=code, reason='RAW_RECORD_UNAVAILABLE_OR_CHANGED'))
            report['state_after'] = after = _state(functions['state'])
            report['mapped_after'] = list(isolation.mapped_native_modes())
            report['state_stable'] = before == after
            report['mapping_stable'] = report['mapped_before'] == report['mapped_after']
            report['snapshot_complete'] = (report['state_stable'] and report['mapping_stable']
                and not report['read_errors'] and not report['sequence_range']['omitted']
                and not before['census']['overflow'] and before['census']['total'] == before['last_sequence'])
            report['status'] = 'READ_ONLY_SNAPSHOT_COMPLETE' if report['snapshot_complete'] else 'READ_ONLY_SNAPSHOT_INCOMPLETE'
    except BaseException as error:
        report['status'] = 'READ_REFUSED'
        report['snapshot_complete'] = False
        report['read_errors'].append(dict(error=repr(error)))
    return report


def diagnostic_summary(report):
    """Small index; the separate artifact keeps all raw rows and exact fields."""
    failures = []
    counts = dict(parent_records=0, native_records=0, prepare_original_calls=0, execute_original_calls=0)
    for row in report['records']:
        value = row['record']
        if row['status']:
            continue
        if value['kind'] == 1:
            counts['parent_records'] += 1
        elif value['kind'] == 2:
            counts['native_records'] += 1
            event = value['native']
            if event['kind'] in (2, 4):
                counts['prepare_original_calls' if event['kind'] == 2 else 'execute_original_calls'] += event['original_calls']
                if event['status']:
                    failures.append({name: event[name] for name in ('kind', 'sequence', 'registration', 'executor',
                        'thread', 'device', 'status', 'original_calls')})
    return {**{name: report.get(name) for name in ('status', 'process_id', 'native_thread_id', 'snapshot_complete',
        'state_before', 'state_after', 'state_stable', 'mapping_stable', 'sequence_range', 'read_errors')},
        'counts': counts, 'failed_native_exits': failures[:16], 'failed_native_exit_count': len(failures),
        'failure_index_omitted': max(0, len(failures) - 16)}
