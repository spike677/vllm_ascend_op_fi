"""Inspect already-loaded opapi handles; never call an operator or load CANN.

The fixed feature order is specific to the inspected torch_npu 2.9.0.post1
wheel. Its exact upstream commit is unavailable. Loaded provider identities and
real gate counters are required in addition to this resolver-shaped diagnostic.
"""
import ctypes
import hashlib
import os
from pathlib import Path


FEATURE_ORDER = (
    'libopapi_math.so', 'libopapi_nn.so', 'libopapi_cv.so',
    'libopapi_transformer.so', 'libopapi_oam.so', 'libopapi_legacy.so',
    'libopapi.so', 'libaclnn_ops_infer.so', 'libaclnn_ops_train.so',
    'libaclnn_math.so', 'libaclnn_sparse.so', 'libaclnn_fft.so', 'libaclnn_rand.so',
)
SYMBOLS = ('CanUsePTACache', 'dcat20_pta_gate_status_v1',
           'aclnnMatmulWeightNzGetWorkspaceSize', 'aclnnQuantMatmulWeightNzGetWorkspaceSize')
GATE_HANDLES = ('libopapi_math.so', 'libopapi_nn.so', 'libopapi.so')


class _DlInfo(ctypes.Structure):
    _fields_ = [('dli_fname', ctypes.c_char_p), ('dli_fbase', ctypes.c_void_p),
                ('dli_sname', ctypes.c_char_p), ('dli_saddr', ctypes.c_void_p)]


def _custom_paths():
    """Mirror the inspected resolver's existing custom files and owner guard."""
    paths, rejected = [], []
    configuration = {key: os.environ.get(key) for key in
                     ('ASCEND_CUSTOM_OPP_PATH', 'ASCEND_OPP_PATH')}
    roots = [Path(item) / 'op_api/lib' for item in
             (configuration['ASCEND_CUSTOM_OPP_PATH'] or '').split(':') if item]
    opp = configuration['ASCEND_OPP_PATH']
    config = Path(opp) / 'vendors/config.ini' if opp else None
    configuration['vendors_config_sha256'] = None
    if config is not None and config.is_file():
        raw = config.read_bytes()
        configuration['vendors_config_sha256'] = hashlib.sha256(raw).hexdigest()
        priority = next((line[len('load_priority='):] for line in raw.decode().splitlines()
                         if line.startswith('load_priority=')), '')
        roots.extend(config.parent / item / 'op_api/lib' for item in priority.split(',') if item)
    for root in roots:
        path = root / 'libcust_opapi.so'
        if not path.is_file():
            continue
        path = path.resolve(strict=True)
        uid, owner = os.getuid(), path.stat().st_uid
        if (uid == 0 and owner != 0) or (uid != 0 and owner not in (0, uid)):
            rejected.append({'path': str(path), 'reason': 'UPSTREAM_OWNER_GUARD'})
            continue
        if str(path) not in paths:
            paths.append(str(path))
    return paths, configuration, rejected


def inspect_loaded_pta_resolution():
    """Read dlsym/dladdr results from existing handles, without calling them."""
    if not hasattr(os, 'RTLD_NOLOAD'):
        raise ValueError('PTA_RESOLVER_REQUIRES_LINUX_RTLD_NOLOAD')
    loader = ctypes.CDLL(None)
    lookup = loader.dlsym
    lookup.argtypes, lookup.restype = [ctypes.c_void_p, ctypes.c_char_p], ctypes.c_void_p
    locate = loader.dladdr
    locate.argtypes, locate.restype = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)], ctypes.c_int
    hashes = {}

    def origin(address):
        info = _DlInfo()
        if not locate(address, ctypes.byref(info)) or not info.dli_fname:
            raise ValueError('PTA_SYMBOL_PROVIDER_UNAVAILABLE')
        path = Path(os.fsdecode(info.dli_fname)).resolve(strict=True)
        if str(path) not in hashes:
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return {'address': hex(address), 'provider': str(path),
                'provider_sha256': hashes[str(path)],
                'symbol': os.fsdecode(info.dli_sname) if info.dli_sname else None}

    custom, configuration, owner_skipped = _custom_paths()
    mapped_custom, mapped_torch_npu = set(), set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and Path(fields[5]).name == 'libcust_opapi.so':
            mapped_custom.add(str(Path(fields[5]).resolve(strict=True)))
        if len(fields) == 6 and Path(fields[5]).name == 'libtorch_npu.so':
            mapped_torch_npu.add(str(Path(fields[5]).resolve(strict=True)))
    unknown_custom = sorted(mapped_custom - set(custom)
                            - {row['path'] for row in owner_skipped})
    rows = []
    for name in (*custom, *FEATURE_ORDER):
        row = {'handle': name, 'custom': name in custom, 'loaded': False, 'symbols': {}}
        try:
            handle = ctypes.CDLL(name, mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
        except OSError as exc:
            row['error'] = str(exc)
        else:
            row['loaded'] = True
            for symbol in SYMBOLS:
                address = lookup(handle._handle, symbol.encode())
                row['symbols'][symbol] = origin(address) if address else None
        rows.append(row)
    winner = next((row for row in rows if row['symbols'].get('CanUsePTACache')), None)
    wheels = [{'path': path, 'file_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
              for path in sorted(mapped_torch_npu)]
    return {'schema': 'dcat20.pta_resolver.loaded.v1', 'configuration': configuration,
            'feature_order': list(FEATURE_ORDER), 'rows': rows,
            'winner_handle': winner['handle'] if winner else None,
            'unknown_loaded_custom': unknown_custom, 'owner_skipped': owner_skipped,
            'loaded_torch_npu_images': wheels,
            'vendor_functions_called': False, 'new_vendor_libraries_loaded': False}


def validate_loaded_pta_resolution(report):
    """Require the winning function/status to be one exact owned gate pair.

An earlier custom handle can legitimately resolve a dependency's main gate.
Handle names are not approval: both actual symbol addresses/provider identities
must equal the corresponding pair from an independently inspected gate handle.
The caller additionally validates all three gates' build/status/wheel identity.
"""
    if report.get('schema') != 'dcat20.pta_resolver.loaded.v1':
        raise ValueError('INVALID_PTA_RESOLVER_REPORT')
    if report.get('unknown_loaded_custom'):
        raise ValueError('UNENUMERATED_LOADED_CUSTOM_OPAPI')
    rows = report['rows']
    if any(row['custom'] and not row['loaded'] for row in rows):
        raise ValueError('CUSTOM_OPAPI_LOOKUP_NOT_OBSERVED')
    gates = []
    for name in GATE_HANDLES:
        entry = next((row for row in rows if row['handle'] == name), None)
        if not entry or not entry['loaded']:
            raise ValueError('ACTUAL_GATE_HANDLE_NOT_LOADED:' + name)
        can = entry['symbols'].get('CanUsePTACache')
        status = entry['symbols'].get('dcat20_pta_gate_status_v1')
        if (not can or not status or can['provider'] != status['provider']
                or can['provider_sha256'] != status['provider_sha256']
                or Path(can['provider']).name != name
                or can['symbol'] != 'CanUsePTACache'
                or status['symbol'] != 'dcat20_pta_gate_status_v1'):
            raise ValueError('CAN_USE_PROVIDER_IS_NOT_OWNED_GATE:' + name)
        gates.append((name, can, status))
    winner = next((row for row in rows if row['symbols'].get('CanUsePTACache')), None)
    if winner:
        for name, can, status in gates:
            if (winner['symbols'].get('CanUsePTACache') == can
                    and winner['symbols'].get('dcat20_pta_gate_status_v1') == status):
                return {**can, 'gate_handle': name, 'winner_handle': winner['handle']}
    raise ValueError('WINNING_OPAPI_PROVIDER_IS_NOT_VERIFIED_GATE')
