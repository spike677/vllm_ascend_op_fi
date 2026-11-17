"""Managed quant loader/arm exclusion based on actual mapped symbol owners.

The existing quantization._LEASE serializes managed check/load/arm operations.
This is not an interceptor for arbitrary user dlopen or direct legacy ABI calls.
Loaded DSOs remain mode owners after a Python scope/lock has been released.
"""
from contextlib import contextmanager
import ctypes as C
import functools
import hashlib
import os
from pathlib import Path
import re
import sys
import threading

from .core import UnsupportedCall


MARKERS = {'legacy': ('dcat_dq_arm', 'dcat_dq_finish'),
           'group': ('dcat_dqg_abi_v1', 'dcat_dqg_state_v1')}


class _DlInfo(C.Structure):
    _fields_ = [('name', C.c_char_p), ('base', C.c_void_p), ('symbol', C.c_char_p), ('address', C.c_void_p)]


def _unescape_path(value):
    return re.sub(r'\\([0-7]{3})', lambda match: chr(int(match.group(1), 8)), value)


def _marker_owner(info, address, rows):
    matches = [row for row in rows if row['start'] <= address < row['end']]
    if len(matches) != 1:
        raise UnsupportedCall('QUANT_SYMBOL_NOT_IN_VERIFIED_EXECUTABLE_MAPPING')
    row = matches[0]
    if row['deleted']:
        raise UnsupportedCall('QUANT_EXECUTABLE_MAPPING_DELETED_IDENTITY_UNKNOWN')
    actual = Path(os.fsdecode(info.name)).resolve(strict=True)
    if actual != Path(row['path']).resolve(strict=True):
        raise UnsupportedCall('QUANT_SYMBOL_AND_MAPPING_FILE_OWNER_DIFFER')
    def matched(status):
        return (status.st_ino == row['inode'] and os.major(status.st_dev) == row['major']
                and os.minor(status.st_dev) == row['minor'])
    with actual.open('rb') as stream:
        status = os.fstat(stream.fileno())
        if not matched(status):
            raise UnsupportedCall('QUANT_MAPPED_FILE_IDENTITY_CHANGED')
        content = stream.read()
        if not content.startswith(b'\x7fELF') or not matched(os.fstat(stream.fileno())) or not matched(actual.stat()):
            raise UnsupportedCall('QUANT_MAPPED_FILE_IDENTITY_CHANGED')
    return dict(path=str(actual), base=int(info.base), inode=status.st_ino, device=status.st_dev,
                sha256=hashlib.sha256(content).hexdigest())


def mapped_native_modes():
    """Inspect existing mappings without loading any new library.

    Basenames and environment toggles are not authority. RTLD_NOLOAD locates an
    existing handle; dladdr identifies the actual owner of each exported marker,
    whose address/range/device/inode/file hash must match /proc/self/maps.
    Unrelated system mappings may expose an overlayfs stat device different
    from their mapped backing device. They are not quant ownership evidence:
    inspect actual marker owners first, then apply the strict identity checks.
    """
    if sys.platform != 'linux' or not hasattr(os, 'RTLD_NOLOAD'):
        raise UnsupportedCall('QUANT_NATIVE_ISOLATION_REQUIRES_LINUX_LOADED_MAPS')
    rows = []
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(None, 5)
        if len(parts) != 6 or 'x' not in parts[1] or not parts[5].startswith('/'):
            continue
        deleted = parts[5].endswith(' (deleted)')
        path = _unescape_path(parts[5][:-10] if deleted else parts[5])
        major, minor = (int(value, 16) for value in parts[3].split(':'))
        start, end = (int(value, 16) for value in parts[0].split('-'))
        rows.append(dict(path=path, inode=int(parts[4]), major=major, minor=minor,
                         deleted=deleted, start=start, end=end))
    runtime = C.CDLL(None)
    runtime.dlopen.argtypes = [C.c_char_p, C.c_int]
    runtime.dlopen.restype = C.c_void_p
    runtime.dlsym.argtypes = [C.c_void_p, C.c_char_p]
    runtime.dlsym.restype = C.c_void_p
    runtime.dladdr.argtypes = [C.c_void_p, C.POINTER(_DlInfo)]
    runtime.dladdr.restype = C.c_int
    runtime.dlclose.argtypes = [C.c_void_p]
    runtime.dlclose.restype = C.c_int
    found = {}
    for path in sorted({row['path'] for row in rows}):
        handle = runtime.dlopen(os.fsencode(path), os.RTLD_NOLOAD | os.RTLD_NOW | os.RTLD_LOCAL)
        if not handle:
            continue  # An executable mapping need not be a loader-owned DSO.
        try:
            for mode, markers in MARKERS.items():
                for symbol in markers:
                    address = runtime.dlsym(handle, symbol.encode())
                    if not address:
                        continue
                    info = _DlInfo()
                    if not runtime.dladdr(address, C.byref(info)) or not info.name or not info.base:
                        raise UnsupportedCall('QUANT_SYMBOL_OWNER_UNAVAILABLE')
                    identity = _marker_owner(info, address, rows)
                    key = (mode, int(info.base))
                    record = found.setdefault(key, dict(mode=mode, **identity, symbols={}))
                    record['symbols'][symbol] = int(address)
        finally:
            if runtime.dlclose(handle):
                raise UnsupportedCall('QUANT_NOLOAD_HANDLE_RELEASE_FAILED')
    return tuple(found[key] for key in sorted(found))


def require_mode(mode):
    if mode not in MARKERS:
        raise ValueError('Unknown managed quant mode')
    mapped = mapped_native_modes()
    foreign = [row for row in mapped if row['mode'] != mode]
    if foreign:
        raise UnsupportedCall('QUANT_NATIVE_MODES_CANNOT_COLOAD:' + mode + ':' + ','.join(row['path'] for row in foreign))
    return mapped


@contextmanager
def managed(mode, *, session_runtime=None):
    from .quantization import _LEASE
    inherited = (session_runtime is not None
                 and getattr(session_runtime, '_managed_session_thread', None) == threading.get_ident())
    acquired = False
    if inherited:
        if not _LEASE.locked():
            raise UnsupportedCall('QUANT_SESSION_LEASE_NOT_HELD')
    else:
        acquired = _LEASE.acquire(blocking=False)
        if not acquired:
            raise UnsupportedCall('QUANT_NATIVE_CHECK_LOAD_ARM_LEASE_BUSY')
    try:
        require_mode(mode)
        yield
        require_mode(mode)
    finally:
        if acquired:
            _LEASE.release()


def managed_loader(mode):
    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            with managed(mode):
                return function(*args, **kwargs)
        return wrapped
    return decorate
