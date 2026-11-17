"""Read-only native intake for the existing typed native adapters."""
from collections import defaultdict
from contextlib import contextmanager
import ctypes
from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import sys
import threading

from fault_injection.runtime_discovery import _process_start_identity


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def preload_environment(path, sha256, *, environment=None):
    """Return a child-process environment for an explicitly pinned interposer."""
    path = Path(path)
    if sys.platform != 'linux' or not path.is_absolute():
        raise ValueError('LINUX_ABSOLUTE_PRELOAD_PATH_REQUIRED')
    path = path.resolve(strict=True)
    if any(char.isspace() or char == ':' for char in str(path)) or _sha256(path) != sha256:
        raise ValueError('PINNED_PRELOAD_LIBRARY_IDENTITY_REQUIRED')
    result = dict(os.environ if environment is None else environment)
    current = result.get('LD_PRELOAD')
    result['LD_PRELOAD'] = str(path) + (' ' + current if current else '')
    return result


class _DlInfo(ctypes.Structure):
    _fields_ = [('path', ctypes.c_char_p), ('base', ctypes.c_void_p),
                ('name', ctypes.c_char_p), ('address', ctypes.c_void_p)]


@dataclass(frozen=True)
class SymbolBinding:
    symbol: str
    address: int
    library_path: str
    library_sha256: str


class PinnedNativeLibrary:
    """Resolve a symbol from a hash-pinned, same-process ELF image.

    Observation defaults to RTLD_NOLOAD. Explicit loading can execute library
    constructors and is only for callers that own the pinned Linux process.
    """
    def __init__(self, path, sha256, *, loaded_only=True):
        if sys.platform != 'linux' or not hasattr(os, 'RTLD_NOLOAD'):
            raise RuntimeError('NATIVE_LOADER_REQUIRES_LINUX_RTLD_NOLOAD')
        path = Path(path)
        if not path.is_absolute():
            raise ValueError('ABSOLUTE_NATIVE_LIBRARY_PATH_REQUIRED')
        self.path = path.resolve(strict=True)
        self.sha256 = sha256
        self.process_start = _process_start_identity()
        self._file = self.path.stat()
        self._check_identity()
        mode = os.RTLD_LOCAL | (os.RTLD_NOLOAD | os.RTLD_LAZY if loaded_only else os.RTLD_NOW)
        self.handle = ctypes.CDLL(str(self.path), mode=mode)
        self._check_identity()

    def _check_identity(self):
        stat = self.path.stat()
        current = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        previous = self._file
        original = (previous.st_dev, previous.st_ino, previous.st_size,
                    previous.st_mtime_ns, previous.st_ctime_ns)
        if (_process_start_identity() != self.process_start or current != original
                or _sha256(self.path) != self.sha256):
            raise RuntimeError('NATIVE_LIBRARY_IDENTITY_CHANGED')

    def bind(self, symbol):
        self._check_identity()
        address = ctypes.cast(getattr(self.handle, symbol), ctypes.c_void_p).value
        loader = ctypes.CDLL('libdl.so.2')
        locate = loader.dladdr
        locate.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
        locate.restype = ctypes.c_int
        info = _DlInfo()
        if not address or locate(address, ctypes.byref(info)) != 1 or not info.path:
            raise RuntimeError('NATIVE_SYMBOL_OWNER_NOT_VISIBLE:' + symbol)
        if Path(os.fsdecode(info.path)).resolve(strict=True) != self.path:
            raise RuntimeError('NATIVE_SYMBOL_RESOLVED_FROM_OTHER_LIBRARY:' + symbol)
        for row in Path('/proc/self/maps').read_text().splitlines():
            extent, _, _, device, inode, *_ = row.split(maxsplit=5)
            first, last = (int(value, 16) for value in extent.split('-'))
            if first <= address < last:
                major, minor = (int(value, 16) for value in device.split(':'))
                if (os.makedev(major, minor), int(inode)) == (self._file.st_dev, self._file.st_ino):
                    return SymbolBinding(symbol, address, str(self.path), self.sha256)
                break
        raise RuntimeError('NATIVE_MAPPED_IMAGE_DIFFERS_FROM_PINNED_FILE:' + symbol)


@dataclass(frozen=True)
class NativeScope:
    process_id: int
    process_start_identity: str
    worker_id: str
    rank: int
    logical_device: int | None
    request_ids: tuple[str, ...] = ()
    phase: str | None = None
    dispatcher_token: str | None = None
    request_observed: bool = False
    model_instance_id: str | None = None

    def __post_init__(self):
        if (type(self.process_id) is not int or self.process_id <= 0 or
                type(self.rank) is not int or self.rank < 0 or
                any(not isinstance(item, str) or not item for item in (self.process_start_identity, self.worker_id)) or
                (self.logical_device is not None and
                 (type(self.logical_device) is not int or self.logical_device < 0)) or
                not isinstance(self.request_ids, tuple) or
                any(not isinstance(item, str) or not item for item in self.request_ids) or
                len(set(self.request_ids)) != len(self.request_ids) or
                self.phase not in (None, 'prefill', 'decode', 'mixed') or
                type(self.request_observed) is not bool or (self.request_observed and not self.request_ids) or
                any(item is not None and (not isinstance(item, str) or not item)
                    for item in (self.dispatcher_token, self.model_instance_id))):
            raise ValueError('COMPLETE_NATIVE_WINDOW_IDENTITY_REQUIRED')


_LAYOUTS = {
    'dynamic_quant_v2': {
        'aclnnDynamicQuantV2': ('prepare_enter', 'prepare_exit', 'execute_enter', 'execute_exit'),
    },
    'writer26': {
        'qmm_cache_run': ('qmm_enter', 'qmm_exit'),
        'fia_writer': ('fia_before', 'fia_gws', 'fia_enter', 'fia_exit'),
        'add_rms_norm_bias': ('add_rms_gws', 'add_rms_enter', 'add_rms_exit'),
    },
}
_WINDOW_LOCKS = {source: threading.Lock() for source in _LAYOUTS}


def _native_calls(source, rows, scope, thread, synchronous):
    prefixes = _LAYOUTS[source]
    families = {kind: operator for operator, kinds in prefixes.items() for kind in kinds}
    groups = defaultdict(list)
    # Registration-only records stay in the raw denominator and never become
    # execution. A zero call token is unresolved, never an address-based join.
    for event in rows:
        operator = families.get(event['kind_name'])
        if operator is not None:
            native_call = event['call']
            groups[(operator, native_call, 0 if native_call else event['sequence'])].append(event)

    records = []
    for (operator, native_call, _), events in groups.items():
        first = events[0]
        observed_kinds = tuple(event['kind_name'] for event in events)
        expected = prefixes[operator]
        failed_prepare = (source == 'dynamic_quant_v2' and observed_kinds == expected[:2]
                          and events[-1]['status'] != 0)
        starts_in_window = bool(native_call and observed_kinds[0] == expected[0])
        same_thread = all(event['thread'] == thread for event in events)
        same_device = all(event['device'] == scope.logical_device for event in events)
        consistent = (len({event['thread'] for event in events}) == 1 and
                      len({event['device'] for event in events}) == 1 and
                      len({event['executor'] for event in events if event['executor']}) <= 1 and
                      len({event.get('registration') for event in events if event.get('registration')}) <= 1)
        complete = starts_in_window and consistent and (observed_kinds == expected or failed_prepare)
        bound = complete and same_thread and same_device and synchronous
        records.append(dict(operator=operator, native_call=native_call,
            first_sequence=first['sequence'], event_sequences=[event['sequence'] for event in events],
            native_thread=first['thread'], logical_device=first['device'],
            executor=next((event['executor'] for event in events if event['executor']), None),
            registration=next((event.get('registration') for event in events if event.get('registration')), None),
            lifecycle_complete=complete, execution_returned=complete and not failed_prepare,
            native_status=events[-1]['status'] if complete else None,
            request_ids=list(scope.request_ids) if bound and scope.request_observed else [],
            dispatcher_token=scope.dispatcher_token if bound else None,
            binding_status='OBSERVED' if bound and scope.dispatcher_token else 'NOT_VISIBLE'))
    return records


class NativeObserver:
    """Capture one bounded window from a typed provider before fault selection.

    Readers own the ABI and validate overflow/counters. This intake preserves
    every raw record, unmatched call, and failure from the selected ABI.
    """
    def __init__(self, reader, *, source, build_sha256):
        if source not in _LAYOUTS:
            raise ValueError('UNSUPPORTED_NATIVE_INVENTORY_SOURCE')
        if (not isinstance(build_sha256, str) or len(build_sha256) != 64 or
                any(char not in '0123456789abcdef' for char in build_sha256)):
            raise ValueError('PINNED_NATIVE_BUILD_SHA256_REQUIRED')
        self.reader = reader
        self.source = source
        self.build_sha256 = build_sha256

    @classmethod
    def dynamic_quant(cls, runtime):
        from fault_injection.deployments.dynamic_quant_inventory import NativeDynamicQuantInventory

        device = getattr(sys.modules.get('torch_npu'), 'npu', None)
        if device is None or not device.is_initialized():
            raise RuntimeError('NATIVE_OBSERVER_REQUIRES_ALREADY_INITIALIZED_NPU')
        pinned = PinnedNativeLibrary(runtime.backend.lib._name, runtime.build_identity['library_sha256'])
        for symbol in ('dcat_dq_inventory_state_v1', 'dcat_dq_inventory_read_v1'):
            pinned.bind(symbol)
        return cls(NativeDynamicQuantInventory(runtime), source='dynamic_quant_v2',
                   build_sha256=runtime.build_identity['build_sha256'])

    @classmethod
    def writer26(cls, library, sha256):
        from fault_injection.deployments.writer26_inventory import NativeWriterInventory

        pinned = PinnedNativeLibrary(library, sha256)
        for symbol in ('dcat26_inventory_state_v1', 'dcat26_inventory_read_v1'):
            pinned.bind(symbol)
        return cls(NativeWriterInventory(library, sha256), source='writer26', build_sha256=sha256)

    @contextmanager
    def capture(self, scope):
        if (not isinstance(scope, NativeScope) or scope.process_id != os.getpid() or
                scope.process_start_identity != _process_start_identity()):
            raise ValueError('NATIVE_WINDOW_REQUIRES_CURRENT_PROCESS_SCOPE')
        lock = _WINDOW_LOCKS[self.source]
        thread = threading.get_native_id()
        synchronous = os.environ.get('TASK_QUEUE_ENABLE') == '0'
        report = dict(schema='demoncat.engineering.native_observation.v1',
            source=self.source, build_sha256=self.build_sha256, scope=asdict(scope),
            native_thread=thread, synchronous_policy=synchronous, coverage_authorized=False,
            complete=False, state_before=None, state_after=None, events=[], calls=[], errors=[])
        if not lock.acquire(blocking=False):
            report['errors'].append('OVERLAPPING_NATIVE_OBSERVATION_WINDOWS')
            yield report
            return
        try:
            try:
                before = self.reader.state()
                report['state_before'] = before
            except (OSError, ValueError, RuntimeError) as error:
                report['errors'].append('NATIVE_INVENTORY_START_REFUSED:' + str(error))
            try:
                yield report
            finally:
                if report['state_before'] is not None:
                    try:
                        if threading.get_native_id() != thread:
                            raise RuntimeError('NATIVE_WINDOW_THREAD_CHANGED')
                        after, events = self.reader.delta(before)
                        report.update(state_after=after, events=events)
                        report['calls'] = _native_calls(self.source, events, scope, thread,
                            synchronous=synchronous and os.environ.get('TASK_QUEUE_ENABLE') == '0')
                        report['complete'] = not any(not call['lifecycle_complete'] for call in report['calls'])
                    except (OSError, ValueError, RuntimeError) as error:
                        report['errors'].append('NATIVE_INVENTORY_END_REFUSED:' + str(error))
        finally:
            lock.release()


def symbol_entry_status(binding, entries):
    """A successful `dladdr` query alone cannot prove interposition.

    `entries` are independently recorded actual function entries with an
    address and a provider fingerprint, not the result of a loader lookup.
    """
    return ('OBSERVED' if any(entry.get('actual_function_entry') is True and
            entry.get('symbol') == binding.symbol and
            entry.get('address') == binding.address and
            entry.get('library_sha256') == binding.library_sha256 for entry in entries)
            else 'NOT_VISIBLE')
