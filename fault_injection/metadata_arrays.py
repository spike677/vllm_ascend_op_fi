"""Read owned SDK metadata-array counters from the exact already-loaded proxies.

This complements, and does not replace, native device/workspace/lease accounting.
No reset API, operator call, allocation control, or late vendor dlopen is exposed.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path


SCHEMA = 'demoncat.metadata_arrays.v1'
COUNTERS = ('capture_calls', 'active_captures', 'getter_calls', 'active_getters',
            'getter_failures', 'getter_exceptions', 'capture_exceptions',
            'acquired_arrays', 'released_arrays', 'live_arrays', 'peak_live_arrays',
            'ownership_errors', 'revision')
MONOTONIC = tuple(key for key in COUNTERS if not key.startswith('active_') and key != 'live_arrays')
ZERO = ('active_captures', 'active_getters', 'getter_failures', 'getter_exceptions',
        'capture_exceptions', 'live_arrays', 'ownership_errors')


class MetadataArrayStateV1(ctypes.Structure):
    _fields_ = [('version', ctypes.c_uint32), ('size', ctypes.c_uint32),
               ('instance', ctypes.c_uint64)] + [(name, ctypes.c_uint64) for name in COUNTERS]


class _DlInfo(ctypes.Structure):
    _fields_ = [('dli_fname', ctypes.c_char_p), ('dli_fbase', ctypes.c_void_p),
               ('dli_sname', ctypes.c_char_p), ('dli_saddr', ctypes.c_void_p)]


def require(condition, reason):
    if not condition:
        raise RuntimeError('METADATA_ARRAYS_' + reason)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stamp(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _mapped_proxies():
    result = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) != 6:
            continue
        name = Path(parts[5].removesuffix(' (deleted)')).name
        if name not in ('libopapi.so', 'libopapi_nn.so'):
            continue
        require(not parts[5].endswith(' (deleted)'), 'DELETED_PROXY_MAPPING')
        path = Path(parts[5]).resolve(strict=True)
        stat = path.stat()
        device = tuple(int(value, 16) for value in parts[3].split(':'))
        require(stat.st_ino == int(parts[4]) and device == (os.major(stat.st_dev), os.minor(stat.st_dev)),
                'MAPPED_FILE_REPLACED')
        result.setdefault(name, set()).add(str(path))
    require(result and all(len(paths) == 1 for paths in result.values()), 'MISSING_OR_AMBIGUOUS_PROXY')
    return {name: next(iter(paths)) for name, paths in result.items()}


class LoadedMetadataArrays:
    """Attach read-only symbols after validating maps, manifest, and dladdr."""
    def __init__(self, *, kind, build_profile):
        require(kind in ('20', '39') and hasattr(os, 'RTLD_NOLOAD'), 'LINUX_AND_KNOWN_ADAPTER_REQUIRED')
        self.pid, self.kind, self.build_profile = os.getpid(), kind, build_profile
        self.paths = _mapped_proxies()
        roots = {Path(path).parent for path in self.paths.values()}
        require(len(roots) == 1, 'MIXED_PROXY_ROOTS')
        root = next(iter(roots))
        profile_path, manifest_path = root / 'version_profile.json', root / 'build_manifest.json'
        profile, manifest = json.loads(profile_path.read_text()), json.loads(manifest_path.read_text())
        content = dict(profile)
        profile_sha = content.pop('profile_sha256', None)
        canonical = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        adapter = 'fault20-isolated-cache-abi0-v1' if kind == '20' else 'matmul-cache-abi0-v1'
        require(profile.get('schema') == 'fault39.cann.build_profile.v1' and profile_sha == canonical == build_profile
                and profile.get('adapter') == adapter, 'BUILD_PROFILE_MISMATCH')
        symbol = 'dcat20_metadata_arrays_v1' if kind == '20' else 'fault39_metadata_arrays_v1'
        spec = profile.get('metadata_array_ownership', {})
        expected_names = ['libopapi.so'] + (['libopapi_nn.so'] if profile.get('proxy_layout') in
                                          ('main_and_nn', 'main_nn_math') else [])
        require(spec.get('schema') == SCHEMA and spec.get('state_version') == 1
                and spec.get('state_symbol') == symbol and spec.get('proxy_images') == expected_names
                and spec.get('reset_supported') is False and spec.get('runtime_quiescent_readback_required') is True
                and set(self.paths) == set(expected_names), 'COUNTER_CAPABILITY_OR_PROXY_SET_MISSING')
        require(manifest.get('schema') == 'fault39.native.build_manifest.v1'
                and manifest.get('profile_sha256') == build_profile
                and manifest.get('files', {}).get('version_profile.json') == _sha(profile_path), 'BUILD_MANIFEST_MISMATCH')
        self.files = {profile_path: _stamp(profile_path), manifest_path: _stamp(manifest_path)}
        loader = ctypes.CDLL(None)
        locate = loader.dladdr
        locate.argtypes, locate.restype = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)], ctypes.c_int
        self.images = []
        for name in expected_names:
            path = Path(self.paths[name])
            digest = _sha(path)
            require(digest == manifest['files'].get(name), 'PROXY_BUILD_HASH_MISMATCH')
            self.files[path] = _stamp(path)
            # RTLD_NOLOAD cannot bring a new vendor image into this process.
            handle = ctypes.CDLL(str(path), mode=os.RTLD_NOLOAD | os.RTLD_NOW)
            fn = getattr(handle, symbol, None)
            require(fn is not None, 'COUNTER_SYMBOL_MISSING')
            info = _DlInfo()
            require(locate(ctypes.cast(fn, ctypes.c_void_p), ctypes.byref(info)) and info.dli_fname
                    and Path(os.fsdecode(info.dli_fname)).resolve(strict=True) == path, 'COUNTER_SYMBOL_WRONG_IMAGE')
            fn.argtypes, fn.restype = [ctypes.POINTER(MetadataArrayStateV1)], ctypes.c_int
            self.images.append((handle, fn, dict(name=name, path=str(path), sha256=digest, symbol=symbol)))
        self.manifest_sha = _sha(manifest_path)

    def read(self):
        require(os.getpid() == self.pid and _mapped_proxies() == self.paths, 'PROCESS_OR_LOADED_IMAGES_CHANGED')
        require(all(_stamp(path) == stamp for path, stamp in self.files.items()), 'BUILD_FILES_CHANGED')
        rows = []
        for handle, fn, identity in self.images:
            state = MetadataArrayStateV1()
            state.version, state.size = 1, ctypes.sizeof(state)
            require(fn(ctypes.byref(state)) == 0, 'COUNTER_READ_FAILED_OR_UNSTABLE')
            rows.append(dict(identity, state={name: getattr(state, name) for name, _ in state._fields_}))
        return dict(schema=SCHEMA, source='ACTUAL_LOADED_PROXY_COUNTERS', pid=self.pid,
                    build_profile_sha256=self.build_profile, build_manifest_sha256=self.manifest_sha,
                    images=rows, scope='SDK_GETTER_ARRAYS_ONLY', reset_supported=False)


class MetadataArraySession:
    """Fail-closed quiescence, stable identities, and positive workload activity."""
    def __init__(self, reader, build_profile):
        require(callable(reader), 'LOADED_COUNTER_READER_REQUIRED')
        self.reader, self.build_profile = reader, build_profile
        self.implementation_sha256 = _sha(Path(__file__))
        self.first = self.previous = self.latest = None

    def read(self, *, require_activity=False):
        value = self.reader()
        require(isinstance(value, dict) and value.get('schema') == SCHEMA
                and value.get('pid') == os.getpid() and value.get('build_profile_sha256') == self.build_profile
                and value.get('reset_supported') is False and value.get('scope') == 'SDK_GETTER_ARRAYS_ONLY'
                and isinstance(value.get('build_manifest_sha256'), str)
                and len(value['build_manifest_sha256']) == 64, 'INVALID_LOADED_COUNTER_REPORT')
        rows = value.get('images')
        require(isinstance(rows, list) and rows, 'EMPTY_LOADED_COUNTER_REPORT')
        names, instances = set(), set()
        identity = (value['pid'], value['build_profile_sha256'], value['build_manifest_sha256'])
        indexed = {}
        for row in rows:
            name, state = row.get('name'), row.get('state', {})
            require(name not in names and name in ('libopapi.so', 'libopapi_nn.so'), 'INVALID_PROXY_SET')
            names.add(name)
            require(all(type(state.get(key)) is int and 0 <= state[key] < 2**64
                        for key in ('instance', *COUNTERS)) and state.get('version') == 1
                    and state.get('size') == ctypes.sizeof(MetadataArrayStateV1), 'COUNTER_ABI_MISMATCH')
            require(state['instance'] > 0 and state['instance'] not in instances, 'SHARED_OR_MISSING_COUNTER_INSTANCE')
            instances.add(state['instance'])
            require(all(state[key] == 0 for key in ZERO) and state['acquired_arrays'] == state['released_arrays'],
                    'GETTER_FAILURE_OR_UNRELEASED_ARRAYS')
            require(isinstance(row.get('sha256'), str) and len(row['sha256']) == 64
                    and isinstance(row.get('path'), str) and bool(row['path'])
                    and isinstance(row.get('symbol'), str) and bool(row['symbol']), 'IMAGE_IDENTITY_MISSING')
            indexed[name] = dict(identity=(row['path'], row['sha256'], row['symbol'], state['instance']), state=dict(state))
        require('libopapi.so' in names, 'MAIN_PROXY_MISSING')
        current = dict(identity=identity, rows=indexed)
        if self.previous is not None:
            require(current['identity'] == self.previous['identity'] and indexed.keys() == self.previous['rows'].keys(),
                    'BUILD_OR_PROXY_SET_CHANGED')
            for name, row in indexed.items():
                prior = self.previous['rows'][name]
                require(row['identity'] == prior['identity'] and all(row['state'][key] >= prior['state'][key]
                        for key in MONOTONIC), 'INSTANCE_CHANGED_OR_COUNTER_RESET')
        if self.first is None:
            self.first = current
        if require_activity:
            require(all(sum(row['state'][key] - self.first['rows'][name]['state'][key] for name, row in indexed.items()) > 0
                        for key in ('capture_calls', 'getter_calls', 'acquired_arrays', 'released_arrays')),
                    'NO_ACTUAL_WORKLOAD_ARRAY_ACTIVITY')
        self.previous, self.latest = current, value
        return value
