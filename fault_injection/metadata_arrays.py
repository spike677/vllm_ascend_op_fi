"""Read owned SDK metadata-array counters from the exact already-loaded proxies.

This complements, and does not replace, native device/workspace/lease accounting.
No reset API, operator call, allocation control, or late vendor dlopen is exposed.
"""
import ctypes
import copy
import hashlib
import json
import os
from pathlib import Path


SCHEMA = 'demoncat.metadata_arrays.v1'
WRITER26_SYMBOLS = {'libfault26_readonly_trace_abi0.so': 'dcat26_trace_metadata_arrays_v1',
                   'libopapi.so': 'dcat20_metadata_arrays_v1', 'libopapi_nn.so': 'dcat20_metadata_arrays_v1',
                   'libdcat26_group_v2.so': 'dcat26_group_metadata_arrays_v1'}
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


def _mapped_proxies(names=('libopapi.so', 'libopapi_nn.so')):
    result = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) != 6:
            continue
        name = Path(parts[5].removesuffix(' (deleted)')).name
        if name not in names:
            continue
        require(not parts[5].endswith(' (deleted)'), 'DELETED_PROXY_MAPPING')
        path = Path(parts[5]).resolve(strict=True)
        stat = path.stat()
        device = tuple(int(value, 16) for value in parts[3].split(':'))
        require(stat.st_ino == int(parts[4]) and device == (os.major(stat.st_dev), os.minor(stat.st_dev)),
                'MAPPED_FILE_REPLACED')
        result.setdefault(name, set()).add(str(path))
    require(result, 'MISSING_PROXY_MAPPING')
    return {name: tuple(sorted(paths)) for name, paths in result.items()}


def writer26_mapping_diagnostic(paths):
    """Keep a missing-image error independently diagnosable after process exit."""
    return dict(expected_images=sorted(WRITER26_SYMBOLS),
                missing_images=sorted(set(WRITER26_SYMBOLS) - set(paths)),
                unexpected_images=sorted(set(paths) - set(WRITER26_SYMBOLS)),
                mapped_paths={name: list(aliases) for name, aliases in sorted(paths.items())})


class LoadedMetadataArrays:
    """Attach read-only symbols after validating maps, manifest, and dladdr."""
    def __init__(self, *, kind, build_profile, anchor):
        require(kind in ('20', '39') and hasattr(os, 'RTLD_NOLOAD'), 'LINUX_AND_KNOWN_ADAPTER_REQUIRED')
        self.pid, self.kind, self.build_profile = os.getpid(), kind, build_profile
        core_name = 'libdcat20_qmm.so' if kind == '20' else 'libfault39_native_v1.so'
        self.mapping_names = ('libopapi.so', 'libopapi_nn.so', core_name)
        self.paths = _mapped_proxies(self.mapping_names)
        loader = ctypes.CDLL(None)
        locate = loader.dladdr
        locate.argtypes, locate.restype = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)], ctypes.c_int

        def origin(fn):
            info = _DlInfo()
            require(locate(ctypes.cast(fn, ctypes.c_void_p), ctypes.byref(info)) and info.dli_fname,
                    'SYMBOL_PROVIDER_UNAVAILABLE')
            return Path(os.fsdecode(info.dli_fname)).resolve(strict=True)

        # Root is bound to the actual loaded runtime export, never inferred
        # from the split proxy's possible SDK bind-mount pathname.
        core_path = origin(anchor)
        require(core_path.name == core_name and str(core_path) in self.paths.get(core_name, ()),
                'RUNTIME_ANCHOR_NOT_MAPPED')
        root = core_path.parent
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
                and set(self.paths) == set(expected_names + [core_name]), 'COUNTER_CAPABILITY_OR_PROXY_SET_MISSING')
        require(kind != '39' or spec.get('cached_registration_witness') == 'fault39_gws_commit_arrays_v1',
                'CACHED_REGISTRATION_WITNESS_BUILD_REQUIRED')
        require(manifest.get('schema') == 'fault39.native.build_manifest.v1'
                and manifest.get('profile_sha256') == build_profile
                and manifest.get('files', {}).get('version_profile.json') == _sha(profile_path), 'BUILD_MANIFEST_MISMATCH')
        require(manifest['files'].get(core_name) == _sha(core_path), 'RUNTIME_ANCHOR_BUILD_HASH_MISMATCH')
        self.files = {profile_path: _stamp(profile_path), manifest_path: _stamp(manifest_path)}
        for alias in self.paths[core_name]:
            require(Path(alias).samefile(core_path), 'AMBIGUOUS_RUNTIME_INSTANCE')
            self.files[Path(alias)] = _stamp(Path(alias))
        self.images = []
        for name in expected_names:
            path = root / name
            digest = _sha(path)
            require(digest == manifest['files'].get(name), 'PROXY_BUILD_HASH_MISMATCH')
            self.files[path] = _stamp(path)
            handles, addresses, providers = [], set(), set()
            for alias in self.paths[name]:
                mapped = Path(alias)
                require(mapped.samefile(path) and _sha(mapped) == digest, 'UNKNOWN_PROXY_ALIAS_OR_DUPLICATE')
                self.files[mapped] = _stamp(mapped)
                # A verified bind-mount alias may live under the SDK path.
                handle = ctypes.CDLL(alias, mode=os.RTLD_NOLOAD | os.RTLD_NOW)
                fn = getattr(handle, symbol, None)
                require(fn is not None, 'COUNTER_SYMBOL_MISSING')
                provider = origin(fn)
                require(str(provider) in self.paths[name] and provider.samefile(path)
                        and _sha(provider) == digest, 'COUNTER_SYMBOL_WRONG_IMAGE')
                fn.argtypes, fn.restype = [ctypes.POINTER(MetadataArrayStateV1)], ctypes.c_int
                handles.append((handle, fn))
                addresses.add(ctypes.cast(fn, ctypes.c_void_p).value)
                providers.add(str(provider))
            require(len(addresses) == 1, 'DUPLICATE_LOADED_COUNTER_INSTANCE')
            self.images.append((handles, handles[0][1], dict(name=name, path=str(path), sha256=digest,
                symbol=symbol, mapped_paths=list(self.paths[name]), symbol_providers=sorted(providers))))
        self.manifest_sha = _sha(manifest_path)

    def read(self):
        require(os.getpid() == self.pid and _mapped_proxies(self.mapping_names) == self.paths,
                'PROCESS_OR_LOADED_IMAGES_CHANGED')
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


class LoadedWriter26MetadataArrays(LoadedMetadataArrays):
    """Same counter contract for the v2 shim AND each actual trace dependency."""
    def __init__(self, *, bundle, manifest_sha256):
        require(hasattr(os, 'RTLD_NOLOAD'), 'LINUX_LOADED_WRITER_REQUIRED')
        root = Path(bundle).resolve(strict=True)
        manifest_path = root / 'build.json'
        require(_sha(manifest_path) == manifest_sha256, 'WRITER_MANIFEST_CHANGED')
        build = json.loads(manifest_path.read_text())
        spec = build.get('metadata_array_ownership', {})
        require(build.get('schema') == 'demoncat.writer26.group_deployment.v2'
                and spec.get('schema') == SCHEMA and spec.get('state_version') == 1
                and spec.get('images') == WRITER26_SYMBOLS and spec.get('reset_supported') is False
                and spec.get('runtime_quiescent_readback_required') is True
                and isinstance(spec.get('parent_receipt_sha256'), str) and len(spec['parent_receipt_sha256']) == 64
                and spec.get('parent_receipt_sha256') == build.get('files', {}).get('metadata-array-ownership.json'),
                'OLD_WRITER_OR_TRACE_DEPENDENCIES_REQUIRE_REBUILD')
        self.pid, self.kind, self.build_profile = os.getpid(), '26', manifest_sha256
        self.manifest_sha = manifest_sha256
        self.mapping_names = tuple(WRITER26_SYMBOLS)
        self.paths = _mapped_proxies(self.mapping_names)
        require(set(self.paths) == set(WRITER26_SYMBOLS), 'INCOMPLETE_WRITER_PROXY_MAPPING:' +
                json.dumps(writer26_mapping_diagnostic(self.paths), sort_keys=True, separators=(',', ':')))
        self.files = {manifest_path: _stamp(manifest_path)}
        parent_proof = root / 'metadata-array-ownership.json'
        require(_sha(parent_proof) == spec['parent_receipt_sha256'], 'WRITER_PARENT_PROOF_CHANGED')
        self.files[parent_proof] = _stamp(parent_proof)
        loader = ctypes.CDLL(None)
        locate = loader.dladdr
        locate.argtypes, locate.restype = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)], ctypes.c_int
        self.images = []
        for name, symbol in WRITER26_SYMBOLS.items():
            path = root / name
            digest = _sha(path)
            require(build['files'].get(name) == digest, 'WRITER_DEPENDENCY_HASH_MISMATCH')
            self.files[path] = _stamp(path)
            handles, addresses, providers = [], set(), set()
            for alias in self.paths[name]:
                mapped = Path(alias)
                require(mapped.samefile(path) and _sha(mapped) == digest, 'UNKNOWN_PROXY_ALIAS_OR_DUPLICATE')
                self.files[mapped] = _stamp(mapped)
                handle = ctypes.CDLL(alias, mode=os.RTLD_NOLOAD | os.RTLD_NOW)
                fn = getattr(handle, symbol, None)
                require(fn is not None, 'COUNTER_SYMBOL_MISSING')
                info = _DlInfo()
                require(locate(ctypes.cast(fn, ctypes.c_void_p), ctypes.byref(info)) and info.dli_fname,
                        'SYMBOL_PROVIDER_UNAVAILABLE')
                provider = Path(os.fsdecode(info.dli_fname)).resolve(strict=True)
                require(str(provider) in self.paths[name] and provider.samefile(path)
                        and _sha(provider) == digest, 'COUNTER_SYMBOL_WRONG_IMAGE')
                fn.argtypes, fn.restype = [ctypes.POINTER(MetadataArrayStateV1)], ctypes.c_int
                handles.append((handle, fn))
                addresses.add(ctypes.cast(fn, ctypes.c_void_p).value)
                providers.add(str(provider))
            require(len(addresses) == 1, 'DUPLICATE_LOADED_COUNTER_INSTANCE')
            self.images.append((handles, handles[0][1], dict(name=name, path=str(path), sha256=digest,
                symbol=symbol, mapped_paths=list(self.paths[name]), symbol_providers=sorted(providers))))


class MetadataArraySession:
    """Fail-closed quiescence, stable identities, and positive workload activity."""
    def __init__(self, reader, build_profile):
        require(callable(reader), 'LOADED_COUNTER_READER_REQUIRED')
        self.reader, self.build_profile = reader, build_profile
        self.implementation_sha256 = _sha(Path(__file__))
        self.first = self.previous = self.latest = None
        self.cached_registrations = None

    def _cached_activity(self, current):
        if self.cached_registrations is None:
            return False
        by_instance = {row['state']['instance']: row for row in current['rows'].values()}
        for row in self.cached_registrations:
            witness = row['metadata_arrays']
            require(isinstance(witness, dict) and witness.get('source') == 'actual_typed_gws_capture',
                    'ACTUAL_CACHE_CAPTURE_WITNESS_REQUIRED')
            b, a = witness.get('before', {}), witness.get('after', {})
            for state in (b, a):
                require(all(type(state.get(key)) is int and 0 <= state[key] < 2**64
                            for key in ('instance', *COUNTERS)) and state.get('version') == 1
                        and state.get('size') == ctypes.sizeof(MetadataArrayStateV1), 'CACHE_WITNESS_ABI_MISMATCH')
                require(all(state[key] == 0 for key in ZERO)
                        and state['acquired_arrays'] == state['released_arrays'], 'CACHE_WITNESS_UNRELEASED_OR_ERROR')
            require(b['instance'] == a['instance'] and a['instance'] in by_instance, 'CACHE_WITNESS_FOREIGN_INSTANCE')
            require(all(a[key] >= b[key] for key in MONOTONIC) and a['revision'] > b['revision']
                    and all(a[key] - b[key] == count for key, count in
                            (('capture_calls', 3), ('getter_calls', 9), ('acquired_arrays', 9), ('released_arrays', 9))),
                    'CACHE_WITNESS_POSITIVE_EXACT_CAPTURE_REQUIRED')
            loaded = by_instance[a['instance']]['state']
            require(all(loaded[key] >= a[key] for key in MONOTONIC), 'CACHE_WITNESS_AFTER_CURRENT_OR_RESET')
        return bool(self.cached_registrations)

    def bind_cached_registrations(self, rows, required_keys):
        """Bind only independent live native registrations used by this group.

        This widens the observation interval to the actual registration-time
        capture. A nonzero process lifetime total alone never grants activity.
        """
        require(self.cached_registrations is None and isinstance(rows, list) and rows,
                'ONE_COMPLETE_CACHE_WITNESS_SET_REQUIRED')
        self.read()
        keys = []
        for row in rows:
            key = row.get('cache_key')
            require(isinstance(key, (list, tuple)) and len(key) == 3 and isinstance(key[0], str) and key[0]
                    and all(type(n) is int and n > 0 for n in key[1:]) and row.get('registration_live') is True,
                    'LIVE_CACHE_REGISTRATION_IDENTITY_REQUIRED')
            keys.append(tuple(key))
        require(len(set(keys)) == len(keys) and set(keys) == set(required_keys), 'CACHE_WITNESS_COVERAGE_INCOMPLETE')
        self.cached_registrations = copy.deepcopy(rows)
        try:
            self._cached_activity(self.previous)
        except BaseException:
            self.cached_registrations = None
            raise
        return self.activity_evidence()

    def activity_evidence(self):
        if self.first is None or self.previous is None:
            return None
        delta = {key: sum(row['state'][key] - self.first['rows'][name]['state'][key]
                         for name, row in self.previous['rows'].items())
                 for key in ('capture_calls', 'getter_calls', 'acquired_arrays', 'released_arrays')}
        return dict(schema='demoncat.metadata_array_activity.v1', pid=self.previous['identity'][0],
                    build_profile_sha256=self.build_profile, group_observed_delta=delta,
                    cached_registration_witnesses=copy.deepcopy(self.cached_registrations),
                    process_lifetime_nonzero_alone_accepted=False)

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
            require(name not in names and name in WRITER26_SYMBOLS, 'INVALID_PROXY_SET')
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
        cached = self._cached_activity(current)
        if require_activity:
            direct = all(sum(row['state'][key] - self.first['rows'][name]['state'][key]
                             for name, row in indexed.items()) > 0
                         for key in ('capture_calls', 'getter_calls', 'acquired_arrays', 'released_arrays'))
            require(direct or cached, 'NO_ACTUAL_WORKLOAD_ARRAY_ACTIVITY')
        self.previous, self.latest = current, value
        return value
