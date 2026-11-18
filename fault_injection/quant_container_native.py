"""Attach to a preloaded, exact four-container diagnostic. Never load it late."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid

from .metadata_arrays import MetadataArrayStateV1, _DlInfo

U64, U32, I32, I64 = C.c_uint64, C.c_uint32, C.c_int32, C.c_int64


def fields(names, dtype=U64):
    return [(name, dtype) for name in names.split()]


class Config(C.Structure):
    _fields_ = fields('version size', U32) + fields('owner_hi owner_lo generation thread max_calls max_bytes')


class Scope(C.Structure):
    _fields_ = fields('version size', U32) + fields('token forward operator_sequence boundary_sequence stage thread')


class Tensor(C.Structure):
    _fields_ = (fields('descriptor address') + [('counts', U64 * 3)]
        + fields('called_mask null_mask truncated_mask') + fields('offset', I64)
        + [(name, I64 * 8) for name in ('shape', 'strides', 'storage')]
        + fields('dtype format', I32) + [('statuses', I32 * 7), ('complete', U32)])


class Event(C.Structure):
    _fields_ = fields('version size operation phase', U32) + fields(
        'sequence call registration scope_token forward operator_sequence boundary_sequence stage '
        'pid thread depth function function_base function_offset caller caller_base caller_offset '
        'caller_kind handle count payload_offset payload_bytes original_calls monotonic_ns error') + fields('status reserved', I32)


class State(C.Structure):
    _fields_ = fields('version size', U32) + fields(
        'installed stopped retired owner_hi owner_lo generation owner_thread scope_token total_calls total_events '
        'retained_events metadata_bytes live_registrations active_calls overflow first_error abi_exceptions '
        'getter_capture_failures outside_scope_calls foreign_scope_thread_calls record_capacity_bytes '
        'payload_capacity_bytes registration_count elapsed_ns resolver_ready') + [
        (name, U64 * 4) for name in ('entries', 'returns', 'original_calls')] + [('arrays', MetadataArrayStateV1)]


class Sizes(C.Structure):
    _fields_ = fields('version size', U32) + fields('config scope tensor event state')


def instance(cls):
    value = cls(); value.version = 1; value.size = C.sizeof(cls)
    return value


def record(value):
    if isinstance(value, C.Structure):
        return {name: record(getattr(value, name)) for name, _ in value._fields_}
    if isinstance(value, C.Array):
        return [record(item) for item in value]
    return value


def require(condition, reason):
    if not condition:
        raise RuntimeError('CONTAINER_TRACE_' + reason)


def validate_bundle(root, expected):
    root = Path(root).resolve(strict=True); raw = (root / 'build.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'BUILD_SHA_CHANGED')
    build = json.loads(raw)
    require(build.get('schema') == 'demoncat.quant.container_deployment.v1'
        and build.get('actual_npu_calls') == 0 and build.get('qualified') is False,
        'READONLY_BUILD_REQUIRED')
    for name, sha in build['files'].items():
        path = (root / name).resolve(strict=True)
        require(path.is_relative_to(root) and hashlib.sha256(path.read_bytes()).hexdigest() == sha,
                'BUNDLE_FILE_CHANGED:' + name)
    profile = json.loads((root / 'profile.json').read_bytes())
    ident = profile.pop('profile_sha256')
    require(ident == build['profile_sha256'] == hashlib.sha256(
        json.dumps(profile, sort_keys=True, separators=(',', ':')).encode()).hexdigest(), 'PROFILE_CHANGED')
    return root, build


class NativeContainers:
    def __init__(self, root, expected):
        require(hasattr(os, 'RTLD_NOLOAD'), 'LINUX_NOLOAD_REQUIRED')
        root, self.build = validate_bundle(root, expected)
        self.manifest_sha256 = expected
        self.path = (root / self.build['library']).resolve(strict=True)
        # An explicit existing handle, not arbitrary scans of overlayfs images.
        self.library = C.CDLL(str(self.path), mode=os.RTLD_NOLOAD | os.RTLD_NOW | os.RTLD_LOCAL)
        locate = C.CDLL(None).dladdr
        locate.argtypes, locate.restype = [C.c_void_p, C.POINTER(_DlInfo)], C.c_int
        self.functions = {}
        signatures = dict(marker=([], U32), profile=([], C.c_char_p), sizes=([C.POINTER(Sizes)], C.c_int),
            state=([C.POINTER(State)], C.c_int), begin=([C.POINTER(Config)], C.c_int),
            scope_begin=([C.POINTER(Config), C.POINTER(Scope)], C.c_int),
            scope_end=([C.POINTER(Config), U64], C.c_int), stop=([C.POINTER(Config)], C.c_int),
            event=([U64, C.POINTER(Event)], C.c_int), payload=([U64, U64, C.c_void_p], C.c_int),
            release=([C.POINTER(Config)], C.c_int))
        base = None
        for name, (args, result) in signatures.items():
            fn = getattr(self.library, 'dcat_qct_' + name + '_v1'); fn.argtypes, fn.restype = args, result
            info = _DlInfo()
            require(locate(C.cast(fn, C.c_void_p), C.byref(info)) and info.dli_fname
                and Path(os.fsdecode(info.dli_fname)).resolve(strict=True) == self.path,
                'ACTUAL_FUNCTION_OWNER_CHANGED:' + name)
            base = info.dli_fbase if base is None else base
            require(info.dli_fbase == base, 'SPLIT_CONTROL_OWNER')
            self.functions[name] = fn
        self.mapping = dict(path=str(self.path), base=base, sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())
        self._mapped_identity()
        require(self.functions['marker']() == 1
                and self.functions['profile']().decode() == self.build['profile_sha256'], 'LOADED_PROFILE_CHANGED')
        sizes = instance(Sizes); self._call('sizes', C.byref(sizes))
        require(all(getattr(sizes, name) == C.sizeof(cls) for name, cls in
                    dict(config=Config, scope=Scope, tensor=Tensor, event=Event, state=State).items()),
                'SDK_ABI_LAYOUT_CHANGED')
        self.pid, self.thread = os.getpid(), threading.get_native_id()
        nonce = uuid.uuid4().int; self.config = instance(Config)
        self.config.owner_hi, self.config.owner_lo = nonce >> 64, nonce & ((1 << 64) - 1)
        self.config.generation, self.config.thread = 1, self.thread
        self.config.max_calls, self.config.max_bytes = 4096, 8 * 1024 * 1024
        self.scopes = []; self.active_scope = None; self.started = False

    def _mapped_identity(self):
        stat = self.path.stat(); matched = False
        for line in Path('/proc/self/maps').read_text().splitlines():
            parts = line.split(maxsplit=5)
            if len(parts) != 6 or parts[5].removesuffix(' (deleted)') != str(self.path):
                continue
            require(not parts[5].endswith(' (deleted)') and int(parts[4]) == stat.st_ino
                and tuple(int(x, 16) for x in parts[3].split(':')) == (os.major(stat.st_dev), os.minor(stat.st_dev)),
                'OWNED_MAPPING_FILE_IDENTITY_CHANGED')
            matched = True
        require(matched, 'OWNED_BIND_MOUNT_MAPPING_REQUIRED')

    def _call(self, name, *args):
        status = self.functions[name](*args)
        require(status == 0, name.upper() + '_FAILED:' + str(status))

    def current(self):
        require(os.getpid() == self.pid and threading.get_native_id() == self.thread, 'WORKER_OR_THREAD_CHANGED')
        self._mapped_identity()
        require(hashlib.sha256(self.path.read_bytes()).hexdigest() == self.mapping['sha256'], 'MAPPED_FILE_CHANGED')

    def query(self):
        value = instance(State); self._call('state', C.byref(value)); return record(value)

    def begin(self):
        self.current(); require(not self.started, 'ONE_DECODE_DIAGNOSTIC_ONLY')
        self._call('begin', C.byref(self.config)); self.started = True

    def enter(self, *, forward, sequence, boundary, stage, evidence):
        # Scope is a current source observation, not admission/target authority.
        require(self.started and self.active_scope is None, 'OVERLAPPING_PUBLIC_SCOPE')
        require(os.getpid() == self.pid and threading.get_native_id() == self.thread, 'SCOPE_WORKER_CHANGED')
        value = instance(Scope); value.token = len(self.scopes) + 1
        value.forward, value.operator_sequence = forward, sequence
        value.boundary_sequence, value.stage, value.thread = boundary, stage, self.thread
        row = dict(native_scope=record(value), evidence=json.loads(json.dumps(evidence)), status='ENTERED')
        self.scopes.append(row); self.active_scope = row
        try:
            self._call('scope_begin', C.byref(self.config), C.byref(value))
        except BaseException:
            row['status'] = 'SCOPE_BEGIN_FAILED'
            if self.query()['scope_token'] == 0:
                self.active_scope = None
            raise

    def leave(self, *, original_calls, error=None):
        require(self.active_scope is not None, 'MISSING_PUBLIC_SCOPE')
        self.active_scope.setdefault('original_calls', original_calls)
        self.active_scope.setdefault('original_error', error)
        attempt = dict(observed_original_calls=original_calls, observed_error=error, native_status=None)
        self.active_scope.setdefault('leave_attempts', []).append(attempt)
        status = self.functions['scope_end'](C.byref(self.config), self.active_scope['native_scope']['token'])
        attempt['native_status'] = status
        require(status == 0, 'SCOPE_END_FAILED:' + str(status))
        self.active_scope['status'] = 'RETURNED' if self.active_scope['original_error'] is None else 'ORIGINAL_FAILED'
        self.active_scope = None

    def snapshot(self):
        before = self.query(); require(not before['active_calls'], 'CALL_STILL_ACTIVE')
        events = []
        for index in range(before['retained_events']):
            value = instance(Event); self._call('event', index, C.byref(value)); row = record(value)
            size = value.payload_bytes
            if size:
                require(size <= 8 * 1024 * 1024, 'PAYLOAD_BOUND_CHANGED')
                raw = C.create_string_buffer(size)
                self._call('payload', value.payload_offset, size, raw)
                if value.operation == 1:
                    require(size == value.count * C.sizeof(Tensor), 'TENSOR_PAYLOAD_LAYOUT_CHANGED')
                    row['tensor_descriptors'] = [record(Tensor.from_buffer_copy(raw.raw, n * C.sizeof(Tensor)))
                                                 for n in range(value.count)]
                elif value.operation == 3:
                    require(size == value.count * 8, 'INTEGER_PAYLOAD_LAYOUT_CHANGED')
                    row['integer_values'] = list((I64 * value.count).from_buffer_copy(raw.raw))
                else:
                    raise RuntimeError('CONTAINER_TRACE_UNEXPECTED_PAYLOAD')
            events.append(row)
        after = self.query()
        require(all(before[key] == after[key] for key in ('total_calls', 'total_events', 'retained_events',
            'metadata_bytes', 'live_registrations', 'active_calls', 'scope_token')), 'SNAPSHOT_NOT_QUIESCENT')
        return dict(schema='demoncat.quant.container_trace.v1', mapping=self.mapping,
            build_sha256=self.manifest_sha256, profile_sha256=self.build['profile_sha256'], state=after, events=events,
            public_scopes=json.loads(json.dumps(self.scopes)), native_qualified=False, request_rows_proven=False,
            resource_scope='OWNED_TRACE_BUFFERS_REGISTRATIONS_SCOPES_AND_SDK_GETTER_ARRAYS',
            preloaded_library_remains_resident=True,
            scope='CREATOR_DESTROY_OBSERVABILITY_ONLY_NOT_GWS_OR_EXECUTOR_PROOF')

    def stop(self):
        self.current(); self._call('stop', C.byref(self.config))

    def release(self):
        self.current(); self._call('release', C.byref(self.config))
        after = self.query()
        require(not any(after[key] for key in ('installed', 'active_calls', 'scope_token', 'live_registrations',
            'record_capacity_bytes', 'payload_capacity_bytes')) and after['arrays']['live_arrays'] == 0,
            'OBSERVER_RESOURCES_NOT_ZERO')
        return after


def assess_observation(value):
    """Pure diagnostic gate; never returns row/Native qualification authority."""
    reasons = []; state = value['state']; events = value['events']; scopes = value['public_scopes']
    def check(condition, reason):
        if not condition:
            reasons.append(reason)
    check(value.get('schema') == 'demoncat.quant.container_trace.v1', 'SCHEMA')
    check(state['installed'] == state['stopped'] == 1 and not any(state[key] for key in
        ('active_calls', 'scope_token', 'live_registrations', 'first_error', 'abi_exceptions',
         'overflow', 'getter_capture_failures', 'foreign_scope_thread_calls')), 'NATIVE_OBSERVATION_NOT_COMPLETE_AND_IDLE')
    check(state['total_events'] == state['retained_events'] == len(events) == state['total_calls'] * 2,
          'INDEPENDENT_EVENT_DENOMINATOR_INCOMPLETE')
    check(state['entries'] == state['returns'] == state['original_calls']
          and all(count > 0 for count in state['entries']), 'FOUR_REAL_FUNCTIONS_NOT_OBSERVED')
    arrays = state['arrays']
    check(arrays['acquired_arrays'] == arrays['released_arrays'] > 0 and not any(arrays[key] for key in
        ('active_captures', 'active_getters', 'live_arrays', 'getter_failures', 'getter_exceptions',
         'capture_exceptions', 'ownership_errors')), 'GETTER_ARRAY_ACCOUNTING_FAILED')
    by_scope = {row['native_scope']['token']: row for row in scopes}
    check(len(by_scope) == len(scopes), 'DUPLICATE_PUBLIC_SCOPE')
    calls = {}; live = {}; stock_lists = custom_ints = 0
    for index, event in enumerate(events, 1):
        check(event['sequence'] == index, 'EVENT_ORDER')
        check(event['error'] == event['status'] == 0, 'NATIVE_CALL_FAILED')
        check(event['function'] > 0 and event['caller'] > 0, 'ACTUAL_FUNCTION_OR_CALLER_PC_MISSING')
        calls.setdefault(event['call'], []).append(event)
        if event['scope_token']:
            row = by_scope.get(event['scope_token'])
            check(row is not None, 'UNKNOWN_SCOPE')
            if row is not None:
                check(all(event[key] == row['native_scope'][key] for key in
                    ('forward', 'operator_sequence', 'boundary_sequence', 'stage', 'thread')), 'CURRENT_SCOPE_IDENTITY_CHANGED')
                check(event['caller_kind'] == (2 if event['stage'] == 1 else 1), 'ACTUAL_CALLER_IMAGE_UNPROVEN')
        if event['phase'] != 2:
            continue
        if event['operation'] in (1, 3):
            check(event['handle'] != 0 and event['registration'] == event['call']
                  and event['registration'] not in live, 'FRESH_CREATOR_REGISTRATION_REQUIRED')
            live[event['registration']] = event
            stock_lists += int(event['operation'] == 1 and event['stage'] == 3 and event['caller_kind'] == 1)
            custom_ints += int(event['operation'] == 3 and event['stage'] == 1 and event['caller_kind'] == 2)
        elif event['handle']:
            created = live.pop(event['registration'], None)
            check(created is not None, 'UNOBSERVED_OR_DUPLICATE_DESTROY')
            if created is not None:
                check(created['handle'] == event['handle'] and created['operation'] + 1 == event['operation'],
                      'DESTROY_REGISTRATION_IDENTITY_CHANGED')
    for pair in calls.values():
        check(len(pair) == 2 and [x['phase'] for x in pair] == [1, 2]
              and [x['original_calls'] for x in pair] == [0, 1], 'ORIGINAL_CALL_PAIR_INCOMPLETE')
        if len(pair) == 2:
            check(all(pair[0][key] == pair[1][key] for key in ('operation', 'pid', 'thread',
                'scope_token', 'function', 'caller', 'caller_kind')), 'CALL_PAIR_IDENTITY_CHANGED')
    check(not live, 'RETAINED_CONTAINER_LIFETIMES')
    check(stock_lists > 0 and custom_ints > 0, 'REQUIRED_NATURAL_STOCK_LIST_AND_CUSTOM_RANGE_ENTRY_NOT_SEEN')
    for row in scopes:
        check(row['status'] == 'RETURNED' and row.get('original_calls') == 1
              and row.get('original_error') is None, 'PUBLIC_SCOPE_INCOMPLETE')
        if row['native_scope']['stage'] in (1, 3):
            operation = 3 if row['native_scope']['stage'] == 1 else 1
            check(any(e['scope_token'] == row['native_scope']['token'] and e['operation'] == operation
                      and e['phase'] == 2 for e in events), 'NATURAL_SCOPE_CREATOR_BYPASSED')
    check({row['native_scope']['stage'] for row in scopes} == {1, 2, 3, 4}, 'FOUR_NATURAL_STAGES_NOT_OBSERVED')
    return dict(status='FOUR_CONTAINER_OBSERVABILITY_COMPLETE' if not reasons else 'OBSERVABILITY_REFUSED',
        reasons=sorted(set(reasons)), stock_gmm2_list_creations=stock_lists, custom_route_range_creations=custom_ints,
        original_calls=state['original_calls'], qualified=False, request_rows_proven=False, c1=False, c2=False, c3=False)
