"""One preloaded WeightNz diagnostic, with no row or fault admission authority."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import threading
import weakref

from . import quant_container_native as base

U64, U32, I32, I64 = base.U64, base.U32, base.I32, base.I64
fields, instance, record = base.fields, base.instance, base.record
PAIR = ('aclnnGroupedMatmulWeightNzGetWorkspaceSize', 'aclnnGroupedMatmulWeightNz')
FORMALS = ('x', 'weight', 'bias', 'scale', 'offset', 'antiquantScale', 'antiquantOffset',
    'perTokenScale', 'groupList', 'activationInput', 'activationQuantScale', 'activationQuantOffset',
    'splitItem', 'groupType', 'groupListType', 'actType', 'tuningConfig', 'quantGroupSize',
    'out', 'activationFeatureOut', 'dynQuantScaleOut', 'workspaceSize', 'executor')
CONTAINERS = (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 16, 18, 19, 20)
FUNCTIONS = {
    'prepare_function': ('opapi', PAIR[0], 29002704),
    'execute_function': ('opapi', PAIR[1], 28878640),
    'device_function': ('device_runtime', 'aclrtGetDevice', 42824),
    'list_size_function': ('nn', 'aclGetTensorListSize', 1089552),
    'int_size_function': ('nn', 'aclGetIntArraySize', 1089336),
}
CALLER_OFFSETS = {1: 0x175bc10, 2: 0x1747cb4}
EXTENSION = dict(version=1, typed_pair=list(PAIR), max_calls=128, max_pairs=32, max_bytes=1024*1024)
_startup_trace = None


def require(condition, reason):
    if not condition:
        raise RuntimeError('GMM2_TRACE_' + reason)


def clone(value):
    return json.loads(json.dumps(value, allow_nan=False))


def process_identity():
    """Same /proc identity as the standard Worker; no fallback nonce is accepted."""
    pid = os.getpid()
    stat = Path('/proc/self/stat').read_text(encoding='utf-8')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text(encoding='utf-8').strip()
    start = stat.rsplit(') ', 1)[1].split()[19]
    require(stat.split(' ', 1)[0] == str(pid) and boot and start.isdecimal(), 'ACTUAL_PROCESS_START_REQUIRED')
    return dict(pid=pid, thread=threading.get_native_id(), process_start_identity=f'proc:{boot}:{pid}:{start}')


class Limits(C.Structure):
    _fields_ = fields('version size', U32) + fields('max_calls max_pairs max_bytes')


class Slot(C.Structure):
    _fields_ = (fields('formal_index kind', U32)
        + fields('handle registration creator_entry creator_return count payload_offset payload_bytes actual_count joined live_at_return')
        + fields('size_status', I32) + fields('reserved', U32))


class Event(C.Structure):
    _fields_ = (fields('version size operation phase', U32) + fields(
        'sequence call prepare_registration owner_hi owner_lo generation scope_token forward operator_sequence '
        'boundary_sequence stage pid thread function function_base function_offset caller caller_base caller_offset '
        'original_calls container_events monotonic_ns error') + fields('device device_status status', I32)
        + fields('reserved', U32) + [('formals', U64*23), ('scalars', I64*5), ('containers', Slot*15),
                                    ('group_list', base.Tensor)]
        + fields('workspace workspace_bytes executor stream'))


class State(C.Structure):
    _fields_ = fields('version size', U32) + fields(
        'installed stopped retired owner_hi owner_lo generation thread total_calls total_events retained_events '
        'active_calls pending_executors untracked_executors overflow first_error abi_exceptions outside_scope_calls '
        'foreign_thread_calls getter_failures record_capacity_bytes elapsed_ns resolver_ready prepares prepare_returns '
        'executes execute_returns original_prepares original_executes paired_execute_returns failed_with_executor '
        'prepare_function execute_function device_function list_size_function int_size_function')


class Sizes(C.Structure):
    _fields_ = fields('version size', U32) + fields('limits slot event state')


def validate_extension(root, expected):
    root, build = base.validate_bundle(root, expected)
    profile = json.loads((root / 'profile.json').read_bytes())
    require(build.get('gmm2_extension') == profile.get('gmm2_extension') == EXTENSION,
            'EXPLICIT_NEW_TYPED_BUILD_REQUIRED')
    require(set(FUNCTIONS[key][0] for key in FUNCTIONS).issubset(profile.get('images', {})),
            'EXACT_ORIGINAL_IMAGE_PROFILES_REQUIRED')
    return root, build, profile


def function_owner(address, expected, *, symbol, offset):
    """dladdr the resolved pointer, then canonicalize and hash the actual file.

    Overlayfs system-map device numbers can differ from stat. They are not a
    file replacement test; Native's pinned executable segments are the loaded
    code check. The diagnostic's own read-only bind mount has a stricter check.
    """
    require(type(address) is int and address > 0, 'RESOLVED_POINTER_MISSING:' + symbol)
    locate = C.CDLL(None).dladdr
    locate.argtypes, locate.restype = [C.c_void_p, C.POINTER(base._DlInfo)], C.c_int
    info = base._DlInfo()
    require(locate(C.c_void_p(address), C.byref(info)) and info.dli_fname and info.dli_fbase,
            'RESOLVED_POINTER_OWNER_MISSING:' + symbol)
    path = Path(os.fsdecode(info.dli_fname)).resolve(strict=True)
    wanted = Path(expected['path']).resolve(strict=True)
    raw = path.read_bytes()
    actual_offset = address - info.dli_fbase
    require(path == wanted and len(raw) == expected['file_bytes']
        and hashlib.sha256(raw).hexdigest() == expected['file_sha256'], 'RESOLVED_IMAGE_CHANGED:' + symbol)
    require(actual_offset == offset and info.dli_sname and os.fsdecode(info.dli_sname) == symbol
        and info.dli_saddr == address and any(segment['vaddr'] <= actual_offset < segment['vaddr'] + segment['size']
        for segment in expected['segments']), 'RESOLVED_TYPED_SYMBOL_CHANGED:' + symbol)
    return dict(symbol=symbol, address=address, base=info.dli_fbase, offset=actual_offset,
        displayed_path=os.fsdecode(info.dli_fname), canonical_path=str(path), file_bytes=len(raw),
        file_sha256=hashlib.sha256(raw).hexdigest(), identity='CANONICAL_FILE_AND_NATIVE_PINNED_EXECUTABLE_SEGMENTS')


class NativeGmm2(base.NativeContainers):
    def __init__(self, root, expected):
        root, _, self.profile = validate_extension(root, expected)
        super().__init__(root, expected)
        locate = C.CDLL(None).dladdr
        locate.argtypes, locate.restype = [C.c_void_p, C.POINTER(base._DlInfo)], C.c_int
        self.gfunctions = {}
        signatures = dict(marker=([], U32), resolve=([], C.c_int),
            sizes=([C.POINTER(Sizes)], C.c_int), state=([C.POINTER(State)], C.c_int),
            begin=([C.POINTER(base.Config), C.POINTER(Limits)], C.c_int),
            event=([U64, C.POINTER(Event)], C.c_int), stop=([C.POINTER(base.Config)], C.c_int),
            release=([C.POINTER(base.Config)], C.c_int))
        for name, (args, result) in signatures.items():
            fn = getattr(self.library, 'dcat_qct_gmm2_' + name + '_v1'); fn.argtypes, fn.restype = args, result
            info = base._DlInfo()
            require(locate(C.cast(fn, C.c_void_p), C.byref(info)) and info.dli_fname
                and Path(os.fsdecode(info.dli_fname)).resolve(strict=True) == self.path
                and info.dli_fbase == self.mapping['base'], 'CONTROL_OWNER_CHANGED:' + name)
            self.gfunctions[name] = fn
        require(self.gfunctions['marker']() == 1, 'TYPED_MARKER_CHANGED')
        sizes = instance(Sizes); self._gcall('sizes', C.byref(sizes))
        require(all(getattr(sizes, name) == C.sizeof(cls) for name, cls in
                    dict(limits=Limits, slot=Slot, event=Event, state=State).items()), 'SDK_TYPED_LAYOUT_CHANGED')
        self.limits = instance(Limits)
        self.limits.max_calls, self.limits.max_pairs, self.limits.max_bytes = 128, 32, 1024*1024
        self.startup = self.worker_identity = None
        self.worker_refs = None
        self.failures = []

    def failure(self, action, error):
        self.failures.append(dict(sequence=len(self.failures) + 1, action=action, error=repr(error)))

    def _gcall(self, name, *args):
        status = self.gfunctions[name](*args)
        require(status == 0, name.upper() + '_FAILED:' + str(status))

    def query_gmm2(self):
        value = instance(State); self._gcall('state', C.byref(value)); return record(value)

    def query_all(self):
        # Pure state readback: no resolution, callbacks, collect, install or retry.
        return dict(containers=self.query(), gmm2=self.query_gmm2())

    def preflight(self):
        self.current()
        require(self.startup is None and not self.started, 'ONE_PRE_LLM_PREFLIGHT_REQUIRED')
        require(os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING') == '0'
            and os.environ.get('TASK_QUEUE_ENABLE') == '0', 'SAME_PROCESS_SYNCHRONOUS_WORKER_REQUIRED')
        identity = process_identity()
        before = self.query_all()
        require(not any(before['containers'][key] for key in ('installed', 'retired', 'total_calls', 'first_error'))
            and not any(before['gmm2'][key] for key in ('installed', 'retired', 'total_calls', 'first_error')),
            'FRESH_PRE_LLM_NATIVE_STATE_REQUIRED')
        self._gcall('resolve')  # NOLOAD lookup/identity only. No aclrtGetDevice call.
        after = self.query_all(); state = after['gmm2']
        require(state['resolver_ready'] == 1 and not state['first_error'] and not state['abi_exceptions']
            and state['total_calls'] == state['installed'] == state['record_capacity_bytes'] == 0,
            'STARTUP_ORIGINAL_RESOLUTION_FAILED')
        owners = {key: function_owner(state[key], self.profile['images'][image], symbol=symbol, offset=offset)
                  for key, (image, symbol, offset) in FUNCTIONS.items()}
        self.startup = dict(schema='demoncat.quant.gmm2_startup.v1', status='RESOLVED_BEFORE_LLM',
            identity=identity, native_build_sha256=self.manifest_sha256, native_profile_sha256=self.build['profile_sha256'],
            native_compile_source_commit=self.build['source_commit'],
            original_functions=owners, state_before=before, state_after=after, operator_calls=0,
            explicit_device_query_calls=0, install_calls=0,
            loader_constructor_driver_calls='NOT_COUNTED_MAY_PROBE_DEVICE_INFORMATION')
        return clone(self.startup)

    def bind_worker(self, worker):
        require(self.startup is not None and self.worker_identity is None, 'ACTUAL_PRE_LLM_HANDLE_REQUIRED')
        self.current(); require(process_identity() == self.startup['identity'], 'PREFLIGHT_PROCESS_OR_THREAD_CHANGED')
        identity = worker.demoncat_worker_identity()
        value = identity.get('worker', {})
        require(identity.get('status') == 'WORKER_IDENTIFIED' and not identity.get('quarantined')
            and value.get('pid') == self.pid and value.get('process_start_identity') == self.startup['identity']['process_start_identity']
            and value.get('rank') == 0 and type(value.get('logical_device')) is int and value['logical_device'] >= 0
            and value['logical_device'] == value.get('observed_logical_device') == value.get('configured_logical_device')
            and value.get('worker_start_identity') and value.get('model_instance_id'), 'ACTUAL_SAME_PROCESS_WORKER_REQUIRED')
        runner = worker.model_runner; model = runner.model
        self.worker_refs = tuple(weakref.ref(item) for item in (worker, runner, model))
        self.worker_identity = clone(identity)
        return clone(identity)

    def verify_worker(self, worker):
        require(self.worker_identity is not None and self.worker_refs is not None, 'HELD_WORKER_REQUIRED')
        require(process_identity() == self.startup['identity'], 'PREFLIGHT_PROCESS_OR_THREAD_CHANGED')
        actual = (worker, worker.model_runner, worker.model_runner.model)
        require(all(ref() is item for ref, item in zip(self.worker_refs, actual)), 'HELD_WORKER_RUNNER_OR_MODEL_CHANGED')
        require(worker.demoncat_worker_identity() == self.worker_identity, 'ACTUAL_WORKER_IDENTITY_CHANGED')
        state = self.query_gmm2()
        require(state['resolver_ready'] == 1 and all(state[key] == row['address']
            for key, row in self.startup['original_functions'].items()), 'STARTUP_RESOLVED_POINTER_CHANGED')

    def begin(self):
        require(self.startup is not None and self.worker_identity is not None, 'BOUND_PREFLIGHT_WORKER_REQUIRED')
        require(process_identity() == self.startup['identity'], 'PREFLIGHT_PROCESS_OR_THREAD_CHANGED')
        # A failed second begin leaves the first journal's owner reachable.
        super().begin()
        try:
            self._gcall('begin', C.byref(self.config), C.byref(self.limits))
        except BaseException as error:
            self.failure('begin_gmm2', error)
            raise

    def stop(self):
        self.current()
        if self.query_gmm2()['installed']:
            self._gcall('stop', C.byref(self.config))
        super().stop()

    def snapshot(self):
        before = self.query_gmm2(); require(not before['active_calls'], 'GMM2_CALL_STILL_ACTIVE')
        value = super().snapshot()
        value['mapping'] = clone(value['mapping'])
        events = []
        require(before['retained_events'] <= 256, 'RETAINED_EVENT_BOUND_CHANGED')
        for index in range(before['retained_events']):
            event = instance(Event); self._gcall('event', index, C.byref(event)); events.append(record(event))
        after = self.query_gmm2()
        require(all(before[key] == after[key] for key in before if key != 'elapsed_ns'), 'GMM2_SNAPSHOT_NOT_QUIESCENT')
        value['gmm2'] = dict(schema='demoncat.quant.gmm2_trace.v1', state=after, events=events,
            formal_names=list(FORMALS), startup=clone(self.startup), worker_identity=clone(self.worker_identity),
            failures=clone(self.failures), original_execute_return_is_not_device_drain=True)
        value['scope'] = 'FOUR_CONTAINER_LIFETIMES_AND_CURRENT_WEIGHTNZ_CALL_SLOTS_NOT_REQUEST_ROW_AUTHORITY'
        return value

    def release(self):
        self.current()
        if self.query_gmm2()['installed']:
            self._gcall('release', C.byref(self.config))
        state = self.query_gmm2()
        require(not any(state[key] for key in ('installed', 'active_calls', 'pending_executors',
            'untracked_executors', 'record_capacity_bytes')), 'GMM2_RESOURCES_NOT_ZERO')
        containers = super().release()
        self.worker_refs = None
        return dict(containers=containers, gmm2=state, resource_scope='OBSERVER_HOST_JOURNALS_ONLY_NO_DEVICE_BUFFERS',
                    device_drain_proven=False, preloaded_library_remains_resident=True)


def prepare_startup(root, expected):
    """Keep this exact pre-LLM object in the installed module, not an RPC copy."""
    global _startup_trace
    require(_startup_trace is None, 'EXCLUSIVE_PRE_LLM_HANDLE_REQUIRED')
    value = NativeGmm2(root, expected)
    _startup_trace = value  # Retain partial failure evidence for the caller.
    return value.preflight()


def startup_trace():
    require(_startup_trace is not None, 'SAME_PROCESS_PRE_LLM_HANDLE_MISSING')
    return _startup_trace


def discard_startup(value):
    global _startup_trace
    require(_startup_trace is value, 'PRE_LLM_HANDLE_OWNER_CHANGED')
    state = value.query_all()
    require(not any(state['containers'][key] for key in ('installed', 'active_calls', 'live_registrations',
        'scope_token', 'record_capacity_bytes', 'payload_capacity_bytes'))
        and not any(state['containers']['arrays'][key] for key in ('live_arrays', 'active_captures', 'active_getters'))
        and not any(state['gmm2'][key] for key in ('installed', 'active_calls', 'pending_executors',
        'untracked_executors', 'record_capacity_bytes')), 'RETAINED_DIAGNOSTIC_OWNER')
    value.worker_refs = None
    _startup_trace = None


def assess_observation(value, *, expected_pairs=26):
    """Rejoin raw journals. Complete means observed calls, never routed row proof."""
    try:
        four = base.assess_observation(value)
    except (KeyError, TypeError, ValueError, IndexError) as error:
        four = dict(status='OBSERVABILITY_REFUSED', reasons=['MALFORMED_CONTAINER_ORIGINALS:' + type(error).__name__],
                    qualified=False, request_rows_proven=False, c1=False, c2=False, c3=False)
    reasons = list(four['reasons']); joined_slots = 0
    def check(condition, reason):
        if not condition:
            reasons.append(reason)
    try:
        g = value['gmm2']; state = g['state']; events = g['events']; startup = g['startup']
        worker = g['worker_identity']['worker']; process = startup['identity']
        owners = startup['original_functions']
        check(g['schema'] == 'demoncat.quant.gmm2_trace.v1' and g['formal_names'] == list(FORMALS), 'TYPED_JOURNAL_SCHEMA')
        check(startup['status'] == 'RESOLVED_BEFORE_LLM' and process['pid'] == worker['pid']
            and process['process_start_identity'] == worker['process_start_identity']
            and worker['rank'] == 0 and worker['logical_device'] == worker['configured_logical_device']
            == worker['observed_logical_device'], 'HELD_PREFLIGHT_WORKER_IDENTITY')
        check(not g['failures'] and state['installed'] == state['stopped'] == state['resolver_ready'] == 1
            and not any(state[key] for key in ('active_calls', 'pending_executors', 'untracked_executors',
                'overflow', 'first_error', 'abi_exceptions', 'outside_scope_calls', 'foreign_thread_calls',
                'getter_failures', 'failed_with_executor')), 'TYPED_OBSERVATION_NOT_COMPLETE_AND_IDLE')
        check(state['total_events'] == state['retained_events'] == len(events) == expected_pairs*4
            and state['total_calls'] == expected_pairs*2 and all(state[key] == expected_pairs for key in
            ('prepares', 'prepare_returns', 'executes', 'execute_returns', 'original_prepares',
             'original_executes', 'paired_execute_returns')), 'TYPED_INDEPENDENT_DENOMINATOR')
        check(all(state[key] == owners[key]['address'] for key in FUNCTIONS), 'STARTUP_ORIGINAL_POINTERS_CHANGED')
        check(all(state[key] == value['state'][key] for key in ('owner_hi', 'owner_lo', 'generation'))
            and state['thread'] == process['thread'] == value['state']['owner_thread'], 'TYPED_GROUP_OWNER')
        scopes = {row['native_scope']['token']: row for row in value['public_scopes']}
        wanted_scopes = {token for token, row in scopes.items() if row['native_scope']['stage'] == 3}
        check(len(wanted_scopes) == expected_pairs, 'NATURAL_GMM2_WINDOW_DENOMINATOR')
        raw = {row['sequence']: row for row in value['events']}
        destroys = {}
        for row in value['events']:
            if row['operation'] in (2, 4) and row['phase'] == 1 and row['handle']:
                destroys.setdefault(row['registration'], []).append(row)
        pairs = {}; preparation = {}; executions = {}; scope_pairs = {}
        for sequence, event in enumerate(events, 1):
            check(event['sequence'] == sequence and event['call'] > 0, 'TYPED_EVENT_ORDER')
            check(event['error'] == event['status'] == event['device_status'] == 0
                and event['device'] == worker['logical_device'], 'ACTUAL_TYPED_CALL_OR_LOGICAL_DEVICE_FAILED')
            check(event['pid'] == process['pid'] and event['thread'] == process['thread']
                and all(event[key] == state[key] for key in ('owner_hi', 'owner_lo', 'generation')), 'TYPED_CALL_LIFETIME_CHANGED')
            check(event['operation'] in (1, 2) and event['phase'] in (1, 2), 'TYPED_OPERATION_OR_PHASE')
            operation = event['operation']
            original = owners['prepare_function' if operation == 1 else 'execute_function']
            check(all(event[key] == original[mapped] for key, mapped in
                (('function', 'address'), ('function_base', 'base'), ('function_offset', 'offset')))
                and event['caller_base'] > 0 and event['caller'] - event['caller_base'] == event['caller_offset']
                == CALLER_OFFSETS.get(operation), 'ACTUAL_TYPED_ORIGINAL_OR_REVIEWED_CALLER_CHANGED')
            scope = scopes.get(event['scope_token'])
            check(scope is not None and event['scope_token'] in wanted_scopes, 'TYPED_CURRENT_PUBLIC_SCOPE_MISSING')
            if scope is not None:
                check(all(event[key] == scope['native_scope'][key] for key in
                    ('forward', 'operator_sequence', 'boundary_sequence', 'stage', 'thread')), 'TYPED_PUBLIC_SCOPE_CHANGED')
            pairs.setdefault(event['call'], []).append(event)
        for pair in pairs.values():
            check(len(pair) == 2 and [row['phase'] for row in pair] == [1, 2]
                and [row['original_calls'] for row in pair] == [0, 1], 'EXACTLY_ONE_TYPED_ORIGINAL_REQUIRED')
            if len(pair) != 2:
                continue
            entry, returned = pair
            check(entry['sequence'] < returned['sequence'] and entry['monotonic_ns'] <= returned['monotonic_ns']
                and all(entry[key] == returned[key] for key in ('call', 'operation', 'prepare_registration',
                    'owner_hi', 'owner_lo', 'generation', 'scope_token', 'pid', 'thread', 'device', 'function',
                    'caller', 'forward', 'operator_sequence', 'boundary_sequence', 'stage')), 'TYPED_RETURN_IDENTITY_CHANGED')
            registration = entry['prepare_registration']
            if entry['operation'] == 2:
                executions.setdefault(registration, []).append(pair)
                check(all(entry[key] == returned[key] for key in ('workspace', 'workspace_bytes', 'executor', 'stream')),
                      'EXECUTE_ARGUMENTS_CHANGED')
                continue
            check(registration == entry['call'] and registration not in preparation, 'FRESH_PREPARE_REGISTRATION_REQUIRED')
            preparation[registration] = pair
            scope_pairs.setdefault(entry['scope_token'], []).append(registration)
            check(entry['formals'] == returned['formals'] and len(entry['formals']) == 23
                and entry['scalars'] == returned['scalars'] and len(entry['scalars']) == 5
                and all((entry['scalars'][i] & ((1 << 64)-1)) == entry['formals'][slot]
                        for i, slot in enumerate((12, 13, 14, 15, 17)))
                and entry['formals'][21] > 0 and entry['formals'][22] > 0, 'ACTUAL_23_FORMALS_REQUIRED')
            check([row['formal_index'] for row in entry['containers']] == list(CONTAINERS)
                and [row['formal_index'] for row in returned['containers']] == list(CONTAINERS), 'NAMED_CONTAINER_SLOTS_REQUIRED')
            for slot, end in zip(entry['containers'], returned['containers']):
                index = slot['formal_index']; kind = 3 if index == 16 else 1
                check(slot['handle'] == entry['formals'][index] and slot['kind'] == kind
                    and slot['joined'] == end['joined'] == end['live_at_return'] == 1
                    and slot['actual_count'] == slot['count'] and slot['size_status'] == 0
                    and all(slot[key] == end[key] for key in slot if key != 'live_at_return'), 'CURRENT_CONTAINER_SLOT_JOIN_FAILED')
                if not slot['handle']:
                    check(not any(slot[key] for key in ('registration', 'creator_entry', 'creator_return',
                        'count', 'actual_count', 'payload_offset', 'payload_bytes')), 'NULL_CONTAINER_INVENTED_REGISTRATION')
                    continue
                first, last = raw.get(slot['creator_entry']), raw.get(slot['creator_return'])
                death = destroys.get(slot['registration'], [])
                check(first is not None and last is not None and len(death) == 1, 'CURRENT_CONTAINER_ORIGINAL_LIFETIME_MISSING')
                if first is None or last is None or len(death) != 1:
                    continue
                check(first['phase'] == 1 and last['phase'] == 2 and first['operation'] == last['operation'] == kind
                    and first['call'] == last['call'] == last['registration'] == slot['registration']
                    and last['handle'] == death[0]['handle'] == slot['handle']
                    and first['sequence'] < last['sequence'] <= entry['container_events']
                    <= returned['container_events'] < death[0]['sequence'], 'CONTAINER_LIVE_INTERVAL_NOT_CURRENT_GWS')
                check(all(first[key] == last[key] == entry[key] for key in ('scope_token', 'pid', 'thread',
                    'forward', 'operator_sequence', 'boundary_sequence', 'stage'))
                    and all(first[key] == slot[key] for key in ('count', 'payload_offset', 'payload_bytes')),
                    'CONTAINER_SOURCE_OR_PAYLOAD_CHANGED')
                payload = first.get('tensor_descriptors', []) if kind == 1 else first.get('integer_values', [])
                check(len(payload) == slot['count'] and slot['payload_bytes'] == slot['count'] * (C.sizeof(base.Tensor) if kind == 1 else 8),
                      'CONTAINER_COPIED_PAYLOAD_INCOMPLETE')
                joined_slots += 1
            descriptor = entry['group_list']
            check(entry['group_list'] == returned['group_list'] and descriptor['descriptor'] == entry['formals'][8],
                  'CURRENT_GROUP_LIST_DESCRIPTOR_CHANGED')
            if entry['formals'][8]:
                check(descriptor['complete'] == 1 and descriptor['called_mask'] == 127
                    and descriptor['null_mask'] == descriptor['truncated_mask'] == 0
                    and all(status == 0 for status in descriptor['statuses']), 'GROUP_LIST_TYPED_GETTERS_INCOMPLETE')
        check(set(scope_pairs) == wanted_scopes and all(len(rows) == 1 for rows in scope_pairs.values()),
              'EXACTLY_ONE_PREPARE_PER_CURRENT_GMM2_WINDOW')
        check(len(preparation) == expected_pairs and set(preparation) == set(executions), 'PREPARE_EXECUTE_REGISTRATION_DENOMINATOR')
        for registration, (entry, returned) in preparation.items():
            executed = executions.get(registration, [])
            check(len(executed) == 1, 'EXACTLY_ONE_MATCHING_EXECUTE_REQUIRED')
            if len(executed) != 1:
                continue
            call, done = executed[0]
            check(returned['sequence'] < call['sequence'] < done['sequence'] and returned['executor'] > 0
                and returned['executor'] == call['executor'] == done['executor']
                and all(entry[key] == call[key] for key in ('scope_token', 'pid', 'thread', 'device',
                    'forward', 'operator_sequence', 'boundary_sequence', 'stage'))
                and call['workspace_bytes'] >= returned['workspace_bytes']
                and (returned['workspace_bytes'] == 0 or call['workspace'] > 0), 'CURRENT_EXECUTOR_WORKSPACE_SCOPE_JOIN_FAILED')
    except (KeyError, TypeError, ValueError, IndexError) as error:
        reasons.append('MALFORMED_TYPED_ORIGINALS:' + type(error).__name__)
    return dict(status='CURRENT_WEIGHTNZ_SLOTS_AND_EXECUTION_OBSERVED' if not reasons else 'OBSERVABILITY_REFUSED',
        reasons=sorted(set(reasons)), expected_pairs=expected_pairs, joined_nonnull_container_slots=joined_slots,
        four_container_assessment=four, qualified=False, request_rows_proven=False, c1=False, c2=False, c3=False,
        scope='CURRENT_TYPED_ENTRY_LIFETIME_ONLY_NOT_ROUTED_TOKEN_REQUEST_JOIN', device_drain_proven=False)
