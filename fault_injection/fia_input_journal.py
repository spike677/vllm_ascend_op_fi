"""Experimental read-only FIA V3 journal, explicitly attached before LLM startup.

This module neither installs an Attention hook nor manages a Worker. The caller
must own the real API window, retain its actual input objects, and write failures.
"""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import threading

from . import quant_container_native as containers
from .metadata_arrays import _DlInfo

U64, U32, I32, I64 = C.c_uint64, C.c_uint32, C.c_int32, C.c_int64
fields = containers.fields
TENSOR_INDICES = (0, 3, 4, *range(7, 23), 24, 25, 26, 40, 41)
CONTAINER_INDICES = (1, 2, 5, 6, 23)
FORMALS = ('query key value pse_shift atten_mask actual_seq_lengths actual_seq_lengths_kv deq_scale1 '
    'quant_scale1 deq_scale2 quant_scale2 quant_offset2 antiquant_scale antiquant_offset block_table '
    'query_padding_size kv_padding_size key_antiquant_scale key_antiquant_offset value_antiquant_scale '
    'value_antiquant_offset key_shared_prefix value_shared_prefix actual_shared_prefix_len query_rope key_rope '
    'key_rope_antiquant_scale num_heads scale pre_tokens next_tokens input_layout num_key_value_heads sparse_mode '
    'inner_precise block_size antiquant_mode softmax_lse_flag key_antiquant_mode value_antiquant_mode '
    'attention_out softmax_lse workspace_size executor').split()


class Limits(C.Structure):
    _fields_ = fields('version size', U32) + fields('max_calls max_pairs max_bytes max_windows')


class ScopeIdentity(C.Structure):
    _fields_ = fields('version size', U32) + [('scope', containers.Scope)] + fields('pid worker model module') + [('call_id', C.c_char * 33)]


class Slot(C.Structure):
    _fields_ = fields('formal kind', U32) + fields('handle registration creator_entry creator_return count actual_count '
        'payload_offset payload_bytes joined live_at_return') + fields('size_status', I32) + fields('reserved', U32)


class Event(C.Structure):
    _fields_ = fields('version size operation phase', U32) + fields(
        'sequence call prepare_registration owner_hi owner_lo generation scope_token forward operator_sequence boundary_sequence '
        'pid thread worker model module function function_base function_offset caller caller_base caller_offset '
        'monotonic_ns original_calls original_status_valid original_exception error executor workspace workspace_bytes') + [
        ('formals', U64 * 44), ('signed_scalars', I64 * 10), ('scale', C.c_double)] + fields('softmax_lse layout_length', U32) + [
        ('layout', C.c_char * 16), ('call_id', C.c_char * 33)] + fields('device device_status status', I32) + fields('reserved', U32) + [
        ('tensors', containers.Tensor * 24), ('containers', Slot * 5)]


class State(C.Structure):
    _fields_ = fields('version size', U32) + fields(
        'installed stopped retired owner_hi owner_lo generation pid thread total_calls total_events retained_events active_calls '
        'pending_executors untracked_executors overflow first_error exceptions foreign_thread_calls outside_scope_calls getter_failures '
        'prepare_returns execute_returns original_prepares original_executes paired_execute_returns failed_prepare_outputs_unknown '
        'record_capacity_bytes resolver_ready prepare_function execute_function device_function list_size_function '
        'int_size_function observer_scope_function window_limit accepted_windows retained_scope_tokens retained_scope_token_characters')


class Sizes(C.Structure):
    _fields_ = fields('version size', U32) + fields('limits scope slot event state') + fields('float16_tag int8_tag nd_tag reserved', I32)


def require(ok, reason):
    if not ok:
        raise RuntimeError('FIA_INPUT_JOURNAL_' + reason)


def record(value):
    if isinstance(value, C.Structure):
        return {name: record(getattr(value, name)) for name, _ in value._fields_}
    if isinstance(value, C.Array):
        return [record(item) for item in value]
    if isinstance(value, bytes):
        return value.decode('ascii')
    return value


def process_identity():
    require(os.name == 'posix', 'LINUX_PROCESS_IDENTITY_REQUIRED')
    # /proc stat comm may contain spaces and parentheses; starttime is field 22.
    stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=os.getpid(), native_tid=threading.get_native_id(),
        process_start_identity='proc:' + Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        + ':' + str(os.getpid()) + ':' + stat[19])


def public_tensor_metadata(value):
    """Actual storage/view addresses only: no clone, conversion or device read."""
    import torch
    require(isinstance(value, torch.Tensor) and value.layout == torch.strided, 'ACTUAL_STRIDED_TENSOR_REQUIRED')
    storage = value.untyped_storage()
    return dict(object_id=id(value), storage_identity=int(storage._cdata), storage_address=int(storage.data_ptr()),
        data_address=int(value.data_ptr()), storage_bytes=int(storage.nbytes()), storage_offset=int(value.storage_offset()),
        element_bytes=int(value.element_size()), shape=list(value.shape), stride=list(value.stride()),
        dtype=str(value.dtype), device=str(value.device))


class NativeFiaInputJournal:
    """One process, explicit begin, bounded actual API windows, no F methods."""
    MAX_LIFECYCLE_EVENTS = 64
    MAX_SCOPE_EVIDENCE_BYTES = 64*1024
    def __init__(self, root, build_sha256):
        self.containers = containers.NativeContainers(root, build_sha256)
        self.library = self.containers.library
        self.root = Path(root).resolve(strict=True)
        self.profile = json.loads((self.root / 'profile.json').read_text())
        require(self.profile.get('candidate') == 'fia_input_journal'
            and self.profile.get('slice_pointer_lineage_proven') is False, 'EXACT_INPUT_JOURNAL_BUNDLE_REQUIRED')
        self.functions = {}
        signatures = dict(marker=([], U32), resolve=([], C.c_int), sizes=([C.POINTER(Sizes)], C.c_int),
            begin=([C.POINTER(containers.Config), C.POINTER(Limits)], C.c_int),
            scope_begin=([C.POINTER(containers.Config), C.POINTER(ScopeIdentity)], C.c_int),
            scope_end=([C.POINTER(containers.Config), U64], C.c_int), state=([C.POINTER(State)], C.c_int),
            event=([U64, C.POINTER(Event)], C.c_int), stop=([C.POINTER(containers.Config)], C.c_int),
            release=([C.POINTER(containers.Config)], C.c_int))
        self.locate = C.CDLL(None).dladdr
        self.locate.argtypes, self.locate.restype = [C.c_void_p, C.POINTER(_DlInfo)], C.c_int
        for name, (args, result) in signatures.items():
            fn = getattr(self.library, 'dcat_fia_input_journal_' + name + '_v1'); fn.argtypes, fn.restype = args, result
            info = _DlInfo()
            require(self.locate(C.cast(fn, C.c_void_p), C.byref(info)) and info.dli_fname
                and Path(os.fsdecode(info.dli_fname)).resolve(strict=True) == self.containers.path
                and info.dli_fbase == self.containers.mapping['base'], 'CONTROL_OWNER_CHANGED:' + name)
            self.functions[name] = fn
        require(self.functions['marker']() == 1, 'MARKER_CHANGED')
        sizes = containers.instance(Sizes); self.call('sizes', C.byref(sizes))
        require(all(getattr(sizes, key) == C.sizeof(cls) for key, cls in
            dict(limits=Limits, scope=ScopeIdentity, slot=Slot, event=Event, state=State).items()), 'SDK_ABI_CHANGED')
        self.sdk_tags = {key: getattr(sizes, key) for key in ('float16_tag', 'int8_tag', 'nd_tag')}
        self.startup = None; self.bound_worker = None; self.started = False; self.active = None; self.scopes = []
        self.installation_attempted = False; self.window_limit = 0; self.limits = None
        self.lifecycle = []; self.first_lifecycle_error = None; self.lifecycle_overflow = 0

    def call(self, name, *args):
        status = self.functions[name](*args)
        require(status == 0, name.upper() + '_FAILED:' + str(status))

    def query(self):
        value = containers.instance(State); self.call('state', C.byref(value)); return record(value)

    def _states(self):
        return dict(fia=self.query(), containers=self.containers.query())

    def _phase(self, component, operation, action=None, before=None, after=None):
        """Keep actual return codes and both queried states, including failures."""
        if len(self.lifecycle) >= self.MAX_LIFECYCLE_EVENTS:
            self.lifecycle_overflow += 1
            require(False, 'LIFECYCLE_BUDGET_EXHAUSTED_OWNER_RETAINED')
        row = dict(sequence=len(self.lifecycle)+1, component=component, operation=operation,
            status='STARTED', native_status=None)
        self.lifecycle.append(row)
        try:
            self.containers.current(); row['before'] = self._states()
            if before is not None: before(row['before'])
            if action is not None:
                row['native_status'] = action()
                if component == 'containers' and operation == 'begin' and row['native_status'] == 0:
                    self.containers.started = True
                require(row['native_status'] == 0, 'LIFECYCLE_' + component.upper() + '_' + operation.upper()
                    + '_FAILED:' + str(row['native_status']))
            row['after'] = self._states()
            if after is not None: after(row['after'])
            row['status'] = 'RETURNED'
            return row['after']
        except BaseException as error:
            row.update(status='FAILED', error=repr(error))
            try: row['after'] = self._states()
            except BaseException as query_error: row['after_query_error'] = repr(query_error)
            if self.first_lifecycle_error is None:
                self.first_lifecycle_error = dict(sequence=row['sequence'], component=component,
                    operation=operation, error=repr(error), native_status=row['native_status'])
            raise

    def _native_phase(self, component, operation, *extra, before=None, after=None):
        functions = self.functions if component == 'fia' else self.containers.functions
        return self._phase(component, operation,
            lambda: functions[operation](C.byref(self.containers.config), *extra), before, after)

    def _returned_control(self, component, operation):
        return any(row['component'] == component and row['operation'] == operation
            and row['native_status'] == 0 for row in self.lifecycle)

    def _owners(self, states):
        config = self.containers.config
        for component, value in states.items():
            if value['installed'] or value['retired']:
                require(all(value[key] == getattr(config, key) for key in ('owner_hi','owner_lo','generation')),
                    'CLEANUP_OWNER_CHANGED:' + component)
                require(value['thread' if component == 'fia' else 'owner_thread'] == config.thread,
                    'CLEANUP_THREAD_CHANGED:' + component)
                if component == 'fia': require(value['pid'] == self.bound_worker['pid'], 'CLEANUP_PROCESS_CHANGED')

    @staticmethod
    def _fia_empty(value):
        return not any(value[key] for key in ('installed','active_calls','pending_executors','untracked_executors',
            'record_capacity_bytes','retained_events','retained_scope_tokens','retained_scope_token_characters'))

    @staticmethod
    def _containers_idle(value):
        return not any(value[key] for key in ('active_calls','scope_token','live_registrations')) and not any(
            value['arrays'][key] for key in ('active_captures','active_getters','live_arrays'))

    def _absent_fia(self, states):
        self._owners(states); value = states['fia']
        require(self._fia_empty(value), 'FIA_ABSENCE_NOT_PROVEN_OWNER_RETAINED')
        if value['retired']:
            require(self._returned_control('fia','release'), 'FIA_RETIREMENT_WITHOUT_OWN_RELEASE')
        else:
            require(value['total_calls'] == 0 and value['accepted_windows'] == 0,
                'FIA_NEVER_INSTALLED_NOT_PROVEN')
        require(self._containers_idle(states['containers']), 'BASE_NOT_IDLE_OWNER_RETAINED')

    def _absent_containers(self, states):
        self._owners(states); value = states['containers']
        require(not value['installed'] and self._containers_idle(value) and not any(value[key] for key in
            ('record_capacity_bytes','payload_capacity_bytes','retained_events')), 'BASE_ABSENCE_NOT_PROVEN')
        require(self._returned_control('containers','release') if value['retired'] else value['total_calls'] == 0,
            'BASE_RETIREMENT_WITHOUT_OWN_RELEASE')

    def _base_idle(self, states):
        self._owners(states)
        require(self._containers_idle(states['containers']), 'BASE_NOT_IDLE_OWNER_RETAINED')

    def preflight(self):
        require(self.startup is None and not self.started, 'ONE_PRE_LLM_PREFLIGHT_REQUIRED')
        self.containers.current(); self.call('resolve')
        state = self.query(); owners = {}
        for field, image in dict(prepare_function='opapi', execute_function='opapi', device_function='device_runtime',
                list_size_function='nn', int_size_function='nn', observer_scope_function='observer').items():
            address = state[field]; info = _DlInfo()
            require(address and self.locate(address, C.byref(info)) and info.dli_fname, 'ORIGINAL_FUNCTION_UNMAPPED:' + field)
            path = Path(os.fsdecode(info.dli_fname)).resolve(strict=True)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            require(digest == self.profile['originals'][image]['file_sha256'], 'ORIGINAL_OWNER_CHANGED:' + field)
            owners[field] = dict(address=address, base=info.dli_fbase, offset=address-info.dli_fbase, path=str(path), sha256=digest)
        self.startup = dict(schema='demoncat.fia_input_journal.startup.v1', identity=process_identity(),
            original_functions=owners, status='RESOLVED_BEFORE_LLM', explicit_operator_calls=0,
            explicit_device_queries=0, install_calls=0, loader_runtime_discovery_may_be_attempted=True)
        return json.loads(json.dumps(self.startup))

    def bind_worker(self, *, pid, process_start_identity, worker_object_id, model_object_id):
        require(self.startup is not None and self.bound_worker is None and not self.started, 'PRE_LLM_FIRST_REQUIRED')
        identity = globals()['process_identity']()
        require(identity == self.startup['identity'] and pid == identity['pid']
            and process_start_identity == identity['process_start_identity']
            and type(worker_object_id) is int and worker_object_id > 0
            and type(model_object_id) is int and model_object_id > 0, 'SAME_ACTUAL_WORKER_REQUIRED')
        self.bound_worker = dict(pid=pid, process_start_identity=process_start_identity,
            worker_object_id=worker_object_id, model_object_id=model_object_id)

    def begin(self, *, max_calls=128, max_pairs=64, max_bytes=16*1024*1024, max_windows=64):
        require(self.bound_worker is not None and not self.installation_attempted
            and process_identity() == self.startup['identity'], 'CURRENT_WORKER_REQUIRED')
        limits = containers.instance(Limits)
        require(all(type(v) is int and v > 0 for v in (max_calls, max_pairs, max_bytes, max_windows)), 'POSITIVE_INTEGER_BUDGET_REQUIRED')
        limits.max_calls, limits.max_pairs, limits.max_bytes, limits.max_windows = max_calls, max_pairs, max_bytes, max_windows
        require(max_calls <= 256 and max_pairs <= 128 and max_bytes <= 16*1024*1024
            and max_windows <= 256 and 2*max_calls*C.sizeof(Event) <= max_bytes, 'BOUNDED_RECORD_BUDGET_REQUIRED')
        self.installation_attempted = True; self.window_limit = max_windows; self.limits = record(limits)
        self._native_phase('containers','begin', before=self._owners,
            after=lambda states: require(states['containers']['installed'] == 1, 'BASE_INSTALLATION_UNCONFIRMED'))
        self._native_phase('fia','begin', C.byref(limits), before=self._owners,
            after=lambda states: require(states['fia']['installed'] == 1, 'FIA_INSTALLATION_UNCONFIRMED'))
        self.started = True

    def enter(self, *, call_id, module_object_id, forward, sequence, boundary, evidence):
        require(self.started and self.active is None and process_identity() == self.startup['identity'], 'NO_OVERLAPPING_OR_FOREIGN_WINDOW')
        require(len(self.scopes) < self.window_limit, 'PUBLIC_WINDOW_BUDGET_EXHAUSTED')
        require(type(call_id) is str and len(call_id) == 32 and all(c in '0123456789abcdef' for c in call_id), 'ACTUAL_LAUNCH_TOKEN_REQUIRED')
        require(all(type(n) is int and n > 0 for n in (module_object_id, forward, sequence, boundary)), 'ACTUAL_SCOPE_IDENTITIES_REQUIRED')
        value = containers.instance(ScopeIdentity); value.scope = containers.instance(containers.Scope)
        value.scope.token = len(self.scopes)+1; value.scope.forward = forward; value.scope.operator_sequence = sequence
        value.scope.boundary_sequence = boundary; value.scope.stage = 3; value.scope.thread = threading.get_native_id()
        value.pid = os.getpid(); value.worker = self.bound_worker['worker_object_id']; value.model = self.bound_worker['model_object_id']
        value.module = module_object_id; value.call_id = call_id.encode('ascii')
        serialized = json.dumps(evidence, allow_nan=False)
        require(len(serialized.encode('utf-8')) <= self.MAX_SCOPE_EVIDENCE_BYTES, 'PUBLIC_SCOPE_EVIDENCE_BUDGET_EXHAUSTED')
        row = dict(identity=record(value), evidence=json.loads(serialized), status='ENTERING')
        self.scopes.append(row); self.active = row
        try:
            self.call('scope_begin', C.byref(self.containers.config), C.byref(value)); row['status'] = 'ENTERED'
        except BaseException as error:
            row.update(status='ENTER_FAILED', error=repr(error))
            if not self.containers.query()['scope_token']: self.active = None
            raise

    def leave(self, *, original_calls, original_error=None):
        require(self.active is not None, 'MISSING_WINDOW')
        self.active.setdefault('original_calls', original_calls); self.active.setdefault('original_error', original_error)
        require(len(self.active.get('leave_statuses',[])) < self.MAX_LIFECYCLE_EVENTS, 'LEAVE_RETRY_BUDGET_OWNER_RETAINED')
        status = self.functions['scope_end'](C.byref(self.containers.config), self.active['identity']['scope']['token'])
        self.active.setdefault('leave_statuses', []).append(status)
        require(status == 0, 'DRAIN_OR_PENDING_EXECUTOR_UNCONFIRMED:' + str(status))
        self.active['status'] = 'RETURNED' if original_error is None else 'ORIGINAL_FAILED'; self.active = None

    def stop(self):
        states = self._phase('both','inspect_stop', before=self._owners)
        if states['fia']['installed']:
            if not states['fia']['stopped']:
                states = self._native_phase('fia','stop', before=self._owners)
        else:
            states = self._phase('fia','confirmed_absent_for_stop', before=self._absent_fia)
        if states['containers']['installed']:
            states = self._phase('containers','confirmed_idle_for_stop', before=self._base_idle)
            if not states['containers']['stopped']:
                self._native_phase('containers','stop', before=self._owners)
        else:
            self._phase('containers','confirmed_absent_for_stop', before=self._absent_containers)
        self.started = False

    def snapshot(self):
        before = self.query(); require(not before['active_calls'], 'ACTIVE_NATIVE_CALL')
        events = []
        for i in range(before['retained_events']):
            value = containers.instance(Event); self.call('event', i, C.byref(value)); events.append(record(value))
        after = self.query(); require(before == after, 'SNAPSHOT_NOT_QUIESCENT')
        result = dict(schema='demoncat.fia_input_journal.observation.v1', source='ACTUAL_NATIVE_RECORDS',
            build_sha256=self.containers.manifest_sha256, mapping=self.containers.mapping,
            startup=self.startup, worker=self.bound_worker, state=after, events=events,
            containers=self.containers.snapshot(), public_scopes=json.loads(json.dumps(self.scopes)),
            formals=FORMALS, tensor_formals=list(TENSOR_INDICES), container_formals=list(CONTAINER_INDICES),
            sdk_tags=self.sdk_tags, limits=self.limits, lifecycle=self.lifecycle,
            first_lifecycle_error=self.first_lifecycle_error, lifecycle_overflow=self.lifecycle_overflow,
            slice_pointer_lineage_proven=False, prepared_internal_tensor_lineage_proven=False,
            tensor_list_members_captured_at='ACTUAL_CREATOR_ENTRY', tensor_list_members_reread_at_gws=False,
            C1=False, C2=False, C3=False, E5=False, qualified=False)
        return json.loads(json.dumps(result))

    def release(self):
        states = self._phase('both','inspect_release', before=self._owners)
        if states['fia']['installed']:
            states = self._native_phase('fia','release', before=self._owners,
                after=lambda state: require(self._fia_empty(state['fia']) and state['fia']['retired'] == 1,
                    'FIA_RELEASE_NOT_CONFIRMED'))
        else:
            states = self._phase('fia','confirmed_absent_for_release', before=self._absent_fia)
        if states['containers']['installed']:
            states = self._native_phase('containers','release', before=self._owners, after=self._absent_containers)
        else:
            states = self._phase('containers','confirmed_absent_for_release', before=self._absent_containers)
        self.started = False
        return json.loads(json.dumps(dict(fia=states['fia'], containers=states['containers'], lifecycle=self.lifecycle,
            first_lifecycle_error=self.first_lifecycle_error, lifecycle_overflow=self.lifecycle_overflow,
            global_wrappers_unloaded=False, preloaded_library_remains_until_process_exit=True)))
