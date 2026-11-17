"""Connect the opt-in Fault20 CANN Run callbacks to qualified injection."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid

from ..core import UnsupportedCall
from ..native import NativeCall, NativeProof, UninstalledObserver, _STACK
from ..providers import ProviderProfile, register_runtime
from .pta_resolver import inspect_loaded_pta_resolution, validate_loaded_pta_resolution


_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_uint64, ctypes.c_char_p)
_SYMBOLS = {
    'aclnnQuantMatmulWeightNz': 'dcat20_qmm_',
    'aclnnMatmulWeightNz': 'dcat20_weight_nz_',
}
_MECHANISM = '20.buffer_substitute'
_INSTALLED, _INFLIGHT, _QUARANTINED, _OWNED, _MATCHED = 1, 2, 4, 8, 16
_PAUSED, _REQUEST_CONTROLLED = 32, 64
_REQUIRED_BUNDLE = frozenset({
    'libnnopbase.so', 'libascendcl.so', 'libdcat20_qmm.so',
    'libopapi.so', 'libopapi_nn.so', 'libopapi_real.so', 'libopapi_nn_real.so',
})


class CtypesMatmulRuntime:
    """Synchronous, process-local callbacks from the actual CANN cache Run."""

    def __init__(self, mechanism, worker, evidence_dir, *, operator):
        if os.environ.get('TASK_QUEUE_ENABLE') != '0' or os.environ.get('VLLM_SERVER_DEV_MODE') != '1':
            raise UnsupportedCall('Start a developer worker with synchronous dispatch')
        if mechanism != _MECHANISM or operator not in _SYMBOLS:
            raise ValueError('Select the scoped Fault20 WeightNz or QMM operator')
        self.operator = operator
        prefix = _SYMBOLS[operator]
        self.mechanism = mechanism
        self.worker = dict(worker)
        if (type(self.worker.get('pid')) is not int or self.worker['pid'] != os.getpid()
                or any(type(self.worker.get(key)) is not int or self.worker[key] < 0
                       for key in ('tp', 'rank', 'logical_device'))
                or not 0 <= self.worker['rank'] < self.worker['tp']
                or self.worker.get('execution_mode') != 'eager'
                or not isinstance(self.worker.get('soc'), str) or not self.worker['soc']):
            raise UnsupportedCall('Expected actual eager worker PID, SoC, TP, rank and logical device')
        self._library = ctypes.CDLL(None)
        try:
            self._install = getattr(self._library, prefix + 'install_v1')
            self._remove = getattr(self._library, prefix + 'remove_v1')
            self._invoke = getattr(self._library, prefix + 'invoke_v1')
            self._receipt = getattr(self._library, prefix + 'receipt_v1')
            profile = self._library.dcat20_qmm_profile_v1
        except AttributeError as exc:
            raise UnsupportedCall('Matching CANN callback bundle was not preloaded') from exc
        self._install.argtypes, self._install.restype = [_CALLBACK, ctypes.c_int32], ctypes.c_int
        self._remove.argtypes, self._remove.restype = [_CALLBACK], ctypes.c_int
        self._invoke.argtypes, self._invoke.restype = [ctypes.c_uint64, ctypes.c_uint32], ctypes.c_uint32
        self._receipt.argtypes, self._receipt.restype = [ctypes.c_uint64], ctypes.c_char_p
        profile.argtypes, profile.restype = [], ctypes.c_char_p
        identity = profile()
        if identity is None or len(identity) != 64 or not all(byte in b'0123456789abcdef' for byte in identity):
            raise UnsupportedCall('Scoped Fault20 bundle returned an invalid build profile')
        self.build_profile = identity.decode('ascii')
        self._pta_configuration = os.environ.get('DEMONCAT_F20_REQUIRE_TYPED_GWS', '')
        self.pta_gate_status = self._verify_pta_gate()
        self.evidence_dir = Path(evidence_dir)
        self._pending = self._callback = None
        self._request_identity = None
        self.last_error = None

    def metadata_arrays(self):
        """Additive read-only resource proof required by the group adapter."""
        from ..metadata_arrays import LoadedMetadataArrays
        if not hasattr(self, '_metadata_arrays'):
            self._metadata_arrays = LoadedMetadataArrays(kind='20', build_profile=self.build_profile,
                anchor=self._library.dcat20_qmm_profile_v1)
        return self._metadata_arrays.read()

    def configure_request_lifecycle(self, *, owner, generation):
        """Bind a real Worker lease once; installation will begin paused."""
        if (self._callback is not None or self._pending is not None
                or self._request_identity is not None):
            raise UnsupportedCall('Fault20 request lifecycle must be configured once before installation')
        if (not isinstance(owner, str) or not owner or '\0' in owner
                or len(owner.encode('utf-8')) > 128 or type(generation) is not int
                or not 0 < generation < 1 << 64):
            raise UnsupportedCall('Expected a bounded real owner and positive uint64 generation')
        prefix = _SYMBOLS[self.operator]
        methods = {}
        try:
            for action in ('install', 'pause', 'resume', 'remove'):
                method = getattr(self._library, prefix + action + '_v2')
                method.argtypes = ([_CALLBACK, ctypes.c_int32, ctypes.c_char_p, ctypes.c_uint64]
                                   if action == 'install' else
                                   [_CALLBACK, ctypes.c_char_p, ctypes.c_uint64])
                method.restype = ctypes.c_int
                methods[action] = method
            for native_prefix in _SYMBOLS.values():
                getattr(self._library, native_prefix + 'state_v2')
        except AttributeError as error:
            raise UnsupportedCall('Fault20 owner/generation request ABI v2 is required') from error
        self._request_methods = methods
        self._request_identity = (owner.encode('utf-8'), generation)

    def _request_action(self, action):
        identity = self._request_identity
        if identity is None or self._callback is None or self._pending is not None:
            raise UnsupportedCall('Fault20 request control has no idle owned callback')
        if self._request_methods[action](self._callback, *identity):
            raise UnsupportedCall('Fault20 native ' + action + ' refused owner/generation/thread or active execution')
        state = self.snapshot()[self.operator]
        if (not state['native_owned'] or not state['request_controlled']
                or state['inflight'] or state['leases']
                or state['paused'] != (action == 'pause')):
            raise UnsupportedCall('Fault20 native request state did not confirm ' + action)
        return state

    def pause(self):
        return self._request_action('pause')

    def resume(self):
        return self._request_action('resume')

    def _verify_pta_gate(self):
        """The opt-in routing change is admission, never proof of a real hit."""
        if os.environ.get('DEMONCAT_F20_REQUIRE_TYPED_GWS', '') != self._pta_configuration:
            raise UnsupportedCall('Fault20 PTA policy changed during the Worker lifetime')
        if self._pta_configuration not in ('', '0', '1'):
            raise UnsupportedCall('Invalid explicit Fault20 PTA policy')
        if self._pta_configuration != '1':
            return {}
        try:
            resolution = inspect_loaded_pta_resolution()
            actual_provider = validate_loaded_pta_resolution(resolution)
        except (AttributeError, OSError, ValueError, TypeError) as error:
            raise UnsupportedCall('Fault20 actual PTA resolver route is unproven: ' + str(error)) from error
        previous = getattr(self, 'pta_resolver_status', None)
        if previous is not None and previous['configuration'] != resolution['configuration']:
            raise UnsupportedCall('Fault20 custom opapi resolver configuration changed')
        identity = {key: actual_provider[key] for key in ('provider', 'provider_sha256', 'address')}
        if getattr(self, '_pta_provider_identity', identity) != identity:
            raise UnsupportedCall('Fault20 actual PTA function provider changed')
        self._pta_provider_identity = identity
        self.pta_resolver_status = resolution
        wheels = resolution.get('loaded_torch_npu_images', [])
        if len(wheels) != 1 or len(wheels[0].get('file_sha256', '')) != 64:
            raise UnsupportedCall('Fault20 requires one actual profiled torch_npu resolver image')
        result = {}
        for name in ('libopapi_math.so', 'libopapi.so', 'libopapi_nn.so'):
            try:
                handle = ctypes.CDLL(name, mode=os.RTLD_NOLOAD | os.RTLD_NOW)
                status = handle.dcat20_pta_gate_status_v1
                status.argtypes, status.restype = [], ctypes.c_char_p
                raw = status()
                if not raw or len(raw) > 65536:
                    raise ValueError('Invalid gate status')
                row = json.loads(raw)
            except (AttributeError, OSError, ValueError, TypeError) as error:
                raise UnsupportedCall('Matched loaded Fault20 PTA proxy required: ' + name) from error
            if (row.get('schema') != 'dcat20.pta_gate.v1' or row.get('profile_sha256') != self.build_profile
                    or row.get('entry') != name
                    or row.get('resolver_wheel_sha256') != wheels[0]['file_sha256']
                    or any(row.get(key) is not True for key in
                           ('requested', 'config_valid', 'identity_valid', 'profile_match', 'original_available', 'ready'))
                    or row.get('refused') is not False):
                raise UnsupportedCall('Fault20 PTA control refused: ' + name + '/' + str(row.get('reason')))
            if name == 'libopapi_math.so' and row.get('original_math_identity_valid') is not True:
                raise UnsupportedCall('Fault20 original math provider identity is unproven')
            result[name] = row
        return result

    def snapshot(self):
        """Query both native bridges, including any callback owned elsewhere."""
        self.pta_gate_status = self._verify_pta_gate()
        states = {}
        for operator, prefix in _SYMBOLS.items():
            requester = ctypes.cast(self._callback, ctypes.c_void_p) if operator == self.operator else None
            try:
                state = getattr(self._library, prefix + 'state_v2')
                version = 2
            except AttributeError:
                if self._request_identity is not None:
                    raise UnsupportedCall('NATIVE_STATE_UNAVAILABLE: request state ABI v2 required')
                try:
                    state = getattr(self._library, prefix + 'state_v1')
                    version = 1
                except AttributeError as exc:
                    raise UnsupportedCall('NATIVE_STATE_UNAVAILABLE: Fault20 state ABI not preloaded') from exc
            if version == 2:
                state.argtypes, state.restype = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64], ctypes.c_uint32
                identity = self._request_identity if operator == self.operator else None
                flags = state(requester, *(identity or (None, 0)))
            else:
                state.argtypes, state.restype = [ctypes.c_void_p], ctypes.c_uint32
                flags = state(requester)
            if (flags & ~(_INSTALLED | _INFLIGHT | _QUARANTINED | _OWNED | _MATCHED | _PAUSED | _REQUEST_CONTROLLED)
                    or flags & (_INFLIGHT | _OWNED | _PAUSED | _REQUEST_CONTROLLED) and not flags & _INSTALLED
                    or flags & _PAUSED and (not flags & _REQUEST_CONTROLLED or flags & _INFLIGHT)):
                raise UnsupportedCall('NATIVE_STATE_UNAVAILABLE: invalid Fault20 state flags')
            states[operator] = {
                'native_registered': bool(flags & _INSTALLED),
                'native_owned': bool(flags & _OWNED),
                'runtime_matched': bool(flags & _MATCHED),
                'inflight': int(bool(flags & _INFLIGHT)),
                'recovery_required': bool(flags & _QUARANTINED),
                'leases': int(bool(flags & (_INFLIGHT | _QUARANTINED))),
                'paused': bool(flags & _PAUSED),
                'enabled': bool(flags & _INSTALLED and not flags & _PAUSED),
                'request_controlled': bool(flags & _REQUEST_CONTROLLED),
            }
        return states

    def _proofs(self, call_id, owner, packet, directory):
        rows = {
            'runtime_call': {**packet, 'worker': self.worker},
            'abi_handshake': {'actual_build_profile': self.build_profile, 'status': packet['runtime_identity']},
            'semantic_slot': {'proof_origin': 'typed_GWS_formals_and_live_Run_slots', 'slots': packet['slots']},
            'live_storage': {'proof_origin': 'native_nonoverlap_and_bounded_allocations', 'slots': packet['slots']},
        }
        proofs = []
        for kind, details in rows.items():
            receipt = dict(schema='demoncat.native.proof.v1', pid=os.getpid(), owner_id=id(owner),
                           call_id=call_id, operator=self.operator, mechanism=self.mechanism,
                           kind=kind, status='verified', details=details)
            raw = json.dumps(receipt, allow_nan=False).encode()
            path = directory / f'{call_id}-{kind}.json'
            with path.open('xb') as output:
                output.write(raw)
            proofs.append(NativeProof(kind, str(path), hashlib.sha256(raw).hexdigest()))
        return tuple(proofs)

    def install_observer(self, callback, candidates, mechanism):
        if mechanism != self.mechanism or self._callback is not None:
            raise UninstalledObserver('Wrong or duplicate MatMul observer')
        # Everything that can fail before the direct native install is local.
        owner_thread = threading.get_ident()
        objects = {id(module): module for _, module, _ in candidates}
        session = uuid.uuid4().hex
        directory = self.evidence_dir / session
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise UninstalledObserver('Cannot create callback evidence directory') from exc

        def dispatch(token, raw):
            pending = None
            try:
                stack = _STACK.get()
                owner = stack[-1][1] if stack else None
                if owner is None or objects.get(id(owner)) is not owner or threading.get_ident() != owner_thread:
                    return 0 if self._invoke(token, 0) == 0 else 1
                if self._pending is not None or not raw or len(raw) > 32768:
                    raise UnsupportedCall('Invalid native callback payload or reentrancy')
                packet = json.loads(raw)
                if (packet.get('version') != 1 or packet.get('operator') != self.operator
                        or packet.get('device') != self.worker['logical_device']
                        or packet.get('runtime_identity') != 'MATCHED_BUILD'
                        or not packet.get('cache_lifecycle')
                        or packet.get('build_profile_sha256') != self.build_profile):
                    raise UnsupportedCall('Native callback identity or worker changed')
                call_id = f'{session}-{token}'
                call = NativeCall(owner, call_id, self.operator,
                                  proofs=self._proofs(call_id, owner, packet, directory))
                pending = {'call': call, 'token': token, 'mode': 0}
                self._pending = pending

                def original():
                    if 'status' in pending:
                        raise UnsupportedCall('Native continuation already consumed')
                    pending['status'] = self._invoke(token, pending['mode'])
                    return pending['status']

                callback(call, original)
                return 0 if pending.get('status') == 0 else 1
            except BaseException as exc:
                # The caller receives an error and no callback failure can
                # cause the original CANN kernel to execute a second time.
                self.last_error = str(exc)
                return 1
            finally:
                if pending is not None and self._pending is pending:
                    self._pending = None

        native_callback = _CALLBACK(dispatch)
        identity = self._request_identity
        status = (self._request_methods['install'](native_callback, self.worker['rank'], *identity)
                  if identity is not None else self._install(native_callback, self.worker['rank']))
        if status:
            raise UninstalledObserver('Native callback refused the loaded ABI, rank or existing owner')
        self._callback = native_callback

        def remove():
            if self._pending is not None:
                raise RuntimeError('Native callback still owns execution or unresolved allocations')
            result = (self._request_methods['remove'](native_callback, *identity) if identity is not None
                      else self._remove(native_callback))
            if result:
                raise RuntimeError('Native callback still owns execution or unresolved allocations; request owner must be paused')
            self._callback = None

        return remove

    def execute_scoped(self, call, invoke_once, mode):
        pending = self._pending
        if pending is None or pending['call'] is not call or pending['mode'] != 0 or mode not in (1, 2, 3):
            raise UnsupportedCall('No matching live native continuation')
        pending['mode'] = mode
        status = invoke_once()
        raw = self._receipt(pending['token'])
        receipt = json.loads(raw) if raw else {}
        if status:
            raise UnsupportedCall(f'Native call or buffer verification failed: mode={mode}, status={status}, '
                                  f'cleanup={receipt.get("resources_released")}')
        return status, receipt


class CtypesMatmulGroupRuntime(CtypesMatmulRuntime):
    """v3 exact target groups; never accepted by the existing C1 adapter.

    The native inventory is queried separately from the callback. Every eligible
    callback reaches this observer, even when no public module currently owns it.
    The trusted whole-request adapter must reconcile both denominators.
    """
    def configure_group_lifecycle(self, *, owner, generation, max_targets, max_faults, max_calls):
        self.configure_request_lifecycle(owner=owner, generation=generation)
        if (type(max_targets) is not int or not 0 < max_targets <= 4096
                or type(max_faults) is not int or not 0 < max_faults <= 8
                or type(max_calls) is not int or not 3 * max_targets <= max_calls <= 65536):
            raise UnsupportedCall('Bounded immutable native group budgets required')
        prefix = _SYMBOLS[self.operator]
        try:
            install = getattr(self._library, prefix + 'install_group_v3')
            self._seal_group = getattr(self._library, prefix + 'seal_group_v3')
            self._invoke_group = getattr(self._library, prefix + 'invoke_group_v3')
            self._group_state = getattr(self._library, prefix + 'group_state_v3')
            self._inventory = self._library.dcat20_independent_run_inventory_v1
        except AttributeError as error:
            raise UnsupportedCall('Native group v3 and pre-owner Run inventory are required') from error
        install.argtypes, install.restype = [_CALLBACK, ctypes.c_int32, ctypes.c_char_p, ctypes.c_uint64,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64], ctypes.c_int
        self._seal_group.argtypes, self._seal_group.restype = [_CALLBACK, ctypes.c_char_p, ctypes.c_uint64,
            ctypes.c_uint32, ctypes.c_char_p], ctypes.c_int
        self._invoke_group.argtypes, self._invoke_group.restype = [ctypes.c_uint64, ctypes.c_uint32, ctypes.c_char_p], ctypes.c_uint32
        self._group_state.argtypes, self._group_state.restype = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64], ctypes.c_char_p
        self._inventory.argtypes, self._inventory.restype = [], ctypes.c_char_p
        self.group_limits = dict(max_targets=max_targets, max_faults=max_faults, max_calls=max_calls)
        self._request_methods['install'] = lambda callback, rank, native_owner, native_generation: install(
            callback, rank, native_owner, native_generation, max_targets, max_faults, max_calls)

    def independent_inventory(self):
        raw = self._inventory()
        if not raw or len(raw) > 16 * 1024 * 1024:
            raise UnsupportedCall('Independent native inventory unavailable or over bounded capacity')
        row = json.loads(raw)
        if (row.get('schema') != 'dcat20.independent_run_inventory.v1'
                or row.get('complete') is not True or row.get('overflow_calls') != 0
                or row.get('total_run_calls') != sum(row.get(name, -1) for name in
                    ('eligible_run_calls', 'rejected_candidate_calls', 'non_target_run_calls'))
                or row.get('eligible_run_calls') != sum(target.get('observations', -1) for target in row.get('targets', []))):
            raise UnsupportedCall('Independent native denominator incomplete; no dropped call may be qualified')
        return row

    def group_state(self):
        if self._request_identity is None:
            raise UnsupportedCall('Group owner lifecycle not configured')
        requester = ctypes.cast(self._callback, ctypes.c_void_p) if self._callback else None
        raw = self._group_state(requester, *self._request_identity)
        if not raw or len(raw) > 4 * 1024 * 1024:
            raise UnsupportedCall('Native group state unavailable')
        state = json.loads(raw)
        if state.get('version') != 3:
            raise UnsupportedCall('Native group state ABI changed')
        return state

    def seal_group(self, target_keys):
        keys = tuple(sorted(target_keys))
        if (not keys or len(set(keys)) != len(keys)
                or any(not isinstance(key, str) or len(key) != 64
                       or any(c not in '0123456789abcdef' for c in key) for key in keys)):
            raise UnsupportedCall('Distinct full target SHA identities required')
        identity = hashlib.sha256(''.join(key + '\n' for key in keys).encode()).hexdigest()
        if self._pending is not None or self._callback is None or self._seal_group(
                self._callback, *self._request_identity, len(keys), identity.encode()):
            raise UnsupportedCall('Native group target set is not completely lossless-qualified and paused')
        state = self.group_state()
        if (not state.get('owned') or not state.get('sealed') or state.get('failed')
                or state.get('group_sha256') != identity or state.get('targets') != len(keys)
                or tuple(sorted(row['key'] for row in state.get('entries', []))) != keys
                or any(not row.get('ready') or row.get('fault_attempts') for row in state['entries'])):
            raise UnsupportedCall('Native group seal readback mismatch')
        return state

    def install_group_observer(self, callback, owner_reader):
        if (self._request_identity is None or not hasattr(self, 'group_limits')
                or self._callback is not None or not callable(callback) or not callable(owner_reader)):
            raise UninstalledObserver('Configured v3 owner and complete trusted observer required')
        owner_thread = threading.get_ident()
        session = uuid.uuid4().hex
        directory = self.evidence_dir / session
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except OSError as error:
            raise UninstalledObserver('Cannot create group callback evidence directory') from error

        def dispatch(token, raw):
            pending = None
            try:
                if threading.get_ident() != owner_thread or self._pending is not None or not raw or len(raw) > 32768:
                    raise UnsupportedCall('Invalid group native callback context or payload')
                packet = json.loads(raw)
                if (packet.get('version') != 1 or packet.get('operator') != self.operator
                        or packet.get('device') != self.worker['logical_device']
                        or packet.get('runtime_identity') != 'MATCHED_BUILD'
                        or not packet.get('cache_lifecycle') or not packet.get('current_typed_call')
                        or packet.get('build_profile_sha256') != self.build_profile):
                    raise UnsupportedCall('Current typed group callback identity changed')
                owner = owner_reader()
                call_id = f'{session}-{token}'
                call = NativeCall(owner, call_id, self.operator,
                    proofs=self._proofs(call_id, owner, packet, directory))
                pending = dict(call=call, token=token, mode=0, target_key=None)
                self._pending = pending

                def original():
                    if 'status' in pending:
                        raise UnsupportedCall('Group native continuation already consumed')
                    key = pending['target_key']
                    pending['status'] = self._invoke_group(token, pending['mode'], key.encode() if key else None)
                    return pending['status']

                callback(call, original, packet)
                return 0 if pending.get('status') == 0 else 1
            except BaseException as error:
                self.last_error = str(error)
                return 1
            finally:
                if pending is not None and self._pending is pending:
                    self._pending = None

        native_callback = _CALLBACK(dispatch)
        identity = self._request_identity
        if self._request_methods['install'](native_callback, self.worker['rank'], *identity):
            raise UninstalledObserver('Native group install refused owner/generation/budgets/runtime')
        self._callback = native_callback

        def remove():
            if self._pending is not None or self._request_methods['remove'](native_callback, *identity):
                raise RuntimeError('Native group cannot release active, foreign or uncertain resources')
            self._callback = None
        return remove

    def execute_group(self, call, invoke_once, mode, target_key):
        pending = self._pending
        if (pending is None or pending['call'] is not call or pending['mode'] != 0 or mode not in (1, 2, 3)
                or not isinstance(target_key, str) or len(target_key) != 64
                or any(c not in '0123456789abcdef' for c in target_key)):
            raise UnsupportedCall('Exact live group continuation and target key required')
        pending.update(mode=mode, target_key=target_key)
        status = invoke_once()
        raw = self._receipt(pending['token'])
        receipt = json.loads(raw) if raw else {}
        if (status or receipt.get('group_v3', {}).get('target_key') != target_key
                or receipt.get('original_calls') != 1 or receipt.get('resources_released') is not True):
            raise UnsupportedCall('Native group execution/effect failed: ' + json.dumps(receipt, sort_keys=True))
        return status, receipt


def register_worker_matmul(mechanism, *, worker_identity, replay, profiles, evidence_dir,
                           first_live_validation=False, reserve_worker=None):
    """Deployment wiring. Worker RPC and replay are supplied by the actual runner."""
    if mechanism != _MECHANISM or not callable(worker_identity):
        raise ValueError('Expected Fault20 and a real worker identity reader')
    profiles = tuple(profiles)
    required = _REQUIRED_BUNDLE
    if os.environ.get('DEMONCAT_F20_REQUIRE_TYPED_GWS') == '1':
        required = required | {'libopapi_math.so', 'libopapi_math_real.so'}
    if not profiles or any(not isinstance(profile, ProviderProfile)
                           or not required.issubset(profile.required_libraries) for profile in profiles):
        raise ValueError('Each profile must fingerprint the loaded Fault20 bundle and both CANN proxies')
    operators = {profile.operator for profile in profiles}
    if len(operators) != 1 or not operators <= _SYMBOLS.keys():
        raise ValueError('Worker deployment requires profiles for one supported native operator')
    operator = next(iter(operators))

    def factory(model):
        worker = dict(worker_identity(model))
        if worker.pop('model', None) is not model:
            raise UnsupportedCall('Worker identity is not bound to the loaded model object')
        return CtypesMatmulRuntime(mechanism, worker, evidence_dir, operator=operator)

    if reserve_worker is not None:
        if not callable(reserve_worker) or not callable(replay):
            raise ValueError('Worker reservation and replay must be trusted callables')
        invoke = replay

        def replay(model):
            return invoke(model)

        replay.reserve_worker = reserve_worker

    register_runtime(mechanism, factory, observer_factory=factory, profiles=profiles, replay=replay,
                     first_live_validation=first_live_validation)
    return factory
