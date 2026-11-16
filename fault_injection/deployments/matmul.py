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


_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_uint64, ctypes.c_char_p)
_SYMBOLS = {
    'aclnnQuantMatmulWeightNz': 'dcat20_qmm_',
    'aclnnMatmulWeightNz': 'dcat20_weight_nz_',
}
_MECHANISM = '20.buffer_substitute'
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
        self.evidence_dir = Path(evidence_dir)
        self._pending = self._callback = None
        self.last_error = None

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
        if self._install(native_callback, self.worker['rank']):
            raise UninstalledObserver('Native callback refused the loaded ABI, rank or existing owner')
        self._callback = native_callback

        def remove():
            if self._pending is not None or self._remove(native_callback):
                raise RuntimeError('Native callback still owns execution or unresolved allocations')
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


def register_worker_matmul(mechanism, *, worker_identity, replay, profiles, evidence_dir,
                           first_live_validation=False):
    """Deployment wiring. Worker RPC and replay are supplied by the actual runner."""
    if mechanism != _MECHANISM or not callable(worker_identity):
        raise ValueError('Expected Fault20 and a real worker identity reader')
    profiles = tuple(profiles)
    if not profiles or any(not isinstance(profile, ProviderProfile)
                           or not _REQUIRED_BUNDLE.issubset(profile.required_libraries) for profile in profiles):
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

    register_runtime(mechanism, factory, observer_factory=factory, profiles=profiles, replay=replay,
                     first_live_validation=first_live_validation)
    return factory
