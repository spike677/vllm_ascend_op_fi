"""Adapt the existing Fault39 native ABI without modifying the model runner."""

from dataclasses import dataclass

from fault_injection.providers import resolve_runtime

from ..runtime import NativePreloadNotAttached
from ._scope import ProviderError, Scope, reject_unknown, require_idle


MECHANISM = '39.matmul_range'


@dataclass(frozen=True)
class Fault39WorkerReplay:
    """Trusted deployment binding from the existing runtime registry.

    reserve_worker(worker, model) must exclude unrelated work. The actual
    scheduler integration in replay(worker, model, phase, provider, owner)
    binds the real request IDs with provider.context and returns ReplayObservation.
    A client-provided request ID cannot implement that contract.
    """

    build_profile_sha256: str
    reserve_worker: object
    replay: object


class Fault39Provider(Scope):
    def __init__(self, worker, model, identity):
        super().__init__(worker, model, identity)
        self.native = self.backend = None
        self.checked = self.attempted = False

    def dispatch(self, action, settings, *, owner, generation):
        if action == 'prepare':
            self.bind(owner, generation)
            reject_unknown(settings)
            self._prepare()
        else:
            self.verify(owner, generation)
            reject_unknown(settings, ('pattern', 'amplitude_policy') if action == 'register' else ())
            if action == 'register':
                self._register(owner, settings)
            elif action == 'check':
                self._check()
            elif action in ('execute', 'enable'):
                if not self.checked or self.attempted:
                    raise ProviderError('FAULT39_SINGLE_QUALIFIED_ATTEMPT_REQUIRED')
                self.attempted = True
                self.backend.enable()
            elif action in ('disable', 'finish', 'abort', 'clean', 'unregister'):
                self._stop(action, owner, generation)
            else:
                raise ProviderError('UNSUPPORTED_PROVIDER_ACTION')
        return self.query(owner=owner, generation=generation)

    def _prepare(self):
        try:
            from demoncat_fault39.provider import CtypesProvider
        except ImportError as exc:
            raise ProviderError('FAULT39_PACKAGE_UNAVAILABLE') from exc
        try:
            self.native = CtypesProvider.attach_preloaded()
        except RuntimeError as exc:
            raise NativePreloadNotAttached(str(exc)) from exc
        try:
            state = self._read_native()
        except ProviderError as exc:
            if exc.code != 'NATIVE_STATE_UNAVAILABLE':
                raise
            raise NativePreloadNotAttached(str(exc)) from exc
        require_idle(state, release_owner=True)

    def _register(self, owner, settings):
        from demoncat_fault39.backend import WorkerEndpoint
        from demoncat_fault39.host_qualification import QualifiedFault39Backend, is_sha256
        from demoncat_fault39.schema import Fault39Config

        binding = resolve_runtime(MECHANISM, self.model)
        if (not isinstance(binding, Fault39WorkerReplay) or not is_sha256(binding.build_profile_sha256)
                or not callable(binding.reserve_worker) or not callable(binding.replay)):
            raise ProviderError('FAULT39_TRUSTED_SCHEDULER_REPLAY_REQUIRED')
        state = self._read_native()
        require_idle(state, release_owner=True)
        if binding.build_profile_sha256 != state['build_profile_sha256']:
            raise ProviderError('FAULT39_BUILD_PROFILE_MISMATCH')
        rank, device = self.identity['rank'], self.identity['logical_device']
        config = Fault39Config(ranks=(rank,), devices=(device,), **settings)
        self.native.bind_worker_rank(rank)
        self.backend = QualifiedFault39Backend(
            {rank: WorkerEndpoint(rank, device, self.native)},
            build_profile_sha256=binding.build_profile_sha256,
            reserve_worker=lambda: binding.reserve_worker(self.worker, self.model),
            worker_identity=self._model_identity,
            replay=lambda phase: binding.replay(self.worker, self.model, phase, self.native, owner),
            owner=owner,
        )
        self.backend.prepare()
        self.backend.register(config)
        self._registered = True

    def _model_identity(self):
        return (self.identity['model_instance_id']
                if getattr(getattr(self.worker, 'model_runner', None), 'model', None) is self.model else None)

    def _check(self):
        if not self._registered or self.backend is None:
            raise ProviderError('FAULT39_REGISTER_REQUIRED')
        require_idle(self._read_native(), release_owner=True)
        self.backend.prepare()
        # Discovery and the three lossless replays run under the exact same
        # exclusive reservation as the attempted injection inside enable().
        self.checked = True

    def _stop(self, action, owner, generation):
        release_owner = action in ('clean', 'unregister')
        if self.backend is not None:
            result = self.backend.unregister() if release_owner else self.backend.disable()
            if result.get('status') != 'DISABLED_CONFIRMED':
                raise ProviderError('NATIVE_RECOVERY_PENDING')
        state = self.query(owner=owner, generation=generation)
        require_idle(state, release_owner=release_owner)
        if release_owner:
            self._registered = False

    def _read_native(self):
        if self.native is None:
            raise ProviderError('NATIVE_STATE_UNAVAILABLE')
        try:
            snapshot = self.native.snapshot()
        except (RuntimeError, ValueError) as exc:
            raise ProviderError('NATIVE_STATE_UNAVAILABLE', str(exc)) from exc
        owner = snapshot.get('owner')
        rank = snapshot.get('worker_rank')
        if (snapshot.get('runtime_loaded') is not True or snapshot.get('runtime_ready') is not True
                or snapshot.get('runtime_identity') != 'MATCHED_BUILD'
                or type(snapshot.get('abi_version')) is not int or snapshot['abi_version'] != 1
                or rank not in (-1, self.identity['rank'])
                or not isinstance(owner, str) or type(snapshot.get('enabled')) is not bool
                or any(type(snapshot.get(key)) is not int or snapshot[key] < 0
                       for key in ('generation', 'inflight', 'leases'))):
            raise ProviderError('NATIVE_STATE_UNAVAILABLE', 'Fault39 build or snapshot did not match')
        if owner and owner != self._ticket[0]:
            raise ProviderError('NATIVE_PROVIDER_OWNED_ELSEWHERE')
        return {
            'native_loaded': True, 'registered': self._registered,
            'native_registered': bool(owner), 'enabled': snapshot['enabled'],
            'inflight': snapshot['inflight'], 'leases': snapshot['leases'],
            'build_profile_sha256': snapshot['build_profile_sha256'],
            'native': {key: value for key, value in snapshot.items() if key != 'candidates'},
        }

    def query(self, *, owner, generation):
        self.verify(owner, generation)
        result = self._read_native()
        if self.backend is not None:
            result['qualification'] = self.backend.query()['qualification']
        return result
