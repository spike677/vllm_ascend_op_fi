"""Attach the selected Fault20 callback to the loaded evaluation model."""

from dataclasses import replace
import ctypes
import os
import sys

from fault_injection.core import UnsupportedCall
from fault_injection.deployments.matmul import CtypesMatmulRuntime
from fault_injection.providers import qualified_binding
from fault_injection.qualification import QualifiedNativeSession, _digest_output

from ..runtime import NativePreloadNotAttached
from ._scope import ProviderError, Scope, reject_unknown, require_idle


MECHANISM = '20.buffer_substitute'
OPERATORS = ('aclnnMatmulWeightNz', 'aclnnQuantMatmulWeightNz')


class Fault20Provider(Scope):
    def __init__(self, worker, model, identity):
        super().__init__(worker, model, identity)
        self.runtime = self.binding = self.session = None
        self.off_expected = None
        self.checked = self.attempted = False

    def _worker_facts(self):
        torch = sys.modules.get('torch')
        npu = getattr(torch, 'npu', None)
        device_name = getattr(npu, 'get_device_name', None)
        config = getattr(self.worker, 'vllm_config', None)
        parallel = getattr(self.worker, 'parallel_config', getattr(config, 'parallel_config', None))
        model_config = getattr(self.worker, 'model_config', getattr(config, 'model_config', None))
        rank = self.identity['rank']
        tp = getattr(parallel, 'tensor_parallel_size', None)
        device = self.identity['logical_device']
        if (not callable(device_name) or type(tp) is not int or type(rank) is not int
                or not 0 <= rank < tp or getattr(model_config, 'enforce_eager', None) is not True):
            raise ProviderError('FAULT20_EAGER_WORKER_IDENTITY_UNPROVEN')
        return {
            'pid': os.getpid(), 'tp': tp, 'rank': rank, 'logical_device': device,
            'soc': device_name(device), 'execution_mode': 'eager',
            'worker_id': self.identity['worker_start_identity'],
            'model_instance_id': self.identity['model_instance_id'],
        }

    def dispatch(self, action, settings, *, owner, generation):
        if action == 'prepare':
            self.bind(owner, generation)
            reject_unknown(settings, ('operator',))
            operator = settings.get('operator', OPERATORS[0])
            if operator not in OPERATORS or self.runtime is not None:
                raise ProviderError('FAULT20_OPERATOR_SCOPE_INVALID')
            if not hasattr(ctypes.CDLL(None), 'dcat20_qmm_profile_v1'):
                raise NativePreloadNotAttached('Fault20 preload symbols are unavailable')
            try:
                # The inspection runtime never installs a callback or writes evidence.
                self.runtime = CtypesMatmulRuntime(MECHANISM, self._worker_facts(), '.', operator=operator)
            except UnsupportedCall as exc:
                raise NativePreloadNotAttached(str(exc)) from exc
            try:
                state = self.query(owner=owner, generation=generation)
            except ProviderError as exc:
                if exc.code != 'NATIVE_STATE_UNAVAILABLE':
                    raise
                raise NativePreloadNotAttached(str(exc)) from exc
            require_idle(state, release_owner=True)
            return state
        self.verify(owner, generation)
        reject_unknown(settings)
        if action == 'register':
            self._register()
        elif action == 'check':
            qualification = self._run(fault=False)
            self.checked = True
            return qualification
        elif action in ('execute', 'enable'):
            if not self.checked or self.attempted:
                raise ProviderError('FAULT20_SINGLE_QUALIFIED_ATTEMPT_REQUIRED')
            self.attempted = True
            return self._run(fault=True)
        elif action in ('disable', 'finish', 'abort'):
            self._recover()
        elif action in ('clean', 'unregister'):
            self._recover()
            state = self.query(owner=owner, generation=generation)
            require_idle(state, release_owner=True)
            self._registered = False
        else:
            raise ProviderError('UNSUPPORTED_PROVIDER_ACTION')
        return self.query(owner=owner, generation=generation)

    def _register(self):
        binding = qualified_binding(MECHANISM)
        if binding is None or not callable(getattr(binding.replay, 'reserve_worker', None)):
            raise ProviderError('FAULT20_TRUSTED_EXCLUSIVE_REPLAY_REQUIRED')
        runtime = binding.factory(self.model)
        worker_facts = self._worker_facts()
        if (not isinstance(runtime, CtypesMatmulRuntime) or runtime.operator != self.runtime.operator
                or runtime.build_profile != self.runtime.build_profile
                or any(runtime.worker.get(key) != value for key, value in worker_facts.items())):
            raise ProviderError('FAULT20_TRUSTED_BINDING_MISMATCH')
        self.runtime = runtime
        self.binding = replace(binding, factory=lambda model: runtime)
        require_idle(self._read_native(), release_owner=True)
        self._registered = True

    def _run(self, *, fault):
        import torch

        if self.binding is None or not self._registered or self.session is not None:
            raise ProviderError('FAULT20_REGISTER_REQUIRED')
        with self.binding.replay.reserve_worker(self.worker, self.model):
            require_idle(self._read_native(), release_owner=True)
            self.session = QualifiedNativeSession(self.model, MECHANISM, self.binding,
                                                  arm='fault' if fault else 'observe')
            try:
                self.session.__enter__()
                evidence = self.session.report()['qualification']
                if not fault:
                    return evidence
                self.off_expected = evidence['replays'][0]['model_output_sha256']
                with torch.inference_mode():
                    output = _digest_output(self.binding.replay(self.model))
                evidence = self.session.assert_coverage()['qualification']
                return {
                    'qualification': evidence, 'baseline_sha256': self.off_expected, 'fault_sha256': output,
                    'model_output_changed': output != self.off_expected, 'off_model_output_verified': True,
                }
            finally:
                # No fault or model hook remains armed across RPC calls.
                self._recover_reserved()

    def _recover_reserved(self):
        if self.session is not None:
            self.session.close()
            self.session = None
        require_idle(self._read_native(), release_owner=True)
        if self.off_expected is not None:
            import torch

            with torch.inference_mode():
                output = _digest_output(self.binding.replay(self.model))
            require_idle(self._read_native(), release_owner=True)
            if output != self.off_expected:
                raise ProviderError('FAULT20_OFF_MODEL_OUTPUT_MISMATCH')
            self.off_expected = None

    def _recover(self):
        if self.session is None and self.off_expected is None:
            require_idle(self._read_native(), release_owner=True)
            return
        with self.binding.replay.reserve_worker(self.worker, self.model):
            self._recover_reserved()

    def _read_native(self):
        if self.runtime is None:
            raise ProviderError('NATIVE_STATE_UNAVAILABLE')
        try:
            states = self.runtime.snapshot()
        except UnsupportedCall as exc:
            raise ProviderError('NATIVE_STATE_UNAVAILABLE', str(exc)) from exc
        matched = all(state['runtime_matched'] for state in states.values())
        if not matched:
            raise ProviderError('NATIVE_STATE_UNAVAILABLE', 'Fault20 CANN build identity did not match')
        registered = any(state['native_registered'] for state in states.values())
        return {
            'native_loaded': True, 'registered': self._registered,
            'native_registered': registered, 'enabled': registered,
            'inflight': sum(state['inflight'] for state in states.values()),
            'leases': sum(state['leases'] for state in states.values()),
            'build_profile_sha256': self.runtime.build_profile,
            'operator': self.runtime.operator, 'native': states,
        }

    def query(self, *, owner, generation):
        self.verify(owner, generation)
        return {**self._read_native(), 'recovery_validation_pending': self.off_expected is not None}
