"""Worker bridge tests with explicit host fakes at the native ABI boundary."""

from contextlib import contextmanager, nullcontext
from importlib.util import find_spec
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from demoncat.runtime.vllm_ascend.worker_extension.providers import provider_capabilities, resolve_provider
from demoncat.runtime.vllm_ascend.worker_extension.providers import fault20, fault39
from demoncat.runtime.vllm_ascend.worker_extension.providers._scope import ProviderError
from demoncat.runtime.vllm_ascend.worker_extension.runtime import NativePreloadNotAttached
from demoncat_fault39.provider import CtypesProvider


OWNER = {'owner': 'lease', 'generation': 1}
IDENTITY = {'rank': 0, 'logical_device': 0, 'pid': os.getpid(),
            'worker_start_identity': 'worker', 'model_instance_id': 'model'}


class Fault20Fixture:
    def __init__(self, mechanism, facts, evidence_dir, *, operator):
        self.operator = operator
        self.worker = facts
        self.build_profile = 'a' * 64
        self.enabled = False
        self.leases = 0

    def snapshot(self):
        return {self.operator: dict(native_registered=self.enabled, inflight=0,
                                    leases=self.leases, runtime_matched=True)}


class Fault39Fixture:
    def __init__(self):
        self.state = dict(owner='', worker_rank=-1, runtime_loaded=True, runtime_ready=True,
                          runtime_identity='MATCHED_BUILD', abi_version=1,
                          inflight=0, leases=0, generation=0,
                          enabled=False, build_profile_sha256='b' * 64)

    def snapshot(self):
        return dict(self.state)


class ProviderTests(unittest.TestCase):
    def test_resolver_has_no_native_side_effect_and_refuses_unknown_version(self):
        self.assertEqual({row['mechanism'] for row in provider_capabilities()},
                         {'20.buffer_substitute','39.matmul_range','26.fia_destination','quant.batch_shared_scale',
                          '15.alias_detach','17.spatial_padding','18.api_half','8.restore_offset','22.wrong_row',
                          '25.reduce_scatter','26.residual_stomp'})
        with patch.object(fault20.ctypes, 'CDLL', side_effect=AssertionError('loaded')):
            resolve_provider(fault20.MECHANISM, '1', object(), object(), IDENTITY)
            with self.assertRaisesRegex(RuntimeError, 'NOT_ADAPTED'):
                resolve_provider(fault20.MECHANISM, 'other', object(), object(), IDENTITY)

    def test_missing_preloads_are_proven_preinstall_refusals(self):
        provider20 = resolve_provider(fault20.MECHANISM, '1', object(), object(), IDENTITY)
        with patch.object(fault20.ctypes, 'CDLL', return_value=object()):
            with self.assertRaises(NativePreloadNotAttached):
                provider20.dispatch('prepare', {}, **OWNER)
        self.assertIsNone(provider20.runtime)
        provider39 = resolve_provider(fault39.MECHANISM, '1', object(), object(), IDENTITY)
        with patch.object(CtypesProvider, 'attach_preloaded', side_effect=RuntimeError('missing')):
            with self.assertRaises(NativePreloadNotAttached):
                provider39.dispatch('prepare', {}, **OWNER)
        self.assertIsNone(provider39.native)

    def test_fault20_readback_and_detach_require_native_idle(self):
        provider = resolve_provider(fault20.MECHANISM, '1', object(), object(), IDENTITY)
        ready = SimpleNamespace(dcat20_qmm_profile_v1=True)
        with (patch.object(fault20.ctypes, 'CDLL', return_value=ready),
              patch.object(provider, '_worker_facts', return_value={'pid': os.getpid()}),
              patch.object(fault20, 'CtypesMatmulRuntime', Fault20Fixture)):
            state = provider.dispatch('prepare', {'operator': 'aclnnQuantMatmulWeightNz'}, **OWNER)
        self.assertEqual(state['operator'], 'aclnnQuantMatmulWeightNz')
        self.assertTrue(state['native_loaded'])
        self.assertFalse(state['registered'])
        with self.assertRaisesRegex(ProviderError, 'OWNER_MISMATCH'):
            provider.query(owner='foreign', generation=1)
        with patch.object(fault20, 'qualified_binding', return_value=None):
            with self.assertRaisesRegex(ProviderError, 'TRUSTED_EXCLUSIVE_REPLAY_REQUIRED'):
                provider.dispatch('register', {}, **OWNER)
        provider.runtime.leases = 1
        self.assertEqual(provider.query(**OWNER)['leases'], 1)
        with self.assertRaisesRegex(ProviderError, 'NATIVE_RECOVERY_PENDING'):
            provider.dispatch('unregister', {}, **OWNER)
        provider.runtime.leases = 0
        self.assertFalse(provider.dispatch('unregister', {}, **OWNER)['native_registered'])

    @unittest.skipIf(find_spec('torch') is None, 'The real replay contract uses torch inference mode')
    def test_fault20_execute_scope_does_not_leave_hooks_armed_and_off_retries(self):
        model = object()
        provider = resolve_provider(fault20.MECHANISM, '1', object(), model, IDENTITY)
        runtime = Fault20Fixture(None, IDENTITY, '.', operator='aclnnMatmulWeightNz')
        provider.runtime = runtime
        provider.bind(**OWNER)
        provider._registered = provider.checked = True
        outputs = iter(('fault', 'wrong-off', 'healthy'))
        entered = []

        @contextmanager
        def reserve(worker, resolved_model):
            self.assertIs(resolved_model, model)
            entered.append('reserve')
            try:
                yield
            finally:
                entered.append('release')

        def replay(resolved_model):
            self.assertIs(resolved_model, model)
            output = next(outputs)
            self.assertEqual(runtime.enabled, output == 'fault')
            entered.append(output)
            return output

        replay.reserve_worker = reserve
        provider.binding = SimpleNamespace(replay=replay)
        qualification = {'replays': [{'model_output_sha256': 'healthy'}]}
        class SessionFixture:
            def __enter__(self):
                runtime.enabled = True

            def close(self):
                runtime.enabled = False

            def report(self):
                return {'qualification': qualification}

            def assert_coverage(self):
                return self.report()

        with (patch.object(fault20, 'QualifiedNativeSession', return_value=SessionFixture()),
              patch.object(fault20, '_digest_output', side_effect=lambda value: value)):
            with self.assertRaisesRegex(ProviderError, 'OFF_MODEL_OUTPUT_MISMATCH'):
                provider.dispatch('execute', {}, **OWNER)
            self.assertFalse(runtime.enabled)
            self.assertTrue(provider.query(**OWNER)['recovery_validation_pending'])
            with self.assertRaisesRegex(ProviderError, 'SINGLE_QUALIFIED_ATTEMPT'):
                provider.dispatch('execute', {}, **OWNER)
            self.assertFalse(provider.dispatch('unregister', {}, **OWNER)['recovery_validation_pending'])
        self.assertEqual(entered, ['reserve', 'fault', 'wrong-off', 'release', 'reserve', 'healthy', 'release'])

    def test_fault39_does_not_arm_a_prepared_service_with_no_scheduler_replay(self):
        provider = resolve_provider(fault39.MECHANISM, '1', object(), object(), IDENTITY)
        native = Fault39Fixture()
        with patch.object(CtypesProvider, 'attach_preloaded', return_value=native):
            state = provider.dispatch('prepare', {}, **OWNER)
        self.assertEqual(state['build_profile_sha256'], 'b' * 64)
        self.assertFalse(state['registered'])
        with patch.object(fault39, 'resolve_runtime', return_value=None):
            with self.assertRaisesRegex(ProviderError, 'TRUSTED_SCHEDULER_REPLAY_REQUIRED'):
                provider.dispatch('register', {}, **OWNER)
        with self.assertRaisesRegex(ProviderError, 'UNSUPPORTED_PROVIDER_SETTINGS'):
            provider.dispatch('register', {'request_ids': ['client-supplied']}, **OWNER)
        native.state['owner'] = 'someone-else'
        with self.assertRaisesRegex(ProviderError, 'OWNED_ELSEWHERE'):
            provider.dispatch('unregister', {}, **OWNER)
        native.state['owner'] = ''
        self.assertFalse(provider.dispatch('unregister', {}, **OWNER)['native_registered'])

    def test_fault39_checks_trusted_profile_before_registering(self):
        provider = resolve_provider(fault39.MECHANISM, '1', object(), object(), IDENTITY)
        with patch.object(CtypesProvider, 'attach_preloaded', return_value=Fault39Fixture()):
            provider.dispatch('prepare', {}, **OWNER)
        binding = fault39.Fault39WorkerReplay('a' * 64, nullcontext, lambda *_: None)
        with patch.object(fault39, 'resolve_runtime', return_value=binding):
            with self.assertRaisesRegex(ProviderError, 'BUILD_PROFILE_MISMATCH'):
                provider.dispatch('register', {}, **OWNER)
        self.assertFalse(provider.query(**OWNER)['registered'])


if __name__ == '__main__':
    unittest.main()
