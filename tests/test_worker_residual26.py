import ctypes as C
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fault_injection.core import UnsupportedCall
from fault_injection.deployments.residual26 import FiaState, Residual26Runtime
from demoncat.runtime.vllm_ascend.worker_extension.providers.fault26 import Fault26ResidualProvider
from demoncat.runtime.vllm_ascend.worker_extension.providers._scope import ProviderError
from demoncat.runtime.vllm_ascend.worker_extension.serving26 import FiaServingSession, ResidualServingSession


class ActualStateFunction:
    def __init__(self, **values): self.values = values
    def __call__(self, pointer):
        state = C.cast(pointer, C.POINTER(FiaState)).contents
        state.runtime_ready = state.allowed = 1
        for name, value in self.values.items(): setattr(state, name, value)
        return 0


class ResidualWorkerTests(unittest.TestCase):
    def test_additive_qmm_state_keeps_real_pending(self):
        runtime = Residual26Runtime.__new__(Residual26Runtime)
        runtime.fia = False
        runtime.lib = SimpleNamespace(dcat26q_state_v1=ActualStateFunction(registered=1, owner_thread=1, pending_consumers=1))
        self.assertEqual(runtime.native_state()['pending_consumers'], 1)
        runtime.lib = SimpleNamespace(dcat26f_state_v1=ActualStateFunction())
        with self.assertRaisesRegex(UnsupportedCall, 'missing'): runtime.native_state()

    def test_resolver_binding_is_mechanism_specific(self):
        provider = Fault26ResidualProvider(object(), object(), {})
        with patch('fault_injection.providers.resolve_runtime', return_value=None) as resolve:
            with self.assertRaisesRegex(ProviderError, 'REVIEWED'):
                provider.dispatch('prepare', {}, owner='o', generation=1)
        self.assertEqual(resolve.call_args.args[0], '26.residual_stomp')

    def test_runtime_budget_checks_actual_supported_not_claim_counter(self):
        runtime = Residual26Runtime.__new__(Residual26Runtime)
        runtime.max_effect_calls = 1
        runtime.supported = 0
        runtime.current = {'packet': object()}
        self.assertTrue(runtime.check(None, '26.residual_stomp', 'baseline').supported)
        runtime.supported = 1  # A lossless pass claims no native fault, but consumes the slot.
        self.assertFalse(runtime.check(None, '26.residual_stomp', 'baseline').supported)

    def test_selected_cache_cannot_drift_between_lossless_and_fault(self):
        selected = dict(module_path='observed', module_object_id=42, operator='aclnnQuantMatmulWeightNz',
            cache=100, output_bytes=128, residual_bytes=128, geometry=[1, 64, 1, 64], allocation={'shift': 32})
        lossless = dict(phase='lossless', native={'selected_target': selected})
        changed = dict(selected, cache=101)
        fault = dict(phase='comparison', native=dict(selected_target=changed, supported=1, claims=1))
        session = ResidualServingSession.__new__(ResidualServingSession)
        session.current = {}; session.rows = [lossless]
        session.baseline_control = False; session.save = lambda: None
        session.recovered = False
        def finished(_self):
            _self.rows.append(fault)
            _self.current = None
        with patch.object(FiaServingSession, 'collect', finished):
            with self.assertRaisesRegex(RuntimeError, 'TARGET_CHANGED'):
                session.collect()
        self.assertIn('TARGET_CHANGED', session.error)
        self.assertFalse(session.recovered)


if __name__ == '__main__': unittest.main()
