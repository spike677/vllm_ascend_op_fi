"""Executed CPU tensor contracts; these tests do not grant NPU qualification."""
from types import SimpleNamespace
import unittest

import torch

from demoncat.runtime.vllm_ascend.worker_extension.providers._scope import ProviderError
from demoncat.runtime.vllm_ascend.worker_extension.providers.tensor_modules import (
    TensorModuleProvider, install_tensor_replay, uninstall_tensor_replay)
from fault_injection.module_faults import ModuleFaultSession


class SimulatedFormat:
    simulation = True
    def __init__(self): self.temps = []
    def format(self, tensor):
        return 2 if any(tensor.untyped_storage()._cdata == x.untyped_storage()._cdata
                        for x in self.temps) else 29
    def convert(self, tensor):
        value = tensor.clone(); self.temps.append(value); return value
    def synchronize(self, tensor): pass


class AliasWorkload(torch.nn.Module):
    def forward(self):
        base = torch.arange(256, dtype=torch.float32).reshape(16, 16).to(torch.bfloat16)
        view = torch.as_strided(base, (8, 8), (16, 1), 17)
        view.copy_(torch.full_like(view, -32))
        return base + 1


class ConvWorkload(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.arbitrary = torch.nn.Conv3d(1, 1, (1, 16, 16), stride=(1, 16, 16), bias=False)
        self.arbitrary.weight.data.fill_(1.)
    def forward(self, x): return self.arbitrary(x) + 1


class NormWorkload(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.arbitrary = torch.nn.RMSNorm(8)
    def forward(self, x): return self.arbitrary(x) + 1


class TensorWorkerProviders(unittest.TestCase):
    def bind(self, mechanism, model, inputs=(), *, replay=None, simulation=True):
        model.eval()
        worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
        binding = install_tensor_replay(worker, mechanism,
            replay=replay or (lambda m, phase: m(*inputs)), synchronize=lambda: None,
            workload_kind='constructed_module', simulation=simulation,
            _format_provider=SimulatedFormat() if mechanism == '15.alias_detach' else None)
        provider = TensorModuleProvider(worker, model, {}, mechanism)
        for action in ('prepare', 'register', 'check'):
            provider.dispatch(action, {}, owner='owner', generation=1)
        return worker, binding, provider

    def test_three_real_tensor_mechanisms_bf_off(self):
        cases = [('15.alias_detach', AliasWorkload(), ()),
                 ('17.spatial_padding', ConvWorkload(), (torch.arange(256.).reshape(1,1,1,16,16),)),
                 ('18.api_half', NormWorkload(), (torch.arange(8.).reshape(1,8),))]
        for mechanism, model, inputs in cases:
            with self.subTest(mechanism=mechanism):
                worker, binding, provider = self.bind(mechanism, model, inputs)
                result = provider.dispatch('execute', {}, owner='owner', generation=1)
                self.assertFalse(result['native_loaded'])
                self.assertEqual(result['execution_boundary'], 'python_hook')
                self.assertEqual(result['state_source'], 'OWNED_PYTHON_HOOKS')
                self.assertTrue(result['qualification']['recovered'])
                effect = result['qualification']['effect']
                self.assertTrue(effect['output_changed'] and effect['off_exact'])
                self.assertEqual(effect['interventions'], 1)
                self.assertFalse(effect['native_execution_claimed'])
                self.assertEqual(len(result['qualification']['phases']), 5)
                self.assertFalse(result['hook_owned'])
                self.assertEqual(result['leases'], 0)
                with self.assertRaises(ProviderError):
                    provider.dispatch('execute', {}, owner='owner', generation=1)
                with self.assertRaises(ValueError):
                    uninstall_tensor_replay(worker, mechanism, binding)
                provider.dispatch('unregister', {}, owner='owner', generation=1)
                uninstall_tensor_replay(worker, mechanism, binding)

    def test_actual_multiple_targets_refused_before_intervention(self):
        model = torch.nn.Sequential(torch.nn.RMSNorm(8), torch.nn.RMSNorm(8))
        _, _, provider = self.bind('18.api_half', model, (torch.arange(8.),))
        with self.assertRaisesRegex(ProviderError, 'ACTUAL_TARGET_ABSENT_OR_AMBIGUOUS'):
            provider.dispatch('execute', {}, owner='owner', generation=1)
        self.assertEqual([r['phase'] for r in provider.rows], ['baseline', 'baseline_repeat'])
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
        provider.dispatch('clean', {}, owner='owner', generation=1)

    def test_missing_binding_and_foreign_owner_refused(self):
        model = NormWorkload().eval()
        worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
        provider = TensorModuleProvider(worker, model, {}, '18.api_half')
        with self.assertRaisesRegex(ProviderError, 'TRUSTED_TENSOR_REPLAY_REQUIRED'):
            provider.dispatch('prepare', {}, owner='owner', generation=1)
        _, _, provider = self.bind('18.api_half', model, (torch.ones(8),))
        with self.assertRaisesRegex(ProviderError, 'PROVIDER_OWNER_MISMATCH'):
            provider.dispatch('execute', {}, owner='stranger', generation=1)

    def test_input_drift_prevents_fault_and_cpu_cannot_claim_npu(self):
        model = NormWorkload()
        _, _, provider = self.bind('18.api_half', model,
            replay=lambda m, phase: m(torch.ones(8) * (2 if phase == 'baseline_repeat' else 1)))
        with self.assertRaisesRegex(ProviderError, 'BASELINE_REPLAY_NOT_EXACTLY_STABLE'):
            provider.dispatch('execute', {}, owner='owner', generation=1)
        provider.dispatch('clean', {}, owner='owner', generation=1)
        _, _, provider = self.bind('18.api_half', NormWorkload(), (torch.ones(8),), simulation=False)
        with self.assertRaisesRegex(ProviderError, 'REAL_NPU_OUTPUT_REQUIRED'):
            provider.dispatch('execute', {}, owner='owner', generation=1)
        provider.dispatch('clean', {}, owner='owner', generation=1)

    def test_failed_model_and_external_hook_ownership_are_recoverable(self):
        model = NormWorkload().eval()
        def replay(m, phase):
            result = m(torch.ones(8))
            if phase == 'fault': raise RuntimeError('intentional replay failure')
            return result
        _, _, provider = self.bind('18.api_half', model, replay=replay)
        with self.assertRaisesRegex(RuntimeError, 'intentional replay failure'):
            provider.dispatch('execute', {}, owner='owner', generation=1)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
        provider.dispatch('clean', {}, owner='owner', generation=1)
        session = ModuleFaultSession(model, '18.api_half')
        session.__enter__()
        handle = session.handles[0]
        hooks = handle.hooks_dict_ref()
        original = hooks[handle.id]
        foreign = lambda *args, **kwargs: None
        hooks[handle.id] = foreign
        self.assertFalse(session.runtime_state()['ownership_verified'])
        with self.assertRaisesRegex(RuntimeError, 'Foreign module hook'):
            session.close()
        self.assertIs(hooks[handle.id], foreign)
        hooks[handle.id] = original
        session.close()
        self.assertTrue(session.runtime_state()['cleanup_confirmed'])

    def test_layer_name_and_coverage_options_not_accepted_over_rpc(self):
        _, _, provider = self.bind('18.api_half', NormWorkload(), (torch.ones(8),))
        for settings in ({'layer':'arbitrary'}, {'every_match':True}, {'target':'all_eligible'}):
            with self.assertRaisesRegex(ProviderError, 'UNSUPPORTED_PROVIDER_SETTINGS'):
                provider.dispatch('execute', settings, owner='owner', generation=1)


if __name__ == '__main__': unittest.main()
