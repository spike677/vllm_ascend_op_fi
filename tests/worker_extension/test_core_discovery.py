"""Public C0 inventories cannot authorize calls or bypass Worker ownership."""
import copy
import unittest
from unittest.mock import patch
import torch

from demoncat.runtime.vllm_ascend.worker_extension import registry, identity
from demoncat.runtime.vllm_ascend.worker_extension.providers import provider_capabilities, resolve_provider
from test_extension import _extended_worker, ReadbackProvider


class SharedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(4, 4)
        self.second = self.first
        self.norm = torch.nn.RMSNorm(4)
        self.calls = 0
    def forward(self, x):
        self.calls += 1
        return self.norm(self.second(self.first(x)))


class CoreDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.device = patch.object(identity, '_current_device', return_value=0)
        self.device.start(); self.addCleanup(self.device.stop)
        self.worker = _extended_worker(); self.worker.load_model()
        self.model = self.worker.model_runner.model = SharedModel()
        self.facts = self.worker.demoncat_worker_identity()['worker']
        self.ticket = self.worker.demoncat_attach(dict(tenant_id='c0', worker=self.facts))['ticket']

    def test_all_aliases_and_unresolved_candidates_are_retained_without_forward(self):
        result = self.worker.demoncat_discover(self.ticket)
        self.assertEqual(result['status'], 'DISCOVERY_SNAPSHOT')
        report = result['discovery']; self.assertEqual(report['worker'], self.facts)
        self.assertEqual(len(report['candidates']), 3)
        shared = next(row for row in report['candidates'] if row['structural']['object_identity'] == id(self.model.first))
        self.assertEqual(shared['structural']['aliases'], ['first', 'second'])
        self.assertTrue(all(row['status'] == 'STRUCTURAL_UNVERIFIED' for row in report['candidates']))
        self.assertFalse(report['actual_call_census']['eligible_denominator_complete'])
        self.assertFalse(report['qualification_granted']); self.assertEqual(self.model.calls, 0)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))
        self.assertEqual(self.worker.demoncat_detach(self.ticket)['status'], 'DETACHED')

    def test_unknown_selector_and_foreign_worker_cannot_drive_discovery(self):
        self.assertEqual(self.worker.demoncat_discover(dict(self.ticket, layer_name='first'))['reason'], 'INVALID_PAYLOAD')
        changed = copy.deepcopy(self.ticket); changed['worker']['model_instance_id'] = 'foreign'
        self.assertEqual(self.worker.demoncat_discover(changed)['reason'], 'WORKER_IDENTITY_MISMATCH')
        self.assertEqual(self.worker.demoncat_fault_rpc('discover', self.ticket)['reason'], 'UNSUPPORTED_ACTION')

    def test_model_replacement_during_inspection_cannot_return_valid_inventory(self):
        from fault_injection.model_discovery import inspect_structure
        def replace(model, **kwargs):
            result = inspect_structure(model, **kwargs)
            self.worker.model_runner.model = SharedModel()
            return result
        with patch('fault_injection.model_discovery.inspect_structure', side_effect=replace):
            result = self.worker.demoncat_discover(self.ticket)
        self.assertEqual(result['status'], 'DISCOVERY_FAILED')
        self.assertFalse(result['qualification_granted'])
        self.assertNotIn('discovery', result)

    def test_current_provider_owner_prevents_new_diagnostic_walk(self):
        provider = ReadbackProvider()
        with patch('demoncat.runtime.vllm_ascend.worker_extension.runtime._resolve_provider', return_value=provider):
            result = self.worker.demoncat_fault_rpc('prepare', dict(self.ticket,
                mechanism='20.buffer_substitute', implementation_version='1', owner='owned'))
        self.assertEqual(result['status'], 'ACTION_COMPLETED')
        self.assertEqual(self.worker.demoncat_discover(self.ticket)['reason'], 'DISCOVERY_REQUIRES_IDLE_WORKER_OWNER_TABLE')


class RegistryTests(unittest.TestCase):
    def test_seventeen_rpc_routes_do_not_grant_attention_fault_admission(self):
        manifest = registry.worker_core_manifest()
        self.assertEqual(manifest['mechanism_count'], 17)
        self.assertEqual(manifest['rpc_provider_count'], 17)
        self.assertEqual(len(provider_capabilities()), 17)
        pending = [r for r in manifest['mechanisms'] if not r['rpc_available']]
        self.assertEqual(len(pending), 0)
        self.assertTrue(all(r['qualification_required'] and not r['qualification_granted'] for r in manifest['mechanisms']))
        attention = [r for r in manifest['mechanisms'] if r['mechanism'].startswith(('28.', '33.'))]
        self.assertEqual(len(attention), 4)
        self.assertTrue(all(r['route']['fault_execution_ready'] is False for r in attention))
        with patch('importlib.import_module') as loader:
            for row in attention:
                with self.assertRaisesRegex(RuntimeError, 'WORKER_PROVIDER_NOT_ADAPTED'):
                    resolve_provider(row['mechanism'], 'unsupported', None, None, {})
            loader.assert_not_called()

    def test_attention_factories_are_distinct_and_require_actual_deployment(self):
        constructors = {'28.fa.legal_tiling': 'FaLegalTilingServingProvider',
            '28.fia.legal_tiling': 'FiaLegalTilingServingProvider',
            '28.fia.denominator_replay': 'FiaDenominatorReplayServingProvider',
            '33.paged_decode': 'PagedDecodeServingProvider'}
        for mechanism, name in constructors.items():
            with self.subTest(mechanism=mechanism), patch('importlib.import_module') as loader:
                resolve_provider(mechanism, '1', 'worker', 'model', {'rank': 7})
                getattr(loader.return_value, name).assert_called_once_with('worker', 'model', {'rank': 7})
                self.assertTrue(loader.call_args.args[0].endswith('.providers.attention_serving'))

    def test_vision_observation_route_requires_explicit_worker_binding(self):
        from types import SimpleNamespace
        for bound in (False, True):
            worker = SimpleNamespace()
            if bound:
                worker._demoncat_vision17_observation_binding = object()
            with patch('importlib.import_module') as loader:
                resolve_provider('17.spatial_padding', '1', worker, 'model', {})
                if bound:
                    loader.return_value.Vision17ObservationProvider.assert_called_once_with(worker, 'model', {})
                    self.assertTrue(loader.call_args.args[0].endswith('.providers.vision17'))
                else:
                    loader.return_value.TensorModuleProvider.assert_called_once_with(worker, 'model', {}, '17.spatial_padding')
        row = next(r for r in registry.rpc_provider_capabilities() if r['mechanism'] == '17.spatial_padding')
        self.assertTrue(row['optional_two_healthy_mm_census'])
        self.assertFalse(row['census_grants_fault_admission'])

    def test_enumeration_is_detached_and_never_loads_native_provider(self):
        with patch('importlib.import_module') as loader:
            manifest = registry.worker_core_manifest()
            manifest['mechanisms'][0]['route']['operators'].clear()
            self.assertTrue(registry.worker_core_manifest()['mechanisms'][0]['route']['operators'])
            loader.assert_not_called()

    def test_existing_native_factory_and_tensor_routes_remain_exact(self):
        with patch('importlib.import_module') as loader:
            resolve_provider('20.buffer_substitute', '1', 'worker', 'model', {'rank': 3})
            loader.return_value.Fault20Provider.assert_called_once_with('worker', 'model', {'rank': 3})
            self.assertTrue(loader.call_args.args[0].endswith('.providers.fault20'))
        with patch('importlib.import_module') as loader:
            resolve_provider('15.alias_detach', '1', 'worker', 'model', {})
            loader.return_value.TensorModuleProvider.assert_called_once_with('worker', 'model', {}, '15.alias_detach')
        for mechanism in ('18.api_half', '8.restore_offset'):
            with patch('importlib.import_module') as loader:
                resolve_provider(mechanism, '1', 'worker', 'model', {})
                loader.return_value.TensorServingProvider.assert_called_once_with('worker', 'model', {}, mechanism)
                self.assertTrue(loader.call_args.args[0].endswith('.providers.tensor_serving'))


if __name__ == '__main__':
    unittest.main()
