import json
import unittest
from unittest.mock import patch

import torch

from fault_injection.capabilities import describe_mechanism, list_mechanisms, inspect_support
from fault_injection.catalog import list_faults
from fault_injection import register_runtime, unregister_runtime


class CapabilityScopeTests(unittest.TestCase):
    def test_every_remaining_variant_has_explicit_non_inherited_scope(self):
        specs = list_mechanisms()
        self.assertEqual(len(specs), 15)
        self.assertEqual({s['name'] for s in specs},
                         {s['name'] for s in list_faults() if not s['name'].startswith('39.')})
        for spec in specs:
            self.assertFalse(spec['current_model_npu_accepted'])
            self.assertFalse(spec['native_launch_verified'])
            self.assertEqual(spec['target'], 'auto')
            self.assertEqual(spec['integration_status'], 'INNER_FRAMEWORK_INTEGRATION_PENDING')
            self.assertEqual(spec['evidence_contract']['rtol'], 1e-5)
            self.assertFalse(spec['evidence_contract']['use_injection_logs_to_choose_first_difference'])
        json.dumps(specs, allow_nan=False)
        with self.assertRaises(ValueError): describe_mechanism('39.matmul_range')

    def test_read_only_discovery_never_calls_forward_or_provider(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.RMSNorm(4)).eval()
        factory = lambda model: self.fail('Provider must not be instantiated during inspection')
        register_runtime('20.buffer_substitute', factory)
        try:
            with patch.object(model, 'forward', side_effect=AssertionError('No forward during discovery')):
                report = inspect_support(model, '20.buffer_substitute')
            self.assertEqual(report['status'], 'STATIC_CANDIDATE_FOUND')
            self.assertEqual(report['candidates'][0]['path'], '0')
            self.assertFalse(report['provider_loaded_or_called'])
            self.assertFalse(model._forward_hooks)
            self.assertFalse(model._forward_pre_hooks)
        finally:
            unregister_runtime('20.buffer_substitute', factory)
        self.assertEqual(inspect_support(model, '20.buffer_substitute')['status'], 'DEPLOYMENT_PROVIDER_REQUIRED')

    def test_missing_dynamic_paths_are_not_marked_supported(self):
        model = torch.nn.Identity().eval()
        expected = {'8.restore_offset':'LEGACY_EVIDENCE_ONLY', '22.wrong_row':'LEGACY_EVIDENCE_ONLY',
                    '25.reduce_scatter':'NOT_IMPLEMENTED', '41.tail_column':'NOT_IMPLEMENTED',
                    '28.fa.legal_tiling':'STRATEGY_BRIDGE_ONLY', '15.alias_detach':'RUNTIME_OBSERVATION_REQUIRED',
                    '17.spatial_padding':'NO_SUPPORTED_CANDIDATE'}
        for name, status in expected.items():
            self.assertEqual(inspect_support(model, name)['status'], status)

    def test_standard_modules_use_structure_and_not_names(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.unrelated_name = torch.nn.RMSNorm(4)
        model = Model().eval()
        report = inspect_support(model, '18.api_half')
        self.assertEqual(report['status'], 'STATIC_CANDIDATE_FOUND')
        self.assertEqual(report['candidates'][0]['path'], 'unrelated_name')
        self.assertTrue(report['runtime_call_validation_required'])
        self.assertFalse(report['current_model_npu_accepted'])

    def test_definitions_are_fresh_copies_and_faults_differ(self):
        first = describe_mechanism('15.alias_detach')
        first['evidence_contract']['preserve_null'] = False
        self.assertTrue(describe_mechanism('15.alias_detach')['evidence_contract']['preserve_null'])
        self.assertNotEqual(describe_mechanism('28.fia.denominator_replay')['mechanism'],
                            describe_mechanism('28.fia.legal_tiling')['mechanism'])
        self.assertNotEqual(describe_mechanism('26.residual_stomp')['mechanism'],
                            describe_mechanism('26.fia_destination')['mechanism'])


if __name__ == '__main__':
    unittest.main()
