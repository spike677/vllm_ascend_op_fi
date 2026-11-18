"""Read-only rejection diagnostics; no AddRms row gate or numerical change."""
import copy
import unittest

import torch

from demoncat.runtime.vllm_ascend.worker_extension.quant_row_provenance import RequestRowProvenance
from demoncat.runtime.vllm_ascend.worker_extension.quant_source_rows import SourceBoundRows
from test_quant_embedding_rows import _LIB  # Actual CPU dispatcher substitute registered by that module.


class AddRmsRowDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.mode = torch.inference_mode()
        self.mode.__enter__()
        self.addCleanup(self.mode.__exit__, None, None, None)
        self.model = torch.nn.Module()
        self.model.weight = torch.nn.Parameter(torch.ones(16, dtype=torch.bfloat16), requires_grad=False)
        self.graph = RequestRowProvenance(max_calls=128)
        self.source = SourceBoundRows(self.graph, self.model, [], {}, simulation=True, operator_aliases={
            torch.ops.cpu_quant_embedding_rows.norm.default: '_C_ascend.npu_add_rms_norm_bias.default'})
        self.graph.boundaries = self.source
        self.rows = dict(request_ids=['cpu-' + str(i) for i in range(8)], row_token_positions=[2]*8)
        self.x1, self.x2 = torch.ones(8, 16, dtype=torch.bfloat16), torch.ones(8, 16, dtype=torch.bfloat16)
        self.graph.begin('cpu-add-rms')
        self.source.begin(self.rows)
        self.addCleanup(lambda: self.graph.end() if self.graph.active else None)
        for tensor in (self.x1, self.x2):
            self.graph.seed(tensor, dict(verified=True, actual_rows=self.rows, source='EXPLICIT_CPU_FIXTURE'))

    def register(self):
        self.graph.register_broadcast_parameter(self.model, 'weight', self.model.weight, kind='_parameters')

    def call(self, *, gamma=None):
        result = torch.ops.cpu_quant_embedding_rows.norm(self.x1, self.x2,
            self.model.weight if gamma is None else gamma)
        event = next(row for row in reversed(self.graph.events)
            if row['operator'] == 'cpu_quant_embedding_rows.norm.default')
        self.assertEqual(event['original_calls'], 1)
        return result, event['add_rms_row_diagnostic']

    def test_existing_eligible_rule_keeps_two_outputs_and_all_actual_checks(self):
        self.register()
        result, diagnostic = self.call()
        self.assertTrue(self.graph.query(result[0])['verified'])
        self.assertTrue(self.graph.query(result[2])['verified'])
        self.assertEqual(diagnostic['actual_rule_accepted_outputs'], 2)
        for phase in ('before', 'after'):
            value = diagnostic[phase]
            self.assertTrue(value['complete'])
            self.assertEqual(value['failures'], [])
            self.assertEqual(value['gamma']['owner_object_id'], id(self.model))
            self.assertEqual(value['gamma']['owner_model_path'], '<root>')
            self.assertEqual(value['gamma']['parameter_name'], 'weight')
            self.assertFalse(value['gamma']['requires_grad'])
            self.assertEqual(value['gamma']['metadata_changed_fields'], [])
        self.assertEqual(len(diagnostic['after']['outputs']), 3)

    def test_actual_unregistered_requires_grad_parameter_is_diagnosed_not_admitted(self):
        self.model.weight = torch.nn.Parameter(torch.ones(16, dtype=torch.bfloat16), requires_grad=True)
        result, diagnostic = self.call()
        self.assertFalse(self.graph.query(result[0])['verified'])
        self.assertEqual(diagnostic['actual_rule_accepted_outputs'], 0)
        value = diagnostic['after']
        self.assertEqual(value['failures'], ['gamma_broadcast_proven'])
        self.assertEqual(value['gamma']['failures'], ['registered'])
        self.assertTrue(value['gamma']['requires_grad'])
        self.assertEqual(value['gamma']['observed_model_parameter_owners'], [dict(
            module_object_id=id(self.model), model_path='<root>', parameter_name='weight')])
        self.assertNotIn(id(self.model.weight), self.graph.parameters)

    def test_unregistered_frozen_parameter_is_still_not_registered_by_diagnostics(self):
        result, diagnostic = self.call()
        self.assertFalse(self.graph.query(result[0])['verified'])
        self.assertEqual(diagnostic['after']['gamma']['failures'], ['registered'])
        self.assertFalse(diagnostic['after']['gamma']['requires_grad'])
        self.assertEqual(self.graph.parameters, {})

    def test_equal_foreign_gamma_has_no_model_owner(self):
        self.register()
        gamma = self.model.weight.clone()
        result, diagnostic = self.call(gamma=gamma)
        self.assertFalse(self.graph.query(result[0])['verified'])
        self.assertEqual(diagnostic['after']['gamma']['observed_model_parameter_owners'], [])
        self.assertEqual(diagnostic['after']['gamma']['failures'], ['registered'])

    def test_changed_parameter_owner_is_separate_from_metadata_or_epoch(self):
        self.register()
        original = self.model.weight
        self.model.weight = torch.nn.Parameter(original.clone(), requires_grad=False)
        result, diagnostic = self.call(gamma=original)
        self.assertFalse(self.graph.query(result[0])['verified'])
        gamma = diagnostic['after']['gamma']
        self.assertTrue(gamma['registered'])
        self.assertEqual(gamma['failures'], ['owner_parameter_same_object'])
        self.assertEqual(gamma['metadata_changed_fields'], [])

    def test_inference_alias_write_is_reported_as_epoch_change_without_version(self):
        self.register()
        self.model.weight.view(16).add_(0)
        result, diagnostic = self.call()
        self.assertFalse(self.graph.query(result[0])['verified'])
        gamma = diagnostic['after']['gamma']
        self.assertTrue(gamma['actual_metadata']['inference_version_unavailable'])
        self.assertEqual(gamma['metadata_changed_fields'], [])
        self.assertEqual(gamma['failures'], ['write_epoch_matches'])
        self.assertGreater(gamma['actual_write_epoch'], gamma['recorded_write_epoch'])

    def test_two_current_rows_with_different_order_remain_refused(self):
        self.register()
        self.graph.seed(self.x2, dict(verified=True, actual_rows=dict(self.rows,
            request_ids=list(reversed(self.rows['request_ids'])))))
        result, diagnostic = self.call()
        self.assertFalse(self.graph.query(result[0])['verified'])
        value = diagnostic['after']
        self.assertTrue(value['first_input']['current'] and value['second_input']['current'])
        self.assertEqual(value['failures'], ['actual_request_rows_equal'])

    def test_diagnostic_of_stale_row_does_not_evict_or_rebind_it(self):
        self.register()
        record = self.graph.proofs[id(self.x1)]
        key = self.graph.storage_key(record['metadata'])
        self.graph.epochs[key] = record['write_epoch'] + 1
        before_keys = tuple(self.graph.proofs)
        before_nodes, before_epochs = copy.deepcopy(self.graph.nodes), dict(self.graph.epochs)
        diagnostic = self.source._row_diagnostic(self.x1)
        self.assertEqual(diagnostic['failures'], ['write_epoch_matches'])
        self.assertIs(self.graph.proofs[id(self.x1)], record)
        self.assertEqual(tuple(self.graph.proofs), before_keys)
        self.assertEqual(self.graph.nodes, before_nodes)
        self.assertEqual(self.graph.epochs, before_epochs)

    def test_report_snapshot_cannot_rewrite_current_diagnostic_or_parameter(self):
        self.register()
        _, diagnostic = self.call()
        expected = copy.deepcopy(diagnostic)
        public = self.graph.report()
        event = next(row for row in public['events'] if 'add_rms_row_diagnostic' in row)
        event['add_rms_row_diagnostic']['after']['gamma']['actual_metadata']['shape'][0] = 999
        event['add_rms_row_diagnostic']['before']['first_input']['checks']['metadata_matches'] = False
        self.assertEqual(diagnostic, expected)
        self.assertEqual(list(self.model.weight.shape), [16])


if __name__ == '__main__':
    unittest.main()
