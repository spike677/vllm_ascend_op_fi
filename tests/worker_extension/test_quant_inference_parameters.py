"""Actual TLS/model/parameter contracts; no production NPU qualification."""
import threading
import unittest

import torch

from demoncat.runtime.vllm_ascend.worker_extension.quant_row_provenance import RequestRowProvenance, metadata
from demoncat.runtime.vllm_ascend.worker_extension.quant_source_rows import SourceBoundRows
from test_quant_embedding_rows import _LIB


class InferenceParameterRowsTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Module()
        self.model.norm = torch.nn.Module()
        # Like source74 gamma: a normal versioned Parameter, with its original
        # requires_grad flag still true when a later inference call uses it.
        self.model.norm.weight = torch.nn.Parameter(torch.ones(16, dtype=torch.bfloat16))
        self.model.eval()
        self.graph = RequestRowProvenance(max_calls=128)
        self.source = SourceBoundRows(self.graph, self.model, [], {}, simulation=True, operator_aliases={
            torch.ops.cpu_quant_embedding_rows.norm.default: '_C_ascend.npu_add_rms_norm_bias.default'})
        self.graph.boundaries = self.source
        self.rows = dict(request_ids=['cpu-' + str(i) for i in range(8)], row_token_positions=[2]*8)
        self.x1, self.x2 = torch.ones(8, 16, dtype=torch.bfloat16), torch.ones(8, 16, dtype=torch.bfloat16)
        self.addCleanup(lambda: self.graph.end() if self.graph.active else None)

    def begin(self):
        self.graph.begin('current-inference-forward')
        self.source.begin(self.rows)

    def register(self, *, path='norm', module=None):
        self.graph.register_inference_broadcast_parameter(self.model, path,
            self.model.norm if module is None else module, 'weight', self.model.norm.weight)

    def accepts(self, tensor=None):
        return self.graph._broadcast_parameter(self.model.norm.weight if tensor is None else tensor, (8, 16))

    def norm(self):
        for tensor in (self.x1, self.x2):
            self.graph.seed(tensor, dict(verified=True, actual_rows=self.rows, source='EXPLICIT_CPU_FIXTURE'))
        output = torch.ops.cpu_quant_embedding_rows.norm(self.x1, self.x2, self.model.norm.weight)
        event = next(row for row in reversed(self.graph.events) if 'add_rms_row_diagnostic' in row)
        self.assertEqual(event['original_calls'], 1)
        return output, event['add_rms_row_diagnostic']

    def positive(self, context, *, inference):
        before = metadata(self.model.norm.weight)
        with context:
            self.begin()
            self.register()
            self.assertTrue(self.accepts())
            output, diagnostic = self.norm()
            self.assertTrue(self.graph.query(output[0])['verified'], diagnostic)
            self.assertTrue(self.graph.query(output[2])['verified'])
            for phase in ('before', 'after'):
                gamma = diagnostic[phase]['gamma']
                self.assertTrue(gamma['requires_grad'] and gamma['registered'] and gamma['broadcast_proven'])
                self.assertTrue(gamma['inference_context']['current'])
                self.assertFalse(gamma['inference_registration']['grad_enabled'])
                self.assertEqual(gamma['inference_registration']['inference_mode_enabled'], inference)
                self.assertEqual(gamma['inference_registration']['module_path'], 'norm')
                self.assertEqual(diagnostic[phase]['failures'], [])
            self.graph.end()
        self.assertTrue(self.model.norm.weight.requires_grad)
        self.assertFalse(self.model.training or self.model.norm.training)
        self.assertEqual(metadata(self.model.norm.weight), before)
        self.assertFalse(self.accepts())

    def test_actual_inference_mode_accepts_unchanged_requires_grad_weight(self):
        self.positive(torch.inference_mode(), inference=True)

    def test_actual_no_grad_forward_accepts_without_inference_tensor_conversion(self):
        self.positive(torch.no_grad(), inference=False)

    def test_training_model_or_owner_is_refused_without_flag_changes(self):
        with torch.no_grad():
            self.begin()
            self.model.train()
            with self.assertRaisesRegex(RuntimeError, 'EVAL_INFERENCE_PARAMETER_OWNER_PATH'):
                self.register()
            self.assertTrue(self.model.training)
            self.assertEqual(self.graph.parameters, {})
            self.model.eval()
            self.model.norm.train()
            with self.assertRaisesRegex(RuntimeError, 'EVAL_INFERENCE_PARAMETER_OWNER_PATH'):
                self.register()
            self.assertTrue(self.model.norm.training)
            self.assertEqual(self.graph.parameters, {})

    def test_grad_enabled_registration_is_refused(self):
        self.begin()
        self.assertTrue(torch.is_grad_enabled())
        with self.assertRaisesRegex(RuntimeError, 'CURRENT_INFERENCE_MODEL_PARAMETER'):
            self.register()
        self.assertEqual(self.graph.parameters, {})

    def test_grad_enabled_use_is_refused_after_valid_registration(self):
        with torch.no_grad():
            self.begin()
            self.register()
            with torch.enable_grad():
                self.assertFalse(self.accepts())
                context = self.graph._inference_parameter_context(
                    self.graph.parameters[id(self.model.norm.weight)], self.model.norm.weight)
                self.assertIn('gradients_disabled', context['failures'])

    def test_switching_actual_inference_mode_is_refused_even_with_no_grad(self):
        with torch.inference_mode():
            self.begin()
            self.register()
            with torch.inference_mode(False), torch.no_grad():
                self.assertFalse(self.accepts())
                context = self.graph._inference_parameter_context(
                    self.graph.parameters[id(self.model.norm.weight)], self.model.norm.weight)
                self.assertEqual(context['failures'], ['same_inference_mode'])

    def test_foreign_thread_cannot_use_or_register_same_model_weight(self):
        with torch.no_grad():
            self.begin()
            self.register()
            results = []
            def other_thread():
                with torch.no_grad():
                    results.append(self.accepts())
                    try:
                        self.register()
                    except RuntimeError as error:
                        results.append(str(error))
            thread = threading.Thread(target=other_thread)
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(results[0], False)
            self.assertIn('CURRENT_INFERENCE_MODEL_PARAMETER', results[1])

    def test_wrong_path_or_foreign_module_cannot_register_by_shape(self):
        with torch.no_grad():
            self.begin()
            with self.assertRaisesRegex(RuntimeError, 'MODULE_PATH'):
                self.register(path='missing')
            with self.assertRaisesRegex(RuntimeError, 'OWNER_PATH'):
                self.register(path='')
            foreign = torch.nn.Module()
            foreign.weight = self.model.norm.weight
            foreign.eval()
            with self.assertRaisesRegex(RuntimeError, 'OWNER_PATH'):
                self.register(module=foreign)
            self.assertEqual(self.graph.parameters, {})

    def test_replaced_owner_path_is_refused_even_with_identical_parameter_object(self):
        with torch.no_grad():
            self.begin()
            self.register()
            old = self.model.norm
            replacement = torch.nn.Module()
            replacement.weight = old.weight
            replacement.eval()
            self.model.norm = replacement
            self.assertFalse(self.accepts(old.weight))
            self.assertTrue(old.weight.requires_grad)

    def test_training_or_requires_grad_flag_change_after_registration_is_refused(self):
        with torch.no_grad():
            self.begin()
            self.register()
            self.model.norm.train()
            self.assertFalse(self.accepts())
            self.model.norm.eval()
            self.model.norm.weight.requires_grad_(False)
            self.assertFalse(self.accepts())

    def test_parameter_replacement_and_versioned_write_are_refused(self):
        with torch.no_grad():
            self.begin()
            self.register()
            original = self.model.norm.weight
            original.add_(0)
            self.assertFalse(self.accepts())
            self.model.norm.weight = torch.nn.Parameter(torch.ones_like(original))
            self.assertFalse(self.accepts())
            self.assertFalse(self.accepts(original))

    def test_inference_tensor_without_version_still_refuses_alias_write_epoch(self):
        with torch.inference_mode():
            self.model.norm.weight = torch.nn.Parameter(torch.ones(16, dtype=torch.bfloat16))
            self.begin()
            self.register()
            self.assertTrue(self.accepts())
            record = self.graph.parameters[id(self.model.norm.weight)]
            self.model.norm.weight.view(16).add_(0)
            self.assertEqual(metadata(self.model.norm.weight), record['metadata'])
            self.assertFalse(self.accepts())
            self.assertGreater(self.graph.epochs[self.graph.storage_key(record['metadata'])], record['write_epoch'])

    def test_same_parameter_with_replaced_storage_is_refused(self):
        with torch.no_grad():
            self.begin()
            self.register()
            weight = self.model.norm.weight
            record = self.graph.parameters[id(weight)]
            replacement = torch.ones_like(weight)
            weight.data = replacement
            self.assertIs(self.model.norm.weight, weight)
            self.assertEqual(metadata(weight)['version'], record['metadata']['version'])
            self.assertNotEqual(metadata(weight)['storage_address'], record['metadata']['storage_address'])
            self.assertFalse(self.accepts())

    def test_late_reregistration_cannot_reset_mutation_epoch(self):
        with torch.no_grad():
            self.begin()
            self.register()
            self.model.norm.weight.add_(0)
            with self.assertRaisesRegex(RuntimeError, 'BEFORE_OBSERVED_CALLS'):
                self.register()
            self.assertFalse(self.accepts())

    def test_request_axis_parameter_stays_nonbroadcast(self):
        with torch.no_grad():
            self.model.norm.weight = torch.nn.Parameter(torch.ones(8, 16, dtype=torch.bfloat16))
            self.begin()
            self.register()
            self.assertFalse(self.accepts())

    def test_pointwise_evidence_retains_actual_inference_parameter_context(self):
        with torch.no_grad():
            self.begin()
            self.register()
            self.graph.seed(self.x1, dict(verified=True, actual_rows=self.rows))
            output = self.x1 + self.model.norm.weight
            self.assertTrue(self.graph.query(output)['verified'])
            event = self.graph.events[-1]
            evidence = event['broadcast_parameters'][0]
            self.assertEqual(evidence['inference_registration']['module_path'], 'norm')
            self.assertTrue(evidence['inference_context']['current'])
            self.assertFalse(evidence['inference_context']['actual']['grad_enabled'])

    def test_legacy_registration_still_rejects_requires_grad_parameter(self):
        with torch.no_grad():
            self.begin()
            with self.assertRaisesRegex(RuntimeError, 'FROZEN_MODEL_PARAMETER_OWNER'):
                self.graph.register_broadcast_parameter(self.model.norm, 'weight', self.model.norm.weight, kind='_parameters')
            self.assertEqual(self.graph.parameters, {})

    def test_new_forward_cannot_reuse_previous_registration(self):
        with torch.no_grad():
            self.begin()
            self.register()
            self.graph.end()
            self.graph.begin('another-actual-forward')
            self.source.begin(self.rows)
            self.assertFalse(self.accepts())
            self.register()
            self.assertTrue(self.accepts())

    def test_existing_serving_provider_closed_loop_in_actual_inference_context(self):
        # Existing real Runtime/provider lifecycle with its explicitly fake
        # Native DSO, not direct graph-only success or production authority.
        from test_quant_group_c1 import QuantGroupC1Tests
        case = QuantGroupC1Tests('test_existing_provider_runs_single_site_and_keeps_full_unknown_denominator')
        case.setUp()
        try:
            before = metadata(case.model.head.weight)
            with torch.inference_mode():
                for _ in range(7):
                    case.phase()
            record = case.session.row_observer.parameters[id(case.model.head.weight)]
            self.assertEqual(record['inference_registration']['module_path'], 'head')
            self.assertTrue(record['inference_registration']['inference_mode_enabled'])
            result = case.dispatch('finish')['serving']
            self.assertTrue(result['c1'] and result['functional_effect_proven'])
            self.assertFalse(result['c2'] or result['c3'] or result['all_eligible_qualification'])
            self.assertFalse(result['owned_hooks'] or result['owned_nested_dispatch_modes'] or result['observer_cleanup_pending'])
            self.assertEqual(metadata(case.model.head.weight), before)
            self.assertTrue(case.model.head.weight.requires_grad)
            case.dispatch('unregister')
        finally:
            case.doCleanups()


if __name__ == '__main__':
    unittest.main()
