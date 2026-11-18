"""Real CPU dispatcher/profile contracts; no expert routing or NPU qualification."""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import warnings

import torch
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

from demoncat.runtime.vllm_ascend.worker_extension.quant_row_provenance import RequestRowProvenance, metadata
from demoncat.runtime.vllm_ascend.worker_extension.quant_source_rows import SourceBoundRows, source_bindings
from demoncat.runtime.vllm_ascend.worker_extension.quant_moe_rows import SharedMoERows


_LIB = torch.library.Library('cpu_quant_shared_moe_rows', 'DEF')
_LIB.define('shared(Tensor(a0!) hidden_states, Tensor router_logits, Tensor? shared_experts_input, str layer_name) -> (Tensor, Tensor)')
_LIB.define('unknown_write(Tensor(a!) x) -> ()')


def get_layer_from_name(layer_name):
    forward_context = _context
    return forward_context.no_compile_layers[layer_name]


def cpu_shared(hidden_states, router_logits, shared_experts_input, layer_name):
    layer = get_layer_from_name(layer_name)
    if layer.pre_resume_write:
        hidden_states.add_(0)
    return layer.runner.forward_impl(layer, hidden_states, router_logits, shared_experts_input)


def unknown_write(x):
    x.copy_(torch.roll(x, 1, 0))


_LIB.impl('shared', cpu_shared, 'CPU')
_LIB.impl('unknown_write', unknown_write, 'CPU')


class CpuRunner:
    def __init__(self, layer, shared):
        self.layer, self.shared_experts = layer, shared
        self.layer_name = layer.layer_name
        self.moe_config = NS(tp_size=1, dp_size=1, ep_size=1, sp_size=1, pcp_size=1,
            tp_rank=0, dp_rank=0, ep_rank=0, pcp_rank=0, is_sequence_parallel=False)
        self.enable_dbo = False
        self.routed_input_transform = None
        self.shared_experts_stream = None

    def forward(self, hidden_states, router_logits):
        return torch.ops.cpu_quant_shared_moe_rows.shared(hidden_states, router_logits, hidden_states, 'cpu-layer')

    def forward_impl(self, layer, hidden_states, router_logits, shared_input):
        if self.layer.swap_input:
            hidden_states = hidden_states.clone()
        return layer.forward_impl(hidden_states, router_logits)


class CpuLayer(torch.nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.layer_name = 'cpu-layer'
        self._shared_experts = shared
        self.runner = CpuRunner(self, shared)
        self.moe_config = self.runner.moe_config
        self._routed_input_transform = self.shared_expert_stream = None
        self.multistream_overlap_shared_expert = self.multistream_overlap_gate = False
        self.pre_resume_write = self.swap_input = self.mutate = self.throw = self.unknown_shared = False
        self.forward_count = 0

    def forward(self, hidden_states, router_logits):
        return self.runner.forward(hidden_states, router_logits)

    def forward_impl(self, hidden_states, router_logits):
        self.forward_count += 1
        _queries.append(('routed', _graph.query(hidden_states)))
        routed_out = torch.roll(hidden_states, 1, 0)
        if self.mutate:
            torch.ops.cpu_quant_shared_moe_rows.unknown_write(hidden_states)
        shared_out = self._forward_shared_experts(hidden_states, None)
        if self.throw:
            raise RuntimeError('CPU_MOE_ORIGINAL_FAILURE')
        return shared_out, routed_out

    def _forward_shared_experts(self, hidden_states, fused_moe_evts):
        part1_out = self._shared_experts_part1(hidden_states)
        return self._shared_experts_part2(hidden_states, part1_out)

    def _shared_experts_part1(self, hidden_states):
        _queries.append(('shared1', _graph.query(hidden_states)))
        return hidden_states.clone()

    def _shared_experts_part2(self, hidden_states, shared_gate_up):
        _queries.append(('shared2', _graph.query(shared_gate_up)))
        if self.unknown_shared:
            return torch.roll(shared_gate_up, 1, 0)
        return shared_gate_up + 1


class CpuModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Module()
        self.experts = CpuLayer(self.shared)

    def forward(self, x, logits):
        return self.experts(x, logits)


class SharedMoERowsTests(unittest.TestCase):
    def setUp(self):
        global _context, _graph, _queries
        self.model = CpuModel().eval()
        self.graph = _graph = RequestRowProvenance(max_calls=256)
        _queries = []
        _context = NS(no_compile_layers={'cpu-layer': self.model.experts})
        roles = {'cpu_shared': 'moe_opaque', 'get_layer_from_name': 'moe_resolve',
            'CpuRunner.forward': 'moe_public_forward', 'CpuRunner.forward_impl': 'moe_runner',
            'CpuLayer.forward_impl': 'moe_layer', 'CpuLayer._forward_shared_experts': 'moe_shared',
            'CpuLayer._shared_experts_part1': 'moe_part1', 'CpuLayer._shared_experts_part2': 'moe_part2'}
        pin = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        bindings, sources = source_bindings({__name__: (pin, roles)})
        self.source = SourceBoundRows(self.graph, self.model, bindings, sources, simulation=True,
            operator_aliases={torch.ops.cpu_quant_shared_moe_rows.shared.default: 'vllm.moe_forward_shared.default'})
        self.source.moe = SharedMoERows(self.source, simulation_library=_LIB)
        self.graph.boundaries = self.source
        self.rows = dict(request_ids=['cpu-' + str(i) for i in range(8)], row_token_positions=[2] * 8)
        self.addCleanup(lambda: self.graph.end() if self.graph.active else None)

    def begin(self, x, *, seed=True, forward='cpu-forward'):
        self.graph.begin(forward)
        self.source.begin(self.rows)
        if seed:
            self.graph.seed(x, dict(verified=True, actual_rows=self.rows, source='EXPLICIT_CPU_FIXTURE'))

    def event(self):
        return next(row for row in self.graph.events if row['operator'] == 'cpu_quant_shared_moe_rows.shared.default')

    def invoke(self, *, seed=True):
        x, logits = torch.arange(32, dtype=torch.float32).reshape(8, 4), torch.ones(8, 2)
        self.begin(x, seed=seed)
        return x, self.model(x, logits)

    def assert_refused(self, reason, *, seed=True):
        with torch.inference_mode():
            _, output = self.invoke(seed=seed)
            diagnostic = self.event()['shared_moe_row_diagnostic']
            self.assertIn(reason, diagnostic['reason'], diagnostic)
            self.assertFalse(self.graph.query(output[0])['verified'])
            self.assertFalse(self.graph.query(output[1])['verified'])
            self.assertTrue(all(not proof['verified'] for _, proof in _queries))
            self.assertEqual(self.model.experts.forward_count, 1)
            self.assertEqual(self.event()['original_calls'], 1)
            self.graph.end()
        self.assertIsNone(sys.getprofile())
        self.assertEqual(_get_current_dispatch_mode_stack(), [])

    def test_actual_parent_invalidation_and_nested_shared_chain_without_routed_proof(self):
        with torch.inference_mode():
            x, logits = torch.arange(32, dtype=torch.float32).reshape(8, 4), torch.ones(8, 2)
            before = metadata(x)
            self.begin(x)
            old_alias = x.view(8, 4)
            shared, routed = self.model(x, logits)
            diagnostic = self.event()['shared_moe_row_diagnostic']
            self.assertIsNone(diagnostic['reason'], diagnostic)
            self.assertTrue(diagnostic['resumed'])
            self.assertTrue(self.event()['mutable'])
            self.assertTrue(self.event()['invalidated_storages'])
            self.assertEqual(metadata(x), before)
            self.assertFalse(self.graph.query(old_alias)['verified'])
            self.assertTrue(self.graph.query(shared)['verified'])
            self.assertFalse(self.graph.query(routed)['verified'])
            self.assertEqual([(name, proof['verified']) for name, proof in _queries],
                             [('routed', False), ('shared1', True), ('shared2', True)])
            report = self.graph.end()
            self.assertTrue(report['complete'], report['source_boundaries']['errors'])
            self.assertTrue(all(event['original_calls'] == 1 for event in report['events']))
            self.assertEqual(report['source_boundaries']['owned_nested_dispatch_modes'], 0)
            self.assertEqual(self.model.experts.forward_count, 1)
        self.assertIsNone(sys.getprofile())
        self.assertEqual(_get_current_dispatch_mode_stack(), [])

    def test_two_actual_inference_forwards_cannot_reuse_suspended_proof(self):
        with torch.inference_mode():
            for i in range(2):
                x, logits = torch.ones(8, 4), torch.ones(8, 2)
                self.begin(x, forward='cpu-' + str(i))
                shared, routed = self.model(x, logits)
                self.assertTrue(self.graph.query(shared)['verified'], self.event()['shared_moe_row_diagnostic'])
                self.assertFalse(self.graph.query(routed)['verified'])
                self.assertEqual(self.event()['shared_moe_row_diagnostic']['actual_owner']['current_forward'], 'cpu-' + str(i))
                self.graph.end()
                self.assertEqual(self.source.operator_stack, [])

    def test_unknown_rows_are_retained_without_seed_or_row_count_inference(self):
        self.assert_refused('MOE_CURRENT_REQUEST_INPUT_ROWS_UNPROVEN', seed=False)

    def test_child_operation_before_runner_cannot_revive_declaration_snapshot(self):
        self.model.experts.pre_resume_write = True
        self.assert_refused('MOE_RESOLUTION_OR_PRE_RESUME_CHILD_CALL_CHANGED')

    def test_actual_mutating_child_invalidates_inference_aliases(self):
        self.model.experts.mutate = True
        self.assert_refused('MOE_INPUT_WRITTEN_OR_REPLACED')

    def test_same_geometry_replacement_input_is_not_original_shared_tensor(self):
        self.model.experts.swap_input = True
        self.assert_refused('MOE_ACTUAL_SHARED_FRAME_INPUT_OR_OWNER_CHANGED')

    def test_unknown_shared_return_cannot_be_certified_from_shape(self):
        self.model.experts.unknown_shared = True
        with torch.inference_mode():
            x, output = self.invoke()
            self.assertIn('MOE_SHARED_RETURN_HAS_NO_COMPLETE_ACTUAL_OPERATOR_CHAIN',
                          self.event()['shared_moe_row_diagnostic']['reason'])
            self.assertFalse(self.graph.query(output[0])['verified'])
            self.assertFalse(self.graph.query(output[1])['verified'])
            self.assertFalse(self.graph.query(x)['verified'])
            self.assertTrue(self.event()['shared_moe_row_diagnostic']['revoked_proof_nodes'])
            self.assertEqual(self.graph.epochs[self.graph.storage_key(metadata(x))], 1)

    def test_real_dispatch_override_does_not_borrow_reviewed_source_frame(self):
        foreign = torch.library.Library('cpu_quant_shared_moe_rows', 'FRAGMENT')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            foreign.impl('shared', cpu_shared, 'CPU', allow_override=True)
        try:
            self.assert_refused('MOE_CURRENT_KERNEL_SHADOWED_OR_UNPROVEN')
        finally:
            foreign._destroy()

    def test_foreign_model_mapping_cannot_borrow_same_shape_input(self):
        global _context
        other = CpuModel().eval()
        _context = NS(no_compile_layers={'cpu-layer': other.experts})
        with torch.inference_mode():
            _, output = self.invoke()
            self.assertIn('MOE_ACTUAL_FORWARD_CONTEXT_LAYER_LOOKUP_REQUIRED',
                          self.event()['shared_moe_row_diagnostic']['reason'])
            self.assertFalse(self.graph.query(output[0])['verified'])
            self.assertEqual(other.experts.forward_count, 1)
            self.assertEqual(self.model.experts.forward_count, 0)

    def test_replaced_shared_module_with_same_structure_is_refused(self):
        replacement = torch.nn.Module().eval()
        self.model.shared = replacement
        self.model.experts._shared_experts = replacement
        self.model.experts.runner.shared_experts = replacement
        self.assert_refused('MOE_ACTUAL_MODEL_RUNNER_SHARED_OWNER_CHANGED')

    def test_layer_and_runner_cannot_use_different_equal_configuration_objects(self):
        self.model.experts.moe_config = NS(**vars(self.model.experts.moe_config))
        self.assert_refused('MOE_SERIAL_UNTRANSFORMED_TP1_SHARED_BRANCH_REQUIRED')

    def test_input_router_storage_alias_is_not_a_reviewed_shared_call(self):
        with torch.inference_mode():
            x = torch.ones(8, 4)
            self.begin(x)
            shared, routed = self.model(x, x)
            self.assertIn('MOE_ACTUAL_SAME_SHARED_INPUT_REQUIRED', self.event()['shared_moe_row_diagnostic']['reason'])
            self.assertFalse(self.graph.query(shared)['verified'])
            self.assertFalse(self.graph.query(routed)['verified'])
            self.assertTrue(self.event()['invalidated_storages'])

    def test_report_diagnostic_has_no_live_tensor_or_mutable_proof_reference(self):
        with torch.inference_mode():
            _, (shared, _) = self.invoke()
            report = self.graph.report()
            row = next(e for e in report['events'] if 'shared_moe_row_diagnostic' in e)
            row['shared_moe_row_diagnostic']['actual_owner']['layer_path'] = 'foreign'
            row['shared_moe_row_diagnostic']['input']['shape'][0] = 123
            self.assertTrue(self.graph.query(shared)['verified'])
            self.assertEqual(self.event()['shared_moe_row_diagnostic']['actual_owner']['layer_path'], 'experts')
            self.assertEqual(self.event()['shared_moe_row_diagnostic']['input']['shape'][0], 8)

    def test_training_refused_without_changing_module_flag(self):
        self.model.train()
        self.assert_refused('MOE_ACTUAL_EVAL_INFERENCE_THREAD_REQUIRED')
        self.assertTrue(self.model.training)

    def test_parallel_overlap_and_transform_fields_stay_unsupported(self):
        for owner, field, value in ((self.model.experts, 'multistream_overlap_shared_expert', True),
            (self.model.experts, 'multistream_overlap_gate', True),
            (self.model.experts.runner, 'routed_input_transform', object()),
            (self.model.experts.runner, 'enable_dbo', True),
            (self.model.experts.runner.moe_config, 'sp_size', 2)):
            with self.subTest(field=field):
                before = getattr(owner, field)
                setattr(owner, field, value)
                self.model.experts.forward_count = 0
                _queries.clear()
                self.assert_refused('MOE_SERIAL_UNTRANSFORMED_TP1_SHARED_BRANCH_REQUIRED')
                setattr(owner, field, before)

    def test_original_exception_unwinds_all_owned_observer_resources(self):
        self.model.experts.throw = True
        with torch.inference_mode():
            with self.assertRaisesRegex(RuntimeError, 'CPU_MOE_ORIGINAL_FAILURE'):
                self.invoke()
            self.assertEqual(self.model.experts.forward_count, 1)
            report = self.graph.end()
            self.assertFalse(report['complete'])
            self.assertEqual(report['source_boundaries']['owned_nested_dispatch_modes'], 0)
            self.assertEqual(report['source_boundaries']['owned_profile_callbacks'], 0)
        self.assertIsNone(sys.getprofile())
        self.assertEqual(_get_current_dispatch_mode_stack(), [])


if __name__ == '__main__':
    unittest.main()
