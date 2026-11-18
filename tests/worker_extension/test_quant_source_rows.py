"""Actual CPU dispatcher/profile tests, never CANN or natural row qualification."""
import hashlib
from pathlib import Path
import sys
import types
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode, _get_current_dispatch_mode_stack

from demoncat.runtime.vllm_ascend.worker_extension.quant_row_provenance import RequestRowProvenance, metadata
from demoncat.runtime.vllm_ascend.worker_extension.quant_source_rows import SourceBoundRows, source_bindings


_OPAQUE = torch.library.Library('cpu_quant_opaque_rows', 'DEF')
_OPAQUE.define('mla(Tensor hidden_states, bool need_gather_q_kv, Tensor(a!) output, str layer_name) -> ()')
_OPAQUE.define('unknown(Tensor input) -> Tensor')


def cpu_unknown(input):
    _inside_queries.append(_query_graph.query(input))
    return input.clone()


_OPAQUE.impl('unknown', cpu_unknown, 'CPU')


def cpu_mla_forward(hidden_states, need_gather_q_kv, output, layer_name):
    self = _current_model
    forward_context = NS(no_compile_layers={layer_name: self})
    kv_cache = self.mla_attn.kv_cache[0]
    output.copy_(hidden_states + 1)
    if self.raise_inside:
        raise RuntimeError('explicit CPU fixture failure')


_OPAQUE.impl('mla', cpu_mla_forward, 'CPU')


class CpuOpaque(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mla_attn = NS(kv_cache=[object()])
        self.raise_inside = False

    def forward(self, x):
        output = torch.empty_like(x)
        torch.ops.cpu_quant_opaque_rows.mla(x, False, output, 'fixture-layer')
        return output


class CpuRows(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W_UK_T = torch.ones(2, 4, 3)

    def query(self, x):
        return torch.bmm(x.transpose(0, 1), self.W_UK_T).transpose(0, 1)

    def no_gather(self, x, label):
        return x if label is False else torch.roll(x, 1, 0)


class SourceRowsTests(unittest.TestCase):
    def setUp(self):
        self.model = CpuRows()
        self.graph = RequestRowProvenance(max_calls=512)
        roles = {'CpuRows.query': 'mla_q', 'CpuRows.no_gather': 'no_gather'}
        pin = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.reviewed = {__name__: (pin, roles)}
        bindings, sources = source_bindings(self.reviewed)
        self.boundary = SourceBoundRows(self.graph, self.model, bindings, sources, simulation=True)
        self.boundary.constants[(id(self.model), 'W_UK_T')] = (
            self.model, self.model.W_UK_T, metadata(self.model.W_UK_T))
        self.graph.boundaries = self.boundary
        self.rows = dict(request_ids=['cpu-' + str(i) for i in range(8)], row_token_positions=list(range(8)))
        self.addCleanup(lambda: self.graph.end() if self.graph.active else None)

    def begin(self, x):
        self.graph.begin('cpu-only-forward')
        self.boundary.begin(self.rows)
        self.graph.seed(x, dict(verified=True, actual_rows=self.rows, source='EXPLICIT_CPU_FIXTURE'))

    def test_reviewed_head_axis_bmm_returns_actual_request_axis(self):
        x = torch.ones(8, 2, 4)
        self.begin(x)
        y = self.model.query(x)
        self.assertTrue(self.graph.query(y)['verified'])
        self.assertEqual(self.graph.query(y)['actual_rows'], self.rows)
        report = self.graph.end()
        self.assertTrue(report['complete'], report['source_boundaries'])
        self.assertTrue(any(node['row_axis'] == 1 for node in report['nodes']))
        self.assertIsNone(sys.getprofile())

    def test_same_geometry_replaced_weight_is_unverified(self):
        x = torch.ones(8, 2, 4)
        self.begin(x)
        self.model.W_UK_T = self.model.W_UK_T.clone()
        self.assertFalse(self.graph.query(self.model.query(x))['verified'])

    def test_changed_loaded_globals_or_copied_code_is_refused(self):
        original = CpuRows.query
        with patch.object(CpuRows, 'query', types.FunctionType(original.__code__, dict(original.__globals__))):
            with self.assertRaisesRegex(RuntimeError, 'FOREIGN_FUNCTION'):
                source_bindings(self.reviewed)

    def test_function_replacement_after_binding_poisoned_before_row_query(self):
        x = torch.ones(8, 2, 4)
        self.begin(x)
        original = CpuRows.query
        with patch.object(CpuRows, 'query', types.FunctionType(original.__code__, original.__globals__)):
            y = self.model.query(x)
        self.assertFalse(self.graph.query(y)['verified'])
        self.assertTrue(self.boundary.errors)

    def test_unproven_row_operand_does_not_inherit_inplace_rows(self):
        with torch.inference_mode():
            x = torch.ones(8, 4)
            other = torch.roll(x, 1, 0)
            self.begin(x)
            x.add_(other)
            self.assertFalse(self.graph.query(x)['verified'])

    def test_proven_both_operands_and_scalar_mutations_keep_rows_but_invalidate_aliases(self):
        with torch.inference_mode():
            x = torch.ones(8, 4)
            self.begin(x)
            alias = x.view(8, 2, 2)
            other = x.clone()
            x.add_(other)
            self.assertTrue(self.graph.query(x)['verified'])
            self.assertFalse(self.graph.query(alias)['verified'])
            x.mul_(2.0)
            self.assertTrue(self.graph.query(x)['verified'])

    def test_actual_full_copy_binds_destination_and_same_extent_base_only(self):
        with torch.inference_mode():
            x = torch.ones(8, 4)
            target = torch.empty_like(x)
            self.begin(x)
            target[:8].copy_(x)
            self.assertTrue(self.graph.query(target)['verified'])
            self.assertTrue(self.graph.query(x)['verified'])
            partial = torch.empty(9, 4)
            partial[:8].copy_(x)
            self.assertFalse(self.graph.query(partial)['verified'])

    def test_default_graph_does_not_enable_axis_bridge(self):
        self.graph.boundaries = None
        x = torch.ones(8, 2, 4)
        self.graph.begin('plain')
        self.graph.seed(x, dict(verified=True, actual_rows=self.rows))
        self.assertFalse(self.graph.query(x.transpose(0, 1))['verified'])

    def test_profile_conflict_refuses_without_replacing_owner(self):
        owner = lambda *args: None
        self.graph.begin('conflict')
        sys.setprofile(owner)
        try:
            with self.assertRaisesRegex(RuntimeError, 'EXCLUSIVE'):
                self.boundary.begin(self.rows)
            self.assertIs(sys.getprofile(), owner)
        finally:
            sys.setprofile(None)

    def opaque(self, model=None):
        global _current_model
        self.model = _current_model = CpuOpaque() if model is None else model
        pin = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        bindings, sources = source_bindings({__name__: (pin, {'cpu_mla_forward': 'mla_opaque'})})
        self.boundary = SourceBoundRows(self.graph, self.model, bindings, sources, simulation=True,
            operator_aliases={torch.ops.cpu_quant_opaque_rows.mla.default: 'vllm.mla_forward.default'})
        self.graph.boundaries = self.boundary

    def test_actual_opaque_callback_observes_inner_original_operations_once(self):
        self.opaque()
        with torch.inference_mode():
            x = torch.ones(8, 4)
            self.begin(x)
            y = self.model(x)
            self.assertTrue(self.graph.query(y)['verified'])
            self.assertTrue(self.graph.query(x)['verified'])
            report = self.graph.end()
        self.assertTrue(report['complete'], report)
        inner = [event for event in report['events'] if event['operator'] in ('aten.add.Tensor', 'aten.copy_.default')]
        self.assertEqual(len(inner), 2)
        self.assertTrue(all(event['original_calls'] == 1 for event in inner))

    def test_opaque_exception_removes_owned_nested_mode_and_poisoned_proof(self):
        self.opaque()
        with torch.inference_mode():
            x = torch.ones(8, 4)
            self.begin(x)
            self.model.raise_inside = True
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                self.model(x)
            self.assertFalse(self.graph.query(x)['verified'])
            self.assertFalse(self.graph.end()['complete'])
        self.assertIsNone(sys.getprofile())

    def test_source_refusal_before_opaque_call_keeps_original_once_and_clean_unwind(self):
        self.opaque()
        with torch.inference_mode():
            x = torch.ones(8, 4)
            self.begin(x)
            original = cpu_mla_forward
            with patch(__name__ + '.cpu_mla_forward', types.FunctionType(original.__code__, original.__globals__)):
                output = self.model(x)  # Registered CPU kernel still calls its original function.
            self.assertTrue(torch.equal(output, x + 1))
            self.assertFalse(self.graph.query(output)['verified'])
            self.assertFalse(self.graph.end()['complete'])
        self.assertIsNone(sys.getprofile())

    def test_unreviewed_opaque_call_cannot_lend_unchanged_input_proof_to_an_inner_projection(self):
        global _inside_queries, _query_graph
        _inside_queries, _query_graph = [], self.graph
        x = torch.ones(8, 4)
        self.begin(x)
        torch.ops.cpu_quant_opaque_rows.unknown(x)
        self.assertEqual(len(_inside_queries), 1)
        self.assertFalse(_inside_queries[0]['verified'])
        self.assertTrue(self.graph.query(x)['verified'])

    def test_foreign_nested_mode_retains_owner_until_foreign_removal_and_retry(self):
        class ForeignMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))
        foreign = ForeignMode()
        class ForeignAtReturn(CpuOpaque):
            def __getattribute__(self, name):
                if name == 'raise_inside':
                    foreign.__enter__()
                    return False
                return super().__getattribute__(name)
        # Keep the reviewed CPU custom-op kernel unchanged: the new mode is
        # installed by the actual model object's final property access.
        self.opaque(ForeignAtReturn())
        x = torch.ones(8, 4)
        self.begin(x)
        output = self.model(x)
        owned = self.boundary.owned_modes[0]
        self.assertIs(_get_current_dispatch_mode(), self.graph)
        self.assertEqual((len(self.boundary.pending), len(self.boundary.stack)), (0, 0))
        try:
            with self.assertRaisesRegex(RuntimeError, 'NESTED_MODE_OWNER_CHANGED'):
                self.graph.end()
            self.assertTrue(self.graph.active and self.boundary.active)
            self.assertTrue(self.graph.report()['cleanup_pending'])
            self.assertEqual(self.boundary.report()['owned_nested_dispatch_modes'], 1)
            self.assertFalse(self.graph.report()['complete'])
            self.assertFalse(self.graph.query(output)['verified'])
            self.assertIs(_get_current_dispatch_mode(), foreign)
            self.assertIs(_get_current_dispatch_mode_stack()[-2], owned)
            with self.assertRaisesRegex(RuntimeError, 'NESTED_MODE_OWNER_CHANGED'):
                self.graph.end()
            self.assertIs(_get_current_dispatch_mode(), foreign)
        finally:
            if _get_current_dispatch_mode() is foreign:
                foreign.__exit__(None, None, None)  # Only the foreign owner removes itself.
        report = self.graph.end()
        self.assertFalse(report['active'] or self.boundary.active)
        self.assertEqual(self.boundary.report()['owned_nested_dispatch_modes'], 0)
        self.assertEqual(_get_current_dispatch_mode_stack(), [])
        self.assertIsNone(sys.getprofile())
        self.assertTrue(torch.equal(output, x + 1))

    def test_nested_mode_exit_before_pop_retains_handle_and_same_owner_retries(self):
        self.opaque()
        x = torch.ones(8, 4)
        self.begin(x)
        real_mode = self.boundary._mode
        def mode_with_failure():
            mode = real_mode()
            mode.__exit__ = lambda *args: (_ for _ in ()).throw(RuntimeError('TRANSIENT_NESTED_EXIT'))
            return mode
        with patch.object(self.boundary, '_mode', side_effect=mode_with_failure):
            output = self.model(x)
        mode = self.boundary.owned_modes[0]
        with self.assertRaisesRegex(RuntimeError, 'TRANSIENT_NESTED_EXIT'):
            self.graph.end()
        self.assertTrue(self.graph.active and self.boundary.active)
        self.assertIs(_get_current_dispatch_mode(), mode)
        del mode.__exit__
        self.graph.end()
        self.assertEqual(_get_current_dispatch_mode_stack(), [])
        self.assertIsNone(sys.getprofile())
        self.assertTrue(torch.equal(output, x + 1))

    def test_nested_mode_exit_after_pop_records_failure_without_double_exit(self):
        self.opaque()
        x = torch.ones(8, 4)
        self.begin(x)
        real_mode = self.boundary._mode
        def mode_with_failure():
            mode = real_mode()
            original_exit = mode.__exit__
            def pop_then_raise(*args):
                original_exit(*args)
                raise RuntimeError('AFTER_NESTED_POP')
            mode.__exit__ = pop_then_raise
            return mode
        with patch.object(self.boundary, '_mode', side_effect=mode_with_failure):
            self.model(x)
        self.assertEqual(self.boundary.owned_modes, [])
        self.assertTrue(any('AFTER_NESTED_POP' in error for error in self.boundary.errors))
        self.assertFalse(self.graph.end()['complete'])
        self.assertEqual(_get_current_dispatch_mode_stack(), [])
        self.assertIsNone(sys.getprofile())

    def test_metadata_release_failure_preserves_active_owner_after_profile_removal(self):
        x = torch.ones(8, 4)
        self.begin(x)
        producer = NS(release=lambda: None)
        self.boundary.metadata_producers = producer
        with patch.object(producer, 'release', side_effect=RuntimeError('TRANSIENT_RELEASE')):
            with self.assertRaisesRegex(RuntimeError, 'TRANSIENT_RELEASE'):
                self.graph.end()
        self.assertTrue(self.graph.active and self.boundary.active)
        self.assertTrue(self.boundary.profile_released)
        self.assertEqual(_get_current_dispatch_mode_stack(), [])
        self.assertIsNone(sys.getprofile())
        foreign = lambda *args: None
        sys.setprofile(foreign)
        try:
            with self.assertRaisesRegex(RuntimeError, 'QUANT_ROW_PROFILE_OWNER_CHANGED'):
                self.graph.end()
            self.assertIs(sys.getprofile(), foreign)
        finally:
            sys.setprofile(None)
        self.graph.end()
        self.assertFalse(self.graph.active or self.boundary.active)

    def test_outer_mode_after_pop_failure_retains_cleanup_owner_and_retries_once(self):
        x = torch.ones(8, 4)
        self.begin(x)
        original_exit = self.graph.__exit__
        def pop_then_raise(*args):
            original_exit(*args)
            raise RuntimeError('OUTER_AFTER_POP')
        with patch.object(self.graph, '__exit__', side_effect=pop_then_raise):
            with self.assertRaisesRegex(RuntimeError, 'OUTER_AFTER_POP'):
                self.graph.end()
        self.assertTrue(self.graph.active and self.graph.mode_released and self.boundary.active)
        self.assertFalse(self.graph.query(x)['verified'])
        self.assertEqual(_get_current_dispatch_mode_stack(), [])
        self.assertIs(sys.getprofile(), self.boundary.callback)
        self.graph.end()
        self.assertFalse(self.graph.active or self.boundary.active)
        self.assertIsNone(sys.getprofile())


if __name__ == '__main__':
    unittest.main()
