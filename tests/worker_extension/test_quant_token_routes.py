"""Real CPU dispatcher/profile tests, not CANN numerical qualification."""
import copy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
import warnings
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode_stack

from demoncat.runtime.vllm_ascend.worker_extension.quant_row_provenance import RequestRowProvenance
from demoncat.runtime.vllm_ascend.worker_extension.quant_source_rows import SourceBoundRows, source_bindings
from demoncat.runtime.vllm_ascend.worker_extension.quant_token_routes import (
    IntegerRouteObserver, TokenRouteProof, SCHEMAS, _integer_bytes)


_LIB = torch.library.Library('cpu_quant_integer_routes', 'DEF')
_LIB.define('shared(Tensor(a0!) hidden_states, Tensor logits, Tensor shared_input, str name) -> (Tensor, Tensor)')
for _role in ('topk', 'routing', 'gmm1', 'gmm2', 'unpermute'):
    _LIB.define(SCHEMAS[_role].split('::', 1)[1])


def _op(role):
    return getattr(torch.ops.cpu_quant_integer_routes, SCHEMAS[role].split('::')[1].split('(')[0]).default


def fake_topk(x, k, k_group, group_count, group_select_mode, renorm, norm_type,
              out_flag, routed_scaling_factor, eps, bias_opt=None):
    _model.actual['topk'] += 1
    ids = torch.tensor([list(range(6)) for _ in range(8)], dtype=torch.int32)
    if _model.bad_dtype:
        ids = ids.float()
    if _model.bad_shape:
        ids = ids[:, :5].contiguous()
    return torch.ones((8, 6)), ids, torch.ones((8, 64))


def fake_routing(x, expert_idx, **kwargs):
    _model.actual['routing'] += 1
    ids = expert_idx.flatten().to(torch.int64)
    order = torch.argsort(ids, stable=True)
    if _model.reverse_ties:
        order = torch.cat([order[expert * 8:(expert + 1) * 8].flip(0) for expert in range(6)])
    inverse = torch.empty(48, dtype=torch.int32)
    inverse[order] = torch.arange(48, dtype=torch.int32)
    if _model.negative:
        inverse[1] *= -1
    return (torch.zeros((48, 4), dtype=torch.int8), inverse,
            torch.bincount(ids, minlength=64), torch.ones(48))


def fake_gmm1(x, weight, group_list, weight_scale, x_scale, **kwargs):
    _model.actual['gmm1'] += 1
    if _model.original_failure:
        raise RuntimeError('CPU_ACTUAL_GROUPED_MLP_FAILURE')
    return torch.zeros((48, 4), dtype=torch.int8), torch.ones(48), torch.ones(())


def fake_gmm2(x, weight, **kwargs):
    _model.actual['gmm2'] += 1
    return [torch.zeros((48, 4))]


def fake_unpermute(permuted_tokens, sorted_indices, probs=None, padded_mode=False, restore_shape=None):
    _model.actual['unpermute'] += 1
    return torch.zeros((8, 4))


def select_experts(hidden_states, router_logits):
    return _op('topk')(router_logits, 6, 1, 1, 0, 1, 0, False, 1., 1e-20)


class CpuDispatcher:
    def token_dispatch(self, token_dispatch_input):
        x, ids = token_dispatch_input.hidden_states, token_dispatch_input.topk_ids
        out = _op('routing')(x, ids, active_num=48, expert_num=64,
            expert_tokens_num_type=1, expert_tokens_num_flag=True, active_expert_range=[0, 64],
            quant_mode=1, row_idx_type=_model.row_idx_type)
        return out

    def token_combine(self, hidden_states, combine_metadata):
        raw = combine_metadata.raw
        if _model.mutate_indices:
            raw.add_(1)
        if _model.replace_indices:
            raw = raw.clone()
        return _op('unpermute')(hidden_states, torch.abs(raw), combine_metadata.probs,
                              padded_mode=_model.padded)


class CpuPipeline:
    def fused_experts(self, hidden_states, topk_ids, probs):
        dispatch_input = NS(hidden_states=hidden_states, topk_ids=topk_ids, topk_weights=probs,
            routing=NS(apply_router_weight_on_input=False, expert_map=None, pertoken_scale=None,
                       global_redundant_expert_num=0))
        routed, indices, counts, scale = self.dispatcher.token_dispatch(dispatch_input)
        result = self._apply_mlp(routed, counts, scale)
        return self.dispatcher.token_combine(result, NS(raw=indices, probs=probs))

    def _apply_mlp(self, routed, counts, scale):
        cumulative = torch.cumsum(counts, 0)
        if _model.swap_counts:
            cumulative = cumulative.clone()
        first, dynamic, auxiliary = _op('gmm1')(routed, torch.ones((64, 4, 8), dtype=torch.int8),
            cumulative, torch.ones((64, 8)), scale)
        # Same as the actual source83 pipeline: retire the consumed temporary.
        if _model.retire_routed:
            routed.set_(torch.empty(0, dtype=torch.int8))
        return _op('gmm2')([first], [torch.ones((64, 4, 4), dtype=torch.int8)],
            per_token_scale=[dynamic], group_list=counts, group_list_type=1)[0]


def shared(hidden_states, logits, shared_input, name):
    probs, ids, auxiliary = select_experts(hidden_states, logits)
    routed = _model.pipeline.fused_experts(hidden_states, ids, probs)
    return hidden_states.clone(), routed


_LIB.impl('shared', shared, 'CPU')
for _role, _implementation in [('topk', fake_topk), ('routing', fake_routing), ('gmm1', fake_gmm1),
                                ('gmm2', fake_gmm2), ('unpermute', fake_unpermute)]:
    _LIB.impl(SCHEMAS[_role].split('::')[1].split('(')[0], _implementation, 'CPU')


class CpuModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pipeline = CpuPipeline()
        self.pipeline.dispatcher = CpuDispatcher()
        self.actual = dict.fromkeys(('topk', 'routing', 'gmm1', 'gmm2', 'unpermute'), 0)
        self.row_idx_type = 0
        self.bad_dtype = self.bad_shape = self.reverse_ties = self.negative = self.padded = False
        self.mutate_indices = self.replace_indices = self.swap_counts = self.original_failure = False
        self.retire_routed = True

    def forward(self, x, logits):
        return torch.ops.cpu_quant_integer_routes.shared(x, logits, x, 'cpu-layer')


class IntegerRouteTests(unittest.TestCase):
    def setUp(self):
        global _model
        self.model = _model = CpuModel().eval()
        self.graph = RequestRowProvenance(max_calls=1024)
        roles = {'shared': 'moe_opaque', 'select_experts': 'route_selector',
            'CpuPipeline.fused_experts': 'route_pipeline', 'CpuPipeline._apply_mlp': 'route_mlp',
            'CpuDispatcher.token_dispatch': 'route_dispatch', 'CpuDispatcher.token_combine': 'route_combine'}
        bindings, sources = source_bindings({__name__: (hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), roles)})
        aliases = {_op(role): SCHEMAS[role].split('(')[0].replace('::', '.') + '.default'
                   for role in ('topk', 'routing', 'gmm1', 'gmm2', 'unpermute')}
        aliases[torch.ops.cpu_quant_integer_routes.shared.default] = 'vllm.moe_forward_shared.default'
        self.source = SourceBoundRows(self.graph, self.model, bindings, sources, simulation=True, operator_aliases=aliases)
        self.source.routes = self.observer = IntegerRouteObserver(self.source)
        self.graph.boundaries = self.source
        self.rows = dict(request_ids=['request-' + str(i) for i in range(8)], row_token_positions=[12] * 8)
        self.addCleanup(self.cleanup)

    def cleanup(self):
        if self.graph.active:
            self.graph.end()
        self.assertIsNone(sys.getprofile())
        self.assertEqual(_get_current_dispatch_mode_stack(), [])

    def invoke(self):
        x, logits = torch.ones((8, 4)), torch.ones((8, 64))
        self.graph.begin('actual-cpu-forward')
        self.source.begin(self.rows)
        self.graph.seed(x, dict(verified=True, actual_rows=self.rows, source='EXPLICIT_CPU_FIXTURE'))
        return self.model(x, logits)

    def record(self):
        return self.observer.report()['records'][0]

    def test_actual_dispatcher_captures_1600_integer_bytes_without_binding_output(self):
        with torch.inference_mode():
            result = self.invoke()
            row = self.record()
            self.assertIsNone(row['first_failure'], row)
            self.assertTrue(row['token_route_proof']['integer_map_valid'])
            self.assertFalse(row['token_route_proof']['routed_output_row_proof'])
            self.assertFalse(self.graph.query(result[1])['verified'])
            self.assertEqual(self.model.actual, dict.fromkeys(self.model.actual, 1))
            self.assertEqual(row['readback_bytes'], 1600)
            self.assertEqual(len(self.observer.helper_events), 5)
            self.assertTrue(all(call['status'] == 'COMPLETE' and 1 <= len(call['operations']) <= 4
                and call['operations'][0]['operator'] == 'aten._to_copy.default'
                and all(op['original_calls'] == 1 for op in call['operations']) for call in self.observer.helper_events))
            self.assertEqual(self.observer.report()['live_tensor_refs'], 0)
            self.graph.end()

    def test_nonstable_expert_ties_use_actual_permutation(self):
        self.model.reverse_ties = True
        with torch.inference_mode():
            self.invoke()
            self.assertIsNone(self.record()['first_failure'], self.record())
            proof = self.record()['token_route_proof']
            self.assertEqual(proof['sorted_slots'][0]['source_token'], 7)

    def assert_refused(self, option, reason):
        setattr(self.model, option, True)
        with torch.inference_mode():
            self.invoke()
            row = self.record()
            self.assertIn(reason, row['first_failure'], row)
            self.assertIsNone(row['token_route_proof'])
            self.assertEqual(self.observer.report()['live_tensor_refs'], 0)

    def test_negative_cannot_be_laundered_by_abs(self):
        self.assert_refused('negative', 'NONNEGATIVE_PERMUTATION')

    def test_padded_unpermute_remains_unknown(self):
        self.assert_refused('padded', 'UNPADDED_UNPERMUTE')

    def test_same_values_replacement_indices_refused(self):
        self.assert_refused('replace_indices', 'OBJECT_OR_WRITE_EPOCH_CHANGED:raw_indices')

    def test_inference_no_version_write_refused(self):
        self.assert_refused('mutate_indices', 'OBJECT_OR_WRITE_EPOCH_CHANGED:raw_indices')

    def test_same_values_replacement_grouplist_refused(self):
        self.assert_refused('swap_counts', 'OBJECT_OR_WRITE_EPOCH_CHANGED:cumulative_counts')

    def test_float_metadata_never_copied(self):
        self.model.bad_dtype = True
        with torch.inference_mode():
            self.invoke()
            self.assertIn('INTEGER_ONLY', self.record()['first_failure'])
            self.assertEqual(self.observer.readback_bytes, 0)
            self.assertEqual(self.observer.helper_events, [])

    def test_unreviewed_scatter_not_reinterpreted_as_gather(self):
        self.model.row_idx_type = 1
        with torch.inference_mode():
            self.invoke()
            self.assertIn('GATHER_COUNT_SUPPORTED', self.record()['first_failure'])

    def test_original_failure_retains_record_and_releases_refs(self):
        self.model.original_failure = True
        with torch.inference_mode():
            with self.assertRaisesRegex(RuntimeError, 'ACTUAL_GROUPED_MLP_FAILURE'):
                self.invoke()
            row = self.record()
            self.assertIn('ORIGINAL_CALL_FAILED', row['first_failure'])
            self.assertFalse(row['stages']['gmm1']['completed'])
            self.assertEqual(self.observer.report()['live_tensor_refs'], 0)
            self.assertEqual(self.observer.report()['live_parent_refs'], 0)

    def test_readback_failure_is_retained_and_model_still_runs_once(self):
        with torch.inference_mode(), patch.object(torch.ops.aten._to_copy, 'default', side_effect=RuntimeError('CPU_COPY_FAILURE')):
            self.invoke()
            self.assertIn('CPU_COPY_FAILURE', self.record()['first_failure'])
            self.assertEqual(self.observer.readback_bytes, 192)
            self.assertEqual(self.observer.helper_events[0]['status'], 'FAILED')
            self.assertEqual(self.model.actual, dict.fromkeys(self.model.actual, 1))
            self.assertEqual(self.source.owned_modes, [])

    def test_reports_cannot_mutate_proof_or_snapshots(self):
        with torch.inference_mode():
            self.invoke()
            original = self.record()
            report = self.observer.report()
            report['records'][0]['snapshots']['topk_ids']['values'][0] = 63
            report['records'][0]['token_route_proof']['sorted_slots'][0]['source_token'] = 123
            report['helper_calls'][0]['operations'].clear()
            self.assertEqual(self.record(), original)

    def test_two_inference_forwards_release_and_reset(self):
        with torch.inference_mode():
            for _ in range(2):
                self.invoke()
                self.assertIsNone(self.record()['first_failure'], self.record())
                self.assertEqual(self.observer.total_parents, 1)
                self.graph.end()

    def test_missing_stage_digest_tamper_wrong_forward_and_arithmetic_rejected(self):
        with torch.inference_mode():
            self.invoke()
            pristine = self.record()
        for mutation, pattern in [
            (lambda d: d['stages'].pop('gmm2'), 'CALLS_INCOMPLETE'),
            (lambda d: d['snapshots']['topk_ids']['values'].__setitem__(0, 63), 'BYTES_CHANGED'),
            (lambda d: d['snapshots']['topk_ids'].__setitem__('forward_id', 'other'), 'IDENTITY_OR_LAYOUT'),
            (lambda d: d['snapshots']['topk_ids'].__setitem__('epoch_after', 99), 'IDENTITY_OR_LAYOUT')]:
            row = copy.deepcopy(pristine)
            mutation(row)
            with self.subTest(pattern=pattern), self.assertRaisesRegex(RuntimeError, pattern):
                TokenRouteProof.validate(row)
        for name, index, value, pattern in [('expert_counts', 0, 9, 'COUNTS_DO_NOT_MATCH'),
            ('cumulative_counts', 1, 17, 'CUMSUM_DOES_NOT_MATCH'),
            ('raw_indices', 1, 0, 'NONNEGATIVE_PERMUTATION')]:
            row = copy.deepcopy(pristine)
            item = row['snapshots'][name]
            item['values'][index] = value
            item['sha256'] = hashlib.sha256(_integer_bytes(item['values'], item['dtype'])).hexdigest()
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, pattern):
                TokenRouteProof.validate(row)

    def test_parent_budget_retains_denominator_and_runs_unobserved_parent_once(self):
        self.observer.max_parents = 1
        with torch.inference_mode():
            self.invoke()
            self.model(torch.ones((8, 4)), torch.ones((8, 64)))
            report = self.observer.report()
            self.assertEqual((report['total_parents'], report['overflow'], len(report['records'])), (2, 1, 1))
            self.assertEqual(self.model.actual, dict.fromkeys(self.model.actual, 2))
            self.assertEqual(report['readback_bytes'], 1600)
            self.assertEqual(report['live_parent_refs'], 0)
            self.assertEqual(sum(event.get('integer_route_evidence', {}).get('status') == 'BUDGET_REFUSED'
                                 for event in self.graph.events), 1)

    def test_actual_registration_override_refused_before_integer_copy(self):
        library = torch.library.Library(_LIB.ns, 'FRAGMENT')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                library.impl('moe_gating_top_k', fake_topk, 'CPU', allow_override=True)
            with torch.inference_mode():
                self.invoke()
                self.assertIn('SHADOWED_DISPATCH', self.record()['first_failure'])
                self.assertEqual(self.observer.readback_bytes, 0)
                self.assertEqual(self.model.actual, dict.fromkeys(self.model.actual, 1))
        finally:
            library._destroy()

    def test_missing_source_boundary_never_uses_matching_shape(self):
        self.source.bindings.pop(CpuDispatcher.token_dispatch.__code__)
        with torch.inference_mode():
            self.invoke()
            self.assertIn('ACTUAL_SOURCE_FRAME_REQUIRED:routing', self.record()['first_failure'])
            self.assertEqual(self.observer.readback_bytes, 192)

    def test_unknown_actual_schema_keeps_raw_call_not_integer_proof(self):
        real = self.observer._registration
        def changed(func, role):
            if role == 'routing':
                raise RuntimeError('ROUTE_UNREVIEWED_SCHEMA_OR_ALIAS:routing')
            return real(func, role)
        with torch.inference_mode(), patch.object(self.observer, '_registration', changed):
            self.invoke()
            self.assertIn('UNREVIEWED_SCHEMA_OR_ALIAS', self.record()['first_failure'])
            self.assertEqual(self.model.actual['routing'], 1)
            self.assertTrue(any(event['operator'] == str(_op('routing')) and event['original_calls'] == 1
                                for event in self.graph.events))

    def test_training_thread_is_not_read_back(self):
        self.invoke()
        self.assertIn('EVAL_INFERENCE_THREAD', self.record()['first_failure'])
        self.assertEqual(self.observer.readback_bytes, 0)

    def test_same_version_content_change_during_copy_refused(self):
        real = torch.ops.aten._to_copy.default
        def changed(tensor, **kwargs):
            result = real(tensor, **kwargs)
            if tensor.dtype == torch.int32:
                self.graph._invalidate([tensor])
            return result
        with torch.inference_mode(), patch.object(torch.ops.aten._to_copy, 'default', changed):
            self.invoke()
            self.assertIn('CHANGED_DURING_READBACK', self.record()['first_failure'])
            self.assertEqual(self.observer.helper_events[0]['status'], 'FAILED')

    def test_cleanup_that_removed_mode_then_raised_keeps_first_failure(self):
        real = self.source._release_mode
        raised = False
        def fail_after_pop(mode):
            nonlocal raised
            real(mode)
            if type(mode).__name__ == 'ReadbackMode' and not raised:
                raised = True
                raise RuntimeError('CPU_POST_MODE_EXIT_FAILURE')
        with torch.inference_mode(), patch.object(self.source, '_release_mode', fail_after_pop):
            self.invoke()
            self.assertIn('POST_MODE_EXIT_FAILURE', self.record()['first_failure'])
            self.assertEqual(self.observer.report()['live_tensor_refs'], 0)
            self.assertEqual(self.source.owned_modes, [])

    def test_proof_snapshot_is_frozen_and_reports_are_detached(self):
        with torch.inference_mode():
            self.invoke()
            evidence = self.record()
            proof = TokenRouteProof.validate(evidence)
            source = proof.source_to_sorted
            evidence['snapshots']['raw_indices']['values'][0] = 999
            returned = proof.report()
            returned['sorted_slots'][0]['expert'] = 999
            self.assertEqual(proof.source_to_sorted, source)
            self.assertNotEqual(proof.report()['sorted_slots'][0]['expert'], 999)
            with self.assertRaises(AttributeError):
                proof.forward_id = 'changed'

    def test_failed_helper_exit_retains_handle_until_same_owner_retries(self):
        real = self.source._release_mode
        raised = False
        def fail_before_pop(mode):
            nonlocal raised
            if type(mode).__name__ == 'ReadbackMode' and not raised:
                raised = True
                raise RuntimeError('CPU_PRE_MODE_EXIT_FAILURE')
            return real(mode)
        with torch.inference_mode(), patch.object(self.source, '_release_mode', fail_before_pop):
            with self.assertRaisesRegex(RuntimeError, 'UNEXPECTED_READBACK_HELPER'):
                self.invoke()
            self.assertTrue(any(type(mode).__name__ == 'ReadbackMode' for mode in self.source.owned_modes))
            self.assertIn('CPU_PRE_MODE_EXIT_FAILURE', self.record()['first_failure'])
            self.assertEqual(self.observer.report()['live_tensor_refs'], 1)
            self.assertEqual(self.observer.report()['retained_helper_modes'], 1)
        self.graph.end()
        self.assertEqual(self.source.owned_modes, [])
        self.assertEqual(self.observer.report()['live_tensor_refs'], 0)
        self.assertEqual(int((torch.ones(1) + 1)[0]), 2)

    def test_foreign_mode_is_never_removed_and_cleanup_can_retry(self):
        class Foreign(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))
        foreign = Foreign()
        real = self.source._release_mode
        inserted = False
        def insert_foreign(mode):
            nonlocal inserted
            if type(mode).__name__ == 'ReadbackMode' and not inserted:
                inserted = True
                foreign.__enter__()
            return real(mode)
        try:
            with torch.inference_mode(), patch.object(self.source, '_release_mode', insert_foreign):
                with self.assertRaisesRegex(RuntimeError, 'UNEXPECTED_READBACK_HELPER'):
                    self.invoke()
            with self.assertRaisesRegex(RuntimeError, 'MODE_OWNER_CHANGED'):
                self.graph.end()
            self.assertTrue(self.source.active)
            self.assertTrue(any(mode is foreign for mode in _get_current_dispatch_mode_stack()))
            self.assertTrue(self.source.owned_modes)
        finally:
            if any(mode is foreign for mode in _get_current_dispatch_mode_stack()):
                foreign.__exit__(None, None, None)
        self.graph.end()
        self.assertEqual(self.source.owned_modes, [])


if __name__ == '__main__':
    unittest.main()
