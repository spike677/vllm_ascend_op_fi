"""Actual CPU source/dispatcher joins; no real Worker or NPU qualification."""
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from demoncat.runtime.vllm_ascend.worker_extension.quant_row_provenance import RequestRowProvenance
from demoncat.runtime.vllm_ascend.worker_extension.quant_source_rows import SourceBoundRows, source_bindings


_LIB = torch.library.Library('cpu_quant_embedding_rows', 'DEF')
_LIB.define('pad(Tensor x, bool is_ep_comm=False) -> Tensor')
_LIB.define('chunk(Tensor x, Tensor residual) -> Tensor')
_LIB.define('norm(Tensor x1, Tensor x2, Tensor gamma, Tensor? beta=None, float epsilon=1e-6) -> (Tensor, Tensor, Tensor)')
_GROUP = None


def get_tp_group():
    return _GROUP


def tensor_model_parallel_all_reduce(input_):
    return get_tp_group().all_reduce(input_)


def pad(x, is_ep_comm=False):
    return tensor_model_parallel_all_reduce(x)


def chunk(x, residual):
    return residual if x.shape[0] == residual.shape[0] else residual[:x.shape[0]]


def norm(x1, x2, gamma, beta=None, epsilon=1e-6):
    residual = x1 + x2
    return residual * gamma, torch.ones(x1.shape[0], 1), residual


_LIB.impl('pad', pad, 'CPU')
_LIB.impl('chunk', chunk, 'CPU')
_LIB.impl('norm', norm, 'CPU')


class CpuGroup:
    world_size = 1
    rank_in_group = 0

    def all_reduce(self, input_):
        if self.world_size == 1:
            return input_
        return input_.clone()


class CpuEmbeddingMethod:
    padding_idx = -1
    sparse = False

    def embedding(self, layer, input_):
        return torch.ops.aten.embedding.default(layer.weight, input_, self.padding_idx, False, self.sparse)


class CpuEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(64 * 4, dtype=torch.float32).reshape(64, 4), requires_grad=False)
        self.comm_group = CpuGroup()
        self.tp_size, self.tp_rank, self.forward_type = 1, 0, None
        self.quant_method = CpuEmbeddingMethod()
        self.permute_ids = self.ep = self.skip_pad = False

    def _forward_origin(self, input_):
        indices = input_.long()
        if self.permute_ids:
            indices = torch.roll(indices, 1, 0)
        result = self.quant_method.embedding(self, indices)
        return result if self.skip_pad else torch.ops.cpu_quant_embedding_rows.pad(result, self.ep)

    def forward(self, input_):
        return self._forward_origin(input_)


class CpuModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = CpuEmbedding()
        self.gamma = torch.nn.Parameter(torch.ones(4), requires_grad=False)
        self.unknown_residual = False

    def forward(self, ids):
        embedding = self.embed(ids)
        residual = embedding.clone()
        if self.unknown_residual:
            residual = torch.roll(residual, 1, 0)
        projected = embedding * 2
        residual = torch.ops.cpu_quant_embedding_rows.chunk(projected, residual)
        y, _, residual = torch.ops.cpu_quant_embedding_rows.norm(projected, residual, self.gamma)
        return y, residual, embedding


class QuantEmbeddingRowsTests(unittest.TestCase):
    def setUp(self):
        global _GROUP
        self.model = CpuModel().eval()
        _GROUP = self.model.embed.comm_group
        self.graph = RequestRowProvenance(max_calls=256)
        roles = {'CpuEmbedding._forward_origin': 'embedding_origin',
            'CpuEmbeddingMethod.embedding': 'embedding_lookup', 'pad': 'embedding_pad',
            'tensor_model_parallel_all_reduce': 'embedding_tp_reduce', 'get_tp_group': 'embedding_get_tp',
            'CpuGroup.all_reduce': 'embedding_comm_return', 'chunk': 'no_chunk'}
        bindings = source_bindings({__name__: (hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), roles)})
        self.source = SourceBoundRows(self.graph, self.model, *bindings, simulation=True, operator_aliases={
            torch.ops.cpu_quant_embedding_rows.pad.default: 'vllm.maybe_pad_and_reduce.default',
            torch.ops.cpu_quant_embedding_rows.chunk.default: 'vllm.maybe_chunk_residual.default',
            torch.ops.cpu_quant_embedding_rows.norm.default: '_C_ascend.npu_add_rms_norm_bias.default'})
        self.graph.boundaries = self.source
        self.rows = dict(request_ids=['cpu-' + str(i) for i in range(8)], row_token_positions=[2]*8,
            input_ids=list(range(1, 9)))
        self.addCleanup(lambda: self.graph.end() if self.graph.active else None)

    def begin(self, ids, *, supplied=None, forward='cpu-embedding'):
        self.graph.begin(forward)
        self.source.begin(self.rows, input_ids=ids if supplied is None else supplied)
        for _, owner in self.model.named_modules():
            for name, parameter in owner._parameters.items():
                self.graph.register_broadcast_parameter(owner, name, parameter, kind='_parameters')

    def test_actual_ids_lookup_tp1_return_and_clone_seed_both_residual_inputs(self):
        with torch.inference_mode():
            ids = torch.arange(1, 9, dtype=torch.int32)
            self.begin(ids)
            y, residual, embedding = self.model(ids)
            for tensor in (y, residual, embedding):
                self.assertTrue(self.graph.query(tensor)['verified'], self.graph.report())
                self.assertEqual(self.graph.query(tensor)['actual_rows'], self.rows)
            seed = self.graph.nodes[0]
            self.assertEqual(seed['operator'], 'ACTUAL_SCHEDULER_IDS_TO_TP1_EMBEDDING')
            actual = seed['evidence']['actual_operators']
            self.assertEqual([row['canonical_operator'] for row in actual],
                ['aten.to.dtype', 'aten.embedding.default', 'vllm.maybe_pad_and_reduce.default'])
            self.assertEqual(actual[1]['actual_scalar_args'], dict(padding_idx=-1, scale_grad_by_freq=False, sparse=False))
            returns = seed['evidence']['source_returns']
            comm = next(row for row in returns if row['role'] == 'embedding_comm_return')
            self.assertEqual(comm['group_object_id'], id(self.model.embed.comm_group))
            self.assertEqual(comm['observed_child_operators'], [])
            self.assertTrue(comm['returned_same_object'])
            report = self.graph.end()
        self.assertTrue(report['complete'], report)
        self.assertTrue(all(row['original_calls'] == 1 for row in report['events']))
        self.assertEqual(report['source_boundaries']['owned_scheduler_input_refs'], 0)
        self.assertIsNone(sys.getprofile())

    def refused(self, change):
        with torch.inference_mode():
            ids = torch.arange(1, 9, dtype=torch.int32)
            self.begin(ids)
            actual = change(ids)
            y, _, embedding = self.model(ids if actual is None else actual)
            self.assertFalse(self.graph.query(y)['verified'])
            self.assertFalse(self.graph.query(embedding)['verified'])
            report = self.graph.end()
        self.assertTrue(all(row['original_calls'] == 1 for row in report['events']))
        self.assertEqual(report['source_boundaries']['owned_scheduler_input_refs'], 0)

    def test_equal_ids_new_tensor_is_not_current_scheduler_object(self):
        self.refused(lambda ids: ids.clone())

    def test_permuted_indices_keep_full_observation_but_never_seed_rows(self):
        self.refused(lambda ids: setattr(self.model.embed, 'permute_ids', True))

    def test_replaced_equal_weight_cannot_borrow_model_parameter_owner(self):
        self.refused(lambda ids: setattr(self.model.embed, 'weight', torch.nn.Parameter(
            self.model.embed.weight.clone(), requires_grad=False)))

    def test_tp2_is_not_treated_as_same_object_tp1(self):
        def change(ids):
            self.model.embed.tp_size = self.model.embed.comm_group.world_size = 2
        self.refused(change)

    def test_foreign_same_size_group_is_not_actual_embedding_comm_owner(self):
        def change(ids):
            global _GROUP
            _GROUP = CpuGroup()
        self.refused(change)

    def test_nondefault_actual_padding_arg_refuses_even_equal_output_rows(self):
        self.refused(lambda ids: setattr(self.model.embed.quant_method, 'padding_idx', 0))

    def test_nondefault_actual_ep_flag_refuses_same_object_result(self):
        self.refused(lambda ids: setattr(self.model.embed, 'ep', True))

    def test_omitted_observed_pad_chain_does_not_gain_seed_from_source_name(self):
        self.refused(lambda ids: setattr(self.model.embed, 'skip_pad', True))

    def test_unknown_residual_does_not_inherit_proven_other_operand_rows(self):
        with torch.inference_mode():
            ids = torch.arange(1, 9, dtype=torch.int32)
            self.begin(ids)
            self.model.unknown_residual = True
            y, _, embedding = self.model(ids)
            self.assertTrue(self.graph.query(embedding)['verified'])
            self.assertFalse(self.graph.query(y)['verified'])

    def test_input_write_epoch_invalidates_scheduler_producer(self):
        self.refused(lambda ids: (ids.add_(1), None)[1])

    def test_unreviewed_group_getter_cannot_lend_equal_group_object(self):
        with torch.inference_mode():
            ids = torch.arange(1, 9, dtype=torch.int32)
            self.begin(ids)
            with patch(__name__ + '.get_tp_group', return_value=self.model.embed.comm_group):
                y, _, _ = self.model(ids)
            self.assertFalse(self.graph.query(y)['verified'])
            self.graph.end()
        self.assertIsNone(sys.getprofile())
        self.assertIsNone(self.source.input_ids)

    def test_foreign_profile_preserves_scheduler_ref_until_owner_restored(self):
        with torch.inference_mode():
            ids = torch.arange(1, 9, dtype=torch.int32)
            self.begin(ids)
            self.assertTrue(self.graph.query(self.model(ids)[0])['verified'])
            foreign = lambda *args: None
            sys.setprofile(foreign)
            try:
                with self.assertRaisesRegex(RuntimeError, 'PROFILE_OWNER_CHANGED'):
                    self.graph.end()
                self.assertIs(sys.getprofile(), foreign)
                self.assertIs(self.source.input_ids, ids)
                self.assertTrue(self.graph.active and self.source.active)
            finally:
                sys.setprofile(self.source.callback)
            self.graph.end()
        self.assertIsNone(sys.getprofile())
        self.assertIsNone(self.source.input_ids)

    def test_failed_source_install_has_no_retained_scheduler_tensor(self):
        self.graph.begin('failed-install')
        ids = torch.arange(1, 9, dtype=torch.int32)
        with patch.object(self.source, '_verify_bindings', side_effect=RuntimeError('CPU_SOURCE_CHANGED')):
            with self.assertRaisesRegex(RuntimeError, 'CPU_SOURCE_CHANGED'):
                self.source.begin(self.rows, input_ids=ids)
        self.assertIsNone(self.source.input_ids)
        self.assertFalse(self.source.active)
        self.assertIsNone(sys.getprofile())
        self.graph.end()

    def test_int64_ids_and_two_inference_forwards_have_fresh_complete_producers(self):
        with torch.inference_mode():
            for number in range(2):
                ids = torch.arange(1, 9, dtype=torch.int64)
                self.begin(ids, forward='cpu-embedding-' + str(number))
                y, _, _ = self.model(ids)
                self.assertTrue(self.graph.query(y)['verified'], self.graph.report())
                self.assertTrue(self.graph.end()['complete'])
                self.assertIsNone(self.source.input_ids)


if __name__ == '__main__':
    unittest.main()
