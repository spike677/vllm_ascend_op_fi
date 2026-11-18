"""Explicit CPU Native-op substitutes test joins, never FIA numerical accuracy."""
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

from demoncat.runtime.vllm_ascend.worker_extension.quant_row_provenance import RequestRowProvenance, metadata
from demoncat.runtime.vllm_ascend.worker_extension.quant_source_rows import SourceBoundRows, source_bindings


_LIB = torch.library.Library('cpu_quant_mla_rows', 'DEF')
_LIB.define('attention(Tensor query, Tensor key, Tensor value, *, Tensor query_rope, Tensor key_rope, '
    'int num_query_heads, int num_key_value_heads, str input_layout, Tensor? atten_mask=None, '
    'int sparse_mode=0, float softmax_scale=1.0, Tensor? block_table=None, int block_size=0, '
    'int[]? actual_seq_qlen=None, int[]? actual_seq_kvlen=None) -> (Tensor, Tensor)')
_LIB.define('up(Tensor input, Tensor weight, *, Tensor? bias=None, Tensor? scale=None, '
    'int[]? perm_x1=None, int[]? perm_x2=None, int[]? perm_y=None, int? batch_split_factor=1) -> Tensor')


def cpu_attention(query, key, value, **kwargs):
    return query.transpose(0, 1).contiguous(), torch.zeros(query.shape[0])


def cpu_up(input, weight, **kwargs):
    return torch.bmm(input, weight).transpose(0, 1).contiguous()


_LIB.impl('attention', cpu_attention, 'CPU')
_LIB.impl('up', cpu_up, 'CPU')


class CpuMla(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.kv_lora_rank, self.v_head_dim = 2, 1, 4, 3
        self.scale, self.fa_quant_layer, self.enable_kv_nz, self.speculative_config = 0.5, False, False, None
        self.W_UV = torch.ones(2, 4, 3)
        self.layout = 'BNSD_NBSD'

    def decode(self, q_nope, q_pe, k_nope, k_pe, block_size, attn_metadata, _extra_context):
        query = q_nope.view(8, self.num_heads, 1, -1).contiguous()
        query_rope = q_pe.view(8, self.num_heads, 1, -1)
        output, _ = torch.ops.cpu_quant_mla_rows.attention(query, k_nope, k_nope,
            query_rope=query_rope, key_rope=k_pe, num_query_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads, input_layout=self.layout, atten_mask=None, sparse_mode=0,
            softmax_scale=self.scale, block_table=attn_metadata.decode.block_table, block_size=block_size,
            actual_seq_qlen=None, actual_seq_kvlen=attn_metadata.decode.seq_lens_list)
        return self.up(output)

    def up(self, x):
        x = x.view(self.num_heads, -1, self.kv_lora_rank)
        x = torch.ops.cpu_quant_mla_rows.up(x, self.W_UV, perm_y=[1, 0, 2])
        return x.reshape(-1, self.num_heads * self.v_head_dim)


class MlaDecodeRowsTests(unittest.TestCase):
    def setUp(self):
        self.model = CpuMla()
        self.query, self.rope = torch.ones(8, 2, 4), torch.ones(8, 2, 2)
        self.key, self.key_rope = torch.zeros(1, 1, 128, 4), torch.zeros(1, 1, 128, 2)
        self.metadata = NS(num_actual_tokens=8, num_decode_tokens=8, num_decodes=8, num_prefills=0, prefill=None,
            decode=NS(block_table=torch.arange(8).reshape(8, 1), seq_lens_list=list(range(1, 9))))
        self.context = NS(capturing=False, is_draft_model=False)
        self.rows = dict(request_ids=['cpu-' + str(i) for i in range(8)], row_token_positions=list(range(8)))
        self.graph = RequestRowProvenance()
        pins = {__name__: (hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            {'CpuMla.decode': 'mla_decode', 'CpuMla.up': 'mla_v'})}
        self.source = SourceBoundRows(self.graph, self.model, *source_bindings(pins), simulation=True,
            operator_aliases={torch.ops.cpu_quant_mla_rows.attention.default: 'npu.npu_fused_infer_attention_score_v2.default',
                              torch.ops.cpu_quant_mla_rows.up.default: 'npu.npu_transpose_batchmatmul.default'})
        self.source.constants[(id(self.model), 'W_UV')] = (self.model, self.model.W_UV, metadata(self.model.W_UV))
        actual_decode = self.metadata.decode
        def cpu_producer(decode, graph):
            if decode is not actual_decode:
                raise RuntimeError('CPU producer fixture owner changed')
            return dict(actual_rows=self.rows, evidence_kind='EXPLICIT_CPU_PRODUCER_BOUNDARY_FIXTURE')
        # Actual index/copy/metadata producer contracts are exercised separately
        # in test_quant_mla_metadata_rows; this tests the consumer interface.
        self.source.metadata_producers = NS(consumer=cpu_producer, watched_storages=lambda: [], release=lambda: None)
        self.graph.boundaries = self.source
        self.graph.begin('cpu-decode')
        self.source.begin(self.rows)
        for tensor in (self.query, self.rope):
            self.graph.seed(tensor, dict(verified=True, actual_rows=self.rows, source='EXPLICIT_CPU_FIXTURE'))
        self.addCleanup(lambda: self.graph.end() if self.graph.active else None)

    def run_decode(self):
        return self.model.decode(self.query, self.rope, self.key, self.key_rope, 128, self.metadata, self.context)

    def test_real_schema_and_object_joins_restore_query_axis(self):
        output = self.run_decode()
        self.assertTrue(self.graph.query(output)['verified'], self.graph.report())
        self.assertEqual(self.graph.query(output)['actual_rows'], self.rows)
        report = self.graph.end()
        self.assertTrue(report['complete'])
        calls = [event for event in report['events'] if event['operator'].startswith('cpu_quant_mla_rows.')]
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(event['original_calls'] == 1 for event in calls))

    def test_unknown_layout_refuses_even_equal_output_shape(self):
        self.model.layout = 'UNREVIEWED_LAYOUT'
        self.assertFalse(self.graph.query(self.run_decode())['verified'])

    def test_query_rope_from_different_actual_request_order_is_refused(self):
        changed = dict(self.rows, request_ids=list(reversed(self.rows['request_ids'])))
        self.graph.seed(self.rope, dict(verified=True, actual_rows=changed))
        self.assertFalse(self.graph.query(self.run_decode())['verified'])

    def test_mixed_prefill_cannot_borrow_decode_rows(self):
        self.metadata.num_prefills = 1
        self.assertFalse(self.graph.query(self.run_decode())['verified'])

    def test_sequence_lengths_changed_after_scheduler_proof_are_refused(self):
        self.metadata.decode.seq_lens_list[-1] += 1
        self.assertFalse(self.graph.query(self.run_decode())['verified'])

    def test_graph_capture_and_fa_quant_remain_unverified(self):
        self.context.capturing = True
        self.assertFalse(self.graph.query(self.run_decode())['verified'])
        self.context.capturing = False
        self.model.fa_quant_layer = True
        self.assertFalse(self.graph.query(self.run_decode())['verified'])

    def test_replaced_derived_weight_does_not_inherit_source_ownership(self):
        self.model.W_UV = self.model.W_UV.clone()
        self.assertFalse(self.graph.query(self.run_decode())['verified'])

    def test_missing_early_producer_cannot_borrow_query_shape_or_values(self):
        self.source.metadata_producers = None
        self.assertFalse(self.graph.query(self.run_decode())['verified'])

    def test_fabricated_metadata_return_is_not_the_observed_producer_object(self):
        self.metadata.decode = NS(**vars(self.metadata.decode))
        self.assertFalse(self.graph.query(self.run_decode())['verified'])


if __name__ == '__main__':
    unittest.main()
