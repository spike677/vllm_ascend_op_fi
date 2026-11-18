"""CPU dispatcher contracts for source-bound noncontiguous MLA singleton views.

The input geometries follow the reviewed split/transpose/view source. They
are not a natural Worker census or an Attention numerical qualification.
"""
import copy
import hashlib
from pathlib import Path
import unittest

import torch

from demoncat.runtime.vllm_ascend.worker_extension.quant_row_provenance import RequestRowProvenance, metadata
from demoncat.runtime.vllm_ascend.worker_extension.quant_source_rows import (
    SourceBoundRows, _singleton_view_axis, source_bindings,
)


class CpuSingletonViews(torch.nn.Module):
    def view(self, x, shape):
        return x.view(shape)

    def view_contiguous(self, x, shape, memory_format=torch.contiguous_format):
        return x.view(shape).contiguous(memory_format=memory_format)


class MlaSingletonViewTests(unittest.TestCase):
    def setUp(self):
        self.model = CpuSingletonViews()
        self.rows = dict(request_ids=['cpu-' + str(i) for i in range(8)], row_token_positions=list(range(8)))
        self.graph = RequestRowProvenance(max_calls=256)
        bindings = source_bindings({__name__: (hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            {'CpuSingletonViews.view': 'mla_rope', 'CpuSingletonViews.view_contiguous': 'mla_decode'})})
        self.source = SourceBoundRows(self.graph, self.model, *bindings, simulation=True)
        self.graph.boundaries = self.source
        self.addCleanup(lambda: self.graph.end() if self.graph.active else None)

    def begin(self, tensor, forward='cpu-singleton'):
        self.graph.begin(forward)
        self.source.begin(self.rows)
        self.graph.seed(tensor, dict(verified=True, actual_rows=self.rows, source='EXPLICIT_CPU_FIXTURE'))

    def test_actual_split_view_keeps_noncontiguous_rows_and_offset(self):
        with torch.inference_mode():
            combined = torch.arange(8 * 2 * 10).reshape(8, 2, 10)
            self.begin(combined)
            _, rope = combined.split([6, 4], dim=-1)
            output = self.model.view(rope, (8, 2, 1, 4))
            proof = self.graph.query(output)
            self.assertTrue(proof['verified'], self.graph.report())
            self.assertEqual(proof['actual_rows'], self.rows)
            self.assertFalse(rope.is_contiguous())
            self.assertEqual(output.storage_offset(), 6)
            node = self.graph.nodes[proof['node_id'] - 1]
            self.assertEqual(node['evidence']['details']['view_rule'], 'ACTUAL_ATEN_VIEW_SINGLETON_COORDINATES_V1')
            self.assertEqual(node['parents'], [self.graph._lookup(rope)['node']['node_id']])
            self.assertEqual(node['evidence']['details']['input_metadata'], metadata(rope))
            self.assertEqual(node['metadata'], metadata(output))
            report = self.graph.end()
        self.assertTrue(report['complete'], report)
        self.assertTrue(all(row['original_calls'] == 1 for row in report['events']))

    def test_actual_transpose_view_preserves_axis_without_collapsing_requests(self):
        with torch.inference_mode():
            x = torch.ones(8, 2, 4)
            self.begin(x)
            # The transpose is itself observed under the reviewed MLA role in
            # the separate existing bmm tests. Seed its known axis here only
            # to isolate this view's exact coordinate proof.
            transposed = x.transpose(0, 1)
            self.graph._bind(transposed, rows=self.rows, parents=(1,), operator='EXPLICIT_CPU_AXIS_ONE',
                evidence=dict(simulation=True), row_axis=1)
            output = self.model.view(transposed, (2, 8, 1, 4))
            record = self.graph._lookup(output)
            self.assertIsNotNone(record)
            self.assertEqual(record['node']['row_axis'], 1)
            # The public projection gate still requires request rows at axis0.
            self.assertFalse(self.graph.query(output)['verified'])

    def test_source64_observed_rope_stride_and_offset(self):
        # source64 first healthy event187: a feature split, not a row slice.
        with torch.inference_mode():
            x = torch.empty(8, 16, 192, dtype=torch.bfloat16)[..., 128:]
            self.assertEqual((list(x.shape), list(x.stride()), x.storage_offset()),
                             ([8, 16, 64], [3072, 192, 1], 128))
            self.begin(x)
            output = self.model.view(x, (8, 16, 1, 64))
            self.assertEqual(list(output.stride()), [3072, 192, 64, 1])
            self.assertEqual(output.storage_offset(), 128)
            self.assertTrue(self.graph.query(output)['verified'])

    def test_source64_observed_query_stride_then_actual_contiguous_copy(self):
        # source64 first healthy event197, after actual head-axis bmm+transpose.
        with torch.inference_mode():
            x = torch.empty(16, 8, 512, dtype=torch.bfloat16).transpose(0, 1)
            self.assertEqual((list(x.shape), list(x.stride())), ([8, 16, 512], [512, 4096, 1]))
            self.begin(x)
            view = self.model.view(x, (8, 16, 1, 512))
            self.assertEqual(list(view.stride()), [512, 4096, 512, 1])
            output = self.model.view_contiguous(x, (8, 16, 1, 512))
            self.assertTrue(self.graph.query(view)['verified'])
            self.assertTrue(self.graph.query(output)['verified'])
            self.assertNotEqual(output.untyped_storage().data_ptr(), x.untyped_storage().data_ptr())

    def test_nondefault_contiguous_memory_format_stays_unknown(self):
        with torch.inference_mode():
            x = torch.empty(16, 8, 512, dtype=torch.bfloat16).transpose(0, 1)
            self.begin(x)
            output = self.model.view_contiguous(x, (8, 16, 1, 512), torch.channels_last)
            self.assertFalse(self.graph.query(output)['verified'])

    def test_contiguous_outside_reviewed_source_frame_stays_unknown(self):
        with torch.inference_mode():
            x = torch.empty(16, 8, 512, dtype=torch.bfloat16).transpose(0, 1)
            self.begin(x)
            view = self.model.view(x, (8, 16, 1, 512))
            self.assertTrue(self.graph.query(view)['verified'])
            self.assertFalse(self.graph.query(view.contiguous())['verified'])

    def test_same_noncontiguous_view_outside_reviewed_frame_stays_unknown(self):
        x = torch.ones(8, 2, 10)[..., 6:]
        self.begin(x)
        output = x.view(8, 2, 1, 4)
        self.assertFalse(self.graph.query(output)['verified'])

    def test_unproven_input_does_not_borrow_same_shape_or_storage(self):
        x = torch.ones(8, 2, 10)[..., 6:]
        self.begin(x)
        unknown = torch.roll(x, 1, 0)
        self.assertFalse(self.graph.query(self.model.view(unknown, (8, 2, 1, 4)))['verified'])

    def test_inference_alias_write_invalidates_both_view_proofs(self):
        with torch.inference_mode():
            base = torch.ones(8, 2, 10)
            x = base[..., 6:]
            self.begin(x)
            output = self.model.view(x, (8, 2, 1, 4))
            self.assertTrue(self.graph.query(output)['verified'])
            base.zero_()
            self.assertFalse(self.graph.query(x)['verified'])
            self.assertFalse(self.graph.query(output)['verified'])
            self.assertFalse(self.graph.query(self.model.view(x, (8, 2, 1, 4)))['verified'])

    def test_two_inference_forwards_do_not_reuse_old_row_proof(self):
        with torch.inference_mode():
            for iteration in range(2):
                x = torch.ones(8, 2, 10)[..., 6:]
                self.begin(x, 'cpu-' + str(iteration))
                output = self.model.view(x, (8, 2, 1, 4))
                self.assertEqual(self.graph.query(output)['forward_id'], 'cpu-' + str(iteration))
                self.assertTrue(self.graph.query(output)['verified'])
                self.assertTrue(self.graph.end()['complete'])
                self.assertFalse(self.graph.query(output)['verified'])

    def test_exact_geometry_predicate_rejects_offset_stride_storage_and_row_changes(self):
        x = torch.ones(8, 2, 10)[..., 6:]
        before, after = metadata(x), metadata(x.view(8, 2, 1, 4))
        self.assertEqual(_singleton_view_axis(before, after, 0), 0)
        mutations = (
            ('offset', 0), ('storage_identity', after['storage_identity'] + 1),
            ('storage_address', after['storage_address'] + 4), ('storage_bytes', after['storage_bytes'] + 4),
            ('dtype', 'torch.int8'), ('device', 'meta'), ('stride', [10, 20, 4, 1]),
            ('shape', [2, 8, 1, 4]), ('shape', [4, 4, 1, 4]),
            ('shape', [16, 1, 1, 4]), ('shape', [8, 1, 1, 8]),
        )
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                changed = copy.deepcopy(after)
                changed[key] = value
                self.assertIsNone(_singleton_view_axis(before, changed, 0))

    def test_decode_attention_bridge_accepts_real_source_noncontiguous_inputs(self):
        from test_quant_mla_decode_rows import MlaDecodeRowsTests
        case = MlaDecodeRowsTests('test_real_schema_and_object_joins_restore_query_axis')
        case.setUp()
        try:
            case.query = torch.ones(2, 8, 4).transpose(0, 1)
            case.rope = torch.ones(8, 2, 6)[..., 4:]
            for tensor in (case.query, case.rope):
                case.graph.seed(tensor, dict(verified=True, actual_rows=case.rows,
                    source='EXPLICIT_CPU_NONCONTIGUOUS_SOURCE_GEOMETRY'))
            output = case.run_decode()
            self.assertTrue(case.graph.query(output)['verified'], case.graph.report())
            report = case.graph.end()
            self.assertTrue(report['complete'], report)
            native = [row for row in report['events'] if row['operator'].startswith('cpu_quant_mla_rows.')]
            self.assertEqual(len(native), 2)
            self.assertTrue(all(row['original_calls'] == 1 and row['row_preserving'] for row in native))
        finally:
            case.doCleanups()


if __name__ == '__main__':
    unittest.main()
