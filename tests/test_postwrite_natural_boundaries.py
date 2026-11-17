"""CPU contracts for real returned storage and view/consumer lineage, not NPU acceptance."""
import unittest

import torch

from fault_injection.deployments.postwrite import PostwriteProvider, PostwriteRefused
from fault_injection.deployments.postwrite_boundaries import projection_return, reads_selected_elements
from tests.test_postwrite_providers import CpuTransport, Projection


class NaturalProjectionBoundaryTests(unittest.TestCase):
    def proof(self, name, args, result, **kwargs):
        return projection_return(getattr(torch.ops.aten, name).default, args, kwargs, result)

    def test_rank_three_linear_bias_and_exact_return_storage(self):
        x = torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4)
        w = torch.ones(5, 4, dtype=torch.bfloat16)
        b = torch.ones(5, dtype=torch.bfloat16)
        y = torch.nn.functional.linear(x, w, b)
        proof, reason = self.proof('linear', (x, w, b), y)
        self.assertIsNone(reason)
        self.assertEqual(proof['expected_output_shape'], [2, 3, 5])
        self.assertEqual(proof['boundary'], 'linear_affine_return')
        self.assertEqual(proof['returned_tensor']['data_ptr'], y.data_ptr())
        self.assertTrue(proof['original_return_storage'])

    def test_matmul_broadcast_and_vector_contracts(self):
        for ashape, bshape in (((2, 1, 3, 4), (1, 5, 4, 6)), ((4,), (2, 4, 6)),
                               ((2, 3, 4), (4,)), ((4,), (4,))):
            with self.subTest(ashape=ashape, bshape=bshape):
                a, b = torch.ones(ashape, dtype=torch.bfloat16), torch.ones(bshape, dtype=torch.bfloat16)
                y = torch.matmul(a, b)
                proof, reason = self.proof('matmul', (a, b), y)
                self.assertIsNone(reason)
                self.assertEqual(proof['expected_output_shape'], list(y.shape))
                self.assertEqual(proof['boundary'], 'matmul_broadcast_return')

    def test_wrong_geometry_bias_dtype_and_aliased_return_refused(self):
        x = torch.ones(2, 4, dtype=torch.bfloat16)
        w = torch.ones(4, 4, dtype=torch.bfloat16)
        for args, result, reason in (
                ((x, w), x, 'PROJECTION_RETURN_ALIASES_OPERAND'),
                ((x, w), torch.ones(4, 2, dtype=torch.bfloat16), 'PROJECTION_RETURN_GEOMETRY_MISMATCH'),
                ((x, w, torch.ones(4)), torch.ones_like(x), 'LINEAR_BIAS_GEOMETRY_MISMATCH'),
                ((x.float(), w), torch.ones_like(x), 'PROJECTION_BF16_OPERANDS_REQUIRED')):
            self.assertEqual(self.proof('linear', args, result)[1], reason)

    def test_noncontiguous_operand_views_read_selected_elements(self):
        producer = torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4).clone()
        self.assertTrue(reads_selected_elements(producer.transpose(0, 2), (0, 1, 4), producer))
        self.assertTrue(reads_selected_elements(producer[:, :, :2], (0, 1, 4), producer))
        self.assertFalse(reads_selected_elements(producer[:, :, 2:], (0,), producer))
        self.assertFalse(reads_selected_elements(producer.clone(), (0,), producer))

    def test_disjoint_same_storage_slice_is_not_a_consumer(self):
        class Slice(Projection):
            def forward(self, x):
                out = torch.mm(x, self.weight)
                return out[1:] + 1
        p = PostwriteProvider(Slice(), CpuTransport(), '39.output_proxy')
        with self.assertRaisesRegex(PostwriteRefused, 'NO_ELIGIBLE_WRITER'):
            p.qualify(torch.ones(2, 4, dtype=torch.bfloat16))

    def test_view_chain_proves_real_value_read_and_restores_return_storage(self):
        class View(Projection):
            def forward(self, x):
                out = torch.mm(x, self.weight)
                return out.view(2, 2, 2).transpose(0, 2) + 1
        p = PostwriteProvider(View(), CpuTransport(), '39.output_proxy')
        x = torch.ones(2, 4, dtype=torch.bfloat16)
        selected = p.qualify(x)
        self.assertEqual(selected['producer_proof']['operator'], 'aten.mm.default')
        reads = selected['consumer_observations']
        self.assertEqual(len(reads), 1)
        self.assertEqual(reads[0]['operator'], 'aten.add.Tensor')
        self.assertEqual(reads[0]['selected_elements_read'], [0])
        baseline, _ = p.run('baseline', x)
        fault, report = p.run('fault', x)
        off, _ = p.run('off', x)
        self.assertEqual(int(torch.isinf(fault).sum()), 1)
        self.assertTrue(report['recovery_confirmed'])
        self.assertTrue(torch.equal(off, baseline))


if __name__ == '__main__':
    unittest.main()
