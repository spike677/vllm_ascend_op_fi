"""Actual CPU dispatcher contracts; no Native or scheduler qualification."""
import gc
import unittest

import torch
import numpy as np
from torch.utils._python_dispatch import TorchDispatchMode

from demoncat.runtime.vllm_ascend.worker_extension.quant_row_provenance import RequestRowProvenance


class RequestRowProvenanceTests(unittest.TestCase):
    def graph(self, x, *, budget=256):
        graph = RequestRowProvenance(max_calls=budget)
        graph.begin('cpu-forward')
        self.addCleanup(lambda: graph.end() if graph.active else None)
        graph.seed(x, dict(verified=True, source='EXPLICIT_CPU_SCHEDULER_FIXTURE_NOT_NATIVE',
            actual_rows=dict(request_ids=['cpu-' + str(i) for i in range(x.shape[0])],
                logical_ids=list(range(x.shape[0])), input_ids=list(range(x.shape[0])))))
        return graph

    def test_real_row_preserving_chain_binds_actual_tensor_and_schema(self):
        x = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        graph = self.graph(x)
        y = (x.clone() + 1).view(8, 2, 2).transpose(1, 2).contiguous().reshape(8, 4)
        a, b = y.split(2, dim=1)
        output = torch.cat([a, b], dim=1)
        proof = graph.query(output)
        self.assertTrue(proof['verified'])
        self.assertEqual(proof['actual_rows']['request_ids'], ['cpu-' + str(i) for i in range(8)])
        report = graph.end()
        self.assertTrue(report['complete'])
        self.assertTrue(any(e['row_preserving'] and e['operator'] == 'aten.cat.default' for e in report['events']))
        self.assertTrue(all(e['original_calls'] == 1 and len(e['schema_sha256']) == 64 for e in report['events']))
        self.assertFalse(report['native_qualified'])

    def test_equal_shape_row_permutation_and_changed_indices_remain_unknown(self):
        x = torch.ones(8, 8)  # Even equal bytes cannot prove request identity.
        graph = self.graph(x)
        index = torch.arange(7, -1, -1)
        self.assertFalse(graph.query(x.index_select(0, index))['verified'])
        index.copy_(torch.arange(8))
        self.assertFalse(graph.query(x.index_select(0, index))['verified'])
        self.assertFalse(graph.query(x.transpose(0, 1))['verified'])
        self.assertFalse(graph.query(torch.roll(x, 1, dims=0))['verified'])
        self.assertTrue(graph.query(x)['verified'])

    def test_unknown_inplace_invalidates_all_live_storage_aliases(self):
        x = torch.ones(8, 4)
        graph = self.graph(x)
        alias = x.view(8, 2, 2)
        self.assertTrue(graph.query(alias)['verified'])
        alias.add_(1)
        self.assertFalse(graph.query(x)['verified'])
        self.assertFalse(graph.query(alias)['verified'])
        self.assertTrue(any(e.get('mutable') and not e['row_preserving'] for e in graph.events))

    def test_inference_tensor_absent_counter_is_explicit_and_write_epoch_still_invalidates(self):
        with torch.inference_mode():
            x = torch.ones(8, 4)
            graph = self.graph(x)
            proof = graph.query(x)
            self.assertTrue(proof['tensor']['inference_version_unavailable'])
            self.assertIsNone(proof['tensor']['version'])
            x.copy_(torch.ones_like(x))
            self.assertFalse(graph.query(x)['verified'])
            graph.end()

    def test_distinct_storage_wrappers_aliasing_same_bytes_cannot_escape_write_epoch(self):
        backing = np.ones((8, 4), dtype=np.float32)
        with torch.inference_mode():
            x, alias = torch.from_numpy(backing), torch.from_numpy(backing)
            self.assertEqual(x.data_ptr(), alias.data_ptr())
            self.assertNotEqual(x.untyped_storage()._cdata, alias.untyped_storage()._cdata)
            graph = self.graph(x)
            alias.add_(1)
            self.assertFalse(graph.query(x)['verified'])
            self.assertEqual(len(graph.events[-1]['invalidated_storages']), 2)
            graph.end()

    def test_unobserved_actual_version_change_rejects_stale_proof(self):
        x = torch.ones(8, 4)
        graph = self.graph(x)
        with torch._C._DisableTorchDispatch():  # Test an external mutation gap.
            x.add_(1)
        self.assertFalse(graph.query(x)['verified'])

    def test_tensor_identity_reuse_cannot_borrow_dead_reference(self):
        x = torch.ones(8, 4)
        graph = self.graph(x)
        record = graph.proofs[id(x)]
        with torch._C._DisableTorchDispatch():
            other = torch.ones_like(x)
        graph.proofs[id(other)] = record  # Explicit simulated allocator/id collision.
        self.assertFalse(graph.query(other)['verified'])
        del x
        gc.collect()
        self.assertIsNone(record['tensor']())
        self.assertFalse(graph.query(other)['verified'])

    def test_unknown_matrix_and_row_mixing_do_not_inherit_from_shape(self):
        x = torch.ones(8, 4)
        graph = self.graph(x)
        self.assertFalse(graph.query(x @ torch.ones(4, 4))['verified'])
        unknown = torch.ones_like(x)
        self.assertFalse(graph.query(x + unknown)['verified'])
        self.assertFalse(graph.query(x + torch.ones(4))['verified'])
        self.assertFalse(graph.query(x + torch.ones(1, 4))['verified'])
        self.assertTrue(graph.query(x + 1)['verified'])

    def test_current_broadcast_parameter_is_distinct_from_unverified_request_data(self):
        module = torch.nn.Module()
        module.register_parameter('bias', torch.nn.Parameter(torch.ones(4), requires_grad=False))
        module.register_parameter('per_request', torch.nn.Parameter(torch.ones(8, 4), requires_grad=False))
        module.register_buffer('unreviewed_buffer', torch.ones(4))
        x = torch.ones(8, 4)
        graph = self.graph(x)
        graph.register_broadcast_parameter(module, 'bias', module.bias, kind='_parameters')
        graph.register_broadcast_parameter(module, 'per_request', module.per_request, kind='_parameters')
        self.assertTrue(graph.query(x + module.bias)['verified'])
        self.assertFalse(graph.query(x + module.per_request)['verified'])
        self.assertFalse(graph.query(x + module.unreviewed_buffer)['verified'])
        module.bias.add_(1)
        self.assertFalse(graph.query(x + module.bias)['verified'])

    def test_cross_request_residual_with_other_verified_order_still_refuses_row_inheritance(self):
        x, y = torch.ones(8, 4), torch.ones(8, 4)
        graph = self.graph(x)
        graph.seed(y, dict(verified=True, source='EXPLICIT_OTHER_CPU_ORDER', actual_rows=dict(
            request_ids=['cpu-' + str(i) for i in range(7, -1, -1)], logical_ids=list(range(7, -1, -1)),
            input_ids=list(range(8)))))
        self.assertFalse(graph.query(x + y)['verified'])
        self.assertTrue(graph.query(x + x.clone())['verified'])

    def test_owned_public_native_projection_bridge_requires_actual_input_proof(self):
        x = torch.ones(8, 4)
        graph = self.graph(x)
        output = x @ torch.ones(4, 6)
        evidence = dict(complete_native_dynamic_formals=True, owned_public_qmm_chain=True)
        self.assertFalse(graph.query(output)['verified'])
        self.assertTrue(graph.projection(x, output, evidence=evidence)['verified'])
        other = torch.ones_like(x)
        self.assertFalse(graph.projection(other, other.clone(), evidence=evidence)['verified'])
        with self.assertRaisesRegex(RuntimeError, 'EXACT_NATIVE_DYNAMIC_QUANT'):
            graph.projection(x, output, evidence=dict(evidence, owned_public_qmm_chain=False))

    def test_capacity_keeps_original_execution_and_refuses_scope(self):
        x = torch.ones(8, 4)
        graph = self.graph(x, budget=1)
        first = x + 1
        second = first + 1
        self.assertFalse(graph.query(second)['verified'])
        report = graph.end()
        self.assertEqual((report['actual_calls'], report['retained_calls'], report['missing_calls']), (2, 1, 1))
        self.assertFalse(report['complete'])
        self.assertTrue(torch.equal(second, torch.full_like(x, 3)))

    def test_foreign_mode_cannot_be_popped_as_own_cleanup(self):
        class Other(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))
        graph = self.graph(torch.ones(8, 4))
        with Other():
            with self.assertRaisesRegex(RuntimeError, 'ROW_DISPATCH_MODE_OWNER_CHANGED'):
                graph.end()
            self.assertTrue(graph.active)
        graph.end()


if __name__ == '__main__':
    unittest.main()
