"""Real public Host/Runtime requests with explicit CPU custom-op/Native doubles."""
import unittest
import sys
from unittest.mock import patch

import torch

from demoncat.runtime.vllm_ascend.worker_extension import serving39_output
from output39_custom_fixture import CustomOperators, CustomProjection
import test_core_output39 as core


class CoreOutput39CustomTests(unittest.TestCase):
    snapshot = core.CoreOutput39Tests.snapshot
    root = core.CoreOutput39Tests.root
    forward = core.CoreOutput39Tests.forward
    dispose = core.CoreOutput39Tests.dispose
    ok = core.CoreOutput39Tests.ok
    start = core.CoreOutput39Tests.start
    enqueue = core.CoreOutput39Tests.enqueue

    def setUp(self):
        self.ops = self.enterContext(CustomOperators())
        core.CoreOutput39Tests.setUp(self)
        self.enterContext(patch.dict(sys.modules, {'torch_npu': self.ops.npu}))

    def protocol(self, consumer):
        model = CustomProjection(consumer)
        if consumer == 'swiglu':
            model.weight = torch.nn.Parameter(torch.full((8, 4), .25, dtype=torch.bfloat16), requires_grad=False)
        self.model.projection = model
        self.model.projection_alias = model
        self.start()
        for index in range(6):
            before = list(self.ops.calls)
            self.assertEqual(self.client.query()['status'], 'QUERY_OK')
            self.assertEqual(self.ops.calls, before)
            self.enqueue(index)
        state = self.ok(self.client.finish())
        evidence = state['evidence']
        self.assertEqual(evidence['selection']['operator'], 'vllm.unquantized_gemm.default')
        self.assertEqual(self.ops.calls, ['gemm', consumer] * 6)
        self.assertEqual(evidence['actual_requests'][3]['report']['new_fault_hits'], 1)
        self.assertTrue(evidence['actual_requests'][4]['report']['exact_baseline_output'])
        self.assertFalse(evidence['c2'] or evidence['c3'])
        self.assertEqual((state['owned_hooks'], state['leases'], state['inflight']), (0, 0, 0))
        self.assertTrue(all(lease.closed for lease in self.transport.opened))
        self.ok(self.client.unregister())
        self.assertEqual(self.client.detach()['status'], 'DETACHED')
        serving39_output.remove_fault39_output_serving_binding(self.worker)
        with torch.inference_mode():
            self.assertTrue(torch.isfinite(self.forward(self.provider.session)).all())
        return state

    def test_custom_gemm_rms_two_healthy_lossless_fault_off_same_worker_cleanup(self):
        self.protocol('rms')

    def test_custom_gemm_swiglu_two_healthy_lossless_fault_off_same_worker_cleanup(self):
        self.protocol('swiglu')

    def test_original_native_lease_receipts_join_current_target_and_exact_recovery(self):
        from test_postwrite_native_recovery import CpuNativeTransport
        self.transport = CpuNativeTransport()
        final = self.protocol('swiglu')
        requests = final['evidence']['actual_requests']
        for index, mode in ((2,0), (3,1)):
            report = requests[index]['report']
            receipt, target = report['native_lease_receipt'], report['actual_target']
            self.assertEqual(receipt['schema'], 'demoncat.postwrite.lease_receipt.v1')
            self.assertEqual(receipt['geometry'], target['producer_proof']['returned_tensor'])
            self.assertEqual(receipt['open']['data_ptr'], receipt['geometry']['data_ptr'])
            self.assertEqual(receipt['open']['bytes'], receipt['geometry']['logical_bytes'])
            self.assertEqual((receipt['owner'], receipt['generation']), (target['owner'], target['generation']))
            self.assertEqual(receipt['applications'][0]['mode'], mode)
            self.assertEqual(receipt['applications'][0]['indices'], target['logical_indices'])
            self.assertEqual(receipt['applications'][0]['changed_elements'], report['intervention']['changed_elements'])
            self.assertEqual([row['operation'] for row in receipt['events']], ['open','apply','restore','release','live'])
            self.assertTrue(receipt['closed'])
            self.assertIsNone(receipt['first_failure'])
            self.assertFalse(receipt['live_readbacks'][-1]['native_live'])
        self.assertIsNone(requests[4]['report']['native_lease_receipt'])
        self.assertEqual(len(self.transport.opened), 2)


if __name__ == '__main__':
    unittest.main()
