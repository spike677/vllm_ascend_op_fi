"""Custom GEMM/RMS originals and original Python Native lease; CPU ABI only."""
import unittest
import sys
from unittest.mock import patch

import torch

import test_core_output39_group as group
from output39_custom_fixture import CustomOperators, CustomProjection
from test_postwrite_native_recovery import CpuNativeTransport
from fault_injection.deployments.postwrite import MAX_POSTWRITE_SPAN_BYTES
from fault_injection.deployments import postwrite_npu_consumers as consumers


class CoreOutput39GroupCustomTests(unittest.TestCase):
    snapshot = group.CoreOutput39GroupTests.snapshot
    root = group.CoreOutput39GroupTests.root
    forward = group.CoreOutput39GroupTests.forward
    dispose = group.CoreOutput39GroupTests.dispose
    start = group.CoreOutput39GroupTests.start
    enqueue = group.CoreOutput39GroupTests.enqueue
    ok = group.CoreOutput39GroupTests.ok
    request = group.CoreOutput39GroupTests.request
    until = group.CoreOutput39GroupTests.until
    recovery = group.CoreOutput39GroupTests.recovery

    def setUp(self):
        self.ops = self.enterContext(CustomOperators())
        group.CoreOutput39GroupTests.setUp(self)
        self.enterContext(patch.dict(sys.modules, {'torch_npu': self.ops.npu}))
        self.model.projection = CustomProjection('rms')
        self.model.projection_alias = self.model.projection
        self.model.other = CustomProjection('swiglu')
        self.model.other.weight = torch.nn.Parameter(torch.full((8,4), .25, dtype=torch.bfloat16), requires_grad=False)
        self.transport = CpuNativeTransport()
        self.transport.lease_capacity = 8
        self.transport.lease_byte_capacity = MAX_POSTWRITE_SPAN_BYTES

    def test_custom_actual_ordinal_consumer_and_each_native_lease_original_join(self):
        self.start()
        for index in range(40):
            if self.provider.complete:
                break
            self.request(index)
        state = self.ok(self.client.finish())
        for label, hits in (('c2', 2), ('c3', 4)):
            report = self.provider.group_reports[label]
            self.assertTrue(report['coverage']['contract_satisfied'])
            self.assertEqual(report['coverage']['hits'], hits)
            receipts = report['native_lease_receipts']
            self.assertEqual(len(receipts), hits)
            for record in receipts:
                receipt = record['receipt']
                self.assertEqual(receipt['schema'], 'demoncat.postwrite.lease_receipt.v1')
                self.assertEqual([event['operation'] for event in receipt['events']], ['open','apply','restore','release','live'])
                self.assertTrue(receipt['closed'])
                self.assertFalse(receipt['live_readbacks'][-1]['native_live'])
                self.assertEqual(receipt['owner'], self.provider.native_owner)
                execution = next(row for row in report['execution_artifacts']
                    if row['target_id'] == record['target_id'] and row['request_id'] == record['actual_request_id'])
                self.assertEqual(receipt['generation'], execution['native_generation'])
                self.assertEqual(receipt['applications'][0]['changed_elements'], execution['changed_elements'])
                self.assertTrue(execution['consumer_observations'])
            for row in report['independent_discovery'][0]['calls']:
                if row['eligible']:
                    self.assertEqual(row['operator'], 'vllm.unquantized_gemm.default')
                    self.assertEqual(row['key'][-1], 0)
                    self.assertTrue(row['producer_proof']['registered_implementation']['current_registration'])
        self.assertFalse(state['evidence']['c2'] or state['evidence']['c3'])
        self.ok(self.client.unregister())

    def test_current_reviewed_consumer_registration_change_refuses_before_new_native_lease(self):
        index = self.until('c2_lossless')
        before = len(self.transport.opened)
        with patch.object(consumers, 'npu_consumer_registration', return_value=(None, 'CPU_CHANGED_REGISTERED_CONSUMER')):
            with self.assertRaisesRegex(RuntimeError, 'OUTPUT39_GROUP_REVIEWED_CONSUMER_REGISTRATION_CHANGED'):
                self.enqueue(index)
        self.assertEqual(len(self.transport.opened), before)
        self.assertTrue(self.client.snapshot()['ownership_retained'])

    def failed_native_cleanup(self, failure):
        index = self.until('c2_fault')
        opened = self.transport.open
        def fail_open(*args):
            lease = opened(*args)
            self.transport.abis[-1].fail = failure
            return lease
        with patch.object(self.transport, 'open', fail_open):
            with self.assertRaisesRegex(RuntimeError, failure.upper()):
                self.enqueue(index)
        self.assertTrue(self.client.snapshot()['ownership_retained'])
        self.assertTrue(self.provider.p._group_leases)
        abi = self.transport.abis[-1]
        restores, releases = len(abi.restore_calls), len(abi.release_calls)
        abi.fail = None
        # The ABI double implements device restoration with Tensor.copy_.
        with torch.inference_mode():
            state = self.recovery(index+1)
        if failure == 'after_release':
            self.assertEqual((len(abi.restore_calls), len(abi.release_calls)), (restores, releases))
        self.assertEqual(state['leases'], 0)
        self.assertFalse(state['evidence']['c2'] or state['evidence']['c3'])
        records = self.provider.group.native_receipts
        self.assertTrue(any(record['receipt']['first_failure'] is not None for record in records))
        self.assertTrue(any(record['receipt']['closed'] for record in records))

    def test_native_before_restore_keeps_original_lease_until_explicit_off_health(self):
        self.failed_native_cleanup('before_restore')

    def test_native_after_release_error_keeps_owner_and_does_not_restore_absent_handle(self):
        self.failed_native_cleanup('after_release')


if __name__ == '__main__':
    unittest.main()
