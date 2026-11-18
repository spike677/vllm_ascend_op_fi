"""Original Python Native lease with an explicit CPU ABI double; no SDK proof."""
from types import SimpleNamespace as NS
import unittest

import torch

from fault_injection.deployments.postwrite import _NativeLease, PostwriteRefused
from test_postwrite_providers import CpuTransport


class CpuABI:
    def __init__(self, tensor):
        self.tensor, self.original = tensor, tensor.clone()
        self.owners, self.next_handle = {}, 0
        self.restore_calls, self.release_calls, self.readbacks = [], [], []
        self.fail = None
        self.fail_readback = False
        self.apply_calls = []

    def dcat_pw_open(self, address, size, device, owner, generation, result):
        self.next_handle += 1
        result._obj.value = self.next_handle
        self.owners[self.next_handle] = (owner, generation)
        return 0

    def owned(self, handle, owner, generation):
        return self.owners.get(handle.value) == (owner, generation)

    def dcat_pw_apply(self, handle, owner, generation, mode, selected, count, changed):
        self.apply_calls.append((handle.value, owner, generation, mode, list(selected)))
        if not self.owned(handle, owner, generation):
            return 2
        if self.fail == 'apply_code':
            return 7
        if mode:
            self.tensor.view(-1)[list(selected)] = {1:float('inf'), 2:-float('inf'), 3:float('nan')}[mode]
        changed._obj.value = count if mode else 0
        return 0

    def dcat_pw_restore(self, handle, owner, generation):
        self.restore_calls.append((handle.value, owner, generation))
        if not self.owned(handle, owner, generation):
            return 2
        if self.fail == 'before_restore':
            raise RuntimeError('CPU_BEFORE_RESTORE')
        self.tensor.copy_(self.original)
        if self.fail == 'after_restore':
            raise RuntimeError('CPU_AFTER_RESTORE')
        return 0

    def dcat_pw_release(self, handle, owner, generation):
        self.release_calls.append((handle.value, owner, generation))
        if not self.owned(handle, owner, generation):
            return 2
        if self.fail == 'before_release':
            raise RuntimeError('CPU_BEFORE_RELEASE')
        del self.owners[handle.value]
        if self.fail == 'after_release':
            raise RuntimeError('CPU_AFTER_RELEASE')
        if self.fail == 'failed_code_after_release':
            return 9
        return 0

    def dcat_pw_live(self, handle, owner, generation):
        self.readbacks.append((handle.value, owner, generation))
        if self.fail_readback:
            raise RuntimeError('CPU_NATIVE_READBACK_FAILED')
        if handle.value not in self.owners:
            return 0
        return 1 if self.owned(handle, owner, generation) else -1


class CpuNativeTransport(CpuTransport):
    """The original Python Native lease, backed only by CPU ABI calls."""
    sha256 = 'a'*64

    def __init__(self):
        self.opened, self.abis = [], []
        self.restore_failure = False

    def open(self, tensor, owner, generation):
        self.validate(tensor)
        abi = CpuABI(tensor)
        lease = _NativeLease(NS(lib=abi), tensor, owner, generation)
        self.abis.append(abi)
        self.opened.append(lease)
        return lease


class NativeRecoveryTests(unittest.TestCase):
    def make(self):
        tensor = torch.arange(8, dtype=torch.bfloat16)
        abi = CpuABI(tensor)
        lease = _NativeLease(NS(lib=abi), tensor, 101, 7)
        tensor.neg_()
        return tensor, abi, lease

    def boundary(self, failure, restored):
        tensor, abi, lease = self.make()
        abi.fail = failure
        with self.assertRaisesRegex(RuntimeError, failure.upper()):
            lease.recover()
        self.assertFalse(lease.closed)
        self.assertEqual(torch.equal(tensor, abi.original), restored)
        abi.fail = None
        lease.recover()
        self.assertTrue(lease.closed)
        self.assertEqual(len(abi.restore_calls), 2)
        self.assertTrue(torch.equal(tensor, abi.original))

    def test_before_restore_failure_retains_owner_until_retry(self):
        self.boundary('before_restore', False)

    def test_after_restore_failure_repeats_only_the_same_owned_span(self):
        self.boundary('after_restore', True)

    def test_before_release_failure_confirms_same_lease_still_live_and_retries(self):
        self.boundary('before_release', True)

    def test_after_release_exception_preserves_error_and_retry_never_restores_absent_handle(self):
        tensor, abi, lease = self.make()
        abi.owners[99] = (202, 8)  # Another live lease is not evidence about this one.
        abi.fail = 'after_release'
        with self.assertRaisesRegex(RuntimeError, 'CPU_AFTER_RELEASE'):
            lease.recover()
        self.assertTrue(lease.closed)
        self.assertEqual(abi.readbacks, [(lease.handle.value, 101, 7)])
        self.assertEqual(abi.owners, {99: (202, 8)})
        abi.fail = None
        lease.recover()
        self.assertEqual(len(abi.restore_calls), 1)
        self.assertEqual(len(abi.release_calls), 1)
        self.assertTrue(torch.equal(tensor, abi.original))

    def test_failed_release_code_also_keeps_error_with_exact_absence_readback(self):
        _, abi, lease = self.make()
        abi.fail = 'failed_code_after_release'
        with self.assertRaisesRegex(PostwriteRefused, 'POSTWRITE_RELEASE_9'):
            lease.recover()
        self.assertTrue(lease.closed)
        lease.recover()
        self.assertEqual(len(abi.restore_calls), 1)

    def test_release_readback_failure_retains_owner_and_recovers_from_a_later_exact_read(self):
        _, abi, lease = self.make()
        abi.fail, abi.fail_readback = 'after_release', True
        with self.assertRaisesRegex(RuntimeError, 'CPU_AFTER_RELEASE'):
            lease.recover()
        self.assertFalse(lease.closed)
        self.assertTrue(lease.release_uncertain)
        self.assertIn('CPU_NATIVE_READBACK_FAILED', lease.release_readback_error)
        with self.assertRaisesRegex(RuntimeError, 'CPU_NATIVE_READBACK_FAILED'):
            lease.recover()
        self.assertEqual(len(abi.restore_calls), 1)
        abi.fail_readback = False
        lease.recover()
        self.assertTrue(lease.closed)
        self.assertEqual(len(abi.restore_calls), 1)

    def test_changed_owner_during_uncertain_release_cannot_confirm_absence(self):
        _, abi, lease = self.make()
        abi.fail, abi.fail_readback = 'after_release', True
        with self.assertRaises(RuntimeError):
            lease.recover()
        abi.fail_readback = False
        lease.owner += 1
        before = len(abi.readbacks)
        with self.assertRaisesRegex(PostwriteRefused, 'POSTWRITE_RELEASE_SCOPE_CHANGED'):
            lease.recover()
        self.assertFalse(lease.closed)
        self.assertEqual(len(abi.readbacks), before)
        lease.owner = 101
        lease.recover()
        self.assertTrue(lease.closed)

    def test_foreign_native_live_result_never_confirms_release(self):
        _, abi, lease = self.make()
        abi.fail = 'before_release'
        original = abi.dcat_pw_release
        def foreign(handle, owner, generation):
            abi.owners[handle.value] = (999, generation)
            raise RuntimeError('CPU_ORIGINAL_RELEASE_FAILURE')
        abi.dcat_pw_release = foreign
        with self.assertRaisesRegex(RuntimeError, 'CPU_ORIGINAL_RELEASE_FAILURE'):
            lease.recover()
        self.assertFalse(lease.closed)
        self.assertTrue(lease.release_uncertain)
        self.assertIn('POSTWRITE_NATIVE_LEASE_OWNER_MISMATCH', lease.release_readback_error)
        abi.owners[lease.handle.value] = (101, 7)
        abi.dcat_pw_release, abi.fail = original, None
        lease.recover()
        self.assertTrue(lease.closed)

    def test_changed_live_storage_refuses_any_restore_call(self):
        tensor, abi, lease = self.make()
        tensor.set_(torch.empty_like(tensor))
        with self.assertRaisesRegex(PostwriteRefused, 'POSTWRITE_LIVE_STORAGE_CHANGED'):
            lease.recover()
        self.assertEqual(abi.restore_calls, [])
        self.assertFalse(lease.closed)

    def test_receipt_records_real_span_and_call_results_without_new_queries(self):
        tensor = torch.ones(8, dtype=torch.bfloat16)
        abi = CpuABI(tensor)
        lease = _NativeLease(NS(lib=abi), tensor, 101, 7, element_offset=2, elements=4)
        # Query and receipt snapshots never extend the recorded ABI history.
        self.assertTrue(lease.live())
        before = len(abi.readbacks)
        snapshot = lease.receipt()
        self.assertEqual(len(abi.readbacks), before)
        self.assertEqual([row['operation'] for row in snapshot['events']], ['open'])
        self.assertEqual(snapshot['open']['data_ptr'], tensor.data_ptr()+4)
        self.assertEqual(snapshot['open']['bytes'], 8)
        self.assertEqual((snapshot['element_offset'], snapshot['elements']), (2,4))
        snapshot['geometry']['shape'].clear()
        snapshot['events'][0]['owner'] = 999
        self.assertEqual(lease.receipt()['geometry']['shape'], [8])
        self.assertEqual(lease.receipt()['events'][0]['owner'], 101)
        lease.apply(0, (0,))
        lease.recover()
        self.assertFalse(lease.live(record=True))  # Existing provider recovery readback.
        receipt = lease.receipt()
        self.assertEqual([row['operation'] for row in receipt['events']], ['open','apply','restore','release','live'])
        self.assertEqual([row['sequence'] for row in receipt['events']], [1,2,3,4,5])
        self.assertTrue(all(row['returned'] and row['return_code'] == 0 for row in receipt['events']))
        self.assertTrue(all((row['handle'],row['owner'],row['generation']) == (lease.handle.value,101,7)
                            for row in receipt['events']))
        self.assertEqual(receipt['applications'][0]['changed_elements'], 0)
        self.assertIsNone(receipt['first_failure'])
        self.assertTrue(receipt['closed'])
        self.assertFalse(receipt['live_readbacks'][-1]['native_live'])
        self.assertEqual((len(abi.restore_calls), len(abi.release_calls), len(abi.readbacks)), (1,1,2))

    def test_failed_apply_and_release_receipts_preserve_first_return_code_through_retry(self):
        tensor = torch.ones(8, dtype=torch.bfloat16)
        abi = CpuABI(tensor)
        lease = _NativeLease(NS(lib=abi), tensor, 101, 7)
        abi.fail = 'apply_code'
        with self.assertRaisesRegex(PostwriteRefused, 'APPLY_7'):
            lease.apply(1, (0,))
        first = lease.receipt()['first_failure']
        self.assertEqual((first['operation'],first['return_code']), ('apply',7))
        abi.fail = 'before_release'
        with self.assertRaisesRegex(RuntimeError, 'CPU_BEFORE_RELEASE'):
            lease.recover()
        pending = lease.receipt()
        self.assertFalse(pending['closed'])
        self.assertTrue(pending['live_readbacks'][-1]['native_live'])
        self.assertFalse(pending['release']['returned'])
        self.assertIsNone(pending['release']['return_code'])
        self.assertEqual(pending['first_failure'], first)
        abi.fail = None
        lease.recover()
        self.assertFalse(lease.live(record=True))
        final = lease.receipt()
        self.assertTrue(final['closed'])
        self.assertEqual(final['first_failure'], first)
        self.assertEqual(len([row for row in final['events'] if row['operation'] == 'release']), 2)

    def test_absent_handle_after_lost_release_is_not_rewritten_as_successful_release(self):
        _, abi, lease = self.make()
        abi.fail = 'after_release'
        with self.assertRaisesRegex(RuntimeError, 'CPU_AFTER_RELEASE'):
            lease.recover()
        receipt = lease.receipt()
        self.assertTrue(receipt['closed'])
        self.assertFalse(receipt['release']['returned'])
        self.assertIsNone(receipt['release']['return_code'])
        self.assertEqual(receipt['first_failure']['operation'], 'release')
        self.assertFalse(receipt['live_readbacks'][-1]['native_live'])
        before = receipt['events']
        lease.recover()
        self.assertEqual(lease.receipt()['events'], before)


if __name__ == '__main__':
    unittest.main()
