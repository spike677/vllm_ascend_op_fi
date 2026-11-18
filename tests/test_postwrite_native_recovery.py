"""Original Python Native lease with an explicit CPU ABI double; no SDK proof."""
from types import SimpleNamespace as NS
import unittest

import torch

from fault_injection.deployments.postwrite import _NativeLease, PostwriteRefused


class CpuABI:
    def __init__(self, tensor):
        self.tensor, self.original = tensor, tensor.clone()
        self.owners, self.next_handle = {}, 0
        self.restore_calls, self.release_calls, self.readbacks = [], [], []
        self.fail = None
        self.fail_readback = False

    def dcat_pw_open(self, address, size, device, owner, generation, result):
        self.next_handle += 1
        result._obj.value = self.next_handle
        self.owners[self.next_handle] = (owner, generation)
        return 0

    def owned(self, handle, owner, generation):
        return self.owners.get(handle.value) == (owner, generation)

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


if __name__ == '__main__':
    unittest.main()
