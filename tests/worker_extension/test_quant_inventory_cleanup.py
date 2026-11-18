"""Confirmed per-patch cleanup progress; explicit CPU observer fixtures only."""
from types import ModuleType
import unittest
from unittest.mock import patch

import torch

import test_quant_inventory as fixtures


class QuantInventoryCleanupTests(unittest.TestCase):
    setUp = fixtures.QuantInventoryTests.setUp
    tearDown = fixtures.QuantInventoryTests.tearDown
    observer = fixtures.QuantInventoryTests.observer

    def test_partial_method_cleanup_retries_only_the_retained_patch(self):
        x = torch.ones(2, 4, dtype=torch.bfloat16)
        baseline = self.model(x)
        observer = self.observer()
        first, last = observer.patches
        foreign = lambda *args: None
        first[0].apply = foreign
        try:
            with self.assertRaisesRegex(RuntimeError, 'QUANT_METHOD_OBSERVER_OWNER_CHANGED'):
                observer.close()
            self.assertTrue(observer.installed)
            self.assertEqual(observer.patches, [first])
            self.assertIs(first[0].apply, foreign)
            self.assertNotIn('apply', vars(last[0]))
        finally:
            first[0].apply = first[2]
        observer.close()
        self.assertFalse(observer.installed or observer.patches or observer.api_patches)
        self.assertTrue(torch.equal(self.model(x), baseline))

    def test_partial_api_cleanup_retries_without_touching_foreign_owner(self):
        x = torch.ones(2, 4, dtype=torch.bfloat16)
        baseline = self.model(x)
        observer = self.observer()
        observer.begin('cleanup-api', 'a' * 64, request_ids=('cleanup-api',))
        first, last = observer.api_patches
        foreign = lambda *args: None
        setattr(first[0], first[1], foreign)
        try:
            with self.assertRaisesRegex(RuntimeError, 'PUBLIC_QUANT_OBSERVER_OWNER_CHANGED'):
                observer.close()
            self.assertEqual(observer.api_patches, [first])
            self.assertEqual(len(observer.patches), 2)
            self.assertIs(getattr(first[0], first[1]), foreign)
            self.assertIs(getattr(last[0], last[1]), last[2])
        finally:
            setattr(first[0], first[1], first[3])
        observer.close()
        self.assertFalse(observer.installed or observer.patches or observer.api_patches)
        self.assertTrue(torch.equal(self.model(x), baseline))

    def test_api_restore_then_raise_keeps_confirmed_progress_for_retry(self):
        class RaiseAfterRestore(ModuleType):
            fail_name = None
            def __setattr__(self, name, value):
                super().__setattr__(name, value)
                if name == self.fail_name:
                    raise RuntimeError('EXPLICIT_CPU_API_RESTORED_THEN_RAISED')
        self.api.__class__ = RaiseAfterRestore
        observer = self.observer()
        observer.begin('cleanup-api', 'a' * 64, request_ids=('cleanup-api',))
        first, last = observer.api_patches
        self.api.fail_name = last[1]
        with self.assertRaisesRegex(RuntimeError, 'API_RESTORED_THEN_RAISED'):
            observer.close()
        self.assertEqual(observer.api_patches, [first])
        self.assertIs(getattr(self.api, last[1]), last[2])
        self.api.fail_name = None
        observer.close()
        self.assertFalse(observer.installed or observer.patches or observer.api_patches)

    def test_method_restore_then_raise_keeps_confirmed_progress_for_retry(self):
        observer = self.observer()
        first, last = observer.patches
        def delete_then_raise(method, name):
            object.__delattr__(method, name)
            if method is last[0] and name == 'apply':
                raise RuntimeError('EXPLICIT_CPU_METHOD_RESTORED_THEN_RAISED')
        with patch.object(type(last[0]), '__delattr__', delete_then_raise, create=True):
            with self.assertRaisesRegex(RuntimeError, 'METHOD_RESTORED_THEN_RAISED'):
                observer.close()
        self.assertEqual(observer.patches, [first])
        self.assertNotIn('apply', vars(last[0]))
        observer.close()
        self.assertFalse(observer.installed or observer.patches or observer.api_patches)

    def test_local_method_restore_then_raise_keeps_exact_saved_callable(self):
        method = self.model.unused.quant_method
        saved = method.apply
        method.apply = saved
        observer = self.observer()
        first, last = observer.patches
        self.assertTrue(last[3])
        def set_then_raise(actual, name, value):
            object.__setattr__(actual, name, value)
            if actual is method and name == 'apply' and value is saved:
                raise RuntimeError('EXPLICIT_CPU_LOCAL_METHOD_RESTORED_THEN_RAISED')
        with patch.object(type(method), '__setattr__', set_then_raise, create=True):
            with self.assertRaisesRegex(RuntimeError, 'LOCAL_METHOD_RESTORED_THEN_RAISED'):
                observer.close()
        self.assertEqual(observer.patches, [first])
        self.assertIs(method.apply, saved)
        observer.close()
        self.assertFalse(observer.installed or observer.patches or observer.api_patches)

    def test_ignored_api_restore_does_not_drop_its_owned_patch(self):
        class IgnoreRestore(ModuleType):
            ignore_name = None
            def __setattr__(self, name, value):
                if name != self.ignore_name:
                    super().__setattr__(name, value)
        self.api.__class__ = IgnoreRestore
        observer = self.observer()
        observer.begin('cleanup-api', 'a' * 64, request_ids=('cleanup-api',))
        patches = list(observer.api_patches)
        self.api.ignore_name = patches[-1][1]
        with self.assertRaisesRegex(RuntimeError, 'PUBLIC_QUANT_OBSERVER_NOT_RESTORED'):
            observer.close()
        self.assertEqual(observer.api_patches, patches)
        self.assertIs(getattr(self.api, patches[-1][1]), patches[-1][3])
        self.api.ignore_name = None
        observer.close()
        self.assertFalse(observer.installed or observer.patches or observer.api_patches)

    def test_actual_inflight_projection_keeps_all_observers_on_close_refusal(self):
        observer = self.observer()
        self.api.npu_quant_matmul = lambda *args, **kwargs: observer.close()
        observer.begin('cleanup-inflight', 'a' * 64, request_ids=('cleanup-inflight',))
        methods, apis = list(observer.patches), list(observer.api_patches)
        with self.assertRaisesRegex(RuntimeError, 'ACTIVE_QUANT_PROJECTION_CANNOT_REMOVE_OBSERVER'):
            self.model(torch.ones(2, 4, dtype=torch.bfloat16))
        self.assertEqual(observer.patches, methods)
        self.assertEqual(observer.api_patches, apis)
        self.assertIsNone(observer.frame)
        observer.close()



if __name__ == "__main__":
    unittest.main()
