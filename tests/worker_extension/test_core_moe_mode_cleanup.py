"""Actual CPU Host/Runtime and dispatch-stack ownership on MoE exit failure."""
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode_stack

import test_core_tensor_serving as core
from fault_injection.moe_census import NaturalMoETrace
from fault_injection.moe_restore import MoERestoreTrace


class PassThrough(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


class PublicMoEModeCleanupTests(unittest.TestCase):
    def setUp(self):
        self.case = core.CoreMoEServingTests(); self.case.setUp()
        self.owned = []

    def tearDown(self):
        # This is fixture emergency teardown only, never the recovery proof.
        for trace, mode, original in reversed(self.owned):
            mode.__exit__ = original
            stack = _get_current_dispatch_mode_stack()
            if any(item is mode for item in stack) and stack[-1] is mode:
                trace._close_native_mode(mode)
        try:
            if self.case.client.snapshot()['ticket'] is not None:
                self.case.client.recover(); self.case.client.unregister(); self.case.client.detach()
        finally:
            self.case.tearDown()

    def fail_actual_exit(self, *, after=False):
        c = self.case; c.start(); c.ok(c.client.advance('actual-nested-mode-failure'))
        original_open = MoERestoreTrace._open_native
        def opened(trace, row):
            mode = original_open(trace, row); original = mode.__exit__
            self.owned.append((trace, mode, original))
            def failed(*exc):
                if after: original(*exc)
                raise RuntimeError('CPU nested exit ' + ('AFTER_POP' if after else 'BEFORE_POP'))
            mode.__exit__ = failed
            return mode
        with patch.object(MoERestoreTrace, '_open_native', opened):
            with self.assertRaises(Exception):
                c.actual('actual-nested-mode-failure')
        self.assertTrue(self.owned)
        return self.owned[0]

    def clean_retry(self, trace):
        c = self.case; first_error = trace.errors[0]
        calls = c.model.forward_calls
        c.recovery()
        self.assertEqual(c.client.detach()['status'], 'DETACHED')
        self.assertFalse(c.client.snapshot()['ownership_retained'])
        self.assertEqual(c.model.forward_calls, calls)
        self.assertEqual(_get_current_dispatch_mode_stack(), [])
        self.assertEqual(trace.owned_modes, [])
        self.assertEqual(trace.pending, {})
        self.assertFalse(trace.active)
        self.assertEqual(trace.errors[0], first_error)
        self.assertFalse(trace.report()['complete'])
        receipts = [row['failed_phase']['independent']['moe_trace'] for row in c.provider.failures
                    if row.get('failed_phase', {}).get('independent') is not None]
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0]['errors'][0], first_error)
        self.assertEqual(receipts[0]['owned_native_modes'], 0)
        self.assertEqual(receipts[0]['pending_python_boundaries'], 0)
        self.assertFalse(receipts[0]['profile_active'])
        self.assertFalse(receipts[0]['complete'])
        self.assertIsNone(c.provider.proof)
        self.assertNotIn('C2', c.provider.reports)
        c.assert_clean()
        c.actual('ordinary-same-worker-after-cleanup')
        self.assertEqual(c.model.forward_calls, calls + 1)
        self.assertEqual(_get_current_dispatch_mode_stack(), [])

    def test_before_pop_keeps_pending_mode_and_public_owner_until_real_retry(self):
        trace, mode, original = self.fail_actual_exit()
        self.assertIn(mode, _get_current_dispatch_mode_stack())
        self.assertEqual(trace.owned_modes, [mode])
        self.assertEqual(len(trace.pending), 1)
        self.assertEqual(set(trace.pending), trace.returned_frames)
        first_error = trace.errors[0]
        for action in ('recover', 'unregister'):
            reply = getattr(self.case.client, action)()
            self.assertEqual(reply['status'], 'PROVIDER_ACTION_FAILED')
            self.assertTrue(reply['ownership_retained'])
            self.assertTrue(self.case.worker._demoncat_worker_fault_runtime._leases)
            self.assertTrue(self.case.provider.binding.leases)
            self.assertIn(mode, _get_current_dispatch_mode_stack())
        self.assertEqual(self.case.client.detach()['status'], 'RECOVERY_REQUIRED')
        self.assertEqual(trace.errors[0], first_error)
        mode.__exit__ = original
        self.clean_retry(trace)

    def test_after_pop_exception_retires_only_absent_mode_and_keeps_first_error(self):
        trace, mode, original = self.fail_actual_exit(after=True)
        self.assertNotIn(mode, _get_current_dispatch_mode_stack())
        self.assertEqual(trace.owned_modes, [])
        self.assertEqual(trace.pending, {})
        self.assertIn('AFTER_POP', trace.errors[0])
        self.assertTrue(self.case.client.snapshot()['ownership_retained'])
        def stale(*exc): raise AssertionError('Already-popped mode exited twice')
        for _, held, _ in self.owned: held.__exit__ = stale
        self.clean_retry(trace)

    def test_foreign_top_is_not_popped_or_treated_as_own_cleanup(self):
        trace, mode, original = self.fail_actual_exit()
        mode.__exit__ = original
        foreign = PassThrough(); foreign.__enter__()
        try:
            reply = self.case.client.recover()
            self.assertEqual(reply['status'], 'PROVIDER_ACTION_FAILED')
            self.assertTrue(reply['ownership_retained'])
            self.assertIs(_get_current_dispatch_mode_stack()[-1], foreign)
            self.assertEqual(trace.owned_modes, [mode])
        finally:
            foreign.__exit__(None, None, None)
        self.clean_retry(trace)

    def test_foreign_thread_and_profile_cannot_release_original_owner(self):
        trace, mode, original = self.fail_actual_exit()
        mode.__exit__ = original
        errors = []
        def close_elsewhere():
            try: trace.close()
            except Exception as error: errors.append(str(error))
        thread = threading.Thread(target=close_elsewhere); thread.start(); thread.join()
        self.assertEqual(errors, ['MOE_PROFILE_OWNERSHIP_CHANGED'])
        foreign = lambda *args: None
        sys.setprofile(foreign)
        try:
            reply = self.case.client.unregister()
            self.assertNotEqual(reply['status'], 'ACTION_COMPLETED')
            self.assertTrue(reply['ownership_retained'])
            self.assertIs(sys.getprofile(), foreign)
            self.assertEqual(trace.owned_modes, [mode])
            self.assertIn(mode, _get_current_dispatch_mode_stack())
        finally:
            sys.setprofile(trace.callback)
        self.clean_retry(trace)


class DirectMoEModeLedgerTests(unittest.TestCase):
    def test_active_frame_and_noop_exit_cannot_report_cleanup(self):
        trace = NaturalMoETrace(SimpleNamespace(named_modules=lambda: ()), [], {}, simulation=True)
        # Avoid a model fixture: the ledger is exercised with a real mode stack.
        trace.__enter__(); mode = trace._enter_native_mode(PassThrough())
        original = mode.__exit__; trace.pending[1] = ({}, mode)
        try:
            with self.assertRaisesRegex(RuntimeError, 'BOUNDARY_STILL_INFLIGHT'): trace.close()
            trace.returned_frames.add(1); mode.__exit__ = lambda *args: None
            with self.assertRaisesRegex(RuntimeError, 'REMOVAL_UNCONFIRMED'): trace.close()
            self.assertEqual(trace.owned_modes, [mode])
            self.assertEqual(len(trace.pending), 1)
            self.assertIs(_get_current_dispatch_mode_stack()[-1], mode)
        finally:
            mode.__exit__ = original; trace.close()
        self.assertEqual(_get_current_dispatch_mode_stack(), [])

    def test_enter_throw_after_push_retains_mode_before_frame_bookkeeping(self):
        trace = NaturalMoETrace(SimpleNamespace(named_modules=lambda: ()), [], {}, simulation=True); trace.__enter__()
        mode = PassThrough(); original = mode.__enter__
        def failed():
            original(); raise RuntimeError('CPU after push')
        mode.__enter__ = failed
        try:
            with self.assertRaisesRegex(RuntimeError, 'after push'): trace._enter_native_mode(mode)
            self.assertEqual(trace.owned_modes, [mode])
            self.assertIn(mode, _get_current_dispatch_mode_stack())
        finally:
            trace.close()
        self.assertEqual(_get_current_dispatch_mode_stack(), [])

    def test_enter_throw_before_push_does_not_invent_an_owned_mode(self):
        trace = NaturalMoETrace(SimpleNamespace(named_modules=lambda: ()), [], {}, simulation=True); trace.__enter__()
        mode = PassThrough()
        def failed(): raise RuntimeError('CPU before push')
        mode.__enter__ = failed
        try:
            with self.assertRaisesRegex(RuntimeError, 'before push'): trace._enter_native_mode(mode)
            self.assertEqual(trace.owned_modes, [])
            self.assertEqual(_get_current_dispatch_mode_stack(), [])
        finally:
            trace.close()


if __name__ == '__main__': unittest.main()
