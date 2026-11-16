"""Exercise the temporary serving probe against the real CPU RPC state machine."""
from pathlib import Path
import argparse
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

try:
    from . import test_extension as fixtures
except ImportError:
    import test_extension as fixtures
from experiments.overnight_20261116 import worker_probe


class ProbeContractTests(unittest.TestCase):
    setUp = fixtures.WorkerExtensionTest.setUp

    def test_memory_budget_has_explicit_bounds_and_rejects_nonfinite(self):
        for value in ('0.1', '0.35', '0.75', '0.85'):
            self.assertEqual(worker_probe.memory_utilization(value), float(value))
        for value in ('0.0999', '0.8501', 'nan', 'inf', '-inf'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                worker_probe.memory_utilization(value)

    def llm(self):
        worker = self.worker
        class ExistingExecutor:
            def collective_rpc(self, method, timeout=None, args=()):
                return [getattr(worker, method)(*args)]
        return ExistingExecutor()

    def test_probe_checks_global_refusal_and_recovers_both_provider_contracts(self):
        self.worker.load_model()
        for kind in ('fault20', 'fault39'):
            with self.subTest(kind=kind):
                error = RuntimeError('deployment replay is intentionally absent')
                error.code = worker_probe.REPLAY_REFUSALS[kind]
                self.provider.callbacks['register'] = Mock(side_effect=error)
                record = {'workers': [self.worker.demoncat_worker_identity()['worker']]}
                before = self.resolver.call_count
                worker_probe.quarantine_case(self.llm(), 'quarantine-' + kind, record)
                self.assertEqual(self.resolver.call_count - before, 1)
                self.assertEqual(record['other_provider']['reason'], 'WORKER_RECOVERY_REQUIRED')
                self.assertEqual(record['recovery']['status'], 'ACTION_COMPLETED')
                self.assertEqual(record['detach']['status'], 'DETACHED')
                self.assertFalse(self.worker.demoncat_capabilities()['quarantined'])

    def test_probe_does_not_report_recovery_when_native_release_fails(self):
        self.worker.load_model()
        error = RuntimeError('missing scheduler integration')
        error.code = worker_probe.REPLAY_REFUSALS['fault39']
        self.provider.callbacks['register'] = Mock(side_effect=error)
        self.provider.callbacks['unregister'] = Mock(side_effect=RuntimeError('unresolved'))
        record = {'workers': [self.worker.demoncat_worker_identity()['worker']]}
        with self.assertRaisesRegex(RuntimeError, 'RECOVERY_NOT_CONFIRMED'):
            worker_probe.quarantine_case(self.llm(), 'quarantine-fault39', record)
        self.assertNotIn('detach', record)
        self.assertTrue(self.worker.demoncat_worker_identity()['quarantined'])

    def test_failed_observer_close_retains_owner_for_isolation(self):
        session = SimpleNamespace(close=Mock(side_effect=RuntimeError('callback in flight')))
        owned = (session, object(), Path('/unused'))
        worker = SimpleNamespace(_demoncat_overnight_observer=owned)
        with self.assertRaisesRegex(RuntimeError, 'callback in flight'):
            worker_probe.observe_finish(worker)
        self.assertIs(worker._demoncat_overnight_observer, owned)

    def test_cleanup_pending_report_cannot_release_observation_owner(self):
        report = dict(events=[], forwards=[], cleanup_pending=True)
        session = SimpleNamespace(close=lambda:None, report=lambda:report)
        native = SimpleNamespace(operator='aclnnMatmulWeightNz', snapshot=lambda:dict(
            call=dict(native_registered=False, inflight=0, leases=0)))
        with TemporaryDirectory() as directory:
            owned = (session, native, Path(directory)/'observation-rank0')
            worker = SimpleNamespace(_demoncat_overnight_observer=owned, rank=0)
            with self.assertRaisesRegex(RuntimeError, 'DID_NOT_DRAIN'):
                worker_probe.observe_finish(worker)
            self.assertIs(worker._demoncat_overnight_observer, owned)


if __name__ == '__main__':
    unittest.main()
