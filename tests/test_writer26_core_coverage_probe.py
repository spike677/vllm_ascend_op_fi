"""Public C2 driver uses real Host/Runtime and only ordinary CPU enqueues."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
import test_core_writer26_coverage as fixture
from test_writer26_worker_core_probe import Driver
from experiments.c2c3_closure_20261117 import providers26_worker_core_probe as probe


class Core26CoverageProbeTests(unittest.TestCase):
    def setUp(self):
        self.case = fixture.CoreWriter26CoverageTests()
        self.case.setUp()
        self.case.make(max_targets=1 if self._testMethodName.startswith('test_blocked') else 128)
        self.addCleanup(self.case.doCleanups)
        self.out = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.driver = Driver(self.case)
        self.recorder = probe.RecordingExecutor(self.case.executor, self.out/'worker_core_rpc')
        self.case.client = WorkerFaultClient(HostAdapter(self.recorder,
            prepare_binding=self.case.client.host._binding), self.case.client.config)
        self.report = dict(status='STARTED', stage='c2_once', c2=False, c3=False,
            requests=[], healthy_before=dict(tokens=[0], text='CPU fixture'))

    def drive(self):
        return probe.drive(self.driver, self.case.client, executor=self.recorder,
            prompt=dict(prompt_token_ids=[1, 2, 3, 4]), params=object(), out=self.out, report=self.report)

    def test_complete_actual_public_driver_has_eight_phases_original_proof_and_health(self):
        with patch.object(self.case.client, 'snapshot', wraps=self.case.client.snapshot) as snapshot:
            result = self.drive()
        self.assertEqual(result['status'], probe.C2_CPU_CONTRACT, result.get('error'))
        self.assertEqual(snapshot.call_count, 1)
        self.assertEqual(len(self.driver.executed), 9)
        self.assertEqual(self.driver.executed, self.driver.enqueued)
        self.assertEqual([row['phase'] for row in result['requests']], list(probe.C2_PHASES))
        self.assertFalse(result['ownership_retained'] or result['c2'] or result['c3'])
        self.assertTrue(result['c2_attempt']['coverage']['contract_satisfied'])
        self.assertEqual(len(result['c2_attempt']['coverage']['live_c2_proof']['issued']), 1)
        final = result['c2_attempt']['original_observations']['originals'][-1]
        data = json.loads(Path(final['path']).read_text())
        self.assertEqual(len(data['inventory']['coverage']['calls']), 4)
        self.assertEqual(len(data['inventory']['coverage']['live_c2_proof']['issued']), 1)
        self.assertEqual(probe.file_sha(Path(final['path'])), final['sha256'])
        for record in result['rpc_receipts']:
            self.assertEqual(probe.file_sha(self.recorder.folder/record['path']), record['sha256'])

    def test_blocked_whole_group_runs_native_release_health_then_ordinary_health_without_fault(self):
        result = self.drive()
        self.assertEqual(result['status'], probe.C2_BLOCKED, result.get('error'))
        self.assertEqual([row['phase'] for row in result['requests']], ['inventory_1', 'inventory_2', 'health'])
        self.assertEqual(len(self.driver.executed), 4)
        self.assertFalse(self.case.native.executes or result['ownership_retained'] or result['c2'])
        self.assertEqual(len(result['c2_attempt']['candidate_index']), 2)
        self.assertIn('EMPTY_OR_OVER_BUDGET_COMPLETE_RESOLVED_TARGET_SET', result['complete_set_blockers'])


if __name__ == '__main__':
    unittest.main()
