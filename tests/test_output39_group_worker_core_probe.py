"""Public group driver invokes only ordinary CPU enqueues and saves originals."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_core_output39_group as fixture
from test_output39_worker_core_probe import Driver
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
from experiments.c2c3_postwrite_20261117 import natural39_worker_core_probe as probe
from experiments.c2c3_postwrite_20261117 import prepare39_worker_core as prepare


class Output39GroupProbeTests(unittest.TestCase):
    def setUp(self):
        self.case = fixture.CoreOutput39GroupTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.out = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.driver = Driver(self.case)
        self.recorder = probe.RecordingExecutor(self.case.executor, self.out/'worker_core_rpc')
        self.case.client = WorkerFaultClient(HostAdapter(self.recorder,
            prepare_binding=self.case.client.host._binding), self.case.client.config)
        self.report = dict(status='STARTED', stage='c2_c3', healthy_before=dict(tokens=[0], text='CPU fixture'))

    def drive(self):
        return probe.drive(self.driver, self.case.client, executor=self.recorder,
            prompt=dict(prompt_token_ids=[1,2,3,4]), params=object(), out=self.out, report=self.report)

    def test_complete_c2_c3_actual_request_sequence_original_rpc_health_and_live_proof(self):
        with patch.object(self.case.client, 'snapshot', wraps=self.case.client.snapshot) as snapshots:
            result = self.drive()
        self.assertEqual(result['status'], probe.GROUP_CPU, result.get('error'))
        self.assertEqual(snapshots.call_count, 1)
        self.assertEqual(self.driver.executed, self.driver.enqueued)
        self.assertEqual(len(result['requests']), 16)
        self.assertEqual(len(self.driver.executed), 17)
        self.assertFalse(result['ownership_retained'] or result['c2'] or result['c3'])
        for row in result['requests']:
            record = row['original_observations']
            self.assertEqual(probe.file_sha(Path(record['path'])), record['sha256'])
        for record in result['rpc_receipts']:
            self.assertEqual(probe.file_sha(self.recorder.folder/record['path']), record['sha256'])
        final = json.loads(Path(result['requests'][-1]['original_observations']['path']).read_text())
        self.assertEqual(final['finished_groups']['c2']['coverage']['hits'], 2)
        self.assertEqual(final['finished_groups']['c3']['coverage']['hits'], 4)

    def test_failed_group_fault_only_actual_off_health_recovery_no_retry(self):
        self.driver.fail_index = 7
        result = self.drive()
        self.assertEqual(result['status'], 'REFUSED_OR_FAILED')
        self.assertIn('CPU_ROOT_FORWARD_FAILED', result['error'])
        self.assertEqual([row['phase'] for row in result['recovery_requests']],
                         ['group_recovery_off','group_recovery_health'])
        self.assertFalse(result['ownership_retained'], result.get('recovery_error'))
        self.assertFalse(result.get('c2', False) or result.get('c3', False))
        self.assertEqual(self.driver.aborted, [self.driver.enqueued[7]])

    def test_prepared_group_case_has_full_set_scope_and_fixed_native_bounds(self):
        row, = prepare.core_cases(source='/s', native='/n', sha256='a'*64,
            build_fingerprint='b'*64, stage='c2_c3', max_live_leases=128)
        self.assertIn('--stage c2_c3 --max-live-leases 128', row['command'])
        self.assertEqual(row['review']['max_complete_targets'], 256)
        self.assertEqual(row['review']['execution_scope']['c2'], 'independent target replays')
        self.assertIn('nonfinite', row['review']['recovery_plan'])
        self.assertFalse(row['review']['legacy_model_local_evidence_inherited'])
        self.assertNotIn('layer', row['command'])


if __name__ == '__main__':
    unittest.main()
