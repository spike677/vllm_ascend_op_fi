"""Long-plan public driver and pre-F budget refusal, CPU only."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
import test_core_writer26_isolated as fixture
from test_writer26_worker_core_probe import Driver
import test_writer26_worker_core_probe as c1_probe
from experiments.c2c3_closure_20261117 import providers26_worker_core_probe as probe


class Core26IsolatedProbeTests(unittest.TestCase):
    def setUp(self):
        self.case = fixture.CoreWriter26IsolatedTests()
        self.case.setUp()
        self.case.make(count=3)
        self.addCleanup(self.case.doCleanups)
        self.out = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.driver = Driver(self.case)
        self.recorder = probe.RecordingExecutor(self.case.executor, self.out/'worker_core_rpc')
        self.case.client = WorkerFaultClient(HostAdapter(self.recorder,
            prepare_binding=self.case.client.host._binding), self.case.client.config)
        self.report = dict(status='STARTED', stage='c2_isolated_once', c2=False, c3=False,
            requests=[], healthy_before=dict(tokens=[0], text='CPU fixture'))

    def drive(self):
        return probe.drive(self.driver, self.case.client, executor=self.recorder,
            prompt=dict(prompt_token_ids=[1, 2, 3, 4]), params=object(), out=self.out, report=self.report)

    def test_driver_freezes_all_actual_targets_and_enqueues_separate_f_off_pairs(self):
        result = self.drive()
        self.assertEqual(result['status'], probe.C2_CPU_CONTRACT, result.get('error'))
        self.assertEqual(len(self.driver.executed), 13)
        self.assertEqual(self.driver.executed, self.driver.enqueued)
        self.assertEqual([row['phase'] for row in result['requests']],
            list(probe.C2_PHASES[:5])+['group_fault', 'group_off']*3+['health'])
        self.assertFalse(result['ownership_retained'] or result['c2'] or result['c3'])
        self.assertEqual(len(result['c2_attempt']['isolated_pairs']), 3)
        self.assertEqual(result['isolated_plan']['group_requests'], 12)
        self.assertFalse(result['isolated_budget_preflight']['refusal_reasons'])
        final = result['c2_attempt']['original_observations']['originals'][-1]
        data = json.loads(Path(final['path']).read_text())
        self.assertEqual(data['observation_kind'], 'FINAL_COMPLETE_REPORT')
        self.assertEqual(len(data['inventory']['coverage']['calls']), 18)
        self.assertEqual(probe.file_sha(Path(final['path'])), final['sha256'])
        for record in result['rpc_receipts']:
            self.assertEqual(probe.file_sha(self.recorder.folder/record['path']), record['sha256'])

    def test_pre_f_budget_refusal_runs_no_fault_and_releases_real_zero_resource_owner(self):
        original = probe.isolated_budget_preflight
        def too_slow(requests, plan, **kwargs):
            values = [dict(row, complete_cycle_seconds=200) for row in requests]
            return original(values, plan, **kwargs)
        with patch.object(probe, 'isolated_budget_preflight', too_slow):
            result = self.drive()
        self.assertEqual(result['status'], 'REFUSED_OR_FAILED')
        self.assertEqual(len(self.driver.executed), 5)
        self.assertEqual([row['phase'] for row in result['requests']], list(probe.C2_PHASES[:5]))
        self.assertFalse(any(row[1] == 'fault' for row in self.case.native.executes))
        self.assertFalse(result['ownership_retained'] or result['c2'] or result['c3'])
        self.assertEqual(result['isolated_budget_preflight']['status'], 'REFUSED_BEFORE_F')


class IsolatedBudgetTests(unittest.TestCase):
    def rows(self, values):
        return [dict(phase=phase, complete_cycle_seconds=duration)
            for phase, duration in zip(probe.C2_PHASES[2:5], values)]

    def test_sixty_four_request_estimate_keeps_existing_limits_and_evidence_reserve(self):
        value = probe.isolated_budget_preflight(self.rows([6, 6, 6]), dict(group_requests=64, full_native_writer_count=30), elapsed_seconds=180)
        self.assertFalse(value['refusal_reasons'])
        self.assertEqual(value['forecast_remaining_seconds'], 1188)
        self.assertEqual(value['python_control_overhead_reserve_seconds'], 360)
        self.assertEqual(value['estimated_evidence_reservation_bytes'], (536999909*128+16)//17)
        self.assertTrue(value['evidence_bytes_are_estimate_not_measured_bound'])
        self.assertEqual((value['existing_session_limit_seconds'], value['existing_root_case_limit_seconds']), (1200, 1800))

    def test_slow_or_late_actual_run_is_refused_before_any_f(self):
        for durations, elapsed in (([6, 6, 10], 180), ([6, 6, 6], 1000)):
            value = probe.isolated_budget_preflight(self.rows(durations), dict(group_requests=64, full_native_writer_count=30), elapsed_seconds=elapsed)
            self.assertEqual(value['status'], 'REFUSED_BEFORE_F')

    def test_missing_actual_cycles_are_not_a_time_budget(self):
        with self.assertRaisesRegex(RuntimeError, 'THREE_ACTUAL_COMPLETE'):
            probe.isolated_budget_preflight(self.rows([6, 6]), dict(group_requests=64, full_native_writer_count=30), elapsed_seconds=180)

    def test_full_native_denominator_scales_control_cost_and_cannot_be_omitted(self):
        value = probe.isolated_budget_preflight(self.rows([1, 1, 1]),
            dict(group_requests=64, full_native_writer_count=135), elapsed_seconds=180)
        self.assertEqual(value['python_control_overhead_reserve_seconds'], 1620)
        self.assertEqual(value['status'], 'REFUSED_BEFORE_F')
        with self.assertRaisesRegex(RuntimeError, 'FULL_NATIVE_DENOMINATOR'):
            probe.isolated_budget_preflight(self.rows([1, 1, 1]), dict(group_requests=64), elapsed_seconds=180)


class IsolatedCaseTests(unittest.TestCase):
    prepare_case = c1_probe.Core26CaseTests.prepare_case

    def test_explicit_case_preserves_native_limits_and_complete_set_no_c3(self):
        row = self.prepare_case('c2_isolated_once', 128)
        self.assertIn('--stage c2_isolated_once --max-targets 128', row['command'])
        self.assertIn('each target isolated', row['review']['execution_scope'])
        self.assertIn('C3 not requested', row['review']['expected_hits'])
        self.assertEqual(row['timeout'], 1800)
        self.assertEqual(row['review']['resource_budget']['independent_events'], 32768)
        self.assertIn('<=1200', row['review']['resource_budget']['isolated_pre_f_timing_gate'])
        self.assertFalse(row['review']['resource_budget']['max_targets_selects_subset'])
        self.assertIn('Two complete unmodified', row['review']['fia_stock_gate'])


if __name__ == '__main__':
    unittest.main()
