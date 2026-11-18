"""Explicit public isolated-to-cumulative driver; complete CPU originals."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
import test_core_writer26_isolated_transition as fixture
import test_writer26_worker_core_probe as c1_probe
from test_writer26_isolated_transition_budget import cost_fixture
from experiments.c2c3_closure_20261117 import providers26_worker_core_probe as probe
from experiments.c2c3_closure_20261117 import providers26_isolated_budget as budget


class Core26IsolatedTransitionProbeTests(unittest.TestCase):
    def setUp(self):
        self.case = fixture.CoreWriter26IsolatedTransitionTests()
        self.case.setUp()
        self.case.make(count=3, max_targets=1 if self._testMethodName.startswith('test_blocked') else 128)
        self.addCleanup(self.case.doCleanups)
        self.out = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.reference, self.cost_paths = cost_fixture(self, self.out)
        self.driver = c1_probe.Driver(self.case)
        self.recorder = probe.RecordingExecutor(self.case.executor, self.out/'worker_core_rpc')
        self.case.client = WorkerFaultClient(HostAdapter(self.recorder,
            prepare_binding=self.case.client.host._binding), self.case.client.config)
        self.report = dict(status='STARTED', stage='c2_isolated_c3', c2=False, c3=False,
            isolated_cost_reference=self.reference, requests=[], healthy_before=dict(tokens=[0], text='CPU fixture'))

    def drive(self):
        return probe.drive(self.driver, self.case.client, executor=self.recorder,
            prompt=dict(prompt_token_ids=[1, 2, 3, 4]), params=object(), out=self.out, report=self.report)

    def test_nineteen_group_requests_then_healthy_cleanup_consume_one_original_live_proof(self):
        with patch.object(self.case.client, 'snapshot', wraps=self.case.client.snapshot) as snapshot:
            result = self.drive()
        self.assertEqual(result['status'], probe.C3_CPU_CONTRACT, result.get('error'))
        self.assertEqual(snapshot.call_count, 1)
        self.assertEqual(self.driver.executed, self.driver.enqueued)
        self.assertEqual(len(self.driver.executed), 20)
        self.assertEqual([(row['run'], row['phase']) for row in result['requests']],
            [('c2', phase) for phase in probe.C2_PHASES[:5]+('group_fault', 'group_off')*2+('health',)]
            + [('c3', phase) for phase in probe.C3_PHASES])
        a, b = result['c2_attempt']['coverage'], result['c3_attempt']['coverage']
        self.assertEqual((a['hits'], b['hits'], a['observed_calls'], b['observed_calls']), (2, 4, 12, 9))
        self.assertEqual(a['live_c2_proof']['issued'][0]['object_binding'], b['live_c2_proof']['consumed']['object_binding'])
        self.assertFalse(result['c2'] or result['c3'] or result['ownership_retained'])
        self.assertEqual(set(result['transition_budget_preflights']), {'c2', 'c3'})
        self.assertEqual(result['transition_budget_preflights']['c3']['c3_remaining_requests'], 4)
        self.assertTrue(result['outer_clock']['continuous_across_generations'])
        for run in ('c2', 'c3'):
            refs = result[run+'_attempt']['original_observations']['originals']
            previous = None
            for ref in refs:
                self.assertEqual(probe.file_sha(Path(ref['path'])), ref['sha256'])
                raw = json.loads(Path(ref['path']).read_text())
                self.assertEqual(raw['previous_original_sha256'], previous)
                self.assertEqual(raw['run'], run)
                previous = ref['sha256']
            self.assertEqual(raw['observation_kind'], 'FINAL_COMPLETE_REPORT')
            self.assertEqual(len(raw['inventory']['candidates']), 3)
        for ref in result['rpc_receipts']:
            self.assertEqual(probe.file_sha(self.recorder.folder/ref['path']), ref['sha256'])

    def test_before_c2_fault_cost_refusal_keeps_every_candidate_and_clean_owner(self):
        original = probe.transition_budget
        def slow(requests, plan, reference, **kwargs):
            return original(requests, plan, reference, **dict(kwargs, elapsed_seconds=1800))
        with patch.object(probe, 'transition_budget', slow):
            result = self.drive()
        self.assertEqual(result['status'], 'REFUSED_OR_FAILED')
        self.assertEqual(len(self.driver.executed), 5)
        self.assertFalse(any(mode == 'fault' for _, mode, _ in self.case.native.executes))
        self.assertEqual(len(self.case.provider.backend.candidate_inventory), 3)
        self.assertFalse(result['ownership_retained'])
        self.assertEqual(result['transition_budget_preflights']['c2']['status'], 'REFUSED_BEFORE_F')

    def test_before_c3_fault_late_outer_clock_preserves_completed_c2_without_new_faults(self):
        original = probe.transition_budget
        def late(requests, plan, reference, **kwargs):
            if kwargs['run'] == 'c3':
                kwargs['elapsed_seconds'] = 1800
            return original(requests, plan, reference, **kwargs)
        with patch.object(probe, 'transition_budget', late):
            result = self.drive()
        self.assertEqual(result['status'], 'REFUSED_OR_FAILED')
        self.assertEqual(len(self.driver.executed), 15)
        self.assertEqual(len([row for row in self.case.native.executes if row[1] == 'fault']), 2)
        self.assertTrue(result['c2_attempt']['coverage']['contract_satisfied'])
        self.assertEqual(len(result['c2_attempt']['isolated_pairs']), 2)
        self.assertFalse(result['ownership_retained'] or result['c3'])
        self.assertEqual(result['transition_budget_preflights']['c3']['status'], 'REFUSED_BEFORE_F')

    def test_blocked_full_census_cannot_choose_subset_or_start_c3(self):
        result = self.drive()
        self.assertEqual(result['status'], probe.C2_BLOCKED, result.get('error'))
        self.assertEqual(len(result['c2_attempt']['candidate_index']), 3)
        self.assertEqual(len(self.driver.executed), 4)
        self.assertFalse(self.case.native.executes or self.case.provider.c3_started or result['ownership_retained'])

    def test_new_c3_model_failure_keeps_old_originals_without_retry(self):
        self.driver.fail_index = 12
        result = self.drive()
        self.assertEqual(result['status'], 'REFUSED_OR_FAILED')
        self.assertEqual(len(self.driver.executed), 13)
        self.assertEqual(self.driver.aborted, [self.driver.enqueued[-1]])
        self.assertEqual(len([row for row in self.case.native.executes if row[1] == 'fault']), 2)
        self.assertTrue(result['c2_attempt']['coverage']['contract_satisfied'])
        self.assertFalse(result['ownership_retained'] or result['c3'])


class IsolatedTransitionCaseTests(unittest.TestCase):
    prepare_case = c1_probe.Core26CaseTests.prepare_case

    def test_case_uses_two_readonly_fixed_cost_originals_and_continuous_case_budget(self):
        cases = c1_probe.cases
        original = cases.main
        def main():
            sys.argv.extend(['--isolated-cost-result', '/source82-result.json', '--isolated-cost-controller', '/source82-controller.json'])
            return original()
        reference = dict(originals={name:dict(path='/originals/'+name+'.json', sha256='a'*64) for name in ('result', 'controller')})
        with patch.object(cases, 'main', main), patch.object(budget, 'load_cost_reference', return_value=reference) as load:
            row = self.prepare_case('c2_isolated_c3', 128)
        self.assertEqual(load.call_count, 1)
        self.assertIn('--stage c2_isolated_c3', row['command'])
        self.assertIn('--isolated-cost-result /cost/source82-result.json', row['command'])
        self.assertIn('--isolated-cost-controller /cost/source82-controller.json', row['command'])
        self.assertIn(['/originals/result.json', '/cost/source82-result.json', 'ro'], row['mounts'])
        self.assertIn(['/originals/controller.json', '/cost/source82-controller.json', 'ro'], row['mounts'])
        self.assertIn('cumulative C3', row['review']['execution_scope'])
        self.assertEqual(row['timeout'], 1800)
        self.assertIn('generation <=1200', row['review']['resource_budget']['isolated_pre_f_timing_gate'])
        self.assertEqual(row['review']['resource_budget']['isolated_group_requests'], '15 + 2 * actual stable eligible count')
        self.assertFalse(row['review']['previous_direct_entry_qualification_inherited'])

    def test_missing_cost_originals_save_refusal_before_startup_or_model_loading(self):
        from fault_injection.deployments import writer26_loader
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'probe'
            real_path = Path
            def mapped(value):
                return out if str(value) == '/evidence/probe' else real_path(value)
            with patch.object(probe, 'Path', side_effect=mapped), patch.object(writer26_loader, 'StartupDependencies') as startup:
                code = probe.main(['--mechanism', '26.fia_destination', '--manifest-sha256', 'a'*64,
                    '--source-root', '/source', '--stage', 'c2_isolated_c3', '--max-targets', '128'])
            self.assertEqual(code, 2)
            startup.assert_not_called()
            result = json.loads((out/'result.json').read_text())
            self.assertIn('ISOLATED_C3_COST_ORIGINAL_REQUIRED', result['error'])
            self.assertEqual(result['requests'], [])
            self.assertFalse(result['c2'] or result['c3'] or result['qualification_granted'])


if __name__ == '__main__':
    unittest.main()
