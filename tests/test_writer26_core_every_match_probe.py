"""Public 26 C3 driver: ordinary enqueues, complete originals and no CPU grants."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
import test_core_writer26_every_match as fixture
import test_writer26_worker_core_probe as c1_probe
from experiments.c2c3_closure_20261117 import providers26_worker_core_probe as probe


class Core26EveryMatchProbeTests(unittest.TestCase):
    def setUp(self):
        self.case = fixture.CoreWriter26EveryMatchTests()
        self.case.setUp()
        self.case.make(count=2, max_targets=1 if self._testMethodName.startswith('test_blocked') else 128)
        self.addCleanup(self.case.doCleanups)
        self.out = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.driver = c1_probe.Driver(self.case)
        self.recorder = probe.RecordingExecutor(self.case.executor, self.out/'worker_core_rpc')
        self.case.client = WorkerFaultClient(HostAdapter(self.recorder,
            prepare_binding=self.case.client.host._binding), self.case.client.config)
        self.report = dict(status='STARTED', stage='c2_c3', c2=False, c3=False,
            requests=[], healthy_before=dict(tokens=[0], text='CPU fixture'))

    def drive(self):
        return probe.drive(self.driver, self.case.client, executor=self.recorder,
            prompt=dict(prompt_token_ids=[1, 2, 3, 4]), params=object(), out=self.out, report=self.report)

    def test_eighteen_actual_enqueues_keep_separate_generation_originals_and_one_host_snapshot(self):
        with patch.object(self.case.client, 'snapshot', wraps=self.case.client.snapshot) as snapshot:
            result = self.drive()
        self.assertEqual(result['status'], probe.C3_CPU_CONTRACT, result.get('error'))
        self.assertEqual(snapshot.call_count, 1)
        self.assertEqual(len(self.driver.executed), 18)
        self.assertEqual(self.driver.executed, self.driver.enqueued)
        phases = [(row['run'], row['phase']) for row in result['requests']]
        self.assertEqual(phases, [('c2', phase) for phase in probe.C2_PHASES]
            + [('c3', phase) for phase in probe.C3_PHASES])
        first, second = result['c2_attempt']['coverage'], result['c3_attempt']['coverage']
        self.assertEqual((first['hits'], second['hits']), (2, 4))
        issued, = first['live_c2_proof']['issued']
        consumed = second['live_c2_proof']['consumed']
        self.assertEqual(issued['object_binding'], consumed['object_binding'])
        self.assertEqual(issued['proof_sha256'], consumed['proof_sha256'])
        self.assertEqual((second['owner'], second['generation']), (first['owner'], first['generation']+1))
        self.assertTrue(first['contract_satisfied'] and second['contract_satisfied'])
        self.assertFalse(result['ownership_retained'] or result['c2'] or result['c3'])
        for record in result['rpc_receipts']:
            self.assertEqual(probe.file_sha(self.recorder.folder/record['path']), record['sha256'])
        for key, calls in (('c2_attempt', 4), ('c3_attempt', 6)):
            final = result[key]['original_observations']['originals'][-1]
            data = json.loads(Path(final['path']).read_text())
            self.assertEqual(probe.file_sha(Path(final['path'])), final['sha256'])
            self.assertEqual(len(data['inventory']['coverage']['calls']), calls)
            self.assertEqual(data['rpc_generation'], first['generation'])
            self.assertEqual(data['native_generation'], result[key]['coverage']['generation'])
            self.assertEqual(data['run'], key.removesuffix('_attempt'))
        receipts = [json.loads((self.recorder.folder/row['path']).read_text()) for row in result['rpc_receipts']]
        collects = [row for row in receipts if row['method'] == 'demoncat_fault_rpc'
            and row['args'][0] == 'advance' and row['args'][1].get('settings') == {'collect': True}]
        self.assertEqual(len(collects), 17)

    def test_blocked_c2_keeps_full_set_and_never_starts_c3_or_faults(self):
        result = self.drive()
        self.assertEqual(result['status'], probe.C2_BLOCKED, result.get('error'))
        self.assertEqual(result['blocked_run'], 'c2')
        self.assertEqual([row['phase'] for row in result['requests']], ['inventory_1', 'inventory_2', 'health'])
        self.assertEqual(len(self.driver.executed), 4)
        self.assertEqual(len(result['c2_attempt']['candidate_index']), 2)
        self.assertFalse(self.case.provider.c3_started or self.case.native.executes
            or result['ownership_retained'] or result['c2'] or result['c3'])
        self.assertNotIn('c3_attempt', result)

    def test_new_c3_model_failure_keeps_original_c2_and_never_retries_fault(self):
        self.driver.fail_index = 10
        result = self.drive()
        self.assertEqual(result['status'], 'REFUSED_OR_FAILED')
        self.assertEqual(len(self.driver.executed), 11)
        self.assertEqual(self.driver.aborted, [self.driver.enqueued[-1]])
        self.assertEqual(len([row for row in self.case.native.executes if row[1] == 'fault']), 2)
        self.assertFalse(result['ownership_retained'] or result['c3'])
        self.assertTrue(self.case.provider.c2_session.backend.finalized)
        self.assertTrue(result['c2_attempt']['coverage']['contract_satisfied'])
        self.assertIn('CPU', result['error'])

    def test_new_c3_allocator_flip_blocks_whole_second_group_with_original_c2_preserved(self):
        backend_type = fixture.core.CoreWriter26Experiment
        allocation = backend_type._allocation
        def changed(backend, call, row):
            if backend.trigger == 'every_match' and backend.phase == 'inventory_2' and call.cache == 1001:
                row['proven_nonadjacent'] = [[(10, 20, 'active_allocated')], [(30, 40, 'active_allocated')]]
                raise RuntimeError('LIVE_ALLOCATOR_BLOCKS_PROVEN_NOT_ADJACENT')
            return allocation(backend, call, row)
        with patch.object(backend_type, '_allocation', changed):
            result = self.drive()
        self.assertEqual(result['status'], probe.C3_BLOCKED, result.get('error'))
        self.assertEqual(result['blocked_run'], 'c3')
        self.assertEqual(len(self.driver.executed), 12)
        self.assertTrue(result['c2_attempt']['coverage']['contract_satisfied'])
        candidates = result['c3_attempt']['candidate_index']
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[1]['status'], 'UNVERIFIED')
        self.assertEqual(len([row for row in self.case.native.executes if row[1] == 'fault']), 2)
        self.assertFalse(result['ownership_retained'] or result['c3'])


class Core26EveryMatchCaseTests(unittest.TestCase):
    prepare_case = c1_probe.Core26CaseTests.prepare_case

    def test_case_uses_public_combined_stage_and_full_census_budget(self):
        row = self.prepare_case('c2_c3', 128)
        self.assertIn('--stage c2_c3 --max-targets 128', row['command'])
        self.assertIn('public Core all_eligible C2 then original-live-proof C3', row['review']['provider'])
        self.assertIn('WorkerFaultClient', row['review']['public_entry'])
        self.assertFalse(row['review']['resource_budget']['max_targets_selects_subset'])
        self.assertEqual(row['review']['resource_budget']['independent_events'], 32768)
        self.assertIn('blocks the whole group; no subset', row['review']['c2_dynamic_eligibility'])
        self.assertIn('original issued proof object', row['review']['recovery_plan'])


if __name__ == '__main__':
    unittest.main()
