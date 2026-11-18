"""Original public drive/enqueue plus experiment-only callable RPCs on CPU."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.worker_core_v11_20261118 import output39_refusal_probe as probe
from experiments.worker_core_v11_20261118 import prepare_output39_refusal as prepare
from experiments.worker_core_v11_20261118.review39_refusal_diagnostic import review
from test_output39_refusal_diagnostic import OriginalTransport
from test_output39_worker_core_probe import Driver
import test_core_output39_group as fixture
import test_core_output39 as c1


class CallableExecutor:
    def __init__(self, case):
        self.case = case
    def collective_rpc(self, method, timeout=None, args=()):
        if callable(method):
            self.case.executor.calls.append((method.__name__,args))
            return [method(self.case.worker,*args)]
        return self.case.executor.collective_rpc(method, timeout=timeout, args=args)


class DiagnosticProbeTests(unittest.TestCase):
    def setUp(self):
        class Serial(fixture.ParallelProjections):
            def forward(self, input_ids, positions, inputs_embeds=None, intermediate_tensors=None):
                self.forward_calls += 1
                hidden = c1.life.CpuModel.forward(self,input_ids,positions,inputs_embeds,intermediate_tensors)
                return self.other(self.projection(hidden))
        self.case = fixture.CoreOutput39GroupTests()
        with patch.object(fixture,'ParallelProjections',Serial):
            self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.case.transport = OriginalTransport()
        self.out = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.driver = Driver(self.case)
        self.recorder = probe.original.RecordingExecutor(CallableExecutor(self.case),self.out/'rpc')
        self.case.client = probe.original.WorkerFaultClient(probe.original.HostAdapter(self.recorder,
            prepare_binding=self.case.client.host._binding), self.case.client.config)
        self.report = dict(status='STARTED',stage='c2_c3',healthy_before=dict(tokens=[0],text='CPU fixture'))
        self.pins = {name:probe.original.file_sha(probe.ROOT/name) for name in probe.SOURCES}
        self.plan = probe.DiagnosticPlan(predecessor_module='projection',refused_module='other',
            operator='aten.mm.default',refused_shape=(4,4),expected_all_calls=3,expected_eligible=2)
        self.enterContext(patch.object(probe.original,'actual_worker',lambda worker:worker))

    def run_driver(self):
        return probe.drive_once(self.driver,self.case.client,executor=self.recorder,
            prompt=dict(prompt_token_ids=[1,2,3,4]),params=object(),out=self.out,report=self.report,
            source_root=probe.ROOT,pins=self.pins,plan=self.plan,cpu_contract=True)

    def test_exact_original_protocol_installs_after_six_then_one_refusal_and_recovers(self):
        result = self.run_driver()
        self.assertEqual(result['status'],'REFUSED_OR_FAILED',result)
        self.assertIn('POSTWRITE_APPLY_4',result['error'])
        self.assertNotIn('diagnostic_close_error',result)
        self.assertNotIn('recovery_error',result)
        self.assertFalse(result['ownership_retained'])
        self.assertEqual(result['diagnostic_install']['actual_completed_requests'],6)
        self.assertEqual(result['diagnostic_install']['phase_budget']['maximum_user_requests'],17)
        self.assertFalse(result['diagnostic_install']['phase_budget']['original_schedule_modified'])
        self.assertEqual(result['diagnostic_budget']['normal_requests'],13)
        self.assertEqual(result['diagnostic_budget']['recovery_requests'],2)
        self.assertEqual([row['phase'] for row in result['recovery_requests']],
                         ['group_recovery_off','group_recovery_health'])
        self.assertEqual(sum(row['phase']=='c3_fault' for row in result['requests']),1)
        self.assertFalse(hasattr(self.case.worker,probe.ATTRIBUTE))
        ref = result['diagnostic_original']
        path = Path(ref['path'])
        self.assertEqual(probe.original.file_sha(path),ref['sha256'])
        saved = json.loads(path.read_text())
        self.assertEqual(review(saved)['issues'],[])
        self.assertEqual(len(saved['records']),5)
        self.assertTrue(ref['trace_removed'])
        self.assertFalse(self.case.transport.lib.leases)
        self.assertEqual(self.driver.executed,self.driver.enqueued)
        self.assertEqual(self.driver.aborted,[self.driver.enqueued[12]])

    def test_wrong_source_refuses_install_after_prefix_without_new_fault(self):
        self.pins[probe.SELF] = '0'*64
        result = self.run_driver()
        self.assertIn('ACTUAL_HELPER_SOURCE_NOT_PINNED',result['error'])
        self.assertEqual(result['diagnostic_budget']['normal_requests'],6)
        self.assertFalse(result['ownership_retained'],result.get('recovery_error'))
        self.assertNotIn('diagnostic_original',result)
        self.assertFalse(hasattr(self.case.worker,probe.ATTRIBUTE))

    def test_not_reproduced_is_explicit_diagnostic_abort_not_fabricated_apply4(self):
        self.case.model = fixture.ParallelProjections().eval()
        self.case.runner.model = self.case.model
        result = self.run_driver()
        self.assertIn('FROZEN_NATIVE_REFUSAL_NOT_REPRODUCED',result['error'])
        self.assertNotIn('POSTWRITE_APPLY_4',result['error'])
        self.assertFalse(result['ownership_retained'],result.get('recovery_error'))
        self.assertEqual(sum(row['phase']=='c3_fault' for row in result['requests']),1)
        data = json.loads(Path(result['diagnostic_original']['path']).read_text())
        last = next(row for row in data['records'] if row['arm']=='cumulative_fault' and row['target']=='refused')
        self.assertEqual(last['apply_event']['return_code'],0)
        self.assertEqual(last['span']['nonfinite_elements'],0)
        self.assertTrue(review(data)['issues'])

    def test_observation_write_error_is_retained_but_does_not_block_original_recovery(self):
        actual = probe.finish_diagnostic
        def fail_write(worker,*args):
            entry = getattr(worker,probe.ATTRIBUTE)
            entry['output'].mkdir()  # Explicit CPU I/O failure, not fabricated Native data.
            return actual(worker,*args)
        with patch.object(probe,'finish_diagnostic',fail_write):
            result = self.run_driver()
        self.assertIn('POSTWRITE_APPLY_4',result['error'])
        self.assertIn('diagnostic_close_error',result)
        self.assertFalse(result['ownership_retained'],result.get('recovery_error'))
        self.assertFalse(self.case.transport.lib.leases)
        # Failed evidence cannot be called complete; helper entry is retained for retry.
        self.assertTrue(hasattr(self.case.worker,probe.ATTRIBUTE))
        self.assertFalse(getattr(self.case.worker,probe.ATTRIBUTE)['observer'].installed)

    def test_budget_never_dispatches_second_cumulative_or_overdue_request(self):
        clock = [0.0]
        client = probe.BoundedDiagnosticClient(self.case.client,self.recorder,probe.ROOT,self.pins,
            self.out/'diagnostic.json',self.report,plan=self.plan,cpu_contract=True,clock=lambda:clock[0],deadline_seconds=1)
        before = len(self.case.executor.calls)
        clock[0] = 2
        with self.assertRaisesRegex(RuntimeError,'DEADLINE'):
            client.advance('never_enqueued_cpu_contract')
        clock[0] = 0
        client.next_phase,client.cumulative_started = 'c3_fault',True
        with self.assertRaisesRegex(RuntimeError,'ONLY_ONE_CUMULATIVE'):
            client.advance('never_enqueued_cpu_contract')
        self.assertEqual(len(self.case.executor.calls),before)

    def test_preparer_exports_one_fixed_case_without_duplicate_collector_or_statistics(self):
        row, = prepare.cases(source='/new-source',native='/new-native',sha256='a'*64,
            build_fingerprint='b'*64,model='/actual-model',card=5)
        self.assertEqual(row['timeout'],1800)
        self.assertEqual(row['review']['resource_budget']['timeout_seconds'],1800)
        self.assertEqual(row['review']['phase_budget']['diagnostic_normal_requests'],39)
        self.assertEqual(row['review']['phase_budget']['original_full_group_schedule'].count('c2_fault'),28)
        self.assertEqual(row['review']['expected_exit_code'],2)
        self.assertEqual(row['review']['maximum_user_requests_including_health'],43)
        self.assertNotIn('--statistics',row['command'])
        self.assertNotIn('--layer',row['command'])
        self.assertNotIn('--stage',row['command'])
        self.assertNotIn('/msprobe_packages',[mount[1] for mount in row['mounts']])
        self.assertTrue(all((probe.ROOT/name).is_file() for name in probe.SOURCES))


if __name__ == '__main__':
    unittest.main()
