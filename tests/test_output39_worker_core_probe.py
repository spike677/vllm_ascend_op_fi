"""Natural driver through real public Host/Runtime and explicit CPU Native ABI."""
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

import test_core_output39 as fixture
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
from experiments.c2c3_postwrite_20261117 import natural39_worker_core_probe as probe
from experiments.c2c3_postwrite_20261117 import prepare39_worker_core as prepare


class Driver:
    def __init__(self, case):
        self.case, self.llm_engine = case, self
        self.engine_core = type('InprocClient', (), {'__module__':'vllm.v1.engine.core_client'})()
        self.output_processor = NS(request_states={})
        self.enqueued, self.executed, self.aborted = [], [], []
        self.fail_index = None

    def has_unfinished_requests(self):
        return bool(self.output_processor.request_states)

    def reset_prefix_cache(self, reset_running_requests):
        self.case.assertFalse(reset_running_requests or self.has_unfinished_requests())
        return True

    def enqueue(self, prompts, params, use_tqdm):
        self.case.assertEqual(prompts, [dict(prompt_token_ids=[1,2,3,4])])
        self.case.assertFalse(use_tqdm or self.has_unfinished_requests())
        request = 'enqueued-'+str(len(self.enqueued))
        self.output_processor.request_states[request] = NS(request_id=request, external_req_id='public-'+request)
        self.enqueued.append(request)
        return [request]

    def wait_for_completion(self, use_tqdm):
        request, = self.output_processor.request_states
        state = self.output_processor.request_states[request]
        self.case.actual_request_id = request
        provider = self.case.worker._demoncat_worker_fault_runtime._leases.get('39.output_proxy')
        if provider is not None:
            self.case.provider = provider.provider
        self.executed.append(request)
        self.case.model.raise_forward = len(self.enqueued)-1 == self.fail_index
        try:
            with torch.inference_mode():
                value = self.case.forward(self.case.provider.session)
        finally:
            self.case.model.raise_forward = False
        del self.output_processor.request_states[request]
        return [NS(request_id=state.external_req_id, outputs=[NS(token_ids=[int(value.argmax())], text='CPU fixture')])]

    def abort_request(self, ids, internal):
        self.case.assertTrue(internal)
        for request in ids:
            self.aborted.append(request)
            self.output_processor.request_states.pop(request, None)


class Output39ProbeTests(unittest.TestCase):
    def setUp(self):
        self.case = fixture.CoreOutput39Tests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.out = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.driver = Driver(self.case)
        self.recorder = probe.RecordingExecutor(self.case.executor, self.out/'worker_core_rpc')
        self.case.client = WorkerFaultClient(HostAdapter(self.recorder,
            prepare_binding=self.case.client.host._binding), self.case.client.config)
        self.report = dict(status='STARTED', healthy_before=dict(tokens=[0], text='CPU fixture'))

    def drive(self):
        return probe.drive(self.driver, self.case.client, executor=self.recorder,
            prompt=dict(prompt_token_ids=[1,2,3,4]), params=object(), out=self.out, report=self.report)

    def receipts(self):
        rows = []
        for item in self.report['rpc_receipts']:
            path = self.recorder.folder/item['path']
            self.assertEqual(probe.file_sha(path), item['sha256'])
            rows.append(json.loads(path.read_text()))
        self.assertEqual(probe.file_sha(self.out/'host_receipt.json'), self.report['host_receipt']['sha256'])
        return rows

    def test_full_real_protocol_records_every_original_rpc_and_only_one_extra_final_snapshot(self):
        with patch.object(self.case.client, 'snapshot', wraps=self.case.client.snapshot) as snapshot:
            result = self.drive()
        self.assertEqual(result['status'], probe.SUCCESS, result.get('error'))
        self.assertEqual(snapshot.call_count, 1)
        self.assertEqual(self.driver.executed, self.driver.enqueued)
        self.assertEqual(len(self.driver.enqueued), 7)
        self.assertTrue(result['c1']['c1_evidence_complete'])
        self.assertFalse(result['c1']['c2'] or result['c1']['c3'] or result['ownership_retained'])
        self.assertEqual(len(result['c1']['actual_requests']), 6)
        rows = self.receipts()
        self.assertEqual(sum(row['method'] == 'demoncat_fault_query' for row in rows), 6)
        self.assertEqual(sum(row['method'] == 'demoncat_fault_rpc' and row['args'][0] == 'advance' for row in rows), 6)
        self.assertEqual(len(rows), len(self.case.executor.calls))

    def test_failed_fault_preserves_receipts_and_only_real_off_health_follow(self):
        self.driver.fail_index = 3
        result = self.drive()
        self.assertEqual(result['status'], 'REFUSED_OR_FAILED')
        self.assertIn('CPU_ROOT_FORWARD_FAILED', result['error'])
        self.assertEqual([row['phase'] for row in result['recovery_requests']], ['off','health'])
        self.assertEqual(self.driver.aborted, [self.driver.enqueued[3]])
        self.assertEqual(len(self.driver.enqueued), 6)
        self.assertFalse(result['ownership_retained'], result.get('recovery_error'))
        self.assertNotIn('c1', result)
        self.receipts()

    def test_uncertain_fault_arm_never_replays_fault_and_persists_error(self):
        def lose(method, args, replies):
            if method == 'demoncat_fault_rpc' and args[0] == 'advance' and len(self.driver.enqueued) == 4:
                self.case.executor.after = None
                raise TimeoutError('CPU_LOST_FAULT_ARM_RESPONSE')
            return replies
        self.case.executor.after = lose
        result = self.drive()
        self.assertEqual(result['status'], 'REFUSED_OR_FAILED')
        self.assertNotIn(self.driver.enqueued[3], self.driver.executed)
        self.assertEqual(self.driver.aborted, [self.driver.enqueued[3]])
        self.assertFalse(result['ownership_retained'], result.get('recovery_error'))
        self.assertEqual(sum('error' in row for row in self.receipts()), 1)

    def test_statistics_refuses_before_import_or_model_creation(self):
        with redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as exc:
            probe.main(['--model','/m','--library','/l','--sha256','a'*64,
                '--build-fingerprint','b'*64,'--out','/evidence/x','--statistics'])
        self.assertEqual(exc.exception.code, 2)
        self.assertIn('OUTPUT39_STATISTICS_BOUNDARY_UNADAPTED', errors.getvalue())
        self.assertEqual(self.case.model.forward_calls, 0)

    def test_case_has_actual_mechanism_new_scope_and_local_complete_evidence(self):
        row, = prepare.core_cases(source='/s', native='/n', sha256='a'*64, build_fingerprint='b'*64)
        self.assertIn('FaultConfig(39,output_proxy)', row['review']['provider'])
        self.assertNotIn('--expected-targets', row['command'])
        self.assertIn('C2_C3_FALSE', row['review']['acceptance'])
        self.assertIn('/evidence/worker-core39-output.stderr.log', row['command'])
        self.assertTrue(all((probe.ROOT/path).is_file() for path in probe.SOURCES))
        for name, relative in (
                ('fault_injection.deployments.postwrite_npu_consumers',
                 'fault_injection/deployments/postwrite_npu_consumers.py'),
                ('demoncat.runtime.vllm_ascend.worker_extension.gemm_lineage',
                 'src/demoncat/runtime/vllm_ascend/worker_extension/gemm_lineage.py')):
            self.assertIn((name, relative), probe.MODULES)
            self.assertEqual(probe.SOURCES.count(relative), 1)


if __name__ == '__main__':
    unittest.main()
