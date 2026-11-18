"""Real public Host/Runtime driver with CPU model/Native/scheduler fixtures."""
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

import test_core_writer26 as fixture
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
from experiments.c2c3_closure_20261117 import providers26_worker_core_probe as probe
from experiments.c2c3_closure_20261117 import providers26_prepare_group_cases as cases


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
        lease = self.case.worker._demoncat_worker_fault_runtime._leases.get(self.case.mechanism)
        if lease is not None:
            self.case.provider = lease.provider.core
        self.executed.append(request)
        self.case.model.processor.raise_forward = len(self.enqueued)-1 == self.fail_index
        try:
            with torch.inference_mode():
                value = self.case.forward(self.case.provider.session)
        finally:
            self.case.model.processor.raise_forward = False
        del self.output_processor.request_states[request]
        return [NS(request_id=state.external_req_id, outputs=[NS(token_ids=[int(value.argmax())], text='CPU fixture')])]

    def abort_request(self, ids, internal):
        self.case.assertTrue(internal)
        for request in ids:
            self.aborted.append(request)
            self.output_processor.request_states.pop(request, None)


class Core26ProbeTests(unittest.TestCase):
    def setUp(self):
        self.case = fixture.CoreWriter26Tests()
        self.case.setUp()
        self.case.make()
        self.addCleanup(self.case.doCleanups)
        self.out = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.driver = Driver(self.case)
        self.recorder = probe.RecordingExecutor(self.case.executor, self.out/'worker_core_rpc')
        self.case.client = WorkerFaultClient(HostAdapter(self.recorder,
            prepare_binding=self.case.client.host._binding), self.case.client.config)
        self.report = dict(status='STARTED', requests=[], healthy_before=dict(tokens=[0], text='CPU fixture'))

    def drive(self):
        return probe.drive(self.driver, self.case.client, executor=self.recorder,
            prompt=dict(prompt_token_ids=[1,2,3,4]), params=object(), out=self.out, report=self.report)

    def test_eight_actual_enqueues_explicit_collect_every_rpc_and_one_final_host_snapshot(self):
        with patch.object(self.case.client, 'snapshot', wraps=self.case.client.snapshot) as snapshot:
            result = self.drive()
        self.assertEqual(result['status'], probe.SUCCESS, result.get('error'))
        self.assertEqual(snapshot.call_count, 1)
        self.assertEqual(len(self.driver.executed), 9)  # Eight plus post-unregister health.
        self.assertEqual(self.driver.executed, self.driver.enqueued)
        self.assertEqual([row['phase'] for row in result['requests']], list(probe.PHASES))
        self.assertFalse(result['ownership_retained'])
        for record in result['rpc_receipts']:
            path = self.recorder.folder/record['path']
            self.assertEqual(probe.file_sha(path), record['sha256'])
        receipts = [json.loads((self.recorder.folder/record['path']).read_text()) for record in result['rpc_receipts']]
        collects = [row for row in receipts if row['method'] == 'demoncat_fault_rpc'
                    and row['args'][0] == 'advance' and row['args'][1].get('settings') == {'collect':True}]
        self.assertEqual(len(collects), 8)
        for request in result['requests']:
            original = request['original_observations']
            path = Path(original['path'])
            self.assertEqual(probe.file_sha(path), original['sha256'])
            raw = json.loads(path.read_text())
            self.assertEqual(len(raw['inventory']['rows'][-1]['candidates']), 2)
        self.assertEqual(probe.file_sha(self.out/'host_receipt.json'), result['host_receipt']['sha256'])
        host = json.loads((self.out/'host_receipt.json').read_text())
        self.assertEqual(host['status'], 'DETACHED')

    def test_actual_model_failure_keeps_receipts_and_never_retries_fault_or_selects_again(self):
        self.driver.fail_index = 2
        result = self.drive()
        self.assertEqual(result['status'], 'REFUSED_OR_FAILED')
        self.assertEqual(len(self.driver.executed), 3)
        self.assertEqual(self.driver.aborted, [self.driver.enqueued[-1]])
        self.assertFalse(any(mode == 'fault' for _, mode, _ in self.case.native.executes))
        self.assertIsNotNone(self.case.provider.backend.failure)
        self.assertFalse(result['ownership_retained'])
        self.assertTrue(result['rpc_receipts'])

    def test_statistics_refuses_before_model_or_bootstrap_and_source_index_is_concrete(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            probe.main(['--mechanism','26.fia_destination','--manifest-sha256','a'*64,
                        '--source-root','/source','--statistics'])
        self.assertEqual(raised.exception.code, 2)
        self.assertTrue(all((probe.ROOT/path).is_file() for path in probe.SOURCES))


class Core26CaseTests(unittest.TestCase):
    def prepare_case(self, stage, budget):
        from test_writer26_group_build import build
        from test_writer26_fia_typed import ABI
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source, native = folder/'source', folder/'native'
            native.mkdir()
            entry = source/'experiments/c2c3_closure_20261117/providers26_worker_core_probe.py'
            entry.parent.mkdir(parents=True)
            entry.write_bytes(Path(probe.__file__).read_bytes())
            document = dict(schema='demoncat.writer26.group_deployment.v2', actual_npu_calls=0,
                legacy_writers_absent=True, actual_image_id='sha256:'+'b'*64, files={},
                fia_typed_profile=build.typed_profile(dict(fia_typed_abi=ABI),
                    dict(status='PASS_CPU_TYPED_FIA_RUNTIME', actual_npu_calls=0, cases=18)))
            (native/'build.json').write_text(json.dumps(document))
            output = folder/'cases.json'
            argv = ['cases','--installed','/installed','--native','/native','--probe',
                '/source/experiments/c2c3_closure_20261117/providers26_worker_core_probe.py',
                '--source-commit','37bbeb4f','--source','/source','--worker-core','--max-targets',str(budget),
                '--stages',stage,'--mechanisms','26.fia_destination','--out',str(output)]
            def path(value):
                text = str(value)
                return native if text == '/native' else source if text == '/source' else (
                    source/text.removeprefix('/source/') if text.startswith('/source/') else Path(value))
            with patch.object(sys, 'argv', argv), patch.object(cases, 'Path', side_effect=path):
                cases.main()
            row, = json.loads(output.read_text())
            return row

    def test_new_case_selects_public_entry_preserves_typed_profile_denominator_and_logs(self):
        row = self.prepare_case('c1_single_effects', 1)
        self.assertIn('providers26_worker_core_probe.py', row['command'])
        self.assertIn('worker-core26.stderr.log', row['command'])
        self.assertIn('--source-root /source', row['command'])
        self.assertIn('WorkerFaultClient', row['review']['public_entry'])
        self.assertFalse(row['review']['previous_direct_entry_qualification_inherited'])
        self.assertEqual(row['review']['resource_budget']['planned_single_site_targets'], 1)
        self.assertEqual(row['review']['resource_budget']['independent_events'], 32768)
        self.assertFalse(row['review']['resource_budget']['max_targets_selects_subset'])
        self.assertIn('Two complete unmodified', row['review']['fia_stock_gate'])
        self.assertIn('smollm2', row['review']['model'])
        self.assertIn(['/source','/source','ro'], row['mounts'])

    def test_complete_set_case_uses_public_entry_full_budget_and_no_c3_or_subset(self):
        row = self.prepare_case('c2_once', 128)
        self.assertIn('--stage c2_once --max-targets 128', row['command'])
        self.assertIn('public Core all_eligible once C2 attempt; not C3', row['review']['provider'])
        self.assertEqual(row['review']['execution_scope'], 'same-request cumulative all stable eligible targets')
        self.assertIsNone(row['review']['resource_budget']['planned_single_site_targets'])
        self.assertEqual(row['review']['resource_budget']['independent_events'], 32768)
        self.assertFalse(row['review']['resource_budget']['max_targets_selects_subset'])
        self.assertIn('blocks the whole group; no subset', row['review']['c2_dynamic_eligibility'])
        self.assertIn('C3 not requested', row['review']['expected_hits'])
        self.assertIn('Two complete unmodified', row['review']['fia_stock_gate'])


if __name__ == '__main__':
    unittest.main()
