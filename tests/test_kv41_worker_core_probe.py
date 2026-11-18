"""Public Core41 driver with real Host/Runtime/model hooks and explicit CPU ABI."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import test_core_kv41 as fixture
from experiments.c2c3_postwrite_20261117 import natural41_worker_core_probe as probe
from experiments.c2c3_postwrite_20261117 import prepare41_worker_core as prepare


class ServingDriver:
    def __init__(self, case):
        self.case, self.llm_engine = case, self
        self.engine_core = type('InprocClient', (), {'__module__': 'vllm.v1.engine.core_client'})()
        self.output_processor = NS(request_states={})
        self.enqueued, self.executed, self.aborted = [], [], []
        self.resets, self.fail_index = 0, None

    def has_unfinished_requests(self):
        return bool(self.output_processor.request_states)

    def reset_prefix_cache(self, reset_running_requests):
        self.case.assertFalse(reset_running_requests)
        self.case.assertFalse(self.has_unfinished_requests())
        self.resets += 1
        return True

    def enqueue(self, prompts, params, use_tqdm):
        self.case.assertEqual(prompts, [{'prompt_token_ids': self.case.prompt}])
        self.case.assertFalse(use_tqdm or self.has_unfinished_requests())
        request_id = 'actual-enqueued-' + str(len(self.enqueued))
        self.output_processor.request_states[request_id] = NS(request_id=request_id, external_req_id='public-'+request_id)
        self.enqueued.append(request_id)
        return [request_id]

    def wait_for_completion(self, use_tqdm):
        self.case.assertFalse(use_tqdm)
        request_id, = self.output_processor.request_states
        state = self.output_processor.request_states[request_id]
        self.executed.append(request_id)
        self.case.model.fail_decode = len(self.enqueued)-1 == self.fail_index
        try:
            value = self.case.actual_request(request_id)
        finally:
            self.case.model.fail_decode = False
        del self.output_processor.request_states[request_id]
        return [NS(request_id=state.external_req_id, outputs=[NS(token_ids=[int(value.argmax())], text='CPU fixture')])]

    def abort_request(self, ids, internal):
        self.case.assertTrue(internal)
        for request_id in ids:
            self.aborted.append(request_id)
            self.output_processor.request_states.pop(request_id, None)


class CoreKV41ProbeTests(unittest.TestCase):
    def setUp(self):
        self.case = fixture.CoreKV41Tests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out, self.driver = Path(self.temp.name), ServingDriver(self.case)
        value = self.case.actual_request('ordinary-warmup')
        self.report = dict(status='STARTED', mechanism='41.tail_column', public_qualification_granted=False,
            healthy_before=dict(tokens=[int(value.argmax())], text='CPU fixture'))

    def drive(self):
        return probe.drive(self.driver, self.case.client, prompt={'prompt_token_ids': self.case.prompt},
            params=object(), out=self.out, report=self.report)

    def receipts(self):
        rows = []
        for index, record in enumerate(self.report['worker_core']['receipts']):
            path = self.out / record['path']
            self.assertEqual(path.name[:3], str(index).zfill(3))
            self.assertEqual(probe.file_sha(path), record['sha256'])
            rows.append(json.loads(path.read_text(encoding='utf-8')))
        receipt = self.report['worker_core']['final_host_receipt']
        self.assertEqual(probe.file_sha(self.out / receipt['path']), receipt['sha256'])
        final = json.loads((self.out / receipt['path']).read_text(encoding='utf-8'))
        self.assertEqual(rows, final['history'])
        self.assertEqual(json.loads((self.out/'result.json').read_text(encoding='utf-8')), self.report)
        return rows

    def test_complete_original_protocol_full_raw_rows_live_proof_and_post_unregister_enqueue(self):
        result = self.drive()
        self.assertEqual(result['status'], probe.SUCCESS, result.get('error'))
        self.assertEqual(self.driver.executed, self.driver.enqueued)
        self.assertEqual(len(self.driver.enqueued), 18)
        self.assertEqual(self.driver.resets, 18)
        self.assertEqual([row['phase'] for row in result['requests']], list(probe.PHASES))
        self.assertEqual((self.case.model.forward_calls, self.case.model.decode_calls), (38, 19))
        for row in result['requests']:
            if row['phase'].endswith('health'):
                self.assertTrue(row['worker']['model_output_sha256'])
            else:
                actual = row['worker']['independent_inventory']
                self.assertEqual(len(actual['calls']), 2)
                self.assertEqual(len(actual['writers']), 2)
                self.assertTrue(all(call['original_calls'] == 1 for call in actual['calls']))
        self.assertEqual((result['c2']['hits'], result['c3']['hits']), (2, 4))
        issued, = result['c2']['live_c2_proof']['issued']
        consumed = result['c3']['live_c2_proof']['consumed']
        self.assertEqual(issued['proof_sha256'], consumed['proof_sha256'])
        self.assertEqual(issued['object_binding'], consumed['object_binding'])
        self.assertEqual(issued['proof']['scope']['evidence_kind'], 'cpu_contract')
        self.assertFalse(issued['serializable_authority'])
        self.assertFalse(result['c2']['real_worker_evidence'])
        self.assertEqual(len(result['worker_core']['serving']['actual_requests']), 17)
        self.assertTrue(result['post_unregister_health']['after_public_unregister'])
        self.assertFalse(result['worker_core']['ownership_retained'])
        self.assertFalse(result['worker_core']['qualification_granted'])
        rows = self.receipts()
        self.assertEqual(sum(row['action'] == 'advance' for row in rows), 17)
        self.assertEqual(sum(row['action'] == 'query' for row in rows), 17)
        self.assertEqual(rows[-1]['action'], 'detach')
        self.assertEqual(result['statistics_status'], 'SEPARATE_RAW_BF_CAPTURE_REQUIRED')

    def test_actual_group_fault_failure_keeps_raw_failure_and_only_off_health_may_follow(self):
        self.driver.fail_index = 8
        result = self.drive()
        self.assertEqual(result['status'], 'REFUSED_OR_FAILED')
        self.assertIn('ACTUAL_CPU_DECODE_FAILED', result['error'])
        self.assertEqual([row['phase'] for row in result['recovery_requests']], ['recovery_group_off', 'recovery_health'])
        self.assertEqual(self.driver.aborted, [self.driver.enqueued[8]])
        self.assertEqual(len(self.driver.enqueued), 12)
        self.assertNotIn('c2', result)
        self.assertNotIn('c3', result)
        self.assertFalse(result['worker_core']['ownership_retained'], result.get('cleanup_error'))
        self.assertFalse(result['failed_attempt_recovery']['contract_satisfied'])
        self.assertTrue(result['failed_attempt_recovery']['recovery']['complete'])
        self.assertTrue(result['post_unregister_health']['after_public_unregister'])
        self.receipts()

    def test_uncertain_fault_advance_is_not_reexecuted_and_its_original_receipt_survives(self):
        def lose(method, args, replies):
            if method == 'demoncat_fault_rpc' and args[0] == 'advance' and len(self.driver.enqueued) == 9:
                self.case.executor.after = None
                raise TimeoutError('lost first C2 fault advance response')
            return replies
        self.case.executor.after = lose
        result = self.drive()
        self.assertEqual(result['status'], 'REFUSED_OR_FAILED')
        self.assertIn('advance:RPC_UNCERTAIN', result['error'])
        self.assertNotIn(self.driver.enqueued[8], self.driver.executed)
        self.assertEqual(self.driver.aborted, [self.driver.enqueued[8]])
        self.assertEqual(len(self.driver.enqueued), 12)
        self.assertFalse(result['worker_core']['ownership_retained'], result.get('cleanup_error'))
        self.assertEqual([row['phase'] for row in result['recovery_requests']], ['recovery_group_off', 'recovery_health'])
        self.assertNotIn('c2', result)
        rows = self.receipts()
        self.assertEqual(sum(row['response']['status'] == 'RPC_UNCERTAIN' for row in rows), 1)

    def test_failed_owned_cleanup_keeps_runtime_owner_and_never_claims_post_unregister_health(self):
        self.driver.fail_index = 8
        with patch.object(fixture.postwrite41.Postwrite41WorkerProvider, 'close', side_effect=RuntimeError('CPU_OWNED_CLOSE_FAILED')):
            result = self.drive()
        self.assertEqual(result['status'], 'REFUSED_OR_FAILED')
        self.assertTrue(result['worker_core']['ownership_retained'])
        self.assertIn('cleanup_error', result)
        self.assertNotIn('post_unregister_health', result)
        self.assertNotIn('c2', result)
        self.receipts()
        self.case.ok(self.case.client.recover())
        self.case.ok(self.case.client.advance('explicit-after-repair-health'))
        self.case.actual_request('explicit-after-repair-health')
        self.case.ok(self.case.client.finish())
        self.case.ok(self.case.client.unregister())


class CoreKV41CaseTests(unittest.TestCase):
    def test_one_original_workload_case_requires_actual_native_receipt_and_keeps_evidence_local(self):
        row, = prepare.core_cases(source='/reviewed/source', native='/reviewed/native', sha256='a'*64,
            build_fingerprint='b'*64, card=5)
        self.assertEqual(row['mounts'], [['/reviewed/source', '/source', 'ro'],
            ['/reviewed/native', '/native', 'ro'], ['/home/gyl/models/Qwen3-0.6B', '/model', 'ro']])
        self.assertEqual(row['timeout'], 1800)
        self.assertEqual(row['review']['physical_cards'], [5])
        self.assertNotIn('--expected-targets', row['command'])
        self.assertIn('--sha256 '+'a'*64, row['command'])
        self.assertIn('/evidence/worker-core41.stderr.log', row['command'])
        self.assertIn('E5 is separate', row['review']['recovery_plan'])
        self.assertIn('NO_INHERITED_CORE41', row['review']['acceptance'])
        self.assertTrue(all((probe.ROOT/name).is_file() for name in probe.SOURCES))
        self.assertIn('fault_injection/coverage_session.py', probe.SOURCES)
        self.assertIn('src/demoncat/runtime/vllm_ascend/worker_extension/providers/fault41.py', probe.SOURCES)
        with self.assertRaisesRegex(ValueError, 'ACTUAL_BUILD_IDENTITIES_REQUIRED'):
            prepare.core_cases(source='/s', native='/n', sha256='claimed', build_fingerprint='b'*64)


if __name__ == '__main__':
    unittest.main()
