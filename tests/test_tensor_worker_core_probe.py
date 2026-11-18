"""Public probe against real Host/Runtime groups and an explicit CPU scheduler."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import test_core_tensor_serving as core
from experiments.c2c3_closure_20261117 import natural_tensor_worker_core_probe as probe
from experiments.c2c3_closure_20261117 import prepare_tensor_worker_core as prepare


class Driver:
    def __init__(self, case):
        self.case = case
        self.llm_engine = self
        self.engine_core = type('InprocClient', (), {'__module__': 'vllm.v1.engine.core_client'})()
        self.output_processor = SimpleNamespace(request_states={})
        self.enqueued, self.executed, self.aborted = [], [], []
        self.fail_index = None
    def has_unfinished_requests(self):
        return bool(self.output_processor.request_states)
    def reset_prefix_cache(self, reset_running_requests):
        assert not reset_running_requests and not self.has_unfinished_requests()
        return True
    def enqueue(self, prompts, params, use_tqdm):
        assert prompts == [{'prompt_token_ids': [1, 2, 3]}] and not use_tqdm and not self.has_unfinished_requests()
        request_id = 'scheduled-' + str(len(self.enqueued))
        self.enqueued.append(request_id)
        self.output_processor.request_states[request_id] = SimpleNamespace(
            request_id=request_id, external_req_id='external-' + request_id)
        return [request_id]
    def wait_for_completion(self, use_tqdm):
        assert not use_tqdm
        request_id, = self.output_processor.request_states
        state = self.output_processor.request_states[request_id]
        self.executed.append(request_id)
        self.case.model.fail_logits = len(self.enqueued)-1 == self.fail_index
        try:
            logits = self.case.actual(request_id)
        finally:
            self.case.model.fail_logits = False
        del self.output_processor.request_states[request_id]
        return [SimpleNamespace(request_id=state.external_req_id,
                                outputs=[SimpleNamespace(token_ids=[int(logits.argmax())])])]
    def abort_request(self, request_ids, internal):
        assert internal
        for request_id in request_ids:
            assert request_id in self.enqueued
            self.aborted.append(request_id)
            self.output_processor.request_states.pop(request_id, None)


class TensorPublicProbeTests(unittest.TestCase):
    fixture = core.CoreTensorServingTests
    def setUp(self):
        self.case = self.fixture()
        self.case.setUp()
        self.addCleanup(self.case.tearDown)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        self.driver = Driver(self.case)
        healthy = self.case.actual('unmanaged-warmup')
        self.report = dict(status='STARTED', mechanism=self.case.mechanism, standard_vllm_worker=False,
            public_qualification_granted=False, historical_scope_inherited=False,
            healthy_before=[int(healthy.argmax())])

    def run_probe(self):
        return probe.drive(self.driver, self.case.client, prompt={'prompt_token_ids': [1, 2, 3]},
                           params=object(), out=self.out, report=self.report)

    def receipts(self):
        rows = []
        for record in self.report['worker_core']['receipts']:
            path = self.out/record['path']
            self.assertEqual(probe.sha_file(path), record['sha256'])
            rows.append(json.loads(path.read_text(encoding='utf-8')))
        self.assertEqual(rows, self.case.client.snapshot()['history'])
        self.assertEqual(json.loads((self.out/'result.json').read_text()), self.report)
        return rows

    def test_ten_actual_requests_and_explicit_collection_with_lossless_and_live_proof(self):
        result = self.run_probe()
        self.assertEqual(result['status'], probe.SUCCESS, result.get('error'))
        self.assertEqual(self.driver.enqueued, self.driver.executed)
        self.assertEqual(len(self.driver.enqueued), 10)
        self.assertEqual(self.driver.aborted, [])
        self.assertEqual((result['c2']['hits'], result['c3']['hits']), (2, 4))
        self.assertEqual(result['healthy_before'], result['healthy_after'])
        self.assertFalse(result['worker_core']['ownership_retained'])
        self.assertEqual((self.case.model.forward_calls, self.case.model.logits_calls), (11, 11))
        rows = self.receipts()
        collect = [row for row in rows if row['action'] == 'advance'
                   and row['response'].get('provider_state', {}).get('serving', {}).get('pending_request') is None]
        self.assertEqual(len(collect), 10)
        self.case.assert_clean()

    def test_failed_fault_is_preserved_while_one_real_off_and_health_recover(self):
        self.driver.fail_index = 4
        result = self.run_probe()
        self.assertEqual(result['status'], 'REFUSED_OR_FAILED')
        self.assertEqual(len(self.driver.enqueued), 7)
        self.assertEqual(self.driver.aborted, ['scheduled-4'])
        self.assertEqual(result['recovery_requests'][0]['phase'], 'recovery_off')
        self.assertEqual(result['health_request']['phase'], 'recovery_health')
        self.assertNotIn('c3', result)
        self.assertFalse(result['worker_core']['ownership_retained'], result.get('cleanup_error'))
        self.receipts()
        self.case.assert_clean()

    def test_two_prepared_cases_are_bounded_and_do_not_inherit_old_counts(self):
        rows = prepare.core_cases('/home/frozen-public-tensor-source', card=5)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row['review']['physical_cards'], [5])
            self.assertEqual(row['review']['resource_budget']['timeout_seconds'], 1800)
            self.assertTrue(all(mount[2] == 'ro' for mount in row['mounts']))
            self.assertIn('NOT_INHERITED', row['review']['acceptance'])
            self.assertIn('natural_tensor_worker_core_probe.py', row['command'])
        self.assertIn('18.api_half', rows[0]['command'])
        self.assertIn('8.restore_offset', rows[1]['command'])
        with self.assertRaises(ValueError):
            prepare.core_cases('/home/../unreviewed')


class MoEPublicProbeTests(TensorPublicProbeTests):
    fixture = core.CoreMoEServingTests


if __name__ == '__main__':
    unittest.main()
