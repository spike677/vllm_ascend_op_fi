"""CPU probe orchestration with the real Host/Runtime/torch RoPE contract.

The fake scheduler supplies actual enqueue state and executes the fixture model
only from wait_for_completion. These tests grant no serving qualification.
"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import test_core_rope22 as core_fixture
from experiments.c2c3_closure_20261117 import natural_rope_worker_core_probe as probe
from experiments.c2c3_closure_20261117 import prepare_rope22_worker_core as prepare


class FixtureServingDriver:
    def __init__(self, case):
        self.case = case
        self.llm_engine = self
        self.engine_core = type('InprocClient', (), {'__module__': 'vllm.v1.engine.core_client'})()
        self.output_processor = SimpleNamespace(request_states={})
        self.enqueued = []
        self.executed = []
        self.aborted = []
        self.resets = 0
        self.fail_index = None
        self.fail_logits = False

    def has_unfinished_requests(self):
        return bool(self.output_processor.request_states)

    def reset_prefix_cache(self, reset_running_requests):
        self.case.assertFalse(reset_running_requests)
        self.case.assertFalse(self.has_unfinished_requests())
        self.resets += 1
        return True

    def enqueue(self, prompts, params, use_tqdm):
        self.case.assertFalse(use_tqdm)
        self.case.assertEqual(prompts, [{'prompt_token_ids': [1, 2, 3]}])
        self.case.assertFalse(self.has_unfinished_requests())
        request_id = 'actual-enqueued-' + str(len(self.enqueued))
        self.output_processor.request_states[request_id] = SimpleNamespace(
            request_id=request_id, external_req_id='public-' + request_id)
        self.enqueued.append(request_id)
        return [request_id]

    def wait_for_completion(self, use_tqdm):
        self.case.assertFalse(use_tqdm)
        request_id, = self.output_processor.request_states
        state = self.output_processor.request_states[request_id]
        self.executed.append(request_id)
        model = self.case.model
        fail = len(self.enqueued) - 1 == self.fail_index
        model.fail_forward, model.fail_logits = fail and not self.fail_logits, fail and self.fail_logits
        try:
            hidden = model(input_ids=torch.tensor([1, 2, 3]), positions=torch.arange(3))
            logits = model.logits(hidden)
        finally:
            model.fail_forward = model.fail_logits = False
        del self.output_processor.request_states[request_id]
        return [SimpleNamespace(request_id=state.external_req_id,
                                outputs=[SimpleNamespace(token_ids=[int(logits.argmax())])])]

    def abort_request(self, request_ids, internal):
        self.case.assertTrue(internal)
        for request_id in request_ids:
            self.case.assertIn(request_id, self.enqueued)
            self.aborted.append(request_id)
            self.output_processor.request_states.pop(request_id, None)


class WorkerCoreRoPEProbeTests(unittest.TestCase):
    def setUp(self):
        self.case = core_fixture.CoreRoPE22Tests()
        self.case.setUp()
        self.addCleanup(self.case.tearDown)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.driver = FixtureServingDriver(self.case)
        # Healthy warmup is outside any managed group, as in the real probe.
        hidden = self.case.model(input_ids=torch.tensor([1, 2, 3]), positions=torch.arange(3))
        healthy = self.case.model.logits(hidden)
        self.report = dict(status='STARTED', mechanism='22.wrong_row.fused_positions',
            standard_vllm_worker=False, public_qualification_granted=False,
            raw_device_launch_payload_observed=False,
            healthy_before=[int(healthy.argmax())], setup={'environment': {'cpu_contract': True}})

    def run_probe(self):
        return probe.drive(self.driver, self.case.client, prompt={'prompt_token_ids': [1, 2, 3]},
                           params=object(), out=self.out, report=self.report)

    def read_receipts(self):
        results = []
        for index, record in enumerate(self.report['worker_core']['receipts']):
            path = self.out / record['path']
            self.assertEqual(path.name[:3], str(index).zfill(3))
            self.assertEqual(probe.sha_file(path), record['sha256'])
            results.append(json.loads(path.read_text(encoding='utf-8')))
        host = self.report['worker_core']['final_host_receipt']
        self.assertEqual(probe.sha_file(self.out / host['path']), host['sha256'])
        snapshot = json.loads((self.out / host['path']).read_text(encoding='utf-8'))
        self.assertEqual(results, snapshot['history'])
        self.assertEqual(json.loads((self.out / 'result.json').read_text(encoding='utf-8')), self.report)
        return results

    def test_ten_real_enqueues_full_rows_and_original_proof_receipts(self):
        result = self.run_probe()
        self.assertEqual(result['status'], probe.SUCCESS, result.get('error'))
        self.assertEqual(self.driver.executed, self.driver.enqueued)
        self.assertEqual(len(self.driver.enqueued), 10)
        self.assertEqual(self.driver.resets, 10)
        self.assertEqual(self.driver.aborted, [])
        self.assertEqual((self.case.model.forward_calls, self.case.model.logits_calls), (11, 11))
        self.assertEqual([row['phase'] for row in result['requests']], list(probe.PHASES[:9]))
        for index, row in enumerate(result['requests']):
            self.assertEqual(row['mapping']['internal_request_id'], self.driver.enqueued[index])
            self.assertEqual(row['worker']['request_id'], self.driver.enqueued[index])
            trace = row['worker']['independent']['rope_trace']
            self.assertEqual(len(trace['readers']), 3)
            self.assertEqual(len(trace['outer_operators']), 3)
            self.assertTrue(trace['complete'])
            self.assertTrue(all(call['original_calls'] == 1 for call in trace['outer_operators']))
        self.assertEqual(result['cleanup']['requests'], self.driver.enqueued[:9])
        self.assertEqual(result['worker_core']['final_cleanup']['requests'], self.driver.enqueued)
        self.assertEqual(result['health_request']['phase'], 'cleanup_healthy')
        self.assertTrue(result['health_request']['worker']['cleanup_before_request'])
        self.assertEqual(result['healthy_before'], result['healthy_after'])
        self.assertEqual((result['c2']['hits'], result['c3']['hits']), (3, 6))
        self.assertEqual((result['c2_open']['targets'], result['c3_open']['targets']), (3, 3))
        issued, = result['c2']['live_c2_proof']['issued']
        used = result['c3']['live_c2_proof']['consumed']
        self.assertEqual(issued['proof_sha256'], used['proof_sha256'])
        self.assertEqual(issued['object_binding'], used['object_binding'])
        self.assertEqual(used['proof']['owner'], result['c2_open']['owner'])
        self.assertEqual(used['proof']['generation'], result['c2_open']['generation'])
        self.assertEqual(used['consumer']['generation'], result['c3_open']['generation'])
        self.assertEqual(used['proof']['scope']['evidence_kind'], 'cpu_contract')
        self.assertFalse(issued['serializable_authority'])
        self.assertFalse(result['c2']['real_worker_evidence'])
        self.assertFalse(result['worker_core']['qualification_granted'])
        self.assertFalse(result['worker_core']['ownership_retained'])
        self.assertEqual(result['statistics_status'], 'SEPARATE_RAW_BF_CAPTURE_REQUIRED')
        receipts = self.read_receipts()
        self.assertEqual([item['action'] for item in receipts].count('advance'), 10)
        self.assertEqual([item['action'] for item in receipts].count('query'), 10)
        self.assertEqual(receipts[-1]['action'], 'detach')
        self.case.assert_clean()

    def test_failed_real_fault_request_is_preserved_and_only_off_is_enqueued(self):
        self.driver.fail_index = 4
        result = self.run_probe()
        self.assertEqual(result['status'], 'REFUSED_OR_FAILED')
        self.assertIn('real root failure', result['error'])
        self.assertEqual(len(self.driver.enqueued), 6)
        self.assertEqual(self.driver.executed, self.driver.enqueued)
        self.assertEqual(self.driver.aborted, [self.driver.enqueued[4]])
        self.assertEqual([item['phase'] for item in result['requests']], list(probe.PHASES[:4]))
        self.assertEqual([item['phase'] for item in result['recovery_requests']], ['recovery_off'])
        self.assertNotIn('c2', result)
        self.assertNotIn('c3', result)
        self.assertFalse(result['worker_core']['ownership_retained'], result.get('cleanup_error'))
        self.read_receipts()
        self.case.assert_clean()

    def test_lost_advance_reply_never_retries_fault_or_hides_failed_receipt(self):
        def lose_reply(method, args, replies):
            if method == 'demoncat_fault_rpc' and args[0] == 'advance' and len(self.driver.enqueued) == 7:
                self.case.executor.after = None
                raise TimeoutError('lost reply after scheduling original C3')
            return replies
        self.case.executor.after = lose_reply
        result = self.run_probe()
        self.assertEqual(result['status'], 'REFUSED_OR_FAILED')
        self.assertIn('advance:RPC_UNCERTAIN', result['error'])
        self.assertEqual(len(self.driver.enqueued), 8)
        self.assertNotIn(self.driver.enqueued[6], self.driver.executed)
        self.assertEqual(self.driver.aborted, [self.driver.enqueued[6]])
        self.assertEqual([item['phase'] for item in result['requests']], list(probe.PHASES[:6]))
        self.assertEqual([item['phase'] for item in result['recovery_requests']], ['recovery_off'])
        self.assertTrue(result['c2']['live_c2_proof']['issued'])
        self.assertNotIn('c3', result)
        self.assertFalse(result['worker_core']['ownership_retained'], result.get('cleanup_error'))
        receipts = self.read_receipts()
        self.assertEqual(sum(item['response']['status'] == 'RPC_UNCERTAIN' for item in receipts), 1)
        self.case.assert_clean()

    def test_failed_cleanup_preserves_owner_and_full_failure_evidence(self):
        self.driver.fail_index = 4
        original = self.case.executor.collective_rpc

        def obstruct_cleanup(method, timeout=None, args=()):
            if method == 'demoncat_fault_rpc' and args[0] == 'recover':
                runtime = self.case.worker._demoncat_worker_fault_runtime
                provider = runtime._leases['22.wrong_row'].provider
                # Actual owned cleanup failure; keep it failing for this probe.
                provider._owned_abort = lambda: (_ for _ in ()).throw(RuntimeError('owned cleanup blocked'))
            return original(method, timeout=timeout, args=args)

        with patch.object(self.case.executor, 'collective_rpc', side_effect=obstruct_cleanup):
            result = self.run_probe()
        self.assertEqual(result['status'], 'REFUSED_OR_FAILED')
        self.assertTrue(result['worker_core']['ownership_retained'])
        self.assertEqual(len(self.driver.enqueued), 5)
        self.assertNotIn('recovery_requests', result)
        self.assertNotIn('c2', result)
        receipts = self.read_receipts()
        self.assertTrue(any(item['response']['status'] == 'PROVIDER_ACTION_FAILED' for item in receipts))
        provider = self.case.worker._demoncat_worker_fault_runtime._leases['22.wrong_row'].provider
        del provider._owned_abort
        # Test teardown explicitly recovers the same owner after the obstacle is gone.
        self.case.ok(self.case.client.recover())
        self.case.ok(self.case.client.advance('teardown-owned-off'))
        hidden = self.case.model(input_ids=torch.tensor([1, 2, 3]), positions=torch.arange(3))
        self.case.model.logits(hidden)
        self.case.ok(self.case.client.finish())
        self.case.ok(self.case.client.unregister())
        self.case.assert_clean()


class WorkerCoreRoPECaseTests(unittest.TestCase):
    def test_bounded_standard_worker_case_retains_raw_artifacts_and_fixed_weights(self):
        case, = prepare.core_cases('/home/reviewed/source49', card=5)
        self.assertEqual(case['mounts'], [['/home/reviewed/source49', '/source', 'ro'],
            ['/home/gyl/models/Qwen3-0.6B', '/model', 'ro']])
        self.assertEqual(case['review']['physical_cards'], [5])
        self.assertEqual(case['env']['VLLM_ENABLE_V1_MULTIPROCESSING'], '0')
        self.assertIn('natural_rope_worker_core_probe.py', case['command'])
        self.assertIn('--out /evidence/probe', case['command'])
        self.assertIn('> /evidence/worker-core22.stdout.log', case['command'])
        self.assertIn('2> /evidence/worker-core22.stderr.log', case['command'])
        self.assertIn('10 actual enqueue requests', case['review']['recovery_plan'])
        self.assertIn('E5 is separate', case['review']['recovery_plan'])
        self.assertIn('NOT_INHERITED_FROM_SOURCE46', case['review']['acceptance'])
        self.assertNotIn('layer', case['command'])
        self.assertNotIn('rank', case['command'])
        self.assertIn('/evidence/probe/worker_core_rpc', case['review']['raw_evidence'])

    def test_case_generator_rejects_unreviewed_source_or_card_and_preserves_output(self):
        for source, card in [('relative', 2), ('/review/../source', 2), ('/review\ncommand', 2), ('/review', 0)]:
            with self.subTest(source=source, card=card), self.assertRaises(ValueError):
                prepare.core_cases(source, card)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'case.json'
            args = ['--source', '/review/source49', '--out', str(path)]
            prepare.main(args)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                prepare.main(args)
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
