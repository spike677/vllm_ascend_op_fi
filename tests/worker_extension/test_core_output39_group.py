"""Actual Host/Runtime and original CPU projections; no NPU qualification."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

import test_core_output39 as c1
from demoncat.runtime.vllm_ascend.worker_extension import serving39_output
from fault_injection.deployments.postwrite import MAX_POSTWRITE_SPAN_BYTES
from fault_injection.core import UnsupportedCall


class ParallelProjections(c1.ActualProjectionModel):
    def __init__(self):
        super().__init__()
        self.other = c1.Projection()

    def forward(self, input_ids, positions, inputs_embeds=None, intermediate_tensors=None):
        self.forward_calls += 1
        hidden = c1.life.CpuModel.forward(self, input_ids, positions, inputs_embeds, intermediate_tensors)
        return self.projection(hidden)+self.other(hidden)


class CoreOutput39GroupTests(unittest.TestCase):
    snapshot = c1.CoreOutput39Tests.snapshot
    root = c1.CoreOutput39Tests.root
    forward = c1.CoreOutput39Tests.forward
    dispose = c1.CoreOutput39Tests.dispose
    start = c1.CoreOutput39Tests.start
    enqueue = c1.CoreOutput39Tests.enqueue

    def setUp(self):
        self.enterContext(patch.object(c1, 'ActualProjectionModel', ParallelProjections))
        install = serving39_output.install_fault39_output_serving_binding
        self.enterContext(patch.object(serving39_output, 'install_fault39_output_serving_binding',
            lambda *a, **kw: install(*a, stage='c2_c3', max_live_leases=8, **kw)))
        c1.CoreOutput39Tests.setUp(self)
        self.transport.lease_capacity = 8
        self.transport.lease_byte_capacity = MAX_POSTWRITE_SPAN_BYTES

    def ok(self, reply):
        response = reply.get('response', {})
        self.assertEqual(reply['status'], 'ACTION_COMPLETED', {key: response.get(key) for key in
            ('action', 'provider_code', 'reason', 'message', 'error_type')})
        return response['provider_state']

    def request(self, index):
        output = self.enqueue(index)
        self.assertEqual(self.client.query()['status'], 'QUERY_OK')
        self.ok(self.client.collect_replay())
        return output

    def test_complete_c2_then_original_live_proof_c3_and_cleanup(self):
        self.start()
        with patch.object(self.worker, 'demoncat_worker_identity', side_effect=AssertionError('recursive RPC')):
            for index in range(40):
                if self.provider.complete:
                    break
                self.request(index)
        self.assertTrue(self.provider.complete)
        state = self.ok(self.client.finish())
        evidence = state['evidence']
        first, second = (evidence['group_summary'][key]['coverage'] for key in ('c2', 'c3'))
        self.assertTrue(first['contract_satisfied'], first['rejection_reasons'])
        self.assertTrue(second['contract_satisfied'], second['rejection_reasons'])
        self.assertEqual(first['planned_targets'], 2)
        self.assertEqual((first['hits'], second['hits']), (2, 4))
        # These describe different values and different algorithms. Compare
        # B/OFF/health within each boundary, not the two hashes to one another.
        for label in ('c2', 'c3'):
            original = self.provider.group_reports[label]
            health = original['actual_health']
            off = next(row for row in original['forward_artifacts'] if row['phase'] == 'off')
            request = next(row for row in self.provider.rows
                if row['actual_internal_request_id'] == health['actual_request_id'])
            self.assertEqual(health['output_sha256'], off['output_sha256'])
            self.assertNotEqual(health['output_sha256'], request['output_sha256'])
            self.assertEqual(health['output_digest']['value_boundary'], 'TUPLE_OF_ACTUAL_ROOT_HIDDEN_AND_COMPLETE_LOGITS')
            self.assertEqual(request['output_digest']['value_boundary'], 'ACTUAL_FINAL_LOGITS_TENSOR_ONLY')
            self.assertEqual(off['output_digest'], health['output_digest'])
        issued, = first['live_c2_proof']['issued']
        consumed = second['live_c2_proof']['consumed']
        self.assertEqual(issued['object_binding'], consumed['object_binding'])
        self.assertEqual(issued['proof_sha256'], consumed['proof_sha256'])
        self.assertLess(issued['proof']['generation'], consumed['consumer']['generation'])
        self.assertFalse(evidence['c2'] or evidence['c3'] or evidence['model_local_qualification_inherited'])
        self.assertEqual((state['owned_hooks'], state['leases'], state['inflight']), (0, 0, 0))
        self.assertTrue(all(lease.closed for lease in self.transport.opened))
        calls = self.model.forward_calls
        self.ok(self.client.unregister())
        self.assertEqual(self.client.detach()['status'], 'DETACHED')
        serving39_output.remove_fault39_output_serving_binding(self.worker)
        with torch.inference_mode():
            self.assertTrue(torch.isfinite(self.forward(self.provider.session)).all())
        self.assertEqual(self.model.forward_calls, calls+1)

    def until(self, phase):
        self.start()
        for index in range(40):
            if self.provider.phase == phase:
                return index
            self.request(index)
        self.fail('Phase not reached: '+phase)

    def recovery(self, index):
        state = self.ok(self.client.recover())
        while state['next_phase'].startswith('group_recovery_'):
            self.request(index)
            index += 1
            state = self.client.query()['response']['provider_state']
        self.ok(self.client.unregister())
        return state

    def test_copied_c2_proof_refused_before_any_c3_fault_and_explicit_off_health_cleanup(self):
        index = self.until('c3_lossless')
        group = self.provider.group
        original = group.c2_live_proof
        group.c2_live_proof = deepcopy(original)
        before = len(self.transport.opened)
        with self.assertRaisesRegex(UnsupportedCall, 'MATCHING_C2_PROOF_REQUIRED'):
            self.enqueue(index)
        self.assertTrue(self.client.snapshot()['ownership_retained'])
        self.assertEqual(len(self.transport.opened)-before, len(group.eligible))  # Only lossless.
        group.c2_live_proof = original
        self.recovery(index+1)

    def test_query_neither_collects_nor_runs_forward_and_native_readback_stays_current(self):
        self.start()
        self.enqueue(0)
        before = (self.model.forward_calls, len(self.transport.opened), self.transport.restores, len(self.provider.originals))
        with patch.object(self.provider, '_collect', side_effect=AssertionError('query collected')):
            self.assertEqual(self.client.query()['status'], 'QUERY_OK')
        self.assertEqual(before, (self.model.forward_calls, len(self.transport.opened),
            self.transport.restores, len(self.provider.originals)))
        self.ok(self.client.collect_replay())
        self.ok(self.client.unregister())

    def test_same_worker_group_cleanup_health_failure_keeps_owner_and_readback_then_retry(self):
        index = self.until('c2_health')
        self.enqueue(index)
        coverage = self.provider.group.session
        def fail(*args):
            raise RuntimeError('CPU_GROUP_HEALTH_READBACK_UNCERTAIN')
        with patch.object(coverage, 'recovery', replace(coverage.recovery, health=fail)):
            result = self.client.collect_replay()
        self.assertEqual(result['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(result['ownership_retained'] and self.provider.binding.leases)
        self.assertIsNone(self.provider.group.c2_live_proof)
        self.assertTrue(coverage.recovery_report()['held_leases'])
        self.ok(self.client.recover())
        self.assertIsNotNone(self.provider.group.c2_live_proof)
        self.assertEqual(self.model.forward_calls, index+1)
        self.assertIn('CPU_GROUP_HEALTH_READBACK_UNCERTAIN', str(self.provider.cleanup_failures))
        # No C3 F has begun; cleanup is safe after the completed C2 recovery.
        self.ok(self.client.unregister())

    def test_finite_original_gate_is_not_relaxed_for_second_c3_site(self):
        index = self.until('c3_fault')
        before = len(self.transport.opened)
        original = self.transport.validate
        def finite(tensor):
            original(tensor)
            if len(self.transport.opened) >= before+1:
                raise c1.PostwriteRefused('CPU_SECOND_C3_NONFINITE_ORIGINAL')
        with patch.object(self.transport, 'validate', finite):
            with self.assertRaises((RuntimeError, ValueError)):
                self.enqueue(index)
        self.assertTrue(self.client.snapshot()['ownership_retained'])
        self.assertFalse(self.provider.group_reports.get('c3'))
        self.recovery(index+1)

    def test_actual_foreign_ticket_refuses_without_model_or_native(self):
        self.start()
        ticket = self.client.snapshot()['ticket']
        for change in ({'owner':'foreign'}, {'generation':ticket['generation']+1}):
            result = self.worker.demoncat_fault_rpc('advance', dict(ticket, **change, settings={'request_id':'foreign'}))
            self.assertEqual(result['status'], 'REJECTED')
        self.assertEqual(self.model.forward_calls, 0)
        self.assertFalse(self.transport.opened)
        self.ok(self.client.unregister())

    def test_unobserved_executable_module_still_blocks_complete_discovery(self):
        self.model.unused = torch.nn.Linear(4, 4, dtype=torch.bfloat16).eval()
        self.start()
        for index in range(5):
            self.request(index)
        self.enqueue(5)
        failed = self.client.collect_replay()
        self.assertEqual(failed['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(failed['ownership_retained'])
        self.assertFalse(self.provider.group.discovery_diagnostics['candidates']['discovery_complete'])
        self.assertIn('POSTWRITE_INDEPENDENT_DISCOVERY_INCOMPLETE', str(self.provider.cleanup_failures))
        self.assertEqual(len(self.transport.opened), 2)  # Existing C1 only.
        self.ok(self.client.unregister())

    def test_group_calls_never_use_model_local_synchronous_driver(self):
        from fault_injection.deployments.postwrite_coverage import ProjectionCoverage
        from fault_injection.deployments.postwrite import PostwriteProvider
        with (patch.object(ProjectionCoverage, '_run_forward', side_effect=AssertionError('sync group forward')),
              patch.object(PostwriteProvider, '_replay', side_effect=AssertionError('model-local replay'))):
            self.start()
            for index in range(40):
                if self.provider.complete:
                    break
                self.request(index)
            self.ok(self.client.finish())
            self.ok(self.client.unregister())


if __name__ == '__main__':
    unittest.main()
