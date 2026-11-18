"""Real Host/Runtime/Torch callbacks; all Native effects are explicit CPU doubles."""
from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import torch

from demoncat.runtime.vllm_ascend.worker_extension import serving26_core as core
import test_core_writer26 as c1


class PersistedNativeProducerModel(c1.ActualWriters):
    """The Native tracker, unlike Python public strong refs, survives requests."""
    def forward(self, input_ids, positions, inputs_embeds=None, intermediate_tensors=None):
        self.root_calls += 1
        x = torch.ones(input_ids.numel(), self.head.in_features, dtype=torch.bfloat16)
        residual = torch.zeros_like(x)
        for writer in self.writers:
            x = writer(x)
            residual = self.producer(x, residual)
        return residual


class CoreWriter26CoverageTests(unittest.TestCase):
    dispose = c1.CoreWriter26Tests.dispose
    start = c1.CoreWriter26Tests.start
    enqueue = c1.CoreWriter26Tests.enqueue
    snapshot = c1.CoreWriter26Tests.snapshot
    root = c1.CoreWriter26Tests.root
    forward = c1.CoreWriter26Tests.forward

    def setUp(self):
        c1.CoreWriter26Tests.setUp(self)
        if os.name == 'nt':
            # The sandbox's temporary root is long; keep real artifact names.
            self.evidence_root = Path('\\\\?\\'+str(self.evidence_root.resolve()))
            self.addCleanup(shutil.rmtree, str(self.evidence_root.parent), ignore_errors=True)

    def ok(self, reply):
        response = reply.get('response', {})
        self.assertEqual(reply['status'], 'ACTION_COMPLETED', dict(
            action=response.get('action'), reason=response.get('reason'), message=response.get('message'),
            error=response.get('error'), error_type=response.get('error_type')))
        return response['provider_state']

    def make(self, *, fia=False, count=2, max_targets=128):
        install = core.install_writer26_core_binding
        def bind(*args, **kwargs):
            return install(*args, stage=core.COMPLETE_SET_STAGE, max_targets=max_targets, **kwargs)
        self.enterContext(patch.object(core, 'install_writer26_core_binding', bind))
        c1.CoreWriter26Tests.make(self, fia=fia, count=count)
        def seal(targets):
            self.assertFalse(self.native.sealed)
            self.assertIsNone(self.native.transaction)
            expected = [(mode, key) for mode in ('reference', 'lossless', 'off') for key in targets]
            self.assertCountEqual([(mode, key) for _, mode, key in self.native.executes], expected)
            self.native.sealed = True
        self.enterContext(patch.object(self.native, 'seal', seal))

    def complete(self, total, eligible):
        self.start()
        with patch.object(self.worker, 'demoncat_worker_identity', side_effect=AssertionError('recursive RPC')):
            for index in range(8):
                self.enqueue(index)
            state = self.ok(self.client.finish())
        report = self.provider.backend.report()
        coverage = report['coverage']
        self.assertTrue(coverage['contract_satisfied'])
        self.assertEqual((len(report['candidates']), len(self.provider.backend.targets)), (total, eligible))
        self.assertEqual((coverage['planned_targets'], coverage['hits']), (eligible, eligible))
        self.assertEqual(coverage['observed_calls'], total*2)
        self.assertEqual(len(coverage['live_c2_proof']['issued']), 1)
        self.assertIsNone(coverage['live_c2_proof']['consumed'])
        self.assertIsNotNone(self.provider.backend.proof)
        self.assertTrue(all(len(row['candidates']) == total for row in report['rows']))
        self.assertFalse(report['c2'] or report['c3'] or state['evidence']['public_core']['qualification_granted'])
        self.assertEqual((state['leases'], state['inflight'], state['serving']['owned_hooks']), (0, 0, 0))
        self.assertEqual(self.model.root_calls, 8)
        # Repeated cleanup does not issue another proof or overwrite an artifact.
        self.ok(self.client.disable())
        self.ok(self.client.unregister())
        self.assertEqual(self.client.detach()['status'], 'DETACHED')
        with torch.inference_mode():
            self.forward(self.provider.session)
        self.assertEqual(self.model.root_calls, 9)
        return report

    def test_actual_public_qmm_all_stable_targets_once_and_post_unregister_health(self):
        self.make(count=3)
        report = self.complete(3, 3)
        self.assertEqual(report['execution_scope'], 'SAME_REQUEST_CUMULATIVE_ALL_STABLE_ELIGIBLE_TARGETS')
        self.assertFalse(report['independent_per_target_replay'])
        self.assertEqual(len(self.native.executes), 15)

    def test_actual_public_typed_fia_all_29_keeps_full_30_and_flat_storage(self):
        self.make(fia=True, count=30)
        report = self.complete(30, 29)
        self.assertEqual(report['candidates'][0]['status'], 'UNSUPPORTED')
        self.assertEqual(report['event_capacity_plan']['additional_owned_execute_records'], 145)
        for outcome in report['outcomes']:
            original = outcome['fia_typed_consumer_snapshot']['before']['original']
            self.assertEqual((original['format'], original['storage_rank']), (47, 1))
            self.assertTrue(outcome['public_consumer_lifetime'])

    def test_query_never_collects_advances_or_runs_model(self):
        self.make()
        self.start()
        self.enqueue(0, collect=False)
        before = deepcopy(self.native.state())
        with patch.object(self.provider.session, 'collect', side_effect=AssertionError('query collected')):
            self.assertEqual(self.client.query()['status'], 'QUERY_OK')
        self.assertEqual(self.native.state(), before)
        self.assertEqual(self.model.root_calls, 1)
        self.ok(self.client.collect_replay())
        self.ok(self.client.unregister())

    def test_budget_rejects_whole_group_without_selecting_a_prefix(self):
        self.make(count=3, max_targets=2)
        self.start()
        self.enqueue(0)
        self.enqueue(1)
        self.assertEqual(len(self.provider.backend.candidate_inventory), 3)
        self.assertEqual(len(self.provider.backend.targets), 3)
        self.assertIn('EMPTY_OR_OVER_BUDGET_COMPLETE_RESOLVED_TARGET_SET', self.provider.backend.blockers)
        self.assertFalse(self.native.executes)
        self.assertEqual(self.provider.backend.phase, 'health')
        self.enqueue(2)
        self.assertEqual(self.client.finish()['status'], 'PROVIDER_ACTION_FAILED')
        self.ok(self.client.unregister())

    def test_real_two_healthy_allocator_flip_blocks_entire_group_keeps_all_calls(self):
        self.make(count=3)
        self.start()
        self.enqueue(0)
        original = self.provider.backend._allocation
        def allocation(call, row):
            if call.cache == 1001:
                row['proven_nonadjacent'] = [[(10, 20, 'active_allocated')], [(30, 40, 'active_allocated')]]
                raise RuntimeError('LIVE_ALLOCATOR_BLOCKS_PROVEN_NOT_ADJACENT')
            return original(call, row)
        with patch.object(self.provider.backend, '_allocation', allocation):
            self.enqueue(1)
        report = self.provider.backend.report()
        self.assertEqual(len(report['candidates']), 3)
        self.assertEqual(report['candidates'][1]['status'], 'UNVERIFIED')
        self.assertIn('HEALTHY_TARGET_LIFETIME_OR_OCCURRENCE_CHANGED', report['candidates'][1]['reasons'])
        self.assertIn('UNRESOLVED_COMPLETE_NATIVE_WRITER_DENOMINATOR', report['blockers'])
        self.assertFalse(self.native.executes or self.native.sealed)
        self.enqueue(2)
        self.ok(self.client.unregister())

    def stale_producer(self):
        self.enterContext(patch.object(c1, 'ActualWriters', PersistedNativeProducerModel))
        self.make(fia=True, count=3)
        # A real CPU-double Native producer before this fresh serving root.
        with torch.inference_mode():
            self.native.producer(torch.ones(13, 576, dtype=torch.bfloat16),
                                 torch.zeros(13, 576, dtype=torch.bfloat16))
        self.start()
        self.enqueue(0)
        self.enqueue(1)
        return self.provider.backend

    def test_absent_current_producer_with_stale_nonzero_native_tracker_is_proven_not_dropped(self):
        backend = self.stale_producer()
        self.assertEqual(len(backend.candidate_inventory), 3)
        first = backend.candidate_inventory[0]
        self.assertEqual(first['status'], 'UNSUPPORTED')
        self.assertEqual(first['reasons'], ['CURRENT_FRESH_FORWARD_HAS_NO_RESIDUAL_PRODUCER'])
        self.assertGreater(first['first_native']['producer'], 0)
        for row in backend.rows:
            from fault_injection.deployments.writer26_inventory import digest
            self.assertEqual(row['complete_inventory']['candidates_sha256'], digest(row['candidates']))
            proof = row['candidates'][0]['current_forward_residual_absence']
            self.assertEqual(proof['earlier_current_forward_producer_count'], 0)
            self.assertEqual(proof['next_actual_producer']['producer'], proof['inherited_producer']+1)
        for index in range(2, 8):
            self.enqueue(index)
        state = self.ok(self.client.finish())
        self.assertEqual(state['evidence']['coverage']['planned_targets'], 2)
        self.assertEqual(state['evidence']['coverage']['observed_calls'], 6)
        self.ok(self.client.unregister())

    def test_absence_exclusion_refuses_gaps_overflow_other_thread_and_current_producer(self):
        backend = self.stale_producer()
        row = backend.rows[1]
        candidate = row['candidates'][0]
        events = row['native_events']
        with patch.dict(row, native_events=events[1:]):
            self.assertEqual(backend._classify_candidate(candidate)[0], 'UNVERIFIED')
        with patch.dict(row['native_after'], overflow_events=1):
            self.assertEqual(backend._classify_candidate(candidate)[0], 'UNVERIFIED')
        with patch.dict(events[0], thread=events[0]['thread']+1):
            self.assertEqual(backend._classify_candidate(candidate)[0], 'UNVERIFIED')
        with patch.dict(candidate['callback'], original_continuations=2):
            self.assertEqual(backend._classify_candidate(candidate)[0], 'UNVERIFIED')
        # An actual producer before the writer prevents the absence proof,
        # even if later lineage checks were inconclusive.
        producer = next(item for item in events if item['kind_name'] == 'add_rms_exit')
        with patch.dict(candidate['native'], sequence=producer['sequence']+1):
            self.assertEqual(backend._classify_candidate(candidate)[0], 'UNVERIFIED')
        self.ok(self.client.unregister())

    def test_typed_budget_keeps_all_candidates_and_refuses_before_reference(self):
        self.make(fia=True, count=3)
        self.start()
        self.enqueue(0)
        typed = self.provider.backend.fia_typed
        original = typed.state
        def state(*args, **kwargs):
            value = original(*args, **kwargs)
            value['capacity'] = value['stored']+1
            return value
        with patch.object(typed, 'state', state):
            self.enqueue(1)
        self.assertIn('COMPLETE_GROUP_PLAN_EXCEEDS_TYPED_FIA_EVENT_CAPACITY', self.provider.backend.blockers)
        self.assertEqual(len(self.provider.backend.candidate_inventory), 3)
        self.assertFalse(self.native.executes or self.native.sealed)
        self.ok(self.client.unregister())

    def test_actual_foreign_owner_and_generation_never_arm_group(self):
        c1.CoreWriter26Tests.test_actual_rpc_foreign_owner_or_generation_never_enters_provider_or_model(self)

    def test_late_callback_never_reexecutes_or_releases_after_clean(self):
        c1.CoreWriter26Tests.test_late_native_callback_does_not_passthrough_or_release(self)

    def test_foreign_or_missing_binding_reservation_is_not_clean(self):
        c1.CoreWriter26Tests.test_clean_keeps_reservation_until_unregister_foreign_missing_ticket_retains_owner(self)

    def test_real_native_generation_change_refuses_cleanup_without_writes(self):
        c1.CoreWriter26Tests.test_native_generation_change_refuses_readback_and_cleanup_without_writes(self)

    def test_failed_final_health_readback_retains_runtime_group_owner_then_retries(self):
        self.make()
        self.start()
        for index in range(8):
            self.enqueue(index)
        coverage = self.provider.backend.coverage
        def failure(*args):
            raise RuntimeError('CPU_FINAL_GROUP_HEALTH_READBACK_FAILED')
        with patch.object(coverage, 'recovery', replace(coverage.recovery, health=failure)):
            failed = self.client.finish()
        self.assertEqual(failed['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(failed['ownership_retained'])
        self.assertTrue(self.provider.binding.leases)
        self.assertTrue(coverage.recovery_report()['held_leases'])
        self.assertFalse(self.provider.backend.finalized)
        self.assertIsNone(self.provider.backend.proof)
        state = self.ok(self.client.unregister())
        self.assertTrue(state['group_recovery']['complete'])
        self.assertEqual(state['leases'], 0)
        self.assertEqual(len(state['cleanup_failures']), 1)
        self.assertEqual(state['cleanup_failures'][0]['action'], 'finish')
        self.assertIn('CPU_FINAL_GROUP_HEALTH_READBACK_FAILED',
            ' '.join(state['cleanup_failures'][0]['group_recovery']['errors']))
        self.assertEqual(self.model.root_calls, 8)


if __name__ == '__main__':
    unittest.main()
