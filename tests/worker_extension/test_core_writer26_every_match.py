"""The existing live C2 proof crosses public RPC phases, never JSON authority."""
from dataclasses import replace
import ctypes
from unittest.mock import patch
import unittest

import torch

from demoncat.runtime.vllm_ascend.worker_extension import serving26_core as core
import test_core_writer26 as c1
import test_core_writer26_coverage as c2


class CoreWriter26EveryMatchTests(unittest.TestCase):
    setUp = c2.CoreWriter26CoverageTests.setUp
    dispose = c1.CoreWriter26Tests.dispose
    ok = c2.CoreWriter26CoverageTests.ok
    start = c1.CoreWriter26Tests.start
    enqueue = c1.CoreWriter26Tests.enqueue
    snapshot = c1.CoreWriter26Tests.snapshot
    root = c1.CoreWriter26Tests.root
    forward = c1.CoreWriter26Tests.forward

    def make(self, *, fia=False, count=3, max_targets=128):
        install = core.install_writer26_core_binding
        def bind(*args, **kwargs):
            return install(*args, stage=core.EVERY_MATCH_STAGE, max_targets=max_targets, **kwargs)
        self.enterContext(patch.object(core, 'install_writer26_core_binding', bind))
        c1.CoreWriter26Tests.make(self, fia=fia, count=count)
        self.seal_offsets = []
        def seal(targets):
            self.assertFalse(self.native.sealed)
            self.assertIsNone(self.native.transaction)
            start = self.seal_offsets[-1]+2*len(targets) if self.seal_offsets else 0
            actual = [(mode, key) for _, mode, key in self.native.executes[start:]]
            expected = [(mode, key) for mode in ('reference', 'lossless', 'off') for key in targets]
            self.assertCountEqual(actual, expected)
            self.seal_offsets.append(len(self.native.executes))
            self.native.sealed = True
        self.enterContext(patch.object(self.native, 'seal', seal))

    def c2(self):
        self.start()
        for index in range(8):
            self.enqueue(index)
        self.assertEqual(self.provider.current_run, 'c3')
        self.assertTrue(self.provider.c2_session.backend.finalized)
        self.assertEqual(self.provider.backend.phase, 'inventory_1')
        self.assertIs(self.provider.backend.c2_proof, self.provider.c2_proof)
        self.assertIs(self.provider.c2_session.backend.proof, self.provider.c2_proof)
        self.assertEqual(self.native.config.generation, self.provider._ticket[1]+1)
        self.assertEqual(self.model.root_calls, 8)  # The transition RPC only installs; it never runs the model.

    def complete(self, total, eligible):
        self.c2()
        with patch.object(self.worker, 'demoncat_worker_identity', side_effect=AssertionError('recursive identity RPC')):
            for index in range(8, 17):
                self.enqueue(index)
            state = self.ok(self.client.finish())
        evidence = state['evidence']
        first, second = evidence['prior_c2']['coverage'], evidence['coverage']
        self.assertEqual((first['planned_targets'], first['hits'], first['observed_calls']), (eligible, eligible, total*2))
        self.assertEqual((second['planned_targets'], second['hits'], second['observed_calls']), (eligible, eligible*2, total*3))
        self.assertTrue(first['contract_satisfied'] and second['contract_satisfied'])
        issued, = first['live_c2_proof']['issued']
        consumed = second['live_c2_proof']['consumed']
        self.assertEqual(issued['proof_sha256'], consumed['proof_sha256'])
        self.assertEqual(issued['object_binding'], consumed['object_binding'])
        self.assertTrue(consumed['exact_issued_object_verified'])
        self.assertEqual(second['live_c2_proof']['issued'], [])
        self.assertEqual(second['generation'], first['generation']+1)
        self.assertEqual(second['owner'], first['owner'])
        self.assertEqual(first['scope'], second['scope'])
        self.assertEqual(len(evidence['candidate_index']), total)
        self.assertEqual(len(evidence['prior_c2']['candidate_index']), total)
        self.assertTrue(all(row['native_count'] == total for row in evidence['request_index']))
        self.assertFalse(evidence['c2'] or evidence['c3'] or evidence['public_core']['qualification_granted'])
        self.assertEqual((state['leases'], state['inflight'], state['serving']['owned_hooks']), (0, 0, 0))
        self.ok(self.client.disable())
        self.ok(self.client.unregister())
        self.assertEqual(self.client.detach()['status'], 'DETACHED')
        with torch.inference_mode():
            self.forward(self.provider.session)
        self.assertEqual(self.model.root_calls, 18)
        return evidence

    def test_qmm_same_runtime_owner_real_requests_consume_original_c2_then_hit_every_match(self):
        self.make()
        evidence = self.complete(3, 3)
        self.assertEqual(evidence['current_run'], 'c3')
        self.assertEqual(len(self.native.executes), 33)

    def test_typed_fia_requalifies_all_29_with_full_30_in_both_generations(self):
        self.make(fia=True, count=30)
        evidence = self.complete(30, 29)
        self.assertEqual(evidence['candidate_index'][0]['status'], 'UNSUPPORTED')
        self.assertEqual(evidence['event_capacity_plan']['remaining_complete_requests'], 7)
        self.assertEqual(evidence['event_capacity_plan']['additional_owned_execute_records'], 174)
        self.assertEqual(len(self.native.executes), 319)

    def test_copied_original_proof_is_refused_before_any_c3_fault(self):
        self.make()
        self.c2()
        self.provider.backend.c2_proof = replace(self.provider.c2_proof)
        for index in range(8, 13):
            self.enqueue(index)
        faults = len([row for row in self.native.executes if row[1] == 'fault'])
        self.actual_request_id = 'copied-c2-cannot-authorize-c3'
        self.ok(self.client.advance(self.actual_request_id))
        with torch.inference_mode(), self.assertRaisesRegex(RuntimeError, 'MATCHING_C2_PROOF_REQUIRED'):
            self.forward(self.provider.session)
        self.assertEqual(len([row for row in self.native.executes if row[1] == 'fault']), faults)
        self.assertTrue(self.client.snapshot()['ownership_retained'])
        self.ok(self.client.unregister())

    def test_c2_cleanup_failure_retains_owner_and_prevents_c3_native_generation(self):
        self.make()
        self.start()
        for index in range(7):
            self.enqueue(index)
        coverage = self.provider.backend.coverage
        def fail(*args):
            raise RuntimeError('CPU C2 HEALTH READBACK FAILED')
        self.enqueue(7, collect=False)
        with patch.object(coverage, 'recovery', replace(coverage.recovery, health=fail)):
            reply = self.client.collect_replay()
        self.assertEqual(reply['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(reply['ownership_retained'])
        self.assertFalse(self.provider.c3_started)
        self.assertEqual(self.native.config.generation, self.provider._ticket[1])
        # Runtime recovery permits cleanup, not retrying another fault stage.
        state = self.ok(self.client.recover())
        self.assertFalse(self.provider.c3_started)
        self.assertEqual(self.model.root_calls, 8)
        self.assertIn('CPU C2 HEALTH READBACK FAILED', str(state['transition_failures']))
        self.ok(self.client.unregister())

    def test_changed_native_generation_never_writes_or_cleans_a_foreign_c3_owner(self):
        self.make()
        self.c2()
        before = len(self.native.executes)
        self.native.config.generation += 1
        try:
            reply = self.client.unregister()
            self.assertEqual(reply['status'], 'PROVIDER_ACTION_FAILED')
            self.assertTrue(reply['ownership_retained'])
            self.assertEqual(len(self.native.executes), before)
        finally:
            self.native.config.generation -= 1
        self.ok(self.client.unregister())

    def test_old_c2_callbacks_cannot_execute_again_after_c3_has_installed(self):
        self.make()
        self.c2()
        old = self.provider.c2_session.backend
        before = len(self.native.executes)
        self.assertFalse(old.installed)
        self.assertEqual(old._callback(ctypes.c_void_p()), 1)
        self.assertEqual(old._completed(ctypes.c_void_p()), 1)
        self.assertEqual(len(self.native.executes), before)
        self.assertTrue(self.provider.backend.installed)
        self.assertIn('WRITER26_LATE_NATIVE_CALLBACK', old.failure)
        self.ok(self.client.unregister())

    def test_c3_install_after_native_open_failure_retains_owner_until_explicit_cleanup(self):
        self.make()
        self.start()
        for index in range(7):
            self.enqueue(index)
        self.enqueue(7, collect=False)
        install = core.Serving26CoreSession.install
        def after_install(session):
            install(session)
            raise RuntimeError('CPU AFTER C3 NATIVE AND HOOK INSTALL')
        with patch.object(core.Serving26CoreSession, 'install', after_install):
            reply = self.client.collect_replay()
        self.assertEqual(reply['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(reply['ownership_retained'])
        self.assertEqual(self.provider.current_run, 'c3')
        self.assertTrue(self.provider.backend.installed)
        self.assertEqual(self.model.root_calls, 8)
        self.assertEqual(len(self.native.executes), 15)
        state = self.ok(self.client.recover())
        self.assertEqual(self.model.root_calls, 8)
        self.assertEqual(state['serving']['owned_hooks'], 0)
        self.assertIn('CPU AFTER C3 NATIVE AND HOOK INSTALL', str(state['transition_failures']))
        self.ok(self.client.unregister())

    def test_c3_factory_failure_preserves_c2_originals_and_zero_resource_cleanup(self):
        self.make()
        self.start()
        for index in range(7):
            self.enqueue(index)
        self.enqueue(7, collect=False)
        binding = self.provider.binding
        original = binding.runtime_factory
        def failed_factory(*args):
            raise RuntimeError('CPU C3 FACTORY REFUSED')
        object.__setattr__(binding, 'runtime_factory', failed_factory)
        try:
            reply = self.client.collect_replay()
        finally:
            object.__setattr__(binding, 'runtime_factory', original)
        self.assertEqual(reply['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(reply['ownership_retained'])
        self.assertFalse(self.native.installed)
        self.assertIsNone(self.provider.backend)
        state = self.ok(self.client.recover())
        self.assertEqual(len(state['evidence']['prior_c2']['coverage']['live_c2_proof']['issued']), 1)
        self.assertIn('CPU C3 FACTORY REFUSED', str(state['transition_failures']))
        self.assertEqual(self.model.root_calls, 8)
        self.ok(self.client.unregister())


if __name__ == '__main__':
    unittest.main()
