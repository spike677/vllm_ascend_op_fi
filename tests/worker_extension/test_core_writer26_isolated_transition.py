"""Real Host/Runtime requests; Native doubles do not grant device qualification."""
from dataclasses import replace
import ctypes
from unittest.mock import patch
import unittest

import torch

from demoncat.runtime.vllm_ascend.worker_extension import serving26_core as core
from demoncat.runtime.vllm_ascend.worker_extension import serving26_isolated as isolated
import test_core_writer26 as c1
import test_core_writer26_coverage as c2


class CoreWriter26IsolatedTransitionTests(unittest.TestCase):
    setUp = c2.CoreWriter26CoverageTests.setUp
    dispose = c1.CoreWriter26Tests.dispose
    ok = c2.CoreWriter26CoverageTests.ok
    start = c1.CoreWriter26Tests.start
    enqueue = c1.CoreWriter26Tests.enqueue
    snapshot = c1.CoreWriter26Tests.snapshot
    root = c1.CoreWriter26Tests.root
    forward = c1.CoreWriter26Tests.forward

    def make(self, *, count=3, max_targets=128, fia=True):
        self.eligible = count-1
        self.c2_requests = 6+2*self.eligible
        install = core.install_writer26_core_binding
        def bind(*args, **kwargs):
            try:
                return install(*args, stage=core.ISOLATED_C3_STAGE, max_targets=max_targets, **kwargs)
            except BaseException as error:
                self.binding_error = error
                raise
        self.enterContext(patch.object(core, 'install_writer26_core_binding', bind))
        c1.CoreWriter26Tests.make(self, fia=fia, count=count)
        self.seal_offsets = []
        def seal(targets):
            self.assertFalse(self.native.sealed)
            self.assertIsNone(self.native.transaction)
            start = self.seal_offsets[-1]+2*len(targets) if self.seal_offsets else 0
            self.assertCountEqual([(mode, key) for _, mode, key in self.native.executes[start:]],
                [(mode, key) for mode in ('reference', 'lossless', 'off') for key in targets])
            self.seal_offsets.append(len(self.native.executes))
            self.native.sealed = True
        self.enterContext(patch.object(self.native, 'seal', seal))

    def c2(self):
        self.start()
        for index in range(self.c2_requests):
            self.enqueue(index)
        self.assertEqual(self.provider.current_run, 'c3')
        old = self.provider.c2_session.backend
        self.assertTrue(old.finalized and not old.installed)
        self.assertIs(self.provider.c2_proof, old.proof)
        self.assertIs(self.provider.backend.c2_proof, old.proof)
        self.assertEqual(self.native.config.generation, self.provider._ticket[1]+1)
        self.assertEqual(self.model.root_calls, self.c2_requests)
        return old

    def complete(self, count):
        self.make(count=count)
        old = self.c2()
        with patch.object(self.worker, 'demoncat_worker_identity', side_effect=AssertionError('recursive identity RPC')):
            for index in range(self.c2_requests, self.c2_requests+9):
                self.enqueue(index)
            state = self.ok(self.client.finish())
        current = self.provider.backend
        first, second = old.report(), current.report()
        a, b = first['coverage'], second['coverage']
        self.assertEqual((a['hits'], b['hits']), (count-1, 2*(count-1)))
        self.assertEqual((a['observed_calls'], b['observed_calls']), (count*2*(count-1), count*3))
        self.assertTrue(a['contract_satisfied'] and b['contract_satisfied'])
        self.assertEqual((a['scope'], a['owner']), (b['scope'], b['owner']))
        self.assertEqual(a['generation']+1, b['generation'])
        issued, = a['live_c2_proof']['issued']
        used = b['live_c2_proof']['consumed']
        self.assertEqual(issued['object_binding'], used['object_binding'])
        self.assertEqual(issued['proof_sha256'], used['proof_sha256'])
        self.assertTrue(used['exact_issued_object_verified'])
        self.assertFalse(b['live_c2_proof']['issued'])
        for value in (first, second):
            self.assertEqual(value['scope'], isolated.TRANSITION_SCOPE)
            self.assertEqual(len(value['candidates']), count)
            self.assertEqual(value['candidates'][0]['status'], 'UNSUPPORTED')
            self.assertTrue(all(len(row['candidates']) == count for row in value['rows']))
            self.assertFalse(value['c2'] or value['c3'])
        self.assertEqual(len(first['isolated_pairs']), count-1)
        self.assertTrue(first['independent_per_target_replay'])
        self.assertFalse(second['independent_per_target_replay'] or second['isolated_pairs'])
        self.assertEqual([row['admissions'] for row in first['rows'] if row['phase'].startswith('group_')], [1]*2*(count-1))
        self.assertEqual([row['admissions'] for row in second['rows'] if row['phase'].startswith('group_')], [count-1]*3)
        self.assertEqual(first['isolated_plan']['combined_group_requests'], 15+2*(count-1))
        self.assertEqual(first['event_capacity_plan']['combined_owned_execute_records'], 11*(count-1))
        self.assertEqual(len(self.native.executes), 11*(count-1))
        self.assertEqual((state['leases'], state['inflight'], state['serving']['owned_hooks']), (0, 0, 0))
        self.ok(self.client.disable())
        self.ok(self.client.unregister())
        self.assertEqual(self.client.detach()['status'], 'DETACHED')
        with torch.inference_mode():
            self.forward(self.provider.session)
        self.assertEqual(self.model.root_calls, self.c2_requests+10)

    def test_complete_thirty_writer_census_twenty_nine_isolated_pairs_then_two_cumulative_requests(self):
        self.complete(30)

    def test_original_proof_same_scope_owner_generation_and_real_off_in_small_complete_set(self):
        self.complete(3)

    def test_copied_proof_never_admits_first_c3_fault(self):
        self.make()
        self.c2()
        self.provider.backend.c2_proof = replace(self.provider.c2_proof)
        for index in range(self.c2_requests, self.c2_requests+5):
            self.enqueue(index)
        faults = len([row for row in self.native.executes if row[1] == 'fault'])
        self.actual_request_id = 'copied-isolated-proof-is-not-authority'
        self.ok(self.client.advance(self.actual_request_id))
        with torch.inference_mode(), self.assertRaisesRegex(RuntimeError, 'MATCHING_C2_PROOF_REQUIRED'):
            self.forward(self.provider.session)
        self.assertEqual(len([row for row in self.native.executes if row[1] == 'fault']), faults)
        self.assertTrue(self.client.snapshot()['ownership_retained'])
        self.ok(self.client.unregister())

    def test_c2_health_failure_retains_owner_and_does_not_open_c3_generation(self):
        self.make()
        self.start()
        for index in range(self.c2_requests-1):
            self.enqueue(index)
        self.enqueue(self.c2_requests-1, collect=False)
        coverage = self.provider.backend.coverage
        def fail(*args):
            raise RuntimeError('CPU ISOLATED C2 HEALTH READBACK FAILED')
        with patch.object(coverage, 'recovery', replace(coverage.recovery, health=fail)):
            reply = self.client.collect_replay()
        self.assertEqual(reply['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(reply['ownership_retained'])
        self.assertFalse(self.provider.c3_started)
        self.assertIsNone(self.provider.c2_proof)
        self.assertEqual(self.native.config.generation, self.provider._ticket[1])
        self.ok(self.client.recover())
        self.ok(self.client.unregister())
        self.assertEqual(self.model.root_calls, self.c2_requests)

    def test_original_live_object_cannot_authorize_changed_implementation_scope(self):
        self.make()
        old = self.c2()
        self.assertIs(self.provider.backend.c2_proof, old.proof)
        self.provider.backend.implementation_sha = 'f'*64
        for index in range(self.c2_requests, self.c2_requests+5):
            self.enqueue(index)
        faults = len([row for row in self.native.executes if row[1] == 'fault'])
        self.actual_request_id = 'original-proof-different-scope'
        self.ok(self.client.advance(self.actual_request_id))
        with torch.inference_mode(), self.assertRaisesRegex(RuntimeError, 'C2_SCOPE_OR_LIFETIME_MISMATCH'):
            self.forward(self.provider.session)
        self.assertEqual(len([row for row in self.native.executes if row[1] == 'fault']), faults)
        self.ok(self.client.unregister())

    def test_old_callbacks_do_not_reenter_after_new_generation_install(self):
        self.make()
        old = self.c2()
        before = len(self.native.executes)
        self.assertEqual(old._callback(ctypes.c_void_p()), 1)
        self.assertEqual(old._completed(ctypes.c_void_p()), 1)
        self.assertEqual(before, len(self.native.executes))
        self.assertTrue(self.provider.backend.installed)
        self.ok(self.client.unregister())

    def test_combined_future_c3_capacity_is_reserved_before_any_c2_fault(self):
        self.make()
        self.start()
        self.enqueue(0)
        original = self.reader.state
        def state():
            value = original()
            per_request = self.provider.backend.rows[0]['native_after']['total_events']-self.provider.backend.rows[0]['native_before']['total_events']
            # Enough for the isolated C2 remainder, insufficient for its C3.
            value['capacity'] = value['stored_events']+per_request*(self.c2_requests-2)
            return value
        with patch.object(self.reader, 'state', state):
            self.enqueue(1)
        backend = self.provider.backend
        self.assertIn('ISOLATED_C2_C3_FULL_PLAN_EXCEEDS_NATIVE_EVENT_CAPACITY', backend.blockers)
        self.assertEqual((len(backend.candidate_inventory), len(backend.targets)), (3, 2))
        self.assertFalse(self.native.executes)
        self.ok(self.client.unregister())

    def test_stage_refuses_qmm_before_native_or_hooks(self):
        self.make(fia=False)
        reply = self.client.prepare()
        self.assertEqual(reply['status'], 'RPC_UNCERTAIN')
        self.assertIn('ISOLATED_C3_REVIEWED_TYPED_FIA_ONLY', str(self.binding_error))
        self.assertIsNone(reply['ticket'])
        self.assertFalse(self.worker._demoncat_worker_fault_runtime._leases)
        self.assertFalse(self.native.installed or self.native.executes)

    def test_old_isolated_class_stays_once_only(self):
        with self.assertRaisesRegex(RuntimeError, 'ISOLATED_WRITER26_IS_C2_ONLY'):
            isolated.CoreWriter26IsolatedCoverage(trigger='every_match')


if __name__ == '__main__':
    unittest.main()
