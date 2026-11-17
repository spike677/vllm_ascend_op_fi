"""CPU executable contracts only: no Native/NPU or public C2/C3 qualification."""
from dataclasses import replace
import json
import threading
import time
import unittest
from unittest.mock import patch
import uuid

from fault_injection.core import UnsupportedCall
from fault_injection.coverage import (Candidate, CandidateSet, CoverageCapability, Eligibility,
    EligibilityResult, ObservedCall, Outcome, Phase, QualifiedTargetSet, RecoveryReport,
    TargetIdentity, TargetLease, TargetLosslessProof)
from fault_injection.coverage_session import (CoverageScope, CoverageSession, GroupRecoveryHooks,
    IndependentCallObserver, SessionBudget, WorkerHealthProof)
from fault_injection.runtime_discovery import RuntimeCandidateInventory


A, B, C = 'a' * 64, 'b' * 64, 'c' * 64


class Fixture:
    def __init__(self, count=2):
        nonce = uuid.uuid4().hex
        self.owner = 'owner-' + nonce
        self.targets = tuple(TargetIdentity(121, 'worker-' + nonce, 0, 0, 'load-' + nonce,
            50, 60 + i, (), 'aclnnMatmul', 'op_exec_cache_run', 'executor-' + str(i),
            'cache-life-1', 'BF16:2x8', A, provider_identity='39.matmul_range:fixture',
            process_start_identity='process-start-' + nonce) for i in range(count))
        self.candidates = CandidateSet(tuple(Candidate(str(i), EligibilityResult(Eligibility.SUPPORTED), target)
                                            for i, target in enumerate(self.targets)), True)
        self.scope = CoverageScope('39.matmul_range', A, B, C, evidence_kind='cpu_contract')
        self.events, self.drain_ok, self.health_ok, self.restore_ok = [], True, True, True

    def observer(self, candidates=None):
        candidates = candidates or self.candidates
        observer = IndependentCallObserver(source='independent_native_trace', observer_id='trace-' + uuid.uuid4().hex,
            implementation_sha256=C, workload_sha256=A, max_calls=40)
        workers = tuple(dict.fromkeys(t.worker for t in candidates.eligible))
        observer.seal_discovery(candidates, observed_targets=candidates.eligible, artifact_sha256=A,
                                expected_workers=workers, observed_workers=workers)
        return observer

    def group(self, generation=1, trigger='once', candidates=None):
        leases = tuple(TargetLease(t, self.owner, generation, 'lease-' + str(i), 8, 4)
                       for i, t in enumerate(self.targets))
        return QualifiedTargetSet(candidates or self.candidates, leases,
            tuple(TargetLosslessProof(lease, 'input', A, (B,), (B,), C, C, 1, True, True) for lease in leases),
            CoverageCapability(single_target=True, multi_target=True, every_match=True, max_concurrency=1,
                max_resources=1024, max_host_bytes=1024, group_recovery=True,
                recovery_scope='target_set', stop_policy='bounded_session'),
            selection='all_eligible', trigger=trigger)

    def hooks(self):
        def stop(lease):
            self.events.append(('stop', lease.lease_id))
        def drain(lease, timeout):
            self.events.append(('drain', lease.lease_id))
            return self.drain_ok
        def recover(lease):
            self.events.append(('recover', lease.lease_id))
            return RecoveryReport(lease, True, self.restore_ok, self.restore_ok,
                                  proof_sha256=A, off_output_sha256=C)
        def health(worker, owner, generation):
            self.events.append(('health', worker))
            return WorkerHealthProof(worker, owner, generation, A, B, B,
                                     0, 0 if self.health_ok else 8, 0, 0)
        return GroupRecoveryHooks(stop, drain, recover, health)

    def session(self, generation=1, trigger='once', c2_proof=None, observer=None, budget=None, group=None, scope=None):
        return CoverageSession(group=group or self.group(generation, trigger), scope=scope or self.scope,
            observer=observer or self.observer(), budget=budget or SessionBudget(20, 20), recovery=self.hooks(),
            c2_proof=c2_proof)

    def call(self, session, target=None, *, outcome=None, eligible=True, phase=Phase.PRE_FAULT_PROPAGATION,
             propagation_added=False, request_ids=('request',), finish=True):
        ticket = session.observer.observe(ObservedCall(target or self.targets[0], uuid.uuid4().hex,
            'forward', request_ids, phase), eligible=eligible, propagation_added=propagation_added)
        decision = session.begin(ticket)
        if finish:
            outcome = outcome or (Outcome.HIT if decision.admitted else Outcome.PASSTHROUGH)
            session.finish(decision, outcome, execution_sha256=A, effect_sha256=B,
                           native_applied=outcome == Outcome.HIT)
        return decision

    def close(self, session):
        session.observer.seal(artifact_sha256=C)
        session.disable()
        return session.report()

    def complete_c2(self):
        session = self.session()
        for target in self.targets:
            self.call(session, target)
        report = self.close(session)
        assert report['contract_satisfied'], report['rejection_reasons']
        return session.c2_proof()


class CoverageSessionTests(unittest.TestCase):
    def test_two_targets_once_and_original_passthrough_have_separate_denominator(self):
        f = Fixture(); session = f.session()
        for target in f.targets:
            self.assertTrue(f.call(session, target).admitted)
            self.assertEqual(f.call(session, target).reason, 'TARGET_ALREADY_ATTEMPTED')
        outsider = replace(f.targets[0], module_object_identity=90)
        self.assertEqual(f.call(session, outsider, eligible=False).reason, 'NON_TARGET')
        report = f.close(session)
        self.assertTrue(report['contract_satisfied'], report['rejection_reasons'])
        self.assertEqual((report['planned_targets'], report['hits'], report['observed_eligible_calls']), (2, 2, 4))
        self.assertFalse(report['public_qualification_granted'])
        self.assertFalse(report['real_worker_evidence'])
        self.assertEqual([e[0] for e in f.events[:2]], ['stop', 'stop'])
        json.dumps(report)
        json.dumps(session.c2_proof().report())

    def test_all_eligible_singleton_is_valid(self):
        f = Fixture(1)
        self.assertEqual(len(f.complete_c2().target_ids), 1)

    def test_unresolved_candidates_or_process_lifetime_fail_closed(self):
        f = Fixture()
        candidate = Candidate('ambiguous', EligibilityResult(Eligibility.AMBIGUOUS, ('two native caches',)))
        candidates = replace(f.candidates, candidates=f.candidates.candidates + (candidate,))
        with self.assertRaisesRegex(UnsupportedCall, 'UNRESOLVED'):
            f.session(group=f.group(candidates=candidates), observer=f.observer(candidates))
        f.targets = tuple(replace(t, process_start_identity=None) for t in f.targets)
        f.candidates = CandidateSet(tuple(Candidate(str(i), EligibilityResult(Eligibility.SUPPORTED), t)
                                        for i, t in enumerate(f.targets)), True)
        with self.assertRaisesRegex(UnsupportedCall, 'LIFETIME'):
            f.session()

    def test_missing_observed_call_or_incomplete_trace_prevents_c2(self):
        for incomplete in (False, True):
            f = Fixture(); session = f.session()
            for target in f.targets:
                f.call(session, target)
            if not incomplete:
                session.observer.observe(ObservedCall(f.targets[0], 'missing', 'forward', ('r',),
                    Phase.PRE_FAULT_PROPAGATION), eligible=True)
            session.observer.seal(artifact_sha256=A, complete=not incomplete)
            session.disable()
            with self.assertRaisesRegex(UnsupportedCall, 'C2_CONTRACT'):
                session.c2_proof()

    def test_foreign_ticket_duplicate_begin_and_duplicate_finish_rejected(self):
        f = Fixture(); session = f.session()
        d = f.call(session, finish=False)
        with self.assertRaisesRegex(UnsupportedCall, 'FABRICATED'):
            session.begin(replace(d.ticket))
        with self.assertRaisesRegex(UnsupportedCall, 'ALREADY_DISPATCHED'):
            session.begin(d.ticket)
        session.finish(d, Outcome.HIT, execution_sha256=A, effect_sha256=B, native_applied=True)
        with self.assertRaisesRegex(UnsupportedCall, 'ALREADY_FINISHED'):
            session.finish(d, Outcome.PASSTHROUGH)
        f.close(session)

    def test_hit_requires_effect_original_and_native_execution(self):
        for kwargs in ({'effect_sha256': None}, {'native_applied': False}, {'original_calls': 2}):
            f = Fixture(1); session = f.session(); d = f.call(session, finish=False)
            values = dict(execution_sha256=A, effect_sha256=B, native_applied=True, original_calls=1)
            values.update(kwargs)
            session.finish(d, Outcome.HIT, **values)
            report = f.close(session)
            self.assertFalse(report['contract_satisfied'])
            self.assertEqual(report['hits'], 0)

    def test_non_target_native_write_is_never_hidden_as_passthrough(self):
        f = Fixture(1); session = f.session()
        f.call(session)
        d = f.call(session, replace(f.targets[0], module_object_identity=99), eligible=False, finish=False)
        session.finish(d, Outcome.PASSTHROUGH, native_applied=True)
        self.assertIn('NON_TARGET_OR_STOPPED_REPLACEMENT', f.close(session)['rejection_reasons'])

    def test_c3_needs_matching_c2_scope_and_new_generation(self):
        f = Fixture()
        with self.assertRaisesRegex(UnsupportedCall, 'MATCHING_C2'):
            f.session(trigger='every_match')
        proof = f.complete_c2()
        for scope in (replace(f.scope, workload_sha256=B), replace(f.scope, evidence_kind='natural_worker'),
                      replace(f.scope, implementation_sha256=A)):
            observer = f.observer()
            if scope.workload_sha256 != A:
                # Changing the workload in both observer and proof still fails the C2 scope binding.
                observer.workload_sha256 = scope.workload_sha256
            with self.assertRaisesRegex(UnsupportedCall, 'C2_SCOPE'):
                f.session(2, 'every_match', proof, observer=observer, scope=scope)
        session = f.session(2, 'every_match', proof)
        for _ in range(3):
            for target in f.targets:
                self.assertTrue(f.call(session, target).admitted)
        report = f.close(session)
        self.assertTrue(report['contract_satisfied'], report['rejection_reasons'])
        self.assertEqual((report['hits'], report['observed_eligible_calls']), (6, 6))

    def test_c3_passthrough_is_a_coverage_failure(self):
        f = Fixture(1); proof = f.complete_c2()
        session = f.session(2, 'every_match', proof)
        f.call(session); f.call(session, outcome=Outcome.PASSTHROUGH)
        self.assertIn('EVERY_MATCH_NOT_PROVEN', f.close(session)['rejection_reasons'])

    def test_new_propagation_target_separate_existing_target_not_exempt(self):
        f = Fixture(1); proof = f.complete_c2()
        session = f.session(2, 'every_match', proof)
        f.call(session)
        new_target = replace(f.targets[0], executor_identity='post-propagation')
        decision = f.call(session, new_target, phase=Phase.POST_FIRST_ANOMALY, propagation_added=True)
        self.assertEqual(decision.reason, 'PROPAGATION_ADDITION')
        report = f.close(session)
        self.assertTrue(report['contract_satisfied'])
        self.assertEqual(len(report['propagation_added_calls']), 1)
        session = f.session(3, 'every_match', proof)
        f.call(session, phase=Phase.POST_FIRST_ANOMALY, propagation_added=True)
        self.assertIn('DISCOVERED_TARGET_CLASSIFICATION_CHANGED', f.close(session)['rejection_reasons'])

    def test_call_budget_stops_new_native_admission(self):
        f = Fixture(); session = f.session(budget=SessionBudget(1, 20))
        f.call(session)
        d = f.call(session, f.targets[1])
        self.assertFalse(d.admitted)
        self.assertEqual(d.reason, 'SESSION_BUDGET_EXHAUSTED')
        self.assertFalse(f.close(session)['contract_satisfied'])

    def test_disable_waits_for_inflight_original_and_native(self):
        f = Fixture(1); session = f.session(); d = f.call(session, finish=False)
        output = []
        thread = threading.Thread(target=lambda: output.append(session.disable(2)))
        thread.start()
        deadline = time.monotonic() + 1
        while not f.events and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertFalse(any(e[0] == 'drain' for e in f.events))
        session.finish(d, Outcome.HIT, execution_sha256=A, effect_sha256=B, native_applied=True)
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertTrue(output[0]['complete'])
        session.observer.seal(artifact_sha256=C)
        self.assertTrue(session.report()['contract_satisfied'])

    def test_failed_drain_keeps_group_owned_until_retry_and_health_zero(self):
        f = Fixture(); session = f.session(); f.drain_ok = False
        for target in f.targets:
            f.call(session, target)
        self.assertFalse(session.disable(0)['complete'])
        with self.assertRaisesRegex(UnsupportedCall, 'ALREADY_OWNED'):
            f.session(2)
        f.drain_ok, f.health_ok = True, False
        self.assertFalse(session.disable()['complete'])
        with self.assertRaisesRegex(UnsupportedCall, 'ALREADY_OWNED'):
            f.session(2)
        f.health_ok = True
        self.assertTrue(session.disable()['complete'])
        session.observer.seal(artifact_sha256=C)
        self.assertTrue(session.report()['contract_satisfied'])
        with self.assertRaisesRegex(UnsupportedCall, 'STALE_OWNER'):
            f.session()

    def test_unproven_requests_refuse_native_admission(self):
        f = Fixture(1); session = f.session()
        self.assertFalse(f.call(session, request_ids=None).admitted)
        self.assertFalse(f.close(session)['contract_satisfied'])

    def test_second_drain_failure_blocks_every_restore_then_retry_redrains_all(self):
        f = Fixture(); session = f.session()
        for target in f.targets:
            f.call(session, target)
        original_hooks = session.recovery
        allow_second = False

        def drain(lease, timeout):
            f.events.append(('drain', lease.lease_id))
            return lease.lease_id == 'lease-0' or allow_second

        session.recovery = replace(original_hooks, drain=drain)
        pending = session.disable()
        self.assertFalse(pending['complete'])
        self.assertEqual(len(pending['held_leases']), 2)
        self.assertEqual(len(pending['pending_leases']), 2)
        self.assertFalse(any(event[0] in ('recover', 'health') for event in f.events))
        with self.assertRaisesRegex(UnsupportedCall, 'ALREADY_OWNED'):
            f.session(2)
        before_retry = len(f.events)
        allow_second = True
        completed = session.disable()
        self.assertTrue(completed['complete'])
        self.assertEqual(completed['held_leases'], [])
        self.assertEqual([event[0] for event in f.events[before_retry:]],
                         ['drain', 'drain', 'recover', 'recover', 'health'])
        session.observer.seal(artifact_sha256=C)
        self.assertTrue(session.report()['contract_satisfied'])

    def test_drain_returning_true_after_timeout_cannot_restore_or_release(self):
        f = Fixture(); session = f.session()
        for target in f.targets:
            f.call(session, target)
        elapsed = [100.0]

        def drain(lease, timeout):
            f.events.append(('drain', lease.lease_id))
            if lease.lease_id == 'lease-1':
                elapsed[0] += 2.0
            return True

        original_hooks = session.recovery
        session.recovery = replace(original_hooks, drain=drain)
        with patch('fault_injection.coverage_session.time.monotonic', side_effect=lambda: elapsed[0]):
            pending = session.disable(timeout_seconds=1)
        self.assertFalse(pending['complete'])
        self.assertEqual(len(pending['held_leases']), 2)
        self.assertTrue(any('GROUP_DRAIN_TIMEOUT' in reason for reason in pending['errors']))
        self.assertFalse(any(event[0] in ('recover', 'health') for event in f.events))
        with self.assertRaisesRegex(UnsupportedCall, 'ALREADY_OWNED'):
            f.session(2)
        session.recovery = original_hooks
        before_retry = len(f.events)
        self.assertTrue(session.disable()['complete'])
        self.assertEqual([event[0] for event in f.events[before_retry:]],
                         ['drain', 'drain', 'recover', 'recover', 'health'])
        session.observer.seal(artifact_sha256=C)
        self.assertTrue(session.report()['contract_satisfied'])

    def test_observer_overflow_cannot_be_sealed_complete(self):
        f = Fixture(1); observer = f.observer(); observer.max_calls = 1
        session = f.session(observer=observer); f.call(session)
        with self.assertRaisesRegex(UnsupportedCall, 'OBSERVER_RESOURCE'):
            f.call(session)
        self.assertFalse(f.close(session)['observation']['complete'])


class RuntimeCandidateInventoryTests(unittest.TestCase):
    def test_complete_observer_inventory_and_unseen_module_are_distinct(self):
        for missing in (False, True):
            f = Fixture()
            observer = IndependentCallObserver(source='independent_native_trace', observer_id='healthy-native-trace',
                implementation_sha256=C, workload_sha256=A, max_calls=20)
            inventory = RuntimeCandidateInventory(expected_modules=[t.module_object_identity for t in f.targets],
                expected_workers=(f.targets[0].worker,))
            for forward in ('healthy-1', 'healthy-2'):
                for candidate in f.candidates.candidates[:1 if missing else 2]:
                    inventory.observe(candidate, forward_id=forward)
                inventory.finish_forward(forward, healthy=True)
            candidates = inventory.seal(observer, artifact_sha256=A)
            self.assertEqual(candidates.discovery_complete, not missing)
            self.assertEqual(observer.discovery is not None, not missing)
            if missing:
                self.assertEqual(candidates.candidates[-1].eligibility.status, Eligibility.NOT_OBSERVED)

    def test_shape_equal_distinct_lifetimes_do_not_merge(self):
        f = Fixture(1)
        inventory = RuntimeCandidateInventory(expected_modules=[60], expected_workers=(f.targets[0].worker,))
        candidate = f.candidates.candidates[0]
        inventory.observe(candidate, forward_id='healthy-1')
        with self.assertRaisesRegex(UnsupportedCall, 'IDENTITY_CHANGED'):
            inventory.observe(replace(candidate, identity=replace(candidate.identity, cache_lifecycle='reused')),
                              forward_id='healthy-2')


if __name__ == '__main__':
    unittest.main()
