"""CPU review of planning and evidence reconciliation, without a native provider."""
from dataclasses import asdict, replace
import json
import unittest

from fault_injection import (
    CallOutcome, Candidate, CandidateSet, CoverageReport, Eligibility, EligibilityResult,
    IndependentObservation, ObservedCall, Outcome, Phase, RecoveryReport, TargetIdentity,
    TargetLease, UnsupportedCall, coverage_phase, list_mechanisms,
)
from fault_injection.catalog import CAPABILITIES
from fault_injection.coverage import CoverageCapability, QualifiedTargetSet, TargetLosslessProof
from fault_injection.integration import integration_manifest
from fault_injection.providers import coverage_capability


FINGERPRINT = 'd' * 64
SELF = TargetIdentity(
    840, 'worker-a', 0, 2, 'load-A', 41, 61, ('MLP_DOWN',), 'aclnnMatmul',
    'op_exec_cache_run', 'executor-1', 'registration-1', 'BF16:1x12800x5120', FINGERPRINT,
)
OTHER = replace(SELF, module_object_identity=62)


def supported(identity, name):
    return Candidate(name, EligibilityResult(Eligibility.SUPPORTED), identity)


class CoveragePlanningTests(unittest.TestCase):
    def test_catalog_map_and_no_global_opening(self):
        catalog = {item.name for item in CAPABILITIES}
        manifest = integration_manifest()
        host = {row['mechanism'] for row in manifest['mechanisms']}
        self.assertEqual(len(catalog), 17)
        self.assertEqual(manifest['mechanism_count'], 15)
        self.assertEqual(catalog - host, {'39.matmul_range', '39.output_proxy'})
        for row in list_mechanisms():
            self.assertFalse(row['coverage']['multi_target'])
            self.assertFalse(row['coverage']['every_match'])
        for name in catalog:
            capability = coverage_capability(name)
            self.assertFalse(capability.multi_target or capability.every_match)

    def test_selection_is_offline_and_refuses_ties(self):
        common = CandidateSet((supported(SELF, 'first'), supported(OTHER, 'second')), True)
        with self.assertRaisesRegex(UnsupportedCall, 'one eligible'):
            common.select()
        self.assertEqual(common.select('all_eligible'), (SELF, OTHER))
        self.assertEqual(common.select('explicit', target=OTHER), (OTHER,))
        self.assertEqual(common.select('max_k_min_m_unique', priority={SELF: (4, 8), OTHER: (8, 16)}), (OTHER,))
        with self.assertRaisesRegex(UnsupportedCall, 'one eligible'):
            common.select('max_k_min_m_unique', priority={SELF: (8, 4), OTHER: (8, 4)})
        with self.assertRaisesRegex(UnsupportedCall, 'incomplete'):
            replace(common, discovery_complete=False).select('all_eligible')
        self.assertFalse(json.loads(json.dumps(common.report()))['execution_authorized'])

    def test_identity_is_not_tensor_shape_executor_address_or_layer_alias(self):
        variants = (
            OTHER, replace(SELF, cache_lifecycle='registration-2'),
            replace(SELF, model_instance_id='reload-B'),
            replace(SELF, logical_device=5), replace(SELF, worker_id='worker-b'),
        )
        candidates = CandidateSet(tuple(supported(row, str(index))
                                        for index, row in enumerate((SELF, *variants))), True)
        self.assertEqual(len(candidates.select('all_eligible')), 6)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            CandidateSet((supported(SELF, 'one'), supported(SELF, 'alias')), True)
        with self.assertRaisesRegex(ValueError, 'lifecycle'):
            replace(SELF, cache_lifecycle='')
        with self.assertRaisesRegex(ValueError, 'complete target'):
            Candidate('unknown', EligibilityResult(Eligibility.SUPPORTED))

    def test_planned_and_observed_denominators_and_propagation(self):
        candidates = CandidateSet((
            supported(SELF, 'first'), supported(OTHER, 'second'),
            Candidate('unseen', EligibilityResult(Eligibility.NOT_OBSERVED, ('branch disabled',))),
        ), False)
        lease = TargetLease(SELF, 'controller-a', 1, 'lease-0', 256)
        second = TargetLease(OTHER, 'controller-a', 1, 'lease-1', 512)
        hidden = replace(OTHER, module_object_identity=73)
        first = ObservedCall(SELF, '1', 'forward-0', ('req-a',), coverage_phase(1, first_anomaly=1))
        later = replace(first, call_id='2', request_ids=None, phase=coverage_phase(2, first_anomaly=1))
        scarce = replace(later, call_id='3')
        outside = replace(later, target=hidden, call_id='4')
        off = replace(first, call_id='5', phase=coverage_phase(5, first_anomaly=1, off_from=5))
        observation = IndependentObservation('independent_native_trace', (first, later, scarce, outside, off), True)
        report = CoverageReport.reconcile((SELF, OTHER, hidden), candidates, (lease, second), observation, (
            CallOutcome(first, lease, Outcome.HIT, FINGERPRINT),
            CallOutcome(later, lease, Outcome.PASSTHROUGH),
            CallOutcome(scarce, lease, Outcome.RESOURCE_LIMITED),
            CallOutcome(off, lease, Outcome.HIT, FINGERPRINT),
        ))
        self.assertEqual((report.planned_targets_hit, report.planned_targets), (1, 3))
        self.assertEqual((report.observed_calls_hit, report.observed_eligible_calls), (1, 4))
        self.assertEqual(report.planned_targets_unobserved, 1)
        self.assertEqual(report.coverage_by_phase[Phase.POST_FIRST_ANOMALY]['outcome_missing'], 1)
        self.assertEqual(report.outcome_counts[Outcome.RESOURCE_LIMITED], 1)
        self.assertEqual(report.off_hits, 1)
        self.assertEqual(report.request_mapping_unproven, 3)
        self.assertEqual(later.request_mapping, 'UNPROVEN_REQUEST_MAPPING')
        self.assertEqual(report.target_outcomes[0]['counts_in_fault_window'][Outcome.HIT], 1)

    def test_missing_denominator_and_stale_or_mismatched_receipts(self):
        candidates = CandidateSet((supported(SELF, 'first'),), True)
        lease = TargetLease(SELF, 'owner', 2, 'lease-2', 8)
        call = ObservedCall(SELF, '1', 'forward-A', None, Phase.PRE_FAULT_PROPAGATION)
        hit = CallOutcome(call, lease, Outcome.HIT, FINGERPRINT)
        no_trace = CoverageReport.reconcile((SELF,), candidates, (lease,), None, (hit,))
        self.assertEqual(no_trace.planned_targets_hit, 1)
        self.assertIsNone(no_trace.observed_eligible_calls)
        self.assertIsNone(no_trace.observed_calls_hit)
        self.assertFalse(no_trace.observation_complete)
        with self.assertRaisesRegex(ValueError, 'independent'):
            IndependentObservation('injector_hit_log', (call,), True)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            IndependentObservation('independent_native_trace', (call, call), True)
        observation = IndependentObservation('independent_workload_observer', (call,), True)
        for mismatch in (
            replace(hit, lease=replace(lease, generation=1)),
            replace(hit, call=replace(call, forward_context='forward-B')),
        ):
            report = CoverageReport.reconcile((SELF,), candidates, (lease,), observation, (mismatch,))
            self.assertEqual((report.planned_targets_hit, report.observed_calls_hit), (0, 0))
            self.assertEqual(report.uncorrelated_outcomes, 1)
            self.assertEqual(report.coverage_by_phase[Phase.PRE_FAULT_PROPAGATION]['outcome_missing'], 1)
        empty_trace = replace(observation, calls=())
        report = CoverageReport.reconcile((SELF,), candidates, (lease,), empty_trace, (hit,))
        self.assertEqual((report.planned_targets_hit, report.observed_calls_hit), (0, 0))
        self.assertEqual(report.uncorrelated_outcomes, 1)

    def test_recovery_retains_owned_lease_until_exact_off(self):
        lease = TargetLease(SELF, 'owner', 3, 'lease-3', 4096)
        report = RecoveryReport(lease, drained=False, released=False, off_exact=None, pending_reason='sync failed')
        self.assertFalse(report.complete)
        synced = replace(report, drained=True, released=True, pending_reason=None)
        self.assertFalse(synced.complete)
        self.assertTrue(replace(synced, off_exact=True).complete)
        self.assertEqual(asdict(report)['lease']['device_bytes'], 4096)


class QualifiedTargetSetTests(unittest.TestCase):
    def setUp(self):
        provider = {
            'provider_identity': 'cpu_fixture.fault39.matmul_range.v1',
            'provider_proof': (('gws', 'gws-instance-1'), ('run_slot', 'self:0,mat2:1')),
        }
        self.first = replace(SELF, **provider)
        self.second = replace(OTHER, **provider)
        self.scope = CoverageCapability(
            single_target=True, multi_target=True, group_recovery=True, recovery_scope='target_set',
            max_concurrency=1, max_resources=512, max_host_bytes=128,
            lossless_modes=('self', 'mat2', 'dual'),
        )

    def configuration(self, second=None):
        second = self.second if second is None else second
        targets = (self.first, second)
        candidates = CandidateSet(tuple(supported(target, str(index))
                                        for index, target in enumerate(targets)), True)
        leases = tuple(TargetLease(target, 'session-owner', 5, str(index), 256, 64)
                       for index, target in enumerate(targets))
        receipts = tuple(
            TargetLosslessProof(lease, mode, FINGERPRINT, ('a' * 64, 'b' * 64), ('a' * 64, 'b' * 64),
                                'c' * 64, 'c' * 64, 1, True, True)
            for lease in leases for mode in self.scope.lossless_modes
        )
        return dict(candidates=candidates, leases=leases, lossless=receipts, capability=self.scope)

    def test_group_pass_still_has_no_execution_authority(self):
        config = self.configuration()
        group = QualifiedTargetSet(**config, selection='all_eligible')
        self.assertEqual(group.leases, config['leases'])
        self.assertFalse(group.execution_authorized)
        self.assertFalse(json.loads(json.dumps(group.report()))['execution_authorized'])
        with self.assertRaisesRegex(UnsupportedCall, 'Trigger'):
            replace(group, trigger='every_match')
        simulated_recurring = replace(group, trigger='every_match', capability=replace(self.scope, every_match=True))
        self.assertFalse(simulated_recurring.execution_authorized)
        with self.assertRaisesRegex(ValueError, 'boolean'):
            replace(self.scope, every_match='false')
        with self.assertRaisesRegex(ValueError, 'immutable'):
            replace(group, leases=list(group.leases))

    def test_eligibility_identity_and_selection_must_agree(self):
        group = QualifiedTargetSet(**self.configuration(), selection='all_eligible')
        outsider = replace(self.first, module_object_identity=68)
        candidates = replace(group.candidates, candidates=group.candidates.candidates
                             + (supported(outsider, 'other'),))
        with self.assertRaisesRegex(UnsupportedCall, 'Selection'):
            replace(group, candidates=candidates)
        self.assertFalse(replace(group, candidates=candidates, selection='explicit').execution_authorized)
        with self.assertRaisesRegex(UnsupportedCall, 'complete discovery'):
            replace(group, candidates=replace(candidates, discovery_complete=False))
        unsupported = Candidate('wrong target', EligibilityResult(Eligibility.UNSUPPORTED, ('shape',)), self.second)
        with self.assertRaisesRegex(UnsupportedCall, 'eligible'):
            replace(group, candidates=CandidateSet((supported(self.first, 'first'), unsupported), True))
        with self.assertRaisesRegex(UnsupportedCall, 'provider identity'):
            QualifiedTargetSet(**self.configuration(replace(self.second, provider_identity='another.v2')))
        with self.assertRaisesRegex(ValueError, 'distinct ordered'):
            replace(self.first, provider_proof=(('a', '1'), ('a', '2')))

    def test_lossless_must_cover_each_lease_and_mode_with_exact_bytes(self):
        config = self.configuration()
        proofs = config['lossless']
        first, *tail = proofs
        for incomplete in (
            proofs[1:], proofs + (first,), (replace(first, original_calls=2), *tail),
            (replace(first, temporary_readback_verified=False), *tail),
            (replace(first, resources_released=False), *tail),
            (replace(first, input_sha256_after=('0' * 64,)), *tail),
            (replace(first, output_sha256='0' * 64), *tail),
            (replace(first, reference_output_sha256='e' * 64, output_sha256='e' * 64), *tail),
            (replace(first, lease=replace(first.lease, generation=4)), *tail),
            (replace(first, mode='reference'), *tail),
        ):
            with self.subTest(proof=incomplete):
                with self.assertRaisesRegex(UnsupportedCall, 'Lossless'):
                    QualifiedTargetSet(**{**config, 'lossless': incomplete})

    def test_budget_and_owner_gate_apply_to_whole_set(self):
        group = QualifiedTargetSet(**self.configuration())
        first, second = group.leases
        for kwargs in (
            {'concurrency': 2},
            {'capability': replace(self.scope, max_resources=511)},
            {'capability': replace(self.scope, max_host_bytes=127)},
            {'capability': replace(self.scope, group_recovery=False)},
            {'leases': (first, replace(second, owner='other-owner'))},
            {'leases': (first, replace(second, generation=6))},
            {'leases': (first, replace(second, lease_id=first.lease_id))},
            {'held_leases': (replace(first, owner='other-owner', lease_id='held'),)},
            {'held_leases': (TargetLease(replace(self.first, module_object_identity=88), 'owner-b', 1, 'held', 1),)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(UnsupportedCall):
                replace(group, **kwargs)

    def test_worker_rank_isolation_is_independent_of_target_count(self):
        foreign_worker = replace(self.second, worker_id='worker-b', rank=1, logical_device=5)
        config = self.configuration(foreign_worker)
        for kwargs in ({}, {'cross_worker': True}, {'rank_isolation': True}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(UnsupportedCall, 'rank isolation'):
                QualifiedTargetSet(**{**config, 'capability': replace(self.scope, **kwargs)})
        qualified = QualifiedTargetSet(**{
            **config, 'capability': replace(self.scope, cross_worker=True, rank_isolation=True)
        })
        self.assertFalse(qualified.execution_authorized)

    def test_partial_rollback_preserves_unsettled_leases_and_off_evidence(self):
        group = QualifiedTargetSet(**self.configuration())
        first, second = group.leases
        off_first = RecoveryReport(first, True, True, True, proof_sha256=FINGERPRINT, off_output_sha256='c' * 64)
        stalled = RecoveryReport(second, False, False, None, pending_reason='native drain timed out')
        partial = group.reconcile_recovery((off_first, stalled))
        self.assertFalse(partial.complete)
        self.assertEqual(partial.pending_leases, (second,))
        self.assertEqual((partial.owner, partial.generation), ('session-owner', 5))
        self.assertEqual(group.reconcile_recovery((off_first,)).pending_leases, (second,))
        completed = RecoveryReport(second, True, True, True, proof_sha256=FINGERPRINT, off_output_sha256='c' * 64)
        self.assertFalse(group.reconcile_recovery((off_first, replace(completed, proof_sha256=None))).complete)
        self.assertFalse(group.reconcile_recovery((off_first, replace(completed, drained='unknown'))).complete)
        self.assertFalse(group.reconcile_recovery((off_first, replace(completed, off_output_sha256='e' * 64))).complete)
        self.assertTrue(group.reconcile_recovery((off_first, completed)).complete)
        for invalid in ((off_first, off_first), (replace(off_first, lease=replace(first, generation=2)),)):
            with self.assertRaisesRegex(ValueError, 'duplicate or unrelated'):
                group.reconcile_recovery(invalid)
