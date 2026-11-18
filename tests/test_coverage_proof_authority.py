"""Independent negative contracts for live C2 authority; all execution is CPU."""
import copy
from dataclasses import fields, replace
import gc
import json
import pickle
import unittest
from unittest.mock import patch
import weakref

from fault_injection.core import UnsupportedCall
from fault_injection.coverage import Candidate, CandidateSet, Eligibility, EligibilityResult
from fault_injection.coverage_session import C2Proof, CoverageScope
from test_coverage_session import A, B, C, Fixture


class C2ProofAuthorityTests(unittest.TestCase):
    def test_issue_and_consumption_audit_bind_exact_live_object_without_becoming_authority(self):
        from fault_injection.coverage_session import _sha
        fixture = Fixture(1)
        c2 = fixture.session()
        fixture.call(c2)
        fixture.close(c2)
        proof = c2.c2_proof()
        report = c2.report()
        audit = report.pop('live_c2_proof')
        issued, = audit['issued']
        self.assertEqual(proof.report_sha256, _sha(report))
        self.assertEqual(issued['proof_sha256'], _sha(proof.report()))
        self.assertEqual(issued['object_binding']['python_object_id'], id(proof))
        serialized = json.loads(json.dumps(issued))
        self.assert_unissued(fixture, serialized)
        # Returning audit data must not expose either an authority object or
        # writable aliases to the stored receipt/proof scope.
        audit['issued'][0]['proof']['scope']['mechanism'] = 'changed'
        self.assertEqual(c2.report()['live_c2_proof']['issued'][0]['proof']['scope']['mechanism'],
                         fixture.scope.mechanism)
        c3 = fixture.session(2, 'every_match', proof)
        consumed = c3.report()['live_c2_proof']['consumed']
        self.assertEqual(consumed['object_binding']['python_object_id'], id(proof))
        self.assertEqual(consumed['proof_sha256'], _sha(proof.report()))
        self.assertEqual(consumed['consumer']['generation'], 2)
        self.assertEqual(consumed['consumer']['owner'], proof.owner)
        self.assertFalse(consumed['serializable_authority'])
        self.assertFalse(consumed['qualification_granted'])
        fixture.call(c3)
        self.assertTrue(fixture.close(c3)['contract_satisfied'])

    def assert_unissued(self, fixture, proof, *, scope=None):
        with self.assertRaisesRegex(UnsupportedCall, 'MATCHING_C2_PROOF_REQUIRED'):
            fixture.session(2, 'every_match', proof, scope=scope)

    def test_original_live_proof_admits_matching_later_generation(self):
        fixture = Fixture(2)
        proof = fixture.complete_c2()
        session = fixture.session(2, 'every_match', proof)
        for _ in range(2):
            for target in fixture.targets:
                fixture.call(session, target)
        report = fixture.close(session)
        self.assertTrue(report['contract_satisfied'], report['rejection_reasons'])
        self.assertFalse(report['public_qualification_granted'])
        self.assertEqual(report['hits'], 4)

    def test_copy_and_replace_cannot_recreate_authority_with_same_fields(self):
        fixture = Fixture(1)
        proof = fixture.complete_c2()
        for clone in (copy.copy(proof), copy.deepcopy(proof), replace(proof)):
            self.assertIsNot(clone, proof)
            self.assertEqual(clone.report(), proof.report())
            self.assert_unissued(fixture, clone)
        # Failed cloned admissions do not consume the legitimate generation.
        session = fixture.session(2, 'every_match', proof)
        fixture.call(session)
        self.assertTrue(fixture.close(session)['contract_satisfied'])

    def test_same_mechanism_cpu_receipt_cannot_be_relabelled_natural_worker(self):
        fixture = Fixture(1)
        proof = fixture.complete_c2()
        natural = replace(proof.scope, evidence_kind='natural_worker')
        forged = replace(proof, scope=natural)
        self.assertEqual(forged.scope.mechanism, proof.scope.mechanism)
        self.assertEqual(forged.target_ids, proof.target_ids)
        self.assert_unissued(fixture, forged, scope=natural)

    def test_json_pickle_or_explicit_constructor_receipt_is_unissued(self):
        fixture = Fixture(1)
        proof = fixture.complete_c2()
        data = json.loads(json.dumps(proof.report()))
        data['scope'] = CoverageScope(**data['scope'])
        data['target_ids'] = tuple(data['target_ids'])
        # Even deliberately retaining the old marker cannot replace issuance.
        data['_issuer'] = proof._issuer
        reconstructed = C2Proof(**data)
        direct = C2Proof(**{field.name: getattr(proof, field.name) for field in fields(C2Proof)})
        restored = pickle.loads(pickle.dumps(proof))
        object.__setattr__(restored, '_issuer', proof._issuer)
        for clone in (reconstructed, direct, restored):
            self.assertEqual(clone.report(), proof.report())
            self.assert_unissued(fixture, clone)

    def test_original_object_top_level_or_nested_field_mutation_is_rejected(self):
        for attribute, changed in (('report_sha256', A), ('discovery_sha256', B),
                                   ('observation_sha256', A), ('recovery_sha256', C),
                                   ('session_id', 'changed-session'), ('generation', 0),
                                   ('owner', 'changed-owner'), ('target_ids', (A,))):
            with self.subTest(attribute=attribute):
                fixture = Fixture(1)
                proof = fixture.complete_c2()
                self.assertNotEqual(getattr(proof, attribute), changed)
                object.__setattr__(proof, attribute, changed)
                self.assert_unissued(fixture, proof)
        fixture = Fixture(1)
        proof = fixture.complete_c2()
        # Scope is itself frozen, but forced mutation must not change the seal.
        object.__setattr__(proof.scope, 'evidence_kind', 'natural_worker')
        self.assert_unissued(fixture, proof, scope=proof.scope)

    def test_overridden_report_cannot_hide_original_field_mutation(self):
        fixture = Fixture(1)
        proof = fixture.complete_c2()
        original_report = proof.report()
        object.__setattr__(proof, 'report_sha256', A)
        object.__setattr__(proof, 'report', lambda: original_report)
        self.assert_unissued(fixture, proof)

    def test_issued_proof_retains_owner_generation_target_and_scope_gates(self):
        fixture = Fixture(1)
        proof = fixture.complete_c2()
        with self.assertRaisesRegex(UnsupportedCall, 'C2_SCOPE_OR_LIFETIME_MISMATCH'):
            fixture.session(1, 'every_match', proof)
        for scope in (replace(fixture.scope, mechanism='26.residual_stomp'),
                      replace(fixture.scope, environment_sha256=A),
                      replace(fixture.scope, implementation_sha256=A)):
            with self.assertRaisesRegex(UnsupportedCall, 'C2_SCOPE_OR_LIFETIME_MISMATCH'):
                fixture.session(2, 'every_match', proof, scope=scope)
        other_owner = replace(fixture.group(2, 'every_match'),
            leases=tuple(replace(lease, owner='other') for lease in fixture.group(2, 'every_match').leases),
            lossless=tuple(replace(p, lease=replace(p.lease, owner='other'))
                           for p in fixture.group(2, 'every_match').lossless))
        with self.assertRaisesRegex(UnsupportedCall, 'C2_SCOPE_OR_LIFETIME_MISMATCH'):
            fixture.session(2, 'every_match', proof, group=other_owner)
        fixture.targets = (replace(fixture.targets[0], cache_lifecycle='new-registration'),)
        fixture.candidates = CandidateSet((Candidate('0', EligibilityResult(Eligibility.SUPPORTED),
                                                     fixture.targets[0]),), True)
        with self.assertRaisesRegex(UnsupportedCall, 'C2_SCOPE_OR_LIFETIME_MISMATCH'):
            fixture.session(2, 'every_match', proof)

    def test_registry_does_not_keep_completed_proof_alive(self):
        fixture = Fixture(1)
        proof = fixture.complete_c2()
        reference = weakref.ref(proof)
        del proof
        gc.collect()
        self.assertIsNone(reference())

    def test_fork_inherited_registry_and_object_identity_cannot_cross_process(self):
        import os
        fixture = Fixture(1)
        proof = fixture.complete_c2()
        # Windows contract: a fork would preserve addresses and weak references
        # but the child has a different actual process ID.
        with patch('fault_injection.coverage_session.os.getpid', return_value=os.getpid() + 1000):
            self.assert_unissued(fixture, proof)
        session = fixture.session(2, 'every_match', proof)
        fixture.call(session)
        self.assertTrue(fixture.close(session)['contract_satisfied'])


if __name__ == '__main__':
    unittest.main()
