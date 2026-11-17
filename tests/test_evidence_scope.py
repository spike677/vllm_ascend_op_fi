"""Scope contracts only; fixtures do not confer model or NPU qualification."""
from dataclasses import replace
import unittest

from fault_injection.core import UnsupportedCall
from fault_injection.coverage import BoundaryTargetIdentity, Candidate, CandidateSet, Eligibility, EligibilityResult, PythonHookBoundary
from fault_injection.coverage_session import CoverageScope
from test_coverage_session import Fixture, A, B, C


def model_fixture(device='npu', scope_kind='MODEL_LOCAL'):
    fixture = Fixture(1)
    old = fixture.targets[0]
    target = BoundaryTargetIdentity(process_id=old.process_id, process_start_identity=old.process_start_identity,
        worker_id=old.worker_id, scope_kind=scope_kind, rank=old.rank, device_type=device,
        logical_device=0 if device == 'npu' else None, model_instance_id=old.model_instance_id,
        model_object_identity=old.model_object_identity, module_object_identity=old.module_object_identity,
        module_roles=('NORM',), operator_api='RMSNorm.forward', tensor_signature=A,
        boundary=PythonHookBoundary('actual-model-forward', B, 0), provider_identity='18.api_half:1')
    fixture.targets = (target,)
    fixture.candidates = CandidateSet((Candidate('actual', EligibilityResult(Eligibility.SUPPORTED), target),), True)
    fixture.scope = CoverageScope('18.api_half', A, B, C, evidence_kind='natural_pretrained_model',
        model_weights_sha256=A, model_source_sha256=B)
    return fixture


class EvidenceScopeTests(unittest.TestCase):
    def test_actual_model_identity_is_required(self):
        for extra in ({}, {'model_weights_sha256': A}, {'model_weights_sha256': A, 'model_source_sha256': 'unknown'}):
            with self.assertRaises(ValueError):
                CoverageScope('18.api_half', A, B, C, evidence_kind='natural_pretrained_model', **extra)

    def test_model_local_is_separate_from_constructed_and_worker(self):
        fixture = model_fixture()
        session = fixture.session()
        fixture.call(session)
        report = fixture.close(session)
        self.assertTrue(report['contract_satisfied'])
        self.assertTrue(report['real_pretrained_model_evidence'])
        self.assertFalse(report['real_worker_evidence'])
        self.assertFalse(report['public_qualification_granted'])
        self.assertEqual(report['scope']['model_weights_sha256'], A)
        self.assertEqual(report['scope']['model_source_sha256'], B)
        proof = session.c2_proof()
        next_session = fixture.session(2, 'every_match', proof)
        fixture.call(next_session); fixture.call(next_session)
        self.assertTrue(fixture.close(next_session)['contract_satisfied'])

    def test_cpu_worker_and_native_identities_cannot_impersonate_model_local(self):
        for fixture in (model_fixture('cpu'), model_fixture(scope_kind='INITIALIZED_WORKER'), Fixture(1)):
            with self.subTest(target=type(fixture.targets[0]).__name__), self.assertRaisesRegex(
                    UnsupportedCall, 'NATURAL_PRETRAINED_NPU_MODEL_REQUIRED'):
                fixture.session(scope=CoverageScope('18.api_half', A, B, C,
                    evidence_kind='natural_pretrained_model', model_weights_sha256=A, model_source_sha256=B))

    def test_model_c2_proof_cannot_transfer_weights_source_or_scope(self):
        fixture = model_fixture()
        proof = fixture.complete_c2()
        for changed in (replace(fixture.scope, model_weights_sha256=C),
                        replace(fixture.scope, model_source_sha256=C),
                        replace(fixture.scope, evidence_kind='constructed_npu'),
                        replace(fixture.scope, evidence_kind='cpu_contract'),
                        replace(fixture.scope, evidence_kind='natural_worker')):
            with self.subTest(scope=changed), self.assertRaises(UnsupportedCall):
                fixture.session(2, 'every_match', proof, scope=changed)


if __name__ == '__main__':
    unittest.main()
