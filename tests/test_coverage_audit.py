import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from fault_injection.coverage_audit import audit_saved_coverage, index_artifacts
from fault_injection.coverage import Phase
from test_coverage_session import Fixture


def saved_report(trigger='once', c2_proof=None, fixture=None, generation=1):
    fixture = fixture or Fixture()
    session = fixture.session(generation, trigger, c2_proof)
    for target in fixture.targets:
        fixture.call(session, target)
        if trigger == 'every_match':fixture.call(session, target)
    fixture.call(session, replace(fixture.targets[0], module_object_identity=900), eligible=False)
    result = fixture.close(session)
    return fixture, session, json.loads(json.dumps(result))


class CoverageAuditTests(unittest.TestCase):
    def test_recomputes_counters_without_promoting_cpu_or_missing_artifacts(self):
        _, _, report = saved_report()
        result = audit_saved_coverage(report)
        self.assertTrue(result['report_consistent'],result['errors'])
        self.assertEqual(result['recomputed'],dict(targets=2,calls=3,eligible_calls=2,hits=2))
        self.assertFalse(result['referenced_artifact_bytes_verified'])
        self.assertFalse(result['public_qualification_granted'])
        self.assertFalse(result['c2_qualification_granted'])
        self.assertFalse(result['c3_qualification_granted'])

    def test_saved_pass_boolean_cannot_hide_bad_ledger_or_recovery(self):
        _, _, original = saved_report()
        mutations = [lambda x:x.update(hits=999), lambda x:x['recovery']['workers'][0].update(active_leases=1),
            lambda x:x['calls'][0]['receipt'].update(native_applied=False),
            lambda x:x['calls'][0].update(target_id='0'*64),
            lambda x:x['calls'][0].update(request_ids=['different']),
            lambda x:x['recovery']['targets'].pop(),
            lambda x:x['calls'].append(copy.deepcopy(x['calls'][0])),
            lambda x:x['recovery'].update(inflight=1),
            lambda x:x.update(public_qualification_granted=True)]
        for mutate in mutations:
            report=copy.deepcopy(original); mutate(report)
            with self.subTest(mutate=mutate):self.assertFalse(audit_saved_coverage(report)['report_consistent'])

    def test_c3_requires_matching_c2_and_cannot_transfer_scope(self):
        fixture, c2_session, c2 = saved_report()
        _, _, c3 = saved_report('every_match',c2_session.c2_proof(),fixture,2)
        self.assertIn('MATCHING_SAVED_C2_MISSING',audit_saved_coverage(c3)['errors'])
        self.assertTrue(audit_saved_coverage(c3,c2_report=c2)['report_consistent'])
        for key in ('implementation_sha256','workload_sha256','environment_sha256'):
            changed=copy.deepcopy(c2);changed['scope'][key]='d'*64
            self.assertIn('C2_SCOPE_OR_LIFETIME_MISMATCH',audit_saved_coverage(c3,c2_report=changed)['errors'])

    def test_non_target_replacement_cannot_hide_as_passthrough(self):
        _, _, report = saved_report()
        report['calls'][-1]['receipt']['native_applied']=True
        self.assertIn('UNREPORTED_NATIVE_INTERVENTION',audit_saved_coverage(report)['errors'])

    def test_hash_index_checks_actual_bytes_without_reading_tensor_contents(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'proof.json';path.write_bytes(b'{"proof":"abc"}\n')
            first=index_artifacts([path]);path.write_bytes(b'{"proof":"abd"}\n')
            second=index_artifacts([path])
            self.assertNotEqual(set(first),set(second))
            self.assertEqual(next(iter(first.values()))[0]['bytes'],16)


if __name__=='__main__':unittest.main()
