import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from engineering_closure.evidence import audit_bundle, audit_checkpoint, STAGES


class EvidenceClosureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / 'original.json'
        self.path.write_text('{"claimed_pass":true}', encoding='utf-8')
        self.reference = dict(path=self.path.name, sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.scope = dict(mechanism='18.api_half', model='fixture-only', cann='8.5.1',
            soc='CPU-contract', tp=1, source_commit='test-fixture',
            implementation_sha256='a'*64, environment_sha256='b'*64, workload_sha256='c'*64)
        self.bundle = dict(schema='demoncat.engineering.evidence_bundle.v1', scope=self.scope,
            stages={'source': [self.reference]})

    def test_bytes_verified_without_promoting_saved_pass(self):
        report = audit_bundle(self.bundle, artifact_root=self.root)
        self.assertTrue(report['reference_integrity_verified'])
        self.assertFalse(report['public_qualification_granted'])
        self.assertIsNone(report['mechanism_coverage'])
        self.assertEqual(len(report['missing_stages']), len(STAGES)-1)
        self.assertEqual(report['stages']['source']['files'][0]['semantic_review'], 'PENDING')

    def test_tampering_and_missing_file_remain_errors(self):
        self.path.write_text('{"claimed_pass":false}')
        self.assertIn('HASH_MISMATCH', audit_bundle(self.bundle, artifact_root=self.root)['errors'][0])
        self.path.unlink()
        self.assertFalse(audit_bundle(self.bundle, artifact_root=self.root)['reference_integrity_verified'])

    def test_path_escape_and_missing_scope_rejected(self):
        self.reference['path'] = '../outside.json'
        report = audit_bundle(self.bundle, artifact_root=self.root)
        self.assertIn('OUTSIDE_ROOT', report['errors'][0])
        del self.scope['model']
        with self.assertRaisesRegex(ValueError, 'SCOPE'):
            audit_bundle(self.bundle, artifact_root=self.root)

    def test_saved_coverage_cannot_be_used_for_another_mechanism(self):
        self.path.write_text(json.dumps({'scope': dict(self.scope, mechanism='20.buffer_substitute')}))
        self.reference['sha256'] = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.bundle['coverage'] = self.reference
        report = audit_bundle(self.bundle, artifact_root=self.root)
        self.assertTrue(any('COVERAGE_SCOPE_MISMATCH' in error for error in report['errors']))

    def test_checkpoint_accounts_for_all_catalog_mechanisms(self):
        repo = Path(__file__).resolve().parents[3]
        report = audit_checkpoint(repo)
        self.assertEqual(report['mechanism_count'], 17)
        self.assertEqual(set(report['historical_reported_c2']),
                         {'18.api_half', '20.buffer_substitute', '39.matmul_range'})
        self.assertTrue(all(row['available'] for row in report['evidence_references']))
        self.assertFalse(report['original_remote_raw_bytes_verified'])


if __name__ == '__main__':
    unittest.main()
