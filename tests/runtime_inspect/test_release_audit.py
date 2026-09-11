"""Release checks must not silently inherit live evidence after runtime changes."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('release_audit', Path(__file__).resolve().parents[2]/'tools/verify_fake_requant_release.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ReleaseAuditTests(unittest.TestCase):
    def test_matching_source(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/'x.py').write_text('stable')
            result = audit.source_equivalence(root, {'files': {'x.py': audit.sha(root/'x.py')}})
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(audit.file_map_mismatches(root, {'x.py': audit.sha(root/'x.py')}), [])
            self.assertEqual(audit.file_map_mismatches(root, {'x.py': '0'*64, 'missing.json': '0'*64}), ['x.py','missing.json'])

    def test_runtime_drift_requires_live(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/'x.py').write_text('changed')
            result = audit.source_equivalence(root, {'files': {'x.py': '0'*64}})
            self.assertEqual(result['status'], 'FAIL')

    def test_only_known_presentation_and_tests_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); p = root/'src/ascend_fi/evaluation/distortion/report.py'
            p.parent.mkdir(parents=True); p.write_text('presentation')
            self.assertEqual(audit.source_equivalence(root, {'files': {p.relative_to(root).as_posix(): '0'*64}})['status'], 'PASS')

    def test_missing_report_not_waived(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(audit.source_equivalence(Path(d), {'files': {'src/ascend_fi/evaluation/distortion/report.py': '0'*64}})['status'], 'FAIL')

    def test_manifest_cannot_escape(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, 'OUTSIDE'):
                audit.source_equivalence(Path(d), {'files': {'../outside.py': '0'*64}})

    def test_empty_manifest_not_evidence(self):
        with self.assertRaisesRegex(ValueError, 'EMPTY'):
            audit.source_equivalence(Path('.'), {'files': {}})

    def test_recovery_needs_text_pieces_and_http(self):
        b = {'http_status': 200, 'response_text': 'a', 'token_pieces': ['a']}
        self.assertTrue(audit.exact_responses({'b': b, 'r': dict(b)}, ('b','r')))
        for change in ({'token_pieces': None}, {'response_text': 'b'}, {'http_status': 500}):
            self.assertFalse(audit.exact_responses({'b': b, 'r': {**b,**change}}, ('b','r')))
