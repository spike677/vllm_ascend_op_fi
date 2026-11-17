"""Negative audit-boundary checks; these do not simulate NPU acceptance."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.c2c3_closure_20261117.review_native39_group_124 import statistics, target_key


class Native39RecordedReviewTests(unittest.TestCase):
    def test_cache_reuse_does_not_merge_actual_sites(self):
        delta = [{'key': ['signature', 123, 7], 'count': 1}]
        self.assertNotEqual(target_key(dict(site=[1, 'linear', 0], native_delta=delta)),
                            target_key(dict(site=[2, 'linear', 0], native_delta=delta)))

    def test_ambiguous_native_call_count_refused(self):
        with self.assertRaisesRegex(ValueError, 'ONE_NATIVE_RUN'):
            target_key(dict(site=[1, 'linear', 0], native_delta=[{'key': ['s', 1, 2], 'count': 2}]))

    def test_missing_capture_is_not_inherited_from_other_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = statistics(root / 'absent', root / 'audit')
            for item in result.values():
                self.assertEqual(item['status'], 'MISSING_RAW_CAPTURE')
                self.assertFalse(item['e5_statistics_comparable'])
                self.assertIsNone(item['first_exceedance'])
                self.assertTrue(item['missing_files'])
            saved = json.loads((root / 'audit/baseline_fault/summary.json').read_text())
            self.assertEqual(saved, result['baseline_fault'])
            with self.assertRaises(FileExistsError):
                statistics(root / 'absent', root / 'audit')


if __name__ == '__main__':
    unittest.main()
