import unittest
from tools.review_r8_baseline import compare


class BaselineTests(unittest.TestCase):
    def test_exact_bytes_not_line_normalization(self):
        r = compare({'site.py': b'a\r\n'}, {'site.py': b'a\n'})
        self.assertFalse(r['original_runtime_bytes_unchanged'])
        self.assertNotEqual(r['differences'][0]['before']['sha256'], r['differences'][0]['after']['sha256'])

    def test_deleted_file_cannot_be_reclassified_as_addition(self):
        r = compare({'site.py': b'a'}, {'new_site.py': b'a'})
        self.assertIsNone(r['differences'][0]['after'])
        self.assertFalse(r['original_runtime_bytes_unchanged'])

    def test_addition_never_grants_qualification_or_automatic_reuse(self):
        r = compare({'site.py': b'a'}, {'site.py': b'a', 'registration.py': b'b'})
        self.assertTrue(r['original_runtime_bytes_unchanged'])
        self.assertEqual(len(r['additions']), 1)
        self.assertFalse(r['qualification_granted'])
        self.assertFalse(r['a3_results_automatically_reused'])

    def test_empty_baseline_is_not_proof(self):
        self.assertFalse(compare({}, {})['original_runtime_bytes_unchanged'])
