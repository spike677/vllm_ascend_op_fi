import tempfile
import unittest
from pathlib import Path

from tools.review_v4_installed_sources import REQUIRED, canonical, layer_rule, review, scan_file


class SourceIntakeTests(unittest.TestCase):
    def test_missing_is_not_default_identity(self):
        with tempfile.TemporaryDirectory() as root:
            result = review(root)
        self.assertEqual(len(result["missing_files"]), len(REQUIRED))
        self.assertIsNone(result["layer"]["candidate"])
        self.assertFalse(result["qualification_granted"])
        self.assertFalse(result["installed_identity_verified"])

    def test_source_is_parsed_not_executed(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "source.py"
            path.write_text("raise RuntimeError('never execute')\ndef route(x):\n return router._select_experts(x)\n")
            row = scan_file(path)
        self.assertEqual(row["candidate_calls"][0]["call"], "router._select_experts")

    def test_exact_diff_and_repeatable(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as upstream:
            for base in (root, upstream):
                for rel in REQUIRED:
                    path = Path(base) / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"def route(x):\n return x\n")
            (Path(root) / REQUIRED[0]).write_bytes(b"def route(x):\n return x + 1\n")
            a = review(root, upstream=upstream)
            self.assertEqual(canonical(a), canonical(review(root, upstream=upstream)))
        self.assertEqual(a["different_files"], [REQUIRED[0]])
        self.assertEqual(a["missing_files"], [])
        self.assertIsNone(a["actual_runtime_branch"])

    def test_traversal_and_absolute_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            for path in ("../escape.py", "/escape.py", "C:/escape.py", "a\\escape.py"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    review(root, extra_paths=[path])

    def test_first_non_hash_candidate_needs_real_configuration(self):
        self.assertTrue(layer_rule({"num_hash_layers": 3}, 3)["first_non_hash_candidate"])
        self.assertFalse(layer_rule({"num_hash_layers": 4}, 3)["candidate"])
        self.assertTrue(layer_rule({"num_hash_layers": 2}, 3)["candidate"])
        self.assertFalse(layer_rule({"num_hash_layers": 2}, 3)["first_non_hash_candidate"])

    def test_bad_configuration_is_unknown(self):
        for count in (None, True, -1, "3"):
            self.assertIsNone(layer_rule({"num_hash_layers": count}, 3)["candidate"])

    def test_selector_extra_file_included(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / "vllm/router.py"
            p.parent.mkdir()
            p.write_text("def select_experts(x):\n return x\n")
            result = review(root, extra_paths=["vllm/router.py"])
        self.assertTrue(next(r for r in result["files"] if r["path"] == "vllm/router.py")["present"])


if __name__ == "__main__":
    unittest.main()
