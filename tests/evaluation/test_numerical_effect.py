import json
import math
import unittest

from ascend_fi.evaluation.numerical_effect import derive_sampled_energy, sampled_numerical_effect


def row(rank, name, error, reference, elements=4, calls=1):
    return {"rank": rank, "module": name, "layer": 0,
            "sampled_error_sq_sum": error, "sampled_input_sq_sum": reference,
            "sampled_elements": elements, "sampled_calls": calls}


def worker(rank, rows):
    return {"rank": rank, "effect_summary": {"schema": "BOUNDED_EFFECT_SUMMARY_V1",
        "contiguous_hit_indices": True, "errors": [],
        "per_module": {r["module"]: r for r in rows}}}


class NumericalEffectTests(unittest.TestCase):
    def derive(self, error, reference, **kwargs):
        return derive_sampled_energy(error, reference, sampled_elements=kwargs.get("elements", 4),
                                     sampled_calls=kwargs.get("calls", 1))

    def test_known_vectors(self):
        before, after = [1., 2., 3., 4.], [1., 2., 2., 4.]
        error = sum((a - b) ** 2 for a, b in zip(before, after))
        energy = sum(a * a for a in before)
        value = self.derive(error, energy)
        self.assertAlmostEqual(value["nmse"], 1 / 30)
        self.assertAlmostEqual(value["relative_l2"], math.sqrt(1 / 30))
        self.assertAlmostEqual(value["sqnr_db"], 10 * math.log10(30))

    def test_zero_error_has_explicit_json_infinity(self):
        value = self.derive(0, 10)
        self.assertEqual(value["status"], "ZERO_ERROR")
        self.assertEqual(value["nmse"], 0)
        self.assertIsNone(value["sqnr_db"])
        self.assertEqual(value["sqnr_status"], "POSITIVE_INFINITY")
        json.dumps(value, allow_nan=False)

    def test_zero_reference_not_normal_zero(self):
        for error in (0, 2):
            value = self.derive(error, 0)
            self.assertEqual(value["status"], "ZERO_REFERENCE_ENERGY")
            self.assertIsNone(value["nmse"])
            json.dumps(value, allow_nan=False)

    def test_missing_and_unsampled(self):
        self.assertEqual(self.derive(None, 3)["status"], "MISSING")
        self.assertEqual(self.derive(0, 0, elements=0, calls=0)["status"], "NO_SAMPLES")
        self.assertEqual(self.derive(1, 3, elements=0)["status"], "INCONSISTENT_COUNTS")
        self.assertEqual(self.derive(1, 3, elements=1, calls=2)["status"], "INCONSISTENT_COUNTS")

    def test_nonfinite_negative_and_bool_rejected(self):
        for value in (math.inf, -math.inf, math.nan):
            derived = self.derive(value, 10)
            self.assertEqual(derived["status"], "NONFINITE_UPSTREAM")
            json.dumps(derived, allow_nan=False)
        self.assertEqual(self.derive(-1, 10)["status"], "INVALID_ENERGY")
        self.assertEqual(self.derive(True, 10)["status"], "INVALID_ENERGY")
        self.assertEqual(self.derive(1, 10, calls=True)["status"], "INVALID_COUNTS")

    def test_ratio_extremes_keep_log_sqnr(self):
        for error, reference, state in ((1e308, 1e-308, "OVERFLOW"), (1e-308, 1e308, "UNDERFLOW")):
            value = self.derive(error, reference)
            self.assertIsNone(value["nmse"])
            self.assertEqual(value["nmse_status"], state)
            self.assertTrue(math.isfinite(value["sqnr_db"]))
            self.assertIsNotNone(value["relative_l2"])
            json.dumps(value, allow_nan=False)

    def test_aggregate_energies_not_db_or_module_averages(self):
        workers = [worker(4, [row(4, "same", 1, 10), row(4, "other", 9, 90)]),
                   worker(9, [row(9, "same", 10, 900)])]
        result = sampled_numerical_effect(workers, expected_ranks=[4, 9], request_generation=7)
        self.assertAlmostEqual(result["aggregate"]["nmse"], 20 / 1000)
        self.assertAlmostEqual(result["aggregate"]["sqnr_db"], 10 * math.log10(50))
        self.assertEqual(result["aggregate"]["sampled_elements"], 12)
        self.assertEqual(set(result["per_rank"]), {"4", "9"})
        self.assertEqual(result["request_generation"], 7)

    def test_missing_rank_is_not_zero(self):
        result = sampled_numerical_effect([worker(4, [row(4, "x", 1, 2)])], expected_ranks=[4, 9])
        self.assertEqual(result["status"], "INCOMPLETE_INPUT")
        self.assertIsNone(result["aggregate"]["nmse"])

    def test_duplicate_rank_and_bad_metadata(self):
        good = worker(4, [row(4, "x", 1, 2)])
        self.assertEqual(sampled_numerical_effect([good, good])["status"], "INCOMPLETE_INPUT")
        self.assertEqual(sampled_numerical_effect([worker(4, [row(9, "x", 1, 2)])])["status"], "INCOMPLETE_INPUT")
        with self.assertRaises(ValueError):
            sampled_numerical_effect([], expected_ranks=[4, 4])

    def test_old_only_rounded_l2_not_reconstructed(self):
        old = row(4, "x", 1, 2)
        del old["sampled_error_sq_sum"]
        old["relative_l2"] = 0.71
        result = sampled_numerical_effect([worker(4, [old])])
        self.assertEqual(result["status"], "INCOMPLETE_INPUT")
        self.assertIsNone(result["aggregate"]["nmse"])

    def test_bad_module_never_silently_dropped(self):
        result = sampled_numerical_effect([worker(4, [row(4, "a", 1, 2), row(4, "b", math.nan, 3)])])
        self.assertEqual(result["status"], "INCOMPLETE_INPUT")
        self.assertEqual(result["per_rank"]["4"]["per_module"]["b"]["status"], "NONFINITE_UPSTREAM")
        json.dumps(result, allow_nan=False)

    def test_cpu_torch_formula_when_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable; deterministic scalar formulas still tested")
        before = torch.tensor([1., 2., 3., 4.], dtype=torch.float32)
        after = torch.tensor([1., 2., 2., 4.], dtype=torch.float32)
        value = self.derive((before - after).square().sum().item(), before.square().sum().item())
        self.assertAlmostEqual(value["relative_l2"], (torch.linalg.vector_norm(before - after) / torch.linalg.vector_norm(before)).item(), places=6)
        self.assertAlmostEqual(value["sqnr_db"], (20 * torch.log10(torch.linalg.vector_norm(before) / torch.linalg.vector_norm(before - after))).item(), places=5)


if __name__ == "__main__":
    unittest.main()
