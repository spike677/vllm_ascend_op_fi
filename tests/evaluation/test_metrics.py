from __future__ import annotations

import math
import unittest

from ascend_fi.evaluation.metrics import (
    aggregate_per_rank,
    aggregate_repeatability,
    extract_gsm8k_answer,
    gsm8k_accuracy,
    safe_ratio,
    token_divergence,
    word_perplexity,
)


class EffectMetricTests(unittest.TestCase):
    def test_token_divergence_includes_length_delta(self) -> None:
        self.assertEqual(
            token_divergence((1, 2, 3), (1, 9)),
            {"first": 1, "count": 2, "rate": 2 / 3},
        )
        self.assertEqual(
            token_divergence(("a",), ("a",)),
            {"first": None, "count": 0, "rate": 0.0},
        )
        self.assertIsNone(token_divergence(None, (1,))["rate"])

    def test_gsm8k_parser_is_deterministic(self) -> None:
        self.assertEqual(extract_gsm8k_answer("work\n#### 1,024"), "1024")
        self.assertEqual(extract_gsm8k_answer("The final number is -2.50."), "-2.5")
        self.assertEqual(gsm8k_accuracy(("#### 3", "#### 4"), ("3", "5")), 0.5)

    def test_ratios_and_perplexity_are_safe(self) -> None:
        self.assertEqual(safe_ratio(3, 2), 1.5)
        self.assertIsNone(safe_ratio(1, 0))
        self.assertAlmostEqual(word_perplexity(math.log(4) * 2, 2), 4.0)
        self.assertIsNone(word_perplexity(1.0, 0))

    def test_per_rank_expected_reject_is_not_a_failure(self) -> None:
        result = aggregate_per_rank(
            {
                "candidate_count_per_device": [3, 0],
                "hit_count_per_device": [3, 0],
                "last_status_per_device": [1, 0],
            },
            {0: 3, 1: 3},
            {0: 100, 1: 101},
            {0: True, 1: True},
            expected_workers=2,
            target_device_mask=1,
        )
        self.assertEqual(result["observed_worker_count"], 2)
        self.assertEqual(result["ranks"]["0"]["hit_count"], 3)
        self.assertTrue(result["ranks"]["1"]["expected_reject"])
        self.assertEqual(result["ranks"]["1"]["reject_reasons"], {"device_selector_expected": 3})

    def test_repeatability_aggregation(self) -> None:
        result = aggregate_repeatability(
            [
                {"resolved": True, "hit": True, "effect_observed": True, "recovery_exact": True, "crash": False},
                {"resolved": True, "hit": True, "effect_observed": False, "recovery_exact": True, "crash": False},
                {"resolved": False, "hit": False, "effect_observed": False, "recovery_exact": False, "crash": True},
            ]
        )
        self.assertEqual(result["target_resolve_rate"], 2 / 3)
        self.assertEqual(result["effect_repeatability"], 1 / 3)
        self.assertEqual(result["exact_recovery_rate"], 2 / 3)
        self.assertEqual(result["crash_rate"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
