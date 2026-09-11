import copy
import json
import math
import unittest

from ascend_fi.evaluation.reference_probability import (
    attach_reference_probability, compare_echo_reference, parse_echo_reference,
)
from tests.evaluation.test_effect_profile import evaluation


def reference():
    return {"model_identity": "FIXTURE_MODEL", "tokenizer_identity": "FIXTURE_TOKENIZER",
            "template_identity": "raw-token-echo", "reference_sha256": "FIXTURE_REFERENCE_HASH",
            "window_identity": "window0:[0,3)", "context_policy": "all preceding reference IDs in this window",
            "logprob_semantics": "unprocessed natural-log reference-token conditional probability",
            "cache_policy": "no shared B/F response cache; prefix cache disabled",
            "token_ids": [1, 10, 11], "score_mask": [False, True, True]}


def record(name="baseline", values=None):
    ref = reference()
    return {"request_id": name, "reference": ref, "http_status": 200,
            "request": {"model": "fixture", "echo": True, "prompt": ref["token_ids"],
                        "logprobs": 1, "max_tokens": 1, "temperature": 0},
            "raw_response": {"usage": {"prompt_tokens": 3}, "choices": [{
                "logprobs": {"token_ids": [1, 10, 11, 12],
                             "token_logprobs": [None, -1., -2., -99.] if values is None else values}}]}}


class ReferenceProbabilityTests(unittest.TestCase):
    def test_reference_tokens_only_not_generated_logprob(self):
        result = parse_echo_reference(reference(), record())
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["scored_token_count"], 2)
        self.assertEqual(result["sum_nll"], 3)
        self.assertEqual(result["mean_nll"], 1.5)
        self.assertAlmostEqual(result["ppl"], math.exp(1.5))

    def test_paired_delta_ratio_and_no_live_claim(self):
        result = compare_echo_reference(reference(), record(), record("fault", [None, -2., -3., 0]))
        self.assertEqual(result["mean_nll_delta"], 1)
        self.assertAlmostEqual(result["ppl_ratio"], math.e)
        self.assertEqual(result["probability_transaction_gate"], "NOT_IMPLEMENTED")

    def test_all_missing_or_one_missing_never_dropped(self):
        for values in ([None, None, None], [None, -1, None]):
            result = parse_echo_reference(reference(), record(values=values))
            self.assertEqual(result["status"], "UNSCORABLE")
            self.assertIsNone(result["mean_nll"])
            self.assertTrue(result["missing_scored_positions"])

    def test_finite_topk_does_not_fill_missing_target_logprob(self):
        row = record(values=[None, None, -1])
        row["raw_response"]["choices"][0]["logprobs"]["top_logprobs"] = [{"x": -.1}] * 3
        self.assertEqual(parse_echo_reference(reference(), row)["reason"], "MISSING_SCORED_REFERENCE_LOGPROB")

    def test_nonfinite_and_invalid_probability(self):
        for value in (math.nan, math.inf, .01, True, "-1"):
            result = parse_echo_reference(reference(), record(values=[None, value, -1]))
            self.assertEqual(result["status"], "UNSCORABLE")
            json.dumps(result, allow_nan=False)

    def test_negative_infinite_reference_probability_is_explicit(self):
        result = parse_echo_reference(reference(), record(values=[None, -math.inf, -1]))
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["sum_nll_status"], "POSITIVE_INFINITY")
        self.assertIsNone(result["ppl"])
        self.assertEqual(result["ppl_status"], "POSITIVE_INFINITY")
        json.dumps(result, allow_nan=False)

    def test_both_infinite_pair_not_ratio_one(self):
        b, f = record(values=[None, -math.inf, -1]), record("fault", [None, -math.inf, -1])
        result = compare_echo_reference(reference(), b, f)
        self.assertEqual(result["ppl_ratio_status"], "UNDEFINED_INFINITY_OVER_INFINITY")
        self.assertIsNone(result["ppl_ratio"])

    def test_ppl_overflow_but_log_ratio_finite(self):
        b, f = record(values=[None, -1000, -1000]), record("fault", [None, -1001, -1001])
        result = compare_echo_reference(reference(), b, f)
        self.assertEqual(result["baseline"]["ppl_status"], "OVERFLOW")
        self.assertAlmostEqual(result["ppl_ratio"], math.e)
        json.dumps(result, allow_nan=False)

    def test_sum_overflow_preserves_finite_mean(self):
        result = parse_echo_reference(reference(), record(values=[None, -1e308, -1e308]))
        self.assertEqual(result["sum_nll_status"], "OVERFLOW")
        self.assertEqual(result["mean_nll"], 1e308)
        json.dumps(result, allow_nan=False)

    def test_token_ids_mask_window_context_must_match(self):
        for name in ("tokenizer_identity", "window_identity", "context_policy", "logprob_semantics", "cache_policy"):
            row = record()
            row["reference"][name] = "changed"
            self.assertEqual(parse_echo_reference(reference(), row)["reason"], "CAPTURED_REFERENCE_IDENTITY_MISMATCH")
        row = record()
        row["raw_response"]["choices"][0]["logprobs"]["token_ids"] = [1, 11, 10]
        self.assertEqual(parse_echo_reference(reference(), row)["reason"], "RETURNED_REFERENCE_TOKEN_IDS_UNPROVEN")

    def test_empty_mask_bos_and_integer_masks_rejected(self):
        for mask, reason in (([False] * 3, "NO_SCORED_TOKENS"),
                             ([True] * 3, "FIRST_TOKEN_HAS_NO_PROVEN_CONTEXT"),
                             ([0, 1, 1], "SCORE_MASK_INVALID")):
            ref, row = reference(), record()
            ref["score_mask"] = mask
            row["reference"] = ref
            self.assertEqual(parse_echo_reference(ref, row)["reason"], reason)

    def test_legacy_projection_not_sufficient_for_alignment(self):
        row = {"http_status": 200, "token_logprobs": [None, -1, -2], "prompt_tokens": 3}
        self.assertEqual(parse_echo_reference(reference(), row)["status"], "UNSCORABLE")
        row = record()
        del row["raw_response"]["choices"][0]["logprobs"]["token_ids"]
        self.assertEqual(parse_echo_reference(reference(), row)["reason"], "RETURNED_REFERENCE_TOKEN_IDS_UNPROVEN")

    def test_shared_response_cache_and_generation_mismatch_rejected(self):
        b, f = record(), record("fault")
        b["response_cache_key"] = f["response_cache_key"] = "same"
        self.assertEqual(compare_echo_reference(reference(), b, f)["reason"], "BASELINE_FAULT_RESPONSE_CACHE_KEY_REUSED")
        b, f = record(), record("fault")
        f["request"]["temperature"] = 1
        self.assertEqual(compare_echo_reference(reference(), b, f)["reason"], "BASELINE_FAULT_REQUEST_CONFIGURATION_MISMATCH")

    def test_helper_extends_existing_result_without_promoting_quality(self):
        original = evaluation()
        attached = attach_reference_probability(original, reference(), record(), record("fault"))
        self.assertEqual(attached.quality_performance, original.quality_performance)
        self.assertEqual(attached.status, original.status)
        self.assertNotIn("reference_probability_offline", original.extra)
        self.assertIn("reference_probability_offline", attached.extra)
        json.dumps(attached.to_dict(), allow_nan=False)
        with self.assertRaises(ValueError):
            attach_reference_probability(attached, reference(), record(), record("fault"))


if __name__ == "__main__":
    unittest.main()
