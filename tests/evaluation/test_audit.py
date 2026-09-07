from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ascend_fi.evaluation.audit import (
    audit_gsm8k_baseline,
    classify_gsm8k_failure,
    summarize_smoke_phenotypes,
    write_baseline_audit_artifacts,
)
from ascend_fi.evaluation.metrics import (
    extract_explicit_final_numeric_answer,
    generation_hit_limit,
    normalize_numeric_answer,
)


class BaselineAuditTests(unittest.TestCase):
    def test_numeric_normalization_and_explicit_final_extraction(self) -> None:
        self.assertEqual(normalize_numeric_answer("$1,024"), "1024")
        self.assertEqual(extract_explicit_final_numeric_answer("work\n#### $1,024."), "1024")
        self.assertEqual(extract_explicit_final_numeric_answer("reasoning has 1024 but no answer"), None)

    def test_parser_false_negative_format_only_and_truncation_categories(self) -> None:
        parser_false_negative = classify_gsm8k_failure(
            {
                "reference_answer": "42",
                "strict_parser_result": None,
                "independent_answer": "42",
                "finish_reason": "stop",
                "generation_length": 8,
                "max_tokens": 128,
                "format_compliant": True,
                "response_text": "#### 42",
            }
        )
        self.assertEqual(parser_false_negative["primary"], "B_PARSER_FALSE_NEGATIVE")
        format_only = classify_gsm8k_failure(
            {
                "reference_answer": "42",
                "strict_parser_result": None,
                "independent_answer": "42",
                "finish_reason": "stop",
                "generation_length": 8,
                "max_tokens": 128,
                "format_compliant": False,
                "response_text": "FINAL_ANSWER=42",
            }
        )
        self.assertEqual(format_only["primary"], "E_OUTPUT_FORMAT_ONLY")
        self.assertTrue(generation_hit_limit("length", 128, 128))
        truncated = classify_gsm8k_failure(
            {
                "reference_answer": "42",
                "strict_parser_result": "4",
                "independent_answer": None,
                "finish_reason": "length",
                "generation_length": 128,
                "max_tokens": 128,
                "response_text": "Step 4",
            }
        )
        self.assertEqual(truncated["primary"], "C_MAX_TOKENS_TRUNCATED")

    def test_audit_and_artifact_writing(self) -> None:
        source = [{"question": "What is 40 + 2?", "answer": "#### 42"}]
        baseline = {
            "gsm8k": [
                {
                    "sample_id": "gsm8k-test-0000",
                    "index": 0,
                    "reference_answer": "42",
                    "predicted_answer": "4",
                    "response_text": "Step 4: continue",
                    "finish_reason": "length",
                    "generation_length": 128,
                    "token_ids": None,
                    "token_pieces": ["Step", " 4"],
                }
            ],
            "smoke": [
                {
                    "sample_id": "short",
                    "response_text": "42",
                    "token_ids": None,
                    "token_pieces": ["42"],
                    "generation_length": 1,
                },
                {
                    "sample_id": "long",
                    "response_text": "safe output",
                    "token_ids": None,
                    "token_pieces": ["safe", " output"],
                    "generation_length": 2,
                },
            ],
        }
        strong = {
            "smoke": [
                {
                    "sample_id": "short",
                    "response_text": "bad",
                    "token_ids": None,
                    "token_pieces": ["bad"],
                    "generation_length": 1,
                },
                {
                    "sample_id": "long",
                    "response_text": "loop loop loop loop loop loop",
                    "token_ids": None,
                    "token_pieces": ["loop"] * 6,
                    "generation_length": 6,
                },
            ]
        }
        smoke = [
            {"id": "short", "max_tokens": 4, "metadata": {"phenotype_class": "short_exact", "expected_regex": "^42$"}},
            {"id": "long", "max_tokens": 6, "metadata": {"phenotype_class": "long_generation"}},
        ]
        audit = audit_gsm8k_baseline(baseline["gsm8k"], source)
        self.assertEqual(audit["truncation_count"], 1)
        phenotype = summarize_smoke_phenotypes(baseline["smoke"], strong["smoke"], smoke)
        self.assertEqual(phenotype["short_output"]["format_violation_count"], 1)
        self.assertEqual(phenotype["long_output"]["periodic_loop_count"], 1)
        with tempfile.TemporaryDirectory() as directory:
            result = write_baseline_audit_artifacts(
                directory,
                baseline_phase=baseline,
                gsm8k_rows=source,
                strong_phase=strong,
                smoke_samples=smoke,
                evaluation_result={
                    "status": "PASS",
                    "phase_matrix": {
                        "baseline": {"wikitext": {"scored_token_count": 56, "word_perplexity": 3.0}},
                        "strong": {"wikitext": {"word_perplexity": 30.0}},
                    },
                },
            )
            self.assertEqual(result["audit"]["status"], "PASS")
            self.assertTrue((Path(directory) / "baseline_gsm8k_failure_audit.md").exists())
            self.assertEqual(
                json.loads((Path(directory) / "baseline_protocol_diff.json").read_text(encoding="utf-8"))[
                    "historical_reference_status"
                ],
                "UNVERIFIED_OBSERVATION_ONLY",
            )
            self.assertIn(
                "scored `56` reference tokens",
                (Path(directory) / "effect_evaluation_summary_audited.md").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
