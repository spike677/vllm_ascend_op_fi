from __future__ import annotations

import unittest
from pathlib import Path

from ascend_fi.contracts import EvaluationStatus
from ascend_fi.discovery import (
    TraceContext,
    adapt_qmm_jsonl,
    load_capabilities,
    resolve_and_rank,
)
from ascend_fi.evaluation import (
    EvaluationRunner,
    RecordedEvaluationExecutor,
    obvious_gibberish,
)
from ascend_fi.validation import RecordedLosslessExecutor, validate_target


REPO = Path(__file__).resolve().parents[2]


def candidate_and_validation():
    fact = adapt_qmm_jsonl(
        REPO / "tests/fixtures/qmm_run_trace.json",
        TraceContext(
            phase="decode",
            rank=0,
            model="Qwen3-32B-w8a8-nopdmix",
            topology="tensor_parallel_2",
        ),
    )[0]
    candidate = resolve_and_rank(
        fact, load_capabilities(REPO / "profiles/capabilities")
    )[0].candidate
    validation = validate_target(
        candidate,
        RecordedLosslessExecutor(
            REPO / "tests/fixtures/qmm_lossless_validation.json"
        ),
    )
    return candidate, validation


class EvaluationTests(unittest.TestCase):
    def test_recorded_baseline_fault_recovery_evaluation(self) -> None:
        candidate, validation = candidate_and_validation()
        executor = RecordedEvaluationExecutor(
            REPO / "tests/fixtures/qmm_evaluation.json"
        )
        result = EvaluationRunner().evaluate(
            "qmm-recorded-e2e-v0", candidate, validation, executor
        )
        self.assertEqual(result.status, EvaluationStatus.PASS)
        self.assertEqual(
            executor.calls, ["baseline", "fault", "disarm", "recovery"]
        )
        self.assertEqual(result.model_effect.first_divergent_token, 1)
        self.assertFalse(result.model_effect.response_equal)
        self.assertFalse(result.model_effect.token_equal)
        self.assertEqual(
            result.model_effect.extra["token_comparison_basis"], "token_pieces"
        )
        self.assertTrue(result.model_effect.extra["obvious_gibberish"])
        self.assertTrue(result.model_effect.extra["recovery_response_equal"])
        self.assertIsNone(result.baseline.token_ids)
        self.assertIsNone(result.quality_performance.ppl)
        self.assertEqual(type(result).from_json(result.to_json()), result)

    def test_failed_validation_prevents_fault_execution(self) -> None:
        candidate, validation = candidate_and_validation()
        object.__setattr__(validation, "restore_exact", False)
        executor = RecordedEvaluationExecutor(
            REPO / "tests/fixtures/qmm_evaluation.json"
        )
        result = EvaluationRunner().evaluate(
            "not-authorized", candidate, validation, executor
        )
        self.assertEqual(result.status, EvaluationStatus.FAIL)
        self.assertEqual(executor.calls, [])

    def test_gibberish_rule_is_deterministic(self) -> None:
        self.assertFalse(obvious_gibberish("The capital of France is Paris."))
        self.assertTrue(obvious_gibberish("Latin中文한글العربية"))

    def test_fault_exception_still_disarms(self) -> None:
        candidate, validation = candidate_and_validation()

        class FailingExecutor:
            def __init__(self):
                self.calls = []

            def run_baseline(self, candidate):
                self.calls.append("baseline")
                return RecordedEvaluationExecutor(
                    REPO / "tests/fixtures/qmm_evaluation.json"
                )._outcome("baseline")

            def run_fault(self, candidate):
                self.calls.append("fault")
                raise RuntimeError("simulated backend failure")

            def disarm(self, candidate):
                self.calls.append("disarm")
                return True

            def run_recovery(self, candidate):
                self.calls.append("recovery")
                raise AssertionError("recovery must not run after fault exception")

        executor = FailingExecutor()
        result = EvaluationRunner().evaluate(
            "fault-exception", candidate, validation, executor
        )
        self.assertEqual(result.status, EvaluationStatus.ERROR)
        self.assertEqual(executor.calls, ["baseline", "fault", "disarm"])


if __name__ == "__main__":
    unittest.main()
