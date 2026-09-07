from __future__ import annotations

import unittest
from pathlib import Path

from ascend_fi.contracts import CandidateStatus, ValidationStatus
from ascend_fi.discovery import TraceContext, adapt_qmm_jsonl, load_capabilities, resolve_and_rank
from ascend_fi.validation import LosslessEvidence, RecordedLosslessExecutor, validate_target


REPO = Path(__file__).resolve().parents[2]


def resolved_candidate():
    fact = adapt_qmm_jsonl(
        REPO / "tests/fixtures/qmm_run_trace.json",
        TraceContext(
            phase="decode",
            rank=0,
            model="Qwen3-32B-w8a8-nopdmix",
            topology="tensor_parallel_2",
        ),
    )[0]
    return resolve_and_rank(
        fact, load_capabilities(REPO / "profiles/capabilities")
    )[0].candidate


class CountingExecutor:
    def __init__(self, evidence: LosslessEvidence):
        self.evidence = evidence
        self.calls = 0

    def execute_lossless(self, candidate):
        self.calls += 1
        return self.evidence


class ValidationTests(unittest.TestCase):
    def test_recorded_qmm_lossless_gate_passes(self) -> None:
        result = validate_target(
            resolved_candidate(),
            RecordedLosslessExecutor(
                REPO / "tests/fixtures/qmm_lossless_validation.json"
            ),
        )
        self.assertEqual(result.status, ValidationStatus.LOSSLESS_VALIDATED)
        self.assertTrue(result.lossless_passed)
        self.assertTrue(result.restore_exact)
        self.assertTrue(result.destructive_injection_allowed)
        self.assertIsNone(result.before_hash)
        self.assertTrue(result.extra["original_storage_untouched"])
        self.assertEqual(type(result).from_json(result.to_json()), result)

    def test_rejected_candidate_does_not_call_executor(self) -> None:
        candidate = resolved_candidate()
        object.__setattr__(candidate.validation, "status", CandidateStatus.REJECTED)
        executor = CountingExecutor(
            LosslessEvidence(True, True, True, 1, model_output_equal=True)
        )
        result = validate_target(candidate, executor)
        self.assertEqual(result.status, ValidationStatus.REJECTED)
        self.assertEqual(executor.calls, 0)

    def test_hash_mismatch_fails_closed(self) -> None:
        executor = CountingExecutor(
            LosslessEvidence(
                target_resolved=True,
                expected_api_slot_hit=True,
                substitution_executed=True,
                hit_count=1,
                before_hash="before",
                restored_hash="different",
                model_output_equal=True,
                original_storage_untouched=False,
                run_status=0,
                copy_status=0,
                sync_status=0,
                temporary_freed=True,
            )
        )
        result = validate_target(resolved_candidate(), executor)
        self.assertEqual(result.status, ValidationStatus.FAILED)
        self.assertFalse(result.destructive_injection_allowed)


if __name__ == "__main__":
    unittest.main()
