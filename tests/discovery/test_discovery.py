from __future__ import annotations

import unittest
from pathlib import Path

from ascend_fi.contracts import CandidateStatus, ProofLevel
from ascend_fi.discovery import (
    Confidence,
    TraceContext,
    adapt_qmm_jsonl,
    load_capabilities,
    resolve_and_rank,
)


REPO = Path(__file__).resolve().parents[2]


class QmmDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = TraceContext(
            phase="decode",
            rank=0,
            model="Qwen3-32B-w8a8-nopdmix",
            topology="tensor_parallel_2",
            runtime={"cann": "8.5.1", "vllm": "0.18.0"},
            evidence_ref="tests/fixtures/qmm_run_trace.json:2",
        )
        self.fact = adapt_qmm_jsonl(
            REPO / "tests/fixtures/qmm_run_trace.json", self.context
        )[0]
        self.capabilities = load_capabilities(REPO / "profiles/capabilities")

    def test_existing_qmm_trace_adapts_to_observed_fact(self) -> None:
        self.assertEqual(self.fact.symbol, "aclnnQuantMatmulWeightNz")
        self.assertEqual(self.fact.shape, (1, 5120))
        self.assertEqual(self.fact.dtype, "int8")
        self.assertEqual(self.fact.runtime_slots["x1"], 0)
        self.assertTrue(self.fact.proof["semantic_slot_proven"])

    def test_qmm_input_capability_is_best_high_confidence_candidate(self) -> None:
        best = resolve_and_rank(self.fact, self.capabilities)[0]
        self.assertEqual(
            best.candidate.capability_id,
            "op_api.quant_matmul.input_buffer_transform",
        )
        self.assertEqual(
            best.candidate.validation.status, CandidateStatus.ACCEPTED
        )
        self.assertEqual(
            best.candidate.proof.level, ProofLevel.SEMANTIC_SLOT_PROVEN
        )
        self.assertEqual(best.candidate.binding.runtime_slot, 0)
        self.assertEqual(best.candidate.binding.tensor_role, "input_activation")
        self.assertEqual(best.confidence, Confidence.HIGH)
        self.assertTrue(best.auto_validation_allowed)

    def test_unknown_symbol_fails_closed(self) -> None:
        object.__setattr__(self.fact, "symbol", "aclnnUnknownQmm")
        self.assertEqual(resolve_and_rank(self.fact, self.capabilities), ())

    def test_missing_semantic_proof_is_not_auto_validated(self) -> None:
        weak = dict(self.fact.proof)
        weak["semantic_slot_proven"] = False
        weak["x1_slot_proven"] = False
        object.__setattr__(self.fact, "proof", weak)
        ranked = resolve_and_rank(self.fact, self.capabilities)
        qmm = next(
            item
            for item in ranked
            if item.candidate.capability_id
            == "op_api.quant_matmul.input_buffer_transform"
        )
        self.assertEqual(
            qmm.candidate.validation.status, CandidateStatus.REJECTED
        )
        self.assertFalse(qmm.auto_validation_allowed)

    def test_candidate_json_round_trip(self) -> None:
        candidate = resolve_and_rank(self.fact, self.capabilities)[0].candidate
        self.assertEqual(type(candidate).from_json(candidate.to_json()), candidate)


if __name__ == "__main__":
    unittest.main()
