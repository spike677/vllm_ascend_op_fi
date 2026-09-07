from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ascend_fi.contracts import (  # noqa: E402
    Candidate,
    CandidateBinding,
    CandidateProof,
    CandidateRecommendation,
    CandidateStatus,
    CandidateValidation,
    Capability,
    EvaluationResult,
    EvaluationStatus,
    ExecutionPlane,
    MechanismMetrics,
    ModelEffectMetrics,
    ObservedTarget,
    ProofLevel,
    QualityPerformanceMetrics,
    ReliabilityMetrics,
    RunRecord,
    ValidationResult,
    ValidationStatus,
    load_capability_manifest,
)


class CapabilityManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest_paths = sorted((REPO_ROOT / "profiles" / "capabilities").glob("*.json"))

    def test_all_manifests_load_and_references_exist(self) -> None:
        self.assertEqual(5, len(self.manifest_paths))
        ids: set[str] = set()
        for path in self.manifest_paths:
            capability = load_capability_manifest(path)
            self.assertEqual(0, capability.schema_version)
            self.assertNotIn(capability.id, ids)
            ids.add(capability.id)
            self.assertTrue(capability.semantic_mapping.candidate_positions)
            for group in (
                capability.canonical.source,
                capability.canonical.build,
                capability.canonical.evidence,
            ):
                self.assertTrue(group)
                for reference in group:
                    self.assertTrue((REPO_ROOT / reference).exists(), reference)

    def test_manifest_unknown_and_optional_fields_are_compatible(self) -> None:
        raw = json.loads(self.manifest_paths[0].read_text(encoding="utf-8"))
        raw["future_top_level"] = {"version": 1}
        raw["supports"]["future_support"] = True
        raw.pop("compatibility", None)
        capability = Capability.from_dict(raw)
        self.assertEqual({"version": 1}, capability.extra["future_top_level"])
        self.assertTrue(capability.supports.extra["future_support"])
        self.assertEqual(capability, Capability.from_json(capability.to_json()))

    def test_fault41_is_not_advertised_as_a_capability(self) -> None:
        self.assertFalse(any("fault41" in path.name.lower() for path in self.manifest_paths))


class ContractRoundTripTests(unittest.TestCase):
    def test_capability_round_trip(self) -> None:
        capability = load_capability_manifest(
            REPO_ROOT
            / "profiles"
            / "capabilities"
            / "op_api_quant_matmul_input_buffer_transform.json"
        )
        self.assertEqual(capability, Capability.from_json(capability.to_json()))

    def test_candidate_round_trip(self) -> None:
        candidate = Candidate(
            candidate_id="candidate:qmm:x1:rank0",
            capability_id="op_api.quant_matmul.input_buffer_transform",
            observed=ObservedTarget(
                execution_plane=ExecutionPlane.OP_API,
                symbol="aclnnQuantMatmulWeightNz",
                rank=0,
                phase="decode",
                shape=(1, 5120),
                dtype="INT8",
                stream="0x1234",
                call_fingerprint="sha256:example",
            ),
            binding=CandidateBinding(
                tensor_role="x1",
                runtime_slot=0,
                storage_role="dynamic_activation",
            ),
            proof=CandidateProof(
                level=ProofLevel.SEMANTIC_SLOT_PROVEN,
                evidence_refs=("evidence/qmm-slot.json",),
            ),
            validation=CandidateValidation(status=CandidateStatus.UNKNOWN),
            recommended=CandidateRecommendation(
                strategy="int4_equivalent",
                params={"max_hits": 1},
            ),
        )
        self.assertEqual(candidate, Candidate.from_json(candidate.to_json()))

    def test_validation_result_round_trip_and_gate(self) -> None:
        unknown = ValidationResult(
            candidate_id="candidate:qmm:x1:rank0",
            resolved=False,
            lossless_attempted=False,
            lossless_passed=False,
        )
        self.assertFalse(unknown.destructive_injection_allowed)
        validated = ValidationResult(
            candidate_id="candidate:qmm:x1:rank0",
            resolved=True,
            lossless_attempted=True,
            lossless_passed=True,
            hit_count=1,
            before_hash="a",
            substituted_hash="a",
            restored_hash="a",
            model_output_equal=True,
            restore_exact=True,
            status=ValidationStatus.LOSSLESS_VALIDATED,
            evidence_refs=("evidence/lossless.json",),
        )
        self.assertTrue(validated.destructive_injection_allowed)
        self.assertEqual(validated, ValidationResult.from_json(validated.to_json()))

    def test_evaluation_result_round_trip_with_null_reservations(self) -> None:
        evaluation = EvaluationResult(
            evaluation_id="eval:qmm:int4:1",
            capability_id="op_api.quant_matmul.input_buffer_transform",
            candidate_id="candidate:qmm:x1:rank0",
            baseline=RunRecord(http_status=200, response_hash="baseline", token_ids=(1, 2)),
            fault=RunRecord(http_status=200, response_hash="fault", token_ids=(1, 3)),
            recovery=RunRecord(http_status=200, response_hash="baseline", token_ids=(1, 2)),
            mechanism=MechanismMetrics(
                resolved=True,
                candidate_count=1,
                hit_count=1,
                reject_count=0,
                changed_elements=12,
                restore_exact=True,
                auto_disarmed=True,
            ),
            model_effect=ModelEffectMetrics(
                http_status=200,
                response_equal=False,
                response_hash="fault",
                token_equal=False,
                first_divergent_token=1,
                generation_length=2,
                generation_length_delta=0,
                crash=False,
                worker_alive=True,
            ),
            reliability=ReliabilityMetrics(
                target_resolve_rate=1.0,
                injection_hit_rate=1.0,
                recovery_rate=1.0,
            ),
            quality_performance=QualityPerformanceMetrics(),
            status=EvaluationStatus.PASS,
        )
        restored = EvaluationResult.from_json(evaluation.to_json())
        self.assertEqual(evaluation, restored)
        self.assertIsNone(restored.quality_performance.ppl)
        self.assertIsNone(restored.quality_performance.ttft)


if __name__ == "__main__":
    unittest.main()
