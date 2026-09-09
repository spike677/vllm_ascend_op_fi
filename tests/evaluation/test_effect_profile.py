from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ascend_fi.contracts import (  # noqa: E402
    EvaluationResult,
    EvaluationStatus,
    MechanismMetrics,
    ModelEffectMetrics,
    QualityPerformanceMetrics,
    ReliabilityMetrics,
)
from ascend_fi.evaluation.effect_profile import (  # noqa: E402
    EffectProfilePolicy,
    EffectStage,
    PrimaryPhenotype,
    aggregate_paired_effects,
    build_fault_effect_profile,
    classify_paired_request,
    write_effect_card,
)
from ascend_fi.evaluation.replay import replay_completed_effect_cycles  # noqa: E402


def evaluation(*, ppl_ratio: float | None = None) -> EvaluationResult:
    return EvaluationResult(
        evaluation_id="eval:fake-requant:unit",
        capability_id="python.fake_requant.activation",
        candidate_id="candidate:qwen:rank0",
        baseline=None,
        fault=None,
        recovery=None,
        mechanism=MechanismMetrics(resolved=True, candidate_count=4, hit_count=4, changed_elements=64, restore_exact=True),
        model_effect=ModelEffectMetrics(response_equal=False, token_equal=False),
        reliability=ReliabilityMetrics(target_resolve_rate=1.0, injection_hit_rate=1.0, recovery_rate=1.0),
        quality_performance=QualityPerformanceMetrics(ppl_ratio=ppl_ratio),
        status=EvaluationStatus.PASS,
    )


def pair(sample_id: str, baseline: dict, fault: dict, **extra: object) -> dict:
    return {"sample_id": sample_id, "scope": "gsm8k", "baseline": baseline, "fault": fault, "mechanism_valid": True, **extra}


class EffectProfileTests(unittest.TestCase):
    def test_divergence_is_not_visible_degradation(self) -> None:
        record = classify_paired_request(
            pair(
                "g1",
                {"response": "4", "correct": True, "resolved": True},
                {"response": "The answer is 4.", "correct": True, "resolved": True},
            )
        )
        self.assertEqual(PrimaryPhenotype.DIVERGED_BUT_VALID.value, record["primary_phenotype"])
        self.assertTrue(record["response_diverged"])
        self.assertFalse(record["visible_degradation"])

    def test_visible_damage_is_baseline_relative_and_deterministic(self) -> None:
        records = aggregate_paired_effects(
            [
                pair("ok", {"correct": True, "resolved": True, "response": "1"}, {"correct": True, "resolved": True, "response": "1"}),
                pair("wrong", {"correct": True, "resolved": True, "response": "2"}, {"correct": False, "resolved": True, "response": "3"}),
                pair("unresolved", {"resolved": True, "response": "2"}, {"resolved": False, "response": ""}),
                pair("invalid", {"worker_alive": False}, {"response": "x"}, mechanism_valid=False),
            ]
        )
        overall = records["overall"]
        self.assertEqual(3, overall["valid_pair_count"])
        self.assertEqual(2, overall["visible_degradation_count"])
        self.assertEqual(2 / 3, overall["visible_degradation_rate"])
        self.assertEqual(1, overall["paired_damage_count"])
        self.assertEqual(1, overall["phenotype_counts"][PrimaryPhenotype.WRONG_ANSWER.value])
        self.assertEqual(1, overall["phenotype_counts"][PrimaryPhenotype.UNRESOLVED_ANSWER.value])

    def test_visible_damage_never_exceeds_response_divergence(self) -> None:
        aggregate = aggregate_paired_effects(
            [pair("error", {"http_status": 200}, {"http_status": 500})]
        )["overall"]
        self.assertEqual(1.0, aggregate["visible_degradation_rate"])
        self.assertEqual(1.0, aggregate["response_divergence_rate"])

    def test_stage_l2_for_current_fake_requant_pattern(self) -> None:
        profile = build_fault_effect_profile(
            evaluation(ppl_ratio=1.0655228418807177),
            [
                pair("short-1", {"response": "a"}, {"response": "b"}),
                pair("short-2", {"response": "a"}, {"response": "b"}),
                pair("short-3", {"response": "a"}, {"response": "b"}),
                pair("short-4", {"response": "a"}, {"response": "a"}),
            ],
        )
        self.assertEqual(EffectStage.L2_TRAJECTORY_DIVERGENCE.value, profile["effect_stage"])
        self.assertEqual(0.75, profile["output_effect"]["response_divergence_rate"])
        self.assertEqual(0.0, profile["visible_degradation"]["rate"])
        self.assertIn("REPEATABLE_TASK_DAMAGE_NOT_MEASURED", profile["limitations"])

    def test_stage_priority_visible_repeatable_pathological_crash(self) -> None:
        wrong = pair("wrong", {"correct": True, "resolved": True}, {"correct": False, "resolved": True})
        visible = build_fault_effect_profile(evaluation(), [wrong])
        self.assertEqual(EffectStage.L3_VISIBLE_DEGRADATION.value, visible["effect_stage"])
        repeatable = build_fault_effect_profile(
            evaluation(), [wrong], reliability={"repeatable_task_damage_cycles": 2}
        )
        self.assertEqual(EffectStage.L4_REPEATABLE_TASK_DAMAGE.value, repeatable["effect_stage"])
        pathological = build_fault_effect_profile(
            evaluation(),
            [pair("loop", {"periodic_loop": False}, {"periodic_loop": True})],
        )
        self.assertEqual(EffectStage.L5_PATHOLOGICAL_COLLAPSE.value, pathological["effect_stage"])
        crashed = build_fault_effect_profile(
            evaluation(),
            [pair("crash", {"worker_alive": True}, {"worker_alive": False})],
        )
        self.assertEqual(EffectStage.L6_SERVICE_FAILURE.value, crashed["effect_stage"])

    def test_profile_artifacts_are_derived_and_do_not_include_raw_responses(self) -> None:
        profile = build_fault_effect_profile(
            evaluation(),
            [pair("x", {"response": "secret baseline"}, {"response": "secret fault"})],
            policy=EffectProfilePolicy(trajectory_divergence_rate=1.0),
        )
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_effect_card(temporary, profile)
            self.assertTrue(all(path.exists() for path in paths.values()))
            stored = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual("FAULT_EFFECT_PROFILE_V1", stored["schema_version"])
            self.assertNotIn("secret baseline", paths["markdown"].read_text(encoding="utf-8"))
            self.assertIn("Fault Effect Dashboard", paths["dashboard"].read_text(encoding="utf-8"))

    def test_read_only_completed_cycle_replay_writes_sanitized_card(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "completed"
            cycle = root / "cycles" / "cycle_01"
            for phase in ("baseline", "fault", "recovery"):
                (cycle / phase).mkdir(parents=True, exist_ok=True)
            (root / "manifest.json").write_text(
                json.dumps({"protocol": "TEST", "wikitext": {"scored_token_count": 3}}),
                encoding="utf-8",
            )
            (cycle / "mechanism.json").write_text(
                json.dumps({"rank0_measurement_hits": [4096], "rank1_measurement_hits": [0]}),
                encoding="utf-8",
            )
            summary = {
                "cycle": 1,
                "service_health": True,
                "probability": {
                    "baseline": {"nll": 2.0, "ppl": 2.0},
                    "fault": {"nll": 3.0, "ppl": 2.2},
                    "recovery": {"nll": 2.0, "ppl": 2.0},
                    "nll_delta": 1.0,
                    "ppl_ratio": 1.1,
                },
                "short": {
                    "baseline_vs_fault": {"per_sample": [{"sample_id": "short", "response_equal": False}]},
                    "baseline_vs_recovery": {"per_sample": [{"sample_id": "short", "response_equal": True}]},
                },
                "long": {
                    "baseline_vs_fault": {"per_sample": [{"sample_id": "long", "response_equal": False}]},
                    "baseline_vs_recovery": {"per_sample": [{"sample_id": "long", "response_equal": True}]},
                },
            }
            (cycle / "cycle_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            rows = {
                "baseline": {"sample_id": "g", "correct": True, "predicted_answer": "1", "http_status": 200, "response_sha256": "a"},
                "fault": {"sample_id": "g", "correct": False, "predicted_answer": None, "http_status": 200, "response_sha256": "b"},
                "recovery": {"sample_id": "g", "correct": True, "predicted_answer": "1", "http_status": 200, "response_sha256": "a"},
            }
            for phase, row in rows.items():
                (cycle / phase / "gsm8k_responses.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            output = Path(temporary) / "derived"
            result = replay_completed_effect_cycles(root, output)
            # This fixture contains summary counters only, not raw numeric/rank
            # evidence. It stays readable but must no longer claim valid L3.
            self.assertEqual(EffectStage.UNCLASSIFIED.value, result["profile"]["effect_stage"])
            self.assertEqual(0, result["profile"]["visible_degradation"]["count"])
            self.assertTrue((output / "effect_card.json").exists())
            self.assertTrue((output / "replay_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
