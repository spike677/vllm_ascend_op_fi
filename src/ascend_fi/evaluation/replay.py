"""Read-only replay of completed baseline/fault/recovery artifact cycles.

The replayer consumes an already-finished evaluation directory.  It never
contacts a model endpoint, arm/disarms a fault, or writes into the input
artifact tree.  It emits only a sanitized Effect Profile and derived card.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

from ascend_fi.contracts import (
    EvaluationResult,
    EvaluationStatus,
    MechanismMetrics,
    ModelEffectMetrics,
    QualityPerformanceMetrics,
    ReliabilityMetrics,
)

from .effect_profile import EffectProfilePolicy, build_fault_effect_profile, write_effect_card


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not isinstance(value.get("sample_id"), str):
            raise ValueError(f"invalid response record at {path}:{line_number}")
        rows[value["sample_id"]] = value
    return rows


def _response_fact(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only fields required for deterministic offline classification."""

    return {
        "correct": row.get("correct"),
        "resolved": row.get("predicted_answer") is not None,
        "http_status": row.get("http_status"),
        "response_hash": row.get("response_sha256"),
        "worker_alive": row.get("worker_alive", True),
    }


def _comparison_pairs(
    cycle_id: str,
    scope: str,
    comparison: Mapping[str, Any],
    *,
    mechanism_valid: bool,
) -> list[dict[str, Any]]:
    rows = comparison.get("per_sample", [])
    if not isinstance(rows, list):
        raise ValueError(f"{cycle_id}/{scope}: per_sample must be a list")
    pairs = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{cycle_id}/{scope}: per_sample item must be an object")
        base = dict(row.get("baseline_features", {}))
        fault = dict(row.get("fault_features", {}))
        fault["response_equal"] = row.get("response_equal")
        pairs.append(
            {
                "sample_id": f"{cycle_id}:{row.get('sample_id', 'unknown')}",
                "scope": scope,
                "mechanism_valid": mechanism_valid,
                "baseline": base,
                "fault": fault,
            }
        )
    return pairs


def _gsm_pairs(cycle_dir: Path, cycle_id: str, *, mechanism_valid: bool) -> tuple[list[dict[str, Any]], bool]:
    baseline = _read_jsonl(cycle_dir / "baseline" / "gsm8k_responses.jsonl")
    fault = _read_jsonl(cycle_dir / "fault" / "gsm8k_responses.jsonl")
    recovery = _read_jsonl(cycle_dir / "recovery" / "gsm8k_responses.jsonl")
    if set(baseline) != set(fault) or set(baseline) != set(recovery):
        raise ValueError(f"{cycle_id}: GSM8K phase sample IDs differ")
    pairs = [
        {
            "sample_id": f"{cycle_id}:{sample_id}",
            "scope": "gsm8k",
            "mechanism_valid": mechanism_valid,
            "baseline": _response_fact(baseline[sample_id]),
            "fault": _response_fact(fault[sample_id]),
        }
        for sample_id in sorted(baseline)
    ]
    exact_recovery = all(
        baseline[sample_id].get("response_sha256") == recovery[sample_id].get("response_sha256")
        for sample_id in baseline
    )
    return pairs, exact_recovery


def _summary_recovery_exact(summary: Mapping[str, Any], group: str) -> bool:
    comparison = summary.get(group, {}).get("baseline_vs_recovery", {})
    rows = comparison.get("per_sample", []) if isinstance(comparison, Mapping) else []
    return isinstance(rows, list) and bool(rows) and all(
        isinstance(row, Mapping) and row.get("response_equal") is True for row in rows
    )


def replay_completed_effect_cycles(
    input_root: str | Path,
    output_dir: str | Path,
    *,
    evaluation_id: str = "fake_requant.effect_evaluation_v1_complete",
    capability_id: str = "python.fake_requant.activation",
    candidate_id: str = "qwen.mlp.down_proj.rank0",
    policy: EffectProfilePolicy | None = None,
) -> dict[str, Any]:
    """Replay completed cycles and write sanitized ``effect_card.*`` derivatives.

    ``input_root`` must have the completed evaluation layout: a ``cycles``
    directory containing ``cycle_*`` directories, each with ``cycle_summary``
    and baseline/fault/recovery GSM8K JSONL files.  Raw prompts and generated
    text are read only to derive hashes/facts, then are excluded from output.
    """

    root = Path(input_root)
    manifest = _read_json(root / "manifest.json")
    cycle_dirs = sorted(path for path in (root / "cycles").iterdir() if path.is_dir())
    if not cycle_dirs:
        raise ValueError(f"no cycle directories under {root}")
    pairs: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    probabilities: list[Mapping[str, Any]] = []
    all_hits: list[int] = []
    recovery_exact: list[bool] = []
    repeatable_damage_cycles = 0

    for cycle_dir in cycle_dirs:
        cycle_id = cycle_dir.name
        summary = _read_json(cycle_dir / "cycle_summary.json")
        mechanism = _read_json(cycle_dir / "mechanism.json")
        rank0_hits = [int(value) for value in mechanism.get("rank0_measurement_hits", [])]
        rank1_hits = [int(value) for value in mechanism.get("rank1_measurement_hits", [])]
        mechanism_valid = bool(rank0_hits) and all(value > 0 for value in rank0_hits) and all(
            value == 0 for value in rank1_hits
        )
        all_hits.extend(rank0_hits)
        gsm_pairs, gsm_recovery_exact = _gsm_pairs(
            cycle_dir, cycle_id, mechanism_valid=mechanism_valid
        )
        pairs.extend(gsm_pairs)
        pairs.extend(
            _comparison_pairs(
                cycle_id,
                "short",
                summary["short"]["baseline_vs_fault"],
                mechanism_valid=mechanism_valid,
            )
        )
        pairs.extend(
            _comparison_pairs(
                cycle_id,
                "long",
                summary["long"]["baseline_vs_fault"],
                mechanism_valid=mechanism_valid,
            )
        )
        cycle_damage = sum(
            item["baseline"].get("correct") is True and item["fault"].get("correct") is False
            for item in gsm_pairs
        )
        repeatable_damage_cycles += int(cycle_damage > 0)
        exact = (
            gsm_recovery_exact
            and _summary_recovery_exact(summary, "short")
            and _summary_recovery_exact(summary, "long")
            and summary.get("service_health") is True
        )
        recovery_exact.append(exact)
        probabilities.append(summary["probability"])
        cycle_rows.append(
            {
                "cycle": summary.get("cycle", cycle_id),
                "mechanism_valid": mechanism_valid,
                "rank0_hit_min": min(rank0_hits) if rank0_hits else None,
                "rank0_hit_max": max(rank0_hits) if rank0_hits else None,
                "rank1_hit_max": max(rank1_hits) if rank1_hits else None,
                "task_damage_count": cycle_damage,
                "recovery_exact": exact,
            }
        )

    def mean(field: str, phase: str) -> float | None:
        values = [item.get(phase, {}).get(field) for item in probabilities]
        numeric = [float(value) for value in values if isinstance(value, (int, float))]
        return fmean(numeric) if numeric else None

    probability = {
        "baseline_nll": mean("nll", "baseline"),
        "fault_nll": mean("nll", "fault"),
        "recovery_nll": mean("nll", "recovery"),
        "baseline_ppl": mean("ppl", "baseline"),
        "fault_ppl": mean("ppl", "fault"),
        "recovery_ppl": mean("ppl", "recovery"),
        "nll_delta": fmean(float(item["nll_delta"]) for item in probabilities),
        "ppl_ratio": fmean(float(item["ppl_ratio"]) for item in probabilities),
        "scored_token_count": int(manifest["wikitext"]["scored_token_count"]),
    }
    result = EvaluationResult(
        evaluation_id=evaluation_id,
        capability_id=capability_id,
        candidate_id=candidate_id,
        baseline=None,
        fault=None,
        recovery=None,
        mechanism=MechanismMetrics(
            resolved=all(row["mechanism_valid"] for row in cycle_rows),
            candidate_count=None,
            hit_count=sum(all_hits),
            changed_elements=None,
            restore_exact=all(recovery_exact),
            extra={
                "rank_policy": "rank0 targeted; rank1 expected reject",
                "rank0_transaction_count": len(all_hits),
                "rank1_zero_hit": all(row["rank1_hit_max"] == 0 for row in cycle_rows),
            },
        ),
        model_effect=ModelEffectMetrics(response_equal=False, token_equal=False, worker_alive=True),
        reliability=ReliabilityMetrics(
            target_resolve_rate=sum(row["mechanism_valid"] for row in cycle_rows) / len(cycle_rows),
            injection_hit_rate=sum(row["mechanism_valid"] for row in cycle_rows) / len(cycle_rows),
            effect_repeatability=repeatable_damage_cycles / len(cycle_rows),
            recovery_rate=sum(recovery_exact) / len(recovery_exact),
        ),
        quality_performance=QualityPerformanceMetrics(
            nll_delta=probability["nll_delta"],
            ppl=probability["fault_ppl"],
            ppl_ratio=probability["ppl_ratio"],
        ),
        status=EvaluationStatus.PASS,
        reason="read-only replay of completed baseline/fault/recovery cycles",
        evidence_refs=(str(root),),
    )
    profile = build_fault_effect_profile(
        result,
        pairs,
        probability=probability,
        reliability={
            "cycle_count": len(cycle_rows),
            "repeatable_task_damage_cycles": repeatable_damage_cycles,
            "recovery_exact_cycle_count": sum(recovery_exact),
        },
        policy=policy,
    )
    profile["replay"] = {
        "mode": "OFFLINE_READ_ONLY",
        "source_manifest_protocol": manifest.get("protocol"),
        "source_root": str(root),
        "cycles": cycle_rows,
        "raw_responses_included": False,
    }
    output = Path(output_dir)
    paths = write_effect_card(output, profile)
    (output / "replay_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "FAULT_EFFECT_PROFILE_REPLAY_V1",
                "input_root": str(root),
                "source_protocol": manifest.get("protocol"),
                "cycle_count": len(cycle_rows),
                "raw_responses_included": False,
                "output_files": {name: path.name for name, path in paths.items()},
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"evaluation": result, "profile": profile, "paths": paths, "cycles": cycle_rows}
