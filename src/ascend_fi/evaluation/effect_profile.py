"""Offline Fault Effect Profile construction.

The evaluator deliberately separates *sensitivity* from *visible damage*:

``response / token trajectory diverges`` does not by itself mean that a fault
has made the model observably worse.  A pair is classified deterministically
from recorded baseline and fault facts, then aggregated into the V1 Effect
Profile.  No model, native runtime, or control plane is used here.

Input records are deliberately mapping based so the module can replay both
live-evaluation artifacts and future evaluator outputs.  Each pair minimally
needs an identifier plus baseline/fault facts.  Supported facts include
``correct``, ``resolved``, ``response``/``response_hash``, ``token_ids`` or
``token_pieces``, ``format_ok``, ``periodic_loop``, ``punctuation_spam``,
``early_stop``, ``runaway``, ``garbage``, ``http_status``, ``worker_alive``,
and ``mechanism_valid``.  Missing optional facts remain unknown; they are not
silently converted into a degradation claim.
"""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from ascend_fi.contracts import EvaluationResult

from .metrics import safe_ratio, token_divergence


class PrimaryPhenotype(str, Enum):
    """One mutually exclusive primary result for a baseline/fault pair."""

    UNCHANGED = "UNCHANGED"
    DIVERGED_BUT_VALID = "DIVERGED_BUT_VALID"
    WRONG_ANSWER = "WRONG_ANSWER"
    UNRESOLVED_ANSWER = "UNRESOLVED_ANSWER"
    FORMAT_DEGRADATION = "FORMAT_DEGRADATION"
    REPETITION_LOOP = "REPETITION_LOOP"
    PUNCTUATION_SPAM = "PUNCTUATION_SPAM"
    EARLY_TERMINATION = "EARLY_TERMINATION"
    RUNAWAY_GENERATION = "RUNAWAY_GENERATION"
    GARBAGE_OUTPUT = "GARBAGE_OUTPUT"
    REQUEST_ERROR = "REQUEST_ERROR"
    SERVICE_CRASH = "SERVICE_CRASH"


class EffectStage(str, Enum):
    """Evidence-aware degradation stages; these are not a single score."""

    UNCLASSIFIED = "UNCLASSIFIED"
    L0_MECHANISM_ONLY = "L0_MECHANISM_ONLY"
    L1_PROBABILITY_SHIFT = "L1_PROBABILITY_SHIFT"
    L2_TRAJECTORY_DIVERGENCE = "L2_TRAJECTORY_DIVERGENCE"
    L3_VISIBLE_DEGRADATION = "L3_VISIBLE_DEGRADATION"
    L4_REPEATABLE_TASK_DAMAGE = "L4_REPEATABLE_TASK_DAMAGE"
    L5_PATHOLOGICAL_COLLAPSE = "L5_PATHOLOGICAL_COLLAPSE"
    L6_SERVICE_FAILURE = "L6_SERVICE_FAILURE"


@dataclass(frozen=True)
class EffectProfilePolicy:
    """Fixed V1 thresholds used before any result is classified."""

    probability_ppl_ratio: float = 1.05
    trajectory_divergence_rate: float = 0.50
    visible_degradation_rate: float = 0.10
    pathological_rate: float = 0.50
    minimum_repeatable_damage_cycles: int = 2


_PATHOLOGICAL = {
    PrimaryPhenotype.REPETITION_LOOP,
    PrimaryPhenotype.PUNCTUATION_SPAM,
    PrimaryPhenotype.EARLY_TERMINATION,
    PrimaryPhenotype.RUNAWAY_GENERATION,
    PrimaryPhenotype.GARBAGE_OUTPUT,
}
_VISIBLE = set(PrimaryPhenotype) - {
    PrimaryPhenotype.UNCHANGED,
    PrimaryPhenotype.DIVERGED_BUT_VALID,
}


def _as_bool(record: Mapping[str, Any], *names: str) -> bool | None:
    for name in names:
        value = record.get(name)
        if isinstance(value, bool):
            return value
    return None


def _http_success(record: Mapping[str, Any]) -> bool | None:
    value = record.get("http_status")
    if value is None:
        return None
    return isinstance(value, int) and 200 <= value < 300


def _sequence(record: Mapping[str, Any]) -> Sequence[object] | None:
    for key in ("token_ids", "token_pieces", "tokens"):
        value = record.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return value
    return None


def _equal_value(baseline: Mapping[str, Any], fault: Mapping[str, Any]) -> bool | None:
    explicit = _as_bool(fault, "response_equal")
    if explicit is not None:
        return explicit
    for key in ("response", "response_hash"):
        if baseline.get(key) is not None and fault.get(key) is not None:
            return baseline.get(key) == fault.get(key)
    return None


def _response_diverged(baseline: Mapping[str, Any], fault: Mapping[str, Any]) -> bool | None:
    response_equal = _equal_value(baseline, fault)
    if response_equal is False:
        return True
    token_equal = _as_bool(fault, "token_equal")
    if token_equal is False:
        return True
    comparison = token_divergence(_sequence(baseline), _sequence(fault))
    if comparison["count"] is not None:
        return bool(comparison["count"])
    return None if response_equal is None and token_equal is None else False


def _baseline_is_usable(baseline: Mapping[str, Any]) -> bool:
    if _as_bool(baseline, "crash") is True or _as_bool(baseline, "worker_alive") is False:
        return False
    http = _http_success(baseline)
    return http is not False


def _mechanism_is_valid(pair: Mapping[str, Any]) -> bool:
    value = _as_bool(pair, "mechanism_valid", "fault_valid", "injection_valid")
    return value is not False


def classify_paired_request(pair: Mapping[str, Any]) -> dict[str, Any]:
    """Classify one recorded pair without judging free-form text semantically.

    The function makes a degradation claim only for a newly introduced,
    deterministic failure signal.  Mere output divergence maps to
    :class:`PrimaryPhenotype.DIVERGED_BUT_VALID` and is excluded from VDR.
    """

    baseline = pair.get("baseline", {})
    fault = pair.get("fault", {})
    if not isinstance(baseline, Mapping) or not isinstance(fault, Mapping):
        raise ValueError("paired request requires object-valued baseline and fault records")

    pair_id = str(pair.get("sample_id", pair.get("id", "unknown")))
    scope = str(pair.get("scope", pair.get("kind", "overall")))
    valid_pair = _baseline_is_usable(baseline) and _mechanism_is_valid(pair)
    diverged = _response_diverged(baseline, fault)
    baseline_correct = _as_bool(baseline, "correct", "answer_correct")
    fault_correct = _as_bool(fault, "correct", "answer_correct")
    baseline_resolved = _as_bool(baseline, "resolved", "answer_resolved")
    fault_resolved = _as_bool(fault, "resolved", "answer_resolved")

    crash = (
        _as_bool(fault, "crash") is True
        or _as_bool(fault, "worker_alive") is False
    ) and not (_as_bool(baseline, "crash") is True or _as_bool(baseline, "worker_alive") is False)
    request_error = (
        _http_success(fault) is False or _as_bool(fault, "http_error", "request_error") is True
    ) and _http_success(baseline) is not False
    flags = {
        "response_diverged": diverged,
        "baseline_correct_to_fault_wrong": baseline_correct is True and fault_correct is False,
        "baseline_resolved_to_fault_unresolved": baseline_resolved is True and fault_resolved is False,
        "format_degradation": _as_bool(baseline, "format_ok") is True
        and _as_bool(fault, "format_ok") is False,
        "repetition_loop": _as_bool(
            baseline, "periodic_loop", "repetition_loop", "repetitive_output"
        ) is not True
        and _as_bool(fault, "periodic_loop", "repetition_loop", "repetitive_output") is True,
        "punctuation_spam": _as_bool(baseline, "punctuation_spam") is not True
        and _as_bool(fault, "punctuation_spam") is True,
        "early_termination": _as_bool(baseline, "early_stop", "early_termination") is not True
        and _as_bool(fault, "early_stop", "early_termination") is True,
        "empty_output": _as_bool(baseline, "empty_output") is not True
        and _as_bool(fault, "empty_output") is True,
        "runaway_generation": _as_bool(baseline, "runaway") is not True
        and _as_bool(fault, "runaway") is True,
        "garbage_output": _as_bool(baseline, "garbage", "gibberish", "obvious_garbage") is not True
        and _as_bool(fault, "garbage", "gibberish", "obvious_garbage") is True,
        "request_error": request_error,
        "service_crash": crash,
    }

    if not valid_pair:
        primary = PrimaryPhenotype.UNCHANGED
        reason = "INVALID_PAIR_EXCLUDED"
    elif flags["service_crash"]:
        primary, reason = PrimaryPhenotype.SERVICE_CRASH, "fault worker/service stopped"
    elif flags["request_error"]:
        primary, reason = PrimaryPhenotype.REQUEST_ERROR, "fault request failed"
    elif flags["baseline_resolved_to_fault_unresolved"]:
        primary, reason = PrimaryPhenotype.UNRESOLVED_ANSWER, "resolved baseline became unresolved"
    elif flags["baseline_correct_to_fault_wrong"]:
        primary, reason = PrimaryPhenotype.WRONG_ANSWER, "correct baseline became incorrect"
    elif flags["format_degradation"]:
        primary, reason = PrimaryPhenotype.FORMAT_DEGRADATION, "format became invalid"
    elif flags["repetition_loop"]:
        primary, reason = PrimaryPhenotype.REPETITION_LOOP, "new periodic repetition"
    elif flags["punctuation_spam"]:
        primary, reason = PrimaryPhenotype.PUNCTUATION_SPAM, "new punctuation spam"
    elif flags["early_termination"] or flags["empty_output"]:
        primary, reason = PrimaryPhenotype.EARLY_TERMINATION, "new early termination"
    elif flags["runaway_generation"]:
        primary, reason = PrimaryPhenotype.RUNAWAY_GENERATION, "new runaway generation"
    elif flags["garbage_output"]:
        primary, reason = PrimaryPhenotype.GARBAGE_OUTPUT, "new garbage output"
    elif diverged is True:
        primary, reason = PrimaryPhenotype.DIVERGED_BUT_VALID, "trajectory diverged without a visible fault signal"
    else:
        primary, reason = PrimaryPhenotype.UNCHANGED, "no deterministic baseline-relative change"

    return {
        "sample_id": pair_id,
        "scope": scope,
        "valid_pair": valid_pair,
        "primary_phenotype": primary.value,
        "secondary_flags": sorted(name for name, value in flags.items() if value is True),
        # A deterministic visible fault (for example a service error) is also
        # a trajectory-level difference even when no textual response exists.
        # This preserves the invariant VDR <= RDR.
        "response_diverged": diverged if diverged is not None else primary in _VISIBLE,
        "visible_degradation": valid_pair and primary in _VISIBLE,
        "pathological": valid_pair and primary in _PATHOLOGICAL,
        "baseline_correct": baseline_correct,
        "fault_correct": fault_correct,
        "baseline_resolved": baseline_resolved,
        "fault_resolved": fault_resolved,
        "reason": reason,
    }


def _aggregate_classifications(classifications: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [item for item in classifications if item["valid_pair"]]
    phenotype_counts = {phenotype.value: 0 for phenotype in PrimaryPhenotype}
    for item in valid:
        phenotype_counts[str(item["primary_phenotype"])] += 1
    visible = sum(item["visible_degradation"] for item in valid)
    diverged = sum(item["response_diverged"] is True for item in valid)
    pathological = sum(item["pathological"] for item in valid)
    baseline_correct = sum(item["baseline_correct"] is True for item in valid)
    paired_damage = sum(
        item["baseline_correct"] is True and item["fault_correct"] is False for item in valid
    )
    return {
        "pair_count": len(classifications),
        "valid_pair_count": len(valid),
        "invalid_pair_count": len(classifications) - len(valid),
        "visible_degradation_count": visible,
        "visible_degradation_rate": safe_ratio(visible, len(valid)),
        "response_divergence_count": diverged,
        "response_divergence_rate": safe_ratio(diverged, len(valid)),
        "pathological_count": pathological,
        "pathological_rate": safe_ratio(pathological, len(valid)),
        "baseline_correct_count": baseline_correct,
        "paired_damage_count": paired_damage,
        "paired_damage_rate": safe_ratio(paired_damage, baseline_correct),
        "phenotype_counts": phenotype_counts,
    }


def aggregate_paired_effects(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate VDR, RDR, task damage and deterministic phenotype counts."""

    classifications = [classify_paired_request(pair) for pair in pairs]
    by_scope: dict[str, list[Mapping[str, Any]]] = {}
    for item in classifications:
        by_scope.setdefault(str(item["scope"]), []).append(item)
    return {
        "overall": _aggregate_classifications(classifications),
        "by_scope": {
            name: _aggregate_classifications(items) for name, items in sorted(by_scope.items())
        },
        "requests": classifications,
    }


def _metric_record(
    name: str,
    value: float | int | bool | None,
    *,
    unit: str,
    direction: str,
    threshold: float | int | None = None,
    sample_count: int | None = None,
    evidence_level: str = "OBSERVED",
) -> dict[str, Any]:
    triggered = None
    if value is not None and threshold is not None:
        triggered = value >= threshold if direction == "higher_is_worse" else value <= threshold
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "direction": direction,
        "threshold": threshold,
        "triggered": triggered,
        "sample_count": sample_count,
        "evidence_level": evidence_level,
    }


def _mechanism_summary(evaluation: EvaluationResult) -> dict[str, Any]:
    mechanism = evaluation.mechanism
    passed = (
        mechanism.resolved is True
        and mechanism.hit_count is not None
        and mechanism.hit_count > 0
        and (mechanism.changed_elements is None or mechanism.changed_elements > 0)
    )
    return {
        "pass": passed,
        "resolved": mechanism.resolved,
        "candidate_count": mechanism.candidate_count,
        "hit_count": mechanism.hit_count,
        "changed_elements": mechanism.changed_elements,
        "restore_exact": mechanism.restore_exact,
        "rank_conformity": mechanism.extra.get("rank_conformity"),
        "reject_reasons": mechanism.reject_reasons,
    }


def _probability_summary(
    evaluation: EvaluationResult, probability: Mapping[str, Any] | None
) -> dict[str, Any]:
    quality = evaluation.quality_performance
    data = dict(probability or {})
    ppl_ratio = data.get("ppl_ratio", quality.ppl_ratio)
    nll_delta = data.get("nll_delta", quality.nll_delta)
    return {
        "baseline_nll": data.get("baseline_nll"),
        "fault_nll": data.get("fault_nll"),
        "recovery_nll": data.get("recovery_nll"),
        "baseline_ppl": data.get("baseline_ppl"),
        "fault_ppl": data.get("fault_ppl", quality.ppl),
        "recovery_ppl": data.get("recovery_ppl"),
        "nll_delta": nll_delta,
        "ppl_ratio": ppl_ratio,
        "scored_token_count": data.get("scored_token_count", quality.extra.get("scored_token_count")),
    }


def _choose_stage(
    *,
    mechanism_pass: bool,
    overall: Mapping[str, Any],
    probability: Mapping[str, Any],
    reliability: Mapping[str, Any],
    policy: EffectProfilePolicy,
) -> EffectStage:
    if overall["phenotype_counts"][PrimaryPhenotype.SERVICE_CRASH.value] > 0:
        return EffectStage.L6_SERVICE_FAILURE
    if overall["pathological_rate"] is not None and overall["pathological_rate"] >= policy.pathological_rate:
        return EffectStage.L5_PATHOLOGICAL_COLLAPSE
    repeatable_damage_cycles = reliability.get("repeatable_task_damage_cycles", 0)
    if (
        overall["paired_damage_count"] > 0
        and isinstance(repeatable_damage_cycles, int)
        and repeatable_damage_cycles >= policy.minimum_repeatable_damage_cycles
    ):
        return EffectStage.L4_REPEATABLE_TASK_DAMAGE
    if (
        overall["visible_degradation_rate"] is not None
        and overall["visible_degradation_rate"] >= policy.visible_degradation_rate
    ):
        return EffectStage.L3_VISIBLE_DEGRADATION
    if (
        overall["response_divergence_rate"] is not None
        and overall["response_divergence_rate"] >= policy.trajectory_divergence_rate
    ):
        return EffectStage.L2_TRAJECTORY_DIVERGENCE
    if probability["ppl_ratio"] is not None and probability["ppl_ratio"] >= policy.probability_ppl_ratio:
        return EffectStage.L1_PROBABILITY_SHIFT
    if mechanism_pass:
        return EffectStage.L0_MECHANISM_ONLY
    return EffectStage.UNCLASSIFIED


def build_fault_effect_profile(
    evaluation: EvaluationResult,
    pairs: Sequence[Mapping[str, Any]],
    *,
    probability: Mapping[str, Any] | None = None,
    reliability: Mapping[str, Any] | None = None,
    policy: EffectProfilePolicy | None = None,
) -> dict[str, Any]:
    """Build the V1 profile while retaining ``EvaluationResult`` as the root contract."""

    active_policy = policy or EffectProfilePolicy()
    paired = aggregate_paired_effects(pairs)
    mechanism = _mechanism_summary(evaluation)
    probability_result = _probability_summary(evaluation, probability)
    reliability_result = {
        "target_resolve_rate": evaluation.reliability.target_resolve_rate,
        "injection_hit_rate": evaluation.reliability.injection_hit_rate,
        "effect_repeatability": evaluation.reliability.effect_repeatability,
        "recovery_rate": evaluation.reliability.recovery_rate,
        "recovery_exact": evaluation.mechanism.restore_exact,
        **dict(reliability or {}),
    }
    stage = _choose_stage(
        mechanism_pass=mechanism["pass"],
        overall=paired["overall"],
        probability=probability_result,
        reliability=reliability_result,
        policy=active_policy,
    )
    metric_records = [
        _metric_record("mechanism_pass", mechanism["pass"], unit="boolean", direction="required"),
        _metric_record(
            "visible_degradation_rate",
            paired["overall"]["visible_degradation_rate"],
            unit="fraction",
            direction="higher_is_worse",
            threshold=active_policy.visible_degradation_rate,
            sample_count=paired["overall"]["valid_pair_count"],
        ),
        _metric_record(
            "response_divergence_rate",
            paired["overall"]["response_divergence_rate"],
            unit="fraction",
            direction="higher_is_worse",
            threshold=active_policy.trajectory_divergence_rate,
            sample_count=paired["overall"]["valid_pair_count"],
        ),
        _metric_record(
            "paired_damage_rate",
            paired["overall"]["paired_damage_rate"],
            unit="fraction",
            direction="higher_is_worse",
            sample_count=paired["overall"]["baseline_correct_count"],
        ),
        _metric_record(
            "ppl_ratio",
            probability_result["ppl_ratio"],
            unit="ratio",
            direction="higher_is_worse",
            threshold=active_policy.probability_ppl_ratio,
            sample_count=probability_result["scored_token_count"],
        ),
        _metric_record(
            "nll_delta",
            probability_result["nll_delta"],
            unit="nats",
            direction="higher_is_worse",
            sample_count=probability_result["scored_token_count"],
        ),
    ]
    limitations: list[str] = []
    if not paired["overall"]["valid_pair_count"]:
        limitations.append("VDR/RDR_NOT_MEASURED_NO_VALID_PAIRED_REQUESTS")
    if probability_result["ppl_ratio"] is None:
        limitations.append("PROBABILITY_EFFECT_NOT_MEASURED")
    if reliability_result.get("repeatable_task_damage_cycles") is None:
        limitations.append("REPEATABLE_TASK_DAMAGE_NOT_MEASURED")
    return {
        "schema_version": "FAULT_EFFECT_PROFILE_V1",
        "evaluation_id": evaluation.evaluation_id,
        "capability_id": evaluation.capability_id,
        "candidate_id": evaluation.candidate_id,
        "effect_stage": stage.value,
        "policy": asdict(active_policy),
        "mechanism": mechanism,
        "task_effect": {
            "baseline_correct_count": paired["overall"]["baseline_correct_count"],
            "paired_damage_count": paired["overall"]["paired_damage_count"],
            "paired_damage_rate": paired["overall"]["paired_damage_rate"],
        },
        "probability_effect": probability_result,
        "output_effect": {
            "response_divergence_count": paired["overall"]["response_divergence_count"],
            "response_divergence_rate": paired["overall"]["response_divergence_rate"],
        },
        "visible_degradation": {
            "count": paired["overall"]["visible_degradation_count"],
            "rate": paired["overall"]["visible_degradation_rate"],
        },
        "phenotype": {
            "counts": paired["overall"]["phenotype_counts"],
            "pathological_count": paired["overall"]["pathological_count"],
            "pathological_rate": paired["overall"]["pathological_rate"],
        },
        "reliability": reliability_result,
        "recovery_gate": {
            "restore_exact": evaluation.mechanism.restore_exact,
            "recovery_rate": reliability_result.get("recovery_rate"),
        },
        "metric_records": metric_records,
        "paired_requests": paired,
        "limitations": limitations,
        "evidence_refs": list(evaluation.evidence_refs),
    }


def render_effect_card_markdown(profile: Mapping[str, Any]) -> str:
    """Render a compact Chinese-facing Effect Card without embedding raw outputs."""

    mechanism = profile["mechanism"]
    task = profile["task_effect"]
    probability = profile["probability_effect"]
    output = profile["output_effect"]
    visible = profile["visible_degradation"]
    phenotype = profile["phenotype"]
    lines = [
        "# Fault Effect Card",
        "",
        f"- 评估：`{profile['evaluation_id']}`",
        f"- 能力：`{profile['capability_id']}`",
        f"- 当前分级：`{profile['effect_stage']}`",
        "",
        "| 维度 | 指标 | 结果 |",
        "| --- | --- | --- |",
        f"| Mechanism | 注入机制通过 | `{mechanism['pass']}` |",
        f"| Task | Paired Damage | `{task['paired_damage_count']}` / `{task['baseline_correct_count']}` (`{task['paired_damage_rate']}`) |",
        f"| Probability | PPL Ratio / NLL Delta | `{probability['ppl_ratio']}` / `{probability['nll_delta']}` |",
        f"| Output | Response Divergence Rate (RDR) | `{output['response_divergence_rate']}` |",
        f"| Visible | Visible Degradation Rate (VDR) | `{visible['rate']}` ({visible['count']}) |",
        f"| Phenotype | pathological count | `{phenotype['pathological_count']}` |",
        "",
        "## 证据边界",
        "",
        "- RDR 表示生成轨迹敏感度；只有 baseline-relative 的可解释退化才计入 VDR。",
        "- `DIVERGED_BUT_VALID` 不计入 VDR，也不等同于任务能力下降。",
        "- Recovery 是实验有效性门禁，不作为故障强度指标。",
    ]
    reliability = profile.get("reliability", {})
    lines.extend(
        [
            "",
            "## 分级依据",
            "",
            f"- 完整 cycle：`{reliability.get('cycle_count', '未记录')}`；可重复任务损伤 cycle：`{reliability.get('repeatable_task_damage_cycles', '未记录')}`。",
            f"- 恢复精确匹配 cycle：`{reliability.get('recovery_exact_cycle_count', '未记录')}`；恢复率：`{profile['recovery_gate']['recovery_rate']}`。",
            "- L4 由跨 cycle 的 Paired Damage 证据触发；它与总体 VDR 阈值是不同的、可并存的判据。",
        ]
    )
    limitations = profile.get("limitations", [])
    if limitations:
        lines.extend(["", "## 尚未测量/限制", ""])
        lines.extend(f"- `{item}`" for item in limitations)
    return "\n".join(lines) + "\n"


def render_effect_dashboard_html(profile: Mapping[str, Any]) -> str:
    """Render a standalone, dependency-free dashboard for an Effect Card."""

    cards = [
        ("Effect Stage", profile["effect_stage"]),
        ("Mechanism", profile["mechanism"]["pass"]),
        ("VDR", profile["visible_degradation"]["rate"]),
        ("RDR", profile["output_effect"]["response_divergence_rate"]),
        ("PPL Ratio", profile["probability_effect"]["ppl_ratio"]),
        ("Paired Damage", profile["task_effect"]["paired_damage_rate"]),
        ("Recovery Rate", profile["recovery_gate"]["recovery_rate"]),
    ]
    card_html = "".join(
        "<section><h2>{}</h2><p>{}</p></section>".format(html.escape(label), html.escape(str(value)))
        for label, value in cards
    )
    phenotypes = "".join(
        "<tr><td>{}</td><td>{}</td></tr>".format(html.escape(name), html.escape(str(value)))
        for name, value in profile["phenotype"]["counts"].items()
    )
    return f"""<!doctype html>
<html lang=\"zh-CN\"><meta charset=\"utf-8\"><title>Fault Effect Dashboard</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;color:#18212f}}.cards{{display:grid;grid-template-columns:repeat(3,minmax(12rem,1fr));gap:1rem}}section{{border:1px solid #d0d7de;border-radius:.5rem;padding:1rem}}h1{{margin-bottom:.25rem}}h2{{font-size:.9rem;color:#57606a;margin:0}}p{{font-size:1.35rem;font-weight:600;margin:.5rem 0 0}}table{{border-collapse:collapse;margin-top:1.5rem;min-width:26rem}}td,th{{border:1px solid #d0d7de;padding:.45rem;text-align:left}}</style>
<h1>Fault Effect Dashboard</h1><p>Evaluation: <code>{html.escape(str(profile['evaluation_id']))}</code></p>
<main class=\"cards\">{card_html}</main>
<h2>Primary phenotype counts</h2><table><tr><th>Phenotype</th><th>Count</th></tr>{phenotypes}</table>
<h2>Evidence boundary</h2><p>RDR measures trajectory sensitivity. VDR requires a deterministic baseline-relative degradation.</p>
</html>"""


def write_effect_card(output_dir: str | Path, profile: Mapping[str, Any]) -> dict[str, Path]:
    """Write JSON, Markdown and standalone HTML derivatives; never raw responses."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "effect_card.json"
    markdown_path = target / "effect_card.md"
    dashboard_path = target / "effect_dashboard.html"
    json_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_effect_card_markdown(profile), encoding="utf-8")
    dashboard_path.write_text(render_effect_dashboard_html(profile), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path, "dashboard": dashboard_path}
