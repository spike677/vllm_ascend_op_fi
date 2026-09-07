"""Real-model Fault Effect Evaluation using the shared ``EvaluationResult``."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Sequence

from ascend_fi.contracts import (
    Candidate,
    EvaluationResult,
    EvaluationStatus,
    MechanismMetrics,
    ModelEffectMetrics,
    QualityPerformanceMetrics,
    ReliabilityMetrics,
    ValidationResult,
)

from .live import (
    EvaluationSample,
    LiveFaultMode,
    LiveQwenFaultExecutor,
    LOSSLESS_QMM_RANK0,
    MILD_INT4_QMM_RANK0,
    STRONG_INT2_QMM_RANK0,
)
from .metrics import (
    aggregate_repeatability,
    extract_gsm8k_answer,
    safe_ratio,
    token_divergence,
    word_perplexity,
)
from .runner import RunOutcome
from .signals import obvious_gibberish, punctuation_spam_rate, repetition_rate


_DISARMED = LiveFaultMode("disarmed", (), "", 0)
_QMM_INPUT_K = 5120
# This is the existing Fault20 v17 controller's documented all-hit bound; it
# is not a new native limit. Keep the live evaluator inside that capability.
_EXISTING_ALL_HIT_MAX_BYTES = 65_536


def load_smoke_pack(path: str | Path) -> list[EvaluationSample]:
    samples: list[EvaluationSample] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"smoke line {line_number} must be an object")
        samples.append(
            EvaluationSample(
                sample_id=str(value["id"]),
                prompt=str(value["prompt"]),
                max_tokens=int(value["max_tokens"]),
                temperature=float(value.get("temperature", 0)),
                seed=int(value.get("seed", 1234)),
                metadata=dict(value.get("metadata", {})),
            )
        )
    if not 20 <= len(samples) <= 50:
        raise ValueError("smoke pack must contain 20 to 50 fixed samples")
    return samples


def load_gsm8k(path: str | Path, *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    samples: list[dict[str, Any]] = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if len(samples) >= limit:
            break
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not isinstance(value.get("question"), str):
            raise ValueError(f"invalid GSM8K row {index}")
        reference = value.get("answer")
        if not isinstance(reference, str) or extract_gsm8k_answer(reference) is None:
            raise ValueError(f"GSM8K row {index} has no deterministic numeric answer")
        samples.append(
            {
                "id": f"gsm8k-test-{index:04d}",
                "index": index,
                "question": value["question"],
                "reference": reference,
            }
        )
    if len(samples) != limit:
        raise ValueError(f"requested {limit} GSM8K rows but found {len(samples)}")
    return samples


def read_wikitext_raw(path: str | Path) -> str:
    lines = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or (stripped.startswith("=") and stripped.endswith("=")):
            continue
        lines.append(stripped)
    if not lines:
        raise ValueError("WikiText raw input contains no document text")
    return "\n".join(lines)


def prepare_wikitext_windows(
    executor: LiveQwenFaultExecutor,
    text: str,
    *,
    window_tokens: int = 8,
    max_windows: int = 8,
) -> dict[str, Any]:
    if window_tokens < 2 or max_windows < 1:
        raise ValueError("WikiText window must have >=2 tokens and >=1 window")
    required_tokens = 1 + (window_tokens - 1) * max_windows
    # The HTTP tokenize endpoint enforces the model context limit.  Only a
    # deterministic source prefix is needed for the fixed V1 subset; eight
    # source characters per requested token is conservative for English text
    # while remaining below a 4096-token context for the default 8 windows.
    source_prefix = text[: min(len(text), required_tokens * 8)]
    token_ids = executor.tokenize(source_prefix)
    usable = min(len(token_ids), 1 + (window_tokens - 1) * max_windows)
    selected = token_ids[:usable]
    windows = []
    for start in range(0, max(0, len(selected) - 1), window_tokens - 1):
        window = selected[start : start + window_tokens]
        if len(window) != window_tokens:
            break
        windows.append(window)
        if len(windows) == max_windows:
            break
    if not windows:
        raise ValueError("WikiText subset is too short for a complete scoring window")
    selected_count = 1 + len(windows) * (window_tokens - 1)
    selected = selected[:selected_count]
    selected_text = executor.detokenize(selected)
    return {
        "window_tokens": window_tokens,
        "windows": windows,
        "selected_token_count": len(selected),
        "scored_token_count": len(windows) * (window_tokens - 1),
        "word_count": len(re.findall(r"\S+", selected_text)),
        "selected_text_sha256": hashlib.sha256(selected_text.encode("utf-8")).hexdigest(),
        "source_prefix_sha256": hashlib.sha256(source_prefix.encode("utf-8")).hexdigest(),
    }


def _mode_for_m(mode: LiveFaultMode, m: int) -> LiveFaultMode:
    if not mode.arm_args:
        return mode
    args = list(mode.arm_args)
    if len(args) != 5:
        raise ValueError(f"unsupported QMM all-hit command shape for {mode.name}: {args}")
    required_bytes = m * _QMM_INPUT_K
    if required_bytes > _EXISTING_ALL_HIT_MAX_BYTES:
        raise ValueError(
            "requested QMM window exceeds the existing all-hit controller "
            f"limit: M={m}, K={_QMM_INPUT_K}, bytes={required_bytes}, "
            f"limit={_EXISTING_ALL_HIT_MAX_BYTES}"
        )
    args[2] = str(m)
    args[4] = str(required_bytes)
    return replace(mode, name=f"{mode.name}_m{m}", arm_args=tuple(args))


def _outcome_dict(sample: EvaluationSample | dict[str, Any], outcome: RunOutcome) -> dict[str, Any]:
    sample_id = sample.sample_id if isinstance(sample, EvaluationSample) else str(sample["id"])
    return {
        "sample_id": sample_id,
        "http_status": outcome.http_status,
        "response_text": outcome.response_text,
        "response_hash": outcome.response_hash,
        "token_ids": None if outcome.token_ids is None else list(outcome.token_ids),
        "token_pieces": None if outcome.token_pieces is None else list(outcome.token_pieces),
        "generation_length": outcome.generation_length,
        "worker_alive": outcome.worker_alive,
        "finish_reason": outcome.finish_reason,
        "error": outcome.error,
    }


def _run_gsm8k(
    executor: LiveQwenFaultExecutor, samples: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    results = []
    for sample in samples:
        prompt = (
            "Solve the grade-school math problem. Show concise reasoning, then end with exactly "
            "'#### <number>'.\n\nQuestion: " + sample["question"] + "\nAnswer:"
        )
        outcome = executor.run_sample(
            EvaluationSample(str(sample["id"]), prompt, 128, metadata={"task": "gsm8k"})
        )
        record = _outcome_dict(sample, outcome)
        record.update(
            {
                "index": sample["index"],
                "reference_answer": extract_gsm8k_answer(sample["reference"]),
                "predicted_answer": extract_gsm8k_answer(outcome.response_text),
            }
        )
        record["correct"] = (
            record["predicted_answer"] is not None
            and record["predicted_answer"] == record["reference_answer"]
        )
        results.append(record)
    return results


def _run_ppl(executor: LiveQwenFaultExecutor, prepared: dict[str, Any]) -> dict[str, Any]:
    rows = []
    total_nll = 0.0
    scored = 0
    for index, window in enumerate(prepared["windows"]):
        result = executor.prompt_loglikelihood(window)
        raw_values = result.get("token_logprobs")
        values = [float(value) for value in raw_values or [] if isinstance(value, (int, float))]
        nll = -sum(values)
        total_nll += nll
        scored += len(values)
        rows.append(
            {
                "window_index": index,
                "http_status": result.get("http_status"),
                "prompt_tokens": result.get("prompt_tokens"),
                "scored_token_count": len(values),
                "nll": nll if values else None,
                "error": result.get("error"),
            }
        )
    complete = (
        scored == prepared["scored_token_count"]
        and all(row["http_status"] == 200 and row["error"] is None for row in rows)
    )
    nll = total_nll if complete else None
    ppl = word_perplexity(nll, prepared["word_count"])
    return {
        "method": (
            "openai_echo_loglikelihood_overlapping_fixed_"
            f"{prepared['window_tokens']}_token_windows"
        ),
        "nll": nll,
        "word_perplexity": ppl,
        "word_count": prepared["word_count"],
        "selected_token_count": prepared["selected_token_count"],
        "scored_token_count": scored,
        "selected_text_sha256": prepared["selected_text_sha256"],
        "complete": complete,
        "windows": rows,
    }


def _public_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return dict(snapshot)


def _run_phase(
    executor: LiveQwenFaultExecutor,
    name: str,
    mode: LiveFaultMode,
    smoke: Sequence[EvaluationSample],
    gsm8k: Sequence[dict[str, Any]],
    wikitext: dict[str, Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    print(f"EVALUATION_PHASE_START name={name} mode={mode.name}", flush=True)
    decode_offset = executor.trace_offset()
    try:
        if mode.arm_args:
            executor.arm(mode)
        elif not executor.disarm(None):
            raise RuntimeError(f"{name}: could not establish disarmed state")
        smoke_rows = [_outcome_dict(sample, executor.run_sample(sample)) for sample in smoke]
        gsm_rows = _run_gsm8k(executor, gsm8k)
        decode_snapshot = executor.runtime_snapshot(decode_offset, mode, target_m=1)
    finally:
        # Always disarm, including the case where the native arm succeeded but
        # the following status/readback raised before ``arm()`` returned.
        decode_disarm = executor.disarm(None)

    ppl_mode = _mode_for_m(mode, int(wikitext["window_tokens"]))
    ppl_offset = executor.trace_offset()
    try:
        if ppl_mode.arm_args:
            executor.arm(ppl_mode)
        elif not executor.disarm(None):
            raise RuntimeError(f"{name} PPL: could not establish disarmed state")
        ppl = _run_ppl(executor, wikitext)
        ppl_snapshot = executor.runtime_snapshot(
            ppl_offset, ppl_mode, target_m=int(wikitext["window_tokens"])
        )
    finally:
        ppl_disarm = executor.disarm(None)

    result = {
        "name": name,
        "mode": mode.name,
        "smoke": smoke_rows,
        "gsm8k": gsm_rows,
        "ppl": ppl,
        "runtime": {
            "decode": _public_snapshot(decode_snapshot),
            "ppl": _public_snapshot(ppl_snapshot),
        },
        "disarm_success": decode_disarm and ppl_disarm,
    }
    phase_dir = artifact_dir / name
    phase_dir.mkdir(parents=True, exist_ok=True)
    (phase_dir / "raw_phase_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"EVALUATION_PHASE_COMPLETE name={name}", flush=True)
    return result


def _sequence(row: dict[str, Any]) -> tuple[Sequence[object] | None, str | None]:
    if row["token_ids"] is not None:
        return row["token_ids"], "token_ids"
    if row["token_pieces"] is not None:
        return row["token_pieces"], "token_pieces"
    return None, None


def _smoke_metrics(baseline: dict[str, Any], phase: dict[str, Any]) -> dict[str, Any]:
    base_rows = baseline["smoke"]
    rows = phase["smoke"]
    if len(base_rows) != len(rows):
        raise ValueError("smoke phase size mismatch")
    divergence_count = 0
    divergence_denominator = 0
    first_by_sample: dict[str, int | None] = {}
    token_comparable = True
    repeats = []
    spam = []
    early = 0
    crashes = 0
    response_equal = 0
    gibberish = 0
    for base, row in zip(base_rows, rows):
        left, left_basis = _sequence(base)
        right, right_basis = _sequence(row)
        if left_basis is None or left_basis != right_basis:
            token_comparable = False
            divergence = {"first": None, "count": None, "rate": None}
        else:
            divergence = token_divergence(left, right)
            divergence_count += int(divergence["count"] or 0)
            divergence_denominator += max(len(left or ()), len(right or ()))
        first_by_sample[row["sample_id"]] = divergence["first"]
        sequence = right
        value = repetition_rate(sequence)
        if value is not None:
            repeats.append(value)
        value = punctuation_spam_rate(row["response_text"])
        if value is not None:
            spam.append(value)
        early += int(
            isinstance(base["generation_length"], int)
            and isinstance(row["generation_length"], int)
            and row["generation_length"] < base["generation_length"]
        )
        crashes += int(row["http_status"] != 200 or row["worker_alive"] is not True)
        response_equal += int(base["response_text"] == row["response_text"])
        gibberish += int(obvious_gibberish(row["response_text"]) is True)
    base_length = sum(row["generation_length"] or 0 for row in base_rows)
    phase_length = sum(row["generation_length"] or 0 for row in rows)
    first_values = [value for value in first_by_sample.values() if value is not None]
    return {
        "sample_count": len(rows),
        "http_success_rate": sum(row["http_status"] == 200 for row in rows) / len(rows),
        "response_equal_rate": response_equal / len(rows),
        "response_equal": response_equal == len(rows),
        "token_comparable": token_comparable,
        "token_divergence_count": divergence_count if token_comparable else None,
        "token_divergence_rate": (
            divergence_count / divergence_denominator
            if token_comparable and divergence_denominator
            else (0.0 if token_comparable else None)
        ),
        "first_divergent_token": min(first_values) if first_values else None,
        "first_divergent_token_by_sample": first_by_sample,
        "generation_length": phase_length,
        "generation_length_delta": phase_length - base_length,
        "output_length_ratio": safe_ratio(phase_length, base_length),
        "repetition_rate": fmean(repeats) if repeats else None,
        "punctuation_spam_rate": fmean(spam) if spam else None,
        "early_stop_rate": early / len(rows),
        "crash_error_rate": crashes / len(rows),
        "obvious_gibberish_rate": gibberish / len(rows),
    }


def _gsm_metrics(baseline: dict[str, Any], phase: dict[str, Any]) -> dict[str, Any]:
    base_rows = baseline["gsm8k"]
    rows = phase["gsm8k"]
    baseline_accuracy = (
        sum(row["correct"] for row in base_rows) / len(base_rows) if base_rows else None
    )
    accuracy = sum(row["correct"] for row in rows) / len(rows) if rows else None
    return {
        "sample_count": len(rows),
        "sample_ids": [row["sample_id"] for row in rows],
        "accuracy": accuracy,
        "accuracy_delta_pp": (
            None if accuracy is None or baseline_accuracy is None else (accuracy - baseline_accuracy) * 100
        ),
        "accuracy_retention": safe_ratio(accuracy, baseline_accuracy),
    }


def _mechanism_summary(phase: dict[str, Any]) -> dict[str, Any]:
    snapshots = [phase["runtime"]["decode"], phase["runtime"]["ppl"]]
    ranks: dict[str, Any] = {}
    for rank in ("0", "1"):
        values = [snapshot["multirank"]["ranks"][rank] for snapshot in snapshots]
        expected_reject_values = [value["expected_reject"] for value in values]
        ranks[rank] = {
            "pid": values[-1]["pid"] or values[0]["pid"],
            "alive": all(value["alive"] is True for value in values),
            "resolved": any(value["resolved"] for value in values),
            "matching_call_count": sum(value["matching_call_count"] for value in values),
            "candidate_count": sum(value["candidate_count"] or 0 for value in values),
            "hit_count": sum(value["hit_count"] or 0 for value in values),
            "reject_count": sum(value["reject_count"] or 0 for value in values),
            "reject_reasons": {},
            "expected_reject": (
                None
                if all(value is None for value in expected_reject_values)
                else all(value is True for value in expected_reject_values)
            ),
        }
        for value in values:
            for reason, count in value["reject_reasons"].items():
                ranks[rank]["reject_reasons"][reason] = ranks[rank]["reject_reasons"].get(reason, 0) + count
    return {
        "resolved": any(snapshot["resolved"] for snapshot in snapshots),
        "candidate_count": sum(snapshot["status"].get("candidate_count", 0) for snapshot in snapshots),
        "hit_count": sum(snapshot["status"].get("total_hit_count", snapshot["status"].get("hit_count", 0)) for snapshot in snapshots),
        "reject_count": sum(snapshot.get("reject_count", 0) for snapshot in snapshots),
        "changed_elements": sum(snapshot.get("changed_elements") or 0 for snapshot in snapshots) or None,
        "temporary_freed": all(snapshot.get("temporary_freed") is True for snapshot in snapshots),
        "service_health": all(snapshot["service_health"] is True for snapshot in snapshots),
        "per_rank": {
            "expected_worker_count": snapshots[0]["multirank"]["expected_worker_count"],
            "observed_worker_count": min(snapshot["multirank"]["observed_worker_count"] for snapshot in snapshots),
            "ranks": ranks,
        },
        "errors": {
            key: any(snapshot["errors"].get(key) is True for snapshot in snapshots)
            for key in ("cann_error", "hccl_error", "oom", "worker_exit")
        },
    }


def _phase_summary(baseline: dict[str, Any], phase: dict[str, Any]) -> dict[str, Any]:
    ppl = phase["ppl"]["word_perplexity"]
    base_ppl = baseline["ppl"]["word_perplexity"]
    return {
        "mode": phase["mode"],
        "smoke": _smoke_metrics(baseline, phase),
        "gsm8k": _gsm_metrics(baseline, phase),
        "wikitext": {
            **{key: phase["ppl"].get(key) for key in (
                "method", "nll", "word_perplexity", "word_count", "selected_token_count",
                "scored_token_count", "selected_text_sha256", "complete",
            )},
            "ppl_ratio": safe_ratio(ppl, base_ppl),
        },
        "mechanism": _mechanism_summary(phase),
        "disarm_success": phase["disarm_success"],
    }


def _repeat_strong(
    executor: LiveQwenFaultExecutor,
    sample: EvaluationSample,
    baseline_row: dict[str, Any],
    *,
    repeat: int,
) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    records = []
    for index in range(repeat):
        offset = executor.trace_offset()
        try:
            executor.arm(STRONG_INT2_QMM_RANK0)
            fault = executor.run_sample(sample)
            snapshot = executor.runtime_snapshot(offset, STRONG_INT2_QMM_RANK0)
        finally:
            disarmed = executor.disarm(None)
        recovery = executor.run_sample(sample)
        fault_row = _outcome_dict(sample, fault)
        recovery_row = _outcome_dict(sample, recovery)
        rank0_hit = snapshot["multirank"]["ranks"]["0"]["hit_count"] or 0
        record = {
            "repeat_index": index,
            "resolved": snapshot["resolved"],
            "hit": rank0_hit > 0,
            "effect_observed": fault_row["response_text"] != baseline_row["response_text"],
            "recovery_exact": recovery_row["response_text"] == baseline_row["response_text"],
            "crash": fault_row["http_status"] != 200 or fault_row["worker_alive"] is not True,
            "disarm_success": disarmed,
            "fault": fault_row,
            "recovery": recovery_row,
            "runtime": _public_snapshot(snapshot),
        }
        records.append(record)
    return records, aggregate_repeatability(records)


def _summary_markdown(result: EvaluationResult) -> str:
    phases = result.extra["phase_matrix"]
    columns = ("baseline", "lossless", "mild", "strong", "recovery")
    labels = ("Baseline", "Lossless", "Mild", "Strong", "Recovery")
    rows = [
        ("HTTP success", lambda p: p["smoke"]["http_success_rate"]),
        ("GSM8K Accuracy", lambda p: p["gsm8k"]["accuracy"]),
        ("Accuracy Retention", lambda p: p["gsm8k"]["accuracy_retention"]),
        ("WikiText PPL", lambda p: p["wikitext"]["word_perplexity"]),
        ("PPL Ratio", lambda p: p["wikitext"]["ppl_ratio"]),
        ("Token divergence rate", lambda p: p["smoke"]["token_divergence_rate"]),
        ("First divergence", lambda p: p["smoke"]["first_divergent_token"]),
        ("Repetition Rate", lambda p: p["smoke"]["repetition_rate"]),
        ("Spam Rate", lambda p: p["smoke"]["punctuation_spam_rate"]),
        ("Early Stop Rate", lambda p: p["smoke"]["early_stop_rate"]),
        ("Crash/Error Rate", lambda p: p["smoke"]["crash_error_rate"]),
        ("rank0 hit", lambda p: p["mechanism"]["per_rank"]["ranks"]["0"]["hit_count"]),
        ("rank1 hit", lambda p: p["mechanism"]["per_rank"]["ranks"]["1"]["hit_count"]),
        ("exact recovery", lambda p: p["smoke"]["response_equal"]),
    ]
    lines = [
        "# Qwen 2-card Fault Effect Evaluation V1",
        "",
        f"Status: `{result.status.value}`",
        "",
        "| Metric | " + " | ".join(labels) + " |",
        "| --- | " + " | ".join("---:" for _ in labels) + " |",
    ]
    for label, getter in rows:
        values = []
        for column in columns:
            value = getter(phases[column])
            values.append("null" if value is None else str(value))
        lines.append("| " + label + " | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Lossless gate",
            "",
            "```json",
            json.dumps(result.extra["lossless_gate"], ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "## Repeatability definition",
            "",
            "V1 counts an effect when the fixed representative completion text differs from its baseline. "
            "Each repeat is `arm -> fault request -> disarm -> recovery request`.",
            "",
            "```json",
            json.dumps(result.extra["repeatability"], ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "## Evidence boundary",
            "",
            "INT4/INT2-equivalent modes preserve INT8 storage and use existing native transforms. "
            "WikiText scoring uses overlapping fixed 128-token teacher-forced echo windows so the "
            "existing QMM selector can target prefill M=128 without adding a new native mechanism.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_qwen_fault_effect_suite(
    evaluation_id: str,
    candidate: Candidate,
    validation: ValidationResult,
    executor: LiveQwenFaultExecutor,
    *,
    smoke_samples: Sequence[EvaluationSample],
    gsm8k_samples: Sequence[dict[str, Any]],
    wikitext_text: str,
    artifact_dir: str | Path,
    repeat: int = 3,
    wikitext_windows: int = 8,
    wikitext_window_tokens: int = 8,
    environment: dict[str, Any] | None = None,
) -> EvaluationResult:
    if not validation.destructive_injection_allowed:
        raise ValueError("live fault suite requires a passed independent lossless gate")
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    wikitext = prepare_wikitext_windows(
        executor,
        wikitext_text,
        window_tokens=wikitext_window_tokens,
        max_windows=wikitext_windows,
    )
    raw_phases: dict[str, dict[str, Any]] = {}
    raw_phases["baseline"] = _run_phase(
        executor, "baseline", _DISARMED, smoke_samples, gsm8k_samples, wikitext, output
    )
    raw_phases["lossless"] = _run_phase(
        executor, "lossless", LOSSLESS_QMM_RANK0, smoke_samples, gsm8k_samples, wikitext, output
    )
    lossless_preflight = _phase_summary(raw_phases["baseline"], raw_phases["lossless"])
    ppl_nll_equal = (
        raw_phases["baseline"]["ppl"]["nll"] is not None
        and raw_phases["baseline"]["ppl"]["nll"] == raw_phases["lossless"]["ppl"]["nll"]
    )
    lossless_m128 = raw_phases["lossless"]["runtime"]["ppl"]
    if not all(
        (
            lossless_preflight["smoke"]["response_equal"] is True,
            ppl_nll_equal,
            lossless_m128["resolved"] is True,
            lossless_m128["temporary_freed"] is True,
            lossless_m128["service_health"] is True,
            raw_phases["lossless"]["disarm_success"] is True,
        )
    ):
        raise RuntimeError("lossless M=1/M=128 preflight failed; destructive phases not started")
    raw_phases["mild"] = _run_phase(
        executor, "mild", MILD_INT4_QMM_RANK0, smoke_samples, gsm8k_samples, wikitext, output
    )
    raw_phases["strong"] = _run_phase(
        executor, "strong", STRONG_INT2_QMM_RANK0, smoke_samples, gsm8k_samples, wikitext, output
    )
    raw_phases["recovery"] = _run_phase(
        executor, "recovery", _DISARMED, smoke_samples, gsm8k_samples, wikitext, output
    )
    summaries = {
        name: _phase_summary(raw_phases["baseline"], phase)
        for name, phase in raw_phases.items()
    }
    repeat_records, repeatability = _repeat_strong(
        executor,
        smoke_samples[0],
        raw_phases["baseline"]["smoke"][0],
        repeat=repeat,
    )
    (output / "repeatability.json").write_text(
        json.dumps(repeat_records, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    strong = summaries["strong"]
    lossless = summaries["lossless"]
    recovery = summaries["recovery"]
    rank0_hit = strong["mechanism"]["per_rank"]["ranks"]["0"]["hit_count"]
    rank1_hit = strong["mechanism"]["per_rank"]["ranks"]["1"]["hit_count"]
    lossless_gate = {
        "status": validation.status.value,
        "target_resolved": validation.resolved,
        "substitution_executed": validation.lossless_attempted and validation.hit_count > 0,
        "real_operator_run_status": validation.extra.get("run_status"),
        "model_output_equal": validation.model_output_equal,
        "service_healthy": lossless["mechanism"]["service_health"],
        "temporary_freed": validation.extra.get("temporary_freed"),
        "restore_exact": validation.restore_exact,
        "phase_response_equal": lossless["smoke"]["response_equal"],
        "ppl_nll_equal": ppl_nll_equal,
        "ppl_window_hit": lossless_m128["resolved"],
        "ppl_window_temporary_freed": lossless_m128["temporary_freed"],
    }
    acceptance = {
        "two_workers_alive": strong["mechanism"]["per_rank"]["observed_worker_count"] == 2
        and all(rank["alive"] for rank in strong["mechanism"]["per_rank"]["ranks"].values()),
        "lossless_gate": all(
            (
                validation.destructive_injection_allowed,
                lossless_gate["model_output_equal"] is True,
                lossless_gate["phase_response_equal"] is True,
                lossless_gate["temporary_freed"] is True,
                lossless_gate["ppl_nll_equal"] is True,
                lossless_gate["ppl_window_hit"] is True,
                lossless_gate["ppl_window_temporary_freed"] is True,
            )
        ),
        "fault_hit_rank_aware": isinstance(rank0_hit, int) and rank0_hit > 0 and rank1_hit == 0,
        "gsm8k_complete": all(summary["gsm8k"]["accuracy"] is not None for summary in summaries.values()),
        "wikitext_complete": all(summary["wikitext"]["complete"] is True for summary in summaries.values()),
        "recovery_exact": recovery["smoke"]["response_equal"] is True,
        "repeatability_complete": len(repeat_records) == repeat
        and repeatability["target_resolve_rate"] == 1.0
        and repeatability["injection_hit_rate"] == 1.0
        and repeatability["exact_recovery_rate"] == 1.0,
        "service_healthy": all(summary["mechanism"]["service_health"] for summary in summaries.values()),
    }
    passed = all(acceptance.values())
    first_baseline = executor.config.gate_sample if not smoke_samples else smoke_samples[0]
    del first_baseline  # documents that RunRecord is sourced from the fixed pack below.
    baseline_outcome = _dict_to_outcome(raw_phases["baseline"]["smoke"][0])
    strong_outcome = _dict_to_outcome(raw_phases["strong"]["smoke"][0])
    recovery_outcome = _dict_to_outcome(raw_phases["recovery"]["smoke"][0])
    result = EvaluationResult(
        evaluation_id=evaluation_id,
        capability_id=candidate.capability_id,
        candidate_id=candidate.candidate_id,
        baseline=baseline_outcome.as_record(),
        fault=strong_outcome.as_record(),
        recovery=recovery_outcome.as_record(),
        mechanism=MechanismMetrics(
            resolved=strong["mechanism"]["resolved"],
            candidate_count=strong["mechanism"]["candidate_count"],
            hit_count=strong["mechanism"]["hit_count"],
            reject_count=strong["mechanism"]["reject_count"],
            changed_elements=strong["mechanism"]["changed_elements"],
            restore_exact=strong["mechanism"]["temporary_freed"],
            extra={"per_rank": strong["mechanism"]["per_rank"], "errors": strong["mechanism"]["errors"]},
        ),
        model_effect=ModelEffectMetrics(
            http_status=strong_outcome.http_status,
            response_equal=strong["smoke"]["response_equal"],
            response_hash=strong_outcome.response_hash,
            token_equal=(strong["smoke"]["token_divergence_count"] == 0 if strong["smoke"]["token_comparable"] else None),
            first_divergent_token=strong["smoke"]["first_divergent_token"],
            generation_length=strong["smoke"]["generation_length"],
            generation_length_delta=strong["smoke"]["generation_length_delta"],
            repetition_rate=strong["smoke"]["repetition_rate"],
            spam_rate=strong["smoke"]["punctuation_spam_rate"],
            early_stop=strong["smoke"]["early_stop_rate"] > 0,
            crash=strong["smoke"]["crash_error_rate"] > 0,
            worker_alive=strong["mechanism"]["service_health"],
            extra={
                "token_divergence_count": strong["smoke"]["token_divergence_count"],
                "token_divergence_rate": strong["smoke"]["token_divergence_rate"],
                "output_length_ratio": strong["smoke"]["output_length_ratio"],
                "early_stop_rate": strong["smoke"]["early_stop_rate"],
                "crash_error_rate": strong["smoke"]["crash_error_rate"],
                "obvious_gibberish_rate": strong["smoke"]["obvious_gibberish_rate"],
            },
        ),
        reliability=ReliabilityMetrics(
            target_resolve_rate=repeatability["target_resolve_rate"],
            injection_hit_rate=repeatability["injection_hit_rate"],
            effect_repeatability=repeatability["effect_repeatability"],
            recovery_rate=repeatability["exact_recovery_rate"],
            extra={"crash_rate": repeatability["crash_rate"], "repeat": repeat},
        ),
        quality_performance=QualityPerformanceMetrics(
            nll_delta=(
                None
                if strong["wikitext"]["nll"] is None or summaries["baseline"]["wikitext"]["nll"] is None
                else strong["wikitext"]["nll"] - summaries["baseline"]["wikitext"]["nll"]
            ),
            ppl=strong["wikitext"]["word_perplexity"],
            ppl_ratio=strong["wikitext"]["ppl_ratio"],
            accuracy=strong["gsm8k"]["accuracy"],
            accuracy_retention=strong["gsm8k"]["accuracy_retention"],
            extra={"gsm8k_accuracy_delta_pp": strong["gsm8k"]["accuracy_delta_pp"]},
        ),
        status=EvaluationStatus.PASS if passed else EvaluationStatus.FAIL,
        reason=(
            "real two-rank lossless/fault/recovery evaluation completed"
            if passed
            else "one or more real-model acceptance gates failed"
        ),
        evidence_refs=(str(executor.config.trace_path), str(output)),
        extra={
            "environment": dict(environment or {}),
            "phase_matrix": summaries,
            "lossless_gate": lossless_gate,
            "repeatability": {"records": repeat_records, "aggregate": repeatability},
            "acceptance": acceptance,
            "result_label": "QWEN_2CARD_FAULT_EFFECT_EVALUATION_V1_PASS" if passed else None,
        },
    )
    (output / "evaluation_result.json").write_text(result.to_json() + "\n", encoding="utf-8")
    (output / "effect_evaluation_summary.md").write_text(
        _summary_markdown(result), encoding="utf-8"
    )
    return result


def _dict_to_outcome(row: dict[str, Any]) -> RunOutcome:
    return RunOutcome(
        http_status=row.get("http_status"),
        response_text=row.get("response_text"),
        token_ids=None if row.get("token_ids") is None else tuple(row["token_ids"]),
        token_pieces=None if row.get("token_pieces") is None else tuple(row["token_pieces"]),
        generation_length=row.get("generation_length"),
        worker_alive=row.get("worker_alive"),
        finish_reason=row.get("finish_reason"),
        error=row.get("error"),
    )
