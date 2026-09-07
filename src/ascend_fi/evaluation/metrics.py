"""Deterministic fault-effect metrics used by recorded and live evaluation."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


_NUMBER = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)")


def safe_ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    """Return a finite ratio, or ``None`` when it is not mathematically defined."""

    if numerator is None or denominator is None or denominator == 0:
        return None
    value = float(numerator) / float(denominator)
    return value if math.isfinite(value) else None


def token_divergence(
    baseline: Sequence[object] | None, observed: Sequence[object] | None
) -> dict[str, int | float | None]:
    """Compare aligned token sequences without treating token pieces as token IDs."""

    if baseline is None or observed is None:
        return {"first": None, "count": None, "rate": None}
    common = min(len(baseline), len(observed))
    different = sum(baseline[index] != observed[index] for index in range(common))
    different += abs(len(baseline) - len(observed))
    first = next(
        (index for index in range(common) if baseline[index] != observed[index]),
        common if len(baseline) != len(observed) else None,
    )
    denominator = max(len(baseline), len(observed))
    return {
        "first": first,
        "count": different,
        "rate": 0.0 if denominator == 0 else different / denominator,
    }


def normalize_numeric_answer(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.replace(",", "").strip()
    try:
        number = float(normalized)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    if number.is_integer():
        return str(int(number))
    return format(number, ".12g")


def extract_gsm8k_answer(text: str | None) -> str | None:
    """Extract a numeric answer deterministically, preferring GSM8K's ``####`` marker."""

    if not text:
        return None
    marked = re.findall(r"####\s*(" + _NUMBER.pattern + r")", text)
    candidates = marked or _NUMBER.findall(text)
    return normalize_numeric_answer(candidates[-1]) if candidates else None


def gsm8k_accuracy(
    references: Sequence[str | None], predictions: Sequence[str | None]
) -> float | None:
    if len(references) != len(predictions):
        raise ValueError("GSM8K references and predictions must have equal length")
    if not references:
        return None
    correct = sum(
        extract_gsm8k_answer(reference) is not None
        and extract_gsm8k_answer(reference) == extract_gsm8k_answer(prediction)
        for reference, prediction in zip(references, predictions)
    )
    return correct / len(references)


def word_perplexity(nll: float | None, word_count: int | None) -> float | None:
    if nll is None or word_count is None or word_count <= 0 or nll < 0:
        return None
    exponent = nll / word_count
    if not math.isfinite(exponent) or exponent > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
        return None
    value = math.exp(exponent)
    return value if math.isfinite(value) else None


def aggregate_repeatability(records: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    """Aggregate a fixed repeat definition documented by the live V1 evaluator."""

    if not records:
        return {
            "target_resolve_rate": None,
            "injection_hit_rate": None,
            "effect_repeatability": None,
            "exact_recovery_rate": None,
            "crash_rate": None,
        }

    def rate(field: str) -> float:
        return sum(record.get(field) is True for record in records) / len(records)

    return {
        "target_resolve_rate": rate("resolved"),
        "injection_hit_rate": rate("hit"),
        "effect_repeatability": rate("effect_observed"),
        "exact_recovery_rate": rate("recovery_exact"),
        "crash_rate": rate("crash"),
    }


def aggregate_per_rank(
    status: Mapping[str, Any],
    matching_calls: Mapping[int, int],
    worker_pids: Mapping[int, int],
    worker_alive: Mapping[int, bool],
    *,
    expected_workers: int,
    target_device_mask: int,
) -> dict[str, Any]:
    """Normalize Fault20 v17 status into explicit rank-aware evidence.

    ``reject_count`` is evaluator-derived from matching QMM calls that the native
    candidate counter did not accept. This is recorded separately from native
    counters to avoid presenting an inferred count as a runtime field.
    """

    candidates = status.get("candidate_count_per_device")
    hits = status.get("hit_count_per_device")
    last = status.get("last_status_per_device")
    candidates = candidates if isinstance(candidates, list) else []
    hits = hits if isinstance(hits, list) else []
    last = last if isinstance(last, list) else []
    ranks: dict[str, Any] = {}
    for rank in range(expected_workers):
        candidate_count = int(candidates[rank]) if rank < len(candidates) else None
        hit_count = int(hits[rank]) if rank < len(hits) else None
        calls = matching_calls.get(rank, 0)
        inferred_reject = None if candidate_count is None else max(0, calls - candidate_count)
        targeted = bool(target_device_mask & (1 << rank))
        reasons: dict[str, int] = {}
        if inferred_reject:
            reasons["device_selector_expected" if not targeted else "signature_or_budget"] = inferred_reject
        ranks[str(rank)] = {
            "pid": worker_pids.get(rank),
            "alive": worker_alive.get(rank),
            "resolved": calls > 0,
            "matching_call_count": calls,
            "candidate_count": candidate_count,
            "hit_count": hit_count,
            "reject_count": inferred_reject,
            "reject_count_source": "evaluator_inference",
            "reject_reasons": reasons,
            "expected_reject": not targeted,
            "last_status": int(last[rank]) if rank < len(last) else None,
        }
    observed = sum(value is not None for value in worker_pids.values())
    return {
        "expected_worker_count": expected_workers,
        "observed_worker_count": observed,
        "ranks": ranks,
    }
