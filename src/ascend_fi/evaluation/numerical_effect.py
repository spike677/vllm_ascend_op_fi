"""Read-only sampled-energy derivation; not a fault or detector policy.

The source population is rank-local activation element *events*, not unique
parameters or independent statistical samples.  Never reconstruct energies from
rounded relative-L2 displays and never average module-level dB values.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

SCHEMA = "SAMPLED_NUMERICAL_EFFECT_V1"
SOURCE_SCHEMA = "BOUNDED_EFFECT_SUMMARY_V1"
SAMPLING = "first call and every 64 subsequent calls per module; reset on arm"


def _finite(value):
    try:
        return math.isfinite(value)
    except (TypeError, OverflowError):
        return False


def _number(value):
    return type(value) in (float, int) and _finite(value) and value >= 0


def derive_sampled_energy(error_energy, reference_energy, *, sampled_elements,
                          sampled_calls):
    """Derive JSON-safe metrics from the same unrounded sampled E/S population.

    Values are null when absent/undefined/unrepresentable, with an explicit
    status.  In particular zero reference energy is not a normal zero error.
    """
    result = {
        "status": "MISSING", "sampled_error_sq_sum": None,
        "sampled_input_sq_sum": None, "sampled_elements": None,
        "sampled_calls": None, "relative_l2": None, "nmse": None,
        "sqnr_db": None, "relative_l2_status": "UNAVAILABLE",
        "nmse_status": "UNAVAILABLE", "sqnr_status": "UNAVAILABLE",
    }
    values = (error_energy, reference_energy, sampled_elements, sampled_calls)
    if any(v is None for v in values):
        return result
    if not all(type(v) is int and v >= 0 for v in (sampled_elements, sampled_calls)):
        result["status"] = "INVALID_COUNTS"
        return result
    result.update(sampled_elements=sampled_elements, sampled_calls=sampled_calls)
    if any(type(v) in (float, int) and not _finite(v)
           for v in (error_energy, reference_energy)):
        result["status"] = "NONFINITE_UPSTREAM"
        return result
    if not all(_number(v) for v in (error_energy, reference_energy)):
        result["status"] = "INVALID_ENERGY"
        return result
    result.update(sampled_error_sq_sum=error_energy,
                  sampled_input_sq_sum=reference_energy)
    if sampled_elements == 0 or sampled_calls == 0:
        result["status"] = ("NO_SAMPLES" if sampled_elements == sampled_calls == 0
                            and error_energy == reference_energy == 0 else "INCONSISTENT_COUNTS")
        return result
    if sampled_elements < sampled_calls:
        result["status"] = "INCONSISTENT_COUNTS"
        return result
    if reference_energy == 0:
        result["status"] = "ZERO_REFERENCE_ENERGY"
        result["nmse_status"] = "UNDEFINED_0_OVER_0" if error_energy == 0 else "POSITIVE_INFINITY"
        result["relative_l2_status"] = result["nmse_status"]
        result["sqnr_status"] = "UNDEFINED_0_OVER_0" if error_energy == 0 else "NEGATIVE_INFINITY"
        return result
    if error_energy == 0:
        result.update(status="ZERO_ERROR", nmse=0.0, relative_l2=0.0,
                      nmse_status="FINITE", relative_l2_status="FINITE",
                      sqnr_status="POSITIVE_INFINITY")
        return result
    # Log-domain SQNR remains finite even if the ratio over/underflows float64.
    log_ratio = math.log(error_energy) - math.log(reference_energy)
    ratio = error_energy / reference_energy
    root = math.sqrt(ratio) if math.isfinite(ratio) and ratio > 0 else None
    if root is None:
        try:
            root = math.exp(log_ratio / 2)
        except OverflowError:
            root = math.inf
    result.update(status="MEASURED", sqnr_db=-10 * log_ratio / math.log(10),
                  sqnr_status="FINITE")
    for key, value in (("nmse", ratio), ("relative_l2", root)):
        state = "FINITE" if math.isfinite(value) and value > 0 else "OVERFLOW" if math.isinf(value) else "UNDERFLOW"
        result[key + "_status"] = state
        result[key] = value if state == "FINITE" else None
    return result


def _aggregate(rows):
    if not rows:
        return derive_sampled_energy(None, None, sampled_elements=None, sampled_calls=None)
    allowed = {"MEASURED", "ZERO_ERROR", "ZERO_REFERENCE_ENERGY", "NO_SAMPLES"}
    if any(row["status"] not in allowed for row in rows):
        result = derive_sampled_energy(None, None, sampled_elements=None, sampled_calls=None)
        result["status"] = "INCOMPLETE_INPUT"
        return result
    try:
        error = math.fsum(row["sampled_error_sq_sum"] for row in rows)
        reference = math.fsum(row["sampled_input_sq_sum"] for row in rows)
    except OverflowError:
        result = derive_sampled_energy(None, None, sampled_elements=None, sampled_calls=None)
        result["status"] = "AGGREGATE_OVERFLOW"
        return result
    return derive_sampled_energy(error, reference,
                                 sampled_elements=sum(row["sampled_elements"] for row in rows),
                                 sampled_calls=sum(row["sampled_calls"] for row in rows))


def sampled_numerical_effect(workers, *, expected_ranks=None, request_generation=None):
    """Consume one request's worker query rows, keeping ranks/modules separate.

    Each worker row has an explicit ``rank`` and an ``effect_summary``.  Callers
    must not concatenate snapshots from different requests or repeated queries:
    cumulative snapshots would double-count.  Missing ranks poison the overall
    result rather than being silently counted as zero.
    """
    workers = list(workers)
    expected = list(expected_ranks) if expected_ranks is not None else None
    if expected is not None and (any(type(r) is not int or r < 0 for r in expected)
                                 or len(expected) != len(set(expected))):
        raise ValueError("expected_ranks must be distinct nonnegative global ranks")
    issues, per_rank, seen = [], {}, set()
    all_rows = []
    for worker in workers:
        rank = worker.get("rank")
        if type(rank) is not int or rank < 0 or rank in seen:
            issues.append("MISSING_OR_DUPLICATE_RANK")
            continue
        seen.add(rank)
        summary = worker.get("effect_summary") or {}
        if summary.get("schema") != SOURCE_SCHEMA or not isinstance(summary.get("per_module"), Mapping):
            issues.append("MISSING_EFFECT_SUMMARY:rank=" + str(rank))
            continue
        if summary.get("errors") or summary.get("contiguous_hit_indices") is not True:
            issues.append("INVALID_EFFECT_SUMMARY:rank=" + str(rank))
        modules = {}
        for name, row in summary["per_module"].items():
            if row.get("rank") != rank or row.get("module") != name:
                issues.append("MODULE_IDENTITY_MISMATCH:rank=" + str(rank))
                continue
            metric = derive_sampled_energy(row.get("sampled_error_sq_sum"), row.get("sampled_input_sq_sum"),
                                            sampled_elements=row.get("sampled_elements"),
                                            sampled_calls=row.get("sampled_calls"))
            metric.update(module=name, layer=row.get("layer"), rank=rank)
            modules[name] = metric
            all_rows.append(metric)
        per_rank[str(rank)] = {"rank": rank, "per_module": modules,
                               "aggregate": _aggregate(list(modules.values()))}
    if expected is not None and seen != set(expected):
        issues.append("EXPECTED_RANK_SET_MISMATCH")
    overall = _aggregate(all_rows)
    if issues:
        overall = derive_sampled_energy(None, None, sampled_elements=None, sampled_calls=None)
        overall["status"] = "INCOMPLETE_INPUT"
    return {"schema": SCHEMA, "request_generation": request_generation,
            "status": overall["status"], "aggregate": overall, "per_rank": per_rank,
            "issues": issues, "expected_ranks": expected, "observed_ranks": sorted(seen),
            "population": "SAMPLED_RANK_LOCAL_ACTIVATION_ELEMENT_EVENTS",
            "sampling": SAMPLING, "sampling_scope": "not a full-population measurement",
            "energy_source": "unrounded EffectStats sampled_error_sq_sum / sampled_input_sq_sum",
            "source_compute_dtype": "FP32 difference and per-call sums; host float accumulation",
            "derivation_dtype": "Python float; energy-first aggregation",
            "aggregation": "sum E and S across selected module/rank events, then derive",
            "absolute_error_contract": "unchanged: reference absolute error uses original tensor dtype",
            "independent_evidence_count": 1, "detector_thresholds_changed": False}
