"""Strict statistics and two independent evidence chains; no tensor exports."""
from dataclasses import dataclass
import hashlib
import json
import math
from .schema import INTEGRATION_STATUS, REQUEST_MAPPING
from .transform import pressure_assessment

COUNT_FIELDS = ("total", "finite", "zero", "positive", "negative", "nan", "posinf", "neginf")
METRIC_FIELDS = ("min", "max", "max_abs", "mean_abs")


def scalar_tag(value):
    """JSON-safe scalar encoding keeps null, NaN, and signed infinity distinct."""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "+Inf" if value > 0 else "-Inf"
    return value


def summarize_values(values):
    """Local CPU oracle. Returned statistics contain no original tensor values."""
    counts = dict.fromkeys(COUNT_FIELDS, 0)
    low = high = max_abs = None
    sum_abs = 0.0
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Numeric samples required; null is not zero")
        counts["total"] += 1
        if math.isnan(value):
            counts["nan"] += 1
        elif math.isinf(value):
            counts["posinf" if value > 0 else "neginf"] += 1
        else:
            counts["finite"] += 1
            counts["zero" if value == 0 else "positive" if value > 0 else "negative"] += 1
            low = value if low is None else min(low, value)
            high = value if high is None else max(high, value)
            max_abs = abs(value) if max_abs is None else max(max_abs, abs(value))
            sum_abs += abs(value)
    counts.update(min=low, max=high, max_abs=max_abs,
                  mean_abs=sum_abs / counts["finite"] if counts["finite"] else None)
    validate_statistics(counts)
    return counts


def validate_statistics(stats):
    missing = set(COUNT_FIELDS + METRIC_FIELDS) - set(stats)
    if missing:
        raise ValueError("INCOMPLETE_STATISTICS: " + ",".join(sorted(missing)))
    if any(type(stats[k]) is not int or stats[k] < 0 for k in COUNT_FIELDS):
        raise ValueError("INVALID_STATISTICS_COUNTS")
    if stats["total"] != stats["finite"] + stats["nan"] + stats["posinf"] + stats["neginf"]:
        raise ValueError("STATISTICS_TOTAL_MISMATCH: zero is a subset of finite")
    if stats["finite"] != stats["zero"] + stats["positive"] + stats["negative"]:
        raise ValueError("FINITE_SIGN_COUNTS_MISMATCH")
    metrics = [stats[k] for k in METRIC_FIELDS]
    if stats["finite"] == 0:
        if any(v is not None for v in metrics):
            raise ValueError("EMPTY_FINITE_METRICS_MUST_BE_NULL")
    else:
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in metrics):
            raise ValueError("INVALID_FINITE_METRICS")
        if stats["min"] > stats["max"] or stats["mean_abs"] < 0 or stats["max_abs"] < stats["mean_abs"]:
            raise ValueError("INCONSISTENT_FINITE_METRICS")
    return stats


def _canonical(value):
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return scalar_tag(value)


def response_digest(response):
    # Restrict to observable model response, never request metadata or tensors.
    selected = {key: response.get(key) for key in ("choices", "usage", "error")}
    return hashlib.sha256(json.dumps(_canonical(selected), sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


class Fault39OutputInspector:
    """Validate the native chain without consuming msprobe first-difference data."""
    def inspect(self, native, baseline_response=None, fault_response=None, recovery=None):
        observation = native.get("last_observation", {})
        incomplete = []
        for name in ("original_self", "original_mat2", "temporary_self", "temporary_mat2", "output"):
            try:
                validate_statistics(observation[name])
            except (KeyError, TypeError, ValueError) as exc:
                incomplete.append(f"{name}: {exc}")
        counts = native.get("counters", {})
        hit_count = counts.get("hit_count", native.get("hits", native.get("hit_count", 0)))
        proof = observation.get("proof", {})
        tensors = proof.get("tensors", [])
        generation_bound = (type(native.get("generation")) is int and native["generation"] > 0
            and observation.get("generation") == native["generation"] == proof.get("generation")
            and bool(native.get("owner")) and observation.get("owner") == native["owner"]
            and type(hit_count) is int and hit_count > 0 and observation.get("hit") == hit_count == proof.get("hit"))
        slots = proof.get("slots", [])
        slot_proven = (proof.get("formal_order_proven") is True and len(slots) == 3
            and all(type(s) is int and s >= 0 for s in slots) and len(set(slots)) == 3)
        addresses_proven = (len(tensors) == 3 and all(t.get("original_address") not in (None, 0, "0", "0x0")
            and t.get("run_address") not in (None, 0, "0", "0x0") for t in tensors))
        mask = proof.get("input_mask")
        replaced = addresses_proven and type(mask) is int and mask in (1, 2, 3) and all(
            (tensors[i]["original_address"] != tensors[i]["run_address"]) == bool(mask & (1 << i)) for i in range(2))
        if addresses_proven:
            replaced = replaced and tensors[2]["original_address"] == tensors[2]["run_address"]
        real_run = observation.get("run_status") == 0
        loaded = native.get("runtime_loaded") is True
        intercepted = native.get("interposed_run_calls", 0) > 0 and native.get("typed_metadata_calls", 0) > 0
        execution = bool(loaded and intercepted and hit_count > 0 and real_run and replaced and slot_proven and generation_bound)
        finite_inputs = not incomplete and all(observation[n]["finite"] == observation[n]["total"]
            for n in ("original_self", "original_mat2", "temporary_self", "temporary_mat2"))
        terminal = (native.get("inflight") == 0 and native.get("leases") == 0 and native.get("last_error") == 0
            and native.get("status") in ("COMPLETED", "DISABLED", "DISABLED_CONFIRMED"))
        released = terminal and all(observation.get(key) == 0 for key in ("pre_sync_status", "copy_status", "post_sync_status", "free_status"))
        output_readback = observation.get("output_source") == "post_run_device_readback"
        output = observation.get("output", {})
        anomaly = sum(output.get(k, 0) for k in ("nan", "posinf", "neginf")) > 0
        exact_zero, changed_valid, dimensions, pressure = False, False, None, None
        if not incomplete and len(tensors) == 3:
            try:
                m, k = tensors[0]["shape"]
                kb, n = tensors[1]["shape"]
                shape_valid = (all(type(v) is int and v > 0 for v in (m, k, kb, n))
                    and k == kb and k % 2 == 0 and list(tensors[2]["shape"]) == [m, n])
                if shape_valid:
                    dimensions = {"M": m, "K": k, "N": n}
                    after_a, after_b = observation["temporary_self"], observation["temporary_mat2"]
                    totals = [m * k, k * n]
                    changed, unchanged = observation.get("changed", []), observation.get("unchanged", [])
                    changed_valid = (len(changed) == len(unchanged) == 2 and all(type(v) is int and v >= 0 for v in changed + unchanged)
                        and all(changed[i] + unchanged[i] == totals[i] for i in range(2)))
                    magnitude = after_a["min"]
                    exact_zero = (mask == 3 and after_a["total"] == after_a["positive"] == m * k
                        and observation.get("per_column_sign_counts_verified") is True
                        and observation.get("self_constant_positive_verified") is True
                        and after_a["min"] == after_a["max"] and magnitude > 0
                        and after_b["total"] == k * n and after_b["positive"] == after_b["negative"] == k * n // 2
                        and after_b["zero"] == 0 and after_b["min"] == -magnitude and after_b["max"] == magnitude
                        and after_b["mean_abs"] == after_b["max_abs"] == magnitude
                        and output["total"] == m * n)
                    exponent = observation.get("exponent")
                    if type(exponent) is int and 0 <= exponent <= 63:
                        pressure = pressure_assessment(k, exponent)
                        exact_zero = exact_zero and magnitude == 2**exponent
                    else:
                        exact_zero = False
            except (TypeError, ValueError, KeyError, OverflowError):
                exact_zero = changed_valid = False
        verdict = "INSUFFICIENT_NATIVE_EVIDENCE"
        if execution and finite_inputs and released and output_readback and changed_valid:
            verdict = "MATMUL_FINITE_INPUT_NONFINITE_REPRODUCED" if anomaly else "REAL_OPERATOR_HIT_VERIFIED"
            if anomaly and exact_zero and sum(observation["changed"]) > 0:
                verdict = "MATMUL_INTERMEDIATE_RANGE_FAILURE_REPRODUCED"
        try:
            request_ids = json.loads(observation.get("actual_request_ids_json", "[]"))
        except (TypeError, ValueError):
            request_ids = []
        mapping = "SINGLE_REQUEST_SCOPE_OBSERVED" if isinstance(request_ids, list) and len(request_ids) == 1 and isinstance(request_ids[0], str) and request_ids[0] and generation_bound else REQUEST_MAPPING
        # Request scope alone never establishes msprobe dump-to-request mapping.
        result = {"schema_version": 1, "verdict": verdict, "native_execution_verified": execution,
                  "finite_inputs_verified": bool(finite_inputs), "resources_released": released,
                  "range_anomaly": "RANGE_ANOMALY_OBSERVED" if anomaly else "RANGE_ANOMALY_NOT_OBSERVED",
                  "statistics_errors": incomplete, "native_request_scope": mapping,
                  "request_mapping": REQUEST_MAPPING, "inner_framework_integration": INTEGRATION_STATUS,
                  "evidence_kind": "NATIVE_STATISTICS", "tensor_payloads": False,
                  "input_device_readback_verified": observation.get("input_device_readback_verified") is True,
                  "temporary_source": observation.get("temporary_source"),
                  "output_source": observation.get("output_source"), "dimensions": dimensions,
                  "exact_cancellation_verified": exact_zero, "pressure": pressure,
                  "statistics": {name: observation.get(name) for name in ("original_self", "original_mat2", "temporary_self", "temporary_mat2", "output")}}
        if baseline_response is not None and fault_response is not None:
            baseline_digest = response_digest(baseline_response)
            fault_digest = response_digest(fault_response)
            result["model_response"] = {"baseline_digest": baseline_digest, "fault_digest": fault_digest,
                                        "changed": baseline_digest != fault_digest,
                                        "baseline_usage": baseline_response.get("usage"),
                                        "fault_usage": fault_response.get("usage")}
        if recovery is not None:
            result["recovery"] = {"verified": recovery.get("disable_status") == "DISABLED_CONFIRMED"
                and recovery.get("hits_before") == recovery.get("hits_after")
                and type(recovery.get("hits_before")) is int
                and recovery.get("service_healthy") is True
                and recovery.get("response_equal") is True
                and recovery.get("leases") == 0
                and recovery.get("inflight") == 0}
        return result
