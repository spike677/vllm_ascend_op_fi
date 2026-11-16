"""Independent statistics first-difference analysis, with no native-log input."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from .inspector import scalar_tag
from .schema import REQUEST_MAPPING


@dataclass(frozen=True)
class ComparisonPolicy:
    absolute_tolerance: float = 0.0
    relative_tolerance: float = 0.0

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in (self.absolute_tolerance, self.relative_tolerance)):
            raise ValueError("Finite nonnegative tolerances required")


def _number_kind(value):
    if value is None:
        return "null", None
    if isinstance(value, bool):
        return "bool", value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in ("nan", "+nan", "-nan"):
            return "nan", None
        if token in ("inf", "+inf", "infinity", "+infinity"):
            return "+inf", None
        if token in ("-inf", "-infinity"):
            return "-inf", None
        return "text", value
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return "nan", None
        if math.isinf(value):
            return "+inf" if value > 0 else "-inf", None
        return "finite", value
    return "other", value


def statistics_equal(left, right, policy=ComparisonPolicy()):
    kind_a, a = _number_kind(left)
    kind_b, b = _number_kind(right)
    if kind_a != kind_b:
        return False
    if kind_a in ("nan", "null"):
        # Missing statistics and paired NaNs are not observed equality.
        return False
    if kind_a == "finite":
        return abs(b - a) <= policy.absolute_tolerance + policy.relative_tolerance * abs(a)
    return a == b


def _first_value_difference(left, right, policy, path="", unobservable=None):
    if left is None or right is None:
        if unobservable is not None:
            unobservable.append(path)
        return None
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return {"field": path, "reason": "STATISTIC_KEYS_DIFFER", "baseline_keys": sorted(left), "fault_keys": sorted(right)}
        for key in sorted(left):
            found = _first_value_difference(left[key], right[key], policy, path + "/" + str(key), unobservable)
            if found:
                return found
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return {"field": path, "reason": "STATISTIC_LENGTH_DIFFERS", "baseline_length": len(left), "fault_length": len(right)}
        for index, (a, b) in enumerate(zip(left, right)):
            found = _first_value_difference(a, b, policy, path + "/" + str(index), unobservable)
            if found:
                return found
        return None
    if isinstance(left, (list, dict)) or isinstance(right, (list, dict)):
        return {"field": path, "reason": "STATISTIC_TYPE_DIFFERS"}
    if statistics_equal(left, right, policy):
        return None
    return {"field": path, "reason": "NONFINITE_UNCOMPARABLE" if "nan" in (_number_kind(left)[0], _number_kind(right)[0]) else "STATISTIC_DIFFERS",
            "baseline": scalar_tag(left), "fault": scalar_tag(right)}


def first_difference(baseline_records, fault_records, policy=ComparisonPolicy()):
    """Compare collector-ordered records with explicit identity and statistics.

    The normalized record schema is {identity: string, statistics: object}.
    No hit log, candidate signature, or target callsite is accepted as an input.
    Sequence alignment failure is evidence insufficiency, not a guessed match.
    """
    base, fault = list(baseline_records), list(fault_records)
    for record in base + fault:
        if set(record) != {"identity", "statistics"} or not isinstance(record["identity"], str) or not isinstance(record["statistics"], dict):
            raise ValueError("Normalized collector statistics records required")
    report = {"schema_version": 1, "source": "MSPROBE_STATISTICS_ONLY",
              "request_mapping": REQUEST_MAPPING, "tensor_payloads": False,
              "comparison_policy": {"absolute_tolerance": policy.absolute_tolerance,
                                    "relative_tolerance": policy.relative_tolerance,
                                    "rule": "abs(F-B) > atol + rtol*abs(B)"}}
    if not base or not fault:
        return dict(report, status="INSUFFICIENT_STATISTICS", baseline_count=len(base), fault_count=len(fault))
    unobservable = []
    for index, (a, b) in enumerate(zip(base, fault)):
        if a["identity"] != b["identity"]:
            return dict(report, status="ALIGNMENT_MISMATCH", index=index,
                        baseline_identity=a["identity"], fault_identity=b["identity"])
        if not a["statistics"] or not b["statistics"]:
            return dict(report, status="INSUFFICIENT_STATISTICS", index=index, identity=a["identity"])
        local_unobservable = []
        difference = _first_value_difference(a["statistics"], b["statistics"], policy, unobservable=local_unobservable)
        unobservable.extend({"index": index, "identity": a["identity"], "field": field} for field in local_unobservable)
        if difference:
            return dict(report, status="FIRST_OBSERVABLE_STATISTICS_DIFFERENCE" if unobservable else "FIRST_STATISTICS_DIFFERENCE", index=index,
                        identity=a["identity"], difference=difference, unobservable_fields=unobservable)
    if len(base) != len(fault):
        return dict(report, status="ALIGNMENT_MISMATCH", index=min(len(base), len(fault)),
                    baseline_count=len(base), fault_count=len(fault))
    return dict(report, status="NO_OBSERVABLE_STATISTICS_DIFFERENCE" if unobservable else "NO_STATISTICS_DIFFERENCE",
                records_compared=len(base), unobservable_fields=unobservable)


def load_statistics_records(path):
    """Read normalized statistics JSON, never .npy/.pt/raw tensor payloads.

    Existing msprobe dump schemas vary. A collector-specific adapter must emit
    ordered normalized records and preserve source hashes; this is not a parser
    that silently treats arbitrary dump.json structures as validated records.
    """
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError("STATISTICS_JSON_REQUIRED")
    content = path.read_bytes()
    document = json.loads(content)
    if not isinstance(document, dict) or document.get("schema") != "fault39.msprobe.statistics.v1":
        raise ValueError("UNSUPPORTED_STATISTICS_SCHEMA")
    if document.get("mode") not in ("statistics", "mix") or document.get("tensor_payloads") is not False:
        raise ValueError("STATISTICS_ONLY_EXPORT_REQUIRED")
    records = document.get("records")
    if not isinstance(records, list):
        raise ValueError("RECORDS_REQUIRED")
    # Validation is shared with the comparator, including unknown-field rejection.
    first_difference(records, records)
    return {"records": records, "source_sha256": hashlib.sha256(content).hexdigest(),
            "collector_identity": document.get("collector_identity"),
            "request_mapping": REQUEST_MAPPING}
