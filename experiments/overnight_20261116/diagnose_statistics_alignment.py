"""Diagnose raw statistics alignment without claiming a global first difference.

Allowed inputs are capture/config/collector metadata and raw dump/stack/construct
JSON. This does not read fault hit logs, validation outputs or tensor payloads.
The existing strict comparator is rerun unchanged and its result is retained.
Exact-name pairs below are limited diagnostics, never a relaxed acceptance gate.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "fault39" / "tools"))
from compare_model_capture import actual_steps, compare_groups
from compare_msprobe import ComparisonPolicy, load_raw

METRICS = ("Max", "Min", "Mean", "Norm")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("+Inf" if value > 0 else "-Inf")
    if isinstance(value, dict):
        return {key: json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [json_safe(child) for child in value]
    return value


def stripped(value):
    if isinstance(value, dict):
        return {key: stripped(child) for key, child in value.items()
                if not (value.get("type") == "torch.Tensor" and key in METRICS)}
    if isinstance(value, list):
        return [stripped(child) for child in value]
    return value


def structural_changes(b, f, path=""):
    """Keep non-statistical changes, including scalar arguments and key order."""
    if type(b) is not type(f):
        return [{"path": path, "reason": "TYPE_CHANGED", "baseline": b, "fault": f}]
    if isinstance(b, dict):
        result = []
        if list(b) != list(f):
            result.append({"path": path, "reason": "KEYS_OR_ORDER_CHANGED",
                           "baseline": list(b), "fault": list(f)})
        for key in sorted(set(b) - set(f)):
            result.append({"path": path + "/" + key, "reason": "MISSING_FROM_FAULT", "baseline": b[key]})
        for key in sorted(set(f) - set(b)):
            result.append({"path": path + "/" + key, "reason": "MISSING_FROM_BASELINE", "fault": f[key]})
        for key in b:
            if key in f:
                result.extend(structural_changes(b[key], f[key], path + "/" + key))
        return result
    if isinstance(b, list):
        result = []
        if len(b) != len(f):
            result.append({"path": path, "reason": "LENGTH_CHANGED", "baseline": len(b), "fault": len(f)})
        for i, (left, right) in enumerate(zip(b, f)):
            result.extend(structural_changes(left, right, path + "/" + str(i)))
        return result
    return [] if b == f else [{"path": path, "reason": "VALUE_CHANGED", "baseline": b, "fault": f}]


def tensor_records(value, path=""):
    if isinstance(value, dict):
        if value.get("type") == "torch.Tensor":
            yield path, value
        else:
            for key, child in value.items():
                yield from tensor_records(child, path + "/" + key)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from tensor_records(child, path + "/" + str(i))


def stack_by_operator(stack):
    """Resolve collector stack-group IDs; group numbering is not a callsite."""
    result = {}
    for group in stack.values():
        if not isinstance(group, list) or len(group) != 2:
            raise ValueError("UNSUPPORTED_STACK_SCHEMA")
        operators, frames = group
        if not isinstance(operators, list) or not isinstance(frames, list):
            raise ValueError("UNSUPPORTED_STACK_SCHEMA")
        for operator in operators:
            if operator in result:
                raise ValueError("DUPLICATE_OPERATOR_STACK: " + operator)
            result[operator] = frames
    return result


def family(operator):
    # Never pair a shifted occurrence when one side has additional calls of the
    # same API/module family. Unique context is still only a diagnostic pairing.
    return re.sub(r"\.forward\.[0-9]+$", ".forward.#",
                  re.sub(r"\.[0-9]+\.forward$", ".#.forward", operator))


def number(value):
    if value is None:
        return "null", None
    if isinstance(value, bool):
        return "invalid", None
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return "invalid", None
    if not isinstance(value, (float, int)):
        return "invalid", None
    if math.isnan(value):
        return "nan", None
    if math.isinf(value):
        return "inf", value
    return "finite", value


def metric(b, f, policy):
    result = {"baseline": b, "fault": f, "exact_change": None,
              "exceeds_tolerance": None, "relative_change": None}
    bk, bn = number(b)
    fk, fn = number(f)
    if "null" in (bk, fk):
        return dict(result, status="UNOBSERVABLE")
    if "nan" in (bk, fk) or "invalid" in (bk, fk):
        return dict(result, status="UNCOMPARABLE")
    if "inf" in (bk, fk):
        return dict(result, status="NONFINITE_OBSERVED", exact_change=bn != fn,
                    exceeds_tolerance=None)
    delta = fn - bn
    threshold = policy.absolute_tolerance + policy.relative_tolerance * abs(bn)
    return dict(result, status="OBSERVABLE_FINITE", exact_change=bn != fn,
                signed_difference=delta, absolute_difference=abs(delta), threshold=threshold,
                exceeds_tolerance=abs(delta) > threshold,
                relative_change=abs(delta) / abs(bn) if bn != 0 else None,
                relative_change_reason=None if bn != 0 else "ZERO_BASELINE_NO_RELATIVE_VALUE")


def diagnose_step(bpath, fpath, policy, *, capture_metadata_aligned=True):
    # Reuse production schema validation; no normalized export replaces raw JSON.
    br, fr = load_raw(bpath), load_raw(fpath)
    b = read_json(Path(bpath) / "dump.json")["data"]
    f = read_json(Path(fpath) / "dump.json")["data"]
    bs, fs = stack_by_operator(br["stack"]), stack_by_operator(fr["stack"])
    bc, fc = br["construct"], fr["construct"]
    bcounts, fcounts = Counter(map(family, b)), Counter(map(family, f))
    common = [key for key in b if key in f]
    same_common_order = common == [key for key in f if key in b]
    report = {"baseline_raw_sha256": br["hashes"], "fault_raw_sha256": fr["hashes"],
              "baseline_operator_order": list(b), "fault_operator_order": list(f),
              "common_operator_relative_order_equal": same_common_order,
              "capture_metadata_aligned": capture_metadata_aligned,
              "strict_structure_equal": br["structure"] == fr["structure"],
              "strict_stack_equal": br["stack"] == fr["stack"],
              "strict_construct_equal": bc == fc,
              "unpaired": {"baseline": [], "fault": []}, "paired_diagnostics": [],
              "changed_paired_fields": [], "unobservable_paired_fields": [],
              "global_first_difference": None}
    for side, data, other, stacks, construct in (("baseline", b, f, bs, bc), ("fault", f, b, fs, fc)):
        for key in data:
            if key not in other:
                report["unpaired"][side].append({
                    "operator": key, "kind": "module" if key.startswith("Module.") else "api",
                    "reason": "NO_EXACT_OPERATOR_ID_ON_OTHER_SIDE", "raw_record": data[key],
                    "stack": stacks.get(key), "construct_present": key in construct,
                    "construct": construct.get(key)})
    for key in common:
        changes = structural_changes(stripped(b[key]), stripped(f[key]), key)
        stack_equal = key in bs and key in fs and bs[key] == fs[key]
        construct_equal = key in bc and key in fc and bc[key] == fc[key]
        reasons = []
        if not capture_metadata_aligned:
            reasons.append("CAPTURE_CONFIGURATION_OR_COLLECTOR_UNALIGNED")
        if changes:
            reasons.append("NON_STATISTICAL_RECORD_CHANGED")
        if not stack_equal:
            reasons.append("STACK_UNALIGNED_OR_MISSING")
        if not construct_equal:
            reasons.append("CONSTRUCT_UNALIGNED_OR_MISSING")
        if bcounts[family(key)] != fcounts[family(key)]:
            reasons.append("OCCURRENCE_FAMILY_COUNT_MISMATCH")
        if not same_common_order:
            reasons.append("COMMON_OPERATOR_ORDER_CHANGED")
        item = {"operator": key, "kind": "module" if key.startswith("Module.") else "api",
                "pairing_status": "REJECTED" if reasons else "ELIGIBLE_PARTIAL_PAIR_ONLY",
                "reasons": reasons, "nonstatistical_changes": changes, "stack_equal": stack_equal,
                "construct_equal": construct_equal,
                "construct_null": bc.get(key) is None or fc.get(key) is None,
                "raw_stack": bs.get(key) if stack_equal else {"baseline": bs.get(key), "fault": fs.get(key)},
                "tensor_statistics": []}
        if not reasons:
            ftensors = dict(tensor_records(f[key]))
            for path, tensor in tensor_records(b[key]):
                fields = {name: metric(tensor[name], ftensors[path][name], policy) for name in METRICS}
                item["tensor_statistics"].append({"identity": key + path,
                    "description": stripped(tensor), "fields": fields})
                for name, values in fields.items():
                    entry = {"identity": key + path, "field": name, **values}
                    if values["exceeds_tolerance"] is True:
                        report["changed_paired_fields"].append(entry)
                    if values["status"] in ("UNOBSERVABLE", "UNCOMPARABLE", "NONFINITE_OBSERVED"):
                        report["unobservable_paired_fields"].append(entry)
        else:
            item["raw_baseline_record"] = b[key]
            item["raw_fault_record"] = f[key]
        report["paired_diagnostics"].append(item)
    return report


def diagnose_groups(baseline, fault, policy, rank=0):
    baseline, fault = Path(baseline), Path(fault)
    strict = compare_groups(baseline, fault, policy, rank)
    bsteps, fsteps = actual_steps(baseline), actual_steps(fault)
    report = {"schema_version": 1, "purpose": "RAW_STATISTICS_ALIGNMENT_DIAGNOSIS_ONLY",
              "source": "RAW_MSPROBE_STATISTICS_MIX", "native_logs_read": False,
              "tensor_payloads_read": False, "generated_responses_read": False,
              "comparison_policy": {"rtol": policy.relative_tolerance,
                    "atol": policy.absolute_tolerance, "rule": "abs(F-B) > atol + rtol*abs(B)"},
              "diagnostic_source_sha256": digest(__file__),
              "strict_comparator_sha256": {name: digest(ROOT / "fault39" / "tools" / name)
                   for name in ("compare_msprobe.py", "compare_model_capture.py")},
              "strict_original_result": strict, "strict_acceptance_overridden": False,
              "global_first_difference": None, "steps": [],
              "status": "DIAGNOSTICS_ONLY_STRICT_ALIGNMENT_RETAINED",
              "limitations": [
                  "Pairing preserves exact operator IDs, non-statistical metadata, resolved stacks and construct entries; it is not a sequence realignment acceptance gate.",
                  "Unpaired APIs, changed scalar arguments and missing context remain explicit; they are not filtered out.",
                  "Common-ID local statistics do not prove globally earliest change when other records are unpaired.",
                  "Equal observable Max/Min/Mean/Norm do not prove tensor equality; null is never counted as equality.",
                  "Null construct entries do not independently establish tensor aliasing or producer-consumer identity.",
                  "Raw stack source lines are evidence from the collector, not separately read injector code or hit logs.",
                  "Actual collector steps are not inferred decode tokens; no request identity is inferred."]}
    for step in sorted(set(bsteps) | set(fsteps)):
        entry = {"step": step, "rank": rank}
        if step not in bsteps or step not in fsteps:
            entry.update(status="UNPAIRED_STEP", missing_from="baseline" if step not in bsteps else "fault")
        else:
            prior = next(item["report"] for item in strict["steps"] if item["step"] == step)
            metadata_aligned = "comparison" in prior or (
                prior.get("status") == "ALIGNMENT_MISMATCH"
                and prior.get("mismatched") in ("structure", "stack", "construct"))
            entry.update(diagnose_step(bsteps[step] / f"rank{rank}", fsteps[step] / f"rank{rank}",
                                      policy, capture_metadata_aligned=metadata_aligned))
        report["steps"].append(entry)
    return report


def scalar_records(value, path=""):
    if isinstance(value, dict):
        if "type" in value and value.get("type") != "torch.Tensor":
            yield {"path": path, "record": value}
        elif value.get("type") != "torch.Tensor":
            for key, child in value.items():
                yield from scalar_records(child, path + "/" + key)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from scalar_records(child, path + "/" + str(i))


def compact_report(report, policy):
    """Publish a diagnostic summary; keep complete raw-derived detail private."""
    result = {key: value for key, value in report.items() if key != "steps"}
    result["summary_only"] = True
    result["steps"] = []
    for row in report["steps"]:
        entry = {key: value for key, value in row.items()
                 if key not in ("unpaired", "paired_diagnostics", "unobservable_paired_fields")}
        entry["unpaired"] = {}
        for side, items in row.get("unpaired", {}).items():
            entry["unpaired"][side] = [{"operator": item["operator"], "kind": item["kind"],
                "reason": item["reason"], "scalar_records": list(scalar_records(item["raw_record"])),
                "first_raw_stack_frame": (item["stack"] or [None])[0],
                "construct_present": item["construct_present"], "construct": item["construct"]}
                for item in items]
        entry["paired_diagnostics"] = []
        entry["unobservable_paired_fields"] = [{key: item[key] for key in ("identity", "field", "status")}
            for item in row.get("unobservable_paired_fields", [])]
        for item in row.get("paired_diagnostics", []):
            detail = {key: value for key, value in item.items()
                      if key not in ("raw_stack", "raw_baseline_record", "raw_fault_record", "tensor_statistics")}
            detail["tensor_statistics_count"] = len(item["tensor_statistics"])
            # Keep module boundary observations; API changes are already recorded
            # in changed_paired_fields. This is not a replacement raw dump.
            if item["kind"] == "module":
                detail["module_statistics"] = item["tensor_statistics"]
            if item["pairing_status"] == "REJECTED":
                b = dict(tensor_records(item["raw_baseline_record"]))
                f = dict(tensor_records(item["raw_fault_record"]))
                observations = []
                for path in b:
                    if path not in f or stripped(b[path]) != stripped(f[path]):
                        continue
                    for name in METRICS:
                        fields = metric(b[path][name], f[path][name], policy)
                        if fields["exact_change"] is True:
                            observations.append({"identity": item["operator"] + path, "field": name,
                                "record_pairing_accepted": False, "arithmetic_only_not_comparison_acceptance": True,
                                **fields})
                detail["rejected_record_same_path_arithmetic"] = observations
            entry["paired_diagnostics"].append(detail)
        result["steps"].append(entry)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fault", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--prior-strict-report", type=Path)
    parser.add_argument("--private-details-output", type=Path,
                        help="Optional private full raw-derived diagnostic; never changes capture files.")
    args = parser.parse_args()
    report = diagnose_groups(args.baseline, args.fault, ComparisonPolicy(args.atol, args.rtol), args.rank)
    if args.prior_strict_report:
        prior = read_json(args.prior_strict_report)
        report["prior_strict_report"] = {"sha256": digest(args.prior_strict_report),
            "same_report_as_recomputed": prior == report["strict_original_result"],
            "status": prior.get("status"), "all_steps_aligned": prior.get("all_steps_aligned"),
            "first_difference": prior.get("first_difference"), "blocking_steps": prior.get("blocking_steps")}
    output_documents = [(args.output, compact_report(report, ComparisonPolicy(args.atol, args.rtol)))]
    if args.private_details_output:
        output_documents.append((args.private_details_output, report))
    for path, document in output_documents:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(json_safe(document), stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"status": report["status"], "strict_status": report["strict_original_result"]["status"],
          "steps": [{"step": row["step"], "changed_paired_fields": len(row.get("changed_paired_fields", [])),
                     "unpaired_baseline": len(row.get("unpaired", {}).get("baseline", [])),
                     "unpaired_fault": len(row.get("unpaired", {}).get("fault", []))} for row in report["steps"]]}))


if __name__ == "__main__":
    main()
