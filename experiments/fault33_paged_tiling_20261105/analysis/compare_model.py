"""Independently compare original full-model decode statistics from two paged-IFA cases.

Inputs are the two case/observation/step0/rank0/dump.json paths. The sibling
stack/construct, case config, collector provenance and statistics receipt are
read; host callbacks, private logits, result hashes and policy labels are not.
"""
import argparse
import json
from pathlib import Path
import re

import msprobe_compare as values

PARENT = re.compile(r"Module\.model\.layers\.(\d+)\.self_attn\.Qwen3Attention\.forward\.\d+\Z")
LAYER = re.compile(r"Module\.model\.layers\.(\d+)(?:\.|$)")
SOURCE = re.compile(r"^File (.+), line \d+, in ([^,]+),")
# These functions transfer or summarize values for the capture. pack_pages and
# PagedDecode.call change data used by the model and stay in its trace.
DIAGNOSTICS = {"cpu_array", "array_digest", "tensor_report", "physical_report", "physical_rows_after"}


def source_frame(frame):
    m = SOURCE.match(frame) if isinstance(frame, str) else None
    values.require(m is not None, "Unparsed original stack frame")
    return m[1].replace("\\", "/"), m[2].strip()


def load(path):
    path = Path(path)
    values.require(path.parts[-4:] == ("observation", "step0", "rank0", "dump.json"),
                   "Expected case/observation/step0/rank0/dump.json")
    case = path.parents[3]
    files, hashes = {}, {}
    for name, file in (("config", case / "config.json"), ("receipt", case / "statistics_receipt.json"),
                       ("dump", path), ("stack", path.with_name("stack.json")), ("construct", path.with_name("construct.json"))):
        files[name], hashes[name] = values.read(file)
    config, receipt = files["config"], files["receipt"]
    values.require(config.get("task") == "statistics" and config.get("level") == "mix"
                   and config.get("step") == [0] and config.get("rank") in ([], [0]) and config.get("async_dump") is False
                   and config["statistics"] == {"scope": [], "list": [], "data_mode": ["all"],
                   "summary_mode": "statistics", "tensor_list": []}, "Unexpected model capture settings")
    collector = values.collector(case, config, hashes)
    values.require(receipt["schema"] == "fault33.pa.model_statistics.v1" and receipt["steps"] == [0] and receipt["rank"] == 0
                   and receipt["config_sha256"] == hashes["config"] and receipt["collector_sha256"] == hashes["collector_provenance"],
                   "Model statistics receipt/config/collector mismatch")
    expected = [{"path": f"observation/step0/rank0/{n}.json", "bytes": path.with_name(n + ".json").stat().st_size,
                 "sha256": hashes[n]} for n in ("dump", "construct", "stack")]
    values.require(receipt["native_files"] == expected, "Receipt does not bind original dump/construct/stack")
    dump, construct, stack = (files[k] for k in ("dump", "construct", "stack"))
    values.require(dump.get("task") == "statistics" and dump.get("level") == "mix" and dump.get("framework") == "pytorch"
                   and dump.get("dump_data_dir") is None and isinstance(dump.get("data"), dict) and dump["data"],
                   "Missing original statistics/mix data")
    index = {}
    for row in stack.values():
        values.require(isinstance(row, list) and len(row) == 2 and isinstance(row[0], list)
                       and isinstance(row[1], list) and row[1], "Malformed raw stack record")
        frames = row[1]
        callsites = [source_frame(s) for s in frames]
        diagnostic = any(f == "/audit/model_adapter.py" and function in DIAGNOSTICS for f, function in callsites)
        for name in row[0]:
            values.require(name not in index, "Duplicate stack mapping")
            index[name] = (frames, diagnostic)
    entries, native_layers = [], {}
    for i, (name, node) in enumerate(dump["data"].items()):
        values.require(name in construct and name in index and isinstance(node, dict), "Missing raw node, construct or stack")
        parents, seen, p = [], {name}, construct[name]
        while p is not None:
            values.require(p in construct and p not in seen, "Cyclic or incomplete module ancestry")
            seen.add(p)
            parents.append(p)
            p = construct[p]
        frames, diagnostic = index[name]
        layer = next((int(m[1]) for p in [name, *parents] if (m := LAYER.match(p))), None)
        native = bool(values.API.fullmatch(name))
        if native:
            match = PARENT.fullmatch(parents[0]) if parents else None
            values.require(match and int(match[1]) == layer and parents[0] in dump["data"] and
                           layer not in native_layers and not diagnostic, "Duplicate or unbound original model IFA")
            values.validate_target(node)
            native_layers[layer] = {"step": 0, "layer_index": layer, "api": name, "parent": parents[0]}
        entries.append({"name": name, "ordinal": i, "layer": layer, "parents": parents, "frames": frames,
                        "diagnostic": diagnostic, "native": native, "node": node})
    values.require(list(native_layers) == list(range(28)), "Expected 28 ordered original IFAs, one per real model layer")
    values.require(list(native_layers.values()) == receipt["native_api_nodes"], "Receipt layer labels disagree with original construct")
    return {"entries": entries, "layers": list(native_layers.values()), "hashes": hashes, "collector": collector,
            "config": {k: v for k, v in config.items() if k != "dump_path"}}


def public(entry):
    return {k: entry[k] for k in ("name", "ordinal", "parents", "layer", "diagnostic", "native")}


def short_stats(stats):
    return {"different_finite_fields": len(stats["statistical_differences"]),
            "equal_finite_fields": stats["matching_finite_statistics"], "unobservable_fields": len(stats["unobservable"]),
            "metadata_differences": len(stats["metadata_differences"])}


def compare(baseline, fault):
    arms = {"B": load(baseline), "F": load(fault)}
    b, f = arms.values()
    def trace_sig(e):
        return (values.canonical(e["name"]), tuple(values.canonical(p) for p in e["parents"]),
                e["frames"], e["layer"], e["diagnostic"], e["native"])
    entries = b["entries"], f["entries"]
    same_trace = len(entries[0]) == len(entries[1]) and all(trace_sig(x) == trace_sig(y) for x, y in zip(*entries))
    same_config = b["config"] == f["config"] and b["collector"] == f["collector"]
    result = {"schema": "fault33.model.independent_msprobe.v1", "status": "INCONCLUSIVE_TRACE_OR_CONFIG",
              "ordered_full_stack_construct_match": same_trace, "config_and_collector_match": same_config,
              "sources": {k: v["hashes"] for k, v in arms.items()}, "collectors": {k: v["collector"] for k, v in arms.items()},
              "native_layer_inventory": {k: v["layers"] for k, v in arms.items()},
              "first_observed_model_api_difference": None, "first_observed_raw_api_difference": None,
              "first_observed_module_difference": None, "first_model_metadata_mismatch": None,
              "native_layers": [], "node_counts": {k: len(v["entries"]) for k, v in arms.items()},
              "compared_fields": {"equal_finite": 0, "different_finite": 0, "unobservable": 0, "metadata_differences": 0},
              "diagnostic_node_count": sum(e["diagnostic"] for e in b["entries"]),
              "diagnostic_source_rule": {"file": "/audit/model_adapter.py", "functions": sorted(DIAGNOSTICS)},
              "analysis_uses_hit_or_policy": False, "physical_fault_location_proven": False, "tensor_equality_proven": False,
              "limits": ["Independent first difference in original finite API summaries in dump order. Module boundaries are reported separately.",
                         "Model ranking excludes only exact diagnostic source frames. Unfiltered first raw API difference is separately retained.",
                         "A metadata mismatch before the first finite difference suppresses localization. Later input differences may reflect propagation.",
                         "Null/missing/nonfinite summaries remain unobservable. Equal summaries may correspond to different full tensors.",
                         "Each layer is one actual framework IFA API record over a whole batch. Dispatch, per-request differences and model accuracy require separate evidence."]}
    if not (same_trace and same_config):
        return result
    first_raw, first_model, first_module = (None,) * 3
    model_mismatch = raw_mismatch = False
    for x, y in zip(*entries):
        is_module = x["name"].startswith("Module.")
        model_api = not is_module and not x["diagnostic"]
        before = values.compare_values(values.inputs(x["node"]), values.inputs(y["node"]))
        after = values.compare_values(x["node"].get("output"), y["node"].get("output"))
        if x["native"]:
            result["native_layers"].append({"B": public(x), "F": public(y), "input": before, "output": after})
        for side, stat in (("input", before), ("output", after)):
            totals = result["compared_fields"]
            totals["equal_finite"] += stat["matching_finite_statistics"]
            totals["different_finite"] += len(stat["statistical_differences"])
            totals["unobservable"] += len(stat["unobservable"])
            totals["metadata_differences"] += len(stat["metadata_differences"])
            if stat["metadata_differences"] and not is_module:
                raw_mismatch = True
                if model_api and not model_mismatch:
                    model_mismatch = True
                    result["first_model_metadata_mismatch"] = {"B": public(x), "F": public(y), "side": side,
                                                               "fields": stat["metadata_differences"][:8]}
            if not stat["statistical_differences"]:
                continue
            if ((not is_module and first_raw is None and not raw_mismatch)
                    or (model_api and first_model is None and not model_mismatch) or (is_module and first_module is None)):
                detail = {"B": public(x), "F": public(y), "boundary": side, "fields": stat["statistical_differences"],
                          "input_summary": short_stats(before), "output_summary": short_stats(after),
                          "source_stack": x["frames"]}
                if not is_module and first_raw is None and not raw_mismatch:
                    first_raw = detail
                if model_api and first_model is None and not model_mismatch:
                    first_model = detail
                if is_module and first_module is None:
                    first_module = detail
    result.update(first_observed_model_api_difference=first_model, first_observed_raw_api_difference=first_raw,
                  first_observed_module_difference=first_module)
    if first_model:
        result["status"] = "ALIGNED_OBSERVED_MODEL_API_DIFFERENCE"
    elif model_mismatch:
        result["status"] = "INCONCLUSIVE_MODEL_API_INPUT_OR_OUTPUT_METADATA"
    else:
        result["status"] = "ALIGNED_NO_OBSERVED_FINITE_MODEL_API_DIFFERENCE"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fault", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = compare(args.baseline, args.fault)
        rc = 2 if report["status"].startswith("INCONCLUSIVE") else 0
    except (ValueError, KeyError, TypeError, OSError, IndexError) as error:
        report = {"schema": "fault33.model.independent_msprobe.v1", "status": "INVALID_CAPTURE_EVIDENCE", "error": str(error),
                  "first_observed_model_api_difference": None}
        rc = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2, allow_nan=False)
        fp.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.output), "error": report.get("error")}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

