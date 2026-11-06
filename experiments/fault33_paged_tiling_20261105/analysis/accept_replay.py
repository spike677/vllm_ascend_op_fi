"""Offline fault-33 replay acceptance; no framework or device imports.

Run with the bridge attempt first. Supply --attempt-fault after the fault and
same-process recovery attempt finishes. All input paths are read-only.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct

import msprobe_compare
import native_evidence

MODES = {"stock_bridge": ["off"] * 2, "baseline": ["off"] * 2, "fault": ["inner1024"] * 2,
         "restore": ["off", "inner1024", "inner1024", "off"]}
HIGH_MANIFEST = "743354feff5a5f91a17e68695bad167fa2584a48d257ec227d0973d071f185af"
CORE = "increFlashAttentionSingleCoreParams."
INNER, LOOPS = CORE + "singleProcessSInnerSize", CORE + "sInnerLoopTimes"
OUTPUT_SHAPES = {"attention": [8, 16, 1, 128], "projection": [8, 256]}
INPUT_SHAPES = {"query": [8, 16, 1, 128], "key": [1024, 8, 128, 128],
                "value": [1024, 8, 128, 128], "block_table": [8, 128]}
GEOMETRY = {"batch": 8, "q_heads": 16, "kv_heads": 8, "kv_length": 16384, "block_size": 128,
            "head_dim": 128, "dtype": "float16", "inner_precise": 0, "deterministic_algorithms": True}
FORMAT = {"uint32_t": "I", "int32_t": "i", "uint64_t": "Q", "int64_t": "q", "float": "f"}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    need(path.is_file() and not path.is_symlink(), f"Missing or symlinked evidence: {path}")
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return msprobe_compare.read(path)[0]


def within(root, relative):
    rel = PurePosixPath(relative)
    need(relative == rel.as_posix() and not rel.is_absolute() and ".." not in rel.parts,
         f"Unsafe evidence reference: {relative}")
    path = root / rel
    need(root.resolve() in path.resolve().parents and not path.is_symlink(), f"Path escapes source: {relative}")
    return path


def abi_fields(schema, raw):
    need(schema["schema"] == "fault33.pa.public_serializer_layout.v1"
         and schema["type"] == "IncreFlashAttentionTilingDataV2"
         and schema["layout_source"] == "public_host_SaveToBuffer_single_field_roundtrip"
         and schema["kernel_argument_raw_observed"] is False and schema["projection_only"] is True
         and 0 < len(raw) == schema["size"] <= 1024**2, "Public ABI source/size mismatch")
    result, used = {}, bytearray(len(raw))
    for f in schema["fields"]:
        name, kind, off, width, count = (f[k] for k in ("name", "type", "offset", "width", "count"))
        need(name not in result and kind in FORMAT and width == struct.calcsize("<" + FORMAT[kind])
             and type(off) is int and off >= 0 and type(count) is int and count >= 1,
             "Invalid field in public ABI")
        end = off + count * width
        need(end <= len(raw) and not any(used[off:end]), f"Overlapping or out-of-bounds ABI field: {name}")
        used[off:end] = bytes([1]) * (end - off)
        fields = struct.unpack_from(f"<{count}{FORMAT[kind]}", raw, off)
        result[name] = fields[0] if count == 1 else list(fields)
    need(result, "Empty ABI projection")
    return result


def verified_abi(vendor, schema_file):
    """Verify the private vendor's exact generation-bound public serializer evidence."""
    a = vendor["public_abi"]
    gen = vendor["generation"]
    need(re.fullmatch(r"a[1-9][0-9]{0,2}", gen) and a["generation"] == gen
         and vendor["vendor_root"] == f"vendor_stage_{gen}/vendors/fault33_pa_transformer"
         and a["layout_kind"] == "public_host_serializer_projection" and a["kernel_argument_raw_observed"] is False,
         "Vendor/ABI generation or kind mismatch")
    names = {"schema": "ifa_abi_schema.json", "compile": "ifa_abi_compile.json", "selftest": "ifa_abi_selftest.json"}
    need(schema_file.name == names["schema"], "Unexpected ABI schema filename")
    for role, name in names.items():
        need(a[role + "_file"] == f"build/evidence_{gen}/{name}" and
             sha(schema_file.with_name(name)) == a[role + "_sha256"], f"Unbound same-generation ABI {role} evidence")
    schema = read(schema_file)
    compile_receipt = read(schema_file.with_name(names["compile"]))
    test = read(schema_file.with_name(names["selftest"]))
    need(schema_file.stat().st_size == a["schema_bytes"] and schema["size"] == a["raw_size"] == test["size"] == 3000
         and len(schema["fields"]) == a["projected_fields"] == test["projected_fields"] == 72,
         "ABI projection size or field count differs from approved vendor")
    need(test["status"] == "PASS_PUBLIC_SERIALIZER_CPU_ROUNDTRIP" and test["stock_abi_validated"] is False
         and test["kernel_argument_raw_observed"] is False and compile_receipt["schema"] == "fault33.pa.abi_decoder_compile.v1"
         and compile_receipt["npu_executed"] is False and compile_receipt["abi_kind"] == "public_host_serializer_projection_not_device_argument_read"
         and all(re.fullmatch(r"[0-9a-f]{64}", compile_receipt[k]) for k in ("source_sha256", "decoder_sha256")),
         "Invalid ABI compiler/self-test receipts")
    return schema


def tensor_meta(meta, shape, dtype):
    width = {"torch.float16": 2, "torch.int32": 4}[dtype]
    need(meta["dtype"] == dtype and meta["shape"] == shape and meta["byteorder"] == "little"
         and meta["logical_bytes"] == width * math.prod(shape) and re.fullmatch("[0-9a-f]{64}", meta["sha256"]),
         f"Invalid tensor record for {shape}/{dtype}")


def output_bytes(case_dir, call):
    result = {}
    for label, shape in OUTPUT_SHAPES.items():
        meta = call["synthetic_projection" if label == "projection" else label]
        tensor_meta(meta, shape, "torch.float16")
        path = case_dir / "private_output" / (call["call_id"] + "_" + label + ".bin")
        need(path.stat().st_size == meta["logical_bytes"] and sha(path) == meta["sha256"],
             f"Private output does not match capture record: {path}")
        raw = path.read_bytes()
        need(all(math.isfinite(x[0]) for x in struct.iter_unpack("<e", raw)), "Nonfinite FP16 result")
        result[label] = raw
    return result


def difference(a, b, batch):
    need(len(a) == len(b) and len(a) and len(a) % (2 * batch) == 0, "Mismatched FP16 geometry")
    row_size = len(a) // (2 * batch)
    count, signed_zeros, max_abs, sum_abs, first = 0, 0, 0., 0., None
    per_request = [0] * batch
    for i, ((x,), (y,)) in enumerate(zip(struct.iter_unpack("<e", a), struct.iter_unpack("<e", b))):
        need(math.isfinite(x) and math.isfinite(y), "Nonfinite comparison")
        if x != y:
            if first is None:
                first = {"batch_row": i // row_size, "flat_index_in_row": i % row_size, "B": x, "F": y}
            count += 1
            per_request[i // row_size] += 1
            delta = abs(x - y)
            max_abs, sum_abs = max(max_abs, delta), sum_abs + delta
        elif x == 0. and a[2*i:2*i+2] != b[2*i:2*i+2]:
            signed_zeros += 1
    return {"bitwise_equal": a == b, "numeric_different_elements": count, "signed_zero_only_differences": signed_zeros,
            "elements": len(a) // 2, "max_abs": max_abs, "mean_abs": sum_abs / (len(a) // 2),
            "per_request_numeric_differences": per_request, "first_numeric_difference": first}


def verify_attempt(root, names):
    completion = read(root / "completion.json")
    need(completion["status"] == "CAPTURE_COMPLETED" and completion["card"] == 5
         and completion["probe"] == "replay.py" and completion["capture"] == "native-logs"
         and completion["device_lock_acquired"] is True and completion["device_lock_released"] is True
         and completion["owned_container_removed"] is True and not completion["cleanup_errors"]
         and completion["external_container_mutations"] is False and
         re.fullmatch("[0-9a-f]{64}", completion["owned_container_id"]), "Attempt was not cleanly completed")
    rows = completion["cases"]
    selected = read(root / "selected_cases.json")
    need([c["case_id"] for c in rows] == list(names) == [c["case_id"] for c in selected],
         "Attempt must contain exactly the registered cases in order")
    manifest = read(root / "source_manifest.json")
    need({"replay.py", "controller.py", "vendor_exec.py", "env.sh", "collector_stage.py", "stock_probe.py"}
         <= manifest.keys(), "Frozen harness incomplete")
    for path, digest in manifest.items():
        need(sha(within(root / "source", path)) == digest, "Frozen source changed: " + path)
    for row, s in zip(rows, selected):
        case_id = row["case_id"]
        need(row["rc"] == 0 and row["status"] == "CAPTURE_COMPLETED_NOT_MECHANISM_ACCEPTANCE"
             and sha(root / case_id / "result.json") == row["result_sha256"], f"Unbound result: {case_id}")
        need(s["file"] in manifest and read(within(root / "source", s["file"])) == s["case"]
             == read(root / case_id / "case.json"), f"Case disagrees with frozen input: {case_id}")
    return completion, manifest, selected


def visible_plan(row):
    selected = {k: v for k, v in row["selected"].items() if k not in ("op", "pid", "tid")}
    flags = row["native_flags"]
    need(flags["deterministic"] == 1, "Actual NNOP deterministic execution flag missing")
    return {"selected": selected, "raw_size": row["native_raw_size"], "sdk_bmm": row["sdk_bmm"],
            "native_deterministic": flags["deterministic"], "native_high_precision_flag": flags["verbose_high_precision_flag"]}


def private_plan(window, mode, schema):
    h = window["public_host"]
    fields = abi_fields(schema, bytes.fromhex(h["raw_hex"]))
    need(fields == h["typed"] and len(fields) == 72 and {INNER, LOOPS, "baseParams.blockSize", "coreSidxEnd"} <= fields.keys()
         and isinstance(fields["coreSidxEnd"], list) and len(fields["coreSidxEnd"]) == 50,
         "Public raw ABI fields disagree with typed observation")
    on = mode != "off"
    need(h["phase"] == "decode" and h["mode"] == mode and h["eligible"] is True
         and h["policy_applied"] is on and h["caller_declared_deterministic"] is True
         and h["kernel_argument_raw_observed"] is False and h["selector_checks"] == len(h["guard_reports"]) == 1,
         "Host mode, deterministic declaration, or per-call hit count invalid")
    guard = h["guard_reports"][0]
    need(all(guard[k] is True for k in ("fixed_inputs", "actual_lengths_all_equal", "native_scope", "legal_request", "eligible"))
         and guard["applied"] is on and (guard["batch"], guard["actual_kv_length"], guard["block_size"]) == (8, 16384, 128)
         and guard["requested_sinner"] == (1024 if on else 0) and fields["baseParams.blockSize"] == 128,
         "Public guard does not match observed call")
    expected = 1024 if on else guard["native_sinner"]
    need(fields[INNER] == expected and guard["native_sinner"] > 0 and fields[LOOPS] > 0,
         "Selector did not produce the declared serialized plan")
    return {"selected_inner": fields[INNER], "outer_update_loops": fields[LOOPS],
            "pre_intervention_native_inner": guard["native_sinner"], "selector_applied": on,
            "raw_sha256": window["public_host_raw_sha256"], "verified_projected_fields": len(fields),
            "typed": fields, "workspace_sizes": h["workspace_sizes"]}


def verify_maps(case_dir, case, result, vendor):
    maps = read(case_dir / "runtime_native_maps.json")
    need(sha(case_dir / "runtime_native_maps.json") == result["runtime_native_maps_sha256"]
         and maps["pid"] == result["calls"][0]["pid"] and maps["libraries"], "Runtime maps binding failed")
    prefix = "/audit/" + vendor["vendor_root"].strip("/") + "/"
    libs = {row["path"]: row for row in vendor["artifacts"]}
    approved = {libs[p]["sha256"] for p in vendor["ifa_registration"]["tiling_libraries"]}
    matched = []
    private = case["operator_variant"] == "private_vendor"
    for item in maps["libraries"]:
        if item["path"].startswith(prefix):
            need(private, "Private operator library mapped in stock process")
            rel = item["path"][len(prefix):]
            need(rel in libs and item["sha256"] == libs[rel]["sha256"] and item["bytes"] == libs[rel]["bytes"],
                 "Mapped private library disagrees with approved manifest")
            if item["sha256"] in approved:
                matched.append(item)
        elif "/audit/vendor_stage_" in item["path"]:
            raise ValueError("Different private vendor generation mapped")
    need(bool(matched) == private, "Missing private tiling library or stock process has private mapping")
    return matched


def verify_statistics(case_dir, result):
    receipt = read(case_dir / "statistics_receipt.json")
    source = read(case_dir / "collector_provenance.json")
    need(receipt == result["statistics_receipt"] and receipt["schema"] == "fault33.pa.statistics_receipt.v1"
         and receipt["steps"] == [0, 1] and receipt["rank"] == 0 and receipt["statistics_collector"] == source
         and source["manifest_sha256"] == HIGH_MANIFEST and source["artifact_count"] == 513
         and receipt["config_sha256"] == sha(case_dir / "config.json")
         and receipt["collector_provenance_sha256"] == sha(case_dir / "collector_provenance.json"),
         "Statistics config/collector receipt disagrees with result")
    expected = []
    for step in (0, 1):
        step_dump = case_dir / "observation" / f"step{step}" / "rank0/dump.json"
        raw = msprobe_compare.load_step(step_dump)
        target = raw["target"]
        need(receipt["native_api_nodes"][step] == {"step": step, "api": target["name"], "parent": target["parents"][0]},
             "Native API or child Module in receipt differs from raw capture")
        call, node = result["calls"][step], target["node"]
        slots = dict(zip(("query", "key", "value"), node["input_args"][:3]))
        slots["block_table"] = node["input_kwargs"]["block_table"]
        need(slots.keys() == INPUT_SHAPES.keys(), "Incomplete native positional tensor geometry")
        for label, observed in slots.items():
            meta = call["inputs_before"][label]
            need(all(observed[k] == meta[k] for k in ("shape", "dtype")), "Statistics from a different input geometry")
        need(all(node["output"][0][k] == call["attention"][k] for k in ("shape", "dtype")),
             "Statistics output geometry differs from native output")
        for kind in ("dump", "stack", "construct"):
            file = case_dir / "observation" / f"step{step}" / "rank0" / (kind + ".json")
            expected.append({"path": file.relative_to(case_dir).as_posix(), "bytes": file.stat().st_size, "sha256": sha(file)})
    need(receipt["native_files"] == expected, "Original statistics trio disagrees with case receipt")
    for file in (case_dir / "observation").rglob("*"):
        need(not file.is_symlink() and file.suffix.lower() not in (".pt", ".pth", ".bin", ".npy", ".npz", ".safetensors"),
             "Tensor payload or symlink in statistics evidence")
    return sha(case_dir / "statistics_receipt.json")


def load_case(root, selected, schema, vendor, vendor_sha, manifest):
    name, case = selected["case_id"], selected["case"]
    case_dir, modes = root / name, MODES[name]
    result = read(case_dir / "result.json")
    private = name != "stock_bridge"
    capture = name in ("baseline", "fault")
    need(case["schema"] == "fault33.pa.replay.case.v1" and all(case[k] == v for k, v in GEOMETRY.items())
         and case["mode"] == modes[0] and case.get("mode_sequence", [case["mode"]] * case["repeats"]) == modes
         and case["repeats"] == len(modes) and case["capture_msprobe"] is capture
         and (case["operator_variant"] == ("private_vendor" if private else "stock")), "Wrong registered case: " + name)
    need(result["schema"] == "fault33.pa.replay.result.v1" and result["case_id"] == name
         and result["status"] == "CAPTURE_COMPLETED" and result["mode"] == modes[0] and result["model_forward"] is False
         and result["deterministic_algorithms"] is True and result["original_api_calls"] == len(result["calls"]) == len(modes),
         "Missing complete deterministic replay: " + name)
    loaded = verify_maps(case_dir, case, result, vendor)
    log = root / (name + ".log")
    windows = native_evidence.parse_log(log)
    tokens = [c["scope_token"] for c in result["calls"]]
    need(len(windows) == len(modes) and len(set(tokens)) == len(tokens)
         and len({c["pid"] for c in result["calls"]}) == 1, "Missing/reused windows or process changed within case")
    hashes = {"case": sha(case_dir / "case.json"), "result": sha(case_dir / "result.json"),
              "log": sha(log), "maps": result["runtime_native_maps_sha256"]}
    if private:
        need(case["approved_vendor_manifest_sha256"] == result["vendor"]["manifest_sha256"] == vendor_sha,
             "Vendor manifest fingerprint mismatch")
        launcher_path = root / (name + ".vendor_launch.json")
        launcher = read(launcher_path)
        need(launcher["verified_vendor"] == result["vendor"] and launcher["case_id"] == name
             and launcher["case_sha256"] == manifest[selected["file"]] and launcher["probe_sha256"] == manifest["replay.py"]
             and launcher["launcher_sha256"] == manifest["vendor_exec.py"] and launcher["pid"] == result["calls"][0]["pid"]
             and launcher["child_environment"]["ACLNN_CACHE_LIMIT"] == "0", "Vendor launch receipt mismatch")
        hashes["launcher"], hashes["host_events"] = sha(launcher_path), sha(case_dir / "host_events.jsonl")
        native_evidence.bind_host(windows, case_dir / "host_events.jsonl")
    else:
        need(result["vendor"] is None and not (case_dir / "host_events.jsonl").exists(), "Unexpected stock host callback")
    if capture:
        hashes["statistics_receipt"] = verify_statistics(case_dir, result)
    else:
        need(result["statistics_receipt"] is None and not (case_dir / "observation").exists(),
             "Unregistered extra statistics group")
    logical = read(case_dir / "logical_inputs.json")
    need(logical == result["logical_inputs"] and set(logical) == {"query", "key", "value"}, "Logical input binding failed")
    for label, meta in logical.items():
        tensor_meta(meta, [8, 16, 1, 128] if label == "query" else [8, 8, 16384, 128], "torch.float16")
    weight = read(case_dir / "projection.json")["weight"]
    tensor_meta(weight, [2048, 256], "torch.float16")
    verified = []
    for i, (call, window, mode) in enumerate(zip(result["calls"], windows, modes)):
        enter = window["enter"]
        need(call["call_id"] == window["call_id"] == f"repeat{i}" and call["mode"] == mode and call["original_api_calls"] == 1
             and all(call[k] == enter[k] for k in ("scope_token", "pid", "tid"))
             and enter["deterministic_algorithms"] is True and enter["block_size"] == 128, "Unbound real native call")
        physical = call["inputs_before"]
        need(physical == call["inputs_after"] and physical.keys() == INPUT_SHAPES.keys(), "Actual native inputs mutated/missing")
        for label, shape in INPUT_SHAPES.items():
            tensor_meta(physical[label], shape, "torch.int32" if label == "block_table" else "torch.float16")
        plan = private_plan(window, mode, schema) if private else None
        verified.append({"call_id": call["call_id"], "inputs": physical, "output": output_bytes(case_dir, call),
                         "visible": visible_plan(window), "plan": plan})
    return {"name": name, "root": case_dir, "calls": verified, "logical": logical, "weight": weight,
            "seeds": [case["seed"], case["projection_seed"]], "software": result["software"], "hashes": hashes,
            "verified_tiling_maps": loaded}


def check_inputs(cases):
    base = cases["baseline"]
    for name, case in cases.items():
        for key in ("seeds", "software", "logical", "weight"):
            need(base[key] == case[key], f"{name} differs in {key}")
        for call in case["calls"]:
            need(call["inputs"] == base["calls"][0]["inputs"], f"{name} changes physical Q/K/V/table data or geometry")


def check_equivalent(standard, actual, check_public):
    need(standard["visible"] == actual["visible"], "Selected key/block/schedule or actual SDK BMM differs")
    need(standard["output"] == actual["output"], "Attention or projection output differs in expected-equivalent calls")
    if check_public:
        for key in ("typed", "workspace_sizes", "pre_intervention_native_inner", "raw_sha256"):
            need(standard["plan"][key] == actual["plan"][key], "Repeated/restored public host plan differs: " + key)


def independent_statistics(cases):
    def dump(name, step):
        return cases[name]["root"] / "observation" / f"step{step}" / "rank0/dump.json"

    pairs = {}
    for left, li, right, ri in (("baseline", 0, "fault", 0), ("baseline", 1, "fault", 1),
                               ("baseline", 0, "baseline", 1), ("fault", 0, "fault", 1)):
        report = msprobe_compare.compare(dump(left, li), dump(right, ri))
        pairs[f"{left}{li}_vs_{right}{ri}"] = {k: report[k] for k in (
            "status", "alignment", "first_observed_api_difference", "native_api", "evidence",
            "unobservable_fields_in_aligned_rows")}
    cross = [v for k, v in pairs.items() if k.startswith("baseline") and "_vs_fault" in k]
    first_ifa_output = all(v["status"] == "ALIGNED_OBSERVED_API_DIFFERENCE" and
                           v["first_observed_api_difference"]["boundary"] == "output" and
                           msprobe_compare.API.fullmatch(v["first_observed_api_difference"]["B_node"]) and
                           not v["native_api"]["input"]["statistical_differences"] for v in cross)
    repeats_stable = all(pairs[k]["status"] == "ALIGNED_NO_OBSERVED_FINITE_API_DIFFERENCE"
                         for k in ("baseline0_vs_baseline1", "fault0_vs_fault1"))
    return {"pairs": pairs, "both_cross_pairs_first_diff_at_native_output": first_ifa_output,
            "both_within_mode_pairs_no_observed_statistical_change": repeats_stable,
            "observed_repeatable_native_first_difference": first_ifa_output and repeats_stable,
            "limits": "Raw finite statistics only. Matching summaries may collide; null/missing fields are unobservable."}


def accept(bridge, fault, vendor_file, schema_file):
    vendor = read(vendor_file)
    schema = verified_abi(vendor, schema_file)
    vendor_sha = sha(vendor_file)
    need(vendor["schema"] == "fault33.pa.private_vendor.v1" and vendor["ifa_registration"]["op_type"] == "IncreFlashAttention"
         and vendor["kernel_argument_raw_observed"] is False and vendor["same_binary_all_profiles"] is True,
         "Wrong approved private vendor")
    cases, manifests, attempts = {}, [], {}
    groups = [(bridge, ("stock_bridge", "baseline"))]
    if fault:
        groups.append((fault, ("fault", "restore")))
    for root, ids in groups:
        completion, manifest, rows = verify_attempt(root, ids)
        attempts[root.name] = {"completion_sha256": sha(root / "completion.json"), "source_manifest_sha256": sha(root / "source_manifest.json"),
                               "start_epoch": completion["start_epoch"], "end_epoch": completion["end_epoch"]}
        manifests.append(manifest)
        for row in rows:
            cases[row["case_id"]] = load_case(root, row, schema, vendor, vendor_sha, manifest)
    need(all(m == manifests[0] for m in manifests), "Bridge and fault attempts use different frozen sources/cases")
    if fault:
        need(attempts[bridge.name]["end_epoch"] <= attempts[fault.name]["start_epoch"], "Fault started before bridge completed")
    check_inputs(cases)
    b = cases["baseline"]["calls"][0]
    for call in cases["stock_bridge"]["calls"]:
        check_equivalent(b, call, False)
    check_equivalent(b, cases["baseline"]["calls"][1], True)
    evidence = {name: {"files": c["hashes"], "verified_private_tiling_maps": c["verified_tiling_maps"],
                       "calls": [{"call_id": c0["call_id"], "actual_launch_projection": c0["visible"],
                                  "public_plan": c0["plan"], "output_sha256": {k: hashlib.sha256(v).hexdigest()
                                                                             for k, v in c0["output"].items()}}
                                 for c0 in c["calls"]]} for name, c in cases.items()}
    result = {"schema": "fault33.offline_replay_acceptance.v1", "status": "BRIDGE_VERIFIED_FAULT_NOT_EVALUATED",
              "stock_off_bridge_verified": True, "attempts": attempts, "cases": evidence, "actual_calls": sum(len(c["calls"]) for c in cases.values()),
              "vendor_manifest_sha256": vendor_sha, "public_serializer_schema_sha256": sha(schema_file),
              "verified_vendor_public_abi": vendor["public_abi"],
              "same_logical_physical_inputs_and_projection_weight": True, "seeds": cases["baseline"]["seeds"],
              "mechanism_accepted": False, "independent_statistics_localized": None, "release_approved": False,
              "kernel_argument_raw_observed": False, "stock_full_tiling_raw_observed": False, "real_model_forward": False}
    if not fault:
        return result
    f = cases["fault"]["calls"][0]
    check_equivalent(f, cases["fault"]["calls"][1], True)
    for index, off in enumerate((True, False, False, True)):
        check_equivalent(b if off else f, cases["restore"]["calls"][index], True)
    bp, fp = b["plan"], f["plan"]
    need(bp["pre_intervention_native_inner"] == fp["pre_intervention_native_inner"], "Native selector already differs before control")
    changed_plan = bp["selected_inner"] != fp["selected_inner"] and bp["outer_update_loops"] != fp["outer_update_loops"]
    contrasts = {key: difference(b["output"][key], f["output"][key], 8) for key in OUTPUT_SHAPES}
    numeric = contrasts["attention"]["numeric_different_elements"] > 0
    stats = independent_statistics(cases)
    mechanism = changed_plan and numeric
    localized = stats["observed_repeatable_native_first_difference"]
    result.update(status=("PASS_CONTROLLED_REPLAY_AND_STATISTICS" if mechanism and localized else "VERIFIED_REPLAY_WITH_LIMITS"),
                  mechanism_accepted=mechanism, independent_statistics_localized=localized,
                  release_approved=bool(mechanism and localized), native_partition_changed=changed_plan,
                  same_mode_two_repeats_bitwise_stable=True, same_process_off_on_on_off_restored=True,
                  full_outputs=contrasts, independent_statistics=stats,
                  limits=["Controlled fixed-layout native replay. Stock and OFF match visible launch attributes and outputs.",
                          "Public host ABI projection is decoded independently and bound to NNOP launch metadata; device argument bytes are not read.",
                          "The synthetic linear projection measures propagation to this consumer. Model and historical incident impact remain untested."])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-bridge", required=True, type=Path)
    parser.add_argument("--attempt-fault", type=Path)
    parser.add_argument("--vendor-manifest", required=True, type=Path)
    parser.add_argument("--abi-schema", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = accept(args.attempt_bridge, args.attempt_fault, args.vendor_manifest, args.abi_schema)
        rc = 0 if result["stock_off_bridge_verified"] and (not args.attempt_fault or result["release_approved"]) else 2
    except (ValueError, KeyError, TypeError, IndexError, OSError, struct.error) as error:
        result = {"schema": "fault33.offline_replay_acceptance.v1", "status": "INVALID_OR_INCOMPLETE_EVIDENCE",
                  "error": str(error), "mechanism_accepted": False, "release_approved": False,
                  "kernel_argument_raw_observed": False, "real_model_forward": False}
        rc = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False, allow_nan=False)
        fp.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output), "error": result.get("error")}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

