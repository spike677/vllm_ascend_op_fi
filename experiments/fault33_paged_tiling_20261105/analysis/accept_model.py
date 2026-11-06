"""Offline four-run complete-model acceptance for the adapted paged-IFA decode.

No model/framework imports. Native plan, observed model propagation and first
differences in the original msprobe statistics are evaluated independently.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct

import accept_replay as replay
import compare_model
import native_evidence

NAMES = ("model_baseline", "model_fault_r0", "model_fault_r1", "model_restore")
LAYERS = list(range(28))
REQUESTS = [f"request_{i:02d}" for i in range(8)]
COMPARISONS = ((0, 1), (0, 2), (0, 3), (1, 2))
VOCAB = 151936
LOGITS = [8, 1, VOCAB]
WIDTH = {"torch.int32": 4, "torch.float16": 2, "torch.float32": 4}
need, sha, read = replay.need, replay.sha, replay.read


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_hash(v):
    return isinstance(v, str) and re.fullmatch("[0-9a-f]{64}", v) is not None


def report_meta(r, shape, dtypes):
    need(r["dtype"] in dtypes and r["shape"] == shape and r["byteorder"] == "little"
         and r["bytes"] == math.prod(shape) * WIDTH[r["dtype"]] and checked_hash(r["sha256"])
         and len(r["request_sha256"]) == 8 and all(checked_hash(v) for v in r["request_sha256"]),
         "Malformed full tensor or per-request hash record")


def validate_payload(meta, raw):
    need(len(raw) == meta["bytes"] and digest(raw) == meta["sha256"] and len(raw) % 8 == 0,
         "Full private output/table hash mismatch")
    row = len(raw) // 8
    need([digest(raw[i*row:(i+1)*row]) for i in range(8)] == meta["request_sha256"],
         "Actual per-request row hashes disagree with record")


def prefill_inputs(case_dir, result):
    docs = {}
    for file, field in (("release.json", "release_sha256"), ("request_mapping.json", "request_mapping_sha256"),
                        ("prefill_cache.json", "prefill_cache_sha256"), ("block_table.json", "block_table_sha256")):
        need(sha(case_dir / file) == result[field], "Unbound model evidence: " + file)
        docs[file] = read(case_dir / file)
    release, mapping, cache, blocks = docs.values()
    need(release["files"]["model.safetensors"]["sha256"] == "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
         and release["files"]["config.json"]["sha256"] == "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"
         and release["versions"] == {"torch": "2.10.0", "torch-npu": "2.10.0.post4", "transformers": "5.5.4"},
         "Different model checkpoint/configuration or runtime")
    req = mapping["rows"]
    need(mapping["schema"] == "fault33.model_known_prefix.v1" and [r["request_id"] for r in req] == REQUESTS
         and [r["batch_row"] for r in req] == list(range(8)) and len(set(r["known_prefix_sha256"] for r in req)) == 8,
         "Incomplete or duplicate request identity")
    for row in req:
        need(row["prefill_length"] == row["decode_position"] == 16383 and 0 <= row["teacher_token_id"] < VOCAB
             and all(checked_hash(row[k]) for k in ("prompt_token_sha256", "known_prefix_sha256")),
             "Invalid fixed prompt/teacher token mapping")
    expected = [list(range(i * 128, (i + 1) * 128)) for i in range(8)]
    need(blocks["values"] == expected, "Unexpected paged decode table")
    report_meta(blocks, [8, 128], {"torch.int32"})
    validate_payload(blocks, struct.pack("<1024i", *range(1024)))
    need(cache["length"] == 16383 and len(cache["layers"]) == len(LAYERS)
         and [c["layer_index"] for c in cache["original_sdpa_calls"]] == LAYERS, "Incomplete original HF prefill")
    for c in cache["original_sdpa_calls"]:
        need(c["api"] == "torch.nn.functional.scaled_dot_product_attention" and c["original_api_calls"] == 1
             and c["scope_active"] is False and c["is_causal"] is True and c["enable_gqa"] is True
             and c["query_shape"] == c["output_shape"] == [8, 16, 16383, 128]
             and c["key_shape"] == c["value_shape"] == [8, 8, 16383, 128], "Unexpected original prefill attention")
    for layer in cache["layers"]:
        for key in ("key", "value"):
            report_meta(layer[key], [8, 8, 16383, 128], {"torch.float16"})
    report_meta(cache["prefill_logits"], LOGITS, {"torch.float16", "torch.float32"})
    return docs


def layer_inputs_and_outputs(c):
    report_meta(c["query"], [8, 16, 1, 128], {"torch.float16"})
    need(c["native_inputs_unchanged"] is True and c["actual_seq_lengths"] == [16384]*8, "Modified native inputs")
    for name in ("key", "value"):
        kv = c["kv"][name]
        need(kv["inverse_verified"] is True and kv["prefill_prefix_verified"] is True
             and kv["logical_shape"] == [8, 8, 16384, 128] and kv["physical_shape"] == [1024, 8, 128, 128]
             and kv["dtype"] == "torch.float16" and all(checked_hash(kv[k]) for k in ("logical_sha256", "physical_sha256"))
             and all(len(kv[k]) == 8 and all(checked_hash(h) for h in kv[k]) for k in ("request_sha256", "physical_request_sha256")),
             "Missing certified page inverse, real prefix or row hashes")
    for name, dims in (("attention", [8, 16, 1, 128]), ("o_proj_input", [8, 1, 2048]), ("o_proj", [8, 1, 1024])):
        report_meta(c[name], dims, {"torch.float16"})
    for key in ("sha256", "request_sha256"):
        need(c["attention"][key] == c["o_proj_input"][key], "Real projection did not consume the produced attention output")
    return {key: c[key] for key in ("query", "kv", "attention", "o_proj_input", "o_proj")}


def coverage_log(path, calls, case):
    """Cross-check complete process log, module order and any actual unscoped IFA."""
    pid = calls[0]["pid"]
    phase, layer, pending = None, None, None
    launches, phases, visits, exits = [], [], [], []
    expected_events = ("model_prefill_enter", "model_prefill_exit", "model_decode_enter", "model_decode_exit")
    with path.open(encoding="utf-8", errors="strict") as fp:
        for line in fp:
            if "NnopbaseExecutorLaunchKernel" in line and "IncreFlashAttention" in line:
                match = native_evidence.LAUNCH.search(line)
                owner = re.search(r"OP\((\d+),", line)
                if owner and int(owner[1]) == pid and re.search(r"Launch (?:dynamic|static) kernel IncreFlashAttention\b", line):
                    need(match is not None, "Unparsed real IFA launch; coverage unknown")
                if match and int(match[1]) == pid:
                    need(phase == "decode" and layer is not None and pending == "enter",
                         "Actual native IFA launch outside a registered model window")
                    launches.append(layer)
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            tag = row.get("fault33") if isinstance(row, dict) else None
            if not tag or row.get("pid") != pid:
                continue
            if tag in expected_events:
                need(len(phases) < 4 and tag == expected_events[len(phases)] and row["case_id"] == case, "Model phase order")
                phases.append(tag)
                phase = {"model_prefill_enter": "prefill", "model_decode_enter": "decode"}.get(tag)
            elif tag == "module_enter":
                need(phase == "decode" and layer is None and pending is None and row["request_ids"] == REQUESTS
                     and row["case_id"] == case and row["layer_index"] == len(visits) < 28, "Unbound module entry")
                layer = row["layer_index"]
                visits.append(layer)
            elif tag in ("enter", "exit"):
                need(layer is not None and row == calls[layer][tag] and row["layer_index"] == layer
                     and (pending is None if tag == "enter" else pending == "enter"), "Original API window not bound to active model layer")
                pending = tag
            elif tag == "module_exit":
                need(pending == "exit" and layer is not None and row["case_id"] == case and row["layer_index"] == layer,
                     "Missing native call in original module")
                exits.append(layer)
                layer, pending = None, None
    need(tuple(phases) == expected_events and layer is None and pending is None and visits == exits == launches == LAYERS,
         "Complete model log does not contain exactly one original IFA per layer")


def validate_attempt(root, expected_names=NAMES):
    complete = read(root / "completion.json")
    need(complete["status"] == "CAPTURE_COMPLETED" and complete["card"] == 5 and complete["probe"] == "model_probe.py"
         and complete["capture"] == "native-logs" and complete["device_lock_acquired"] is True and complete["device_lock_released"] is True
         and complete["owned_container_removed"] is True and complete["cleanup_errors"] == [] and complete["external_container_mutations"] is False,
         "Full model attempt did not finish cleanly")
    need([c["case_id"] for c in complete["cases"]] == list(expected_names), "Missing, duplicate or unexpected model case order")
    need(checked_hash(complete["owned_container_id"]) and read(root/"container_identity.json")["id"] == complete["owned_container_id"],
         "Cleaned container identity differs from the original receipt")
    manifest = read(root / "source_manifest.json")
    frozen = root / "source"
    files = list(frozen.rglob("*"))
    need(not frozen.is_symlink() and not any(p.is_symlink() for p in files)
         and {p.relative_to(frozen).as_posix() for p in files if p.is_file()} == set(manifest),
         "Frozen source inventory differs from manifest")
    for n, h in manifest.items():
        need(sha(replay.within(frozen, n)) == h, "Frozen model harness differs: " + n)
    need({"model_probe.py", "model_adapter.py", "vendor_exec.py", "controller.py", "replay.py"} <= manifest.keys(),
         "Frozen model harness missing")
    selected = read(root / "selected_cases.json")
    need([s["case_id"] for s in selected] == list(expected_names), "Selected cases not in registered order")
    times = [complete["start_epoch"], *(t for c in complete["cases"] for t in (c["start"], c["end"])), complete["end_epoch"]]
    need(all(type(t) in (float, int) and math.isfinite(t) for t in times) and all(x < y for x, y in zip(times, times[1:])),
         "Case intervals are out of order, overlap or fall outside the controller window")
    for c, s in zip(complete["cases"], selected):
        case = s["case_id"]
        need(c["rc"] == 0 and c["status"] == "CAPTURE_COMPLETED_NOT_MECHANISM_ACCEPTANCE"
             and sha(root / case / "result.json") == c["result_sha256"] and s["file"] in manifest
             and s["case"]["case_id"] == case and read(root / case / "result.json")["case_sha256"] == manifest[s["file"]]
             and read(replay.within(root/"source", s["file"])) == s["case"] == read(root/case/"case.json"),
             "Unbound model result or changed case: " + case)
        before, after = c["pre_case_resources"], c["post_case_resources"]
        samples = [before, *c["resources"], after]
        need(all(s["selected_card"] == 5 and s["eligible"] is True and s["blocking_selected_card_foreign_npu_processes"] == []
                 and s["unapproved_device_capable_containers"] == [] for s in samples)
             and all(s["require_selected_card_idle"] is True and s["selected_card_table_idle"] is True for s in (before, after)),
             "Case has a non-eligible selected-card resource sample")
    return selected, manifest


def validate_attempts(roots):
    """Bind one complete matrix or four complete single-case attempts in execution order."""
    roots = [roots] if isinstance(roots, Path) else list(roots)
    need(len(roots) in (1, len(NAMES)) and len(set(p.name for p in roots)) == len(roots)
         and len(set(p.resolve() for p in roots)) == len(roots), "Expected one or four distinct complete attempts")
    bindings, manifest, manifest_sha, prior_end = {}, None, None, None
    owned = set()
    for i, root in enumerate(roots):
        selected, current = validate_attempt(root, NAMES if len(roots) == 1 else (NAMES[i],))
        receipt = read(root / "completion.json")
        need(prior_end is None or prior_end <= receipt["start_epoch"], "Controller attempts overlap or are out of execution order")
        need(receipt["owned_container_id"] not in owned, "Reused owned container across independent attempts")
        owned.add(receipt["owned_container_id"])
        checksum = sha(root / "source_manifest.json")
        need(manifest is None or (manifest == current and checksum == manifest_sha),
             "Source or frozen case manifest differs between attempts")
        manifest, manifest_sha, prior_end = current, checksum, receipt["end_epoch"]
        for row in selected:
            bindings[row["case_id"]] = (root, row)
    need(tuple(bindings) == NAMES, "Expected one execution of each registered model case")
    return bindings, manifest


def load_case(root, selected, manifest, schema, vendor, vendor_sha):
    name, cfg = selected["case_id"], selected["case"]
    mode = "inner1024" if name.startswith("model_fault") else "off"
    case_dir = root / name
    result = read(case_dir / "result.json")
    need(cfg["schema"] == "fault33.pa.model.case.v1" and cfg["mode"] == mode and cfg["capture_msprobe"] is True
         and cfg["execution_kind"] == "model_paged_adapter" and cfg["operator_variant"] == "private_vendor"
         and cfg["approved_vendor_manifest_sha256"] == vendor_sha == result["vendor_manifest_sha256"]
         and cfg["approved_collector_manifest_sha256"] == replay.HIGH_MANIFEST and all(cfg[k] == v for k, v in replay.GEOMETRY.items()),
         "Different model geometry, collector, vendor, or execution mode")
    need(result["schema"] == "fault33.model_paged_adapter.result.v1" and result["model_forward"] is True
         and result["status"] == "CAPTURE_COMPLETED" and result["case_id"] == name and result["mode"] == mode
         and result["all_layers"] == LAYERS and result["original_api_calls"] == result["eligible_decode_ifa_calls"] == 28
         and result["original_prefill_sdpa_calls"] == 28 and result["covered_request_layer_pairs"] == 224
         and result["request_count"] == 8 and result["deterministic_algorithms"] is True
         and result["case_sha256"] == manifest[selected["file"]], "Incomplete model coverage or unbound result")
    fixed = prefill_inputs(case_dir, result)
    ledger = read(case_dir / "model_calls.json")
    calls = ledger["calls"]
    need(sha(case_dir / "model_calls.json") == result["calls_sha256"] and ledger["module_enters"] == ledger["module_exits"] == LAYERS
         and [c["layer_index"] for c in calls] == LAYERS and len(set(c["scope_token"] for c in calls)) == 28 and
         len(set(c["pid"] for c in calls)) == 1, "Case call census or fresh invocations incomplete")
    launcher = read(root / (name + ".vendor_launch.json"))
    need(launcher["case_id"] == name and launcher["case_sha256"] == result["case_sha256"] and launcher["pid"] == calls[0]["pid"]
         and launcher["probe_sha256"] == manifest["model_probe.py"] and launcher["launcher_sha256"] == manifest["vendor_exec.py"]
         and launcher["verified_vendor"]["manifest_sha256"] == vendor_sha and launcher["child_environment"]["ACLNN_CACHE_LIMIT"] == "0",
         "Wrong model/vendor executable or retained ACLNN cache")
    maps = replay.verify_maps(case_dir, cfg, {"calls": calls, "runtime_native_maps_sha256": result["runtime_native_maps_sha256"]}, vendor)
    log, host = root / (name + ".log"), case_dir / "host_events.jsonl"
    need(sha(host) == result["host"]["sha256"] and result["host"]["callbacks"] == 28
         and result["host"]["applied"] == (0 if mode == "off" else 28), "Host ledger or reported count differs")
    windows = native_evidence.bind_host(native_evidence.parse_log(log), host)
    need(len(windows) == 28, "Missing real model IFA execution windows")
    coverage_log(log, calls, name)
    layers = []
    for i, (w, call) in enumerate(zip(windows, calls)):
        need(w["enter"] == call["enter"] and w["call_id"] == call["call_id"] == f"decode_layer_{i:02d}"
             and call["original_api_calls"] == 1 and call["request_ids"] == REQUESTS and call["enter"]["case_id"] == name
             and call["enter"]["monotonic_ns"] <= w["public_host"]["callback_begin_monotonic_ns"]
             <= w["public_host"]["callback_end_monotonic_ns"] <= call["exit"]["monotonic_ns"], "Unbound model layer/host/launch window")
        layers.append({"index": i, "visible": replay.visible_plan(w), "plan": replay.private_plan(w, mode, schema),
                       "values": layer_inputs_and_outputs(call)})
    need(result["private_logits_file"] == "private_output/decode_logits.bin" and
         result["statistics_receipt_sha256"] == sha(case_dir / "statistics_receipt.json"), "Unbound logits or native statistics")
    need(read(case_dir/"config.json").get("precision") == "high" and
         read(case_dir/"collector_provenance.json")["manifest_sha256"] == replay.HIGH_MANIFEST,
         "The recorded model statistics collector does not match the frozen case")
    logits = result["decode_logits"]
    report_meta(logits, LOGITS, {"torch.float16", "torch.float32"})
    file = case_dir / "private_output/decode_logits.bin"
    need(not file.is_symlink() and file.stat().st_size == logits["bytes"], "Missing raw logits or invalid size")
    raw = file.read_bytes()
    validate_payload(logits, raw)
    return {"name": name, "dir": case_dir, "fixed": fixed, "layers": layers, "dtype": logits["dtype"], "logits": raw,
            "proof": {"result_sha256": sha(case_dir/"result.json"), "calls_sha256": result["calls_sha256"],
                      "statistics_receipt_sha256": result["statistics_receipt_sha256"], "log_sha256": sha(log),
                      "host_sha256": sha(host), "maps_sha256": result["runtime_native_maps_sha256"], "maps": maps,
                      "logits_sha256": logits["sha256"], "prefix_hashes": {key: result[key] for key in
                      ("request_mapping_sha256", "prefill_cache_sha256", "release_sha256", "block_table_sha256")}}}


def compare_logits(a, b, dtype, vocab=VOCAB):
    fmt = {"torch.float16": "<e", "torch.float32": "<f"}[dtype]
    width = struct.calcsize(fmt)
    need(len(a) == len(b) == 8*vocab*width, "Logit matrix geometry differs")
    rows = []
    for row in range(8):
        aa = a[row*vocab*width:(row+1)*vocab*width]
        bb = b[row*vocab*width:(row+1)*vocab*width]
        best = [-math.inf, -math.inf]
        best_id, ties = [None, None], [0, 0]
        different = signed_zero = 0
        max_abs = sum_abs = 0.
        for i, ((x,), (y,)) in enumerate(zip(struct.iter_unpack(fmt, aa), struct.iter_unpack(fmt, bb))):
            need(math.isfinite(x) and math.isfinite(y), "Nonfinite model logits")
            for j, value in enumerate((x, y)):
                if value > best[j]:
                    best[j], best_id[j], ties[j] = value, i, 1
                elif value == best[j]:
                    ties[j] += 1
            if x != y:
                diff = abs(x-y)
                different += 1
                max_abs, sum_abs = max(max_abs, diff), sum_abs+diff
            elif x == 0 and aa[i*width:(i+1)*width] != bb[i*width:(i+1)*width]:
                signed_zero += 1
        rows.append({"request_id": REQUESTS[row], "bitwise_equal": aa == bb, "elements": vocab,
                     "numeric_differences": different, "signed_zero_only": signed_zero, "max_abs": max_abs,
                     "mean_abs": sum_abs / vocab, "B_top1": best_id[0], "F_top1": best_id[1],
                     "B_argmax_ties": ties[0], "F_argmax_ties": ties[1], "top1_changed": best_id[0] != best_id[1]})
    return {"elements": 8*vocab, "dtype": dtype, "bitwise_equal": a == b, "rows": rows,
            "numeric_differences": sum(r["numeric_differences"] for r in rows),
            "requests_with_numeric_difference": sum(r["numeric_differences"] > 0 for r in rows),
            "top1_changed_requests": sum(r["top1_changed"] for r in rows), "top1_tie_rule": "First vocabulary index at the maximum"}


def same_inputs(a, b):
    return a["query"] == b["query"] and a["kv"] == b["kv"]


def compare_layers(a, b, same_mode):
    rows = []
    for x, y in zip(a, b):
        left, right = x["values"], y["values"]
        inputs_equal = same_inputs(left, right)
        if same_mode:
            need(left == right and x["visible"] == y["visible"] and x["plan"]["typed"] == y["plan"]["typed"]
                 and x["plan"]["raw_sha256"] == y["plan"]["raw_sha256"]
                 and x["plan"]["workspace_sizes"] == y["plan"]["workspace_sizes"], "Same-mode model layer or verified plan differs")
        changes = {}
        for key in ("attention", "o_proj"):
            changes[key] = [REQUESTS[i] for i, (u, v) in enumerate(zip(left[key]["request_sha256"], right[key]["request_sha256"])) if u != v]
            need((left[key]["sha256"] == right[key]["sha256"]) == (not changes[key]), "Inconsistent row/full hashes")
        if not same_mode and x["index"] == 0:
            need(inputs_equal, "Original first-layer Q and logical/physical K/V values differ before policy application")
        p, q = x["plan"], y["plan"]
        rows.append({"layer": x["index"], "full_and_per_row_QKV_hashes_equal": inputs_equal, "changed_requests": changes,
                     "pre_intervention_native_inner_equal": p["pre_intervention_native_inner"] == q["pre_intervention_native_inner"],
                     "B_inner_loops": [p["selected_inner"], p["outer_update_loops"]],
                     "F_inner_loops": [q["selected_inner"], q["outer_update_loops"]],
                     "verified_partition_changed": p["selected_inner"] != q["selected_inner"] and
                                                 p["outer_update_loops"] != q["outer_update_loops"],
                     "public_tiling_sha256": [p["raw_sha256"], q["raw_sha256"]]})
    return rows


def accept(attempt, vendor_file, abi_file):
    vendor = read(vendor_file)
    schema = replay.verified_abi(vendor, abi_file)
    need(vendor["schema"] == "fault33.pa.private_vendor.v1" and vendor["ifa_registration"]["op_type"] == "IncreFlashAttention"
         and vendor["kernel_argument_raw_observed"] is False, "Unrecognized private vendor")
    vendor_sha = sha(vendor_file)
    bindings, manifest = validate_attempts(attempt)
    cases = [load_case(root, row, manifest, schema, vendor, vendor_sha) for root, row in bindings.values()]
    roots = list(dict.fromkeys(root for root, _ in bindings.values()))
    receipts = {}
    for root in roots:
        complete = read(root/"completion.json")
        receipts[root.name] = {
            "case_ids": [c["case_id"] for c in complete["cases"]],
            "start_epoch": complete["start_epoch"], "end_epoch": complete["end_epoch"],
            "completion_sha256": sha(root/"completion.json"), "source_manifest_sha256": sha(root/"source_manifest.json"),
            "selected_cases_sha256": sha(root/"selected_cases.json"), "container_identity_sha256": sha(root/"container_identity.json"),
        }
    need(all(c["fixed"] == cases[0]["fixed"] and c["dtype"] == cases[0]["dtype"] for c in cases),
         "Model checkpoint/tokenizer, prompts/teacher, prefill KV/logits or block table changed across arms")
    pairs = {}
    for i, j in COMPARISONS:
        a, b = cases[i], cases[j]
        same = (i,j) in ((0,3),(1,2))
        name = a["name"] + "_vs_" + b["name"]
        layers = compare_layers(a["layers"], b["layers"], same)
        logits = compare_logits(a["logits"], b["logits"], a["dtype"])
        if same:
            need(logits["bitwise_equal"], "Non-repeatable fault or unsuccessful fresh-process OFF logits recovery")
        stats = compare_model.compare(*(c["dir"]/"observation/step0/rank0/dump.json" for c in (a,b)))
        raw_first = next((x for x in layers if x["changed_requests"]["attention"]), None)
        obs = stats["first_observed_model_api_difference"]
        agrees = bool(raw_first and raw_first["full_and_per_row_QKV_hashes_equal"] and obs and obs["B"]["native"]
                      and obs["B"]["layer"] == raw_first["layer"] and obs["boundary"] == "output"
                      and obs["input_summary"]["different_finite_fields"] == 0 and obs["input_summary"]["equal_finite_fields"] > 0)
        pairs[name] = {"layers": layers, "logits": logits, "first_full_attention_hash_difference_layer": raw_first["layer"] if raw_first else None,
                       "matched_inputs_at_first_attention_hash_difference": raw_first["full_and_per_row_QKV_hashes_equal"] if raw_first else None,
                       "independent_msprobe_first_agrees_with_complete_attention_hashes": agrees,
                       "independent_statistics": {k: stats[k] for k in ("status", "first_observed_model_api_difference",
                        "first_observed_raw_api_difference", "first_model_metadata_mismatch", "compared_fields",
                        "native_layers", "sources", "collectors", "ordered_full_stack_construct_match", "config_and_collector_match")}}
    reports = list(pairs.values())
    switched = all(row["verified_partition_changed"] and row["pre_intervention_native_inner_equal"] for rpt in reports[:2] for row in rpt["layers"])
    native_changed = all(r["matched_inputs_at_first_attention_hash_difference"] is True for r in reports[:2])
    impact = all(r["logits"]["numeric_differences"] > 0 and any(row["changed_requests"]["o_proj"] for row in r["layers"]) for r in reports[:2])
    located = all(r["independent_msprobe_first_agrees_with_complete_attention_hashes"] for r in reports[:2]) and all(
        r["independent_statistics"]["status"] == "ALIGNED_NO_OBSERVED_FINITE_MODEL_API_DIFFERENCE" for r in reports[2:])
    ready = switched and native_changed and impact and located
    return {"schema": "fault33.model.offline_acceptance.v1", "status": "PASS_ADAPTED_MODEL_AND_STATISTICS" if ready else "VERIFIED_MODEL_WITH_LIMITS",
            "actual_decode_ifa_launches": 112, "applied_fault_calls": 56, "eligible_off_calls_without_intervention": 56,
            "request_layer_incidences_all_runs": 896, "request_layer_incidences_under_control": 448,
            "original_prefill_sdpa_calls": 112, "same_known_inputs_and_prefill_hashes": True,
            "fault_repeat_and_fresh_process_off_recovery_verified": True, "public_plan_switch_and_native_attention_effect": switched and native_changed,
            "downstream_projection_and_real_model_logits_changed": impact, "independent_msprobe_native_first_difference_verified": located,
            "release_approved": ready, "pairs": pairs, "sources": {c["name"]: c["proof"] for c in cases},
            "controller_completion_sha256": receipts[roots[0].name]["completion_sha256"] if len(roots) == 1 else None,
            "frozen_source_manifest_sha256": receipts[roots[0].name]["source_manifest_sha256"], "controller_attempts": receipts,
            "case_to_attempt": {name: root.name for name, (root, _) in bindings.items()},
            "vendor_manifest_sha256": vendor_sha, "public_serializer_schema_sha256": sha(abi_file),
            "verified_vendor_public_abi": vendor["public_abi"],
            "public_host_serializer_projection_only": True, "kernel_argument_raw_observed": False,
            "real_model_forward": True, "historical_serving_incident_reproduced": False,
            "limits": ["HF prefill uses original SDPA. This model's decode deliberately selects paged IFA through an adapter.",
                       "Every layer is one batched kernel call serving eight requests. Request-layer incidences are counted separately.",
                       "Complete model tensors are compared by captured full/per-request hashes; only private logits are re-read element by element.",
                       "Recovery in this matrix uses a fresh process. Same-process disarm is tested separately in operator replay.",
                       "The four runs do not establish production scheduling behavior or a task-accuracy regression."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True, type=Path, action="append",
                        help="One complete matrix, or repeat for four clean singleton attempts in B/F0/F1/R execution order")
    parser.add_argument("--vendor-manifest", required=True, type=Path)
    parser.add_argument("--abi-schema", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        r = accept(args.attempt, args.vendor_manifest, args.abi_schema)
        rc = 0 if r["release_approved"] else 2
    except (ValueError, KeyError, TypeError, IndexError, OSError, struct.error) as error:
        r = {"schema": "fault33.model.offline_acceptance.v1", "status": "INVALID_OR_INCOMPLETE_EVIDENCE",
             "error": str(error), "release_approved": False}
        rc = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as fp:
        json.dump(r, fp, indent=2, allow_nan=False, ensure_ascii=False)
        fp.write("\n")
    print(json.dumps({"status": r["status"], "output": str(args.output), "error": r.get("error")}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

