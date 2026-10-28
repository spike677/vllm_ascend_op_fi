"""Check this public evidence subset with only Python's standard library.

PASS means the selected package is intact and its records agree. This does not
recompute the private raw-tensor numerical acceptance or run a new experiment.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
API = "NPU.npu_dynamic_quant.127.forward"
MODULE = "Module.model.layers.31.mlp.down_proj.AscendRowParallelLinear.forward.0"
RUNS = {"run002": ("baseline", 2, "H", "letter_counts_B2"),
        "run004": ("fault", 2, "F", "letter_counts_B2"),
        "run012": ("baseline", 8, "H", "even_count_B4"),
        "run014": ("fault", 8, "F", "even_count_B4")}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    # Original msprobe reports contain NaN; preserve and parse their native form.
    return json.loads((ROOT / path).read_bytes())


def digest(data):
    return hashlib.sha256(data).hexdigest()


checks = load("SHA256.json")["files"]
actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()}
require(actual == set(checks) | {"SHA256.json"}, "Missing or unlisted public file")
for name, info in checks.items():
    rel = PurePosixPath(name)
    require(not rel.is_absolute() and ".." not in rel.parts and ":" not in name and "\\" not in name,
            "Unsafe manifest path")
    require(rel.suffix in {".json", ".md", ".py", ".cpp"} or name == ".gitattributes",
            "Unexpected binary file")
    data = (ROOT / name).read_bytes()
    require(len(data) == info["bytes"] and digest(data) == info["sha256"], "Checksum mismatch: " + name)
    if rel.suffix == ".json":
        json.loads(data)
    else:
        data.decode("utf-8")
require((ROOT / ".gitattributes").read_text().strip() == "* -text", "Byte-preservation rule changed")
selection = load("selection.json")
selected = {item["published_path"]: item for item in selection["original_files"]}
require(len(selected) == len(selection["original_files"]), "Duplicate selection")
for name, item in selected.items():
    require(checks[name] == {"bytes": item["source_bytes"], "sha256": item["source_sha256"]},
            "Original-byte source mapping mismatch")
for name, expected in load("source/SHA256.json").items():
    require(digest((ROOT / "source" / name).read_bytes()) == expected, "Frozen source mismatch")

completed = load("records/completed.json")
execution = load("records/execution.json")
require(completed["status"] == "EXECUTED" and completed["runs"] == len(execution) == 17,
        "Original matrix completion mismatch")
require(all(item["return"] == 0 for item in execution), "Original failed run")
execution_by_id = {item["id"]: item for item in execution}
receipts = load("records/derived_completion_receipts.json")
require(receipts["derived"] is True, "Completion receipt must be marked derived")
by_run = {item["run_id"]: item for item in receipts["runs"]}
require(set(by_run) == set(RUNS), "Receipt run coverage mismatch")
receipt_sources = {item["run_id"]: item for item in selection["derived_receipt_sources"]}
dumps = {}
for run, (role, rows, stage, condition) in RUNS.items():
    base = f"{role}/rows{rows}_{run}"
    receipt = by_run[run]
    require(receipt["source_sha256"] == receipt_sources[run]["source_sha256"], "Receipt provenance mismatch")
    require(receipt["status"] == "completed" and receipt["observe"] is True and receipt["backend"] is True,
            "Incomplete selected run")
    require((receipt["published_role"], receipt["stage"], receipt["condition"]) == (role, stage, condition),
            "Receipt run mapping mismatch")
    require(receipt["observation_window"] == [30, 32] and receipt["observation_step"] == 0,
            "Unexpected observation window")
    require(execution_by_id[run]["stage"] == stage and execution_by_id[run]["condition"] == condition,
            "Execution mapping mismatch")
    for rank in (0, 1):
        config_name = f"{base}/config_rank{rank}.json"
        config = load(config_name)
        require(config["task"] == "statistics" and config["level"] == "mix", "Wrong capture mode")
        require(config["statistics"]["summary_mode"] == "md5" and config["statistics"]["tensor_list"] == ["."],
                "Wrong summary/raw-tensor source configuration")
        require(selected[config_name]["source_path"] == f"outputs/fresh_matrix/{run}/config_rank{rank}.json",
                "Config source run/rank mismatch")
        require(receipt["rank_contract_shapes"][str(rank)] == {"shape": [rows, 13824], "dtype": "torch.bfloat16"},
                "Real row count mismatch")
        hits = receipt["backend_hits"][str(rank)]
        require(hits["prepare_hits"] == hits["execute_hits"] == 1 and
                hits["prepare_errors"] == hits["execution_errors"] == 0 and
                hits["fault_executions"] == int(role == "fault"), "Backend execution receipt mismatch")
        for kind in ("dump", "stack", "construct"):
            name = f"{base}/observation/step0/rank{rank}/{kind}.json"
            require(selected[name]["source_path"] == f"outputs/fresh_matrix/{run}/observation/step0/rank{rank}/{kind}.json",
                    "Native observation source mismatch")
            obj = load(name)
            if kind == "dump":
                require(obj["task"] == "statistics" and obj["level"] == "mix" and len(obj["data"]) == 113,
                        "Wrong native dump mode or node count")
                require(f"/{run}/observation/step0/rank{rank}/" in obj["dump_data_dir"], "Wrong native rank directory")
                node = obj["data"][API]
                require(node["input_args"][0]["shape"] == [rows, 13824] and
                        node["output"][0]["shape"] == [rows, 13824] and node["output"][1]["shape"] == [rows],
                        "Wrong quantization tensor metadata shapes")
                dumps[(run, rank)] = obj
            elif kind == "stack":
                stacks = [frames for names, frames in obj.values() if API in names]
                require(len(stacks) == 1 and len(stacks[0]) > 0, "Missing native API stack")
            else:
                require(API in obj and obj[API] == MODULE, "Native parent module mismatch")

for baseline, fault in (("run002", "run004"), ("run012", "run014")):
    pair = "pair" + baseline[3:] + "_" + fault[3:]
    official_dir = ROOT / "reports/official" / pair
    findings = list(official_dir.glob("diff_analyze_*.json"))
    require(len(findings) == 1, "Expected one official diff analysis")
    official_candidates = json.loads(findings[0].read_bytes())
    require(set(official_candidates) == {"rank_0", "rank_1"}, "Wrong official candidate ranks")
    for rank in (0, 1):
        a = dumps[(baseline, rank)]["data"][API]
        b = dumps[(fault, rank)]["data"][API]
        require(a["input_args"] == b["input_args"], "Observed boundary input metadata differs")
        require(a["output"][0]["md5"] != b["output"][0]["md5"] and
                a["output"][1]["md5"] != b["output"][1]["md5"], "Missing Q/scale digest difference")
        require(b["output"][1]["Min"] == b["output"][1]["Max"], "Fault scale summary is not uniform")
        report = load(f"reports/independent/{pair}_rank{rank}.json")
        first = report["first_difference"]
        require(report["status"] == "DIFFERENCE" and first["api"] == API and not first["input_differences"] and
                first["output_differences"] and report["earlier_observed_nodes_exact"] == 72 and
                report["first_node_parent_module"] == MODULE, "Independent original report boundary mismatch")
        files = list(official_dir.glob(f"compare_result_rank{rank}_*.json"))
        require(len(files) == 1, "Expected one official rank comparison")
        comparison = json.loads(files[0].read_bytes())
        first_changed = next((name for name, value in comparison.items() if value["is_same"] is False), None)
        require(first_changed == API, "Official first observed difference mismatch")
        require([item["op_name"] for item in official_candidates[f"rank_{rank}"]] == [API],
                "Official source candidate mismatch")
require(load("reports/independent/summary.json")["status"] == "PASS" and
        load("reports/official/acceptance_summary.json")["status"] == "PASS", "Original acceptance status mismatch")
print(json.dumps({"status": "PASS", "meaning": "Public subset integrity and original-record consistency only",
                  "files": len(actual), "bytes": sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()),
                  "selected_runs": 4, "configurations": 8, "native_jsons": 24,
                  "tensor_binary_files": 0, "new_npu_execution": False}, ensure_ascii=False))
