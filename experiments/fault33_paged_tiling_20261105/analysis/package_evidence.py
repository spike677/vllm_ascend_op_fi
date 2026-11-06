"""Export exact original statistics and bounded JSON metadata from completed fault-33 cases.

Example: python package_evidence.py --attempt-bridge outputs/bridge --attempt-fault outputs/control
         --attempt-model outputs/model --output /fresh/public --include-audit --archive /fresh/public.tar.gz
         python package_evidence.py --verify /fresh/public
This does not run a model or make an acceptance decision.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile

import accept_model
import accept_replay
import native_evidence
from msprobe_compare import unique_keys

GROUPS = {"bridge": ("stock_bridge", "baseline"), "fault": ("fault", "restore"), "model": accept_model.NAMES}
STEPS = {"stock_bridge": 0, "baseline": 2, "fault": 2, "restore": 0, **dict.fromkeys(accept_model.NAMES, 1)}
CASES = {"common": {"case.json", "result.json", "runtime_native_maps.json"},
         "statistics": {"config.json", "collector_provenance.json", "statistics_receipt.json"},
         "replay": {"logical_inputs.json", "projection.json"},
         "model": {"release.json", "request_mapping.json", "prefill_cache.json", "block_table.json", "model_calls.json"}}
AUDIT = {"completion.json", "source_manifest.json", "selected_cases.json", "resource_policy.json",
         "container_identity.json", "launch_command.json"}
SCRIPTS = {"controller.py", "env.sh", "vendor_exec.py", "collector_stage.py", "stock_probe.py", "replay.py",
           "model_probe.py", "model_adapter.py", "build_cases.py", "build_model_cases.py", "host_inventory.py",
           "cpu_inventory.py", "gate_check.py", "cpu_model_check.py"}
FORBIDDEN = {"private_output", "native_logs", "cache", ".cache", ".git", "__pycache__"}
BAD_KEYS = {"tensor_payload", "tensor_values", "tensor_data", "raw_tensor", "raw_tensor_data", "activation_data",
            "data_base64", "blob_base64", "tensor_bytes", "serialized_tensor", "payload_base64"}
TENSOR_KEYS = {"type", "dtype", "shape", "Max", "Min", "Mean", "Norm", "requires_grad", "data_name"}
MAX_JSON = 256 * 1024**2
need, sha = accept_replay.need, accept_replay.sha
INVENTORY = "INVENTORY.json"


def json_data(path):
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_JSON, f"Missing or oversized JSON: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique_keys)


def permitted_source(name):
    p = PurePosixPath(name)
    if len(p.parts) == 1:
        return name in SCRIPTS
    return p.parts[0] == "setup" and len(p.parts) <= 5 and p.suffix == ".json" and all(
        re.fullmatch(r"[A-Za-z0-9_-]+(?:\.json)?", part) for part in p.parts[1:])


def permitted(name):
    p = PurePosixPath(name)
    if not name or p.as_posix() != name or p.is_absolute() or "\\" in name or ":" in name or \
            any(s in ("..", ".") or s in FORBIDDEN for s in p.parts):
        return False
    parts = p.parts
    if len(parts) >= 3 and parts[0] == "audit" and parts[1] in GROUPS:
        return (len(parts) == 3 and parts[2] in AUDIT) or (
            len(parts) >= 4 and parts[2] == "source" and permitted_source("/".join(parts[3:]))) or (
            len(parts) == 5 and parts[1:3] == ("model", "attempts") and
            parts[3] in GROUPS["model"] and parts[4] in AUDIT)
    if len(parts) < 3 or parts[0] not in GROUPS or parts[1] not in GROUPS[parts[0]]:
        return False
    group, case = parts[:2]
    n = STEPS[case]
    if len(parts) == 3:
        allowed = CASES["common"] | CASES["model" if group == "model" else "replay"] | {"native_execution.json"}
        if n:
            allowed |= CASES["statistics"]
        if case != "stock_bridge":
            allowed |= {"host_events.jsonl", "vendor_launch.json"}
        return parts[2] in allowed
    return len(parts) == 6 and parts[2] == "observation" and parts[3] in {f"step{i}" for i in range(n)} and \
        parts[4] == "rank0" and parts[5] in {"dump.json", "stack.json", "construct.json"}


def validate_json_tree(doc, *, allow_host_raw=False):
    def walk(v, host=False):
        if isinstance(v, dict):
            host = allow_host_raw and v.get("schema") == "fault33.pa.typed_host.v1"
            if v.get("type") == "torch.Tensor":
                need(v.keys() <= TENSOR_KEYS and isinstance(v.get("dtype"), str) and isinstance(v.get("shape"), list)
                     and not v.get("data_name"), "Unexpected payload/checksum/field in native Tensor statistic")
            if "shape" in v and "dtype" in v:
                need(not ({"buffer", "raw_data", "array", "storage", "flat_data", "base64", "data"} & v.keys()),
                     "Unreviewed embedded tensor storage")
                if "values" in v:
                    need(v["dtype"] == "torch.int32" and v["shape"] == [8, 128] and
                         v["values"] == [list(range(i*128, (i+1)*128)) for i in range(8)],
                         "Only the registered model block table may include its control indices")
            for key, child in v.items():
                need(key.lower() not in BAD_KEYS, f"Unreviewed tensor payload field: {key}")
                if key == "raw_hex":
                    need(host and isinstance(child, str) and v.get("raw_size") == 3000 and len(child) == 6000
                         and re.fullmatch(r"[a-fA-F0-9]+", child), "Only the bounded 3000-byte public host tiling projection is allowed")
                else:
                    walk(child)
        elif isinstance(v, list):
            need(not (len(v) > 32 and all(type(x) is float for x in v)), "Unreviewed array of floating point values")
            need(not (len(v) > 512 and all(type(x) in (int, float) for x in v)), "Unreviewed large numeric payload")
            for child in v:
                walk(child)
        elif isinstance(v, str):
            need(not v.startswith("data:application/octet-stream;base64,") and
                 not (len(v) >= 1024 and re.fullmatch(r"[A-Za-z0-9+/=]+", v)), "Unreviewed encoded value")
    walk(doc)


def validate_file(path, dest):
    need(permitted(dest), "File outside the published allowlist: " + dest)
    ext = path.suffix.lower()
    if ext in (".py", ".sh"):
        need(path.stat().st_size <= 2*1024**2 and b"\0" not in path.read_bytes(), "Invalid frozen text source")
        path.read_text(encoding="utf-8")
    elif ext == ".jsonl":
        need(path.stat().st_size <= MAX_JSON, "Oversized host event ledger")
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line, object_pairs_hook=unique_keys)
                need(event.get("schema") == "fault33.pa.typed_host.v1", "Unexpected host ledger record")
                validate_json_tree(event, allow_host_raw=True)
    else:
        validate_json_tree(json_data(path), allow_host_raw=dest.endswith("/native_execution.json"))


def exact_observation(folder, n):
    root = folder / "observation"
    if not n:
        need(not root.exists(), "Unregistered extra statistics group")
        return []
    expected = [f"observation/step{i}/rank0/{kind}.json" for i in range(n) for kind in ("dump", "stack", "construct")]
    actual = set()
    for f in root.rglob("*"):
        need(not f.is_symlink(), f"Symlink in native statistics: {f}")
        if f.is_file():
            actual.add(f.relative_to(folder).as_posix())
    need(set(expected) == actual, "Original statistics tree contains missing/extra files or tensor payloads")
    return expected


class Selection:
    def __init__(self):
        self.files = {}
        self.groups = {}
        self.roots = []

    def add(self, root, relative, dest, expected=None):
        need(permitted(dest) and dest not in self.files, "Invalid/duplicate public path: " + dest)
        path = accept_replay.within(root, relative)
        need(all(not p.is_symlink() for p in (path, *[q for q in path.parents if root in q.parents])),
             "Symlink in origin path")
        validate_file(path, dest)
        h = sha(path)
        need(expected is None or h == expected, "Capture receipt hash mismatch: " + relative)
        self.files[dest] = {"src": path, "bytes": path.stat().st_size, "sha256": h}

    def generated(self, dest, obj):
        need(permitted(dest) and dest not in self.files and dest.endswith("/native_execution.json"), "Unregistered derived report")
        validate_json_tree(obj, allow_host_raw=True)
        raw = (json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
        need(len(raw) <= MAX_JSON, "Derived launch evidence exceeds JSON size bound")
        self.files[dest] = {"raw": raw, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def select_group(s, group, root, audit):
    roots = [p.resolve(strict=True) for p in (root if isinstance(root, (tuple, list)) else [root])]
    need((len(roots) == 1 or (group == "model" and len(roots) == 4))
         and len(set(roots)) == len(roots) and all(p.is_dir() and p not in s.roots for p in roots)
         and group not in s.groups, "Expected distinct original attempt directories")
    s.roots.extend(roots)
    ids = GROUPS[group]
    root = roots[0]
    if group == "model":
        bindings, manifest = accept_model.validate_attempts(roots)
    else:
        _, manifest, selected = accept_replay.verify_attempt(root, ids)
        bindings = {row["case_id"]: (root, row) for row in selected}
    need(list(bindings) == list(ids), "Selected cases do not match the registered group")
    info = {"cases": list(ids), "source_manifest_sha256": sha(root / "source_manifest.json"), "full_logs": {}}
    s.groups[group] = info
    if len(roots) == 1:
        info.update(attempt_name=root.name, completion_sha256=sha(root / "completion.json"))
    else:
        info.update(attempt_layout="four_single_case_attempts", attempts={})
        # Separate controller ledgers are essential to the provenance of this layout.
        audit = True
        for case, (case_root, _) in bindings.items():
            need(sha(case_root / "source_manifest.json") == info["source_manifest_sha256"],
                 "Original source manifests are not byte-identical")
            complete = json_data(case_root/"completion.json")
            info["attempts"][case] = {"attempt_name": case_root.name,
                "completion_sha256": sha(case_root / "completion.json"), "source_manifest_sha256": info["source_manifest_sha256"],
                "start_epoch": complete["start_epoch"], "end_epoch": complete["end_epoch"]}
            for name in sorted(AUDIT):
                s.add(case_root, name, f"audit/model/attempts/{case}/{name}")
        verify_split_order(info)
    if audit and len(roots) == 1:
        for name in sorted(AUDIT):
            s.add(root, name, f"audit/{group}/{name}")
    if audit:
        # validate_attempts separately checks every source copy in each original attempt.
        for name, checksum in manifest.items():
            need(permitted_source(name), "Unreviewed frozen source path: " + name)
            s.add(root, "source/"+name, f"audit/{group}/source/{name}", checksum)
    for name, (root, row) in bindings.items():
        folder = root / name
        prefix = f"{group}/{name}/"
        result = json_data(folder/"result.json")
        need(result["status"] == "CAPTURE_COMPLETED" and result["case_id"] == name, "Incomplete case")
        for item in sorted(CASES["common"] | CASES["model" if group == "model" else "replay"]):
            s.add(root, f"{name}/{item}", prefix+item)
        n = STEPS[name]
        trios = exact_observation(folder, n)
        if n:
            receipt = json_data(folder / "statistics_receipt.json")
            if group == "model":
                need(result["statistics_receipt_sha256"] == sha(folder/"statistics_receipt.json"), "Model statistics receipt binding")
                collector_hash = receipt["collector_sha256"]
            else:
                need(result["statistics_receipt"] == receipt, "Replay statistics receipt binding")
                collector_hash = receipt["collector_provenance_sha256"]
            need(receipt["steps"] == list(range(n)) and receipt["rank"] == 0, "Statistics steps/rank differ")
            for doc, expected in (("statistics_receipt.json", None), ("config.json", receipt["config_sha256"]),
                                  ("collector_provenance.json", collector_hash)):
                s.add(root, f"{name}/{doc}", prefix+doc, expected)
            listed = {r["path"]: r for r in receipt["native_files"]}
            need(len(listed) == len(trios) == len(receipt["native_files"]) and set(listed) == set(trios), "Wrong native receipt inventory")
            for rel in trios:
                need((folder/rel).stat().st_size == listed[rel]["bytes"], "Native statistics size differs")
                s.add(root, f"{name}/{rel}", prefix+rel, listed[rel]["sha256"])
        log = root/(name+".log")
        log_stat = log.stat()
        windows = native_evidence.parse_log(log)
        if name != "stock_bridge":
            host = folder/"host_events.jsonl"
            native_evidence.bind_host(windows, host)
            s.add(root, f"{name}/host_events.jsonl", prefix+"host_events.jsonl")
            launch_name = name+".vendor_launch.json"
            launch = json_data(root/launch_name)
            need(launch["case_id"] == name and launch["case_sha256"] == manifest[row["file"]]
                 and launch["launcher_sha256"] == manifest["vendor_exec.py"]
                 and launch["probe_sha256"] == manifest["model_probe.py" if group == "model" else "replay.py"],
                 "Private vendor launcher/frozen source mismatch")
            s.add(root, launch_name, prefix+"vendor_launch.json")
        need(len(windows) == result["original_api_calls"] == (28 if group == "model" else len(accept_replay.MODES[name])),
             "Extracted launch count differs from complete capture")
        h = sha(log)
        after = log.stat()
        need((after.st_size, after.st_mtime_ns, after.st_ino) == (log_stat.st_size, log_stat.st_mtime_ns, log_stat.st_ino),
             "Full log changed during selection")
        s.groups[group]["full_logs"][name] = {"source_filename": log.name, "bytes": after.st_size, "sha256": h}
        s.generated(prefix+"native_execution.json", {"schema": "fault33.public.native_execution_subset.v1",
            "case_id": name, "source_log": s.groups[group]["full_logs"][name], "calls": windows,
            "kernel_argument_raw_observed": False, "public_host_raw_is_tiling_metadata": True,
            "full_original_log_included": False, "analysis_source_sha256": sha(Path(native_evidence.__file__).resolve())})


def overlaps(left, right):
    return left == right or left in right.parents or right in left.parents


def verify_split_order(info):
    ids = GROUPS["model"]
    attempts = info.get("attempts", {})
    need(set(attempts) == set(ids), "Split bundle requires four registered model runs")
    need(len({a["completion_sha256"] for a in attempts.values()}) == len(ids),
         "Reused controller receipt in split model bundle")
    previous_end = None
    for name in ids:
        a = attempts[name]
        start, end = a["start_epoch"], a["end_epoch"]
        need(all(type(t) in (int, float) and math.isfinite(t) and t > 0 for t in (start, end))
             and start < end and (previous_end is None or previous_end <= start),
             "Overlapping or misordered split model attempts")
        need(a["source_manifest_sha256"] == info["source_manifest_sha256"], "Split source generation differs")
        previous_end = end


def verify_split_audit(root, info, required):
    verify_split_order(info)
    first = None
    for case in GROUPS["model"]:
        a = info["attempts"][case]
        directory = f"audit/model/attempts/{case}"
        required.update(f"{directory}/{n}" for n in AUDIT)
        completion = f"{directory}/completion.json"
        manifest_path = f"{directory}/source_manifest.json"
        need(sha(root/completion) == a["completion_sha256"] and
             sha(root/manifest_path) == a["source_manifest_sha256"], "Split audit receipt hash mismatch")
        complete = json_data(root/completion)
        container = json_data(root/directory/"container_identity.json")
        selected = json_data(root/directory/"selected_cases.json")
        manifest = json_data(root/manifest_path)
        cases = complete["cases"]
        need(complete["status"] == "CAPTURE_COMPLETED" and complete["probe"] == "model_probe.py" and
             complete["capture"] == "native-logs" and complete["card"] == 5 and
             complete["device_lock_acquired"] is True and complete["device_lock_released"] is True and
             complete["owned_container_removed"] is True and complete["cleanup_errors"] == [] and
             complete["external_container_mutations"] is False and
             complete["owned_container_id"] == container["id"] and
             complete["start_epoch"] == a["start_epoch"] and complete["end_epoch"] == a["end_epoch"] and
             len(cases) == 1 and cases[0]["case_id"] == case and cases[0]["rc"] == 0 and
             complete["start_epoch"] < cases[0]["start"] < cases[0]["end"] < complete["end_epoch"] and
             cases[0]["status"] == "CAPTURE_COMPLETED_NOT_MECHANISM_ACCEPTANCE" and
             cases[0]["result_sha256"] == sha(root/"model"/case/"result.json"),
             "Split controller receipt does not bind a successful, cleaned case")
        need(len(selected) == 1 and selected[0]["case_id"] == case and
             selected[0]["case"] == json_data(root/"model"/case/"case.json"), "Split case selection differs")
        source = selected[0]["file"]
        need(source in manifest and permitted_source(source) and source.startswith("setup/")
             and sha(root/"audit/model/source"/source) == manifest[source]
             and json_data(root/"audit/model/source"/source) == selected[0]["case"]
             and json_data(root/"model"/case/"result.json")["case_sha256"] == manifest[source],
             "Split case differs from frozen source")
        first = first if first is not None else manifest
        need(manifest == first, "Split frozen source declarations differ")
    return first


def fresh_targets(selection, output, archive):
    output = output.resolve()
    targets = [output] + ([archive.resolve()] if archive else [])
    need(all(not p.exists() and not p.is_symlink() for p in targets), "Output bundle/archive must be new")
    need(not archive or (archive.name.endswith(".tar.gz") and not overlaps(*targets)), "Expected distinct tar.gz outside bundle")
    need(not any(overlaps(p, root) for p in targets for root in selection.roots), "Output overlaps source evidence")
    return targets


def export(selection, output, archive=None):
    targets = fresh_targets(selection, output, archive)
    root = targets[0]
    root.mkdir(parents=True, exist_ok=False)
    inventory = []
    for name in sorted(selection.files):
        entry = selection.files[name]
        dest = root/PurePosixPath(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as fp:
            if "raw" in entry:
                fp.write(entry["raw"])
            else:
                with entry["src"].open("rb") as source:
                    shutil.copyfileobj(source, fp, length=1024**2)
        need(dest.stat().st_size == entry["bytes"] and sha(dest) == entry["sha256"], "Copied bytes changed: " + name)
        inventory.append({"path": name, "bytes": entry["bytes"], "sha256": entry["sha256"]})
    record = {"schema": "fault33.public_evidence_inventory.v1", "groups": selection.groups, "files": inventory,
              "source_files_preserved_bytewise": True, "full_logs_included": False, "private_outputs_included": False,
              "scope": "Original statistics, whitelisted metadata and derived native log excerpts; no acceptance decision."}
    with (root/INVENTORY).open("x", encoding="utf-8") as fp:
        json.dump(record, fp, indent=2, ensure_ascii=False, allow_nan=False)
        fp.write("\n")
    summary = verify(root)
    if archive:
        arc = targets[1]
        arc.parent.mkdir(parents=True, exist_ok=True)
        with arc.open("xb") as fp, gzip.GzipFile(fileobj=fp, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w|") as tar:
            for name in [INVENTORY, *(r["path"] for r in inventory)]:
                tar.add(root/PurePosixPath(name), arcname=name, recursive=False)
        summary["archive_sha256"] = sha(arc)
        summary["archive_bytes"] = arc.stat().st_size
    return summary


def verify(root):
    need(root.is_dir() and not root.is_symlink(), "Expected an ordinary bundle directory")
    record = json_data(root/INVENTORY)
    need(record["schema"] == "fault33.public_evidence_inventory.v1" and
         record["full_logs_included"] is False and record["private_outputs_included"] is False, "Wrong inventory scope")
    expected = {r["path"]: r for r in record["files"]}
    need(expected and len(expected) == len(record["files"]) and all(permitted(n) for n in expected),
         "Non-allowlisted or duplicated inventory entry")
    required = set()
    need(record["groups"] and record["groups"].keys() <= GROUPS.keys(), "Unregistered attempt group")
    for g, info in record["groups"].items():
        need(info["cases"] == list(GROUPS[g]) and set(info["full_logs"]) == set(GROUPS[g]), "Incomplete original case census")
        for case in GROUPS[g]:
            names = CASES["common"] | CASES["model" if g == "model" else "replay"] | {"native_execution.json"}
            if STEPS[case]:
                names |= CASES["statistics"] | {f"observation/step{i}/rank0/{kind}.json" for i in range(STEPS[case]) for kind in ("dump","stack","construct")}
            if case != "stock_bridge":
                names |= {"host_events.jsonl", "vendor_launch.json"}
            required.update(f"{g}/{case}/{n}" for n in names)
            evidence = json_data(root/g/case/"native_execution.json")
            need(evidence["source_log"] == info["full_logs"][case] and evidence["case_id"] == case,
                 "Native excerpt origin differs from inventory")
        if "attempt_layout" in info or "attempts" in info:
            need(g == "model" and info.get("attempt_layout") == "four_single_case_attempts",
                 "Unrecognized multiple-attempt group")
            frozen = verify_split_audit(root, info, required)
        elif any(n.startswith(f"audit/{g}/") for n in expected):
            required.update(f"audit/{g}/{n}" for n in AUDIT)
            frozen = json_data(root/"audit"/g/"source_manifest.json")
        else:
            frozen = None
        if frozen is not None:
            for n, checksum in frozen.items():
                rel = f"audit/{g}/source/{n}"
                need(rel in expected and expected[rel]["sha256"] == checksum, "Missing or altered frozen source copy")
                required.add(rel)
    need(required == expected.keys(), "Incomplete or extra files for declared groups")
    actual = set()
    for p in root.rglob("*"):
        need(not p.is_symlink(), "Symlink in public bundle")
        if p.is_file() and p != root/INVENTORY:
            name = p.relative_to(root).as_posix()
            need(name in expected, "Unexpected public file: " + name)
            actual.add(name)
            row = expected[name]
            need(p.stat().st_size == row["bytes"] and sha(p) == row["sha256"], "Offline content mismatch: " + name)
            validate_file(p, name)
    need(actual == expected.keys(), "Missing files from public bundle")
    return {"files": len(expected), "bytes": sum(r["bytes"] for r in expected.values()),
            "inventory_sha256": sha(root/INVENTORY), "groups": sorted(record["groups"]),
            "integrity_verified": True, "generated_acceptance_decision": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attempt-bridge", type=Path)
    p.add_argument("--attempt-fault", type=Path)
    p.add_argument("--attempt-model", type=Path, action="append",
                   help="One four-case attempt, or repeat four times for four clean single-case attempts")
    p.add_argument("--output", type=Path)
    p.add_argument("--archive", type=Path)
    p.add_argument("--include-audit", action="store_true")
    p.add_argument("--verify", type=Path)
    args = p.parse_args()
    roots = {g: getattr(args, "attempt_"+g) for g in GROUPS}
    if args.verify:
        need(not any(roots.values()) and not args.output and not args.archive and not args.include_audit, "Use --verify alone")
        result = verify(args.verify)
    else:
        need(args.output and any(roots.values()), "Select at least one registered attempt and a fresh output")
        selection = Selection()
        for group, root in roots.items():
            if root:
                select_group(selection, group, root, args.include_audit)
        result = export(selection, args.output, args.archive)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

