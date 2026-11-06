"""Shared validation of native msprobe statistics, without a target operator."""
import hashlib,json,math,re
from pathlib import Path
FRAME=re.compile(r"^File .+, line \d+, in [^,]+,")
STATS=("Max","Min","Mean","Norm")
FORBIDDEN={"data_name","md5","MD5","checksum","tensor_data"}
PATCHED_SHA={"task_local_fp32_high_v1":"3aa7ede049bc6cb4d54a572a85f907214a98890a3c9544ed7f75e25df1306141", "task_local_cpu_fp64_high_v1":"6688aa3e90849027b12458add4c9da6ac4b334b57e835b742df8403c2fc474b2"}
def require(ok, message):
    if not ok:
        raise ValueError(message)


def unique_keys(items):
    result = {}
    for key, value in items:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def read(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 256 * 1024**2,
            f"Missing, symlinked or oversized JSON: {path}")
    raw = path.read_bytes()
    return json.loads(raw, object_pairs_hook=unique_keys), hashlib.sha256(raw).hexdigest()


def canonical(name):
    if name.startswith("Module."):
        return re.sub(r"\.(forward|backward)\.\d+$", r".\1", name)
    return re.sub(r"\.\d+\.(forward|backward)$", r".\1", name)


def is_tensor(item):
    return isinstance(item, dict) and item.get("type") == "torch.Tensor"


def collector(case, config, hashes):
    precision = config.get("precision", "low")
    require(precision in ("low", "high"), "Unknown statistics precision")
    path = case / "collector_provenance.json"
    if not path.exists():
        require(precision != "high", "High precision needs actual collector provenance")
        return {"configured_precision": precision, "implementation_verified": False}
    record, hashes["collector_provenance"] = read(path)
    loaded = record.get("actual_loaded", {})
    require(isinstance(loaded, dict) and isinstance(loaded.get("path"), str) and
            loaded["path"] == loaded.get("method_code_file") and
            re.fullmatch("[0-9a-f]{64}", loaded.get("sha256", "")), "Missing collector code identity")
    require(record.get("precision") == precision, "Collector precision disagrees with config")
    if precision == "high":
        require(record.get("modified_official_collector") is True and record.get("variant") in PATCHED_SHA
                and loaded["sha256"] == record.get("patched_processor_sha256") == PATCHED_SHA[record["variant"]],
                "High precision collector differs from disclosed task-local implementation")
    return {"configured_precision": precision, "variant": record.get("variant"), "actual_loaded": loaded,
            "modified_official_collector": record.get("modified_official_collector"),
            "manifest_sha256": record.get("manifest_sha256"), "implementation_verified": False,
            "verification": "Reported import/code identity, no independent rehash of installed package"}


def load_step(path):
    path = Path(path)
    require(path.name == "dump.json" and path.parent.name == "rank0"
            and re.fullmatch(r"step\d+", path.parents[1].name) and path.parents[2].name == "observation",
            "Expected case/observation/stepN/rank0/dump.json")
    require(not any(p.is_symlink() for p in path.parents[:4]), "Symlink in observation path")
    step = int(path.parents[1].name[4:])
    case = path.parents[3]
    hashes = {}
    config, hashes["config"] = read(case / "config.json")
    require(config.get("task") == "statistics" and config.get("level") == "mix" and
            config.get("async_dump") is False and step in config.get("step", []) and
            config.get("rank") in ([], [0]), "Require synchronous statistics/mix capture and configured rank/step")
    settings = config.get("statistics", {})
    require(settings.get("summary_mode") == "statistics" and settings.get("data_mode") == ["all"] and
            all(settings.get(k) == [] for k in ("list", "scope", "tensor_list")),
            "Require unfiltered statistics mode and no tensor list")
    disclosure = collector(case, config, hashes)
    dump, hashes["dump"] = read(path)
    stack, hashes["stack"] = read(path.with_name("stack.json"))
    construct, hashes["construct"] = read(path.with_name("construct.json"))
    require(dump.get("framework") == "pytorch" and dump.get("level") == "mix" and
            dump.get("task") == "statistics" and dump.get("dump_data_dir") is None
            and isinstance(dump.get("data"), dict) and dump["data"], "Malformed raw mix/statistics dump")
    require(isinstance(stack, dict) and isinstance(construct, dict), "Missing stack or construct")
    frames = {}
    for row in stack.values():
        require(isinstance(row, list) and len(row) == 2 and isinstance(row[0], list) and
                isinstance(row[1], list) and row[1] and all(isinstance(s, str) and FRAME.match(s) for s in row[1]),
                "Malformed source stack")
        for name in row[0]:
            require(name not in frames, "Duplicate stack attribution")
            frames[name] = row[1]
    entries = []
    for name, node in dump["data"].items():
        require(isinstance(node, dict) and name in frames and name in construct, f"Missing real stack/construct: {name}")
        chain, seen, parent = [], {name}, construct[name]
        while parent is not None:
            require(parent in construct and parent not in seen, f"Invalid construct ancestry: {name}")
            seen.add(parent)
            chain.append(parent)
            parent = construct[parent]
        entries.append({"name": name, "node": node, "frames": frames[name], "parents": chain})
    return {"entries": entries, "hashes": hashes, "step": step,
            "config": {k: v for k, v in config.items() if k != "dump_path"}, "collector": disclosure}



def safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def compare_values(left, right):
    result = {"metadata_differences": [], "statistical_differences": [], "unobservable": [],
              "matching_finite_statistics": 0}

    def visit(a, b, path):
        if is_tensor(a) and is_tensor(b):
            for value in (a, b):
                require(isinstance(value.get("dtype"), str) and isinstance(value.get("shape"), list)
                        and all(type(n) is int and n >= 0 for n in value["shape"]), "Malformed tensor descriptor")
                require(not any(key in value and value[key] is not None for key in FORBIDDEN),
                        "Tensor payload/checksum outside statistics contract")
            meta_a, meta_b = ({k: v for k, v in val.items() if k not in STATS} for val in (a, b))
            if meta_a != meta_b:
                result["metadata_differences"].append({"path": path, "B": meta_a, "F": meta_b})
                return
            for key in STATS:
                av, bv = a.get(key), b.get(key)
                item = {"path": f"{path}.{key}", "B": safe(av), "F": safe(bv),
                        "B_present": key in a, "F_present": key in b}
                allowed = (int, float, bool) if a["dtype"] == "torch.bool" else (int, float)
                if type(av) not in allowed or type(bv) not in allowed or not (math.isfinite(av) and math.isfinite(bv)):
                    result["unobservable"].append(item)
                elif av != bv:
                    result["statistical_differences"].append(item)
                else:
                    result["matching_finite_statistics"] += 1
        elif is_tensor(a) or is_tensor(b) or type(a) is not type(b):
            result["metadata_differences"].append({"path": path, "reason": "structure or type differs"})
        elif isinstance(a, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    result["metadata_differences"].append({"path": f"{path}.{key}", "reason": "missing field"})
                else:
                    visit(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, list):
            if len(a) != len(b):
                result["metadata_differences"].append({"path": path, "B_length": len(a), "F_length": len(b)})
            else:
                for i, (aa, bb) in enumerate(zip(a, b)):
                    visit(aa, bb, f"{path}[{i}]")
        elif safe(a) != safe(b):
            result["metadata_differences"].append({"path": path, "B": safe(a), "F": safe(b)})

    visit(left, right, "$")
    return result


def inputs(node):
    return {key: node[key] for key in ("input_args", "input_kwargs", "parameters") if key in node}


def signature(entry):
    return canonical(entry["name"]), [canonical(p) for p in entry["parents"]], entry["frames"]


