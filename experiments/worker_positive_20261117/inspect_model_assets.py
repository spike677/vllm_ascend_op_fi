"""Bounded, read-only checkpoint inventory. Does not load weights or runtimes."""
import argparse
import hashlib
import json
import os
from pathlib import Path


ROOTS = ("/home/gyl/models", "/home/ws/models", "/home/haoran/models", "/models",
         "/AIdata/models", "/root/.cache/huggingface/hub", "/home")
SKIP = {".git", "node_modules", "__pycache__", ".venv", "venv", "build", "site-packages"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def inspect_config(path):
    raw = path.read_bytes()
    if len(raw) > 5_000_000:
        return None
    config = json.loads(raw)
    family = str(config.get("model_type", ""))
    name = path.parent.name
    if not any(token in (name + " " + family).lower() for token in ("deepseek", "glm")):
        return None
    files = sorted(p for pattern in ("*.safetensors", "pytorch_model*.bin") for p in path.parent.glob(pattern) if p.is_file())
    indices = []
    for index in sorted(path.parent.glob("*.index.json")):
        content = index.read_bytes()
        doc = json.loads(content)
        if not isinstance(doc.get("weight_map"), dict):
            continue
        shards = sorted(set(doc["weight_map"].values()))
        indices.append({"name": index.name, "sha256": sha(content), "declared_shard_count": len(shards),
                        "missing_shards": [name for name in shards if not (path.parent / name).is_file()],
                        "parameter_names_count": len(doc["weight_map"])})
    return {"path": str(path.parent), "resolved_path": str(path.parent.resolve()),
            "config_sha256": sha(raw), "model_type": family, "architectures": config.get("architectures"),
            "name_is_not_architecture": True,
            "selected_config": {key: config.get(key) for key in (
                "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
                "torch_dtype", "quantization_config", "auto_map") if key in config},
            "weight_file_count": len(files), "weight_bytes": sum(p.stat().st_size for p in files),
            "weight_indices": indices,
            "tokenizer_files": [name for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
                                if (path.parent / name).is_file()],
            "acceptance": "FILE_PRESENCE_ONLY_NO_WEIGHT_LOAD_OR_NPU_ACCEPTANCE"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("REFUSE_OVERWRITE")
    report = {"schema_version": 1, "roots": [], "models": [], "errors": [], "npu_calls": 0,
              "max_depth": 6, "max_directories_per_root": 20000, "weights_loaded": False}
    seen = set()
    for root_text in ROOTS:
        root = Path(root_text)
        item = {"path": root_text, "exists": root.is_dir(), "directories_seen": 0, "truncated": False}
        report["roots"].append(item)
        if not root.is_dir():
            continue
        for current, children, names in os.walk(root, followlinks=False):
            item["directories_seen"] += 1
            if item["directories_seen"] > report["max_directories_per_root"]:
                item["truncated"] = True
                break
            path = Path(current)
            depth = len(path.relative_to(root).parts)
            children[:] = sorted(name for name in children if name not in SKIP) if depth < report["max_depth"] else []
            if "config.json" not in names:
                continue
            candidate = path / "config.json"
            resolved = str(candidate.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                model = inspect_config(candidate)
                if model is not None:
                    report["models"].append(model)
            except (OSError, ValueError, TypeError) as error:
                report["errors"].append({"path": str(candidate), "error": str(error)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "models": len(report["models"]), "roots": report["roots"],
                      "errors": len(report["errors"])}))


if __name__ == "__main__":
    main()
