"""Offline source intake. Never imports installed model code or grants admission."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

REQUIRED = (
    "vllm_ascend/models/deepseek_v4/model.py",
    "vllm_ascend/ops/fused_moe/fused_moe.py",
    "vllm_ascend/ops/fused_moe/routed_experts.py",
    "vllm_ascend/ops/fused_moe/token_dispatcher.py",
    "vllm_ascend/ops/fused_moe/prepare_finalize.py",
    "vllm_ascend/ops/fused_moe/moe_comm_method.py",
    "vllm_ascend/models/common/ops/sequence_parallel.py",
)
TERMS = ("router", "topk", "top_k", "select_experts", "dispatch", "combine",
         "prepare", "finalize", "num_hash_layers", "sp_shard")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def checked_file(root, relative):
    root = Path(root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or "\\" in relative or ":" in relative:
        raise ValueError("SOURCE_PATH_OUTSIDE_ROOT")
    path = (root / rel).resolve()
    if not path.is_relative_to(root):
        raise ValueError("SOURCE_PATH_OUTSIDE_ROOT")
    return path


def scan_file(path):
    raw = path.read_bytes()
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError("SOURCE_FILE_TOO_LARGE")
    tree = ast.parse(raw, filename=str(path))
    definitions, calls, imports = [], [], []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions.append(dict(name=node.name, line=node.lineno, end_line=node.end_lineno))
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if any(term in name.lower() for term in TERMS):
                calls.append(dict(call=name, line=node.lineno))
        if isinstance(node, ast.ImportFrom):
            if any(term in (node.module or "").lower() for term in TERMS):
                imports.append(dict(module=node.module, names=[x.name for x in node.names], line=node.lineno))
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                definitions=definitions, candidate_calls=calls, candidate_imports=imports)


def layer_rule(config, layer_index):
    count = config.get("num_hash_layers")
    if type(count) is not int or count < 0 or type(layer_index) is not int or layer_index < 0:
        return dict(num_hash_layers=count, layer_index=layer_index, candidate=None,
                    reason="CONFIG_OR_LAYER_INDEX_UNPROVEN")
    return dict(num_hash_layers=count, layer_index=layer_index,
                candidate=layer_index >= count,
                first_non_hash_candidate=layer_index == count,
                reason="SOURCE_RULE_ONLY_REQUIRES_ACTUAL_HASH_FLAG_DRAFT_FLAG_AND_NAMED_MODULE")


def review(root, *, upstream=None, extra_paths=(), config=None, layer_index=None):
    paths = sorted(set(REQUIRED).union(extra_paths))
    rows, missing, different = [], [], []
    for rel in paths:
        path = checked_file(root, rel)
        row = dict(path=rel, present=path.is_file(), upstream_sha256=None, matches_upstream=None)
        if path.is_file():
            row.update(scan_file(path))
            if upstream:
                ref = checked_file(upstream, rel)
                if ref.is_file():
                    row["upstream_sha256"] = hashlib.sha256(ref.read_bytes()).hexdigest()
                    row["matches_upstream"] = row["sha256"] == row["upstream_sha256"]
                    if not row["matches_upstream"]:
                        different.append(rel)
        else:
            missing.append(rel)
        rows.append(row)
    identity = [dict(path=r["path"], sha256=r.get("sha256"), bytes=r.get("bytes")) for r in rows]
    return dict(schema="dcat.v4.source.intake.v1", files=rows, missing_files=missing,
                different_files=different, source_manifest_sha256=hashlib.sha256(canonical(identity)).hexdigest(),
                layer=layer_rule(config or {}, layer_index),
                actual_runtime_branch=None, installed_identity_verified=False,
                selector_implementation_review_required=True,
                status="STATIC_CANDIDATES_ONLY", qualification_granted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Extracted installed package tree; no imports")
    parser.add_argument("--upstream", help="Separate verified v0.30.0 candidate source tree")
    parser.add_argument("--extra", action="append", default=[], help="Actual vLLM selector/FusedMoE implementation path")
    parser.add_argument("--config")
    parser.add_argument("--layer-index", type=int)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else None
    print(canonical(review(args.root, upstream=args.upstream, extra_paths=args.extra,
                           config=config, layer_index=args.layer_index)).decode("utf-8"))


if __name__ == "__main__":
    main()
