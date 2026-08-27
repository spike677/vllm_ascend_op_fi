#!/usr/bin/env python3
"""Summarize read-only aclnnAscendQuantV3 JSONL traces."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def classify_granularity(x_shape: list[int], scale_shape: list[int], axis: int | None) -> str:
    if not scale_shape or scale_shape == [1]:
        return "shared-scalar-candidate"
    if len(x_shape) >= 2 and scale_shape == [x_shape[-1]]:
        if axis is None:
            return "per-channel-broadcast-candidate-axis-unobserved"
        if axis in (-1, len(x_shape) - 1):
            return "per-channel-hidden-axis"
        return "per-channel-shape-with-unexpected-axis"
    if x_shape and scale_shape == [x_shape[0]]:
        return "per-row-or-per-token-candidate"
    if len(x_shape) >= 2 and scale_shape[0] == x_shape[0]:
        return "per-row-group-candidate"
    return "other-or-broadcasted"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    calls: dict[tuple[int, int], dict[str, Any]] = {}
    malformed = 0
    with args.metadata.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "aclnnAscendQuantV3" not in line and "quant_getws" not in line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            pid = event.get("pid")
            invocation = event.get("invocation")
            if not isinstance(pid, int) or not isinstance(invocation, int):
                continue
            call = calls.setdefault((pid, invocation), {"pid": pid, "invocation": invocation})
            kind = event.get("event")
            if kind == "tensor_metadata":
                call.setdefault("tensors", {})[event.get("role", "unknown")] = {
                    key: event.get(key)
                    for key in (
                        "present",
                        "dtype",
                        "format",
                        "view_shape",
                        "storage_shape",
                        "strides",
                        "view_offset",
                        "numel",
                        "raw_addr",
                    )
                }
            elif kind == "quant_parameters":
                call["parameters"] = {
                    key: event.get(key)
                    for key in ("status", "sqrt_mode", "round_mode", "dst_type", "axis")
                }
            elif kind == "tensor_value_summary":
                call.setdefault("value_summaries", {})[event.get("role", "unknown")] = {
                    key: event.get(key)
                    for key in (
                        "dtype",
                        "numel",
                        "bytes",
                        "copy_status",
                        "fnv1a64",
                        "finite_count",
                        "nonfinite_count",
                        "min",
                        "max",
                        "mean",
                    )
                }

    signatures: collections.Counter[str] = collections.Counter()
    address_sets: dict[str, set[str]] = collections.defaultdict(set)
    value_sets: dict[str, set[str]] = collections.defaultdict(set)
    complete_calls = []
    for call in calls.values():
        tensors = call.get("tensors", {})
        if "x" not in tensors or "scale" not in tensors:
            continue
        x_shape = tensors["x"].get("view_shape") or []
        scale_shape = tensors["scale"].get("view_shape") or []
        axis = call.get("parameters", {}).get("axis")
        granularity = classify_granularity(x_shape, scale_shape, axis)
        scale_summary = call.get("value_summaries", {}).get("scale", {})
        semantic_scale_kind = granularity
        if (
            granularity == "per-channel-hidden-axis"
            and scale_summary.get("copy_status") == 0
            and scale_summary.get("finite_count") == scale_summary.get("numel")
            and scale_summary.get("min") == scale_summary.get("max")
        ):
            semantic_scale_kind = "expanded-scalar-on-hidden-axis"
        signature = {
            "x_shape": x_shape,
            "scale_shape": scale_shape,
            "offset_shape": tensors.get("offset", {}).get("view_shape"),
            "y_shape": tensors.get("y", {}).get("view_shape"),
            "x_dtype": tensors["x"].get("dtype"),
            "scale_dtype": tensors["scale"].get("dtype"),
            "axis": axis,
            "granularity": granularity,
            "semantic_scale_kind": semantic_scale_kind,
        }
        signatures[json.dumps(signature, sort_keys=True, ensure_ascii=False)] += 1
        hidden = str(x_shape[-1]) if x_shape else "unknown"
        scale_addr = tensors["scale"].get("raw_addr")
        if scale_addr:
            address_sets[f"scale_h{hidden}"].add(scale_addr)
        offset_addr = tensors.get("offset", {}).get("raw_addr")
        if offset_addr:
            address_sets[f"offset_h{hidden}"].add(offset_addr)
        scale_hash = call.get("value_summaries", {}).get("scale", {}).get("fnv1a64")
        if scale_hash:
            value_sets[f"scale_h{hidden}"].add(scale_hash)
        complete_calls.append(call)

    result = {
        "source": str(args.metadata),
        "malformed_lines": malformed,
        "complete_quant_calls": len(complete_calls),
        "unique_signatures": [
            {"count": count, **json.loads(signature)}
            for signature, count in sorted(signatures.items())
        ],
        "unique_addresses": {key: sorted(values) for key, values in sorted(address_sets.items())},
        "unique_value_hashes": {key: sorted(values) for key, values in sorted(value_sets.items())},
        "calls": sorted(complete_calls, key=lambda item: (item["pid"], item["invocation"])),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
