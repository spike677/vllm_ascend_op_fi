#!/usr/bin/env python3
"""Offline full-coverage QMM -> next AddRms declared-span audit.

The live queue is a strict runtime gate, but cache/GWS scheduling means it can
be intentionally conservative.  This analyzer reuses the already validated
L4 ordering rule: QMM Run events between two consecutive real AddRms executes
belong to the latter consumer.  The consumer is accepted only when the prior
AddRms output.x storage exactly equals the current AddRms input.x2 storage.

No tensor content is read and no runtime state is modified.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


INTERVAL_EVENT = "fault26_qmm_residual_interval"
ROLES = (
    "x1", "x2", "x1_scale", "x2_scale", "y_scale", "x1_offset",
    "x2_offset", "y_offset", "bias", "out", "workspace",
)


def load(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                events.append(value)
    return events


def pointer(value: Any) -> int:
    if not isinstance(value, str):
        return 0
    try:
        return int(value, 16)
    except ValueError:
        return 0


def classify(begin: int, size: int, residual_begin: int, residual_size: int) -> tuple[str, int]:
    if begin <= 0 or size <= 0 or residual_begin <= 0 or residual_size <= 0:
        return "unknown", 0
    end = begin + size
    residual_end = residual_begin + residual_size
    if begin < residual_end and residual_begin < end:
        return "overlap", 0
    if end == residual_begin:
        return "tail_adjacent", 0
    if residual_end == begin:
        return "reverse_adjacent", 0
    if end < residual_begin:
        return "qmm_before_residual_gap", residual_begin - end
    return "residual_before_qmm_gap", -(begin - residual_end)


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    gaps = [int(sample["signed_gap_bytes"]) for sample in samples]
    nonzero = [abs(value) for value in gaps if value]
    return {
        "samples": len(samples),
        "relations": dict(sorted(Counter(sample["relation"] for sample in samples).items())),
        "tail_adjacent": sum(sample["relation"] == "tail_adjacent" for sample in samples),
        "reverse_adjacent": sum(sample["relation"] == "reverse_adjacent" for sample in samples),
        "overlap": sum(sample["relation"] == "overlap" for sample in samples),
        "signed_gap_bytes": {
            "min": min(gaps) if gaps else None,
            "median": int(statistics.median(gaps)) if gaps else None,
            "max": max(gaps) if gaps else None,
            "min_nonzero_absolute": min(nonzero) if nonzero else None,
        },
        "distinct_runtime_address_pairs": len({
            (sample["span_begin"], sample["residual_begin"]) for sample in samples
        }),
        "allocator_span_known": False,
    }


def analyze_pid(pid: int, events: list[dict[str, Any]]) -> dict[str, Any]:
    summaries_by_executor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tensors_by_executor_role: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    executes_by_executor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    interval_groups: dict[tuple[str, int], dict[str, Any]] = {}

    for event in events:
        kind = event.get("event")
        if kind == "fault26_typed_gws" and event.get("api") == "aclnnAddRmsNormBias":
            summaries_by_executor[str(event.get("executor", ""))].append(event)
        elif kind == "fault26_typed_gws_tensor" and event.get("api") == "aclnnAddRmsNormBias":
            key = (str(event.get("executor", "")), str(event.get("role", "")))
            tensors_by_executor_role[key].append(event)
        elif kind == "fault26_execute_enter" and event.get("api") == "aclnnAddRmsNormBias":
            executes_by_executor[str(event.get("executor", ""))].append(event)
        elif kind == INTERVAL_EVENT:
            key = (str(event.get("cache", "")), int(event.get("run_index", 0)))
            group = interval_groups.setdefault(key, {
                "ts_ns": int(event.get("ts_ns", 0)),
                "device": int(event.get("device", -1)),
                "slot_map_proven": bool(event.get("slot_map_proven")),
                "output_slot_proven": bool(event.get("output_slot_proven")),
                "run_status": int(event.get("run_status", -1)),
                "roles": {},
            })
            group["ts_ns"] = min(group["ts_ns"], int(event.get("ts_ns", 0)))
            group["roles"][str(event.get("role", ""))] = event

    records_by_executor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    gws_complete = True
    for executor, summaries in summaries_by_executor.items():
        summaries.sort(key=lambda event: int(event.get("sequence", 0)))
        role_lists: dict[str, list[dict[str, Any]]] = {}
        for role in ("input.x2", "output.x"):
            values = tensors_by_executor_role.get((executor, role), [])
            values.sort(key=lambda event: int(event.get("sequence", 0)))
            role_lists[role] = values
            if len(values) != len(summaries):
                gws_complete = False
        for index, summary in enumerate(summaries):
            records_by_executor[executor].append({
                "summary": summary,
                "x2": role_lists["input.x2"][index] if index < len(role_lists["input.x2"]) else {},
                "x_out": role_lists["output.x"][index] if index < len(role_lists["output.x"]) else {},
            })

    consumers: list[dict[str, Any]] = []
    fifo_complete = True
    for executor, executes in executes_by_executor.items():
        executes.sort(key=lambda event: int(event.get("ts_ns", 0)))
        records = records_by_executor.get(executor, [])
        if len(records) != len(executes):
            fifo_complete = False
        for index, execute in enumerate(executes):
            consumers.append({
                "execute": execute,
                "record": records[index] if index < len(records) else {},
            })
    consumers.sort(key=lambda item: int(item["execute"].get("ts_ns", 0)))

    qmm_groups = sorted(interval_groups.values(), key=lambda group: group["ts_ns"])
    qmm_cursor = 0
    previous_execute_ns = 0
    mapped = 0
    safe = 0
    boundary_rejected = 0
    qmm_after_last = 0
    distribution: Counter[int] = Counter()
    samples_by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    samples_by_device_role: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)

    previous_consumer: dict[str, Any] | None = None
    for ordinal, consumer in enumerate(consumers, 1):
        execute_ns = int(consumer["execute"].get("ts_ns", 0))
        interval: list[dict[str, Any]] = []
        while qmm_cursor < len(qmm_groups):
            qmm_ns = qmm_groups[qmm_cursor]["ts_ns"]
            if qmm_ns <= previous_execute_ns:
                qmm_cursor += 1
                continue
            if qmm_ns > execute_ns:
                break
            interval.append(qmm_groups[qmm_cursor])
            qmm_cursor += 1
        distribution[len(interval)] += 1
        mapped += len(interval)

        record = consumer.get("record", {})
        x2 = record.get("x2", {}) if isinstance(record, dict) else {}
        residual_begin = pointer(x2.get("storage"))
        residual_bytes = int(x2.get("storage_bytes", 0) or 0)
        previous_out = {}
        if previous_consumer is not None:
            previous_record = previous_consumer.get("record", {})
            if isinstance(previous_record, dict):
                previous_out = previous_record.get("x_out", {})
        residual_link = (
            previous_consumer is not None
            and residual_begin != 0
            and residual_begin == pointer(previous_out.get("storage"))
            and residual_bytes != 0
            and residual_bytes == int(previous_out.get("storage_bytes", 0) or 0)
        )

        if residual_link:
            safe += len(interval)
            for group in interval:
                if not group["slot_map_proven"] or group["run_status"] != 0:
                    continue
                for role, span in group["roles"].items():
                    if role not in ROLES or not span.get("span_known"):
                        continue
                    if role == "out" and not group["output_slot_proven"]:
                        continue
                    begin = pointer(span.get("span_begin"))
                    size = int(span.get("span_bytes", 0) or 0)
                    relation, gap = classify(begin, size, residual_begin, residual_bytes)
                    if relation == "unknown":
                        continue
                    sample = {
                        "relation": relation,
                        "signed_gap_bytes": gap,
                        "span_begin": begin,
                        "span_bytes": size,
                        "residual_begin": residual_begin,
                        "residual_bytes": residual_bytes,
                    }
                    samples_by_role[role].append(sample)
                    samples_by_device_role[(int(group["device"]), role)].append(sample)
        elif previous_consumer is not None:
            boundary_rejected += len(interval)
        previous_consumer = consumer
        previous_execute_ns = execute_ns

    qmm_after_last = len(qmm_groups) - qmm_cursor
    output = samples_by_role.get("out", [])
    output_relations = Counter(sample["relation"] for sample in output)
    if not output:
        verdict = "NO_SAFE_OUTPUT_INTERVALS"
    elif output_relations.get("tail_adjacent"):
        verdict = "QMM_OUTPUT_TAIL_ADJACENCY_OBSERVED_DECLARED_SPANS"
    elif output_relations.get("reverse_adjacent"):
        verdict = "QMM_OUTPUT_REVERSE_ADJACENCY_OBSERVED_DECLARED_SPANS"
    elif output_relations.get("overlap"):
        verdict = "QMM_OUTPUT_RESIDUAL_OVERLAP_OBSERVED_DECLARED_SPANS"
    else:
        verdict = "QMM_OUTPUT_RESIDUAL_NOT_ADJACENT_IN_DECLARED_SPANS"

    return {
        "pid": pid,
        "verdict": verdict,
        "gws_complete": gws_complete,
        "executor_fifo_complete": fifo_complete,
        "add_rms_gws": sum(len(values) for values in summaries_by_executor.values()),
        "add_rms_executes": len(consumers),
        "qmm_runs": len(qmm_groups),
        "output_slot_proven_runs": sum(
            bool(group["output_slot_proven"]) for group in qmm_groups
        ),
        "qmm_to_next_consumer_mapped": mapped,
        "qmm_after_last_consumer": qmm_after_last,
        "safe_qmm_candidates": safe,
        "boundary_rejected_qmm": boundary_rejected,
        "consumer_qmm_count_distribution": {
            str(key): value for key, value in sorted(distribution.items())
        },
        "roles": {
            role: summarize(samples) for role, samples in sorted(samples_by_role.items())
        },
        "per_device": {
            str(device): {
                role: summarize(samples)
                for (sample_device, role), samples in sorted(samples_by_device_role.items())
                if sample_device == device
            }
            for device in sorted({device for device, _ in samples_by_device_role})
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    events = load(args.inventory)
    by_pid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if "pid" in event:
            by_pid[int(event["pid"])].append(event)
    results = [analyze_pid(pid, values) for pid, values in sorted(by_pid.items())]
    results = [result for result in results if result["qmm_runs"] or result["add_rms_executes"]]
    overall_pass = bool(results) and all(
        result["verdict"] == "QMM_OUTPUT_RESIDUAL_NOT_ADJACENT_IN_DECLARED_SPANS"
        and result["gws_complete"]
        and result["executor_fifo_complete"]
        and result["qmm_to_next_consumer_mapped"] == result["qmm_runs"]
        and result["qmm_after_last_consumer"] == 0
        and result["safe_qmm_candidates"] > 0
        and result["output_slot_proven_runs"] == result["qmm_runs"]
        and result["roles"].get("out", {}).get("samples")
            == result["safe_qmm_candidates"]
        for result in results
    )
    report = {
        "classification": (
            "FAULT26_QMM_RESIDUAL_DECLARED_INTERVAL_AUDIT_PASS"
            if overall_pass else "FAULT26_QMM_RESIDUAL_DECLARED_INTERVAL_AUDIT_INCOMPLETE"
        ),
        "pass": overall_pass,
        "scope": {
            "device_memory_modified": False,
            "interval_semantics": "declared tensor storage/workspace spans",
            "allocator_block_span": "unknown",
            "physical_oob_proven": False,
        },
        "pids": results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
