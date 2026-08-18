#!/usr/bin/env python3
"""Analyze Fault26 P0-P4 residual writer checkpoints and visible neighbors.

The analyzer is deliberately conservative: neighbor results cover only live
QMM tensor/workspace *declared spans* observed for the same Run. They are not
allocator block boundaries and are never described as physical allocations.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}: {exc}") from exc
            if isinstance(event, dict):
                events.append(event)
    return events


def pointer(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 16 if value.startswith("0x") else 10)
        except ValueError:
            return 0
    return 0


def first_change_window(checkpoint: str) -> str:
    return {
        "P1": "AFTER_LEGAL_PRODUCER_BEFORE_QMM",
        "P2": "QMM_EXECUTION_WINDOW",
        "P3": "AFTER_LAST_QMM_BEFORE_NEXT_ADDRMS_WRAPPER",
        "P4": "ADDRMS_WRAPPER_TO_INTERNAL_RUN",
    }.get(checkpoint, "UNKNOWN_WINDOW")


def nearest_visible_neighbors(
    residual_begin: int,
    residual_end: int,
    spans: list[dict[str, Any]],
) -> dict[str, Any]:
    lower: dict[str, Any] | None = None
    upper: dict[str, Any] | None = None
    overlaps: list[dict[str, Any]] = []
    for span in spans:
        begin = pointer(span.get("span_begin"))
        end = pointer(span.get("span_end"))
        if begin == 0 or end <= begin:
            continue
        candidate = {
            "role": span.get("role"),
            "begin": f"0x{begin:x}",
            "end": f"0x{end:x}",
        }
        if end <= residual_begin:
            gap = residual_begin - end
            if lower is None or gap < lower["gap_bytes"]:
                lower = {**candidate, "gap_bytes": gap}
        elif begin >= residual_end:
            gap = begin - residual_end
            if upper is None or gap < upper["gap_bytes"]:
                upper = {**candidate, "gap_bytes": gap}
        else:
            overlaps.append(candidate)
    return {"lower": lower, "upper": upper, "overlaps": overlaps}


def summarize_group(
    key: tuple[int, int],
    checkpoints: list[dict[str, Any]],
    intervals: dict[tuple[int, str, int], list[dict[str, Any]]],
) -> dict[str, Any]:
    pid, session = key
    ordered = sorted(checkpoints, key=lambda event: int(event.get("sequence", 0)))
    counts = Counter(str(event.get("checkpoint", "")) for event in ordered)
    read_failures = [event for event in ordered if event.get("read_ok") is not True]
    device_writes = [event for event in ordered if event.get("device_memory_modified") is not False]
    generations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in ordered:
        generations[int(event.get("producer_sequence", 0))].append(event)
    consumer_confirmed = {
        producer
        for producer, events in generations.items()
        if {"P3", "P4"}.issubset(
            {str(event.get("checkpoint", "")) for event in events}
        )
    }
    all_changes = [
        event
        for event in ordered
        if event.get("checkpoint") != "P0"
        and int(event.get("changed_bytes_from_previous", 0)) > 0
    ]
    changes = [
        event
        for event in all_changes
        if int(event.get("producer_sequence", 0)) in consumer_confirmed
    ]
    expired_or_boundary_changes = [
        event
        for event in all_changes
        if int(event.get("producer_sequence", 0)) not in consumer_confirmed
    ]

    qmm_pairs: dict[tuple[int, str, int], Counter[str]] = defaultdict(Counter)
    neighbor_observations: list[dict[str, Any]] = []
    for event in ordered:
        checkpoint = str(event.get("checkpoint", ""))
        cache = str(event.get("qmm_cache", ""))
        run_index = int(event.get("qmm_run_index", 0))
        producer = int(event.get("producer_sequence", 0))
        if producer not in consumer_confirmed:
            continue
        if checkpoint in {"P1", "P2"} and cache not in {"", "0x0", "(nil)"}:
            qmm_pairs[(producer, cache, run_index)][checkpoint] += 1
        if checkpoint != "P1":
            continue
        residual_begin = pointer(event.get("residual_begin"))
        residual_end = pointer(event.get("residual_end"))
        if residual_begin == 0 or residual_end <= residual_begin:
            continue
        spans = intervals.get((pid, cache, run_index), [])
        neighbors = nearest_visible_neighbors(residual_begin, residual_end, spans)
        neighbor_observations.append(
            {
                "producer_sequence": producer,
                "qmm_cache": cache,
                "qmm_run_index": run_index,
                **neighbors,
            }
        )

    complete_qmm_windows = sum(
        pair.get("P1", 0) == 1 and pair.get("P2", 0) == 1
        for pair in qmm_pairs.values()
    )
    full_generations = 0
    for events in generations.values():
        present = {str(event.get("checkpoint", "")) for event in events}
        if {"P0", "P1", "P2", "P3", "P4"}.issubset(present):
            full_generations += 1

    lower_gaps = [
        item["lower"]["gap_bytes"]
        for item in neighbor_observations
        if item.get("lower") is not None
    ]
    upper_gaps = [
        item["upper"]["gap_bytes"]
        for item in neighbor_observations
        if item.get("upper") is not None
    ]
    overlap_count = sum(len(item["overlaps"]) for item in neighbor_observations)
    first_change = changes[0] if changes else None
    passed = (
        counts["P0"] > 0
        and full_generations > 0
        and len(qmm_pairs) > 0
        and complete_qmm_windows == len(qmm_pairs)
        and not read_failures
        and not device_writes
        and not changes
    )
    return {
        "pid": pid,
        "session": session,
        "pass": passed,
        "classification": (
            "FAULT26_RESIDUAL_WRITER_TIMELINE_NO_UNEXPECTED_CHANGE"
            if passed
            else "FAULT26_RESIDUAL_WRITER_TIMELINE_CHANGE_OR_INCOMPLETE"
        ),
        "legal_producer_api": "aclnnAddRmsNormBias",
        "checkpoint_counts": dict(sorted(counts.items())),
        "residual_generations": len(generations),
        "consumer_confirmed_generations": len(consumer_confirmed),
        "expired_or_boundary_generations": len(generations) - len(consumer_confirmed),
        "full_p0_p4_generations": full_generations,
        "qmm_windows": len(qmm_pairs),
        "complete_p1_p2_qmm_windows": complete_qmm_windows,
        "read_failures": len(read_failures),
        "device_write_events": len(device_writes),
        "unexpected_changes": len(changes),
        "expired_or_boundary_changes_not_classified_as_writer_fault": len(
            expired_or_boundary_changes
        ),
        "first_unexpected_change": (
            None
            if first_change is None
            else {
                "sequence": first_change.get("sequence"),
                "checkpoint": first_change.get("checkpoint"),
                "window": first_change_window(str(first_change.get("checkpoint", ""))),
                "producer_sequence": first_change.get("producer_sequence"),
                "consumer_sequence": first_change.get("consumer_sequence"),
                "qmm_cache": first_change.get("qmm_cache"),
                "qmm_run_index": first_change.get("qmm_run_index"),
                "changed_bytes_from_previous": first_change.get(
                    "changed_bytes_from_previous"
                ),
                "first_diff_from_previous": first_change.get(
                    "first_diff_from_previous"
                ),
            }
        ),
        "visible_live_qmm_declared_span_neighbors": {
            "scope": "same-Run QMM tensor/workspace declared spans only",
            "allocator_block_boundaries_known": False,
            "observations": len(neighbor_observations),
            "lower_neighbor_observations": len(lower_gaps),
            "upper_neighbor_observations": len(upper_gaps),
            "minimum_lower_gap_bytes": min(lower_gaps) if lower_gaps else None,
            "minimum_upper_gap_bytes": min(upper_gaps) if upper_gaps else None,
            "exact_lower_adjacency": sum(gap == 0 for gap in lower_gaps),
            "exact_upper_adjacency": sum(gap == 0 for gap in upper_gaps),
            "overlap_count": overlap_count,
            "examples": neighbor_observations[:20],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    events = load(args.inventory)

    intervals: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        kind = event.get("event")
        if kind == "fault26_qmm_residual_interval" and event.get("span_known") is True:
            intervals[
                (
                    int(event.get("pid", 0)),
                    str(event.get("cache", "")),
                    int(event.get("run_index", 0)),
                )
            ].append(event)
        elif kind == "fault26_residual_checkpoint":
            groups[(int(event.get("pid", 0)), int(event.get("session", 0)))].append(event)

    results = [
        summarize_group(key, checkpoints, intervals)
        for key, checkpoints in sorted(groups.items())
        if key[1] > 0
    ]
    overall_pass = bool(results) and all(result["pass"] for result in results)
    output = {
        "classification": (
            "FAULT26_RESIDUAL_WRITER_TIMELINE_AUDIT_PASS"
            if overall_pass
            else "FAULT26_RESIDUAL_WRITER_TIMELINE_AUDIT_INCOMPLETE_OR_CHANGED"
        ),
        "pass": overall_pass,
        "sessions": results,
        "scope": {
            "device_memory_modified": False,
            "checkpoint_reads": "synchronized full-storage D2H",
            "allocator_block_span": "unknown",
            "kernel_internal_temporary_buffers": "unknown",
        },
    }
    rendered = json.dumps(output, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
