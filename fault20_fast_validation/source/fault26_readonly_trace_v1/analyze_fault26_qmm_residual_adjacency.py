#!/usr/bin/env python3
"""Audit declared QMM/storage intervals against the next AddRms residual.

This parser is intentionally read-only.  It only accepts runtime events whose
typed QMM slot map and QMM->next-AddRms lifecycle relation were both proven.
The reported spans are declared tensor/workspace ranges, never allocator block
spans.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EVENT = "fault26_qmm_residual_interval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def role_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    known = [event for event in events if event.get("span_known")]
    valid = [event for event in known if event.get("relation_valid")]
    gaps = [int(event["signed_gap_bytes"]) for event in valid]
    nonzero_abs = [abs(value) for value in gaps if value != 0]
    pairs = {
        (event.get("span_begin"), event.get("residual_begin"))
        for event in valid
    }
    return {
        "events": len(events),
        "known_declared_spans": len(known),
        "valid_relations": len(valid),
        "relations": dict(sorted(Counter(
            str(event.get("relation", "unknown")) for event in valid
        ).items())),
        "tail_adjacent": sum(event.get("relation") == "tail_adjacent" for event in valid),
        "reverse_adjacent": sum(event.get("relation") == "reverse_adjacent" for event in valid),
        "overlap": sum(event.get("relation") == "overlap" for event in valid),
        "signed_gap_bytes": {
            "min": min(gaps) if gaps else None,
            "median": int(statistics.median(gaps)) if gaps else None,
            "max": max(gaps) if gaps else None,
            "p05": percentile(gaps, 0.05),
            "p95": percentile(gaps, 0.95),
            "min_nonzero_absolute": min(nonzero_abs) if nonzero_abs else None,
        },
        "distinct_runtime_address_pairs": len(pairs),
        "allocator_span_known": any(event.get("allocator_span_known") for event in events),
    }


def main() -> int:
    args = parse_args()
    parse_errors = 0
    all_events: list[dict[str, Any]] = []
    with args.log.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            if event.get("event") == EVENT:
                event["_line"] = line_number
                all_events.append(event)

    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in all_events:
        by_role[str(event.get("role", "unknown"))].append(event)

    proven = [
        event for event in all_events
        if event.get("slot_map_proven")
        and event.get("lifecycle_match")
        and int(event.get("run_status", -1)) == 0
    ]
    proven_by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in proven:
        proven_by_role[str(event.get("role", "unknown"))].append(event)

    run_keys = {
        (event.get("pid"), event.get("device"), event.get("cache"), event.get("run_index"))
        for event in all_events
    }
    lifecycle_run_keys = {
        (event.get("pid"), event.get("device"), event.get("cache"), event.get("run_index"))
        for event in all_events if event.get("lifecycle_match")
    }
    slot_run_keys = {
        (event.get("pid"), event.get("device"), event.get("cache"), event.get("run_index"))
        for event in all_events if event.get("slot_map_proven")
    }
    output_events = proven_by_role.get("out", [])
    output_slot_proven = [event for event in output_events if event.get("output_slot_proven")]
    output_valid = [event for event in output_slot_proven if event.get("relation_valid")]
    output_relations = Counter(str(event.get("relation")) for event in output_valid)

    if not all_events:
        verdict = "NO_INTERVAL_EVENTS"
    elif not slot_run_keys:
        verdict = "QMM_SLOT_MAP_NOT_PROVEN"
    elif not lifecycle_run_keys:
        verdict = "QMM_NEXT_ADDRMS_LIFECYCLE_NOT_PROVEN"
    elif not output_valid:
        verdict = "QMM_OUTPUT_RESIDUAL_INTERVAL_NOT_COMPARABLE"
    elif output_relations.get("tail_adjacent", 0):
        verdict = "QMM_OUTPUT_TAIL_ADJACENCY_OBSERVED_DECLARED_SPANS"
    elif output_relations.get("reverse_adjacent", 0):
        verdict = "QMM_OUTPUT_REVERSE_ADJACENCY_OBSERVED_DECLARED_SPANS"
    elif output_relations.get("overlap", 0):
        verdict = "QMM_OUTPUT_RESIDUAL_OVERLAP_OBSERVED_DECLARED_SPANS"
    else:
        verdict = "QMM_OUTPUT_RESIDUAL_NOT_ADJACENT_IN_DECLARED_SPANS"

    result = {
        "verdict": verdict,
        "scope": {
            "interval_semantics": "declared tensor storage/workspace spans",
            "allocator_block_span": "unknown",
            "device_memory_modified": False,
        },
        "input": str(args.log),
        "parse_errors": parse_errors,
        "events": len(all_events),
        "pids": sorted({int(event["pid"]) for event in all_events}),
        "devices": sorted({int(event["device"]) for event in all_events}),
        "qmm_runs": len(run_keys),
        "slot_map_proven_runs": len(slot_run_keys),
        "lifecycle_matched_runs": len(lifecycle_run_keys),
        "output": {
            "events_after_all_gates": len(output_events),
            "output_slot_proven": len(output_slot_proven),
            "valid_relations": len(output_valid),
            "relations": dict(sorted(output_relations.items())),
        },
        "roles": {
            role: role_summary(events)
            for role, events in sorted(proven_by_role.items())
        },
        "ungated_role_counts": {
            role: len(events) for role, events in sorted(by_role.items())
        },
        "interpretation": {
            "tail_adjacent": "span_end == residual_begin; supports forward overrun reachability only for declared spans",
            "reverse_adjacent": "residual_end == span_begin; a forward output-tail overrun does not explain this ordering",
            "overlap": "declared spans overlap; inspect aliasing/reuse before calling it OOB",
            "gap": "declared spans are separated by the recorded signed byte distance",
            "allocator_warning": "aclTensor storage_bytes is not the allocator's reserved block size",
        },
    }

    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if verdict not in {
        "NO_INTERVAL_EVENTS",
        "QMM_SLOT_MAP_NOT_PROVEN",
        "QMM_NEXT_ADDRMS_LIFECYCLE_NOT_PROVEN",
        "QMM_OUTPUT_RESIDUAL_INTERVAL_NOT_COMPARABLE",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
