#!/usr/bin/env python3
"""Audit full residual-lifetime API coverage from Fault26 v18 trace events.

This analyzer does not infer a writer from API order. It establishes which
visible L4 Runs/typed execute wrappers occurred between synchronized residual
hash checkpoints, and whether ReduceSum was actually present in the armed
request. The result remains scoped to visible trace events and declared
residual storage; allocator blocks and kernel-internal work are not visible.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


API_SUFFIX = re.compile(r"_[0-9]+$")


def normalized_api(event: dict[str, Any]) -> str:
    value = str(event.get("api_info") or event.get("api") or "")
    return API_SUFFIX.sub("", value)


def load_by_pid(path: Path) -> dict[int, list[dict[str, Any]]]:
    events: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}: {exc}") from exc
            if not isinstance(event, dict):
                continue
            pid = int(event.get("pid", 0))
            if pid > 0:
                events[pid].append(event)
    for pid_events in events.values():
        pid_events.sort(key=lambda event: int(event.get("sequence", 0)))
    return events


def api_counts(events: Iterable[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for event in events:
        kind = event.get("event")
        if kind not in {"fault26_run_enter", "fault26_execute_enter"}:
            continue
        api = normalized_api(event)
        if api:
            counts[api] += 1
    return counts


def summarize_pid(
    pid: int,
    events: list[dict[str, Any]],
    session: int,
) -> dict[str, Any]:
    checkpoints = [
        event
        for event in events
        if event.get("event") == "fault26_residual_checkpoint"
        and int(event.get("session", 0)) == session
    ]
    if not checkpoints:
        return {
            "pid": pid,
            "session": session,
            "complete": False,
            "reason": "session_not_observed",
        }

    first_sequence = min(int(event.get("sequence", 0)) for event in checkpoints)
    last_sequence = max(int(event.get("sequence", 0)) for event in checkpoints)
    before = [event for event in events if int(event.get("sequence", 0)) < first_sequence]
    during = [
        event
        for event in events
        if first_sequence <= int(event.get("sequence", 0)) <= last_sequence
    ]
    after = [event for event in events if int(event.get("sequence", 0)) > last_sequence]

    generations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in checkpoints:
        generations[int(event.get("producer_sequence", 0))].append(event)

    event_by_sequence = {
        int(event.get("sequence", 0)): event
        for event in events
        if int(event.get("sequence", 0)) > 0
    }
    interval_api_counts: dict[str, Counter[str]] = defaultdict(Counter)
    covered_run_events = 0
    covered_execute_events = 0
    consumer_confirmed = 0
    complete_generations = 0
    changing_intervals: list[dict[str, Any]] = []

    for producer, generation in sorted(generations.items()):
        ordered = sorted(generation, key=lambda event: int(event.get("sequence", 0)))
        present = {str(event.get("checkpoint", "")) for event in ordered}
        if {"P3", "P4"}.issubset(present):
            consumer_confirmed += 1
        if not {"P0", "P3", "P4"}.issubset(present):
            continue
        complete_generations += 1
        previous = ordered[0]
        for current in ordered[1:]:
            previous_sequence = int(previous.get("sequence", 0))
            current_sequence = int(current.get("sequence", 0))
            label = f"{previous.get('checkpoint')}->{current.get('checkpoint')}"
            for sequence in range(previous_sequence + 1, current_sequence):
                event = event_by_sequence.get(sequence)
                if event is None:
                    continue
                kind = event.get("event")
                if kind not in {"fault26_run_enter", "fault26_execute_enter"}:
                    continue
                api = normalized_api(event)
                if not api:
                    continue
                interval_api_counts[label][api] += 1
                if kind == "fault26_run_enter":
                    covered_run_events += 1
                else:
                    covered_execute_events += 1
            if (
                current.get("read_ok") is True
                and int(current.get("changed_bytes_from_previous", 0)) > 0
            ):
                changing_intervals.append(
                    {
                        "producer_sequence": producer,
                        "window": label,
                        "sequence": current_sequence,
                        "changed_bytes": int(
                            current.get("changed_bytes_from_previous", 0)
                        ),
                    }
                )
            previous = current

    def selected_counts(source: list[dict[str, Any]]) -> dict[str, int]:
        counts = api_counts(source)
        return {
            api: counts.get(api, 0)
            for api in (
                "aclnnReduceSum",
                "aclnnFusedInferAttentionScoreV3",
                "aclnnAddRmsNormBias",
                "aclnnQuantMatmulWeightNz",
            )
        }

    all_during_counts = api_counts(during)
    return {
        "pid": pid,
        "session": session,
        "complete": complete_generations > 0,
        "sequence_range": [first_sequence, last_sequence],
        "checkpoint_count": len(checkpoints),
        "residual_generations": len(generations),
        "consumer_confirmed_generations": consumer_confirmed,
        "complete_generations": complete_generations,
        "legal_producer_api": "aclnnAddRmsNormBias",
        "selected_api_counts": {
            "before_session": selected_counts(before),
            "during_session": selected_counts(during),
            "after_session": selected_counts(after),
        },
        "all_visible_apis_during_session": dict(all_during_counts.most_common()),
        "visible_apis_between_adjacent_hash_checkpoints": {
            label: dict(counts.most_common())
            for label, counts in sorted(interval_api_counts.items())
        },
        "covered_run_enter_events": covered_run_events,
        "covered_typed_execute_enter_events": covered_execute_events,
        "changing_live_intervals": changing_intervals,
        "reducesum_to_residual_lineage_proven": False,
        "reducesum_lineage_reason": (
            "no_aclnnReduceSum_execute_in_armed_session"
            if all_during_counts.get("aclnnReduceSum", 0) == 0
            else "typed_output_to_residual_alias_not_proven"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    by_pid = load_by_pid(args.inventory)
    workers = [
        summarize_pid(pid, events, args.session)
        for pid, events in sorted(by_pid.items())
    ]
    observed = [worker for worker in workers if worker.get("complete")]
    no_live_changes = bool(observed) and all(
        not worker.get("changing_live_intervals") for worker in observed
    )
    no_session_reducesum = bool(observed) and all(
        worker["selected_api_counts"]["during_session"]["aclnnReduceSum"] == 0
        for worker in observed
    )
    output = {
        "classification": (
            "FAULT26_RESIDUAL_LIFETIME_VISIBLE_API_AUDIT_PASS"
            if no_live_changes
            else "FAULT26_RESIDUAL_LIFETIME_VISIBLE_API_AUDIT_INCOMPLETE_OR_CHANGED"
        ),
        "pass": no_live_changes,
        "fixed_request_reducesum_observed": not no_session_reducesum,
        "reducesum_to_residual_lineage_proven": False,
        "workers": workers,
        "scope": {
            "device_memory_modified": False,
            "evidence_source": "existing v18 D2H-only trace",
            "visible_l4_events_only": True,
            "allocator_blocks_known": False,
            "kernel_internal_temporaries_known": False,
            "wrong_device_pointer_paths_excluded": False,
            "historical_defect_fixed_proven": False,
        },
    }
    rendered = json.dumps(output, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if no_live_changes else 1


if __name__ == "__main__":
    raise SystemExit(main())
