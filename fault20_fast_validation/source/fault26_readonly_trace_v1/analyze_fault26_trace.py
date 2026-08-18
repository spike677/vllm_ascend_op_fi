#!/usr/bin/env python3
"""Summarize Fault26 read-only JSONL without reading tensor contents."""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class TensorSlot:
    sequence: int
    formal_ordinal: int
    slot_candidate: int
    slot_proven: bool
    address: int
    size: int
    workspace_overlap: bool
    workspace_gap: int


@dataclass
class Run:
    sequence: int
    pid: int
    tid: int
    device: int
    api: str
    execute_api: str
    cache: str
    run_index: int
    stream: str
    workspace: int
    workspace_size: int
    address_count: int
    slot_proven: bool
    status: int
    tensors: list[TensorSlot] = field(default_factory=list)


def parse_pointer(value: Any) -> int:
    if value in (None, "", "(nil)", "0x0"):
        return 0
    if isinstance(value, int):
        return value
    try:
        return int(str(value), 16)
    except ValueError:
        return 0


def load_events(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as error:
                print(
                    f"warning: skip invalid JSON line {line_number}: {error}",
                    file=sys.stderr,
                )
                continue
            if isinstance(value, dict):
                yield value


def run_key(event: dict[str, Any]) -> tuple[int, str, int]:
    return (
        int(event.get("pid", -1)),
        str(event.get("cache", "")),
        int(event.get("run_index", 0)),
    )


def build_runs(events: Iterable[dict[str, Any]]) -> tuple[list[Run], dict[str, int]]:
    run_by_key: dict[tuple[int, str, int], Run] = {}
    slots_by_key: dict[tuple[int, str, int], list[TensorSlot]] = collections.defaultdict(list)
    counters: collections.Counter[str] = collections.Counter()
    for event in events:
        name = str(event.get("event", ""))
        counters[name] += 1
        if name == "fault26_tensor_slot":
            slots_by_key[run_key(event)].append(
                TensorSlot(
                    sequence=int(event.get("sequence", 0)),
                    formal_ordinal=int(event.get("formal_ordinal", -1)),
                    slot_candidate=int(event.get("slot_candidate", -1)),
                    slot_proven=bool(event.get("slot_proven", False)),
                    address=parse_pointer(event.get("run_address")),
                    size=int(event.get("storage_bytes", 0)),
                    workspace_overlap=bool(event.get("workspace_overlap", False)),
                    workspace_gap=int(event.get("workspace_gap_bytes", 0)),
                )
            )
        elif name == "fault26_run_exit":
            key = run_key(event)
            # In inventory mode tensor registration is disabled and run_index
            # intentionally stays zero.  Keep every dynamic execution instead
            # of collapsing all calls for the same cache into one record.
            if int(event.get("run_index", 0)) == 0:
                key = (
                    int(event.get("pid", -1)),
                    str(event.get("cache", "")),
                    -int(event.get("sequence", 0)),
                )
            run_by_key[key] = Run(
                sequence=int(event.get("sequence", 0)),
                pid=int(event.get("pid", -1)),
                tid=int(event.get("tid", -1)),
                device=int(event.get("device", -1)),
                api=str(event.get("api_info", "")),
                execute_api=str(event.get("execute_api", "")),
                cache=str(event.get("cache", "")),
                run_index=int(event.get("run_index", 0)),
                stream=str(event.get("stream", "")),
                workspace=parse_pointer(event.get("workspace")),
                workspace_size=int(event.get("workspace_size", 0)),
                address_count=int(event.get("address_count", 0)),
                slot_proven=bool(event.get("slot_proven", False)),
                status=int(event.get("status", -1)),
            )
    runs = list(run_by_key.values())
    for key, run in run_by_key.items():
        run.tensors = sorted(
            slots_by_key.get(key, []), key=lambda tensor: tensor.formal_ordinal
        )
    runs.sort(key=lambda run: run.sequence)
    return runs, dict(counters)


def api_kind(run: Run) -> str:
    value = f"{run.api} {run.execute_api}".lower()
    if "fusedinferattention" in value or "inferattention" in value:
        return "fia"
    if "reducesum" in value:
        return "reduce_sum"
    if "add" in value or "rmsnorm" in value or "residual" in value:
        return "residual_consumer_candidate"
    return "other"


def ranges_overlap(first: TensorSlot, second: TensorSlot) -> bool:
    if first.address == 0 or second.address == 0 or first.size == 0 or second.size == 0:
        return False
    return first.address < second.address + second.size and second.address < first.address + first.size


def relation(first: Run, second: Run) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for left in first.tensors:
        for right in second.tensors:
            if not ranges_overlap(left, right):
                continue
            results.append(
                {
                    "from_api": first.api or first.execute_api,
                    "from_sequence": first.sequence,
                    "from_formal": left.formal_ordinal,
                    "from_slot": left.slot_candidate,
                    "to_api": second.api or second.execute_api,
                    "to_sequence": second.sequence,
                    "to_formal": right.formal_ordinal,
                    "to_slot": right.slot_candidate,
                    "exact_address": left.address == right.address,
                    "overlap": True,
                    "address": hex(max(left.address, right.address)),
                }
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument(
        "--window",
        type=int,
        default=128,
        help="maximum later target runs compared after a FIA run",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    events = list(load_events(args.trace))
    runs, event_counts = build_runs(events)
    dynamic_api_counts = collections.Counter(
        str(event.get("api_info", ""))
        for event in events
        if event.get("event") == "fault26_run_exit"
    )
    normalized_dynamic_api_counts: collections.Counter[str] = collections.Counter()
    for api, count in dynamic_api_counts.items():
        normalized_dynamic_api_counts[re.sub(r"_\d+$", "", api)] += count
    api_counts = collections.Counter((run.api or run.execute_api) for run in runs)
    kind_counts = collections.Counter(api_kind(run) for run in runs)
    slot_proof = collections.Counter(run.slot_proven for run in runs)
    workspace_known = sum(1 for run in runs if run.workspace and run.workspace_size)
    workspace_overlaps = [
        {
            "api": run.api or run.execute_api,
            "sequence": run.sequence,
            "formal": tensor.formal_ordinal,
            "slot": tensor.slot_candidate,
        }
        for run in runs
        for tensor in run.tensors
        if tensor.workspace_overlap
    ]

    cross_relations: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        if api_kind(run) != "fia":
            continue
        later = runs[index + 1 : index + 1 + max(args.window, 0)]
        for candidate in later:
            if candidate.pid != run.pid or candidate.device != run.device:
                continue
            if api_kind(candidate) not in {
                "reduce_sum",
                "residual_consumer_candidate",
            }:
                continue
            cross_relations.extend(relation(run, candidate))

    reduce_runs = [run for run in runs if api_kind(run) == "reduce_sum"]
    consumer_runs = [
        run for run in runs if api_kind(run) == "residual_consumer_candidate"
    ]
    reduce_to_consumer: list[dict[str, Any]] = []
    for reduce_run in reduce_runs:
        for consumer in consumer_runs:
            if consumer.sequence <= reduce_run.sequence:
                continue
            if consumer.sequence - reduce_run.sequence > args.window:
                break
            if consumer.pid == reduce_run.pid and consumer.device == reduce_run.device:
                reduce_to_consumer.extend(relation(reduce_run, consumer))

    summary = {
        "trace": str(args.trace),
        "event_counts": event_counts,
        "run_count": len(runs),
        "api_counts": dict(api_counts),
        "dynamic_run_exit_api_counts": dict(dynamic_api_counts),
        "normalized_dynamic_run_exit_api_counts": dict(normalized_dynamic_api_counts),
        "kind_counts": dict(kind_counts),
        "slot_proven_runs": slot_proof.get(True, 0),
        "slot_unproven_runs": slot_proof.get(False, 0),
        "workspace_size_known_runs": workspace_known,
        "workspace_tensor_overlaps": workspace_overlaps,
        "fia_to_victim_address_relations": cross_relations,
        "reduce_to_consumer_address_relations": reduce_to_consumer,
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    print(f"runs: {summary['run_count']}")
    print(f"slot proven/unproven: {summary['slot_proven_runs']}/{summary['slot_unproven_runs']}")
    print(f"workspace size known: {workspace_known}")
    print("normalized dynamic APIs:")
    for api, count in normalized_dynamic_api_counts.most_common():
        print(f"  {count:8d}  {api or '<unknown>'}")
    print(f"workspace/tensor overlaps: {len(workspace_overlaps)}")
    print(f"FIA -> victim address relations: {len(cross_relations)}")
    print(f"ReduceSum -> consumer address relations: {len(reduce_to_consumer)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
