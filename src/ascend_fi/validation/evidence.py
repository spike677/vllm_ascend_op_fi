"""Evidence boundary for existing lossless target-validation backends."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ascend_fi.contracts import Candidate


@dataclass(frozen=True)
class LosslessEvidence:
    target_resolved: bool
    expected_api_slot_hit: bool
    substitution_executed: bool
    hit_count: int
    reject_count: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)
    before_hash: str | None = None
    substituted_hash: str | None = None
    restored_hash: str | None = None
    model_output_equal: bool | None = None
    original_storage_untouched: bool | None = None
    run_status: int | None = None
    copy_status: int | None = None
    sync_status: int | None = None
    temporary_freed: bool | None = None
    evidence_refs: tuple[str, ...] = ()


class LosslessExecutor(Protocol):
    """Developer/backend adapter; never a production controller interface."""

    def execute_lossless(self, candidate: Candidate) -> LosslessEvidence: ...


class RecordedLosslessExecutor:
    """Replay a frozen validation record without arming a live backend."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def execute_lossless(self, candidate: Candidate) -> LosslessEvidence:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("recorded lossless fixture must be an object")
        events = payload.get("runtime_events")
        comparison = payload.get("comparison", {})
        if not isinstance(events, list) or not isinstance(comparison, dict):
            raise ValueError("fixture requires runtime_events and comparison")
        matching = []
        for event in events:
            if not isinstance(event, dict) or event.get("event") != "lossless_substitution":
                continue
            shape = event.get("shape")
            if (
                candidate.observed.symbol in str(event.get("api_info", ""))
                and shape == list(candidate.observed.shape or ())
            ):
                matching.append(event)
        if len(matching) != 1:
            return LosslessEvidence(
                target_resolved=False,
                expected_api_slot_hit=False,
                substitution_executed=False,
                hit_count=0,
                reject_count=max(1, len(events) - len(matching)),
                reject_reasons={"expected_unique_lossless_event": 1},
                evidence_refs=(str(self.path),),
            )
        event = matching[0]
        original = event.get("original")
        temporary = event.get("temporary")
        refs = payload.get("evidence_refs", [])
        if not isinstance(refs, list) or not all(isinstance(item, str) for item in refs):
            raise ValueError("evidence_refs must be a string list")
        return LosslessEvidence(
            target_resolved=True,
            expected_api_slot_hit=candidate.binding.runtime_slot == 0,
            substitution_executed=temporary is not None and temporary != original,
            hit_count=int(event.get("hit", 0) > 0),
            reject_count=int(payload.get("reject_count", 0)),
            reject_reasons={
                str(key): int(value)
                for key, value in payload.get("reject_reasons", {}).items()
            },
            before_hash=comparison.get("before_hash"),
            substituted_hash=comparison.get("substituted_hash"),
            restored_hash=comparison.get("restored_hash"),
            model_output_equal=comparison.get("model_output_equal"),
            original_storage_untouched=True,
            run_status=event.get("run_status"),
            copy_status=event.get("copy_status"),
            sync_status=event.get("sync_status"),
            temporary_freed=event.get("freed"),
            evidence_refs=tuple(refs) + (str(self.path),),
        )
