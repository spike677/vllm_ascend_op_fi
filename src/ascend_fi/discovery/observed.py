"""Normalized, read-only facts emitted by Ascend execution observers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ascend_fi.contracts import ExecutionPlane


@dataclass(frozen=True)
class ObservedFact:
    """One runtime observation before capability selection.

    Addresses and call indices are evidence only. They are deliberately kept
    outside the stable product target contract.
    """

    execution_plane: ExecutionPlane
    symbol: str
    rank: int | None = None
    phase: str | None = None
    call_index: int | None = None
    call_fingerprint: str | None = None
    stream: str | int | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    runtime_slots: dict[str, int] = field(default_factory=dict)
    tensor_roles: dict[str, dict[str, Any]] = field(default_factory=dict)
    proof: dict[str, Any] = field(default_factory=dict)
    address_metadata: dict[str, Any] = field(default_factory=dict)
    producer_metadata: dict[str, Any] = field(default_factory=dict)
    consumer_metadata: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()
