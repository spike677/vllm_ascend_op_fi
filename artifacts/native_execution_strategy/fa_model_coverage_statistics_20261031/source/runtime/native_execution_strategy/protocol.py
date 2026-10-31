"""Adapter boundary; implementations must call the genuine native operation.

No generic code here emulates CANN, mutates serialized tiling, guesses an ABI,
or turns requested policy labels into observed execution evidence.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

from .models import InvocationContext, NativeStrategy, StrategyObservation
from .policies import PolicyDecision


@runtime_checkable
class ExecutionStrategyAdapter(Protocol):
    """An adapter verifies its own source, ABI, legal scope and evidence.

The harness owns exactly one original API invocation inside native_scope.
    observe_current must look up independently bound native evidence, not run an
    extra operation or synthesize a default tiling from shape.
legal_alternatives enumerates only supported candidates for this real context;
validate_candidate checks the full native guard/certificate, not only IDs.
"""
    name: str
    version: str
    on_unsupported: str

    def inspect(self, actual_call: Mapping[str, Any]) -> InvocationContext: ...

    def unsupported_reason(self, context: InvocationContext, *, controlled: bool) -> str | None: ...

    def observe_current(self, context: InvocationContext) -> StrategyObservation: ...

    def legal_alternatives(self, context: InvocationContext,
                           current: NativeStrategy) -> Iterable[NativeStrategy]: ...

    def validate_candidate(self, context: InvocationContext, candidate: NativeStrategy) -> bool: ...

    def native_scope(self, decision: PolicyDecision | None, invocation_id: str,
                     evidence_dir: Path, *, context: InvocationContext,
                     current: NativeStrategy | None = None) -> AbstractContextManager[Any]: ...

    def parse_native_evidence(self, context: InvocationContext,
                              record: Mapping[str, Any]) -> StrategyObservation: ...
