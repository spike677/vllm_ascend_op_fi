"""Discovery candidate contract shared by evaluation and fault research."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from ._base import JsonContract, as_mapping, enum_value, extra_fields, require
from .capability import ExecutionPlane


class ProofLevel(str, Enum):
    OBSERVED = "OBSERVED"
    STRUCTURAL_MATCH = "STRUCTURAL_MATCH"
    SLOT_PROVEN = "SLOT_PROVEN"
    SEMANTIC_SLOT_PROVEN = "SEMANTIC_SLOT_PROVEN"
    LOSSLESS_VALIDATED = "LOSSLESS_VALIDATED"
    E2E_VALIDATED = "E2E_VALIDATED"


class CandidateStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class ObservedTarget(JsonContract):
    execution_plane: ExecutionPlane | None = None
    symbol: str | None = None
    rank: int | None = None
    phase: str | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    stream: str | int | None = None
    call_fingerprint: str | None = None
    extra: dict[str, Any] = field(default_factory=dict, compare=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObservedTarget":
        known = {
            "execution_plane",
            "symbol",
            "rank",
            "phase",
            "shape",
            "dtype",
            "stream",
            "call_fingerprint",
        }
        raw_plane = data.get("execution_plane")
        raw_shape = data.get("shape")
        if raw_shape is not None and (
            not isinstance(raw_shape, list) or not all(isinstance(item, int) for item in raw_shape)
        ):
            raise ValueError("observed.shape must be an integer list or null")
        return cls(
            execution_plane=None
            if raw_plane is None
            else enum_value(ExecutionPlane, raw_plane, "observed.execution_plane"),
            symbol=data.get("symbol"),
            rank=data.get("rank"),
            phase=data.get("phase"),
            shape=None if raw_shape is None else tuple(raw_shape),
            dtype=data.get("dtype"),
            stream=data.get("stream"),
            call_fingerprint=data.get("call_fingerprint"),
            extra=extra_fields(data, known),
        )


@dataclass(frozen=True)
class CandidateBinding(JsonContract):
    tensor_role: str | None = None
    runtime_slot: int | None = None
    storage_role: str | None = None
    extra: dict[str, Any] = field(default_factory=dict, compare=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateBinding":
        known = {"tensor_role", "runtime_slot", "storage_role"}
        return cls(
            tensor_role=data.get("tensor_role"),
            runtime_slot=data.get("runtime_slot"),
            storage_role=data.get("storage_role"),
            extra=extra_fields(data, known),
        )


@dataclass(frozen=True)
class CandidateProof(JsonContract):
    level: ProofLevel
    evidence_refs: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict, compare=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateProof":
        known = {"level", "evidence_refs"}
        refs = data.get("evidence_refs", [])
        if not isinstance(refs, list) or not all(isinstance(item, str) for item in refs):
            raise ValueError("proof.evidence_refs must be a string list")
        return cls(
            level=enum_value(ProofLevel, require(data, "level"), "proof.level"),
            evidence_refs=tuple(refs),
            extra=extra_fields(data, known),
        )


@dataclass(frozen=True)
class CandidateValidation(JsonContract):
    status: CandidateStatus = CandidateStatus.UNKNOWN
    reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict, compare=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateValidation":
        known = {"status", "reason"}
        return cls(
            status=enum_value(
                CandidateStatus, data.get("status", CandidateStatus.UNKNOWN.value), "validation.status"
            ),
            reason=data.get("reason"),
            extra=extra_fields(data, known),
        )


@dataclass(frozen=True)
class CandidateRecommendation(JsonContract):
    strategy: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict, compare=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateRecommendation":
        known = {"strategy", "params"}
        params = dict(as_mapping(data.get("params", {}), "recommended.params"))
        return cls(
            strategy=data.get("strategy"),
            params=params,
            extra=extra_fields(data, known),
        )


@dataclass(frozen=True)
class Candidate(JsonContract):
    candidate_id: str
    capability_id: str
    observed: ObservedTarget
    binding: CandidateBinding
    proof: CandidateProof
    validation: CandidateValidation = field(default_factory=CandidateValidation)
    recommended: CandidateRecommendation = field(default_factory=CandidateRecommendation)
    extra: dict[str, Any] = field(default_factory=dict, compare=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Candidate":
        known = {
            "candidate_id",
            "capability_id",
            "observed",
            "binding",
            "proof",
            "validation",
            "recommended",
        }
        return cls(
            candidate_id=str(require(data, "candidate_id")),
            capability_id=str(require(data, "capability_id")),
            observed=ObservedTarget.from_dict(
                as_mapping(require(data, "observed"), "observed")
            ),
            binding=CandidateBinding.from_dict(
                as_mapping(require(data, "binding"), "binding")
            ),
            proof=CandidateProof.from_dict(as_mapping(require(data, "proof"), "proof")),
            validation=CandidateValidation.from_dict(
                as_mapping(data.get("validation", {}), "validation")
            ),
            recommended=CandidateRecommendation.from_dict(
                as_mapping(data.get("recommended", {}), "recommended")
            ),
            extra=extra_fields(data, known),
        )
