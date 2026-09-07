"""Lossless target validation result contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from ._base import JsonContract, enum_value, extra_fields, require


class ValidationStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    PENDING = "PENDING"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    LOSSLESS_VALIDATED = "LOSSLESS_VALIDATED"


@dataclass(frozen=True)
class ValidationResult(JsonContract):
    candidate_id: str
    resolved: bool
    lossless_attempted: bool
    lossless_passed: bool
    hit_count: int = 0
    reject_count: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)
    before_hash: str | None = None
    substituted_hash: str | None = None
    restored_hash: str | None = None
    model_output_equal: bool | None = None
    restore_exact: bool | None = None
    status: ValidationStatus = ValidationStatus.UNKNOWN
    reason: str | None = None
    evidence_refs: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict, compare=True)

    @property
    def destructive_injection_allowed(self) -> bool:
        return (
            self.resolved
            and self.lossless_attempted
            and self.lossless_passed
            and self.restore_exact is True
            and self.status is ValidationStatus.LOSSLESS_VALIDATED
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationResult":
        known = {
            "candidate_id",
            "resolved",
            "lossless_attempted",
            "lossless_passed",
            "hit_count",
            "reject_count",
            "reject_reasons",
            "before_hash",
            "substituted_hash",
            "restored_hash",
            "model_output_equal",
            "restore_exact",
            "status",
            "reason",
            "evidence_refs",
        }
        refs = data.get("evidence_refs", [])
        reasons = data.get("reject_reasons", {})
        if not isinstance(refs, list) or not all(isinstance(item, str) for item in refs):
            raise ValueError("evidence_refs must be a string list")
        if not isinstance(reasons, Mapping):
            raise ValueError("reject_reasons must be an object")
        return cls(
            candidate_id=str(require(data, "candidate_id")),
            resolved=bool(require(data, "resolved")),
            lossless_attempted=bool(require(data, "lossless_attempted")),
            lossless_passed=bool(require(data, "lossless_passed")),
            hit_count=int(data.get("hit_count", 0)),
            reject_count=int(data.get("reject_count", 0)),
            reject_reasons={str(key): int(value) for key, value in reasons.items()},
            before_hash=data.get("before_hash"),
            substituted_hash=data.get("substituted_hash"),
            restored_hash=data.get("restored_hash"),
            model_output_equal=data.get("model_output_equal"),
            restore_exact=data.get("restore_exact"),
            status=enum_value(
                ValidationStatus,
                data.get("status", ValidationStatus.UNKNOWN.value),
                "status",
            ),
            reason=data.get("reason"),
            evidence_refs=tuple(refs),
            extra=extra_fields(data, known),
        )
