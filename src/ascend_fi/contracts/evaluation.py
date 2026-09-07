"""Unified baseline/fault/recovery evaluation result contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from ._base import JsonContract, as_mapping, enum_value, extra_fields, require


class EvaluationStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"


@dataclass(frozen=True)
class RunRecord(JsonContract):
    http_status: int | None = None
    response_hash: str | None = None
    token_ids: tuple[int, ...] | None = None
    generation_length: int | None = None
    worker_alive: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict, compare=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunRecord":
        known = {"http_status", "response_hash", "token_ids", "generation_length", "worker_alive"}
        tokens = data.get("token_ids")
        if tokens is not None and (
            not isinstance(tokens, list) or not all(isinstance(item, int) for item in tokens)
        ):
            raise ValueError("token_ids must be an integer list or null")
        return cls(
            http_status=data.get("http_status"),
            response_hash=data.get("response_hash"),
            token_ids=None if tokens is None else tuple(tokens),
            generation_length=data.get("generation_length"),
            worker_alive=data.get("worker_alive"),
            extra=extra_fields(data, known),
        )


@dataclass(frozen=True)
class MechanismMetrics(JsonContract):
    resolved: bool | None = None
    candidate_count: int | None = None
    hit_count: int | None = None
    reject_count: int | None = None
    reject_reasons: dict[str, int] = field(default_factory=dict)
    changed_elements: int | None = None
    restore_exact: bool | None = None
    auto_disarmed: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict, compare=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MechanismMetrics":
        known = {
            "resolved",
            "candidate_count",
            "hit_count",
            "reject_count",
            "reject_reasons",
            "changed_elements",
            "restore_exact",
            "auto_disarmed",
        }
        reasons = data.get("reject_reasons", {})
        if not isinstance(reasons, Mapping):
            raise ValueError("mechanism.reject_reasons must be an object")
        return cls(
            resolved=data.get("resolved"),
            candidate_count=data.get("candidate_count"),
            hit_count=data.get("hit_count"),
            reject_count=data.get("reject_count"),
            reject_reasons={str(key): int(value) for key, value in reasons.items()},
            changed_elements=data.get("changed_elements"),
            restore_exact=data.get("restore_exact"),
            auto_disarmed=data.get("auto_disarmed"),
            extra=extra_fields(data, known),
        )


@dataclass(frozen=True)
class ModelEffectMetrics(JsonContract):
    http_status: int | None = None
    response_equal: bool | None = None
    response_hash: str | None = None
    token_equal: bool | None = None
    first_divergent_token: int | None = None
    generation_length: int | None = None
    generation_length_delta: int | None = None
    repetition_rate: float | None = None
    spam_rate: float | None = None
    early_stop: bool | None = None
    crash: bool | None = None
    worker_alive: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict, compare=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelEffectMetrics":
        known = {item.name for item in __import__("dataclasses").fields(cls) if item.name != "extra"}
        return cls(
            **{key: data.get(key) for key in known},
            extra=extra_fields(data, known),
        )


@dataclass(frozen=True)
class ReliabilityMetrics(JsonContract):
    target_resolve_rate: float | None = None
    injection_hit_rate: float | None = None
    effect_repeatability: float | None = None
    recovery_rate: float | None = None
    extra: dict[str, Any] = field(default_factory=dict, compare=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReliabilityMetrics":
        known = {
            "target_resolve_rate",
            "injection_hit_rate",
            "effect_repeatability",
            "recovery_rate",
        }
        return cls(
            target_resolve_rate=data.get("target_resolve_rate"),
            injection_hit_rate=data.get("injection_hit_rate"),
            effect_repeatability=data.get("effect_repeatability"),
            recovery_rate=data.get("recovery_rate"),
            extra=extra_fields(data, known),
        )


@dataclass(frozen=True)
class QualityPerformanceMetrics(JsonContract):
    mean_logprob_delta: float | None = None
    max_logprob_delta: float | None = None
    nll_delta: float | None = None
    kl_divergence: float | None = None
    ppl: float | None = None
    ppl_ratio: float | None = None
    accuracy: float | None = None
    accuracy_retention: float | None = None
    ttft: float | None = None
    tpot: float | None = None
    throughput: float | None = None
    extra: dict[str, Any] = field(default_factory=dict, compare=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QualityPerformanceMetrics":
        known = {item.name for item in __import__("dataclasses").fields(cls) if item.name != "extra"}
        return cls(
            **{key: data.get(key) for key in known},
            extra=extra_fields(data, known),
        )


@dataclass(frozen=True)
class EvaluationResult(JsonContract):
    evaluation_id: str
    capability_id: str
    candidate_id: str | None
    baseline: RunRecord | None
    fault: RunRecord | None
    recovery: RunRecord | None
    mechanism: MechanismMetrics
    model_effect: ModelEffectMetrics
    reliability: ReliabilityMetrics
    quality_performance: QualityPerformanceMetrics
    status: EvaluationStatus = EvaluationStatus.UNKNOWN
    reason: str | None = None
    evidence_refs: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict, compare=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationResult":
        known = {
            "evaluation_id",
            "capability_id",
            "candidate_id",
            "baseline",
            "fault",
            "recovery",
            "mechanism",
            "model_effect",
            "reliability",
            "quality_performance",
            "status",
            "reason",
            "evidence_refs",
        }

        def optional_run(key: str) -> RunRecord | None:
            value = data.get(key)
            if value is None:
                return None
            return RunRecord.from_dict(as_mapping(value, key))

        refs = data.get("evidence_refs", [])
        if not isinstance(refs, list) or not all(isinstance(item, str) for item in refs):
            raise ValueError("evidence_refs must be a string list")
        return cls(
            evaluation_id=str(require(data, "evaluation_id")),
            capability_id=str(require(data, "capability_id")),
            candidate_id=data.get("candidate_id"),
            baseline=optional_run("baseline"),
            fault=optional_run("fault"),
            recovery=optional_run("recovery"),
            mechanism=MechanismMetrics.from_dict(
                as_mapping(require(data, "mechanism"), "mechanism")
            ),
            model_effect=ModelEffectMetrics.from_dict(
                as_mapping(require(data, "model_effect"), "model_effect")
            ),
            reliability=ReliabilityMetrics.from_dict(
                as_mapping(require(data, "reliability"), "reliability")
            ),
            quality_performance=QualityPerformanceMetrics.from_dict(
                as_mapping(require(data, "quality_performance"), "quality_performance")
            ),
            status=enum_value(
                EvaluationStatus,
                data.get("status", EvaluationStatus.UNKNOWN.value),
                "status",
            ),
            reason=data.get("reason"),
            evidence_refs=tuple(refs),
            extra=extra_fields(data, known),
        )
