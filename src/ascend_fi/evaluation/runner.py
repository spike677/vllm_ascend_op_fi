"""Unified baseline/fault/recovery evaluation runner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ascend_fi.contracts import (
    Candidate,
    EvaluationResult,
    EvaluationStatus,
    MechanismMetrics,
    ModelEffectMetrics,
    QualityPerformanceMetrics,
    ReliabilityMetrics,
    RunRecord,
    ValidationResult,
)

from .signals import first_difference, obvious_gibberish, punctuation_spam_rate, repetition_rate


@dataclass(frozen=True)
class RunOutcome:
    http_status: int | None
    response_text: str | None
    token_ids: tuple[int, ...] | None = None
    token_pieces: tuple[str, ...] | None = None
    generation_length: int | None = None
    worker_alive: bool | None = None
    finish_reason: str | None = None
    error: str | None = None
    mechanism: dict[str, Any] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()

    @property
    def response_hash(self) -> str | None:
        if self.response_text is None:
            return None
        return hashlib.sha256(self.response_text.encode("utf-8")).hexdigest()

    def as_record(self) -> RunRecord:
        return RunRecord(
            http_status=self.http_status,
            response_hash=self.response_hash,
            token_ids=self.token_ids,
            generation_length=self.generation_length,
            worker_alive=self.worker_alive,
            extra={
                "response_text": self.response_text,
                "token_pieces": None if self.token_pieces is None else list(self.token_pieces),
                "finish_reason": self.finish_reason,
                "error": self.error,
                "evidence_refs": list(self.evidence_refs),
            },
        )


class EvaluationExecutor(Protocol):
    """Developer/backend adapter, not a product controller contract."""

    def run_baseline(self, candidate: Candidate) -> RunOutcome: ...
    def run_fault(self, candidate: Candidate) -> RunOutcome: ...
    def disarm(self, candidate: Candidate) -> bool: ...
    def run_recovery(self, candidate: Candidate) -> RunOutcome: ...


class RecordedEvaluationExecutor:
    """Replay a deterministic saved fixture while preserving phase order."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("recorded evaluation fixture must be an object")
        self.payload = payload
        self.calls: list[str] = []

    def _outcome(self, phase: str) -> RunOutcome:
        raw = self.payload.get(phase)
        if not isinstance(raw, dict):
            raise ValueError(f"fixture phase {phase} must be an object")
        text = raw.get("response_text")
        expected_hash = raw.get("response_hash")
        if text is not None and not isinstance(text, str):
            raise ValueError(f"{phase}.response_text must be a string or null")
        if text is not None and expected_hash is not None:
            actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if actual != expected_hash:
                raise ValueError(f"{phase}.response_hash does not match response_text")
        ids = raw.get("token_ids")
        pieces = raw.get("token_pieces")
        if ids is not None and (not isinstance(ids, list) or not all(isinstance(item, int) for item in ids)):
            raise ValueError(f"{phase}.token_ids must be an integer list or null")
        if pieces is not None and (not isinstance(pieces, list) or not all(isinstance(item, str) for item in pieces)):
            raise ValueError(f"{phase}.token_pieces must be a string list or null")
        refs = raw.get("evidence_refs", self.payload.get("evidence_refs", []))
        return RunOutcome(
            http_status=raw.get("http_status"),
            response_text=text,
            token_ids=None if ids is None else tuple(ids),
            token_pieces=None if pieces is None else tuple(pieces),
            generation_length=raw.get("generation_length"),
            worker_alive=raw.get("worker_alive"),
            finish_reason=raw.get("finish_reason"),
            error=raw.get("error"),
            mechanism=dict(raw.get("mechanism", {})),
            evidence_refs=tuple(refs),
        )

    def run_baseline(self, candidate: Candidate) -> RunOutcome:
        self.calls.append("baseline")
        return self._outcome("baseline")

    def run_fault(self, candidate: Candidate) -> RunOutcome:
        self.calls.append("fault")
        return self._outcome("fault")

    def disarm(self, candidate: Candidate) -> bool:
        self.calls.append("disarm")
        value = self.payload.get("disarm_success")
        if not isinstance(value, bool):
            raise ValueError("fixture disarm_success must be a boolean")
        return value

    def run_recovery(self, candidate: Candidate) -> RunOutcome:
        self.calls.append("recovery")
        return self._outcome("recovery")


class EvaluationRunner:
    def evaluate(
        self,
        evaluation_id: str,
        candidate: Candidate,
        validation: ValidationResult,
        executor: EvaluationExecutor,
    ) -> EvaluationResult:
        if not validation.destructive_injection_allowed:
            return _not_authorized(evaluation_id, candidate, validation)
        try:
            baseline = executor.run_baseline(candidate)
            try:
                fault = executor.run_fault(candidate)
            finally:
                # Once fault execution starts, always request disarm even if
                # the backend raises before returning an outcome.
                disarm_success = executor.disarm(candidate)
            recovery = executor.run_recovery(candidate)
        except Exception as error:
            return _error_result(evaluation_id, candidate, f"evaluation executor failed: {error}")
        return _build_result(
            evaluation_id,
            candidate,
            baseline,
            fault,
            recovery,
            disarm_success,
        )


def _comparison_sequence(outcome: RunOutcome) -> tuple[tuple[object, ...] | None, str | None]:
    if outcome.token_ids is not None:
        return tuple(outcome.token_ids), "token_ids"
    if outcome.token_pieces is not None:
        return tuple(outcome.token_pieces), "token_pieces"
    return None, None


def _build_result(
    evaluation_id: str,
    candidate: Candidate,
    baseline: RunOutcome,
    fault: RunOutcome,
    recovery: RunOutcome,
    disarm_success: bool,
) -> EvaluationResult:
    baseline_tokens, baseline_basis = _comparison_sequence(baseline)
    fault_tokens, fault_basis = _comparison_sequence(fault)
    basis = baseline_basis if baseline_basis == fault_basis else None
    token_equal = None if basis is None else baseline_tokens == fault_tokens
    first_divergent = first_difference(baseline_tokens, fault_tokens) if basis else None
    response_equal = (
        None
        if baseline.response_text is None or fault.response_text is None
        else baseline.response_text == fault.response_text
    )
    recovery_equal = (
        baseline.response_text is not None
        and recovery.response_text is not None
        and baseline.response_text == recovery.response_text
        and (
            baseline_tokens == _comparison_sequence(recovery)[0]
            if baseline_tokens is not None
            else True
        )
    )
    mechanism_raw = fault.mechanism
    hit_count = mechanism_raw.get("hit_count")
    mechanism_ok = (
        mechanism_raw.get("resolved") is True
        and isinstance(hit_count, int)
        and hit_count > 0
    )
    health_ok = all(
        outcome.http_status == 200 and outcome.worker_alive is True
        for outcome in (baseline, fault, recovery)
    )
    passed = mechanism_ok and health_ok and disarm_success and recovery_equal
    evidence_refs = tuple(
        dict.fromkeys(
            candidate.proof.evidence_refs
            + baseline.evidence_refs
            + fault.evidence_refs
            + recovery.evidence_refs
        )
    )
    sequence_for_signals = fault_tokens
    mechanism = MechanismMetrics(
        resolved=mechanism_raw.get("resolved"),
        candidate_count=mechanism_raw.get("candidate_count"),
        hit_count=hit_count,
        reject_count=mechanism_raw.get("reject_count"),
        reject_reasons=dict(mechanism_raw.get("reject_reasons", {})),
        changed_elements=mechanism_raw.get("changed_elements"),
        restore_exact=mechanism_raw.get("restore_exact"),
        auto_disarmed=mechanism_raw.get("auto_disarmed"),
        extra={"disarm_success": disarm_success},
    )
    model_effect = ModelEffectMetrics(
        http_status=fault.http_status,
        response_equal=response_equal,
        response_hash=fault.response_hash,
        token_equal=token_equal,
        first_divergent_token=first_divergent,
        generation_length=fault.generation_length,
        generation_length_delta=(
            None
            if baseline.generation_length is None or fault.generation_length is None
            else fault.generation_length - baseline.generation_length
        ),
        repetition_rate=repetition_rate(sequence_for_signals),
        spam_rate=punctuation_spam_rate(fault.response_text),
        early_stop=(
            None
            if baseline.generation_length is None or fault.generation_length is None
            else fault.generation_length < baseline.generation_length
        ),
        crash=(fault.http_status is not None and fault.http_status >= 500) or fault.worker_alive is False,
        worker_alive=fault.worker_alive,
        extra={
            "token_comparison_basis": basis,
            "empty_output": fault.response_text is not None and not fault.response_text.strip(),
            "obvious_gibberish": obvious_gibberish(fault.response_text),
            "recovery_response_equal": recovery_equal,
        },
    )
    return EvaluationResult(
        evaluation_id=evaluation_id,
        capability_id=candidate.capability_id,
        candidate_id=candidate.candidate_id,
        baseline=baseline.as_record(),
        fault=fault.as_record(),
        recovery=recovery.as_record(),
        mechanism=mechanism,
        model_effect=model_effect,
        reliability=ReliabilityMetrics(
            target_resolve_rate=1.0 if mechanism_raw.get("resolved") is True else 0.0,
            injection_hit_rate=(1.0 if mechanism_ok else 0.0),
            effect_repeatability=None,
            recovery_rate=1.0 if recovery_equal else 0.0,
        ),
        quality_performance=QualityPerformanceMetrics(),
        status=EvaluationStatus.PASS if passed else EvaluationStatus.FAIL,
        reason=(
            "mechanism hit, service remained healthy, and recovery matched baseline"
            if passed
            else "mechanism, health, disarm, or recovery requirement failed"
        ),
        evidence_refs=evidence_refs,
        extra={"disarm_success": disarm_success, "recovery_exact": recovery_equal},
    )


def _not_authorized(
    evaluation_id: str, candidate: Candidate, validation: ValidationResult
) -> EvaluationResult:
    return _empty_result(
        evaluation_id,
        candidate,
        EvaluationStatus.FAIL,
        f"destructive evaluation not allowed: {validation.reason or validation.status.value}",
    )


def _error_result(evaluation_id: str, candidate: Candidate, reason: str) -> EvaluationResult:
    return _empty_result(evaluation_id, candidate, EvaluationStatus.ERROR, reason)


def _empty_result(
    evaluation_id: str,
    candidate: Candidate,
    status: EvaluationStatus,
    reason: str,
) -> EvaluationResult:
    return EvaluationResult(
        evaluation_id=evaluation_id,
        capability_id=candidate.capability_id,
        candidate_id=candidate.candidate_id,
        baseline=None,
        fault=None,
        recovery=None,
        mechanism=MechanismMetrics(resolved=False),
        model_effect=ModelEffectMetrics(),
        reliability=ReliabilityMetrics(),
        quality_performance=QualityPerformanceMetrics(),
        status=status,
        reason=reason,
    )
