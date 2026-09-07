"""Developer workflow composing discovery, validation, evaluation, and profile output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ascend_fi.contracts import Candidate, Capability, EvaluationResult, ValidationResult
from ascend_fi.discovery import ObservedFact, RankedCandidate, resolve_and_rank
from ascend_fi.profiles import ModelCapabilityProfile, generate_model_profile
from ascend_fi.validation import LosslessExecutor, validate_target

from .runner import EvaluationExecutor, EvaluationRunner


class WorkflowRejected(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkflowResult:
    ranked_candidate: RankedCandidate
    validation: ValidationResult
    evaluation: EvaluationResult
    profile: ModelCapabilityProfile

    @property
    def candidate(self) -> Candidate:
        return self.ranked_candidate.candidate


def run_discovery_evaluation(
    *,
    fact: ObservedFact,
    capabilities: Iterable[Capability],
    lossless_executor: LosslessExecutor,
    evaluation_executor: EvaluationExecutor,
    evaluation_id: str,
    repository_root: str | Path | None = None,
) -> WorkflowResult:
    capability_list = tuple(capabilities)
    ranked = resolve_and_rank(fact, capability_list)
    eligible = next((item for item in ranked if item.auto_validation_allowed), None)
    if eligible is None:
        raise WorkflowRejected("no candidate is eligible for automatic lossless validation")
    by_id = {capability.id: capability for capability in capability_list}
    capability = by_id[eligible.candidate.capability_id]
    validation = validate_target(eligible.candidate, lossless_executor)
    if not validation.destructive_injection_allowed:
        raise WorkflowRejected(validation.reason or "lossless validation failed")
    evaluation = EvaluationRunner().evaluate(
        evaluation_id,
        eligible.candidate,
        validation,
        evaluation_executor,
    )
    profile = generate_model_profile(
        capability,
        eligible.candidate,
        validation,
        evaluation,
        repository_root=repository_root,
    )
    return WorkflowResult(eligible, validation, evaluation, profile)
