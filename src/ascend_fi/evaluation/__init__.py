"""Baseline/fault/recovery evaluation without product control-plane ownership."""

from .runner import (
    EvaluationExecutor,
    EvaluationRunner,
    RecordedEvaluationExecutor,
    RunOutcome,
)
from .signals import first_difference, obvious_gibberish, punctuation_spam_rate, repetition_rate
from .workflow import WorkflowRejected, WorkflowResult, run_discovery_evaluation

__all__ = [
    "EvaluationExecutor",
    "EvaluationRunner",
    "RecordedEvaluationExecutor",
    "RunOutcome",
    "WorkflowRejected",
    "WorkflowResult",
    "first_difference",
    "obvious_gibberish",
    "punctuation_spam_rate",
    "repetition_rate",
    "run_discovery_evaluation",
]
