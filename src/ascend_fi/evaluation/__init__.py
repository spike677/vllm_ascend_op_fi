"""Baseline/fault/recovery evaluation without product control-plane ownership."""

from .runner import (
    EvaluationExecutor,
    EvaluationRunner,
    RecordedEvaluationExecutor,
    RunOutcome,
)
from .signals import first_difference, obvious_gibberish, punctuation_spam_rate, repetition_rate

__all__ = [
    "EvaluationExecutor",
    "EvaluationRunner",
    "RecordedEvaluationExecutor",
    "RunOutcome",
    "first_difference",
    "obvious_gibberish",
    "punctuation_spam_rate",
    "repetition_rate",
]
