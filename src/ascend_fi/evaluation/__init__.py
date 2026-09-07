"""Baseline/fault/recovery evaluation without product control-plane ownership."""

from .runner import (
    EvaluationExecutor,
    EvaluationRunner,
    RecordedEvaluationExecutor,
    RunOutcome,
)
from .live import (
    EvaluationSample,
    LiveExecutorConfig,
    LiveFaultMode,
    LiveQwenFaultExecutor,
    LOSSLESS_QMM_RANK0,
    MILD_INT4_QMM_RANK0,
    OpenAICompletionClient,
    STRONG_INT2_QMM_RANK0,
)
from .metrics import (
    aggregate_per_rank,
    aggregate_repeatability,
    extract_gsm8k_answer,
    gsm8k_accuracy,
    safe_ratio,
    token_divergence,
    word_perplexity,
)
from .signals import first_difference, obvious_gibberish, punctuation_spam_rate, repetition_rate
from .suite import (
    load_gsm8k,
    load_smoke_pack,
    prepare_wikitext_windows,
    read_wikitext_raw,
    run_qwen_fault_effect_suite,
)
from .workflow import WorkflowRejected, WorkflowResult, run_discovery_evaluation

__all__ = [
    "EvaluationExecutor",
    "EvaluationSample",
    "EvaluationRunner",
    "LiveExecutorConfig",
    "LiveFaultMode",
    "LiveQwenFaultExecutor",
    "LOSSLESS_QMM_RANK0",
    "MILD_INT4_QMM_RANK0",
    "OpenAICompletionClient",
    "RecordedEvaluationExecutor",
    "RunOutcome",
    "STRONG_INT2_QMM_RANK0",
    "WorkflowRejected",
    "WorkflowResult",
    "first_difference",
    "aggregate_per_rank",
    "aggregate_repeatability",
    "extract_gsm8k_answer",
    "gsm8k_accuracy",
    "obvious_gibberish",
    "punctuation_spam_rate",
    "load_gsm8k",
    "load_smoke_pack",
    "prepare_wikitext_windows",
    "read_wikitext_raw",
    "repetition_rate",
    "safe_ratio",
    "token_divergence",
    "word_perplexity",
    "run_discovery_evaluation",
    "run_qwen_fault_effect_suite",
]
