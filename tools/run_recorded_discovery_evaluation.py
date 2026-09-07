"""Run the V0 pipeline against committed fixtures; never contacts or arms a service."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ascend_fi.discovery import TraceContext, adapt_qmm_jsonl, load_capabilities
from ascend_fi.evaluation import RecordedEvaluationExecutor, run_discovery_evaluation
from ascend_fi.validation import RecordedLosslessExecutor


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    context = TraceContext(
        phase="decode",
        rank=0,
        model="Qwen3-32B-w8a8-nopdmix",
        topology="tensor_parallel_2",
        runtime={"cann": "8.5.1", "vllm": "0.18.0"},
        evidence_ref="tests/fixtures/qmm_run_trace.json:2",
    )
    fact = adapt_qmm_jsonl(REPO / "tests/fixtures/qmm_run_trace.json", context)[0]
    result = run_discovery_evaluation(
        fact=fact,
        capabilities=load_capabilities(REPO / "profiles/capabilities"),
        lossless_executor=RecordedLosslessExecutor(
            REPO / "tests/fixtures/qmm_lossless_validation.json"
        ),
        evaluation_executor=RecordedEvaluationExecutor(
            REPO / "tests/fixtures/qmm_evaluation.json"
        ),
        evaluation_id="qmm-recorded-discovery-evaluation-v0",
        repository_root=REPO,
    )
    payload = {
        "candidate": result.candidate.to_dict(),
        "ranking": {
            "score": result.ranked_candidate.score,
            "confidence": result.ranked_candidate.confidence.value,
            "auto_validation_allowed": result.ranked_candidate.auto_validation_allowed,
            "reasons": list(result.ranked_candidate.reasons),
        },
        "validation": result.validation.to_dict(),
        "evaluation": result.evaluation.to_dict(),
        "profile": result.profile.to_dict(),
        "execution_mode": "recorded_fixture_only",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
