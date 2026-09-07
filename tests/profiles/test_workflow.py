from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ascend_fi.contracts import EvaluationStatus, ValidationStatus
from ascend_fi.discovery import TraceContext, adapt_qmm_jsonl, load_capabilities
from ascend_fi.evaluation import RecordedEvaluationExecutor, run_discovery_evaluation
from ascend_fi.validation import RecordedLosslessExecutor


REPO = Path(__file__).resolve().parents[2]


class WorkflowTests(unittest.TestCase):
    def test_qmm_fixture_closes_v0_workflow_and_generates_profile(self) -> None:
        context = TraceContext(
            phase="decode",
            rank=0,
            model="Qwen3-32B-w8a8-nopdmix",
            topology="tensor_parallel_2",
            runtime={"cann": "8.5.1", "vllm": "0.18.0"},
            evidence_ref="tests/fixtures/qmm_run_trace.json:2",
        )
        fact = adapt_qmm_jsonl(
            REPO / "tests/fixtures/qmm_run_trace.json", context
        )[0]
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
        self.assertEqual(result.validation.status, ValidationStatus.LOSSLESS_VALIDATED)
        self.assertEqual(result.evaluation.status, EvaluationStatus.PASS)
        profile = result.profile.to_dict()
        self.assertEqual(
            profile["capability"]["id"],
            "op_api.quant_matmul.input_buffer_transform",
        )
        self.assertEqual(profile["candidate"]["binding"]["runtime_slot"], 0)
        self.assertEqual(
            profile["candidate"]["binding"]["stability"],
            "rediscover_per_environment",
        )
        self.assertNotIn("call_index", result.profile.to_json())
        self.assertNotIn("0x2000", result.profile.to_json())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "profile.json"
            result.profile.write_json(output)
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
