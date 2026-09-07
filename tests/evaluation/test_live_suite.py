from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from ascend_fi.discovery import TraceContext, adapt_qmm_jsonl, load_capabilities, resolve_and_rank
from ascend_fi.evaluation import (
    EvaluationSample,
    LiveExecutorConfig,
    LiveQwenFaultExecutor,
    STRONG_INT2_QMM_RANK0,
    run_qwen_fault_effect_suite,
)
from ascend_fi.evaluation.live import CommandResult
from ascend_fi.evaluation.suite import _run_phase
from ascend_fi.evaluation.suite import _mode_for_m
from ascend_fi.validation import validate_target


REPO = Path(__file__).resolve().parents[2]


def candidate():
    fact = adapt_qmm_jsonl(
        REPO / "tests/fixtures/qmm_run_trace.json",
        TraceContext(phase="decode", rank=0, model="Qwen3-32B-w8a8-nopdmix", topology="tensor_parallel_2"),
    )[0]
    return resolve_and_rank(fact, load_capabilities(REPO / "profiles/capabilities"))[0].candidate


class FakeControl:
    def __init__(self):
        self.enabled = False
        self.mode = ""
        self.target_m = 1
        self.candidates = [0, 0]
        self.hits = [0, 0]
        self.calls = []

    def run(self, argv, *, env=None):
        command = argv[-1] if len(argv) == 2 else argv[1] if len(argv) > 1 else ""
        self.calls.append(tuple(argv[1:]))
        if command == "status":
            value = {
                "enabled": int(self.enabled),
                "candidate_count": sum(self.candidates),
                "hit_count": sum(self.hits),
                "total_hit_count": sum(self.hits),
                "candidate_count_per_device": self.candidates,
                "hit_count_per_device": self.hits,
                "last_status_per_device": [1 if self.hits[0] else 0, 0],
            }
            return CommandResult(0, json.dumps(value))
        if command == "disarm":
            self.enabled = False
            self.mode = ""
            return CommandResult(0, "")
        if command.startswith("arm-"):
            self.enabled = True
            self.mode = command
            self.target_m = int(argv[3])
            self.candidates = [0, 0]
            self.hits = [0, 0]
            return CommandResult(0, "")
        return CommandResult(2, "", "unexpected control command")


class FakeClient:
    def __init__(self, control: FakeControl, trace: Path):
        self.control = control
        self.trace = trace

    def health(self):
        return True

    def tokenize(self, model, text):
        return list(range(400))

    def detokenize(self, model, token_ids):
        return "word " * len(token_ids)

    def _emit(self, event):
        with self.trace.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

    def completion(self, payload):
        prompt = payload["prompt"]
        m = len(prompt) if isinstance(prompt, list) else 1
        for rank in (0, 1):
            self._emit(
                {
                    "event": "qmm_run_trace",
                    "pid": os.getpid(),
                    "device": rank,
                    "rank": rank,
                    "x1": {"shape": [m, 5120]},
                }
            )
        if self.control.enabled and self.control.target_m == m:
            self.control.candidates[0] += 1
            self.control.hits[0] += 1
            name = {
                "arm-all-qmm-lossless": "lossless_substitution",
                "arm-all-qmm-int4": "int4_equivalent_substitution",
                "arm-all-qmm-int2": "int2_equivalent_substitution",
            }[self.control.mode]
            self._emit(
                {
                    "event": name,
                    "outcome": "success",
                    "pid": os.getpid(),
                    "device": 0,
                    "rank": 0,
                    "hit": self.control.hits[0],
                    "run_slot": 0,
                    "original": "0x1",
                    "temporary": "0x2",
                    "temporary_used": True,
                    "temporary_freed": True,
                    "freed": True,
                    "host_freed": True,
                    "changed_elements": 12 if "lossless" not in name else 0,
                    "run_status": 0,
                    "h2d_status": 0,
                    "copy_status": 0,
                    "post_sync_status": 0,
                    "sync_status": 0,
                    "producer_match": True,
                }
            )
        if payload.get("echo"):
            logprob = -2.0 if self.control.mode == "arm-all-qmm-int2" else -1.0
            count = len(prompt)
            values = [None] + [logprob] * (count - 1) + [logprob]
            return 200, {
                "choices": [{"text": "", "logprobs": {"token_logprobs": values, "tokens": []}}],
                "usage": {"prompt_tokens": count, "completion_tokens": 1},
            }, None
        text = "stable #### 1"
        if self.control.mode == "arm-all-qmm-int4":
            text = "mild #### 1"
        elif self.control.mode == "arm-all-qmm-int2":
            text = "fault #### 0"
        tokens = text.split()
        return 200, {
            "choices": [{"text": text, "finish_reason": "stop", "logprobs": {"tokens": tokens}}],
            "usage": {"completion_tokens": len(tokens)},
        }, None


class LiveSuiteTests(unittest.TestCase):
    def test_existing_all_hit_byte_limit_is_enforced_before_arm(self):
        with self.assertRaisesRegex(ValueError, "existing all-hit controller limit"):
            _mode_for_m(STRONG_INT2_QMM_RANK0, 128)

    def test_live_adapter_matrix_uses_shared_evaluation_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.jsonl"
            trace.write_text("", encoding="utf-8")
            control = FakeControl()
            executor = LiveQwenFaultExecutor(
                LiveExecutorConfig(
                    model_name="test-model",
                    control_command=("ctl",),
                    trace_path=trace,
                    worker_alive_probe=lambda pid: True,
                ),
                FakeClient(control, trace),
                command_runner=control,
            )
            target = candidate()
            validation = validate_target(target, executor)
            self.assertTrue(validation.destructive_injection_allowed)
            result = run_qwen_fault_effect_suite(
                "live-fixture-matrix",
                target,
                validation,
                executor,
                smoke_samples=[EvaluationSample("smoke", "prompt", 4)],
                gsm8k_samples=[{"id": "gsm", "index": 0, "question": "zero plus one", "reference": "#### 1"}],
                wikitext_text="word " * 500,
                artifact_dir=root / "out",
                repeat=3,
                wikitext_windows=1,
            )
            self.assertEqual(
                result.extra["result_label"],
                "QWEN_2CARD_FAULT_EFFECT_EVALUATION_V1_PASS",
                result.extra["acceptance"],
            )
            self.assertEqual(result.mechanism.extra["per_rank"]["ranks"]["0"]["hit_count"], 3)
            self.assertEqual(result.mechanism.extra["per_rank"]["ranks"]["1"]["hit_count"], 0)
            self.assertEqual(result.extra["phase_matrix"]["baseline"]["mechanism"]["per_rank"]["ranks"]["0"]["hit_count"], 0)
            self.assertIsNone(result.extra["phase_matrix"]["recovery"]["mechanism"]["per_rank"]["ranks"]["1"]["expected_reject"])
            self.assertEqual(result.reliability.effect_repeatability, 1.0)
            self.assertEqual(result.reliability.recovery_rate, 1.0)
            self.assertGreater(result.quality_performance.ppl_ratio, 1.0)
            self.assertTrue((root / "out/evaluation_result.json").exists())
            self.assertTrue((root / "out/effect_evaluation_summary.md").exists())
            self.assertEqual(type(result).from_json(result.to_json()), result)

    def test_phase_exception_always_disarms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.jsonl"
            trace.write_text("", encoding="utf-8")
            control = FakeControl()

            class FailingClient(FakeClient):
                def completion(self, payload):
                    if self.control.mode == "arm-all-qmm-int2":
                        raise RuntimeError("simulated request failure after arm")
                    return super().completion(payload)

            executor = LiveQwenFaultExecutor(
                LiveExecutorConfig(
                    model_name="test-model",
                    control_command=("ctl",),
                    trace_path=trace,
                    worker_alive_probe=lambda pid: True,
                ),
                FailingClient(control, trace),
                command_runner=control,
            )
            with self.assertRaisesRegex(RuntimeError, "after arm"):
                _run_phase(
                    executor,
                    "strong",
                    STRONG_INT2_QMM_RANK0,
                    [EvaluationSample("smoke", "prompt", 4)],
                    [],
                    {
                        "window_tokens": 128,
                        "windows": [list(range(128))],
                        "selected_token_count": 128,
                        "scored_token_count": 127,
                        "word_count": 128,
                        "selected_text_sha256": "x",
                    },
                    root / "out",
                )
            self.assertFalse(control.enabled)
            self.assertIn(("disarm",), control.calls)


if __name__ == "__main__":
    unittest.main()
