"""Live adapter for an already-running developer-mode Qwen/Fault20 harness.

The adapter owns orchestration only.  Native tensor mutation stays in the
existing ``fault20_buffer_ctl``/preload capability.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterator
from typing import Any, Callable, Mapping, Protocol, Sequence

from ascend_fi.contracts import Candidate
from ascend_fi.validation import LosslessEvidence

from .metrics import aggregate_per_rank
from .runner import RunOutcome


@dataclass(frozen=True)
class EvaluationSample:
    sample_id: str
    prompt: str
    max_tokens: int
    temperature: float = 0.0
    seed: int = 1234
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LiveFaultMode:
    name: str
    arm_args: tuple[str, ...]
    event_name: str
    target_device_mask: int


LOSSLESS_QMM_RANK0 = LiveFaultMode(
    name="lossless",
    arm_args=("arm-all-qmm-lossless", "1", "1", "5120", "5120"),
    event_name="lossless_substitution",
    target_device_mask=1,
)
MILD_INT4_QMM_RANK0 = LiveFaultMode(
    name="mild_int4_equivalent",
    arm_args=("arm-all-qmm-int4", "1", "1", "5120", "5120"),
    event_name="int4_equivalent_substitution",
    target_device_mask=1,
)
STRONG_INT2_QMM_RANK0 = LiveFaultMode(
    name="strong_int2_equivalent",
    arm_args=("arm-all-qmm-int2", "1", "1", "5120", "5120"),
    event_name="int2_equivalent_substitution",
    target_device_mask=1,
)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str = ""


class CommandRunner(Protocol):
    def run(self, argv: Sequence[str], *, env: Mapping[str, str] | None = None) -> CommandResult: ...


class SubprocessCommandRunner:
    def run(self, argv: Sequence[str], *, env: Mapping[str, str] | None = None) -> CommandResult:
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        completed = subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            env=process_env,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)


class CompletionClient(Protocol):
    def health(self) -> bool: ...
    def completion(self, payload: Mapping[str, Any]) -> tuple[int | None, dict[str, Any] | None, str | None]: ...
    def tokenize(self, model: str, text: str) -> list[int]: ...
    def detokenize(self, model: str, token_ids: Sequence[int]) -> str: ...


class OpenAICompletionClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(
                self.base_url + "/health", timeout=min(self.timeout_seconds, 10.0)
            ) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            return False

    def _post(
        self, path: str, payload: Mapping[str, Any]
    ) -> tuple[int | None, dict[str, Any] | None, str | None]:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(dict(payload), ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8", errors="replace")
                value = json.loads(raw)
                return response.status, value if isinstance(value, dict) else None, None
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            return error.code, None, body
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            return None, None, f"{type(error).__name__}: {error}"

    def completion(
        self, payload: Mapping[str, Any]
    ) -> tuple[int | None, dict[str, Any] | None, str | None]:
        return self._post("/v1/completions", payload)

    def tokenize(self, model: str, text: str) -> list[int]:
        status, raw, error = self._post("/tokenize", {"model": model, "prompt": text})
        tokens = None if raw is None else raw.get("tokens")
        if status != 200 or not isinstance(tokens, list) or not all(isinstance(item, int) for item in tokens):
            raise RuntimeError(f"tokenize failed: status={status} error={error}")
        return tokens

    def detokenize(self, model: str, token_ids: Sequence[int]) -> str:
        status, raw, error = self._post(
            "/detokenize", {"model": model, "tokens": list(token_ids)}
        )
        text = None if raw is None else raw.get("prompt", raw.get("text"))
        if status != 200 or not isinstance(text, str):
            raise RuntimeError(f"detokenize failed: status={status} error={error}")
        return text


@dataclass(frozen=True)
class LiveExecutorConfig:
    model_name: str
    control_command: tuple[str, ...]
    trace_path: Path
    service_log_path: Path | None = None
    expected_worker_count: int = 2
    worker_alive_probe: Callable[[int], bool] | None = None
    lossless_mode: LiveFaultMode = LOSSLESS_QMM_RANK0
    fault_mode: LiveFaultMode = MILD_INT4_QMM_RANK0
    gate_sample: EvaluationSample = EvaluationSample(
        sample_id="lossless-gate",
        prompt="What is the capital of France? Answer with one word.",
        max_tokens=16,
    )


class LiveQwenFaultExecutor:
    """Adapter compatible with both ``LosslessExecutor`` and ``EvaluationExecutor``."""

    def __init__(
        self,
        config: LiveExecutorConfig,
        client: CompletionClient,
        *,
        command_runner: CommandRunner | None = None,
    ):
        self.config = config
        self.client = client
        self.command_runner = command_runner or SubprocessCommandRunner()
        self.last_lossless: dict[str, Any] | None = None

    def _control(self, *args: str) -> dict[str, Any] | None:
        result = self.command_runner.run(
            self.config.control_command + tuple(args),
            env={"VLLM_SERVER_DEV_MODE": "1"},
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"fault control {' '.join(args)} failed ({result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        if not result.stdout.strip():
            return None
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"stdout": result.stdout.strip()}
        return value if isinstance(value, dict) else {"value": value}

    def status(self) -> dict[str, Any]:
        return self._control("status") or {}

    def arm(self, mode: LiveFaultMode) -> dict[str, Any]:
        self.disarm(None)
        result = self._control(*mode.arm_args)
        status = self.status()
        if status.get("enabled") != 1:
            raise RuntimeError(f"fault mode {mode.name} did not arm: {status}")
        return result or status

    def disarm(self, candidate: Candidate | None = None) -> bool:
        try:
            self._control("disarm")
            return self.status().get("enabled") == 0
        except Exception:
            return False

    def trace_offset(self) -> int:
        try:
            return self.config.trace_path.stat().st_size
        except FileNotFoundError:
            return 0

    def trace_events(self, offset: int = 0) -> list[dict[str, Any]]:
        return list(self.iter_trace_events(offset))

    def iter_trace_events(self, offset: int = 0) -> Iterator[dict[str, Any]]:
        """Stream native JSONL from an offset without materializing a multi-GB trace."""

        try:
            with self.config.trace_path.open("rb") as handle:
                handle.seek(offset)
                for raw_line in handle:
                    try:
                        value = json.loads(raw_line.decode("utf-8", errors="replace"))
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict):
                        yield value
        except FileNotFoundError:
            return

    def run_sample(self, sample: EvaluationSample, **overrides: Any) -> RunOutcome:
        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "prompt": sample.prompt,
            "max_tokens": sample.max_tokens,
            "temperature": sample.temperature,
            "seed": sample.seed,
            "logprobs": 1,
        }
        payload.update(overrides)
        http_status, raw, error = self.client.completion(payload)
        if raw is None:
            return RunOutcome(
                http_status=http_status,
                response_text=None,
                worker_alive=self.client.health(),
                error=error,
                evidence_refs=(str(self.config.trace_path),),
            )
        choices = raw.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        response_text = choice.get("text") if isinstance(choice.get("text"), str) else None
        logprobs = choice.get("logprobs") if isinstance(choice.get("logprobs"), dict) else {}
        raw_ids = choice.get("token_ids")
        if raw_ids is None:
            raw_ids = logprobs.get("token_ids")
        token_ids = (
            tuple(raw_ids)
            if isinstance(raw_ids, list) and all(isinstance(value, int) for value in raw_ids)
            else None
        )
        raw_pieces = logprobs.get("tokens")
        token_pieces = (
            tuple(str(value) for value in raw_pieces)
            if isinstance(raw_pieces, list)
            else None
        )
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        generation_length = usage.get("completion_tokens")
        if not isinstance(generation_length, int):
            generation_length = len(token_ids or token_pieces or ()) or None
        return RunOutcome(
            http_status=http_status,
            response_text=response_text,
            token_ids=token_ids,
            token_pieces=token_pieces,
            generation_length=generation_length,
            worker_alive=self.client.health(),
            finish_reason=choice.get("finish_reason"),
            error=error,
            evidence_refs=(str(self.config.trace_path),),
        )

    def tokenize(self, text: str) -> list[int]:
        return self.client.tokenize(self.config.model_name, text)

    def detokenize(self, token_ids: Sequence[int]) -> str:
        return self.client.detokenize(self.config.model_name, token_ids)

    def prompt_loglikelihood(self, prompt: str | Sequence[int]) -> dict[str, Any]:
        status, raw, error = self.client.completion(
            {
                "model": self.config.model_name,
                "prompt": prompt if isinstance(prompt, str) else list(prompt),
                "max_tokens": 1,
                "temperature": 0,
                "seed": 1234,
                "logprobs": 1,
                "echo": True,
            }
        )
        if raw is None:
            return {"http_status": status, "error": error, "token_logprobs": None}
        choices = raw.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        logprobs = choice.get("logprobs") if isinstance(choice.get("logprobs"), dict) else {}
        values = logprobs.get("token_logprobs")
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        prompt_tokens = usage.get("prompt_tokens")
        if isinstance(values, list) and isinstance(prompt_tokens, int):
            values = values[:prompt_tokens]
        return {
            "http_status": status,
            "error": error,
            "token_logprobs": values if isinstance(values, list) else None,
            "prompt_tokens": prompt_tokens,
        }

    def run_baseline(self, candidate: Candidate) -> RunOutcome:
        if not self.disarm(candidate):
            raise RuntimeError("baseline could not establish disarmed control")
        return self.run_sample(self.config.gate_sample)

    def run_fault(self, candidate: Candidate) -> RunOutcome:
        self.arm(self.config.fault_mode)
        before = self.trace_offset()
        outcome = self.run_sample(self.config.gate_sample)
        snapshot = self.runtime_snapshot(before, self.config.fault_mode)
        return RunOutcome(
            **{
                **outcome.__dict__,
                "mechanism": {
                    "resolved": snapshot["resolved"],
                    "candidate_count": snapshot["status"].get("candidate_count"),
                    "hit_count": snapshot["status"].get("total_hit_count", snapshot["status"].get("hit_count")),
                    "reject_count": snapshot.get("reject_count"),
                    "reject_reasons": snapshot.get("reject_reasons", {}),
                    "changed_elements": snapshot.get("changed_elements"),
                    "restore_exact": snapshot.get("temporary_freed"),
                    "auto_disarmed": snapshot["status"].get("enabled") == 0,
                    "per_rank": snapshot["multirank"],
                },
            }
        )

    def run_recovery(self, candidate: Candidate) -> RunOutcome:
        if not self.disarm(candidate):
            raise RuntimeError("recovery could not establish disarmed control")
        return self.run_sample(self.config.gate_sample)

    def execute_lossless(self, candidate: Candidate) -> LosslessEvidence:
        if not self.disarm(candidate):
            raise RuntimeError("lossless gate could not establish disarmed control")
        baseline = self.run_sample(self.config.gate_sample)
        offset = self.trace_offset()
        try:
            self.arm(self.config.lossless_mode)
            observed = self.run_sample(self.config.gate_sample)
            snapshot = self.runtime_snapshot(offset, self.config.lossless_mode)
        finally:
            disarmed = self.disarm(candidate)
        recovery = self.run_sample(self.config.gate_sample)
        event_count = int(snapshot["injection_event_count"])
        temporary_ok = event_count > 0 and snapshot["all_temporary_distinct"] is True
        freed = event_count > 0 and snapshot["temporary_freed"] is True
        output_equal = (
            baseline.response_text is not None
            and baseline.response_text == observed.response_text == recovery.response_text
            and (baseline.token_ids is None or baseline.token_ids == observed.token_ids == recovery.token_ids)
            and (baseline.token_pieces is None or baseline.token_pieces == observed.token_pieces == recovery.token_pieces)
        )
        self.last_lossless = {
            "baseline": baseline,
            "lossless": observed,
            "recovery": recovery,
            "snapshot": snapshot,
            "disarm_success": disarmed,
        }
        return LosslessEvidence(
            target_resolved=snapshot["resolved"],
            expected_api_slot_hit=(
                event_count > 0
                and candidate.binding.runtime_slot == 0
                and snapshot["all_producer_match"] is True
            ),
            substitution_executed=temporary_ok,
            hit_count=event_count,
            reject_count=int(snapshot.get("reject_count") or 0),
            reject_reasons=dict(snapshot.get("reject_reasons", {})),
            model_output_equal=output_equal,
            original_storage_untouched=temporary_ok,
            run_status=0 if snapshot["all_run_status_zero"] is True else None,
            copy_status=0 if snapshot["all_copy_status_zero"] is True else None,
            sync_status=0 if snapshot["all_sync_status_zero"] is True else None,
            temporary_freed=freed and disarmed,
            evidence_refs=(str(self.config.trace_path),),
        )

    def runtime_snapshot(
        self, offset: int, mode: LiveFaultMode, *, target_m: int | None = None
    ) -> dict[str, Any]:
        status = self.status()
        worker_pids: dict[int, int] = {}
        matching_calls: dict[int, int] = {}
        injection_count = 0
        injection_samples: list[dict[str, Any]] = []
        all_temporary_distinct = True
        all_producer_match = True
        all_run_status_zero = True
        all_copy_status_zero = True
        all_sync_status_zero = True
        all_freed = True
        changed_total = 0
        changed_seen = False
        if target_m is None:
            target_m = (
                int(mode.arm_args[2])
                if len(mode.arm_args) == 5 and mode.arm_args[2].isdigit()
                else 1
            )
        for event in self.iter_trace_events(offset):
            rank = event.get("rank", event.get("device"))
            x1 = event.get("x1") if isinstance(event.get("x1"), dict) else {}
            if event.get("event") == "qmm_run_trace" and x1.get("shape") == [target_m, 5120]:
                if isinstance(rank, int):
                    matching_calls[rank] = matching_calls.get(rank, 0) + 1
                    pid = event.get("pid")
                    if isinstance(pid, int):
                        worker_pids[rank] = pid
            if event.get("event") == mode.event_name and event.get("outcome", "success") == "success":
                injection_count += 1
                if len(injection_samples) < 8:
                    injection_samples.append(event)
                all_temporary_distinct &= event.get("temporary") != event.get("original")
                all_producer_match &= event.get("producer_match") is True
                all_run_status_zero &= event.get("run_status") == 0
                all_copy_status_zero &= event.get("h2d_status", event.get("copy_status")) == 0
                all_sync_status_zero &= event.get("post_sync_status", event.get("sync_status")) == 0
                all_freed &= (
                    (event.get("temporary_freed") is True or event.get("freed") is True)
                    and event.get("free_status", 0) == 0
                    and event.get("host_freed", True) is True
                )
                changed = event.get("changed_elements")
                if isinstance(changed, int):
                    changed_total += changed
                    changed_seen = True
        probe = self.config.worker_alive_probe or (lambda pid: Path(f"/proc/{pid}").exists())
        worker_alive = {rank: probe(pid) for rank, pid in worker_pids.items()}
        multirank = aggregate_per_rank(
            status,
            matching_calls,
            worker_pids,
            worker_alive,
            expected_workers=self.config.expected_worker_count,
            target_device_mask=mode.target_device_mask,
            injection_active=bool(mode.arm_args),
        )
        rejects = sum(
            rank.get("reject_count") or 0 for rank in multirank["ranks"].values()
        )
        reject_reasons: dict[str, int] = {}
        for rank in multirank["ranks"].values():
            for reason, count in rank["reject_reasons"].items():
                reject_reasons[reason] = reject_reasons.get(reason, 0) + count
        temporary_freed = injection_count > 0 and all_freed
        errors = self._service_errors()
        return {
            "mode": mode.name,
            "status": status,
            "injection_event_count": injection_count,
            "injection_event_samples": injection_samples,
            "resolved": injection_count > 0,
            "changed_elements": changed_total if changed_seen else None,
            "temporary_freed": temporary_freed,
            "all_temporary_distinct": injection_count > 0 and all_temporary_distinct,
            "all_producer_match": injection_count > 0 and all_producer_match,
            "all_run_status_zero": injection_count > 0 and all_run_status_zero,
            "all_copy_status_zero": injection_count > 0 and all_copy_status_zero,
            "all_sync_status_zero": injection_count > 0 and all_sync_status_zero,
            "reject_count": rejects,
            "reject_reasons": reject_reasons,
            "multirank": multirank,
            "service_health": self.client.health(),
            "errors": errors,
        }

    def _service_errors(self) -> dict[str, bool | None]:
        path = self.config.service_log_path
        if path is None or not path.exists():
            return {"cann_error": None, "hccl_error": None, "oom": None, "worker_exit": None}
        with path.open("rb") as handle:
            size = handle.seek(0, 2)
            handle.seek(max(0, size - 512_000))
            text = handle.read().decode("utf-8", errors="replace")
        return {
            "cann_error": bool(re.search(r"(?:CANN|ACL|EZ)\w*.*(?:ERROR|failed)", text, re.I)),
            "hccl_error": bool(re.search(r"HCCL.*(?:ERROR|failed)", text, re.I)),
            "oom": bool(re.search(r"out of memory|OOM", text, re.I)),
            "worker_exit": bool(re.search(r"Worker_TP\d+.*(?:exited|terminating|shutting down)", text, re.I)),
        }
