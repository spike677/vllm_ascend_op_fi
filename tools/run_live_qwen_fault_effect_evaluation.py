#!/usr/bin/env python3
"""Run the real two-rank Qwen Fault Effect Evaluation V1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ascend_fi.discovery import TraceContext, adapt_qmm_run_trace, load_capabilities, resolve_and_rank
from ascend_fi.evaluation import (
    LiveExecutorConfig,
    LiveQwenFaultExecutor,
    OpenAICompletionClient,
    load_gsm8k,
    load_smoke_pack,
    read_wikitext_raw,
    run_qwen_fault_effect_suite,
)
from ascend_fi.validation import validate_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--service-log", type=Path)
    parser.add_argument("--smoke", required=True, type=Path)
    parser.add_argument("--gsm8k", required=True, type=Path)
    parser.add_argument("--gsm-limit", type=int, default=10)
    parser.add_argument("--wikitext", required=True, type=Path)
    parser.add_argument("--wikitext-windows", type=int, default=8)
    parser.add_argument(
        "--wikitext-window-tokens",
        type=int,
        default=8,
        help="fixed teacher-forced token window; 8 stays within the existing 65536-byte all-hit limit",
    )
    parser.add_argument("--capabilities", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--environment-json", type=Path)
    parser.add_argument("--evaluation-id", default="qwen-2card-fault-effect-v1")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    executor = LiveQwenFaultExecutor(
        LiveExecutorConfig(
            model_name=args.model_name,
            control_command=(str(args.control),),
            trace_path=args.trace,
            service_log_path=args.service_log,
        ),
        OpenAICompletionClient(args.base_url),
    )
    if not executor.client.health():
        raise SystemExit("live Qwen service health check failed")

    # Ensure a current decode trace exists before adapting a live candidate.
    executor.disarm(None)
    executor.run_sample(executor.config.gate_sample)
    context = TraceContext(
        phase="decode",
        model="Qwen3-32B-w8a8-nopdmix",
        topology="tensor_parallel_2",
        evidence_ref=str(args.trace),
    )
    latest = None
    # The gate request immediately above guarantees a current candidate near
    # EOF. Bound the scan so a long-running native trace never becomes a
    # discovery-time memory/CPU hazard.
    with args.trace.open("rb") as handle:
        start = max(0, args.trace.stat().st_size - 64 * 1024 * 1024)
        handle.seek(start)
        if start:
            handle.readline()
        for raw_line in handle:
            try:
                record = json.loads(raw_line.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            x1 = record.get("x1") if isinstance(record.get("x1"), dict) else {}
            if (
                record.get("event") != "qmm_run_trace"
                or record.get("rank", record.get("device")) != 0
                or x1.get("shape") != [1, 5120]
            ):
                continue
            fact = adapt_qmm_run_trace(record, context)
            if (
                fact is not None
                and fact.rank == 0
                and fact.shape == (1, 5120)
                and fact.proof.get("semantic_slot_proven") is True
            ):
                latest = fact
    if latest is None:
        raise SystemExit("no current rank0 [1,5120] semantic-slot QMM candidate")
    ranked = resolve_and_rank(latest, load_capabilities(args.capabilities))
    eligible = next(
        (
            item.candidate
            for item in ranked
            if item.candidate.capability_id == "op_api.quant_matmul.input_buffer_transform"
        ),
        None,
    )
    if eligible is None:
        raise SystemExit("target QMM input-buffer capability did not resolve")
    (args.output / "candidate.json").write_text(eligible.to_json() + "\n", encoding="utf-8")

    validation = validate_target(eligible, executor)
    (args.output / "lossless_validation.json").write_text(
        validation.to_json() + "\n", encoding="utf-8"
    )
    if executor.last_lossless is not None:
        lossless_runtime = {
            "baseline": executor.last_lossless["baseline"].as_record().to_dict(),
            "lossless": executor.last_lossless["lossless"].as_record().to_dict(),
            "recovery": executor.last_lossless["recovery"].as_record().to_dict(),
            "snapshot": {
                key: value
                for key, value in executor.last_lossless["snapshot"].items()
                if key != "events"
            },
            "disarm_success": executor.last_lossless["disarm_success"],
        }
        (args.output / "lossless_gate_runtime.json").write_text(
            json.dumps(lossless_runtime, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if not validation.destructive_injection_allowed:
        print(validation.to_json())
        return 2
    if args.validate_only:
        print(validation.to_json())
        return 0

    environment = {}
    if args.environment_json:
        value = json.loads(args.environment_json.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("environment JSON must be an object")
        environment = value
    result = run_qwen_fault_effect_suite(
        args.evaluation_id,
        eligible,
        validation,
        executor,
        smoke_samples=load_smoke_pack(args.smoke),
        gsm8k_samples=load_gsm8k(args.gsm8k, limit=args.gsm_limit),
        wikitext_text=read_wikitext_raw(args.wikitext),
        artifact_dir=args.output,
        repeat=args.repeat,
        wikitext_windows=args.wikitext_windows,
        wikitext_window_tokens=args.wikitext_window_tokens,
        environment=environment,
    )
    print(result.to_json())
    return 0 if result.extra.get("result_label") == "QWEN_2CARD_FAULT_EFFECT_EVALUATION_V1_PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
