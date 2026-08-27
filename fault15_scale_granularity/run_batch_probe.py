#!/usr/bin/env python3
"""Submit deterministic 1/2/4/8-sequence batches to a vLLM completions API."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def post_json(url: str, payload: dict, timeout: float) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18015")
    parser.add_argument("--model", default="qwen3-32b-w8a8-fault15-trace")
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for batch_size in (int(value) for value in args.batches.split(",") if value.strip()):
        prompts = [f"请用一个短句说明矩阵乘法。样本编号{i}。" for i in range(batch_size)]
        payload = {
            "model": args.model,
            "prompt": prompts,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "seed": 20260811,
            "return_token_ids": True,
        }
        started = time.time()
        status, response = post_json(f"{args.base_url}/v1/completions", payload, args.timeout)
        ended = time.time()
        record = {
            "requested_batch_size": batch_size,
            "http_status": status,
            "started_unix": started,
            "ended_unix": ended,
            "elapsed_seconds": ended - started,
            "choice_count": len(response.get("choices", [])),
            "response": response,
        }
        path = args.output_dir / f"batch_{batch_size}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary.append({key: record[key] for key in record if key != "response"})
        print(json.dumps(summary[-1], ensure_ascii=False))

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
