#!/usr/bin/env python3
"""Generate a no-rerun validity audit from completed Qwen V1 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ascend_fi.evaluation.audit import write_baseline_audit_artifacts


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain an object")
        rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-phase", type=Path, required=True)
    parser.add_argument("--strong-phase", type=Path, required=True)
    parser.add_argument("--gsm8k", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--evaluation-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = write_baseline_audit_artifacts(
        args.output,
        baseline_phase=_json(args.baseline_phase),
        gsm8k_rows=_jsonl(args.gsm8k),
        strong_phase=_json(args.strong_phase),
        smoke_samples=_jsonl(args.smoke),
        evaluation_result=_json(args.evaluation_result) if args.evaluation_result else None,
    )
    audit = result["audit"]
    print(
        "BASELINE_VALIDITY_AUDIT "
        f"status={audit['status']} label={audit['result_label']} "
        f"strict={audit['strict_correct_count']}/{audit['sample_count']} "
        f"resolved={audit['answer_resolved_count']}"
    )
    return 0 if audit["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
