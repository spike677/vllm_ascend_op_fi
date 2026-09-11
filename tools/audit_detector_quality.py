"""Offline full-text frozen detector regression and independent-label audit."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ascend_fi.evaluation.detector_audit import audit_cases, historical_explanation_cases, write_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cases", type=Path, help="full-text cases with per-detector label provenance")
    source.add_argument("--historical-explanation-audit", type=Path, help="hash-verified legacy complete-text audit; expectations are not independent gold")
    parser.add_argument("--raw-source", type=Path, help="optional historical raw ZIP/directory to retain original prompt and token pieces")
    parser.add_argument("--suite", type=Path, default=Path(__file__).resolve().parents[1] / "profiles/evaluation/distortion_probe_suite_v1.json")
    parser.add_argument("--out", type=Path, required=True, help="new report path; existing files are never overwritten")
    args = parser.parse_args()
    if args.raw_source and not args.historical_explanation_audit:
        parser.error("--raw-source requires --historical-explanation-audit")
    if args.cases:
        result = write_audit(args.cases, args.suite, args.out)
    else:
        result = audit_cases(historical_explanation_cases(args.historical_explanation_audit, args.raw_source), args.suite)
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=True, indent=2, allow_nan=False)
            stream.write("\n")
    print(result["regression"]["status"], result["quality_status"])
    return 1 if result["regression"]["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
