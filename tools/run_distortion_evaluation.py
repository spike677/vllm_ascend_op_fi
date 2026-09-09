"""Distortion-first evaluation. Offline by default; no embedded fault controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ascend_fi.evaluation.distortion.contracts import load_suite, FaultConfigurationRecord
from ascend_fi.evaluation.distortion.evaluator import evaluate_cycles
from ascend_fi.evaluation.distortion.replay import load_legacy
from ascend_fi.evaluation.distortion.report import write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("distortion", "analysis", "both"), default="distortion")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-suite", type=Path, default=Path(__file__).resolve().parents[1] / "profiles/evaluation/distortion_probe_suite_v1.json")
    parser.add_argument("--live-adapter", help="Existing lifecycle adapter module:factory (owns lock/arm/cleanup); no second controller")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--fault-config", type=Path)
    parser.add_argument("--cycles", type=int, default=3)
    args = parser.parse_args()
    frozen = load_suite(args.probe_suite)
    if args.replay:
        if args.live_adapter or args.base_url:
            parser.error("replay must not contact the live service")
        if args.output.resolve() == args.replay.resolve() or args.replay.resolve() in args.output.resolve().parents:
            parser.error("derived output must be outside the original evidence tree")
        if args.replay.is_dir() and (args.replay / 'live_raw.json').is_file():
            raw_path = args.replay / 'live_raw.json'
            raw = json.loads(raw_path.read_text(encoding='utf-8'))
            if raw.get('schema') != 'NORMAL_DIAGNOSTIC_LIVE_V1' or raw['suite'] != frozen:
                raise ValueError('live replay schema/frozen suite mismatch')
            suite, config, cycles = raw['suite'], FaultConfigurationRecord.from_dict(raw['config']), raw['cycles']
            analysis, provenance = {'status': 'NOT_REQUESTED'}, {**raw['provenance'],
                'raw_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(), 'source': str(raw_path)}
        else:
            suite, config, cycles, analysis, provenance = load_legacy(args.replay, frozen)
    else:
        from ascend_fi.evaluation.distortion.live import run_existing_lifecycle
        if not args.live_adapter or not args.fault_config or not args.base_url or not args.model:
            parser.error("live requires explicit existing adapter, fault config, base URL and model")
        suite, config, cycles, analysis, provenance = run_existing_lifecycle(args, frozen)
    result = evaluate_cycles(suite, config, cycles, analysis=analysis,
                             evaluation_id=args.output.name)
    package = Path(__file__).resolve().parents[1] / "src/ascend_fi/evaluation/distortion"
    provenance["evaluator_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.glob("*.py"))}
    provenance["frozen_suite_file_sha256"] = hashlib.sha256(args.probe_suite.read_bytes()).hexdigest()
    paths = write_report(args.output, result, provenance, mode=args.mode)
    d = result.extra["distortion"]
    print(json.dumps({"paths": paths, "mechanism": d["mechanism_gate"],
                      "stable_distorted": d["stable_distorted_probes"], "recovery": d["recovery"],
                      "robust_gate": d["robust_gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
