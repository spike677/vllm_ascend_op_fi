"""Create an offline Fault Effect Profile from a completed evaluation directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from ascend_fi.evaluation.replay import replay_completed_effect_cycles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path, help="completed evaluation artifact directory")
    parser.add_argument("output_dir", type=Path, help="new directory for sanitized effect_card artifacts")
    arguments = parser.parse_args()
    result = replay_completed_effect_cycles(arguments.input_root, arguments.output_dir)
    print(f"EFFECT_PROFILE_REPLAY_COMPLETE stage={result['profile']['effect_stage']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
