"""Actual pretrained DeepSeek MoE all-target / every-match Worker probe.

Shares the genuine enqueued request lifecycle with Worker18, but installs the
source-bound actual MoE input remap and reports a separate mechanism/scope.
No model/router layer is constructed or replaced by this entry point.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from experiments.c2c3_closure_20261117.natural_rms_worker_probe import main


if __name__ == '__main__':
    raise SystemExit(main('8.restore_offset'))
