"""Offline fixed-tolerance raw statistics audit; no NPU/Worker/API calls."""
import argparse
import hashlib
import json
from pathlib import Path

from fault_injection.statistics_audit import compare_raw_statistics, compact_summary


def main():
    parser = argparse.ArgumentParser()
    for name in ('baseline', 'fault', 'baseline-config', 'fault-config',
                 'baseline-collector', 'fault-collector'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    options = vars(args).copy(); folder = options.pop('out')
    report = compare_raw_statistics(**options)
    folder.mkdir(parents=True, exist_ok=False)
    full = folder/'independent_full.json'
    full.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    summary = compact_summary(report)
    summary['independent_full_sha256'] = hashlib.sha256(full.read_bytes()).hexdigest()
    (folder/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], first_exceedance=report['first_exceedance'],
                         e5_statistics_comparable=report['e5_statistics_comparable'], out=str(folder))))


if __name__ == '__main__':
    main()
