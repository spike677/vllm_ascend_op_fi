"""Audit every actual paired statistics step; no runtime or expected-location input."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def captures(root):
    root = Path(root).resolve(strict=True)
    result = {}
    for path in sorted((root / 'observation').glob('step*/rank*/dump.json')):
        relative = path.relative_to(root).as_posix()
        match = re.fullmatch(r'observation/step(0|[1-9][0-9]*)/rank(0|[1-9][0-9]*)/dump.json', relative)
        if not match or path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError('UNSAFE_OR_NONCANONICAL_CAPTURE:' + relative)
        key = tuple(map(int, match.groups()))
        if key in result:
            raise ValueError('DUPLICATE_STEP_RANK')
        result[key] = path.parent
    if not result:
        raise ValueError('NO_ACTUAL_STATISTICS_STEPS')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'fault', 'comparator', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--comparator-sha256', required=True)
    args = parser.parse_args()
    if sha(args.comparator) != args.comparator_sha256:
        raise ValueError('COMPARATOR_BYTES_CHANGED')
    spec = importlib.util.spec_from_file_location('_reviewed_statistics_comparator', args.comparator)
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)
    if str(comparator.RTOL) != '0.00001' or str(comparator.ATOL) != '1E-8':
        raise ValueError('FIXED_BASELINE_REFERENCE_TOLERANCE_CHANGED')
    b, f = captures(args.baseline), captures(args.fault)
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='demoncat.raw_statistics_series.v1',
        comparator_sha256=args.comparator_sha256, driver_sha256=sha(__file__),
        rtol='1e-5', atol='1e-8', reference='baseline', runtime_evidence_read=False,
        private_tensors_read=False, expected_location_read=False,
        baseline_steps=[list(key) for key in sorted(b)], fault_steps=[list(key) for key in sorted(f)],
        actual_step_rank_sets_equal=set(b) == set(f), pairs=[], public_qualification_granted=False)
    # A missing step is reported explicitly. Common steps can still be observed,
    # but no successful pair is allowed to imply complete-series comparability.
    for step, rank in sorted(set(b) & set(f)):
        result = comparator.compare_raw_statistics(
            baseline=b[step, rank], fault=f[step, rank],
            baseline_config=args.baseline / 'config.json', fault_config=args.fault / 'config.json',
            baseline_collector=args.baseline / 'collector_identity.json',
            fault_collector=args.fault / 'collector_identity.json')
        path = args.out / ('step' + str(step) + '-rank' + str(rank) + '-full.json')
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        summary = comparator.compact_summary(result)
        summary.update(step=step, rank=rank, full_report_sha256=sha(path), full_report_path=path.name)
        report['pairs'].append(summary)
    report['status_counts'] = dict(Counter(row['status'] for row in report['pairs']))
    report['all_steps_e5_comparable'] = (report['actual_step_rank_sets_equal']
        and bool(report['pairs']) and all(row['e5_statistics_comparable'] for row in report['pairs']))
    target = args.out / 'series-summary.json'
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(summary_sha256=sha(target), pairs=len(report['pairs']),
        step_sets_equal=report['actual_step_rank_sets_equal'], status_counts=report['status_counts'],
        all_steps_e5_comparable=report['all_steps_e5_comparable'], out=str(args.out))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
