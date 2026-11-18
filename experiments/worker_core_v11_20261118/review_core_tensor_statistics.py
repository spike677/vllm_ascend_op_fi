"""Read only the declared native step0=B, step1=repeat, step2=F statistics.

No runtime, sidecar window, hit, expected-position or private-array receipt is
read. The mapping is part of this frozen experiment, not discovered from logs.
"""
import argparse
import hashlib
import json
from pathlib import Path

from fault_injection import statistics_audit as comparator
from experiments.c2c3_closure_20261117.audit_statistics_series import captures

COMPARATOR_SHA256 = 'd5633600f8bde98590c4eaf367eb16d76d4e6c9ad5b0cb2f82d434ae6a377ca2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review(root):
    root = Path(root)
    if sha(comparator.__file__) != COMPARATOR_SHA256:
        raise ValueError('FROZEN_COMPARATOR_BYTES_CHANGED')
    if str(comparator.RTOL) != '0.00001' or str(comparator.ATOL) != '1E-8':
        raise ValueError('FIXED_TOLERANCE_CHANGED')
    found = captures(root)
    if set(found) != {(0, 0), (1, 0), (2, 0)}:
        raise ValueError('EXACT_THREE_ACTUAL_STEP_RANK_ORIGINALS_REQUIRED')
    config, _ = comparator.read_json(root/'config.json')
    settings = config.get('statistics', {})
    if not all(settings.get(k) == [] for k in ('scope', 'list', 'tensor_list')) or settings.get('data_mode') != ['all']:
        raise ValueError('FILTERED_CAPTURE_NOT_FULL_MODEL')
    summaries, full = {}, {}
    for name, left, right in [('self', 0, 0), ('repeat', 0, 1), ('bf', 0, 2), ('repeat_f', 1, 2)]:
        value = comparator.compare_raw_statistics(baseline=found[left, 0], fault=found[right, 0],
            baseline_config=root/'config.json', fault_config=root/'config.json',
            baseline_collector=root/'collector_identity.json', fault_collector=root/'collector_identity.json')
        full[name] = value; summaries[name] = comparator.compact_summary(value)
    controls = all(summaries[k]['e5_statistics_comparable'] and summaries[k]['exceedance_count'] == 0
                   for k in ('self', 'repeat'))
    first = summaries['bf']['first_exceedance']
    again = summaries['repeat_f']['first_exceedance']
    stable = first is not None and again is not None and all(first[k] == again[k] for k in ('identity', 'field'))
    return dict(schema='demoncat.public_tensor_statistics_raw_review.v1',
        step_mapping={'baseline': 0, 'baseline_repeat': 1, 'fault': 2}, rank=0,
        rtol='1e-5', atol='1e-8', comparator_sha256=COMPARATOR_SHA256,
        reviewer_sha256=sha(__file__), pairs=summaries,
        baseline_controls_pass=controls, first_location_stable_against_independent_baseline=stable,
        observable_difference_supported=bool(controls and stable and summaries['bf']['e5_statistics_comparable']),
        full_numeric_prefix_observed=bool(first and not first['preceding_unobservable_fields']
                                          and not first['preceding_incomparable_fields']),
        runtime_or_hit_logs_read=False, sidecar_receipts_read=False, private_arrays_read=False,
        expected_first_operator_read=False, statistics_bytes_rewritten=False,
        null_means_equal=False, raw_only=True, current_live_scope_identity_verified=False,
        fault_repeat_stability_measured=False, root_cause_identified=False, e5=False,
        public_qualification_granted=False), full


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--statistics', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    summary, full = review(args.statistics)
    args.out.mkdir(parents=True, exist_ok=False)
    for name, report in full.items():
        (args.out/(name+'-full.json')).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    summary['full_report_sha256'] = {name: sha(args.out/(name+'-full.json')) for name in full}
    output = args.out/'summary.json'
    output.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(summary_sha256=sha(output), observable_difference_supported=summary['observable_difference_supported'],
                         first_exceedance=summary['pairs']['bf']['first_exceedance'], e5=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
