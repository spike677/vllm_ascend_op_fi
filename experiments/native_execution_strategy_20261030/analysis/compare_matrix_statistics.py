"""Bounded, preregistered comparisons using only the independent msprobe reader.

Discovery: four same-batch repeats. Each selected controlled target: all twelve
within-case baseline comparisons, four matching-call repeats, eight independent
process repeats, and four stock/private native comparisons. Full plan: 88 pairs.
No profile, host callback, private tensor or numerical fingerprint is consulted.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import compare_replay_statistics as comparison


TARGETS = ('query_tile', 'key_group', 'joint')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def build_plan(group, discovery, controlled):
    """Plan depends only on preregistered case IDs/steps, never observed values."""
    plan = []

    def add(label, kind, left_attempt, left, ls, right_attempt, right, rs):
        plan.append({'comparison_id': label, 'kind': kind,
            'B': {'case_directory': str(left_attempt / left), 'completion': str(left_attempt / 'completion.json'), 'step': ls},
            'F': {'case_directory': str(right_attempt / right), 'completion': str(right_attempt / 'completion.json'), 'step': rs}})

    if group in ('all', 'discovery'):
        for batch in (1, 2, 4, 8):
            add(f'discovery_b{batch}_r0_vs_r1', 'stock_independent_repeat', discovery,
                f'discovery_b{batch}_r0', 0, discovery, f'discovery_b{batch}_r1', 0)
    for target in TARGETS if group == 'all' else (group,) if group in TARGETS else ():
        attempt = controlled[target]
        for batch in (1, 8):
            prefix = f'controlled_{target}_b{batch}'
            for repeat in (0, 1):
                case = prefix + f'_r{repeat}'
                for step in (1, 2, 3):
                    add(case + f'_s0_vs_s{step}', 'off_restore' if step == 3 else 'native_vs_controlled',
                        attempt, case, 0, attempt, case, step)
                add(case + '_s1_vs_s2', 'same_process_matching_repeat', attempt, case, 1, attempt, case, 2)
                add(case + '_stock_vs_private_s0', 'stock_vs_private_native', discovery,
                    f'discovery_b{batch}_r{repeat}', 0, attempt, case, 0)
            for step in range(4):
                add(prefix + f'_r0_vs_r1_s{step}', 'controlled_independent_repeat', attempt,
                    prefix + '_r0', step, attempt, prefix + '_r1', step)
    require(len(plan) == (88 if group == 'all' else 4 if group == 'discovery' else 28), 'Plan count differs')
    require(len({row['comparison_id'] for row in plan}) == len(plan), 'Duplicate comparison ID')
    return plan


def summarize(item, report, report_file):
    complete = report.get('alignment', {}).get('complete') is True
    first = report.get('first_observed_api_difference')
    main = report.get('attention_score_output_observed_difference') if complete else None
    aux = report.get('auxiliary_output_observed_differences') if complete else None
    if not complete:
        observed_class = 'INCONCLUSIVE_OR_INVALID'
    elif main:
        observed_class = 'PRIMARY_ATTENTION_SUMMARY_DIFFERENCE'
    elif aux:
        observed_class = 'AUXILIARY_ONLY_OBSERVED_DIFFERENCE'
    else:
        observed_class = 'NO_OBSERVED_NATIVE_RETURN_SUMMARY_DIFFERENCE'
    native_rows = [row for row in report.get('comparisons', []) if comparison.API.fullmatch(row['B']['name'])]
    native = native_rows[0] if complete and len(native_rows) == 1 else None
    primary_unknown = len(native['return_slots'][0]['unobservable']) if native else None
    return {'comparison_id': item['comparison_id'], 'kind': item['kind'], 'B': item['B'], 'F': item['F'],
        'report_file': report_file.name, 'report_sha256': sha(report_file),
        'status': report['status'], 'alignment_complete': complete,
        'observed_class': observed_class,
        'first_observed_api': first['B']['name'] if first else None,
        'first_difference_boundary': report.get('first_difference_boundary'),
        'native_input_observed_difference': bool(native['input']['differences']) if native else None,
        'primary_attention_output_observed_difference': main,
        'primary_attention_unobservable_fields': primary_unknown,
        'auxiliary_output_observed_differences': aux,
        'unobservable_statistic_fields': report.get('unobservable_statistic_fields'),
        'evidence_sha256': report.get('evidence_sha256'),
        'statistical_restore_observed': (False if main else True if primary_unknown == 0 else None)
            if item['kind'] == 'off_restore' and complete else None,
        'tensor_equality_or_bitwise_restore_proven': False,
        'physical_fault_location_proven': False, 'mechanism_accepted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=('all', 'discovery', *TARGETS), required=True)
    parser.add_argument('--discovery-attempt', type=Path, required=True)
    parser.add_argument('--controlled-attempt', action='append', default=[], help='target=absolute_attempt_directory')
    parser.add_argument('--output', type=Path, required=True, help='Fresh analysis-only directory')
    parser.add_argument('--max-seconds', type=int, default=120)
    args = parser.parse_args()
    require(10 <= args.max_seconds <= 600, 'Bounded CPU analysis duration required')
    require(not args.output.exists(), 'Analysis output must be new')
    controlled = {}
    for value in args.controlled_attempt:
        target, path = value.split('=', 1)
        require(target in TARGETS and target not in controlled, 'Unknown/duplicate controlled target')
        controlled[target] = Path(path).resolve(strict=True)
    needed = set(TARGETS) if args.group == 'all' else {args.group} if args.group in TARGETS else set()
    require(set(controlled) == needed, 'Provide exactly the controlled targets selected for this report')
    discovery = args.discovery_attempt.resolve(strict=True)
    plan = build_plan(args.group, discovery, controlled)
    destination = args.output.resolve()
    attempts = {discovery, *controlled.values()}
    for source in attempts:
        require(not destination.is_relative_to(source) and not source.is_relative_to(destination), 'Output overlaps original evidence')
        completion = json.loads((source / 'completion.json').read_bytes())
        require(completion.get('status') == 'CAPTURE_COMPLETED' and completion.get('owned_container_removed') is True
                and completion.get('cleanup_errors') == [] and completion.get('device_lock_released') is True,
                'Wait for the entire attempt to complete cleanly: ' + str(source))
    destination.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    rows = []
    for item in plan:
        require(time.monotonic() - start < args.max_seconds, 'CPU comparison time budget exhausted; partial files retained')
        left, right = item['B'], item['F']
        try:
            report = comparison.analyze(left['case_directory'], right['case_directory'], left['completion'], right['completion'],
                                        left['step'], right['step'])
        except (ValueError, KeyError, TypeError, OSError) as error:
            report = {'schema': 'nes.independent_statistics_comparison.v1', 'status': 'INVALID_CAPTURE_EVIDENCE',
                'error': str(error), 'mechanism_accepted': False, 'physical_fault_location_proven': False}
        report_path = destination / (item['comparison_id'] + '.json')
        with report_path.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
        rows.append(summarize(item, report, report_path))
    summary = {'schema': 'nes.independent_statistics_matrix.v1', 'status': 'COMPARISONS_COMPLETE_NOT_MECHANISM_ACCEPTANCE',
        'group': args.group, 'planned_comparisons': len(plan), 'completed_comparisons': len(rows),
        'comparison_counts_by_kind': dict(Counter(row['kind'] for row in rows)),
        'outcome_counts': dict(Counter(row['observed_class'] for row in rows)),
        'status_counts': dict(Counter(row['status'] for row in rows)), 'comparisons': rows,
        'source_sha256': {'compare_matrix_statistics.py': sha(__file__), 'compare_replay_statistics.py': sha(comparison.__file__)},
        'elapsed_cpu_process_wall_seconds': time.monotonic() - start,
        'localization_inputs_only': ['config.json', 'dump.json', 'stack.json', 'construct.json'],
        'integrity_inputs_only': ['completion.json', 'receipt.json', 'collector_provenance.json', 'result.json status/case_id/receipt_sha256'],
        'not_opened': ['native host/hit/profile logs', 'private tensor files', 'step resource metadata'],
        'limits': ['Null fields remain unobservable; finite summaries can collide and do not establish tensor equality.',
            'Seed/offset or other auxiliary differences are reported separately, never promoted to primary attention faults.',
            'Stock/private source stacks may differ. No frames are dropped or rewritten to force alignment.',
            'Restore means no observed primary summary difference when aligned, not bitwise recovery or root-cause proof.'],
        'mechanism_accepted': False, 'physical_fault_location_proven': False, 'model_impact': None, 'native_calls': 0}
    with (destination / 'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': summary['status'], 'group': args.group, 'comparisons': len(rows),
        'outcomes': summary['outcome_counts'], 'summary_sha256': sha(destination / 'summary.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
