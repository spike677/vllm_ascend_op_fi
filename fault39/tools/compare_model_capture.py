"""Compare complete B/F capture groups using raw msprobe artifacts only.

Each group contains config.json, collector_identity.json and
observation/stepN/rankR/{dump,stack,construct}.json. Native validation reports,
target identities, generated response records and full tensors are not read.
"""
import argparse
import json
from pathlib import Path
import re

from compare_msprobe import ComparisonPolicy, compare

STEP_NAME = re.compile(r'step(0|[1-9][0-9]*)\Z')
DIFFERENCE = {'FIRST_STATISTICS_DIFFERENCE', 'FIRST_OBSERVABLE_STATISTICS_DIFFERENCE'}
EQUAL = {'NO_STATISTICS_DIFFERENCE', 'NO_OBSERVABLE_STATISTICS_DIFFERENCE'}


def actual_steps(group):
    """Use numeric step order, without renaming or fabricating missing steps."""
    observation = Path(group) / 'observation'
    if not observation.is_dir():
        raise ValueError('OBSERVATION_DIRECTORY_MISSING')
    steps = {}
    for entry in observation.iterdir():
        if not entry.is_dir() or not entry.name.startswith('step'):
            continue
        matched = STEP_NAME.fullmatch(entry.name)
        if not matched:
            raise ValueError('INVALID_STEP_DIRECTORY: ' + entry.name)
        steps[int(matched.group(1))] = entry
    return dict(sorted(steps.items()))


def compare_groups(baseline, fault, policy=ComparisonPolicy(1e-8, 1e-5), rank=0):
    if type(rank) is not int or rank < 0:
        raise ValueError('NONNEGATIVE_RANK_REQUIRED')
    baseline, fault = Path(baseline), Path(fault)
    report = {
        'schema_version': 1, 'source': 'RAW_MSPROBE_STATISTICS_MIX',
        'native_logs_read': False, 'private_tensors_read': False,
        'generated_responses_read': False, 'request_mapping': 'UNPROVEN_REQUEST_MAPPING',
        'rank': rank, 'step_semantics': 'Actual collector steps in numeric order; not inferred token numbers',
        'comparison_policy': {'rtol': policy.relative_tolerance, 'atol': policy.absolute_tolerance,
                              'rule': 'abs(F-B) > atol + rtol*abs(B)'},
        'limitations': [
            'Null statistics remain unobservable; equality of hidden values is not established.',
            'Later generated inputs can differ between B/F; this comparator does not infer their cause.',
            'Step numbers and matching stacks do not independently prove request identity.',
            'Statistical first difference does not uniquely establish a physical fault mechanism.',
        ],
        'steps': [], 'first_difference': None, 'blocking_steps': [],
    }
    try:
        bsteps, fsteps = actual_steps(baseline), actual_steps(fault)
    except (OSError, ValueError) as error:
        return dict(report, status='INVALID_CAPTURE_LAYOUT', error=str(error))
    report.update(baseline_steps=list(bsteps), fault_steps=list(fsteps))
    mismatch = set(bsteps) != set(fsteps)
    report['missing_from_baseline'] = sorted(set(fsteps) - set(bsteps))
    report['missing_from_fault'] = sorted(set(bsteps) - set(fsteps))
    earliest = None
    prior_blocked = False
    any_unobservable = False
    for step in sorted(set(bsteps) | set(fsteps)):
        item = {'step': step, 'relative_path': f'observation/step{step}/rank{rank}'}
        if step not in bsteps or step not in fsteps:
            result = {'status': 'STEP_MISSING', 'missing_group': 'baseline' if step not in bsteps else 'fault'}
        else:
            try:
                result = compare(bsteps[step] / f'rank{rank}', fsteps[step] / f'rank{rank}', policy,
                    baseline_config=baseline / 'config.json', fault_config=fault / 'config.json',
                    baseline_collector=baseline / 'collector_identity.json',
                    fault_collector=fault / 'collector_identity.json')
            except (OSError, ValueError, TypeError, KeyError) as error:
                result = {'status': 'INVALID_STEP_EVIDENCE', 'error_type': type(error).__name__, 'error': str(error)}
        item['report'] = result
        report['steps'].append(item)
        comparison = result.get('comparison', {})
        status = result.get('status', comparison.get('status'))
        item['status'] = status
        any_unobservable |= bool(comparison.get('unobservable_fields'))
        nan_incomparable = comparison.get('difference', {}).get('reason') == 'NONFINITE_UNCOMPARABLE'
        if status not in DIFFERENCE | EQUAL or nan_incomparable:
            report['blocking_steps'].append(step)
            if earliest is None:
                prior_blocked = True
        elif status in DIFFERENCE and earliest is None:
            earliest = {'step': step, 'relative_path': item['relative_path'],
                'identity': comparison['identity'], 'record_index': comparison['index'],
                'difference': comparison['difference'],
                'earlier_steps_comparable': not prior_blocked,
                'unobservable_fields_before_or_at_first': comparison.get('unobservable_fields', [])}
    report['all_steps_aligned'] = bool(bsteps) and not mismatch and not report['blocking_steps']
    if mismatch:
        report['status'] = 'STEP_SET_MISMATCH'
        report['first_difference'] = None
    elif not bsteps:
        report['status'] = 'INSUFFICIENT_STATISTICS'
    elif earliest is not None and not earliest['earlier_steps_comparable']:
        report['status'] = 'EARLIEST_DIFFERENCE_UNPROVEN'
        report['later_observed_difference'] = earliest
    elif earliest is not None:
        report['status'] = 'FIRST_OBSERVABLE_STATISTICS_DIFFERENCE'
        report['first_difference'] = earliest
    elif report['blocking_steps']:
        report['status'] = 'INCOMPLETE_COMPARISON'
    else:
        report['status'] = 'NO_OBSERVABLE_STATISTICS_DIFFERENCE' if any_unobservable else 'NO_STATISTICS_DIFFERENCE'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--fault', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--atol', type=float, default=1e-8)
    args = parser.parse_args()
    report = compare_groups(args.baseline, args.fault, ComparisonPolicy(args.atol, args.rtol), args.rank)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    summary = {key: report.get(key) for key in
        ('status', 'baseline_steps', 'fault_steps', 'all_steps_aligned', 'blocking_steps')}
    first = report.get('first_difference')
    summary['first_difference'] = {key: first[key] for key in
        ('step', 'relative_path', 'identity', 'record_index', 'difference')} if first else None
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
