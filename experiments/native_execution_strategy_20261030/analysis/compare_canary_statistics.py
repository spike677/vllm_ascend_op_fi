"""Separate four-pair canary report using the unchanged independent comparator."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import compare_replay_statistics as comparison
from compare_matrix_statistics import sha, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    attempt = args.attempt.resolve(strict=True)
    case = (attempt / args.case_id).resolve(strict=True)
    if case.parent != attempt or case.name != args.case_id:
        raise ValueError('Case must be a direct child of the supplied attempt')
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(attempt) or attempt.is_relative_to(output):
        raise ValueError('Output must be fresh and separate from original evidence')
    completion = attempt / 'completion.json'
    loaded = comparison.load(case, completion, 0)
    if json.loads((case / 'config.json').read_bytes()).get('step') != [0, 1, 2, 3]:
        raise ValueError('This canary requires exactly four registered capture steps')
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    start = time.monotonic()
    for left, right in ((0, 1), (0, 2), (0, 3), (1, 2)):
        if time.monotonic() - start > 120:
            raise ValueError('Bounded CPU comparison time exhausted; partial reports retained')
        item = {'comparison_id': args.case_id + f'_s{left}_vs_s{right}',
            'kind': 'off_restore' if right == 3 else 'same_process_matching_repeat' if left == 1 else 'native_vs_controlled',
            'B': {'case_directory': str(case), 'completion': str(completion), 'step': left},
            'F': {'case_directory': str(case), 'completion': str(completion), 'step': right}}
        report = comparison.analyze(case, case, completion, completion, left, right)
        path = output / (item['comparison_id'] + '.json')
        with path.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
        rows.append(summarize(item, report, path))
    summary = {'schema': 'nes.independent_statistics_canary.v1',
        'status': 'FOUR_COMPARISONS_COMPLETE_NOT_MECHANISM_ACCEPTANCE',
        'scope': 'Separate validated-alias canary; excluded from the formal 20-case/56-call/88-pair matrix',
        'case_id': args.case_id, 'comparisons': rows, 'comparison_count': len(rows),
        'outcomes': dict(Counter(row['observed_class'] for row in rows)),
        'collector_disclosure': loaded['collector'],
        'source_sha256': {'compare_canary_statistics.py': sha(__file__),
            'compare_replay_statistics.py': sha(comparison.__file__),
            'compare_matrix_statistics.py': sha(Path(__file__).with_name('compare_matrix_statistics.py'))},
        'localization_inputs_only': ['config.json', 'dump.json', 'stack.json', 'construct.json'],
        'not_opened': ['host/profile/hit logs', 'private tensor files', 'policy intent', 'resource measurements'],
        'limits': ['Summaries do not establish tensor equality or physical fault location.',
            'Auxiliary output differences and unobservable fields are retained separately.',
            'This report does not validate the alias, runtime manifest or certificate itself.'],
        'mechanism_accepted': False, 'physical_fault_location_proven': False, 'model_impact': None, 'native_calls': 0}
    path = output / 'summary.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': summary['status'], 'comparison_count': len(rows),
        'outcomes': summary['outcomes'], 'summary_sha256': sha(path)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
