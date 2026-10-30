"""All 88 fixed comparisons on disclosed CPU fixtures, without framework imports."""
import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

import compare_matrix_statistics as matrix
import test_compare_replay_statistics as fixture

HERE = Path(__file__).resolve().parent


def main():
    checks = []
    temporary_parent = Path('\\\\?\\' + str(HERE)) if os.name == 'nt' else HERE
    with tempfile.TemporaryDirectory(prefix='matrix_fixture_', dir=temporary_parent) as directory:
        root = Path(directory)
        attempts = {'discovery': root / 'discovery'}
        attempts.update({target: root / target for target in matrix.TARGETS})
        for target, attempt in attempts.items():
            rows = []
            for batch in (1, 2, 4, 8) if target == 'discovery' else (1, 8):
                for repeat in (0, 1):
                    case = f'discovery_b{batch}_r{repeat}' if target == 'discovery' else f'controlled_{target}_b{batch}_r{repeat}'
                    steps = []
                    for step in range(1 if target == 'discovery' else 4):
                        dump, stack, construct = fixture.payload(batch=batch)
                        if target != 'discovery':
                            # Source mismatch is deliberate; the report must not erase it.
                            for entry in stack.values():
                                entry[1].append('File /audit/runtime/native_execution_strategy/session.py, line 100, in execute, \n value = invoke_original()')
                            if step in (1, 2):
                                dump['data'][fixture.API]['output'][0]['Mean'] = 0.26
                                dump['data'][fixture.MODULE]['output'][0]['Mean'] = 0.26
                        dump['data'][fixture.API]['output'][4]['Mean'] = 0.1 + repeat + step
                        steps.append((dump, stack, construct))
                    _, completion = fixture.capture(attempt, case, step_documents=steps)
                    rows.extend(json.loads(completion.read_bytes())['cases'])
            fixture.write(attempt / 'completion.json', {'status': 'CAPTURE_COMPLETED', 'owned_container_removed': True,
                'cleanup_errors': [], 'device_lock_released': True, 'cases': rows})
        plan = matrix.build_plan('all', attempts['discovery'], {t: attempts[t] for t in matrix.TARGETS})
        assert len(plan) == 88 and len({row['comparison_id'] for row in plan}) == 88
        checks.append('preregistered_88_unique_pairs_include_all_modes_repeats_and_stock_private')
        arguments = ['matrix', '--group', 'all', '--discovery-attempt', str(attempts['discovery']), '--output', str(root / 'reports')]
        for target in matrix.TARGETS:
            arguments += ['--controlled-attempt', target + '=' + str(attempts[target])]
        previous = sys.argv
        try:
            sys.argv = arguments
            with redirect_stdout(io.StringIO()):
                assert matrix.main() == 0
        finally:
            sys.argv = previous
        summary = json.loads((root / 'reports/summary.json').read_bytes())
        assert summary['completed_comparisons'] == 88 and summary['native_calls'] == 0
        assert summary['outcome_counts']['INCONCLUSIVE_OR_INVALID'] == 12
        assert all(row['alignment_complete'] is False and row['first_observed_api'] is None
                   for row in summary['comparisons'] if row['kind'] == 'stock_vs_private_native')
        checks.append('stock_private_full_source_stack_mismatch_preserved_no_forced_alignment')
        assert all(row['primary_attention_output_observed_difference'] is True
                   for row in summary['comparisons'] if row['kind'] == 'native_vs_controlled')
        checks.append('all_24_requested_control_comparisons_report_observed_fixture_primary_difference')
        assert all(row['statistical_restore_observed'] is True and row['tensor_equality_or_bitwise_restore_proven'] is False
                   for row in summary['comparisons'] if row['kind'] == 'off_restore')
        checks.append('all_12_off_restores_are_statistics_only_and_auxiliary_changes_preserved')
        assert all(row['auxiliary_output_observed_differences'] == ['seed']
                   for row in summary['comparisons'] if row['kind'] in ('stock_independent_repeat', 'same_process_matching_repeat'))
        checks.append('auxiliary_seed_variation_never_promoted_to_primary_attention_difference')
        assert all(matrix.sha(root / 'reports' / row['report_file']) == row['report_sha256'] for row in summary['comparisons'])
        checks.append('every_child_report_sha_bound')
        fake_report = copy.deepcopy(json.loads((root / 'reports' / next(row['report_file'] for row in summary['comparisons'] if row['kind'] == 'off_restore')).read_bytes()))
        for row in fake_report['comparisons']:
            if matrix.comparison.API.fullmatch(row['B']['name']):
                row['return_slots'][0]['unobservable'] = [{'reason': 'fixture_null'}]
        record = matrix.summarize({'comparison_id': 'null', 'kind': 'off_restore', 'B': {}, 'F': {}}, fake_report, root / 'reports/summary.json')
        assert record['statistical_restore_observed'] is None and record['primary_attention_unobservable_fields'] == 1
        checks.append('unobservable_primary_statistics_cannot_claim_restore')
    report = {'schema': 'nes.statistics_matrix.cpu_test.v1', 'status': 'PASS_CPU_FIXTURES',
        'checks': [{'name': name, 'passed': True} for name in checks], 'native_calls': 0,
        'framework_imports': False, 'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__), HERE / 'compare_matrix_statistics.py', HERE / 'compare_replay_statistics.py')}}
    path = HERE / 'matrix_statistics_cpu_checks.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps({'status': report['status'], 'checks': len(checks), 'sha256': matrix.sha(path)}))


if __name__ == '__main__':
    main()
