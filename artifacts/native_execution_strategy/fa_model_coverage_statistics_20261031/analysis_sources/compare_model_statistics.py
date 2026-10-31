"""Source-only full-model statistical first difference, with propagation labels."""
import argparse
from collections import Counter
import json
from pathlib import Path

from model_statistics_io import SDPA, load_capture, require, sha
from statistics_values import compare_values

RUN_PAIRS = ((0, 1, 'calibration_vs_native'), (1, 2, 'native_vs_fault_a'),
             (1, 3, 'native_vs_fault_b'), (2, 3, 'fault_repeat'), (1, 4, 'native_vs_off'))


def public(entry):
    return {k: entry[k] for k in ('name', 'ordinal', 'parents', 'stack', 'layer_index')}


def compare_phase(loaded, left_step, right_step):
    left, right = loaded['steps'][left_step], loaded['steps'][right_step]
    require(left['phase']['stage'] == right['phase']['stage'], 'Do not align prefill and decode as one operation')
    a = [entry for entry in left['entries'] if not entry['diagnostic']]
    b = [entry for entry in right['entries'] if not entry['diagnostic']]
    trace_equal = not left['issues'] and not right['issues'] and [x['key'] for x in a] == [x['key'] for x in b]
    metadata_mismatches, rows, changed_modules = [], [], []
    total_unobservable = 0
    for before, after in zip(a, b):
        if before['key'] != after['key']:
            continue
        inputs = compare_values({k: before['node'].get(k) for k in ('input_args', 'input_kwargs', 'parameters')},
                                {k: after['node'].get(k) for k in ('input_args', 'input_kwargs', 'parameters')})
        outputs = compare_values(before['node'].get('output'), after['node'].get('output'))
        total_unobservable += len(inputs['unobservable']) + len(outputs['unobservable'])
        if inputs['metadata_differences'] or outputs['metadata_differences']:
            metadata_mismatches.append({'node': before['name'], 'input': inputs['metadata_differences'], 'output': outputs['metadata_differences']})
        row = {'B': public(before), 'F': public(after), 'input': inputs, 'output': outputs}
        if before['name'].startswith('Module.'):
            if inputs['differences'] or outputs['differences']:
                changed_modules.append(before['name'])
        else:
            rows.append(row)
    aligned = trace_equal and not metadata_mismatches
    changed = [row for row in rows if row['input']['differences'] or row['output']['differences']]
    first = changed[0] if aligned and changed else None
    sdpa_rows = [row for row in rows if SDPA.fullmatch(row['B']['name'])]
    layers = []
    by_layer = {row['B']['layer_index']: row for row in sdpa_rows}
    for layer in range(28):
        row = by_layer.get(layer)
        observable = aligned and row is not None
        input_diff = bool(row['input']['differences']) if observable else None
        output_diff = bool(row['output']['differences']) if observable else None
        if not observable:
            label = 'INCONCLUSIVE_ALIGNMENT_OR_METADATA'
        elif input_diff:
            label = 'INPUT_ALREADY_DIFFERENT_WITH_OUTPUT_DIFFERENCE' if output_diff else 'INPUT_ALREADY_DIFFERENT_WITH_NO_OBSERVED_OUTPUT_DIFFERENCE'
        elif output_diff:
            label = 'OUTPUT_DIFFERENCE_WITH_NO_OBSERVED_INPUT_STATISTICAL_DIFFERENCE'
        else:
            label = 'NO_OBSERVED_SDPA_STATISTICAL_DIFFERENCE'
        layers.append({'layer_index': layer, 'classification': label,
            'input_already_different': input_diff, 'output_difference': output_diff,
            'input_unobservable_fields': len(row['input']['unobservable']) if row else None,
            'output_unobservable_fields': len(row['output']['unobservable']) if row else None,
            'comparison': row, 'independent_fault_origin_proven': False,
            'per_request_attention_difference': 'UNOBSERVABLE_FROM_BATCH_SUMMARIES'})
    if not aligned:
        status = 'INCONCLUSIVE_ALIGNMENT_OR_METADATA'
    elif first:
        status = 'ALIGNED_OBSERVED_MODEL_API_STATISTICAL_DIFFERENCE'
    elif total_unobservable:
        status = 'ALIGNED_NO_OBSERVED_DIFFERENCE_WITH_UNOBSERVABLE_FIELDS'
    else:
        status = 'ALIGNED_AVAILABLE_MODEL_STATISTICS_MATCH'
    return {'schema': 'nes.model_coverage.phase_statistics.v1', 'status': status,
        'B_step': left_step, 'F_step': right_step, 'stage': left['phase']['stage'],
        'alignment_complete': aligned, 'full_model_source_trace_match': trace_equal,
        'metadata_mismatches': metadata_mismatches, 'B_issues': left['issues'], 'F_issues': right['issues'],
        'raw_node_counts': {'B': len(left['entries']), 'F': len(right['entries'])},
        'model_node_counts': {'B': len(a), 'F': len(b)},
        'diagnostic_classification': {'source': '/audit/model_helpers.py', 'function': 'tensor_snapshot',
            'B_nodes': [x['name'] for x in left['entries'] if x['diagnostic']],
            'F_nodes': [x['name'] for x in right['entries'] if x['diagnostic']],
            'raw_files_modified': False, 'real_sdpa_excluded': False},
        'first_observed_model_api_difference': first,
        'first_difference_boundary': ('input' if first['input']['differences'] else 'output') if first else None,
        'changed_api_nodes': [{'name': row['B']['name'], 'ordinal': row['B']['ordinal'],
            'layer_index': row['B']['layer_index'], 'input_changed': bool(row['input']['differences']),
            'output_changed': bool(row['output']['differences'])} for row in changed],
        'changed_module_nodes': changed_modules, 'unobservable_fields': total_unobservable,
        'layers': layers, 'tensor_equality_proven': False, 'physical_fault_location_proven': False,
        'mechanism_accepted': False}


def analyze(loaded, destination):
    require(len(loaded['steps']) in (2, 10), 'Unexpected registered model steps')
    destination = Path(destination)
    require(not destination.exists() and not destination.resolve().is_relative_to(loaded['root'])
            and not loaded['root'].is_relative_to(destination.resolve()), 'Fresh separate report directory required')
    destination.mkdir(parents=True, exist_ok=False)
    run_reports, phase_report_index = [], []
    pairs = RUN_PAIRS if len(loaded['steps']) == 10 else ()
    for before, after, name in pairs:
        phases = []
        for phase in (0, 1):
            report = compare_phase(loaded, before * 2 + phase, after * 2 + phase)
            file = destination / (name + '_' + report['stage'] + '.json')
            with file.open('x', encoding='utf-8') as stream:
                json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
            record = {'file': file.name, 'sha256': sha(file), 'B_step': report['B_step'], 'F_step': report['F_step'],
                'stage': report['stage'], 'status': report['status'], 'alignment_complete': report['alignment_complete'],
                'first_observed_model_api': report['first_observed_model_api_difference']['B']['name'] if report['first_observed_model_api_difference'] else None,
                'first_boundary': report['first_difference_boundary'],
                'layer_classifications': dict(Counter(row['classification'] for row in report['layers'])),
                'unobservable_fields': report['unobservable_fields']}
            phases.append((report, record)); phase_report_index.append(record)
        complete = all(report['alignment_complete'] for report, _ in phases)
        first = next(({'stage': report['stage'], 'B_step': report['B_step'], 'F_step': report['F_step'],
                       'boundary': report['first_difference_boundary'], 'node': report['first_observed_model_api_difference']}
                      for report, _ in phases if report['first_observed_model_api_difference']), None) if complete else None
        run_reports.append({'comparison_id': name, 'B_run_index': before, 'F_run_index': after,
            'alignment_complete': complete, 'phases': [ref for _, ref in phases],
            'first_observed_model_api_difference': first,
            'interpretation': 'Ordered prefill then fixed teacher-forced decode; one global first difference only when both complete traces align.',
            'tensor_equality_proven': False, 'physical_fault_location_proven': False})
    return {'schema': 'nes.model_coverage.independent_statistics.v1',
        'status': 'FORMAL_COMPARISONS_COMPLETE' if pairs else 'PILOT_CAPTURE_SCHEMA_VERIFIED_NO_COMPARATIVE_RUN',
        'case_identity': loaded['identity'], 'evidence_files': loaded['files'], 'evidence_sha256': loaded['hashes'],
        'collector_disclosure': loaded['collector'], 'run_comparison_count': len(run_reports),
        'phase_comparison_count': len(phase_report_index), 'runs': run_reports, 'phase_reports': phase_report_index,
        'localization_inputs_only': ['config.json', 'original dump.json', 'original stack.json', 'original construct.json'],
        'integrity_inputs': ['result status/receipt/phase and source bindings', 'completion', 'receipt', 'collector provenance', 'phase manifest'],
        'not_opened': ['native_calls per-call files', 'host/hit/profile logs', 'private tensors', 'policy/Session decisions'],
        'limits': ['Whole-batch statistics cannot locate which request changed or prove tensor equality.',
            'Later layers can inherit differences from earlier layers; a changed layer is not an independently established origin.',
            'Calibration/manual-scope and Session stacks can differ and are never forced into alignment.',
            'Diagnostic classification is restricted to the reviewed tensor_snapshot function; original SDPA records are preserved.',
            'Native launch coverage, actual strategy switches and model output hashes require separate evidence.'],
        'mechanism_accepted': False, 'physical_fault_location_proven': False, 'native_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--completion', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(load_capture(args.case_dir, args.completion), args.output)
    report['analysis_source_sha256'] = {p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name('model_statistics_io.py'), Path(__file__).with_name('statistics_values.py'))}
    file = args.output / 'summary.json'
    with file.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': report['status'], 'phase_comparisons': report['phase_comparison_count'], 'sha256': sha(file)}))


if __name__ == '__main__':
    main()
