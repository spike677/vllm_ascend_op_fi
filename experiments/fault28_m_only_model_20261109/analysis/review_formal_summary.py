"""Recompute formal msprobe inference, then separately review sealed output hashes.

No host log, private tensor, model execution, network or device operation is used.
CLI: --inputs existing_acceptance_suite.json [--acceptance acceptance.json]
     --output NEW_REVIEW.json
The two cases must be locally available with their original relative layout.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from audit_model_coverage import audit
from compare_model_statistics import RUN_PAIRS, compare_phase
from model_statistics_io import child, load_capture, read, require, sha

HERE = Path(__file__).resolve().parent
SCHEMA = 'nes.model_coverage.local_formal_review.v1'
HASH = re.compile('[0-9a-f]{64}')
RESTORE = {'calibration_vs_native', 'fault_repeat', 'native_vs_off'}


def local_path(value, base=None):
    path = Path(value)
    if not path.is_absolute() and base is not None:
        path = base / path
    path = path.absolute()
    if os.name == 'nt' and not str(path).startswith('\\\\?\\'):
        path = Path('\\\\?\\' + str(path))
    return path.resolve(strict=True)


def source_hashes(names):
    return {name: sha(HERE / name) for name in names}


def snapshot(value, batch, shape=None):
    require(isinstance(value, dict) and HASH.fullmatch(value.get('full_sha256', '')),
            'Missing complete logical tensor SHA')
    rows = value.get('per_batch_sha256')
    require(isinstance(rows, list) and len(rows) == batch
            and all(isinstance(x, str) and HASH.fullmatch(x) for x in rows),
            'Missing independently mapped per-request tensor SHA')
    require(isinstance(value.get('shape'), list) and value['shape'][0] == batch,
            'Snapshot batch geometry differs')
    if shape is not None:
        require(value['shape'] == shape and value.get('dtype') == 'torch.float16',
                'Snapshot differs from original FP16 SDPA geometry')
    return value


def changed_slots(left, right):
    require(left['shape'] == right['shape'] and left['dtype'] == right['dtype'],
            'Hash comparison tensor geometry/dtype differs')
    require(len(left['per_batch_sha256']) == len(right['per_batch_sha256']), 'Row hash count differs')
    slots = [i for i, (a, b) in enumerate(zip(left['per_batch_sha256'], right['per_batch_sha256'])) if a != b]
    require((left['full_sha256'] != right['full_sha256']) == bool(slots),
            'Whole-tensor and row hashes contradict')
    return slots


def verified_statistics(loaded, summary_path):
    """Freeze the inference before opening any runtime output-hash sidecars."""
    report, digest = read(summary_path)
    require(report.get('analysis_source_sha256') == source_hashes(
        ('compare_model_statistics.py', 'model_statistics_io.py', 'statistics_values.py')),
        'Independent statistics analyzer source differs')
    require(report.get('schema') == 'nes.model_coverage.independent_statistics.v1'
            and report.get('status') == 'FORMAL_COMPARISONS_COMPLETE'
            and report.get('case_identity') == loaded['identity']
            and report.get('evidence_sha256') == loaded['hashes']
            and report.get('mechanism_accepted') is False, 'Statistics report is not bound to original evidence')
    require(report.get('run_comparison_count') == 5 and report.get('phase_comparison_count') == 10,
            'Formal comparison matrix incomplete')
    expected_pairs = {(a * 2 + s, b * 2 + s) for a, b, _ in RUN_PAIRS for s in (0, 1)}
    actual, report_refs = {}, []
    for ref in report['phase_reports']:
        pair = (ref['B_step'], ref['F_step'])
        require(pair in expected_pairs and pair not in actual, 'Unknown or duplicate phase comparison')
        path = child(summary_path.parent, ref['file'])
        saved, observed = read(path)
        require(observed == ref['sha256'], 'Phase statistics report SHA differs')
        recalculated = compare_phase(loaded, *pair)
        require(saved == recalculated, 'Phase inference differs from independent raw recomputation')
        actual[pair] = recalculated
        report_refs.append({'file': ref['file'], 'sha256': observed})
    require(set(actual) == expected_pairs, 'Missing phase comparison')
    require(len(report['runs']) == 5, 'Missing run summary')
    runs = []
    for saved_run, (a, b, name) in zip(report['runs'], RUN_PAIRS):
        phases = [actual[a * 2 + s, b * 2 + s] for s in (0, 1)]
        aligned = all(x['alignment_complete'] for x in phases)
        first = next(({'stage': x['stage'], 'B_step': x['B_step'], 'F_step': x['F_step'],
            'boundary': x['first_difference_boundary'], 'node': x['first_observed_model_api_difference']}
            for x in phases if x['first_observed_model_api_difference']), None) if aligned else None
        require(saved_run['comparison_id'] == name and saved_run['B_run_index'] == a
                and saved_run['F_run_index'] == b and saved_run['alignment_complete'] == aligned
                and saved_run['first_observed_model_api_difference'] == first,
                'Global first difference summary disagrees with raw phase order')
        runs.append({'comparison_id': name, 'alignment_complete': aligned,
            'first_observed_model_api_difference': first,
            'phase_unobservable_fields': {x['stage']: x['unobservable_fields'] for x in phases}})
    return actual, runs, {'summary_sha256': digest, 'phase_reports': report_refs}


def verified_hash_calls(root, loaded, framework):
    result, _ = read(root / 'result.json')
    refs = result.get('calls', [])
    require(len(refs) == 280 and result.get('original_api_calls') == 280,
            'Require ten full-model phases, not an isolated attention replay')
    expected_files = {x['file'] for x in refs}
    require(len(expected_files) == 280 and expected_files ==
            {p.relative_to(root).as_posix() for p in (root / 'native_calls').glob('call_*/call.json')},
            'Missing/extra call sidecar')
    batch = framework['calls'][0]['request_count']
    calls, fingerprints = {}, {}
    for ref, denominator in zip(refs, framework['calls']):
        call, digest = read(child(root, ref['file']))
        step, layer = denominator['step'], denominator['layer_index']
        phase = loaded['steps'][step]['phase']
        require(digest == ref['sha256'] and call.get('status') == 'CAPTURE_COMPLETED'
                and call.get('step') == ref['step'] == step and call.get('layer_index') == ref['layer_index'] == layer
                and call.get('call_id') == ref['call_id'] == phase['sdpa_call_ids'][layer]
                and call.get('module_call_id') == denominator['module_call_id']
                and call.get('request_ids') == denominator['request_ids']
                and call.get('request_mapping_file') == phase['request_mapping_file']
                and call.get('request_mapping_sha256') == phase['request_mapping_sha256'],
                'Call hashes are not bound to the independent framework/request denominator')
        for name in ('query', 'key', 'value'):
            shape = denominator[name + '_shape']
            before = snapshot(call['inputs_before'][name], batch, shape)
            after = snapshot(call['inputs_after'][name], batch, shape)
            require(not changed_slots(before, after), 'Original SDPA modified logical input values')
        snapshot(call['output'], batch, denominator['query_shape'])
        calls[step, layer] = call
        fingerprints[ref['file']] = digest
    for step, capture in loaded['steps'].items():
        logits = snapshot(capture['phase']['logits'], batch)
        require(len(logits['shape']) == 3 and logits['shape'][1] == 1, 'Expected complete last-token logits per request')
    return calls, fingerprints


def review_case(spec):
    root = local_path(spec['case_dir'])
    loaded = load_capture(root, local_path(spec.get('completion', root.parent / 'completion.json')))
    require(len(loaded['steps']) == 10, 'This review requires formal data; pilot is only schema admission')
    framework = audit(loaded, root / 'model_config.json')
    saved, framework_sha = read(local_path(spec['framework_report']))
    require(saved.get('analysis_source_sha256') == source_hashes(
        ('audit_model_coverage.py', 'model_statistics_io.py', 'statistics_values.py')), 'Framework analyzer source differs')
    for key in ('schema', 'status', 'case_identity', 'evidence_sha256', 'calls', 'phases',
                'framework_sdpa_calls', 'fault_prefill_shape_candidates', 'fault_decode_unsupported_calls',
                'fault_prefill_layer_request_pairs'):
        require(saved.get(key) == framework[key], 'Framework raw recomputation differs: ' + key)
    # This call completes the source-only statistical inference before hash files are opened.
    phases, runs, report_binding = verified_statistics(loaded, local_path(spec['statistics_report']))
    calls, call_sha = verified_hash_calls(root, loaded, framework)
    batch = framework['calls'][0]['request_count']
    request_ids = framework['calls'][0]['request_ids']
    reviews, concerns = [], []
    for a, b, name in RUN_PAIRS:
        for offset in (0, 1):
            left, right = a * 2 + offset, b * 2 + offset
            statistical = phases[left, right]
            layers, collisions, statistic_only_changes = [], [], []
            for layer in range(28):
                before, after = calls[left, layer], calls[right, layer]
                changed_inputs = [k for k in ('query', 'key', 'value')
                                  if changed_slots(before['inputs_before'][k], after['inputs_before'][k])]
                slots = changed_slots(before['output'], after['output'])
                observed = statistical['layers'][layer]
                if slots and observed['output_difference'] is False:
                    collisions.append({'layer_index': layer, 'changed_request_slots': slots,
                        'classification': 'UNOBSERVABLE_OR_SUMMARY_COLLISION' if observed['output_unobservable_fields'] else 'SUMMARY_COLLISION'})
                if not slots and observed['output_difference'] is True:
                    statistic_only_changes.append(layer)
                layers.append({'layer_index': layer, 'changed_input_names': changed_inputs,
                    'changed_output_request_slots': slots, 'changed_output_request_ids': [request_ids[i] for i in slots],
                    'statistics_classification': observed['classification'],
                    'statistics_input_unobservable_fields': observed['input_unobservable_fields'],
                    'statistics_output_unobservable_fields': observed['output_unobservable_fields']})
            logits_slots = changed_slots(loaded['steps'][left]['phase']['logits'], loaded['steps'][right]['phase']['logits'])
            stable = not logits_slots and not any(x['changed_input_names'] or x['changed_output_request_slots'] for x in layers)
            if name in RESTORE and not stable:
                concerns.append(name + ':' + statistical['stage'] + ':RESTORATION_OR_REPEAT_HASH_DIFFERENCE')
            if name != 'calibration_vs_native' and not statistical['alignment_complete']:
                concerns.append(name + ':' + statistical['stage'] + ':SOURCE_ALIGNMENT_UNKNOWN')
            reviews.append({'comparison_id': name, 'stage': statistical['stage'], 'B_step': left, 'F_step': right,
                'alignment_complete': statistical['alignment_complete'], 'layers': layers,
                'statistical_summary_collisions': collisions,
                'statistical_difference_with_equal_output_hash_layers': statistic_only_changes,
                'logits_changed_request_slots': logits_slots, 'logits_changed_request_ids': [request_ids[i] for i in logits_slots],
                'restoration_or_repeat_required': name in RESTORE, 'all_input_output_logits_hashes_match': stable,
                'complete_private_logits_bytes_rechecked': False})
    return {'case_id': root.name, 'batch_size': batch, 'request_ids': request_ids,
        'case_identity': loaded['identity'], 'evidence_sha256': loaded['hashes'],
        'framework_report_sha256': framework_sha, 'statistics_report_binding': report_binding,
        'call_metadata_sha256': call_sha, 'collector_disclosure': loaded['collector'],
        'framework_calls': framework['framework_sdpa_calls'],
        'fault_prefill_candidates': framework['fault_prefill_shape_candidates'],
        'fault_prefill_request_pairs': framework['fault_prefill_layer_request_pairs'],
        'source_only_statistical_inference': runs, 'separate_runtime_hash_review': reviews, 'concerns': concerns}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--acceptance', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    suite_path = local_path(args.inputs)
    suite, suite_sha = read(suite_path)
    require(suite.get('schema') == 'nes.model_coverage.acceptance_inputs.v1' and len(suite.get('cases', [])) == 2,
            'Require formal B1/B8 local suite')
    specs = [{k: local_path(v, suite_path.parent) for k, v in row.items()} for row in suite['cases']]
    output = args.output.absolute()
    if os.name == 'nt' and not str(output).startswith('\\\\?\\'):
        output = Path('\\\\?\\' + str(output))
    require(not output.exists() and all(not output.resolve().is_relative_to(x['case_dir']) for x in specs),
            'Review must be a new report outside original cases')
    try:
        cases = [review_case(spec) for spec in specs]
        require(sorted(x['batch_size'] for x in cases) == [1, 8], 'Missing distinct B1/B8 formal cases')
        one, eight = sorted(cases, key=lambda x: x['batch_size'])
        require(one['request_ids'] == eight['request_ids'][:1], 'Target request is not stable across B1/B8')
        for key in ('runtime_manifest_sha256', 'checkpoint_manifest_sha256', 'tokenizer_manifest_sha256', 'source_fingerprints_sha256'):
            require(one['case_identity'][key] == eight['case_identity'][key], 'Mixed execution generation: ' + key)
        acceptance = None
        if args.acceptance:
            accepted, accepted_sha = read(local_path(args.acceptance))
            indexed = {x['case_id']: x for x in accepted['cases']}
            require(set(indexed) == {x['case_id'] for x in cases}, 'Acceptance case set differs')
            for row in cases:
                linked = indexed[row['case_id']]
                require(linked['framework_report_sha256'] == row['framework_report_sha256']
                    and linked['completion_sha256'] == row['evidence_sha256']['completion']
                    and linked['independent_msprobe_inference']['report_sha256'] == row['statistics_report_binding']['summary_sha256'],
                    'Acceptance links different source/statistical evidence')
            acceptance = {'sha256': accepted_sha, 'reported_status': accepted['status'],
                'reported_formal_coverage_accepted': accepted['formal_coverage_accepted'],
                'native_execution_revalidated_by_this_review': False}
        concerns = [f"{x['case_id']}:{issue}" for x in cases for issue in x['concerns']]
        result = {'schema': SCHEMA, 'status': 'REVIEW_COMPLETE_WITH_CONCERNS' if concerns else 'PASS_LOCAL_FORMAL_STATISTICS_AND_HASH_REVIEW',
            'cases': cases, 'concerns': concerns, 'bound_external_acceptance': acceptance,
            'statistics_phase_comparisons_recomputed': 20, 'framework_calls_recounted': 560,
            'limits': ['First differences are recomputed only from original statistics, stack and construct.',
                'Hashes are reviewed afterwards; they never select or replace the statistical first difference.',
                'Whole-batch summaries may collide and cannot attribute statistical differences to a request.',
                'Per-request SHA changes describe logical-value differences, not accuracy loss.',
                'No private logits or tensor bytes, native host hit data, raw native windows or whole launch log are read.',
                'This local metadata review does not repeat native raw/POD/certificate execution acceptance.'],
            'native_execution_accepted_by_this_review': False, 'model_accuracy_degradation_proven': False}
    except (ValueError, KeyError, TypeError, OSError) as error:
        result = {'schema': SCHEMA, 'status': 'FAIL_OR_INCOMPLETE_LOCAL_REVIEW', 'error': str(error)}
    result.update(inputs_sha256=suite_sha, analysis_source_sha256=source_hashes(
        ('review_formal_summary.py', 'audit_model_coverage.py', 'compare_model_statistics.py', 'model_statistics_io.py', 'statistics_values.py')),
        network_calls=0, native_calls=0, private_tensor_files_opened=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'sha256': sha(output), 'error': result.get('error')}))
    return 0 if result['status'] == 'PASS_LOCAL_FORMAL_STATISTICS_AND_HASH_REVIEW' else 1


if __name__ == '__main__':
    raise SystemExit(main())
