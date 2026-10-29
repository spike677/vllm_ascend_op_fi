"""Cross-check captured formal run values against prior diagnostic runs.

This reads runtime JSON only, not msprobe statistics or host selector logs.
It must never supply or overwrite the independent native msprobe first difference.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

DIAGNOSTIC_HF = '06c7832f94e76af31567f7f833b8a385ad3e1e4bd499bc76393db587c8c82e3b'
HIGH_HF = 'aecb9c0c8f961e545012f160b94b6ae159339f7e3316cb317bf8dcdc4cecda09'
HIGH_SELECTOR = '9f9d69750a1ba363c700ce7bb3d3bae1e3eab472179177ef77ac3c825965e9dd'
HIGH_PROCESSOR = '3aa7ede049bc6cb4d54a572a85f907214a98890a3c9544ed7f75e25df1306141'
FA_VENDOR = '63a5171d2031aa1a8ae5b37c8c9b488614779af7759841e224079a7e746d7b07'


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--formal-run', default='bf_statistics_a2')
    ap.add_argument('--compare-values-across-collector-revision', action='store_true',
                    help='Record and verify distinct frozen probe revisions; do not claim source equivalence')
    ap.add_argument('--stock-off-comparator', type=Path, required=True)
    ap.add_argument('--effect-comparator', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    util = load_module(args.stock_off_comparator, 'stock')
    effect = load_module(args.effect_comparator, 'effect')
    evidence = {}
    report = {'schema': 'fault28.formal_runtime_consistency.v1', 'status': 'FAIL_RUNTIME_CROSSCHECK',
              'scope': 'Runtime hash/metadata consistency only; independent msprobe inference not read or changed',
              'msprobe_inputs_read': False, 'host_control_inputs_read': False, 'first_fault_location_inferred': False,
              'compare_values_across_collector_revision': args.compare_values_across_collector_revision,
              'probe_source_equivalence_claimed': not args.compare_values_across_collector_revision,
              'errors': [], 'arms': [], 'input_sha256': evidence}
    for arm, diagnostic_run, diagnostic_case in [('B', 'private_off_compat_a2', 'fa_off_b8'),
                                                ('F', 'private_on_validation_a2', 'fa_on_b8')]:
        row = {'arm': arm, 'errors': [], 'forwards': []}
        report['arms'].append(row)
        try:
            formal = args.root / 'outputs' / args.formal_run / arm
            diagnostic = args.root / 'outputs' / diagnostic_run / diagnostic_case
            results, calls = [], []
            for folder in (diagnostic, formal):
                path = folder / 'result.json'
                result = util.read_json(path)
                evidence[str(path)] = util.sha_file(path)
                require(result['forward_count'] == 2, 'Expected two actual forwards')
                results.append(result)
                calls.append(util.load_calls(folder, result, evidence))
                source_path = folder.parent / 'source_manifest.json'
                source_manifest = util.read_json(source_path)
                evidence[str(source_path)] = util.sha_file(source_path)
                for name, expected in result['source_sha256'].items():
                    frozen = util.contained(folder.parent / 'source', name)
                    actual = util.sha_file(frozen)
                    require(source_manifest.get(name) == expected == actual, 'Recorded probe revision not bound to frozen source')
                    evidence[str(frozen)] = actual
            a, b = results
            require(a['capture_msprobe'] is False and b['capture_msprobe'] is True, 'Wrong capture modes')
            row['probe_source_sha256'] = {'diagnostic': a['source_sha256'], 'formal': b['source_sha256']}
            if args.compare_values_across_collector_revision:
                require(args.formal_run == 'bf_statistics_high_a1' and
                        a['source_sha256'] == {'hf_runtime_probe.py': DIAGNOSTIC_HF} and
                        b['source_sha256'] == {'hf_runtime_probe.py': HIGH_HF, 'collector_stage.py': HIGH_SELECTOR},
                        'Revision exception is restricted to the explicit collector integration')
                selection = b.get('statistics_collector', {})
                require(selection.get('variant') == 'task_local_fp32_high_v1' and selection.get('precision') == 'high'
                        and selection.get('modified_official_collector') is True and
                        selection.get('patched_processor_sha256') == selection.get('actual_loaded', {}).get('sha256') == HIGH_PROCESSOR,
                        'Collector-specific revision exception lacks actual task-local selection')
            else:
                require(a['source_sha256'] == b['source_sha256'], 'Diagnostic/formal probe source differs')
            for field in ('batch_size', 'prompt_length', 'target_index', 'layer_index', 'versions', 'native_sources',
                          'checkpoint_config', 'options', 'actual_sdpa_context_flags'):
                require(a[field] == b[field], 'Diagnostic/formal configuration differs: ' + field)
            require(a['private_vendor']['manifest_sha256'] == b['private_vendor']['manifest_sha256'], 'Vendor binary differs')
            if args.compare_values_across_collector_revision:
                require(b['private_vendor']['manifest_sha256'] == FA_VENDOR, 'Not the reviewed native FA vendor')
            for forward in (0, 1):
                ca, cb = (items[forward * 28:(forward + 1) * 28] for items in calls)
                comparisons = [util.compare_calls(x, y) for x, y in zip(ca, cb)]
                full_errors = effect.full_inputs_equal(util, ca[1], cb[1])
                logits = effect.compare_logits(util, a['forward_results'][forward], b['forward_results'][forward])
                errors = []
                if not all(c['target_inputs_and_output_equal'] for c in comparisons):
                    errors.append('Formal target layer input/output differs from diagnostic')
                errors.extend(full_errors)
                if not (logits['metadata_equal'] and logits['sha_equal'] and logits['argmax_equal']):
                    errors.append('Formal logits differ from diagnostic')
                row['forwards'].append({'forward_index': forward, 'status': 'PASS' if not errors else 'FAIL',
                                        'errors': errors, 'all_28_target_calls_equal': not any(c['differences'] for c in comparisons),
                                        'call_comparisons': comparisons, 'full_layer1_native_inputs_equal': not full_errors,
                                        'layer1_full_inputs_sha256': {k: v['logical_sha256'] if v else None
                                                                     for k, v in cb[1]['selected_layer_full_inputs_before'].items()},
                                        'layer1_target_output_sha256': cb[1]['target_output']['logical_sha256'],
                                        'logits': logits})
                row['errors'].extend(errors)
        except Exception as exc:
            row['errors'].append(type(exc).__name__ + ': ' + str(exc))
        row['status'] = 'PASS_RUNTIME_CROSSCHECK' if len(row['forwards']) == 2 and not row['errors'] else 'FAIL_RUNTIME_CROSSCHECK'
    require(report['arms'][0].get('probe_source_sha256', {}).get('formal') ==
            report['arms'][1].get('probe_source_sha256', {}).get('formal'), 'Formal B/F probe revisions differ')
    if all(row['status'] == 'PASS_RUNTIME_CROSSCHECK' for row in report['arms']):
        report['status'] = 'PASS_RUNTIME_CROSSCHECK'
    require(not args.output.exists(), 'Output must be fresh')
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'status': report['status'], 'output': str(args.output)}))
    return 0 if report['status'] == 'PASS_RUNTIME_CROSSCHECK' else 1


if __name__ == '__main__':
    raise SystemExit(main())
