"""Recompute the small a5 stock FIA calibration summary from included JSON only."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = ('stock_b1', 'stock_b1_repeat', 'stock_b8', 'stock_b8_repeat')


def read(relative):
    return json.loads((ROOT / relative).read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(record):
    return {key: record[key] for key in ('shape', 'dtype', 'logical_bytes', 'logical_sha256', 'raw_order', 'byteorder')}


def strip_phase(value):
    if isinstance(value, dict):
        return {k: strip_phase(v) for k, v in value.items()
                if k not in ('readback_phase', 'private_raw_file', 'publication')}
    if isinstance(value, list):
        return [strip_phase(v) for v in value]
    return value


def build():
    completion = read('raw/completion.json')
    assert completion['status'] == 'CAPTURE_COMPLETED' and completion['owned_container_removed']
    assert completion['device_lock_released'] and not completion['cleanup_errors']
    source_manifest = read('raw/source_manifest.json')
    for path in (ROOT / 'raw/source').rglob('*'):
        if path.is_file():
            assert sha(path) == source_manifest[str(path.relative_to(ROOT / 'raw/source')).replace('\\', '/')]
    case_summaries, results, calls, logits = {}, {}, {}, {}
    for case in CASES:
        base = 'raw/' + case + '/'
        result, receipt, scheduler = [read(base + name + '.json') for name in ('result', 'receipt', 'scheduler')]
        profile = read('raw/' + case + '.native_profile.v4.json')
        frozen_case = read('raw/source/setup/cases_1024/' + case + '.json')
        assert result['status'] == receipt['status'] == 'CAPTURE_COMPLETED'
        assert result['receipt_sha256'] == sha(ROOT / base / 'receipt.json')
        assert result['source_case_sha256'] == sha(ROOT / 'raw/source/setup/cases_1024' / (case + '.json'))
        assert result['capture_msprobe'] is False and receipt['msprobe_enabled'] is False
        assert result['all_requests_in_one_prefill'] and result['prefill_certified']
        assert result['operator_variant'] == 'stock' and result['environment_before_native_init']['LD_PRELOAD'] is None
        assert result['environment_before_native_init']['ASCEND_CUSTOM_OPP_PATH'] is None
        assert result['environment_before_native_init']['ACLNN_CACHE_LIMIT'] == '0'
        assert len(result['native_calls']) == 28 and sorted(x['layer_index'] for x in result['native_calls']) == list(range(28))
        assert {x['api'] for x in result['native_calls']} == {'npu_fused_infer_attention_score'}
        assert profile['inputs']['case_result_sha256'] == sha(ROOT / base / 'result.json')
        assert profile['status'] == 'INCOMPLETE_PROFILE' and len(profile['calls']) == 28 and not profile['errors']
        assert scheduler['target_position'] == scheduler['target_row'] == 1023 and scheduler['target_output_row'] == 0
        assert scheduler['total'] == result['batch_size'] * result['prompt_length']
        assert result['prompt_length'] == 1024 and all(row == frozen_case['prompt_token_ids'][0]
                                                     for row in frozen_case['prompt_token_ids'])
        selected = []
        for layer in (0, 1):
            call_path = ROOT / base / 'native_calls' / f'call_{layer:03d}.json'
            call = json.loads(call_path.read_bytes())
            assert sha(call_path) == result['native_calls'][layer]['sha256']
            assert call['layer_index'] == layer and call['status'] == 'returned' and call['target_mapping_certified']
            assert call['input_integrity']['exact_logical_hashes_and_metadata_unchanged']
            assert strip_phase(call['arguments_before']) == strip_phase(call['arguments_after'])
            assert strip_phase(call['target_tensors_before']) == strip_phase(call['target_tensors'])
            mapping = call['target_mapping']
            assert mapping['first_row'] == 0 and mapping['last_row'] == 1023 and mapping['count'] == 1024
            assert mapping['prompt_token_sha256'] == frozen_case['expected_target_token_sha256']
            assert mapping['keys_are_target_request_all_prefill_tokens'] and mapping['absolute_query_position'] == 1023
            assert call['arguments_before']['kwargs']['block_table'] is None
            selected.append({'layer': layer, 'input_integrity_independently_rechecked_from_records': True,
                             'target_mapping': mapping,
                             'target_inputs_before': {k: fingerprint(v) for k, v in call['target_tensors_before'].items()},
                             'output_target_all': fingerprint(call['output_target_all']),
                             'output_target_last': fingerprint(call['output_target_last']),
                             'mask': fingerprint(call['arguments_before']['kwargs']['atten_mask']),
                             'precision_arguments': call['precision_arguments']})
            calls[(case, layer)] = call
        selection = [r for c in profile['calls'] for r in c['selected_tiling_records']]
        launches = [r for c in profile['calls'] for r in c['selected_launch_records']]
        assert len(selection) == len(launches) == 28
        assert all(c['selected_key_launch_correlated'] and not c['raw_tiling_complete'] for c in profile['calls'])
        assert all(len(c['selection_launch_correlations']) == 1 and c['selection_launch_correlations'][0]['correlated']
                   for c in profile['calls'])
        signature = lambda row: (row['tiling_key_decimal'], row['num_blocks'], row['schedule_mode'])
        assert Counter(map(signature, selection)) == Counter(map(signature, launches))
        profile_counts = [{'tiling_key_decimal': k, 'num_blocks': b, 'schedule_mode': s, 'calls': n}
                          for (k, b, s), n in sorted(Counter(map(signature, selection)).items())]
        logits[case] = read(base + 'logits.json')
        results[case] = result
        case_summaries[case] = {'capture_status': result['status'], 'profile_status': profile['status'],
                               'batch_size': result['batch_size'], 'prompt_length': result['prompt_length'],
                               'actual_fia_call_count': 28, 'selection_launch_correlated_calls': 28,
                               'selected_and_launched_signatures': profile_counts,
                               'raw_tiling_complete': False,
                               'missing_profile_fields': sorted({x for c in profile['calls'] for x in c['missing_evidence']}),
                               'native_thread_identity_verified': all(c['native_thread_identity_verified'] for c in profile['calls']),
                               'non_utf8_line_count': profile['non_utf8_line_count'],
                               'native_log_sha256': profile['inputs']['log_sha256'],
                               'target_layers': selected, 'target_logits': fingerprint(logits[case]),
                               'target_argmax_token_id': result['requests'][0]['output_token_ids'][0]}
    assert all(r['options'] == results[CASES[0]]['options'] for r in results.values())
    assert all(r['source_sha256'] == results[CASES[0]]['source_sha256'] for r in results.values())

    def compare(left, right):
        layers = []
        for layer in (0, 1):
            a, b = calls[(left, layer)], calls[(right, layer)]
            layers.append({'layer': layer,
                           'target_input_hash_equal': {k: fingerprint(v) == fingerprint(b['target_tensors_before'][k])
                                                       for k, v in a['target_tensors_before'].items()},
                           'output_target_all_hash_equal': fingerprint(a['output_target_all']) == fingerprint(b['output_target_all']),
                           'output_target_last_hash_equal': fingerprint(a['output_target_last']) == fingerprint(b['output_target_last']),
                           'mask_hash_equal': fingerprint(a['arguments_before']['kwargs']['atten_mask']) == fingerprint(b['arguments_before']['kwargs']['atten_mask']),
                           'explicit_precision_arguments_equal': a['precision_arguments'] == b['precision_arguments']})
        return {'left': left, 'right': right, 'layers': layers,
                'target_logits_hash_equal': fingerprint(logits[left]) == fingerprint(logits[right]),
                'target_argmax_token_equal': results[left]['requests'][0]['output_token_ids'] == results[right]['requests'][0]['output_token_ids']}

    return {'schema': 'fault28.stock_fia_calibration.summary.v1', 'status': 'CALIBRATION_COMPLETED_PROFILE_INCOMPLETE',
            'attempt': 'model_stock_a5', 'physical_card': completion['card'],
            'options': results[CASES[0]]['options'], 'versions': results[CASES[0]]['versions'],
            'checkpoint_conversion': results[CASES[0]]['checkpoint_conversion'],
            'same_options_and_probe_source_across_cases': True,
            'cases': case_summaries,
            'repeat_comparisons': [compare('stock_b1', 'stock_b1_repeat'), compare('stock_b8', 'stock_b8_repeat')],
            'batch_comparison': compare('stock_b1', 'stock_b8'),
            'stock_same_input_attention_fault_established': False,
            'earliest_upstream_divergence_localized': False,
            'fault_injection_performed': False, 'mechanism_accepted': False, 'msprobe_collected': False,
            'cleanup': {'owned_container_removed': completion['owned_container_removed'],
                        'device_lock_released': completion['device_lock_released'],
                        'external_container_mutations': completion['external_container_mutations'],
                        'all_host_devices_idle_claimed': False},
            'limits': ['Comparison uses capture-time logical SHA-256 records; private tensor bytes are intentionally absent, so numerical distances cannot be recomputed from this package.',
                       'The same key/block count/schedule does not establish the same hidden M/S1/S2 fields or absence of SplitKV.',
                       'Layer0 target Q_all and K_all already differ between B1 and B8. Identical Q_last and V_all do not mean identical attention inputs because K_all differs.',
                       'At layer1 all recorded target Q/K/V fields differ; its output difference cannot identify a same-input FIA failure.',
                       'Two runs per batch size establish only the observed repeat stability, not universal determinism.',
                       'Selection/launch name/key/PID/block/schedule match within synchronized API windows; a distinct launch thread remains explicitly recorded, not silently treated as same-thread proof.',
                       'Complete raw tiling bytes/length and strict native-thread correlation remain missing; all four profile reports remain INCOMPLETE_PROFILE.',
                       'No fault mode, private operator, tensor injection or msprobe statistics collection was used in these stock calibration runs.']}


if __name__ == '__main__':
    result = build()
    path = ROOT / 'summary.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=True, allow_nan=False)
    print(json.dumps({'status': result['status'], 'summary_sha256': sha(path)}))
