"""Summarize HF a3 JSON/logged tiling metadata without tensor binaries or NPU."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent
CASES = ('stock_b1', 'stock_b1_repeat', 'stock_b8', 'stock_b8_repeat')


def read(relative):
    return json.loads((ROOT / relative).read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target(record):
    assert record['hashed_slice'] == 'batch[0:1], all heads/positions'
    assert record['dtype'] == 'torch.float16' and record['byteorder'] == 'little'
    shape = [1, *record['shape'][1:]]
    assert record['logical_bytes'] == math.prod(shape) * 2
    return {'logical_shape': shape, 'dtype': record['dtype'], 'logical_sha256': record['logical_sha256'],
            'logical_bytes': record['logical_bytes'], 'raw_order': record['raw_order'], 'byteorder': record['byteorder']}


def build():
    source = read('raw/source_manifest.json')
    for path in (ROOT / 'raw/source').rglob('*'):
        if path.is_file():
            assert sha(path) == source[str(path.relative_to(ROOT / 'raw/source')).replace('\\', '/')]
    completion = read('raw/completion.json')
    assert completion['status'] == 'CAPTURE_COMPLETED' and completion['owned_container_removed']
    assert completion['device_lock_released'] and not completion['cleanup_errors']
    results, calls, summaries, profiles = {}, {}, {}, {}
    for case in CASES:
        base = 'raw/' + case + '/'
        result = read(base + 'result.json')
        profile = read('raw/' + case + '.fa_log_profile/profile.json')
        frozen = read('raw/source/setup/cases_2048/' + case + '.json')
        assert result['status'] == 'CAPTURE_COMPLETED' and result['operator_variant'] == 'stock'
        assert result['capture_msprobe'] is False and result['forward_count'] == 1
        assert result['prefill_certified'] and result['all_requests_in_one_prefill']
        assert result['prompt_length'] == 2048 and result['source_case_sha256'] == sha(ROOT / 'raw/source/setup/cases_2048' / (case + '.json'))
        assert all(row == frozen['prompt_token_ids'][0] for row in frozen['prompt_token_ids'])
        assert len(result['native_calls']) == len(profile['calls']) == 28
        assert [x['layer_index'] for x in result['native_calls']] == list(range(28))
        assert {x['api'] for x in result['native_calls']} == {'torch.nn.functional.scaled_dot_product_attention'}
        assert profile['status'] == 'CAPTURED_UNDECODED_FRAMING_CANDIDATE'
        assert profile['source_sha256']['result'] == sha(ROOT / base / 'result.json')
        assert profile['extractor_sha256'] == sha(ROOT / 'analysis_source/extract_fa_log_profile.py')
        layers = []
        for layer in (0, 1):
            path = ROOT / base / 'native_calls' / f'call_{layer:03d}.json'
            call = json.loads(path.read_bytes())
            assert sha(path) == result['native_calls'][layer]['sha256']
            assert call['layer_index'] == layer and call['original_api_calls'] == 1 and call['status'] == 'returned'
            assert call['actual_layout'] == 'BNSD' and call['target_mapping_certified']
            assert call['target_mapping']['batch_slot'] == 0 and call['target_mapping']['last_token_position'] == 2047
            assert call['target_mapping']['token_sha256'] == frozen['expected_target_token_sha256']
            differences = []
            for name, before in call['target_inputs_before'].items():
                after = call['target_inputs_after'][name]
                assert target(before) == target(after)
                for key in set(before) | set(after):
                    if before.get(key) != after.get(key):
                        assert key == 'actual_format' and before[key] == 0 and after[key] == 2
                        differences.append('target_inputs.' + name + '.actual_format')
            normalizations = call['input_format_normalizations']
            assert sorted(differences) == sorted(x['field'] for x in normalizations)
            assert all(x['before'] == 0 and x['after'] == 2 for x in normalizations)
            assert call['attention_mask'] is None and call['scalar_arguments']['is_causal'] is True
            layers.append({'layer': layer, 'target_inputs': {k: target(v) for k, v in call['target_inputs_before'].items()},
                           'target_output': target(call['target_output']),
                           'target_input_values_and_nonformat_metadata_unchanged': True,
                           'format_normalizations': normalizations, 'scalar_arguments': call['scalar_arguments']})
            calls[(case, layer)] = call
        for i, call in enumerate(profile['calls']):
            assert call['enter']['layer_index'] == i and call['enter']['native_index'] == i
            assert call['tiling_key'] == '22619644976' and call['block_dim'] == 20
            assert call['declared_raw_bytes'] == 1464 and call['aligned_log_bytes'] == 1472
            assert call['framing_verified'] is False and call['schema_decoded'] is False
            assert len(call['launches']) == 1 and [int(x['index']) for x in call['chunks']] == [0, 1]
            assert all(x['op'] == call['launches'][0]['op'] and x['tid'] == call['launches'][0]['tid'] for x in call['chunks'])
            words = [int(w) for row in call['chunks'] for w in row['words'].split()]
            aligned = struct.pack('<' + 'I' * len(words), *words)
            profile_root = ROOT / 'raw' / (case + '.fa_log_profile')
            a_path = profile_root / f'call_{i:03d}.aligned_log.bin'
            p_path = profile_root / f'call_{i:03d}.declared_prefix.bin'
            assert a_path.read_bytes() == aligned and len(aligned) == call['aligned_log_bytes']
            assert p_path.read_bytes() == aligned[:call['declared_raw_bytes']]
            assert sha(a_path) == call['aligned_log.bin_sha256'] and sha(p_path) == call['declared_prefix.bin_sha256']
        signatures = Counter((c['tiling_key'], c['block_dim'], c['declared_raw_bytes'], c['aligned_log_bytes'],
                              c['kernel_workspace_count'], c['api_workspace_bytes']) for c in profile['calls'])
        summaries[case] = {'capture_status': result['status'], 'profile_status': profile['status'],
                           'batch_size': result['batch_size'], 'prompt_length': 2048,
                           'sdpa_calls': 28, 'actual_fa_launch_records': 28,
                           'launch_signatures': [{'key': k, 'blocks': b, 'declared_bytes': d, 'aligned_bytes': a,
                                                   'kernel_workspace_count': w, 'api_workspace_bytes': ws, 'calls': n}
                                                  for (k,b,d,a,w,ws),n in sorted(signatures.items())],
                           'schedule_mode_from_profile': None,
                           'unique_declared_prefix_hashes': sorted({c['declared_prefix.bin_sha256'] for c in profile['calls']}),
                           'unique_aligned_log_hash_count': len({c['aligned_log.bin_sha256'] for c in profile['calls']}),
                           'target_layers': layers,
                           'target_logits': target(result['forward_results'][0]['target_logits']),
                           'target_argmax_token_id': result['forward_results'][0]['requests'][0]['next_token_argmax'],
                           'full_tensor_bytes_included': False}
        results[case], profiles[case] = result, profile
    def compare(left, right):
        layers = []
        for layer in (0, 1):
            a, b = calls[(left, layer)], calls[(right, layer)]
            layers.append({'layer': layer, 'target_input_hash_equal': {k: target(v) == target(b['target_inputs_before'][k])
                           for k,v in a['target_inputs_before'].items()},
                           'target_output_hash_equal': target(a['target_output']) == target(b['target_output']),
                           'scalar_arguments_equal': a['scalar_arguments'] == b['scalar_arguments']})
        return {'left': left, 'right': right, 'layers': layers,
                'target_logits_hash_equal': summaries[left]['target_logits'] == summaries[right]['target_logits'],
                'target_argmax_equal': summaries[left]['target_argmax_token_id'] == summaries[right]['target_argmax_token_id'],
                'declared_prefix_hash_equal_by_layer': [a['declared_prefix.bin_sha256'] == b['declared_prefix.bin_sha256']
                                                        for a,b in zip(profiles[left]['calls'],profiles[right]['calls'])],
                'aligned_log_hash_equal_by_layer': [a['aligned_log.bin_sha256'] == b['aligned_log.bin_sha256']
                                                    for a,b in zip(profiles[left]['calls'],profiles[right]['calls'])]}
    normalized_options = [{k:v for k,v in r['options'].items() if k != 'batch_size'} for r in results.values()]
    assert all(x == normalized_options[0] for x in normalized_options)
    assert len({json.dumps(r['source_sha256'], sort_keys=True) for r in results.values()}) == 1
    return {'schema': 'fault28.stock_hf_fa_calibration.summary.v1',
            'status': 'CALIBRATION_COMPLETED_RAW_TILING_FRAMING_CANDIDATE', 'attempt': 'hf_stock_2048_a3',
            'physical_card': completion['card'], 'options_except_batch': normalized_options[0],
            'versions': results[CASES[0]]['versions'], 'same_probe_source_across_cases': True,
            'cases': summaries, 'repeat_comparisons': [compare('stock_b1','stock_b1_repeat'),compare('stock_b8','stock_b8_repeat')],
            'batch_comparison': compare('stock_b1','stock_b8'),
            'fault_injection_performed': False, 'mechanism_accepted': False, 'msprobe_collected': False,
            'raw_framing_verified': False, 'typed_schema_decode_included': False,
            'stock_same_input_attention_fault_established': False,
            'cleanup': {'owned_container_removed': completion['owned_container_removed'],
                        'device_lock_released': completion['device_lock_released'], 'all_host_devices_idle_claimed': False},
            'limits': ['HF native SDPA/FlashAttentionScore path is distinct from the vLLM FIA calibration and uses a 2048-token prompt.',
                       'Target records retain the original full-batch shape but hash batch[0:1]; logical shape here is independently reconstructed and checked against byte count.',
                       'Q/K format 0 to 2 normalization is explicitly allowed only with identical logical SHA and all remaining recorded metadata unchanged; it is not a numeric fault.',
                       'This calibration records target-input integrity, not the later formal full-B8 before/after integrity contract.',
                       'Different B1/B8 target Q/K/V values before layer0 prevent identifying the observed output difference as a same-input FA failure.',
                       'The declared 1464-byte prefix is an unverified framing candidate. The full 1472 bytes are retained; no trailing bytes are silently discarded or assigned a schema.',
                       'Typed decoding and public-context byte comparison are not included; M/S1/S2 legality, reduction order, precision and non-SplitKV are not established by these files.',
                       'API workspace includes helper operators; it is not the raw kernel workspace-size vector. Schedule mode is not supplied by this extracted profile.',
                       'No private tensor binary or full log is included, so readers cannot recompute numerical error from the package.',
                       'No msprobe collection or fault injection was performed in these stock runs.']}


if __name__ == '__main__':
    summary = build()
    with (ROOT/'summary.json').open('x',encoding='utf-8') as stream:
        json.dump(summary,stream,indent=2,ensure_ascii=True,allow_nan=False)
    print(json.dumps({'status':summary['status'],'summary_sha256':sha(ROOT/'summary.json')}))
