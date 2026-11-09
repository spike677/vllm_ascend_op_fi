"""Synthetic CPU model traces only; never framework/device execution evidence."""
import copy
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import audit_model_coverage as coverage
import compare_model_statistics as comparison
import model_statistics_io as reader
from statistics_values import MANIFEST_SHA, ORIGINAL_SHA, PATCHED_SHA

HERE = Path(__file__).resolve().parent


def h(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    return reader.sha(path)


def tensor(shape, mean=0.25, unknown=False):
    return {'type': 'torch.Tensor', 'dtype': 'torch.float16', 'shape': shape, 'requires_grad': False,
        **{key: None if unknown else value for key, value in [('Max', 1.0), ('Min', -1.0), ('Mean', mean), ('Norm', 32.0)]}}


def fixture(root, name, batch, count=10):
    directory = root / name
    config_model = {'num_hidden_layers': 28, 'num_attention_heads': 16, 'num_key_value_heads': 8, 'head_dim': 128, 'hidden_size': 1024}
    model_config = root / (name + '_model_config.json')
    config_sha = write(model_config, config_model)
    write(directory / 'module_inventory.json', {'config_sha256': config_sha,
        'layers': [{'layer_index': i, 'layer_name': f'model.layers.{i}.self_attn', 'class': 'Qwen3Attention'} for i in range(28)]})
    config = {'task': 'statistics', 'level': 'mix', 'precision': 'high', 'rank': [], 'step': list(range(count)), 'async_dump': False,
        'dump_path': '/evidence/' + name + '/observation', 'statistics': {'scope': [], 'list': [], 'data_mode': ['all'], 'summary_mode': 'statistics', 'tensor_list': []}}
    cfgsha = write(directory / 'config.json', config)
    processor_path = '/audit/collector_high_a1/packages/msprobe/core/dump/data_dump/data_processor/pytorch_processor.py'
    provenance = {'variant': 'task_local_fp32_high_v1', 'precision': 'high', 'modified_official_collector': True,
        'manifest_sha256': MANIFEST_SHA, 'original_processor_sha256': ORIGINAL_SHA, 'patched_processor_sha256': PATCHED_SHA,
        'artifact_count': 513, 'provenance': 'Synthetic fixture; uniformly modified collector explicitly disclosed.',
        'policy': 'Uniform FP32 temporary for high statistics; original tensors unchanged.',
        'actual_loaded': {'module': 'msprobe.core.dump.data_dump.data_processor.pytorch_processor',
            'path': processor_path, 'method_code_file': processor_path, 'sha256': PATCHED_SHA}}
    provsha = write(directory / 'collector_provenance.json', provenance)
    phases, native_files, receipts, events = [], [], [], []
    for step in range(count):
        run, phase_index = divmod(step, 2)
        stage, mode = ('prefill', 'decode')[phase_index], reader.MODES[run]
        qlen, kvlen = (2048, 2048) if phase_index == 0 else (1, 2049)
        rows = [{'request_id': f'P{i}', 'batch_slot': i, 'prompt_token_ids_sha256': h(f'P{i}'),
            'prompt_length': 2048, 'query_length': qlen, 'kv_length': kvlen,
            'query_positions': list(range(2048)) if phase_index == 0 else [2048]} for i in range(batch)]
        mapping_file = f'phases/step{step}/request_mapping.json'
        mapsha = write(directory / mapping_file, {'step': step, 'run_index': run, 'mode': mode, 'stage': stage,
            'request_rows': rows, 'actual_input_ids_sha256': h(stage + '_tokens'), 'actual_position_ids_sha256': h(stage + '_positions')})
        data, stack, construct, apis, parents = {}, {}, {}, [], []
        def add(node, value, parent, frames):
            stack[str(len(stack))] = [[node], frames]
            data[node] = value; construct[node] = parent
        add('Functional.embedding.0.forward', {'input_args': [], 'input_kwargs': {}, 'output': [tensor([batch, qlen, 1024])]}, None,
            ['File /audit/model_probe.py, line 400, in model_forward, \n output = model(**inputs)'])
        for layer in range(28):
            module_path = f'model.layers.{layer}.self_attn'
            parent = f'Module.{module_path}.Qwen3Attention.forward.0'
            api = f'Functional.scaled_dot_product_attention.{layer}.forward'
            is_fault = run in (2, 3)
            input_mean = 0.2501 if is_fault and layer > 0 else 0.25
            output_mean = 0.2502 if is_fault and (stage == 'prefill' or layer > 0) else 0.25
            frames = ['File /audit/model_probe.py, line 200, in _invoke, \n returned = original(*args, **kwargs)']
            if run:
                frames.append('File /audit/runtime/native_execution_strategy/session.py, line 100, in execute, \n value = invoke_original()')
            diagnostic = f'Tensor.detach.{layer}.forward'
            add(diagnostic, {'input_args': [tensor([batch, 16, qlen, 128], 0.9 if is_fault else 0.25)], 'input_kwargs': {},
                'output': [tensor([batch, 16, qlen, 128], 0.9 if is_fault else 0.25)]}, parent,
                ['File /audit/model_helpers.py, line 60, in tensor_snapshot, \n payload = tensor.detach().cpu()'])
            args = [tensor([batch, heads, tokens, 128], input_mean) for heads, tokens in ((16, qlen), (8, kvlen), (8, kvlen))]
            add(api, {'input_args': args, 'input_kwargs': {'is_causal': {'type': 'bool', 'value': phase_index == 0}},
                'output': [tensor([batch, 16, qlen, 128], output_mean)]}, parent, frames)
            add(parent, {'input_args': [tensor([batch, qlen, 1024], input_mean)], 'input_kwargs': {},
                'output': [tensor([batch, qlen, 1024], output_mean)]}, None,
                ['File /transformers/modeling_qwen3.py, line 200, in forward, \n attention_output = attention_interface(...)'])
            apis.append(api); parents.append(parent)
            module_id = f's{step}_layer{layer}'
            common = {'step': step, 'layer_index': layer, 'layer_name': module_path,
                'module_call_id': module_id, 'request_ids': [row['request_id'] for row in rows]}
            events += [{**common, 'event': 'module_enter', 'hidden_shape': [batch, qlen, 1024]},
                       {**common, 'event': 'module_exit', 'output_shape': [batch, qlen, 1024]}]
        dump = {'task': 'statistics', 'level': 'mix', 'framework': 'pytorch', 'dump_data_dir': None, 'data': data}
        for kind, payload in [('dump', dump), ('stack', stack), ('construct', construct)]:
            file = directory / f'observation/step{step}/rank0/{kind}.json'
            fingerprint = write(file, payload)
            native_files.append({'path': file.relative_to(directory).as_posix(), 'sha256': fingerprint, 'bytes': file.stat().st_size})
        receipts.append({'step': step, 'functional_sdpa_nodes': apis, 'parents': parents, 'dump_nodes': len(data)})
        phases.append({'step': step, 'run_index': run, 'mode': mode, 'stage': stage,
            'request_mapping_file': mapping_file, 'request_mapping_sha256': mapsha, 'layer_count': 28,
            'sdpa_call_ids': [f'call_s{step}_l{i}' for i in range(28)],
            'cache_length_before': 0 if phase_index == 0 else 2048, 'cache_length_after': kvlen,
            'logits': {'POISON_NOT_LOCALIZATION_INPUT': 'not genuine numeric evidence'}, 'private_logits_file': 'DO_NOT_OPEN.bin'})
    phase_sha = write(directory / 'phase_manifest.json', {'schema': 'nes.model_phases.v1', 'phases': phases})
    (directory / 'module_events.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in events), encoding='utf-8')
    receipt_sha = write(directory / 'receipt.json', {'schema': 'nes.model_coverage.statistics_receipt.v1', 'steps': receipts,
        'native_files': native_files, 'config_sha256': cfgsha, 'collector_provenance_sha256': provsha,
        'statistics_collector': provenance, 'diagnostic_functions': ['tensor_snapshot']})
    result_sha = write(directory / 'result.json', {'schema': 'nes.model_coverage.capture.v1', 'status': 'CAPTURE_COMPLETED',
        'case_id': name, 'case_sha256': h('case'), 'runtime_manifest_sha256': h('runtime'),
        'checkpoint_manifest_sha256': h('checkpoint'), 'tokenizer_manifest_sha256': h('tokenizer'),
        'source_fingerprints_sha256': h('sources'), 'versions': {'fixture': True},
        'phase_manifest_file': 'phase_manifest.json', 'phase_manifest_sha256': phase_sha, 'receipt_sha256': receipt_sha,
        'module_inventory_sha256': reader.sha(directory / 'module_inventory.json'),
        'module_events_sha256': reader.sha(directory / 'module_events.jsonl'),
        'calls': [{'file': 'DO_NOT_OPEN_NATIVE.json', 'claimed_first': 'Tensor.fake_wrong_location'}]})
    completion = root / (name + '_completion.json')
    write(completion, {'status': 'CAPTURE_COMPLETED', 'owned_container_removed': True, 'cleanup_errors': [], 'device_lock_released': True,
        'cases': [{'case_id': name, 'rc': 0, 'status': 'CAPTURE_COMPLETED_NOT_MECHANISM_ACCEPTANCE', 'result_sha256': result_sha}]})
    (directory / 'DO_NOT_OPEN_NATIVE.json').write_bytes(b'\xffNOT_JSON_FAULT_LABELS_FORBIDDEN')
    (directory / 'DO_NOT_OPEN.bin').write_bytes(b'PRIVATE_NOT_LOCALIZATION_INPUT')
    return directory, completion, model_config


def reject(function):
    try:
        function()
    except (ValueError, KeyError, TypeError):
        return
    raise AssertionError('Expected rejection')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'cpu_analysis_report.json')
    args = parser.parse_args()
    assert not args.output.exists()
    checks = []
    temporary_parent = Path(chr(92) * 2 + '?' + chr(92) + str(HERE)) if os.name == 'nt' else HERE
    with tempfile.TemporaryDirectory(prefix='coverage_', dir=temporary_parent) as directory:
        root = Path(directory)
        summaries = []
        for batch in (1, 8):
            case, completion, config = fixture(root, f'b{batch}', batch)
            loaded = reader.load_capture(case, completion)
            audit = coverage.audit(loaded, config)
            assert audit['framework_sdpa_calls'] == 280 and audit['fault_prefill_shape_candidates'] == 56
            assert audit['fault_decode_unsupported_calls'] == 56 and audit['fault_prefill_layer_request_pairs'] == 56 * batch
            assert audit['native_fa_coverage_verified'] is False
            summaries.append(audit)
            report = comparison.analyze(loaded, root / f'report_b{batch}')
            assert report['run_comparison_count'] == 5 and report['phase_comparison_count'] == 10
            assert report['runs'][0]['alignment_complete'] is False and report['runs'][0]['first_observed_model_api_difference'] is None
            for index in (1, 2):
                first = report['runs'][index]['first_observed_model_api_difference']
                assert first['stage'] == 'prefill' and first['boundary'] == 'output'
                assert first['node']['B']['name'] == 'Functional.scaled_dot_product_attention.0.forward'
            for index in (3, 4):
                assert report['runs'][index]['alignment_complete'] is True and report['runs'][index]['first_observed_model_api_difference'] is None
            phase = comparison.compare_phase(loaded, 2, 4)
            assert phase['layers'][0]['input_already_different'] is False and phase['layers'][1]['input_already_different'] is True
            assert all(x['independent_fault_origin_proven'] is False for x in phase['layers'])
            assert len(phase['diagnostic_classification']['F_nodes']) == 28
            for ref in report['phase_reports']:
                assert reader.sha(root / f'report_b{batch}' / ref['file']) == ref['sha256']
            checks.append(f'b{batch}_full_formal_count_mapping_first_difference_propagation_restore_and_SHA')
        assert sum(x['framework_sdpa_calls'] for x in summaries) == 560
        assert sum(x['fault_prefill_layer_request_pairs'] for x in summaries) == 504
        checks.append('framework_call_count560_distinct_from_fault_request_incidence504')
        for name, mutate in [
            ('missing_module_event', lambda value: value['events'].pop()),
            ('duplicate_module_event', lambda value: value['events'].append(copy.deepcopy(value['events'][0]))),
            ('wrong_request_order', lambda value: value['events'][0].update(request_ids=list(reversed(value['events'][0]['request_ids'])))),
            ('wrong_hidden2048_vs_actual1024', lambda value: value['events'][0].update(hidden_shape=[8, 2048, 2048])),
            ('wrong_decode_query_position', lambda value: value['steps'][1]['mapping']['request_rows'][0].update(query_positions=[2047])),
            ('duplicate_prompt_tokens', lambda value: value['steps'][0]['mapping']['request_rows'][1].update(prompt_token_ids_sha256=value['steps'][0]['mapping']['request_rows'][0]['prompt_token_ids_sha256'])),
            ('missing_original_sdpa', lambda value: value['steps'][0]['sdpa'].pop()),
            ('nonfresh_prefill_cache', lambda value: value['steps'][0]['phase'].update(cache_length_before=2048))]:
            changed = copy.deepcopy(loaded); mutate(changed)
            reject(lambda: coverage.audit(changed, config)); checks.append(name + '_rejected')
        changed = copy.deepcopy(loaded)
        node = next(x for x in changed['steps'][4]['entries'] if reader.SDPA.fullmatch(x['name']))
        for field in ('Max', 'Min', 'Mean', 'Norm'):
            node['node']['output'][0][field] = None
        phase = comparison.compare_phase(changed, 2, 4)
        assert phase['layers'][0]['output_difference'] is False and phase['layers'][0]['output_unobservable_fields'] == 4
        checks.append('null_primary_statistics_remain_unobservable_not_equal_tensor_claim')
        case, completion, config = fixture(root, 'pilot', 1, 2)
        pilot = reader.load_capture(case, completion)
        assert coverage.audit(pilot, config)['framework_sdpa_calls'] == 56
        assert comparison.analyze(pilot, root / 'pilot_reports')['phase_comparison_count'] == 0
        checks.append('pilot2steps_schema_only_never_claims_formal_comparisons')
        (case / 'observation/step0/rank0/hidden.pt').write_bytes(b'forbidden')
        reject(lambda: reader.load_capture(case, completion)); checks.append('tensor_payload_rejected')
    report = {'schema': 'nes.model_coverage.analysis_cpu.v1', 'status': 'PASS_SYNTHETIC_CPU_FIXTURES',
        'checks': [{'name': name, 'passed': True} for name in checks], 'native_calls': 0, 'framework_imports': False,
        'actual_model_capture_verified': False,
        'source_sha256': {name: reader.sha(HERE / name) for name in ('statistics_values.py', 'model_statistics_io.py',
            'audit_model_coverage.py', 'compare_model_statistics.py', 'test_model_analysis.py')}}
    file = args.output
    with file.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps({'status': report['status'], 'checks': len(checks), 'sha256': reader.sha(file)}))


if __name__ == '__main__':
    main()
