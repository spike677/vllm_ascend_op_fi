"""Derive every framework SDPA denominator from raw msprobe and module events."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from model_statistics_io import load_capture, read, require, sha


def request_rows(mapping, stage):
    rows = mapping.get('request_rows')
    require(isinstance(rows, list) and len(rows) in (1, 8), 'Expected explicit B1/B8 request map')
    require([row.get('batch_slot') for row in rows] == list(range(len(rows))), 'Batch row permutation/gap')
    require(len({row.get('request_id') for row in rows}) == len(rows)
            and all(isinstance(row.get('request_id'), str) and row['request_id'] for row in rows), 'Request IDs must be unique')
    for row in rows:
        require(row.get('prompt_length') == 2048 and re.fullmatch('[0-9a-f]{64}', row.get('prompt_token_ids_sha256', '')), 'Unbound prompt length/hash')
        require(row.get('query_length') == (2048 if stage == 'prefill' else 1)
                and row.get('kv_length') == (2048 if stage == 'prefill' else 2049)
                and row.get('query_positions') == (list(range(2048)) if stage == 'prefill' else [2048]), 'Wrong actual query/cache positions')
    require(len({row['prompt_token_ids_sha256'] for row in rows}) == len(rows), 'B8 distinct prompt contract violated')
    for field in ('actual_input_ids_sha256', 'actual_position_ids_sha256'):
        require(re.fullmatch('[0-9a-f]{64}', mapping.get(field, '')), 'Missing actual model input fingerprint')
    return rows


def audit(loaded, model_config_file):
    config, config_sha = read(model_config_file)
    require(config_sha == loaded['inventory']['config_sha256'], 'Inventory is not bound to the actual model config')
    require(all(config.get(k) == value for k, value in {'num_hidden_layers': 28, 'num_attention_heads': 16,
                'num_key_value_heads': 8, 'head_dim': 128}.items()), 'Unexpected model architecture')
    hidden_size = config.get('hidden_size')
    require(type(hidden_size) is int and hidden_size > 0, 'Missing model hidden dimension')
    declared_events = loaded['events']
    require(all(row.get('event') in ('module_enter', 'module_exit') and row.get('step') in loaded['steps'] for row in declared_events),
            'Unrecognized/unregistered module event')
    coverage, phase_summaries = [], []
    request_baseline, stage_fingerprints = None, {}
    all_module_ids, all_wrapper_ids = set(), set()
    for step, trace in loaded['steps'].items():
        phase, mapping = trace['phase'], trace['mapping']
        stage = phase['stage']
        require(phase.get('cache_length_before') == (0 if stage == 'prefill' else 2048)
                and phase.get('cache_length_after') == (2048 if stage == 'prefill' else 2049),
                'KV cache boundaries differ from fresh prefill plus one teacher-forced decode')
        rows = request_rows(mapping, stage); batch = len(rows)
        identity = [{k: row[k] for k in ('request_id', 'batch_slot', 'prompt_token_ids_sha256', 'prompt_length')} for row in rows]
        if request_baseline is None:
            request_baseline = identity
        require(identity == request_baseline, 'Requests reordered/replaced between model runs/phases')
        fingerprints = {k: mapping[k] for k in ('actual_input_ids_sha256', 'actual_position_ids_sha256')}
        if stage not in stage_fingerprints:
            stage_fingerprints[stage] = fingerprints
        require(stage_fingerprints[stage] == fingerprints, 'Teacher-forced inputs/positions differ between runs')
        require(not trace['issues'], 'Raw call stack incomplete')
        apis = trace['sdpa']
        require(len(apis) == 28 and [api['layer_index'] for api in apis] == list(range(28)), 'Raw msprobe misses, duplicates, or reorders attention layers')
        qlen, kvlen = (2048, 2048) if stage == 'prefill' else (1, 2049)
        events = [row for row in declared_events if row['step'] == step]
        require(len(events) == 56, 'Every raw SDPA requires one independent module enter and exit')
        step_ids = []
        for layer, api in enumerate(apis):
            module_path = f'model.layers.{layer}.self_attn'
            matching = [row for row in events if row.get('layer_index') == layer]
            require(len(matching) == 2 and [row['event'] for row in matching] == ['module_enter', 'module_exit'], 'Missing/duplicate/reversed independent module events')
            enter, leave = matching
            require(enter.get('module_call_id') == leave.get('module_call_id') and isinstance(enter.get('module_call_id'), str), 'Module pre/post call identity mismatch')
            module_id = enter['module_call_id']
            require(module_id not in all_module_ids, 'Module call ID reused across phases/layers')
            all_module_ids.add(module_id); step_ids.append(module_id)
            require(all(row.get('layer_name') == module_path and row.get('request_ids') == [r['request_id'] for r in rows] for row in matching),
                    'Module observer request/layer binding mismatch')
            require(enter.get('hidden_shape') == [batch, qlen, hidden_size] and leave.get('output_shape') == [batch, qlen, hidden_size], 'Actual module geometry differs')
            args = api['node'].get('input_args')
            require(isinstance(args, list) and len(args) >= 3, 'Original SDPA does not expose actual Q/K/V')
            for tensor, shape in zip(args[:3], ([batch, 16, qlen, 128], [batch, 8, kvlen, 128], [batch, 8, kvlen, 128])):
                require(tensor.get('type') == 'torch.Tensor' and tensor.get('dtype') == 'torch.float16'
                        and tensor.get('shape') == shape, 'Raw SDPA Q/K/V geometry differs from request mapping')
            outputs = api['node'].get('output')
            require(isinstance(outputs, list) and len(outputs) == 1 and outputs[0].get('shape') == [batch, 16, qlen, 128]
                    and outputs[0].get('dtype') == 'torch.float16', 'Raw SDPA output geometry mismatch')
            coverage.append({'step': step, 'run_index': phase['run_index'], 'mode': phase['mode'], 'stage': stage,
                'layer_index': layer, 'layer_name': module_path, 'framework_api': api['name'],
                'original_dump_ordinal': api['ordinal'], 'module_parent': api['parents'][0],
                'module_call_id': module_id, 'request_ids': [r['request_id'] for r in rows],
                'batch_slots': [r['batch_slot'] for r in rows], 'request_count': batch,
                'query_shape': args[0]['shape'], 'key_shape': args[1]['shape'], 'value_shape': args[2]['shape'],
                'context_class': 'PREFILL_SHAPE_CANDIDATE_REQUIRES_NATIVE_PROOF' if stage == 'prefill' else 'UNSUPPORTED_DECODE_SHAPE',
                'native_kernel_verified_by_this_tool': False,
                'per_request_attention_numerical_effect': 'UNOBSERVABLE_FROM_WHOLE_BATCH_STATISTICS'})
        require(len(phase.get('sdpa_call_ids', [])) == 28 and len(set(phase['sdpa_call_ids'])) == 28, 'Declared wrapper observation count is incomplete')
        require(not all_wrapper_ids.intersection(phase['sdpa_call_ids']), 'Wrapper call ID reused across phases')
        all_wrapper_ids.update(phase['sdpa_call_ids'])
        # This declaration is compared after the raw denominator is established; it never selects its members.
        phase_summaries.append({'step': step, 'run_index': phase['run_index'], 'mode': phase['mode'], 'stage': stage,
            'framework_sdpa_calls': len(apis), 'module_enter_events': 28, 'module_exit_events': 28,
            'layer_request_pairs': 28 * batch, 'request_count': batch,
            'declared_wrapper_sdpa_call_ids': phase['sdpa_call_ids'],
            'observer_diagnostic_nodes': sum(entry['diagnostic'] for entry in trace['entries'])})
    faults = [row for row in coverage if row['mode'] == 'controlled']
    return {'schema': 'nes.model_coverage.framework_denominator.v1', 'status': 'FRAMEWORK_DENOMINATOR_VERIFIED',
        'case_identity': loaded['identity'], 'model_config_file': str(model_config_file), 'model_config_sha256': config_sha,
        'evidence_files': loaded['files'], 'evidence_sha256': loaded['hashes'], 'collector_disclosure': loaded['collector'],
        'phase_count': len(phase_summaries), 'framework_sdpa_calls': len(coverage),
        'layer_request_pairs': sum(row['request_count'] for row in coverage),
        'fault_prefill_shape_candidates': sum(row['stage'] == 'prefill' for row in faults),
        'fault_decode_unsupported_calls': sum(row['stage'] == 'decode' for row in faults),
        'fault_prefill_layer_request_pairs': sum(row['request_count'] for row in faults if row['stage'] == 'prefill'),
        'phases': phase_summaries, 'calls': coverage,
        'denominator_sources': ['original_msprobe_SDPA_nodes', 'original_construct_attention_parents',
            'original_source_stacks', 'actual_model_config', 'module_inventory', 'unconditional_module_enter_exit'],
        'not_opened': ['native_calls files', 'host/profile/hit logs', 'private tensors', 'policy/Session decisions'],
        'native_fa_coverage_verified': False, 'strategy_switch_coverage_verified': False,
        'per_request_tensor_equality_proven': False, 'mechanism_accepted': False,
        'limits': ['Framework SDPA coverage does not prove native FA launch or actual strategy changes.',
            'One B8 SDPA is one call carrying eight requests, not eight native calls.',
            'Request row evidence is the frozen model input mapping cross-checked against actual geometry; whole-tensor statistics do not reveal each row value.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--completion', type=Path, required=True)
    parser.add_argument('--model-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.resolve().is_relative_to(args.case_dir.resolve()), 'Fresh separate report required')
    report = audit(load_capture(args.case_dir, args.completion), args.model_config)
    report['analysis_source_sha256'] = {p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name('model_statistics_io.py'), Path(__file__).with_name('statistics_values.py'))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': report['status'], 'framework_sdpa_calls': report['framework_sdpa_calls'], 'sha256': sha(args.output)}))


if __name__ == '__main__':
    main()
