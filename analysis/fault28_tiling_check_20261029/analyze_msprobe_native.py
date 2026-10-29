"""Compare ONLY native msprobe config/dump/stack/construct; no control or tensor data.

The result concerns recorded statistics and API boundaries, never exact tensor
equality, a device-kernel timeline, tiling policy, or an injection success flag.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

STATISTICS = ('Max', 'Min', 'Mean', 'Norm')
PROVENANCE = {'data_name', 'dump_path', 'dump_data_dir', 'md5'}
FRAME = re.compile(r'^File (.+), line (\d+), in ([^,]+),')
# Exact source-function pairs, not every frame belonging to an observer wrapper.
DIAGNOSTICS = {
    ('hf_runtime_probe.py', 'record'), ('hf_runtime_probe.py', 'identity'),
    ('model_hook.py', 'tensor_record'), ('model_hook.py', 'identity'),
    ('model_hook.py', 'describe'), ('model_hook.py', 'fingerprints'),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def read(path):
    payload = Path(path).read_bytes()
    return json.loads(payload.decode('utf-8'), object_pairs_hook=no_duplicate_keys), hashlib.sha256(payload).hexdigest()


def canonical(name):
    if name is None:
        return None
    if name.startswith('Module.'):
        return re.sub(r'\.(forward|backward)\.\d+$', r'.\1', name)
    return re.sub(r'\.\d+\.(forward|backward)$', r'.\1', name)


def frames(texts):
    result = []
    for text in texts:
        require(isinstance(text, str), 'Non-string stack frame')
        match = FRAME.match(text)
        if match:
            path = match[1].replace('\\', '/')
            result.append((path, int(match[2]), match[3].strip()))
    return result


def ancestry(name, construct):
    require(name in construct, 'Dump node missing from construct: ' + name)
    chain, visited = [], {name}
    parent = construct[name]
    while parent is not None:
        require(isinstance(parent, str) and parent in construct, 'Missing or invalid construct parent')
        require(parent not in visited, 'Construct contains a cycle')
        visited.add(parent)
        chain.append(canonical(parent))
        parent = construct[parent]
    return tuple(chain)


def attention_kind(name):
    name = canonical(name)
    if 'scaled_dot_product_attention' in name and '_scaled_dot_product_' not in name:
        return 'PYTORCH_SDPA_API'
    if '_scaled_dot_product_flash_attention' in name:
        return 'ATEN_FLASH_ATTENTION_OPERATOR'
    if 'npu_fused_infer_attention_score' in name:
        return 'TORCH_NPU_FIA_API'
    if 'npu_fusion_attention' in name:
        return 'TORCH_NPU_FUSION_ATTENTION_API'
    if re.search(r'(^|\.)FlashAttentionScore(\.|$)', name):
        return 'EXPLICIT_FLASH_ATTENTION_SCORE_RECORD'
    return None


def load(directory, config_path):
    directory = Path(directory)
    config, config_sha = read(config_path)
    require(config.get('task') == 'statistics' and config.get('level') == 'mix',
            'Config must request statistics/mix')
    settings = config.get('statistics', {})
    require(settings.get('summary_mode') == 'statistics' and settings.get('tensor_list') == [],
            'Require summary_mode=statistics and tensor_list=[]; md5/tensor mode is a different contract')
    values, hashes = {}, {'config': config_sha}
    for kind in ('dump', 'stack', 'construct'):
        values[kind], hashes[kind] = read(directory / (kind + '.json'))
    dump, stack, construct = values['dump'], values['stack'], values['construct']
    require(dump.get('task') == 'statistics' and dump.get('level') == 'mix' and
            isinstance(dump.get('data'), dict) and dump['data'], 'Missing statistics/mix native data')
    require(isinstance(stack, dict) and stack and isinstance(construct, dict) and construct,
            'Missing native stack or construct')
    index = {}
    for item in stack.values():
        require(isinstance(item, list) and len(item) == 2 and
                isinstance(item[0], list) and isinstance(item[1], list), 'Unknown native stack schema')
        for name in item[0]:
            require(name not in index or index[name] == item[1], 'Ambiguous native stack for ' + name)
            index[name] = item[1]
    entries, diagnostics, raw, issues, counts = [], [], [], [], Counter()
    for ordinal, (name, node) in enumerate(dump['data'].items()):
        require(isinstance(node, dict), 'Invalid dump node: ' + name)
        chain = ancestry(name, construct)
        texts = index.get(name, [])
        parsed = frames(texts)
        if not texts or len(parsed) != len(texts):
            issues.append({'node': name, 'issue': 'missing_or_unparsed_stack'})
        diagnostic = next((f'{Path(p).name}:{f}' for p, _, f in parsed
                           if (Path(p).name, f) in DIAGNOSTICS), None)
        key = (canonical(name), chain, tuple(parsed))
        entry = {'name': name, 'ordinal': ordinal, 'api': canonical(name), 'parents': chain,
                 'stack': texts, 'node': node, 'attention_kind': attention_kind(name),
                 'diagnostic': diagnostic}
        entry['key'] = (*key, counts[key])
        counts[key] += 1
        raw.append(entry)
        if diagnostic:
            diagnostics.append(entry)
        else:
            entries.append(entry)
    # Paths are provenance; all other configuration options must agree across arms.
    comparable_config = {k: v for k, v in config.items() if k not in ('dump_path',)}
    return {'entries': entries, 'diagnostics': diagnostics, 'raw': raw, 'issues': issues, 'hashes': hashes,
            'config': comparable_config, 'framework': dump.get('framework'),
            'unrestricted': not settings.get('scope') and not settings.get('list'),
            'directory': str(directory), 'config_path': str(config_path)}


def public(entry):
    return {k: entry[k] for k in ('name', 'ordinal', 'api', 'parents', 'stack', 'attention_kind', 'diagnostic')}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def scalar(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': repr(value)}
    return value


def compare_values(left, right):
    result = {'numerical_differences': [], 'metadata_differences': [],
              'unobservable_statistics': [], 'matching_finite_statistic_count': 0,
              'tensor_slot_count': 0}

    def visit(a, b, path):
        tensor_a = isinstance(a, dict) and a.get('type') in ('torch.Tensor', 'numpy.ndarray')
        tensor_b = isinstance(b, dict) and b.get('type') in ('torch.Tensor', 'numpy.ndarray')
        if tensor_a or tensor_b:
            if not (tensor_a and tensor_b):
                result['metadata_differences'].append({'slot': path, 'reason': 'tensor_structure_mismatch'})
                return
            result['tensor_slot_count'] += 1
            geometry = True
            for tensor in (a, b):
                require(isinstance(tensor.get('dtype'), str) and isinstance(tensor.get('shape'), list)
                        and all(type(d) is int and d >= 0 for d in tensor['shape']),
                        'Malformed tensor dtype/shape in native record')
            for field in ('type', 'dtype', 'shape'):
                if field not in a or field not in b or a.get(field) != b.get(field):
                    geometry = False
                    result['metadata_differences'].append({'slot': path + '.' + field,
                                                          'B': a.get(field), 'F': b.get(field)})
            # Whole-batch statistics with different geometry do not align a target slice.
            if not geometry:
                return
            for field in STATISTICS:
                av, bv = a.get(field), b.get(field)
                item = {'slot': path + '.' + field, 'B': scalar(av), 'F': scalar(bv)}
                if finite(av) and finite(bv):
                    if av != bv:
                        result['numerical_differences'].append({**item, 'kind': 'finite_statistic_difference'})
                    else:
                        result['matching_finite_statistic_count'] += 1
                elif type(av) in (int, float) and type(bv) in (int, float) and scalar(av) != scalar(bv):
                    # Explicit nonfinite numeric summaries are evidence; null is not.
                    result['numerical_differences'].append({**item, 'kind': 'nonfinite_summary_state_difference'})
                else:
                    result['unobservable_statistics'].append(item)
            return
        if type(a) is not type(b):
            result['metadata_differences'].append({'slot': path, 'reason': 'value_type_mismatch',
                                                  'B_type': type(a).__name__, 'F_type': type(b).__name__})
        elif isinstance(a, dict):
            keys_a, keys_b = set(a) - PROVENANCE, set(b) - PROVENANCE
            for key in sorted(keys_a | keys_b):
                if key not in keys_a or key not in keys_b:
                    result['metadata_differences'].append({'slot': path + '.' + key, 'reason': 'missing_field'})
                else:
                    visit(a[key], b[key], path + '.' + key)
        elif isinstance(a, list):
            if len(a) != len(b):
                result['metadata_differences'].append({'slot': path, 'reason': 'sequence_length_mismatch',
                                                      'B': len(a), 'F': len(b)})
            else:
                for i, (av, bv) in enumerate(zip(a, b)):
                    visit(av, bv, path + '.' + str(i))
        elif scalar(a) != scalar(b):
            result['metadata_differences'].append({'slot': path, 'B': scalar(a), 'F': scalar(b)})

    visit(left, right, '')
    return result


def compare_node(left, right):
    before = compare_values({k: left['node'].get(k) for k in ('input_args', 'input_kwargs', 'parameters')},
                            {k: right['node'].get(k) for k in ('input_args', 'input_kwargs', 'parameters')})
    after = compare_values(left['node'].get('output'), right['node'].get('output'))
    return {'B': public(left), 'F': public(right), 'input': before, 'output': after,
            'input_summaries_match_where_observable': not before['numerical_differences'] and
                not before['metadata_differences'],
            'input_values_proven_equal': False}


def analyze(baseline, fault, baseline_config, fault_config, examples=20, expected_sdpa_calls=28):
    b, f = load(baseline, baseline_config), load(fault, fault_config)
    keys_b, keys_f = [e['key'] for e in b['entries']], [e['key'] for e in f['entries']]
    lookup_b, lookup_f = {e['key']: e for e in b['entries']}, {e['key']: e for e in f['entries']}
    common = set(keys_b) & set(keys_f)
    ordered_common = [k for k in keys_b if k in common] == [k for k in keys_f if k in common]
    configs_match = b['config'] == f['config'] and b['framework'] == f['framework']
    trace_aligned = keys_b == keys_f and not b['issues'] and not f['issues'] and configs_match
    comparisons = [compare_node(lookup_b[k], lookup_f[k]) for k in keys_b if k in common]
    shape_mismatches = [x for x in comparisons if x['input']['metadata_differences'] or x['output']['metadata_differences']]
    shape_compatible = not shape_mismatches
    complete = trace_aligned and shape_compatible
    changed = [x for x in comparisons if x['input']['numerical_differences'] or x['output']['numerical_differences']]
    api_changed = [x for x in changed if not x['B']['name'].startswith('Module.')]
    output_changed = [x for x in api_changed if x['output']['numerical_differences']]
    first = api_changed[0] if complete and api_changed else None
    first_output = output_changed[0] if complete and output_changed else None
    raw_first = None
    for left, right in zip(b['raw'], f['raw']):
        if left['key'] != right['key']:
            raw_first = {'kind': 'raw_record_sequence_or_callsite_difference', 'B': public(left), 'F': public(right)}
            break
        comparison = compare_node(left, right)
        if any(comparison[part][field] for part in ('input', 'output')
               for field in ('numerical_differences', 'metadata_differences')):
            raw_first = {'kind': 'raw_recorded_statistical_or_metadata_difference', **comparison}
            break
    if raw_first is None and len(b['raw']) != len(f['raw']):
        raw_first = {'kind': 'raw_record_count_difference_after_common_prefix',
                     'common_prefix_nodes': min(len(b['raw']), len(f['raw']))}
    prefix_unknown = sum(len(x[part]['unobservable_statistics']) for x in comparisons
                         if first is not None and x['B']['ordinal'] < first['B']['ordinal']
                         for part in ('input', 'output'))
    if not complete:
        status = 'INCONCLUSIVE_ALIGNMENT_OR_METADATA'
    elif api_changed:
        status = 'ALIGNED_OBSERVED_API_STATISTICAL_DIFFERENCE'
    elif any(x[part]['unobservable_statistics'] for x in comparisons for part in ('input', 'output')):
        status = 'ALIGNED_NO_OBSERVED_API_DIFFERENCE_WITH_UNOBSERVABLE_STATISTICS'
    else:
        status = 'ALIGNED_AVAILABLE_STATISTICS_MATCH'

    def coverage(capture):
        nodes = [public(e) for e in capture['entries'] if e['attention_kind']]
        counts = Counter(e['attention_kind'] for e in capture['entries'] if e['attention_kind'])
        module_pattern = re.compile(r'\.layers\.(\d+)\.self_attn\.[^.]+\.forward$')
        modules = [e for e in capture['entries'] if e['name'].startswith('Module.') and module_pattern.search(e['api'])]
        sdpa_owners = []
        for entry in capture['entries']:
            if entry['attention_kind'] == 'PYTORCH_SDPA_API':
                owners = [p for p in entry['parents'] if module_pattern.search(p)]
                sdpa_owners.append({'node': entry['name'], 'attention_module_ancestors': owners,
                                    'layer': int(module_pattern.search(owners[0]).group(1)) if len(owners) == 1 else None})
        module_layers = [int(module_pattern.search(e['api']).group(1)) for e in modules]
        expected = list(range(expected_sdpa_calls))
        passed = (counts['PYTORCH_SDPA_API'] == expected_sdpa_calls
                  and sorted(module_layers) == expected
                  and len(sdpa_owners) == expected_sdpa_calls
                  and all(x['layer'] is not None for x in sdpa_owners)
                  and sorted(x['layer'] for x in sdpa_owners) == expected)
        return {'record_counts_by_boundary': dict(counts), 'nodes': nodes,
                'expected_sdpa_calls': expected_sdpa_calls,
                'hf_attention_module_layers': module_layers, 'sdpa_module_ownership': sdpa_owners,
                'complete_hf_sdpa_module_coverage': passed,
                'pytorch_sdpa_recorded': bool(counts['PYTORCH_SDPA_API']),
                'explicit_flash_attention_score_recorded': bool(counts['EXPLICIT_FLASH_ATTENTION_SCORE_RECORD']),
                'cann_kernel_launch_proven': False}

    attention_coverage = {'B': coverage(b), 'F': coverage(f)}
    return {'schema': 'fault28.msprobe_native_comparison.v1', 'status': status,
            'analysis_inputs': 'ONLY each config.json plus dump.json, stack.json, construct.json; no receipt/control/runtime/hostlog/raw tensor read',
            'input_sha256': {'B': b['hashes'], 'F': f['hashes']},
            'alignment': {'method': 'API family + complete parsed source stack + complete construct ancestry + occurrence; order checked',
                          'config_and_framework_match': configs_match, 'same_filtered_sequence': keys_b == keys_f,
                          'common_order_matches': ordered_common, 'complete_alignment': complete,
                          'metadata_compatible': shape_compatible, 'matched_nodes': len(common),
                          'B_unmatched': [public(e) for e in b['entries'] if e['key'] not in common][:examples],
                          'F_unmatched': [public(e) for e in f['entries'] if e['key'] not in common][:examples],
                          'stack_issues': {'B': b['issues'], 'F': f['issues']},
                          'metadata_mismatch_examples': shape_mismatches[:examples]},
            'first_observed_model_api_numerical_difference': first,
            'first_observed_model_api_output_difference': first_output,
            'raw_first_recorded_difference_including_diagnostics': raw_first,
            'unobservable_statistic_fields_before_first_observed_difference': prefix_unknown,
            'aligned_api_numerically_changed_count': len(api_changed),
            'common_pair_difference_examples_not_global_first': changed[:examples],
            'unobservable_statistic_field_count': sum(len(x[part]['unobservable_statistics'])
                                                      for x in comparisons for part in ('input', 'output')),
            'attention_boundary_coverage': attention_coverage,
            'diagnostic_filter': {'exact_source_function_pairs': sorted([list(x) for x in DIAGNOSTICS]),
                                  'B_excluded': [public(e) for e in b['diagnostics']],
                                  'F_excluded': [public(e) for e in f['diagnostics']],
                                  'observe_wrappers_not_excluded': True},
            'capture_scope_unrestricted': b['unrestricted'] and f['unrestricted'],
            'independent_target_slice_localization_supported': False,
            'tensor_equality_proven': False, 'fault_mechanism_accepted': False,
            'limits': [
                'First means the first observed differing API record in dump order after documented diagnostic exclusions; not the first physical corruption or a kernel execution timeline.',
                'Module output hooks are completion records and are kept separately from the model API first-difference claim.',
                'Null/missing/nonfinite pairs are not equal-value evidence. Identical Max/Min/Mean/Norm cannot prove elementwise equality.',
                'Input summaries matching cannot prove equal Q/K/V/mask values or locate an injected operation.',
                'Different batch shapes are not aligned target-request slices; target slice and causal-token mapping need separate evidence.',
                'A PyTorch SDPA record, ATen flash record, or torch_npu attention API is not by itself a direct CANN FlashAttentionScore launch record.',
                'Tiling, precision, legal-key coverage, SplitKV, original inputs, injection hits and logits propagation are outside this native-only analysis.',
                'Missing/reordered nodes, incompatible arguments/config, and unparsed stacks suppress a global first-difference claim.',
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True, help='Native step/rank directory')
    parser.add_argument('--fault', type=Path, required=True, help='Native step/rank directory')
    parser.add_argument('--baseline-config', type=Path, required=True)
    parser.add_argument('--fault-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh JSON report')
    parser.add_argument('--examples', type=int, default=20)
    parser.add_argument('--expected-sdpa-calls', type=int, default=28,
                        help='Declared model coverage per captured step; default Qwen3-0.6B 28 layers')
    args = parser.parse_args()
    require(1 <= args.examples <= 100, 'Examples must be between 1 and 100')
    require(1 <= args.expected_sdpa_calls <= 200, 'Expected SDPA calls outside bounded model coverage')
    result = analyze(args.baseline, args.fault, args.baseline_config, args.fault_config,
                     args.examples, args.expected_sdpa_calls)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=True, allow_nan=False)
    print(json.dumps({'status': result['status'], 'complete_alignment': result['alignment']['complete_alignment'],
                      'fault_mechanism_accepted': False, 'output': str(args.output)}))
    return 0 if result['alignment']['complete_alignment'] and all(
        x['complete_hf_sdpa_module_coverage'] for x in result['attention_boundary_coverage'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
