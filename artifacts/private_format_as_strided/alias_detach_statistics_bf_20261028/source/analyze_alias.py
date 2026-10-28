"""Independent, standard-library-only fault27 evidence acceptance.

Native comparison reads only config/dump/stack/construct, never runtime or
injection controls. Runtime/reference and control acceptance are separate.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys


STAGES = ['base_created', 'view_created', 'view_written', 'base_read', 'consumed']
STATISTICS = ('Max', 'Min', 'Mean', 'Norm')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def identity(tensor):
    return tensor['identity']


def address(value):
    require(type(value) in (int, str), 'Identity must be an integer or integer string')
    number = int(value, 0) if isinstance(value, str) else value
    require(number > 0, 'Null/invalid tensor or storage identity')
    return number


def same_storage(left, right):
    a, b = identity(left), identity(right)
    cdata_equal = address(a['storage_cdata']) == address(b['storage_cdata'])
    pointer_equal = address(a['storage_ptr']) == address(b['storage_ptr'])
    require(cdata_equal == pointer_equal, 'StorageImpl and live storage-pointer evidence disagree')
    return cdata_equal


def tensor_values(tensor, label):
    require(isinstance(tensor, dict), label + ': missing tensor observation')
    meta = identity(tensor)
    for field in ('object_id', 'tensor_cdata', 'storage_cdata', 'storage_ptr', 'tensor_ptr'):
        address(meta[field])
    require(meta['dtype'] == 'torch.bfloat16' and meta['device'].startswith('npu:'), label + ': not actual NPU BF16')
    require(isinstance(meta['shape'], list) and all(type(n) is int and n > 0 for n in meta['shape']), label + ': bad shape')
    require(len(meta['stride']) == len(meta['shape']) and all(type(n) is int and n >= 0 for n in meta['stride']), label + ': bad stride')
    require(type(meta['storage_offset']) is int and meta['storage_offset'] >= 0, label + ': bad offset')
    bits, values = tensor['bf16_bits'], tensor['values_flat']
    require(len(bits) == len(values) == math.prod(meta['shape']), label + ': incomplete logical payload')
    require(all(type(n) is int and 0 <= n <= 65535 for n in bits), label + ': invalid BF16 bit values')
    decoded = [struct.unpack('<f', struct.pack('<I', n << 16))[0] for n in bits]
    require(all(finite(v) for v in values + decoded) and values == decoded, label + ': values do not match BF16 bits')
    digest = hashlib.sha256(b''.join(struct.pack('<H', n) for n in bits)).hexdigest()
    require(tensor['sha256'] == digest, label + ': value SHA mismatch')
    rb = tensor['readback']
    require(rb['source_identity_before'] == meta == rb['source_identity_after'], label + ': readback changed identity')
    require(rb['identity_unchanged'] is True and rb['direct_nd_values_equal'] is True, label + ': readback control failed')
    require(rb['direct_before_sha256'] == rb['direct_after_sha256'] == digest, label + ': readback changed values')
    require(rb['source_bf16_bits_before'] == rb['source_bf16_bits_after'] == bits,
            label + ': direct before/after value bits disagree with ND observation')
    require(isinstance(rb['method'], str) and rb['method'], label + ': readback method missing')
    return values


def geometry(tensor, shape, stride, offset, actual_format, label):
    meta = identity(tensor)
    require(meta['shape'] == shape and meta['stride'] == stride and meta['storage_offset'] == offset,
            label + ': logical geometry differs')
    require(meta['actual_format'] == actual_format, label + ': actual NPU format differs')


def reference_values(path):
    reference = read(path)
    indices = [17 + i * 16 + j for i in range(8) for j in range(8)]
    initial, source = list(range(256)), [-64 - i for i in range(64)]
    updated = initial.copy()
    for at, value in zip(indices, source):
        updated[at] = value
    expected = {'base_shape': [16, 16], 'view_size': [8, 8], 'view_stride': [16, 1], 'offset': 17,
                'indices': indices, 'untouched_indices': [i for i in range(256) if i not in indices],
                'initial_base_flat': initial, 'initial_view_flat': [initial[i] for i in indices],
                'write_values_flat': source, 'expected_updated_base_flat': updated,
                'expected_stale_base_flat': initial, 'expected_updated_consumer_flat': [v + 1 for v in updated],
                'expected_stale_consumer_flat': [v + 1 for v in initial], 'target_count': 64, 'untouched_count': 192}
    require(all(reference.get(k) == v for k, v in expected.items()), 'CPU reference differs from independent integer-index reconstruction')
    require(len(set(indices)) == 64 and max(indices) < 256, 'Invalid oracle indices')
    return expected


def delta(left, right):
    require(len(left) == len(right), 'Compared logical lengths differ')
    return [i for i, (a, b) in enumerate(zip(left, right)) if a != b]


def load_runtime(directory, reference, reference_sha):
    data = read(directory / 'runtime.json')
    require(data['status'] == 'completed', 'Runtime observation incomplete')
    require('arm' not in data and 'hits' not in data, 'Runtime observation mixes injection control labels')
    require(data['rank'] == 0 and data['forward'] == 0 and type(data['pid']) is int and data['pid'] > 0,
            'Unexpected rank/forward/PID')
    require(data['reference_sha256'] == reference_sha, 'Observer used a different CPU reference')
    require(data.get('versions') and data.get('source_sha256'), 'Runtime provenance missing')
    require(all(re.fullmatch('[0-9a-f]{64}', value) for value in data['source_sha256'].values()), 'Invalid source fingerprint')
    require([s['stage'] for s in data['stages']] == STAGES, 'Incomplete/reordered observation stages')
    base_id = identity(data['stages'][0]['base'])
    for stage in data['stages']:
        name = stage['stage']
        require(stage['synchronized'] is True, 'Observation was not synchronized at ' + name)
        require(identity(stage['base']) == base_id, 'Original base object/storage changed at ' + name)
        geometry(stage['base'], [16, 16], [16, 1], 0, 29, name + '.base')
        for key in ('base', 'view', 'etensor', 'original_base_read_nd', 'consumer', 'write_values'):
            if stage.get(key) is not None:
                tensor_values(stage[key], name + '.' + key)
    first = data['stages'][0]
    require(first.get('view') is None and first.get('etensor') is None, 'Objects appeared before view creation')
    require(first['base']['values_flat'] == reference['initial_base_flat'], 'Initial original base differs from oracle')
    unmatched = data['unmatched_passthrough']
    require(identity(unmatched['base']) == base_id, 'Unmatched control did not use the same live base')
    require(tensor_values(unmatched['base'], 'unmatched.base') == reference['initial_base_flat'], 'Unmatched call changed original base')
    require(tensor_values(unmatched['view'], 'unmatched.view') == reference['initial_view_flat'], 'Unmatched call changed view values')
    geometry(unmatched['view'], [8, 8], [16, 1], 17, 29, 'unmatched.view')
    require(same_storage(unmatched['base'], unmatched['view']), 'Unmatched call did not retain native alias')
    return data


def evaluate_arm(data, reference, is_fault):
    observations = []
    first_view = data['stages'][1]['view']
    for index, stage in enumerate(data['stages']):
        name, base = stage['stage'], stage['base']
        written = index >= 2
        expected_base = reference['initial_base_flat'] if is_fault or not written else reference['expected_updated_base_flat']
        require(base['values_flat'] == expected_base, name + ': original base values violate contract')
        if index == 0:
            continue
        view, temporary = stage['view'], stage.get('etensor')
        require(identity(view) == identity(first_view), name + ': returned view object/storage changed')
        geometry(view, [8, 8], [16, 1], 17, 2 if is_fault else 29, name + '.view')
        require(view['values_flat'] == reference['write_values_flat' if written else 'initial_view_flat'], name + ': view values incorrect')
        aliases_base = same_storage(base, view)
        require(aliases_base is not is_fault, name + ': incorrect view/original-base storage relation')
        if is_fault:
            require(temporary is not None, name + ': temporary tensor missing')
            geometry(temporary, [16, 16], [16, 1], 0, 2, name + '.etensor')
            require(identity(temporary) == identity(data['stages'][1]['etensor']), name + ': temporary identity changed')
            require(not same_storage(base, temporary) and same_storage(view, temporary), name + ': independent temporary/view relationship failed')
            expected_temp = reference['expected_updated_base_flat' if written else 'initial_base_flat']
            require(temporary['values_flat'] == expected_temp, name + ': temporary contents incorrect')
        else:
            require(temporary is None, name + ': healthy arm unexpectedly has temporary tensor')
        observations.append({'stage': name, 'view_aliases_original_base': aliases_base,
                             'view_aliases_temporary': same_storage(view, temporary) if temporary is not None else None,
                             'original_base_vs_correct_updated_indices': delta(base['values_flat'], reference['expected_updated_base_flat']),
                             'original_base_vs_initial_indices': delta(base['values_flat'], reference['initial_base_flat'])})
    read_stage = data['stages'][3]
    write_stage = data['stages'][2]
    require(write_stage['copy_return_identity'] == identity(write_stage['view']), 'copy_ did not return the same view object')
    require(write_stage['write_values']['values_flat'] == reference['write_values_flat'], 'Actual copy_ source differs from frozen values')
    require(identity(write_stage['write_values'])['shape'] == [8, 8], 'Actual copy_ source shape differs')
    read_tensor = read_stage['original_base_read_nd']
    geometry(read_tensor, [16, 16], [16, 1], 0, 2, 'base_read.original_base_read_nd')
    require(read_tensor['values_flat'] == read_stage['base']['values_flat'], 'Named original-base read does not equal actual original base')
    final = data['stages'][4]
    require(final['consumer_input_identity'] == identity(final['base']), 'Consumer did not receive same original base object/storage')
    expected_consumer = reference['expected_stale_consumer_flat' if is_fault else 'expected_updated_consumer_flat']
    require(identity(final['consumer'])['shape'] == [16, 16], 'Consumer shape differs')
    require(final['consumer']['values_flat'] == expected_consumer, 'Actual consumer does not match original base + 1')
    require(final['consumer']['values_flat'] == [v + 1 for v in final['base']['values_flat']], 'Consumer arithmetic does not follow observed base')
    return {'passed': True, 'stages': observations, 'unmatched_passthrough_observed_native_alias': True,
            'consumer_used_same_original_base': True, 'all_logical_checks_exact': True}


def runtime_joint(baseline, fault, reference_path):
    reference = reference_values(reference_path)
    paths = {'reference': reference_path, 'B/runtime': baseline / 'runtime.json'}
    b = load_runtime(baseline, reference, sha(reference_path))
    report = {'scope': 'Common observer plus independent logical oracle; no injection-success log used.',
              'baseline': evaluate_arm(b, reference, False)}
    if fault is not None:
        f = load_runtime(fault, reference, sha(reference_path))
        paths['F/runtime'] = fault / 'runtime.json'
        require(b['pid'] != f['pid'], 'B/F must use fresh independent processes')
        require(b['versions'] == f['versions'] and b['source_sha256'] == f['source_sha256'], 'B/F software/observer sources differ')
        report['fault'] = evaluate_arm(f, reference, True)
        structural = []
        numerical = []
        for bs, fs in zip(b['stages'], f['stages']):
            if bs.get('view') is not None and fs.get('view') is not None:
                if same_storage(bs['base'], bs['view']) != same_storage(fs['base'], fs['view']):
                    structural.append(bs['stage'])
            changed = delta(bs['base']['values_flat'], fs['base']['values_flat'])
            if changed:
                numerical.append({'stage': bs['stage'], 'changed_indices': changed})
                require(changed == reference['indices'], 'Base pair differs outside exactly the 64 target positions')
        consumer_changed = delta(b['stages'][-1]['consumer']['values_flat'], f['stages'][-1]['consumer']['values_flat'])
        require(consumer_changed == reference['indices'], 'Consumer pair differs outside exactly the 64 target positions')
        require(structural and structural[0] == 'view_created', 'Structural divergence not first observed at view creation')
        require(numerical and numerical[0]['stage'] == 'view_written', 'Original base numerical divergence did not follow actual copy_')
        report.update(first_observed_storage_relation_difference=structural[0],
                      first_observed_original_base_value_difference=numerical[0],
                      consumer_changed_indices=consumer_changed, target_elements=64, untouched_elements=192,
                      boundary='Runtime observes the original base immediately after copy_. Native module-only statistics may first expose this at the subsequent named base-read module.')
    report['input_sha256'] = {name: sha(path) for name, path in paths.items()}
    report['status'] = 'PASS_RUNTIME_JOINT'
    return report


def stack_index(stack):
    result = {}
    for key, value in stack.items():
        if isinstance(value, list) and len(value) == 2 and isinstance(value[0], list):
            for name in value[0]:
                result[name] = value[1]
        elif key.startswith(('Module.', 'Torch.', 'Tensor.', 'Functional.', 'NPU.')):
            result[key] = value
    return result


def summary_comparison(left, right, path='$'):
    result = {'numeric_differences': [], 'metadata_differences': [], 'unobservable': [], 'compared_numeric_fields': 0}
    def walk(a, b, at):
        if isinstance(a, dict) and isinstance(b, dict):
            if a.get('type') in ('torch.Tensor', 'numpy.ndarray') or b.get('type') in ('torch.Tensor', 'numpy.ndarray'):
                for key in ('type', 'dtype', 'shape'):
                    if a.get(key) != b.get(key):
                        result['metadata_differences'].append(at + '.' + key)
                for key in STATISTICS:
                    av, bv = a.get(key), b.get(key)
                    if not finite(av) or not finite(bv):
                        result['unobservable'].append(at + '.' + key)
                    else:
                        result['compared_numeric_fields'] += 1
                        if av != bv:
                            result['numeric_differences'].append({'path': at + '.' + key, 'B': av, 'F': bv})
                return
            for key in dict.fromkeys([*a, *b]):
                if key in ('data_name', 'md5'):
                    continue
                if key not in a or key not in b:
                    result['metadata_differences'].append(at + '.' + key)
                else:
                    walk(a[key], b[key], at + '.' + key)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                result['metadata_differences'].append(at + '.length')
            else:
                for i, pair in enumerate(zip(a, b)):
                    walk(*pair, at + '[' + str(i) + ']')
        elif type(a) != type(b) or a != b:
            result['metadata_differences'].append(at)
    walk(left, right, path)
    result['interpretation'] = 'NUMERIC_DIFFERENCE' if result['numeric_differences'] else 'NO_DIFFERENCE_IN_AVAILABLE_STATISTICS'
    return result


def load_native(directory):
    config = read(directory / 'config.json')
    require(config['task'] == 'statistics' and config['level'] == 'mix', 'Wrong native capture mode')
    require(config['statistics']['summary_mode'] == 'statistics', 'Native summary mode differs')
    require(config['step'] == [0] and config['rank'] in ([], [0]), 'Wrong native step/rank')
    require(config['statistics'].get('tensor_list', []) == [], 'Full tensor capture requested')
    require(config.get('async_dump', False) is False, 'Asynchronous capture unsupported')
    root = directory / 'observation/step0/rank0'
    dump, stack, construct = [read(root / (kind + '.json')) for kind in ('dump', 'stack', 'construct')]
    require(dump['task'] == 'statistics' and dump['level'] == 'mix' and dump.get('data'), 'Missing native statistics data')
    require(isinstance(construct, dict) and construct and isinstance(stack, dict) and stack, 'Missing stack/construct evidence')
    data = dump['data']
    modules = [name for name in data if name.startswith('Module.')]
    require(modules, 'No semantic module observations')
    indexed = stack_index(stack)
    require(all(indexed.get(name) for name in modules), 'Semantic module stack missing')
    containers = {construct.get(name) for name in modules if isinstance(construct.get(name), str)}
    leaves = [name for name in modules if name not in containers]
    require(leaves, 'No leaf semantic modules could be identified from native construct')
    return {'data': data, 'modules': modules, 'stack': indexed, 'construct': construct, 'config': config,
            'semantic_modules': leaves, 'container_modules': [name for name in modules if name in containers],
            'sources': {'config.json': sha(directory / 'config.json'),
                        **{kind + '.json': sha(root / (kind + '.json')) for kind in ('dump', 'stack', 'construct')}}}


def native_operation_coverage(capture):
    """Verify observed operation ownership, independently of injector counters."""
    modules = capture['semantic_modules']
    def one_module(attribute):
        candidates = [name for name in modules if '.' + attribute + '.' in name]
        require(len(candidates) == 1, 'Missing/ambiguous semantic module: ' + attribute)
        return candidates[0]
    create, write_module = one_module('create_view'), one_module('write_view')
    one_module('read_original_base')
    one_module('consume_original_base')
    def belongs(name, module):
        visited = set()
        parent = capture['construct'].get(name)
        while isinstance(parent, str) and parent not in visited:
            if parent == module:
                return True
            visited.add(parent)
            parent = capture['construct'].get(parent)
        return False
    as_strided = [name for name in capture['data'] if re.search(r'\.as_strided\.\d+\.forward$', name)]
    selected = [name for name in as_strided if belongs(name, create)]
    copies = [name for name in capture['data'] if re.search(r'\.copy_\.\d+\.forward$', name) and belongs(name, write_module)]
    require(len(selected) == 1, 'CreateView must own exactly one captured original as_strided API')
    require(len(copies) == 1, 'WriteView must own exactly one captured original copy_ API')
    selected_stack = capture['stack'].get(selected[0], [])
    require(re.search(r'\bin selected_as_strided\b', '\n'.join(map(str, selected_stack))),
            'Captured selected as_strided lacks its controlled-callsite stack')
    require(capture['stack'].get(copies[0]), 'Captured WriteView copy_ stack missing')
    unmatched = [name for name in as_strided if re.search(
        r'\bin unmatched_as_strided\b', '\n'.join(map(str, capture['stack'].get(name, []))))]
    require(len(unmatched) == 1 and not belongs(unmatched[0], create),
            'Unmatched native passthrough callsite not independently observed outside CreateView')
    return {'status': 'PASS_NATIVE_OPERATION_COVERAGE',
            'selected_as_strided': {'name': selected[0], 'owner_module': create, 'stack': selected_stack},
            'write_copy': {'name': copies[0], 'owner_module': write_module, 'stack': capture['stack'][copies[0]]},
            'unmatched_as_strided': {'name': unmatched[0], 'stack': capture['stack'][unmatched[0]]},
            'scope': 'Actual native msprobe API records plus construct ancestry and callsite stacks. Separate from first-difference search and injector success counts.'}


def aligned_trace(b, f):
    """Align by API kind, complete native stack and semantic parent, not IDs.

    Observer.stage's base/view/temporary source lines and stage callsites remain
    distinct. Fault-only operations remain unmatched instead of shifting slots.
    """
    def groups(capture):
        grouped = defaultdict(list)
        for index, (name, record) in enumerate(capture['data'].items()):
            frames = capture['stack'].get(name)
            if not frames or name.startswith('Module.'):
                continue
            kind = re.sub(r'\.\d+\.forward$', '', name)
            key = (kind, json.dumps(frames, sort_keys=True), json.dumps(capture['construct'].get(name), sort_keys=True))
            grouped[key].append((index, name, record, frames))
        return grouped
    bg, fg = groups(b), groups(f)
    changed, unaligned = [], []
    compared = 0
    unobservable = 0
    for key in dict.fromkeys([*bg, *fg]):
        bs, fs = bg[key], fg[key]
        if len(bs) != len(fs):
            unaligned.append({'api_type': key[0], 'B_count': len(bs), 'F_count': len(fs),
                              'stack_sha256': hashlib.sha256(key[1].encode()).hexdigest()})
            continue
        for left, right in zip(bs, fs):
            difference = summary_comparison(left[2].get('output'), right[2].get('output'))
            unobservable += len(difference['unobservable'])
            if difference['metadata_differences']:
                unaligned.append({'api_type': key[0], 'B_name': left[1], 'F_name': right[1],
                                  'reason': 'Aligned stack but incompatible output metadata'})
                continue
            compared += 1
            if difference['numeric_differences']:
                frames_text = '\n'.join(map(str, left[3]))
                changed.append({'B_trace_index': left[0], 'F_trace_index': right[0],
                                'B_name': left[1], 'F_name': right[1], 'api_type': key[0],
                                'parent': b['construct'].get(left[1]), 'stack': left[3],
                                'common_observer_readback': bool(re.search(r'\bin read_tensor\b', frames_text)),
                                'output': difference})
    changed.sort(key=lambda item: item['B_trace_index'])
    order_preserved = [item['F_trace_index'] for item in changed] == sorted(item['F_trace_index'] for item in changed)
    require(order_preserved, 'Changed aligned trace nodes have inconsistent execution order')
    return {'first_aligned_trace_numerical_difference': changed[0] if changed else None,
            'aligned_api_pairs': compared, 'changed_aligned_api_pairs': len(changed),
            'unobservable_statistic_fields': unobservable, 'unmatched_groups': unaligned,
            'scope': 'Earliest numeric output difference among API pairs with matching type, full native stack, parent and occurrence count. Includes shared observer readback. Fault-only unmatched calls are not globally aligned; no absolute claim about all unpaired trace nodes.'}


def native_only(baseline, fault):
    b = load_native(baseline)
    report = {'scope': 'Native msprobe config/dump/stack/construct only. No runtime, reference, arm controls, or expected fault position read.',
              'input_sha256': {'B': b['sources']}, 'baseline_module_sequence': b['modules'],
              'native_operation_coverage': {'B': native_operation_coverage(b)},
              'semantic_leaf_module_sequence': b['semantic_modules'], 'excluded_container_modules': b['container_modules'],
              'limits': ['Null/nonfinite statistics are unobservable, never equal measurements.',
                         'Equal summaries do not establish elementwise equality or storage aliasing.',
                         'The semantic first difference is searched only over leaf module outputs; enclosing AliasMechanism output is excluded using construct parent relationships.',
                         'Only Tensor/ndarray numeric summary fields count as numerical differences. Scalar format codes and other non-tensor changes are metadata.',
                         'Injected extra API calls are not aligned by global numerical suffix.',
                         'This report does not identify a unique native kernel root cause.']}
    if fault is None:
        report['status'] = 'PASS_NATIVE_BASELINE_CAPTURE'
        return report
    f = load_native(fault)
    report['native_operation_coverage']['F'] = native_operation_coverage(f)
    report['input_sha256']['F'] = f['sources']
    bc, fc = dict(b['config']), dict(f['config'])
    bc.pop('dump_path', None); fc.pop('dump_path', None)
    require(bc == fc, 'Native B/F capture configurations differ beyond output path')
    require(b['modules'] == f['modules'], 'Semantic module sequence/count mismatch; comparison ambiguous')
    require(b['semantic_modules'] == f['semantic_modules'], 'Leaf semantic module hierarchy differs')
    comparisons = []
    for name in b['semantic_modules']:
        left, right = b['data'][name], f['data'][name]
        comparison = {'module': name, 'output': summary_comparison(left.get('output'), right.get('output')),
                      'input': summary_comparison(left.get('input_args'), right.get('input_args')),
                      'B_stack': b['stack'][name], 'F_stack': f['stack'][name],
                      'stacks_equal': b['stack'][name] == f['stack'][name],
                      'parents_equal': b['construct'].get(name) == f['construct'].get(name)}
        require(comparison['stacks_equal'] and comparison['parents_equal'], 'Semantic module callsite/parent mismatch: ' + name)
        require(not comparison['output']['metadata_differences'], 'Semantic module output geometry mismatch: ' + name)
        comparisons.append(comparison)
    changed = [x for x in comparisons if x['output']['numeric_differences']]
    report['module_comparisons'] = comparisons
    report['first_observable_numerical_module_difference'] = changed[0]['module'] if changed else None
    report['first_semantic_module_difference'] = changed[0]['module'] if changed else None
    report['changed_module_outputs'] = [x['module'] for x in changed]
    report['trace_comparison_including_observer'] = aligned_trace(b, f)
    report['status'] = 'ANALYZED_NATIVE_STATISTICS_ONLY'
    return report


def control_gate(directory, arm):
    control = read(directory / 'control.json')
    require(control['arm'] == arm and control['hits'] == (1 if arm == 'F' else 0), 'Unexpected injection control hit count')
    require(control['selected_original_calls'] == 1, 'Selected native call count is not exactly one')
    require(control['unmatched_calls'] >= 1 and control['unmatched_calls'] == control['unmatched_original_calls'], 'Unmatched native passthrough count failed')
    require(control['errors'] == [], 'Injector recorded errors')
    receipt = read(directory / 'receipt.json')
    require(receipt['status'] == 'completed' and receipt['capture_requested'] is True and receipt['capture_completed'] is True, 'Native capture completion missing')
    require(receipt['rank'] == 0 and receipt['step'] == 0 and receipt['forward'] == 0, 'Receipt step/rank/forward differs')
    require(receipt['full_tensor_files'] == [], 'Full tensor files declared in statistics capture')
    require(receipt['runtime_sha256'] == sha(directory / 'runtime.json') and receipt['control_sha256'] == sha(directory / 'control.json'), 'Receipt source hash mismatch')
    require(receipt['config_sha256'] == sha(directory / 'config.json') and receipt['exit_code'] == 0, 'Capture config hash or exit status failed')
    expected = {'observation/step0/rank0/' + k + '.json' for k in ('dump', 'stack', 'construct')}
    require({item['path'] for item in receipt['native_files']} == expected and len(receipt['native_files']) == 3, 'Receipt native file set differs')
    for item in receipt['native_files']:
        path = directory / item['path']
        require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'Native file hash/size mismatch')
    require(receipt['dump_nodes'] == len(read(directory / 'observation/step0/rank0/dump.json')['data']), 'Native node count mismatch')
    require(not any(p.suffix.lower() in ('.pt', '.pth', '.npy', '.npz', '.safetensors') for p in directory.rglob('*') if p.is_file()), 'Tensor binary in statistics arm')
    return {'status': 'PASS_CONTROL_AND_COMPLETION', 'hits': control['hits'], 'selected_original_calls': control['selected_original_calls'],
            'unmatched_calls': control['unmatched_calls'], 'unmatched_original_calls': control['unmatched_original_calls'],
            'control_sha256': sha(directory / 'control.json'), 'receipt_sha256': sha(directory / 'receipt.json'),
            'scope': 'Separate control gate; not used to infer where data first diverges.'}


def attempt(function, *args):
    try:
        return function(*args)
    except (ValueError, KeyError, TypeError, OSError, OverflowError, struct.error) as error:
        return {'status': 'FAIL', 'error': type(error).__name__ + ': ' + str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--fault', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = {'main': args.output, 'native': args.output.with_name(args.output.stem + '.native.json'),
             'joint': args.output.with_name(args.output.stem + '.joint.json')}
    require(len(set(paths.values())) == 3 and not any(p.exists() for p in paths.values()), 'Output already exists')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    native = attempt(native_only, args.baseline, args.fault)
    joint = attempt(runtime_joint, args.baseline, args.fault, args.reference)
    controls = {'B': attempt(control_gate, args.baseline, 'B')}
    if args.fault is not None:
        controls['F'] = attempt(control_gate, args.fault, 'F')
    errors = {key: report.get('error') for key, report in {'native': native, 'joint': joint, **controls}.items() if report['status'] == 'FAIL'}
    if not errors and args.fault is not None:
        first = native['first_observable_numerical_module_difference']
        if not first or not any(label in first for label in ('.read_original_base.', '.consume_original_base.')):
            errors['native_phase_consistency'] = 'Discovered numerical difference is absent or not in post-copy original-base read/consumer phases'
    status = 'FAIL' if errors else 'PASS_BASELINE' if args.fault is None else 'PASS_MECHANISM_AND_STATISTICS'
    write(paths['native'], native)
    write(paths['joint'], joint)
    report = {'status': status, 'errors': errors, 'controls': controls,
              'reports': {key: {'file': str(paths[key]), 'sha256': sha(paths[key])} for key in ('native', 'joint')},
              'analyzer_sha256': sha(Path(__file__)),
              'scope': 'Controlled alias-detach mechanism and observed statistics; not model-chain reachability or historical CANN defect reproduction.'}
    write(paths['main'], report)
    print(json.dumps({'status': status, 'errors': errors, 'output': str(args.output)}, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
