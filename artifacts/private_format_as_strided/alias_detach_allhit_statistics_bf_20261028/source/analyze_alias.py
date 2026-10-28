"""Independent, standard-library-only fault15 evidence acceptance.

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


STAGES = ['pre_view', 'view_created', 'view_written', 'base_read', 'consumed']
FORWARDS = (0, 1)
CALLS = (0, 1, 2)
SEMANTIC_STAGES = ('read_base_before', 'create_view', 'write_view', 'read_original_base', 'consume_original_base')
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
    """Reconstruct each sequential write without importing the probe/reference generator."""
    supplied = read(path)
    indices = [17 + row * 16 + col for row in range(8) for col in range(8)]
    initial = list(range(256))
    def scatter(base, values):
        result = base.copy()
        for at, value in zip(indices, values):
            result[at] = value
        return result
    expected = {'base_shape': [16, 16], 'view_size': [8, 8], 'view_stride': [16, 1], 'offset': 17,
                'indices': indices, 'untouched_indices': [i for i in range(256) if i not in indices],
                'initial_base_flat': initial, 'initial_view_flat': [initial[i] for i in indices],
                'target_count': 64, 'untouched_count': 192, 'forward_count': 2,
                'calls_per_forward': 3, 'matched_calls_per_arm': 6}
    require(all(supplied.get(k) == v for k, v in expected.items()), 'Reference geometry/count differs from independent reconstruction')
    require(len(supplied['calls']) == 3, 'Reference must contain three sequential calls')
    before = initial.copy()
    expected['calls'] = []
    for call in CALLS:
        values = [-64 * (call + 1) - j for j in range(64)]
        after = scatter(before, values)
        item = {'call': call, 'baseline_pre_base_flat': before, 'baseline_post_base_flat': after,
                'fault_pre_base_flat': initial, 'fault_post_base_flat': initial,
                'baseline_pre_view_flat': [before[i] for i in indices],
                'fault_pre_view_flat': [initial[i] for i in indices], 'write_values_flat': values,
                'baseline_consumer_flat': [v + 1 for v in after], 'fault_consumer_flat': [v + 1 for v in initial],
                'expected_fault_temporary_post_base_flat': scatter(initial, values)}
        require(all(supplied['calls'][call].get(k) == v for k, v in item.items()), 'Reference sequential call differs: ' + str(call))
        require(all(before[i] != value and initial[i] != value for i, value in zip(indices, values)), 'Oracle write is idempotent')
        expected['calls'].append(item)
        before = after.copy()
    guard_source = [-16 - j for j in range(64)]
    guard_after = scatter(initial, guard_source)
    guard = {'initial_base_flat': initial, 'initial_view_flat': [initial[i] for i in indices],
             'write_values_flat': guard_source, 'expected_updated_base_flat': guard_after,
             'expected_updated_consumer_flat': [v + 1 for v in guard_after]}
    require(all(supplied['guards']['unregistered_base'].get(k) == v for k, v in guard.items()), 'Reference guard differs')
    expected['guards'] = {'unregistered_base': guard}
    return expected


def delta(left, right):
    require(len(left) == len(right), 'Compared logical lengths differ')
    return [i for i, (a, b) in enumerate(zip(left, right)) if a != b]


def validate_tensor_tree(value, label):
    if isinstance(value, dict):
        if 'values_flat' in value or 'bf16_bits' in value:
            tensor_values(value, label)
        else:
            for key, child in value.items():
                validate_tensor_tree(child, label + '.' + str(key))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            validate_tensor_tree(child, label + '[' + str(i) + ']')


def load_runtime(directory, reference, reference_sha):
    data = read(directory / 'runtime.json')
    require(data['status'] == 'completed', 'Runtime observation incomplete')
    require('arm' not in data and 'hits' not in data, 'Runtime mixes injection control labels')
    require('allhit' in data.get('schema', '') and data['schema'].endswith('.v1'), 'Unexpected runtime schema')
    require(data['rank'] == 0 and data['forwards'] == [0, 1] and type(data['pid']) is int and data['pid'] > 0,
            'Unexpected runtime rank/forwards/PID')
    require(data['reference_sha256'] == reference_sha, 'Observer used a different CPU reference')
    require(data.get('versions') and data.get('source_sha256'), 'Runtime provenance missing')
    require(all(re.fullmatch('[0-9a-f]{64}', value) for value in data['source_sha256'].values()), 'Invalid source fingerprint')
    expected_keys = [(f, c, 'f%d_c%d' % (f, c), f) for f in FORWARDS for c in CALLS]
    require([(r['forward'], r['call'], r['case_id'], r['base_slot']) for r in data['cases']] == expected_keys,
            'Missing, repeated, reordered or mislabeled runtime cases')
    bases = {}
    for row in data['cases']:
        require('arm' not in row and 'hits' not in row, 'Case mixes injection controls')
        label = row['case_id']
        require([stage['stage'] for stage in row['stages']] == STAGES, label + ': incomplete/reordered stages')
        base = row['stages'][0]['base']
        if row['call'] == 0:
            bases[row['forward']] = base
            require(tensor_values(base, label + '.initial') == reference['initial_base_flat'], label + ': fresh base initial values differ')
        for stage in row['stages']:
            name = label + '.' + stage['stage']
            require(stage['synchronized'] is True, name + ': observation unsynchronized')
            require(identity(stage['base']) == identity(bases[row['forward']]), name + ': original base identity changed')
            geometry(stage['base'], [16, 16], [16, 1], 0, 29, name + '.base')
            validate_tensor_tree(stage, name)
        pre = row['stages'][0]
        require(pre.get('view') is None and pre.get('etensor') is None, label + ': prior view/temporary leaked into pre_view')
        observed_before = pre['base_before_read_nd']
        geometry(observed_before, [16, 16], [16, 1], 0, 2, label + '.base_before_read_nd')
        require(observed_before['values_flat'] == pre['base']['values_flat'], label + ': named before-read differs from original base')
    require(not same_storage(bases[0], bases[1]) and identity(bases[0])['object_id'] != identity(bases[1])['object_id'],
            'Forward bases are not distinct live registered objects/storages')
    require([(g['forward'], g['kind']) for g in data['guards']] == [(f, k) for f in FORWARDS for k in ('other_callsite', 'unregistered_base')],
            'Missing/reordered passthrough guards')
    for guard in data['guards']:
        validate_tensor_tree(guard, 'guard.' + str(guard['forward']) + '.' + guard['kind'])
    return data


def evaluate_case(row, reference, is_fault):
    label = row['case_id']
    expected = reference['calls'][row['call']]
    prefix = 'fault' if is_fault else 'baseline'
    pre_values, post_values = expected[prefix + '_pre_base_flat'], expected[prefix + '_post_base_flat']
    stages = {stage['stage']: stage for stage in row['stages']}
    view_initial = expected[prefix + '_pre_view_flat']
    view_identity = identity(stages['view_created']['view'])
    observations = []
    for index, name in enumerate(STAGES):
        stage = stages[name]
        written = index >= 2
        base = stage['base']
        require(base['values_flat'] == (post_values if written else pre_values), label + '.' + name + ': original base state violates sequential oracle')
        if index == 0:
            continue
        view, temporary = stage['view'], stage.get('etensor')
        require(identity(view) == view_identity, label + '.' + name + ': returned view identity changed')
        geometry(view, [8, 8], [16, 1], 17, 2 if is_fault else 29, label + '.' + name + '.view')
        require(view['values_flat'] == (expected['write_values_flat'] if written else view_initial), label + '.' + name + ': view does not follow its own current input/write')
        aliases_base = same_storage(base, view)
        require(aliases_base is not is_fault, label + '.' + name + ': incorrect live view/base storage relationship')
        if is_fault:
            require(temporary is not None, label + ': independent temporary missing')
            geometry(temporary, [16, 16], [16, 1], 0, 2, label + '.temporary')
            require(identity(temporary) == identity(stages['view_created']['etensor']), label + ': temporary identity changed')
            require(not same_storage(base, temporary) and same_storage(view, temporary), label + ': view does not alias independent temporary')
            temporary_values = expected['expected_fault_temporary_post_base_flat'] if written else pre_values
            require(temporary['values_flat'] == temporary_values, label + ': temporary values violate current-call oracle')
        else:
            require(temporary is None, label + ': healthy call unexpectedly has a temporary')
        observations.append({'stage': name, 'view_aliases_original_base': aliases_base,
                             'view_aliases_temporary': same_storage(view, temporary) if temporary is not None else None})
    write_stage = stages['view_written']
    require(write_stage['copy_return_identity'] == view_identity, label + ': copy_ did not return its original view')
    require(write_stage['write_values']['values_flat'] == expected['write_values_flat'], label + ': actual copy source differs')
    require(identity(write_stage['write_values'])['shape'] == [8, 8], label + ': copy source shape differs')
    require(all(pre_values[i] != value for i, value in zip(reference['indices'], expected['write_values_flat'])), label + ': write is idempotent at a target')
    actual_changes = delta(stages['pre_view']['base']['values_flat'], write_stage['base']['values_flat'])
    require(actual_changes == ([] if is_fault else reference['indices']), label + ': observed per-call original-base update footprint differs')
    nd = stages['base_read']['original_base_read_nd']
    geometry(nd, [16, 16], [16, 1], 0, 2, label + '.original_base_read_nd')
    require(nd['values_flat'] == post_values, label + ': named after-read differs from original base')
    final = stages['consumed']
    require(final['consumer_input_identity'] == identity(final['base']), label + ': consumer did not receive same original base')
    require(identity(final['consumer'])['shape'] == [16, 16], label + ': consumer shape differs')
    require(final['consumer']['values_flat'] == expected[prefix + '_consumer_flat'] == [v + 1 for v in final['base']['values_flat']],
            label + ': actual consumer does not equal current original base + 1')
    return {'case_id': label, 'forward': row['forward'], 'call': row['call'], 'passed': True,
            'stages': observations, 'actual_original_base_changed_indices': actual_changes,
            'target_write_non_idempotent_count': 64, 'non_target_unchanged_count': 192,
            'consumer_used_same_original_base': True, 'all_logical_checks_exact': True}


def evaluate_guards(data, reference):
    registered = {row['forward']: row['stages'][0]['base'] for row in data['cases'] if row['call'] == 0}
    unregistered = []
    results = []
    for guard in data['guards']:
        label = 'guard.f%d.%s' % (guard['forward'], guard['kind'])
        before, after, view = guard['base_before'], guard['base_after'], guard['view_created']
        require(identity(before) == identity(after), label + ': original base identity changed')
        geometry(before, [16, 16], [16, 1], 0, 29, label + '.base')
        geometry(view, [8, 8], [16, 1], 17, 29, label + '.view')
        require(same_storage(before, view), label + ': unmatched view lost native alias')
        require(before['values_flat'] == reference['initial_base_flat'] and view['values_flat'] == reference['initial_view_flat'],
                label + ': guard initial values differ')
        if guard['kind'] == 'other_callsite':
            require(identity(before) == identity(registered[guard['forward']]), label + ': wrong registered base')
            require(after['values_flat'] == before['values_flat'], label + ': read-only guard changed base')
            require(all(guard.get(k) is None for k in ('view_written', 'write_values', 'consumer_input_identity', 'consumer')), label + ': read-only guard contains write/consumer')
        else:
            require(all(not same_storage(before, base) for base in [*registered.values(), *unregistered]), label + ': unregistered base aliases another live base')
            unregistered.append(before)
            expected = reference['guards']['unregistered_base']
            require(identity(guard['view_written']) == identity(view), label + ': copy_ returned different view identity')
            require(guard['view_written']['values_flat'] == guard['write_values']['values_flat'] == expected['write_values_flat'], label + ': guard copy source/result differs')
            require(identity(guard['write_values'])['shape'] == [8, 8], label + ': guard copy source shape differs')
            require(after['values_flat'] == expected['expected_updated_base_flat'], label + ': native guard write failed')
            require(guard['consumer_input_identity'] == identity(before), label + ': guard consumer changed original base')
            require(identity(guard['consumer'])['shape'] == [16, 16], label + ': guard consumer shape differs')
            require(guard['consumer']['values_flat'] == expected['expected_updated_consumer_flat'], label + ': native guard consumer differs')
        results.append({'forward': guard['forward'], 'kind': guard['kind'], 'native_alias_preserved': True, 'logical_checks_exact': True})
    return results


def verify_retained_temporaries(data):
    """The probe keeps both registered bases and every fault temporary alive."""
    bases = [row['stages'][0]['base'] for row in data['cases'] if row['call'] == 0]
    temporaries = []
    for row in data['cases']:
        temporary = row['stages'][1]['etensor']
        require(temporary is not None, row['case_id'] + ': temporary missing')
        require(all(not same_storage(temporary, base) for base in bases),
                row['case_id'] + ': temporary aliases a live registered base')
        require(all(not same_storage(temporary, previous) for previous in temporaries),
                row['case_id'] + ': temporary reuses retained prior temporary storage')
        temporaries.append(temporary)
    require(len(temporaries) == 6, 'Expected six distinct retained temporaries')
    return {'distinct_retained_temporary_storages': 6, 'independent_of_all_registered_bases': True,
            'scope': 'Within one F process with both registered bases and all six temporaries retained alive.'}


def runtime_joint(baseline, fault, reference_path):
    reference = reference_values(reference_path)
    b = load_runtime(baseline, reference, sha(reference_path))
    report = {'scope': 'Common observer plus independent sequential integer oracle; no injection-success log read.',
              'baseline': {'cases': [evaluate_case(row, reference, False) for row in b['cases']],
                           'guards': evaluate_guards(b, reference)}}
    paths = {'reference': reference_path, 'B/runtime': baseline / 'runtime.json'}
    if fault is not None:
        f = load_runtime(fault, reference, sha(reference_path))
        require(b['pid'] != f['pid'], 'B/F must use independent processes')
        require(b['versions'] == f['versions'] and b['source_sha256'] == f['source_sha256'], 'B/F software/observer sources differ')
        report['fault'] = {'cases': [evaluate_case(row, reference, True) for row in f['cases']],
                           'guards': evaluate_guards(f, reference), 'retained_temporaries': verify_retained_temporaries(f)}
        comparisons = []
        for bc, fc in zip(b['cases'], f['cases']):
            require(bc['case_id'] == fc['case_id'], 'Case pairing mismatch')
            bs, fs = bc['stages'], fc['stages']
            pre_difference = delta(bs[0]['base']['values_flat'], fs[0]['base']['values_flat'])
            require(pre_difference == ([] if bc['call'] == 0 else reference['indices']), 'Prior-state difference violates sequential expectation')
            base_differences = [{'stage': x['stage'], 'changed_indices': delta(x['base']['values_flat'], y['base']['values_flat'])} for x, y in zip(bs, fs)]
            require(all(x['changed_indices'] == reference['indices'] for x in base_differences[2:]), 'Post-write B/F difference footprint differs')
            structural = [x['stage'] for x, y in zip(bs, fs) if x.get('view') is not None and y.get('view') is not None and
                          same_storage(x['base'], x['view']) != same_storage(y['base'], y['view'])]
            require(structural and structural[0] == 'view_created', 'Current call structural relation did not diverge at view creation')
            require(delta(bs[-1]['consumer']['values_flat'], fs[-1]['consumer']['values_flat']) == reference['indices'], 'Consumer pair differs outside target indices')
            require(bs[2]['view']['values_flat'] == fs[2]['view']['values_flat'], 'Actual written view values differ')
            first = next(x for x in base_differences if x['changed_indices'])
            require(first['stage'] == ('view_written' if bc['call'] == 0 else 'pre_view'), 'Observed sequential first difference differs')
            comparisons.append({'case_id': bc['case_id'], 'forward': bc['forward'], 'call': bc['call'],
                                'first_observed_storage_relation_difference': structural[0],
                                'first_observed_original_base_value_difference': first,
                                'input_difference_class': 'NO_PRIOR_VALUE_DIFFERENCE' if not pre_difference else 'INHERITED_PREEXISTING_VALUE_DIFFERENCE',
                                'initial_views_equal_between_arms': bs[1]['view']['values_flat'] == fs[1]['view']['values_flat'],
                                'written_views_equal_between_arms': True, 'consumer_changed_indices': reference['indices']})
        report['case_comparisons'] = comparisons
        report['boundary'] = 'Each fresh base first differs after its first copy_. Later calls inherit differing inputs; their as_strided output difference is not a newly established numerical origin.'
        paths['F/runtime'] = fault / 'runtime.json'
    report['input_sha256'] = {name: sha(path) for name, path in paths.items()}
    report['status'] = 'PASS_RUNTIME_JOINT_ALL_ELIGIBLE'
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


def load_native(directory, step):
    config = read(directory / 'config.json')
    require(config['task'] == 'statistics' and config['level'] == 'mix', 'Wrong native capture mode')
    require(config['statistics']['summary_mode'] == 'statistics', 'Wrong native summary mode')
    require(config['step'] == [0, 1] and config['rank'] in ([], [0]), 'Wrong native steps/rank')
    require(config['statistics'].get('tensor_list', []) == [] and config.get('async_dump', False) is False,
            'Full tensor or asynchronous capture unsupported')
    root = directory / ('observation/step%d/rank0' % step)
    dump, stack, construct = [read(root / (kind + '.json')) for kind in ('dump', 'stack', 'construct')]
    require(dump['task'] == 'statistics' and dump['level'] == 'mix' and dump.get('data'), 'Missing native statistics data')
    require(isinstance(construct, dict) and construct and isinstance(stack, dict) and stack, 'Missing stack/construct')
    data, indexed = dump['data'], stack_index(stack)
    modules = [name for name in data if name.startswith('Module.')]
    require(modules and all(indexed.get(name) for name in modules), 'Missing module callsite evidence')
    return {'step': step, 'data': data, 'modules': modules, 'stack': indexed, 'construct': construct, 'config': config,
            'sources': {'config.json': sha(directory / 'config.json'),
                        **{'step%d/%s.json' % (step, kind): sha(root / (kind + '.json')) for kind in ('dump', 'stack', 'construct')}}}


def ancestry(capture, name):
    result = [name]
    parent = capture['construct'].get(name)
    while isinstance(parent, str):
        require(parent not in result, 'Cycle in native construct ancestry')
        result.append(parent)
        parent = capture['construct'].get(parent)
    return result


def scoped_capture(capture, owner):
    selected = {name: value for name, value in capture['data'].items() if owner in ancestry(capture, name)}
    require(selected, 'Empty native module scope: ' + owner)
    return {**capture, 'data': selected, 'modules': [name for name in selected if name.startswith('Module.')]}


def one_named_module(capture, pattern, label, require_dump=True):
    names = capture['data'] if require_dump else dict.fromkeys([*capture['data'], *capture['construct'], *[v for v in capture['construct'].values() if isinstance(v, str)]])
    matches = [name for name in names if re.search(pattern, name)]
    require(len(matches) == 1, 'Missing/ambiguous native ' + label + ': ' + str(matches))
    return matches[0]


def native_operation_coverage(capture):
    coverage, case_modules = [], []
    all_as_strided = [name for name in capture['data'] if re.search(r'\.as_strided\.\d+\.forward$', name)]
    all_copies = [name for name in capture['data'] if re.search(r'\.copy_\.\d+\.forward$', name)]
    covered_as_strided, covered_copies = [], []
    for call in CALLS:
        prefix = r'^Module\.invocations\.' + str(call) + r'\.'
        owner = one_named_module(capture, prefix + r'InvocationBlock\.forward\.\d+$', 'invocation owner', False)
        modules = {stage: one_named_module(capture, prefix + re.escape(stage) + r'\.[^.]+\.forward\.\d+$', stage) for stage in SEMANTIC_STAGES}
        require(all(owner in ancestry(capture, name) for name in modules.values()), 'Semantic module escaped invocation owner')
        positions = [list(capture['data']).index(name) for name in modules.values()]
        require(positions == sorted(positions), 'Semantic stage execution order differs')
        selected = [name for name in all_as_strided if modules['create_view'] in ancestry(capture, name)]
        copies = [name for name in all_copies if modules['write_view'] in ancestry(capture, name)]
        require(len(selected) == len(copies) == 1, 'Every invocation must own exactly one captured as_strided and copy_')
        require(re.search(r'\bin selected_as_strided\b', '\n'.join(map(str, capture['stack'].get(selected[0], [])))),
                'Selected native as_strided lacks controlled-callsite stack')
        require(capture['stack'].get(copies[0]), 'Captured copy_ stack missing')
        for stage in ('read_base_before', 'read_original_base'):
            observable = summary_comparison(capture['data'][modules[stage]].get('output'), capture['data'][modules[stage]].get('output'))
            require(observable['compared_numeric_fields'] >= 4 and not observable['unobservable'], 'Named ND base observation is not numerically observable: ' + modules[stage])
        covered_as_strided.extend(selected); covered_copies.extend(copies)
        case_modules.extend(modules.values())
        coverage.append({'case_id': 'f%d_c%d' % (capture['step'], call), 'forward': capture['step'], 'call': call,
                         'owner': owner, 'modules': modules,
                         'selected_as_strided': {'name': selected[0], 'owner_module': modules['create_view'], 'stack': capture['stack'][selected[0]]},
                         'write_copy': {'name': copies[0], 'owner_module': modules['write_view'], 'stack': capture['stack'][copies[0]]}})
    require([name for name in capture['data'] if name in case_modules] == case_modules, 'Invocations are missing, interleaved or reordered')
    guards = []
    for kind in ('other_callsite', 'unregistered_base'):
        owner = one_named_module(capture, r'^Module\.guards\.' + kind + r'\.[^.]+\.forward\.\d+$', 'guard ' + kind, False)
        selected = [name for name in all_as_strided if owner in ancestry(capture, name)]
        copies = [name for name in all_copies if owner in ancestry(capture, name)]
        require(len(selected) == 1 and len(copies) == int(kind == 'unregistered_base'), 'Native guard API coverage differs: ' + kind)
        frames = capture['stack'].get(selected[0], [])
        expected_callsite = 'unmatched_as_strided' if kind == 'other_callsite' else 'selected_as_strided'
        require(re.search(r'\bin ' + expected_callsite + r'\b', '\n'.join(map(str, frames))), 'Guard stack does not establish its distinct eligibility condition')
        require(all(capture['stack'].get(name) for name in copies), 'Guard copy_ stack missing')
        covered_as_strided.extend(selected); covered_copies.extend(copies)
        guards.append({'kind': kind, 'owner': owner, 'as_strided': selected[0], 'copies': copies, 'stack': frames})
    require(set(covered_as_strided) == set(all_as_strided) and len(covered_as_strided) == len(all_as_strided) == 5,
            'Unaccounted, duplicate or missing as_strided capture')
    require(set(covered_copies) == set(all_copies) and len(covered_copies) == len(all_copies) == 4,
            'Unaccounted, duplicate or missing copy_ capture')
    return {'status': 'PASS_NATIVE_OPERATION_COVERAGE', 'step': capture['step'], 'cases': coverage, 'guards': guards,
            'as_strided_count': 5, 'copy_count': 4,
            'scope': 'Native dump plus construct ancestry and callsite stacks. No hit counters used.'}


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
    report = {'scope': 'Native config/dump/stack/construct only; runtime, logical oracle and injection controls are not read.',
              'steps': [], 'limits': ['Null/nonfinite statistics are unobservable, never equal measurements.',
                                    'Equal summaries do not prove elementwise equality or storage identity.',
                                    'First difference is searched over aligned shared APIs and named semantic leaf outputs; F-only calls stay unpaired.',
                                    'Later invocations can enter with preexisting differences. A differing view output does not establish a new numerical origin.',
                                    'Scalar metadata/format changes do not count as tensor numerical differences.',
                                    'No unique native-kernel root-cause or model-path claim.']}
    all_case_comparisons = []
    for step in FORWARDS:
        b = load_native(baseline, step)
        bc = native_operation_coverage(b)
        row = {'step': step, 'input_sha256': {'B': b['sources']}, 'native_operation_coverage': {'B': bc}}
        if fault is not None:
            f = load_native(fault, step)
            fc = native_operation_coverage(f)
            bconfig, fconfig = dict(b['config']), dict(f['config'])
            bconfig.pop('dump_path', None); fconfig.pop('dump_path', None)
            require(bconfig == fconfig, 'B/F native capture configuration differs beyond output path')
            require(b['modules'] == f['modules'], 'Native B/F module sequence differs')
            row['input_sha256']['F'] = f['sources']
            row['native_operation_coverage']['F'] = fc
            comparisons = []
            for bcase, fcase in zip(bc['cases'], fc['cases']):
                require(bcase['case_id'] == fcase['case_id'] and bcase['owner'] == fcase['owner'] and bcase['modules'] == fcase['modules'],
                        'Native case ownership differs')
                module_comparisons = []
                for stage, name in bcase['modules'].items():
                    require(b['stack'][name] == f['stack'][name] and b['construct'].get(name) == f['construct'].get(name), 'Native module callsite/parent mismatch')
                    left, right = b['data'][name], f['data'][name]
                    comparison = {'stage': stage, 'module': name, 'output': summary_comparison(left.get('output'), right.get('output')),
                                  'input': summary_comparison(left.get('input_args'), right.get('input_args'))}
                    require(not comparison['output']['metadata_differences'], 'Native module output geometry differs')
                    module_comparisons.append(comparison)
                changed = [item for item in module_comparisons if item['output']['numeric_differences']]
                before = module_comparisons[0]['output']
                input_status = 'OBSERVED_PREEXISTING_NUMERICAL_DIFFERENCE' if before['numeric_differences'] else 'NO_DIFFERENCE_IN_AVAILABLE_PRE_INPUT_STATISTICS'
                first = changed[0] if changed else None
                item = {'case_id': bcase['case_id'], 'forward': step, 'call': bcase['call'],
                        'module_comparisons': module_comparisons, 'input_difference_class': input_status,
                        'first_observable_numerical_module_difference': first['module'] if first else None,
                        'first_observable_numerical_stage': first['stage'] if first else None,
                        'trace_comparison_including_observer': aligned_trace(scoped_capture(b, bcase['owner']), scoped_capture(f, fcase['owner'])),
                        'interpretation': 'Difference already present before this view construction; do not attribute its numerical origin to this invocation.' if before['numeric_differences'] else
                                          'No pre-input summary difference observed; this does not independently prove elementwise input equality.'}
                comparisons.append(item); all_case_comparisons.append(item)
            row['case_comparisons'] = comparisons
            guard_comparisons = []
            for bg, fg in zip(bc['guards'], fc['guards']):
                require(bg['kind'] == fg['kind'] and bg['owner'] == fg['owner'], 'Native guard ownership differs')
                compared = aligned_trace(scoped_capture(b, bg['owner']), scoped_capture(f, fg['owner']))
                require(compared['changed_aligned_api_pairs'] == 0 and not compared['unmatched_groups'], 'Native guard observations differ or cannot be aligned')
                guard_comparisons.append({'kind': bg['kind'], 'comparison': compared})
            row['guard_comparisons'] = guard_comparisons
            row['whole_step_trace_comparison'] = aligned_trace(b, f)
        report['steps'].append(row)
    if fault is not None:
        report['case_comparisons'] = all_case_comparisons
    report['status'] = 'ANALYZED_NATIVE_STATISTICS_ONLY' if fault is not None else 'PASS_NATIVE_BASELINE_CAPTURE'
    return report


def control_gate(directory, arm):
    control = read(directory / 'control.json')
    require(control['arm'] == arm and control['hits'] == (6 if arm == 'F' else 0), 'Unexpected all-eligible injection hit count')
    require(control.get('policy') == 'every_eligible_call', 'Wrong injection policy')
    require(control['selector'].get('forward_filter') is None and control['selector'].get('hit_limit') is None, 'Policy retains a forward or hit-budget filter')
    require(control['selected_original_calls'] == 6 and control['unmatched_calls'] == control['unmatched_original_calls'] == 4,
            'Selected or guard original-call count differs')
    require(control['errors'] == [], 'Injector recorded errors')
    events = control['events']
    require(len(events) == 10 and all(event.get('completed') is True for event in events), 'Missing/incomplete dispatch event')
    matched = [event for event in events if event.get('matched') is True]
    require([(e['forward'], e['call'], e['case_id'], e['registered_base_slot']) for e in matched] ==
            [(f, c, 'f%d_c%d' % (f, c), f) for f in FORWARDS for c in CALLS], 'Control eligible event mapping differs')
    require(all(sum(e['forward'] == f and e.get('matched') is False for e in events) == 2 for f in FORWARDS), 'Control guard events differ per forward')
    receipt = read(directory / 'receipt.json')
    require(receipt['status'] == 'completed' and receipt['capture_requested'] is True and receipt['capture_completed'] is True,
            'Capture did not complete')
    require(receipt['rank'] == 0 and receipt['steps'] == receipt['forwards'] == [0, 1], 'Receipt rank/steps/forwards differ')
    require(receipt['full_tensor_files'] == [] and receipt['exit_code'] == 0, 'Tensor payload files or failed exit')
    for name in ('runtime', 'control', 'config'):
        require(receipt[name + '_sha256'] == sha(directory / (name + '.json')), 'Receipt source hash differs: ' + name)
    expected = {'observation/step%d/rank0/%s.json' % (step, kind) for step in FORWARDS for kind in ('dump', 'stack', 'construct')}
    require(len(receipt['native_files']) == 6 and {item['path'] for item in receipt['native_files']} == expected, 'Receipt native file set differs')
    for item in receipt['native_files']:
        path = directory / item['path']
        require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'Native file hash/length differs')
    require(set(receipt['dump_nodes']) == {'0', '1'}, 'Receipt step node counts missing')
    for step in FORWARDS:
        require(receipt['dump_nodes'][str(step)] == len(read(directory / ('observation/step%d/rank0/dump.json' % step))['data']), 'Native dump node count differs')
    runtime = read(directory / 'runtime.json')
    registration = runtime['capture_api_registration']
    require(registration.get('restored') is True and registration.get('pending_restorations') == [] and
            registration.get('registrations') == ['Torch.as_strided', 'Tensor.copy_'], 'Official custom APIs not registered/restored as required')
    require(not any(p.suffix.lower() in ('.pt', '.pth', '.npy', '.npz', '.safetensors') for p in directory.rglob('*') if p.is_file()), 'Tensor binary in statistics arm')
    return {'status': 'PASS_CONTROL_AND_COMPLETION', 'hits': control['hits'], 'selected_original_calls': 6,
            'unmatched_calls': 4, 'unmatched_original_calls': 4, 'eligible_events_per_forward': 3,
            'control_sha256': sha(directory / 'control.json'), 'receipt_sha256': sha(directory / 'receipt.json'),
            'scope': 'Separate injection/control/completion gate, never used to select numerical first differences.'}


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
        for row in native['case_comparisons']:
            expected = ('read_original_base', 'consume_original_base') if row['call'] == 0 else ('read_base_before',)
            if row['first_observable_numerical_stage'] not in expected:
                errors['native_phase_consistency.' + row['case_id']] = 'Independently found first difference is inconsistent with fresh/sequential input propagation.'
    status = 'FAIL' if errors else 'PASS_BASELINE' if args.fault is None else 'PASS_ALL_ELIGIBLE_MECHANISM_AND_STATISTICS'
    write(paths['native'], native)
    write(paths['joint'], joint)
    report = {'status': status, 'errors': errors, 'controls': controls,
              'reports': {key: {'file': str(paths[key]), 'sha256': sha(paths[key])} for key in ('native', 'joint')},
              'analyzer_sha256': sha(Path(__file__)),
              'scope': 'All six eligible controlled view calls across two fresh-base forwards; native statistics and sequential runtime evidence are independent. Not model-chain reachability or historical CANN defect reproduction.'}
    write(paths['main'], report)
    print(json.dumps({'status': status, 'errors': errors, 'output': str(args.output)}, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
