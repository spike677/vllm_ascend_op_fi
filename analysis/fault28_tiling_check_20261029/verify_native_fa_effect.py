"""Strict runtime/native-FA ON/guard/recovery diagnostics, independent of msprobe.

Read-only CPU analysis. Model first differences are derived from ordered value
hashes before host selector events are considered. No result here is a substitute
for independent native msprobe localization or historical fault reproduction.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def require(ok, message):
    if not ok:
        raise ValueError(message)


def mark(errors, ok, message):
    if not ok:
        errors.append(message)


def first_value_difference(stock, candidate):
    """Derive actual order from layer/QKV input/output observations, not selector."""
    require(len(stock) == len(candidate) == 28, 'Need 28 actual layer records')
    differences = []
    for a, b in zip(stock, candidate):
        require(a['layer_index'] == b['layer_index'], 'Layer order differs')
        for name in ('query', 'key', 'value'):
            sa = a['target_inputs_before'][name]['logical_sha256']
            sb = b['target_inputs_before'][name]['logical_sha256']
            if sa != sb:
                differences.append({'layer_index': b['layer_index'], 'phase': 'input', 'argument': name,
                                    'baseline_sha256': sa, 'observed_sha256': sb})
        sa, sb = a['target_output']['logical_sha256'], b['target_output']['logical_sha256']
        if sa != sb:
            differences.append({'layer_index': b['layer_index'], 'phase': 'output', 'argument': 'target_output',
                                'baseline_sha256': sa, 'observed_sha256': sb})
    return {'first_observed_value_difference': differences[0] if differences else None,
            'ordered_value_differences': differences,
            'scope': 'Runtime logical target hashes, not msprobe statistics or a causal kernel trace'}


def full_inputs_equal(util, a, b):
    errors = []
    for phase in ('selected_layer_full_inputs_before', 'selected_layer_full_inputs_after'):
        av, bv = a[phase], b[phase]
        require(set(av) == set(bv) == {'query', 'key', 'value', 'attn_mask'}, 'Incomplete full native input set')
        for name in av:
            if av[name] is None or bv[name] is None:
                mark(errors, av[name] is None and bv[name] is None, phase + '.' + name + ': mask presence differs')
            else:
                require(av[name]['hashed_slice'] == bv[name]['hashed_slice'] == 'entire logical tensor', 'Only partial native input captured')
                mark(errors, util.record_projection(av[name]) == util.record_projection(bv[name]), phase + '.' + name + ': full logical value/geometry differs')
    mark(errors, a['scalar_arguments'] == b['scalar_arguments'], 'Native scalar arguments differ')
    mark(errors, a['attention_mask'] == b['attention_mask'] is None, 'Expected unchanged implicit causal SDPA mask')
    return errors


def compare_logits(util, a, b):
    ra, rb = a['requests'], b['requests']
    require(len(ra) == len(rb), 'Logits request count differs')
    for x, y in zip(ra, rb):
        for key in ('prompt_index', 'batch_slot', 'prompt_token_sha256', 'last_token_position'):
            require(x[key] == y[key], 'Logits prefix/request mismatch: ' + key)
    pa, pb = util.record_projection(a['target_logits']), util.record_projection(b['target_logits'])
    metadata_equal = all(pa[k] == pb[k] for k in pa if k != 'logical_sha256')
    return {'metadata_equal': metadata_equal, 'sha_equal': pa['logical_sha256'] == pb['logical_sha256'],
            'baseline_sha256': pa['logical_sha256'], 'observed_sha256': pb['logical_sha256'],
            'argmax_equal': ra[0]['next_token_argmax'] == rb[0]['next_token_argmax'],
            'baseline_argmax': ra[0]['next_token_argmax'], 'observed_argmax': rb[0]['next_token_argmax']}


def model_forward(util, baseline, observed, baseline_forward, observed_forward, expect_change):
    errors = []
    comparisons = [util.compare_calls(a, b) for a, b in zip(baseline, observed)]
    derived = first_value_difference(baseline, observed)
    full_errors = full_inputs_equal(util, baseline[1], observed[1])
    logits = compare_logits(util, baseline_forward, observed_forward)
    first = derived['first_observed_value_difference']
    mark(errors, logits['metadata_equal'], 'Logits geometry/dtype changed')
    if expect_change:
        mark(errors, comparisons[0]['target_inputs_and_output_equal'], 'Layer0 target inputs/output changed before selected layer')
        errors.extend(full_errors)
        # Actual first is computed above; this is a predeclared acceptance assertion only.
        mark(errors, first is not None and first['layer_index'] == 1 and first['phase'] == 'output',
             'Derived first target value difference is not selected layer1 output')
        selected_diffs = comparisons[1]['differences']
        mark(errors, selected_diffs and all(d['field'] == 'target_output.logical_sha256' for d in selected_diffs),
             'Selected API changes input/metadata or lacks output bit difference')
        mark(errors, any(d['layer_index'] > 1 for d in derived['ordered_value_differences']), 'No observed downstream propagation')
        mark(errors, not logits['sha_equal'], 'No observed logits propagation')
    else:
        mark(errors, all(c['target_inputs_and_output_equal'] for c in comparisons), 'Guard/recovery target call differs from OFF')
        errors.extend(full_errors)
        mark(errors, logits['sha_equal'] and logits['argmax_equal'], 'Guard/recovery logits differ from OFF')
    return {'status': 'PASS_MODEL_VALUE_CONTRACT' if not errors else 'FAIL_MODEL_VALUE_CONTRACT',
            'errors': errors, 'expected_selected_output_change': expect_change, **derived,
            'full_native_selected_inputs_equal': not full_errors, 'full_native_input_errors': full_errors,
            'call_comparisons': comparisons, 'logits': logits,
            'msprobe_used': False, 'host_selector_used_for_first_difference': False}


def load_case(util, root, run, case_id, evidence):
    folder = root / 'outputs' / run / case_id
    path = folder / 'result.json'
    result = util.read_json(path)
    evidence[str(path)] = util.sha_file(path)
    calls = util.load_calls(folder, result, evidence)
    require(result['operator_variant'] == 'private_vendor' and result['capture_msprobe'] is False,
            'Expected uncaptured private-vendor diagnostic run')
    require(result['batch_size'] in (1, 8) and result['prompt_length'] == 2048 and result['target_index'] == 0,
            'Unsupported diagnostic workload')
    return {'folder': folder, 'result': result, 'calls': calls}


def native_profiles(util, extractor, case, evidence):
    folder, result = case['folder'], case['result']
    log_path = folder.parent / (result['case_id'] + '.log')
    digest = util.sha_file(log_path)
    with log_path.open('rb') as stream:
        profile = extractor.extract(stream, result['case_id'], result['native_calls'])
    require(util.sha_file(log_path) == digest, 'Native log changed during analysis')
    evidence[str(log_path)] = digest
    for row in profile:
        row['_raw'] = row['_aligned_payload'][:row['declared_raw_bytes']]
    require(len({row['enter']['pid'] for row in profile}) == 1, 'A case must contain one actual model process')
    for row, call in zip(profile, case['calls']):
        require(row['enter']['forward_index'] == call['forward_index'], 'Native marker forward differs from call ledger')
    events_path = folder / 'fa_host_events.jsonl'
    events = [json.loads(x) for x in events_path.read_text(encoding='utf8').splitlines() if x.strip()]
    evidence[str(events_path)] = util.sha_file(events_path)
    require(len(events) == result['forward_count'], 'Expected exactly one host event per forward')
    require(len({e['token'] for e in events}) == len(events), 'Host event tokens reused')
    require(all(event['pid'] == profile[0]['enter']['pid'] for event in events), 'Public callbacks differ from model process')
    case.update(profiles=profile, events=events)
    return profile


def profile_projection(row):
    return {k: row[k] for k in ('tiling_key', 'declared_raw_bytes', 'block_dim', 'kernel_workspace_count')}


def verify_host(util, schema, case, forward, expected_mode, expected_applied):
    event = case['events'][forward]
    call = case['calls'][forward * 28 + 1]
    profile = case['profiles'][forward * 28 + 1]
    scope = call['host_scope']
    errors = []
    require(event['schema'] == 'fault28.fa.typed_host.v1', 'Unknown host schema')
    require(event['token'] == scope['token'] and event['pid'] == scope['pid'] == profile['enter']['pid'] and
            event['tid'] == scope['thread_id'] == profile['enter']['thread_id'], 'Selected public callback/real API scope differs')
    require(event['forward'] == str(forward) and event['layer'] == 1 and event['phase'] == 'prefill', 'Host scope geometry mismatch')
    require(scope['environment_restored'] is True, 'Scoped environment not restored')
    raw = bytes.fromhex(event['raw_hex'])
    require(event['raw_size'] == len(raw), 'Malformed public-context raw length')
    pod = util.decode_pod(raw, schema)
    typed_errors = util.typed_matches(event['typed'], pod)
    mark(errors, not typed_errors, 'Typed/POD mismatch: ' + ','.join(typed_errors))
    mark(errors, raw == profile['_raw'], 'Public-context raw does not match actual native launched prefix')
    for key, expected in (
        ('mode', expected_mode), ('eligible', case['result']['batch_size'] == 8),
        ('policy_applied', expected_applied), ('callback_status', 0),
        ('tiling_key', profile['tiling_key']), ('block_dim', profile['block_dim']),
        ('workspace_count', profile['kernel_workspace_count']),
    ):
        mark(errors, event[key] == expected, 'Host ' + key + ' differs from contract/actual native launch')
    require(len(event['workspace_sizes']) == event['workspace_count'] == 1, 'Missing public workspace vector')
    tile = {'m': 64, 's2_base': 128, 'n_ratio': 4, 's2_group': 512} if expected_applied else {
            'm': 128, 's2_base': 128, 'n_ratio': 8, 's2_group': 1024}
    for key, value in tile.items():
        mark(errors, event['typed'][key] == value, 'Actual typed tile differs: ' + key)
    return {'status': 'PASS_PUBLIC_CONTEXT_AND_CONTROL' if not errors else 'FAIL_PUBLIC_CONTEXT_AND_CONTROL',
            'errors': errors, 'mode': event['mode'], 'eligible': event['eligible'], 'policy_applied': event['policy_applied'],
            'callback_status': event['callback_status'], 'actual_tiling': {k: event['typed'][k] for k in tile},
            'public_raw_sha256': util.sha_bytes(raw), 'actual_native_raw_sha256': util.sha_bytes(profile['_raw']),
            'actual_native_launch': profile['launches'][0], 'schedule_mode': event['schedule_mode'],
            'workspace_sizes': event['workspace_sizes'], 'full_public_pod105': pod}


def derive_without_selector(oracle, event):
    stripped = copy.deepcopy(event)
    for key in ('mode', 'eligible', 'policy_applied', 'token', 'pid', 'tid', 'forward', 'layer', 'phase'):
        stripped.pop(key, None)
    result = oracle.derive(stripped, target_batch=0, target_query=2047)
    require(result['mode_or_hit_fields_used'] is False and result['query_split_between_cores'] is False,
            'Oracle unexpectedly used selector or SplitKV ownership')
    return result


def trace_pair(oracle, before, after, expected_change):
    a, b = derive_without_selector(oracle, before), derive_without_selector(oracle, after)
    errors = []
    for trace in (a, b):
        mark(errors, trace['every_real_query_owned_once'] and trace['each_query_legal_keys_covered_once'],
             'Missing/duplicate causal query/key work')
    heads_a = sorted(a['target_tasks'], key=lambda t: t['query_head'])
    heads_b = sorted(b['target_tasks'], key=lambda t: t['query_head'])
    require([t['query_head'] for t in heads_a] == [t['query_head'] for t in heads_b] == list(range(before['typed']['n1'])),
            'Oracle target head coverage differs')
    group_changes = []
    for x, y in zip(heads_a, heads_b):
        gx, gy = x['target_legal_key_groups'], y['target_legal_key_groups']
        require(gx[0][0] == gy[0][0] == 0 and gx[-1][1] == gy[-1][1] == 2048, 'Target legal causal range changed')
        group_changes.append(gx != gy)
    mark(errors, all(group_changes) if expected_change else not any(group_changes), 'Target reduction grouping violates change/restoration contract')
    # Actual host policy may alter scheduling, but native precision/mask/key semantics must stay fixed.
    fixed = ('template', 'b', 'n1', 'n2', 'g', 's1', 's2', 'd', 'dv', 'dtype_enum', 'layout',
             'calc_bytes', 'high_precision', 'impl_mode', 'sparse_mode', 'has_mask', 'has_dropout', 'has_pse',
             'keep_prob', 'scale', 'sparse_type', 'enable_l1_reuse', 's1_sparse_valid_size', 's2_sparse_valid_size')
    for key in fixed:
        mark(errors, before['typed'][key] == after['typed'][key], 'Precision/mask/key semantics changed: ' + key)
    mark(errors, before['tiling_key'] == after['tiling_key'] and before['schedule_mode'] == after['schedule_mode'],
         'Native kernel key or schedule mode changed')
    return {'status': 'PASS_SOURCE_DERIVED_LEGAL_REDUCTION' if not errors else 'FAIL_SOURCE_DERIVED_LEGAL_REDUCTION',
            'errors': errors, 'target_grouping_changed': all(group_changes),
            'baseline_trace': a, 'observed_trace': b,
            'not_hardware_trace': True, 'not_numerical_equivalence_proof': True,
            'selector_fields_removed_before_derive': True}


def stable_configuration(a, b):
    for key in ('batch_size', 'prompt_length', 'target_index', 'layer_index', 'source_sha256', 'checkpoint_config',
                'versions', 'environment', 'actual_sdpa_context_flags'):
        require(a[key] == b[key], 'OFF/ON source/configuration changed: ' + key)
    require(a['private_vendor']['manifest_sha256'] == b['private_vendor']['manifest_sha256'], 'OFF/ON vendor binary differs')
    for key in a['options']:
        if key != 'full_prefill_forward_count':
            require(a['options'][key] == b['options'][key], 'OFF/ON execution option changed: ' + key)
    require(a['environment']['ACLNN_CACHE_LIMIT'] == '0', 'Executor/tiling cache must be off')


def analyze_case(util, oracle, schema, baseline, observed, modes, on_reference=None):
    result = {'case_id': observed['result']['case_id'], 'status': 'FAIL_NATIVE_FA_EFFECT_DIAGNOSTICS',
              'errors': [], 'forwards': []}
    stable_configuration(baseline['result'], observed['result'])
    require(observed['result']['forward_count'] == len(modes) == 2, 'Need two actual forwards')
    batch = observed['result']['batch_size']
    require(observed['result']['fa_mode_plan']['modes'] == modes, 'Runtime mode plan differs')
    require(len({e['pid'] for e in observed['events']}) == 1, 'Two forwards must execute in same process')
    base_calls = baseline['calls'][:28]
    for forward, mode in enumerate(modes):
        changed = mode == 'on' and batch == 8
        calls = observed['calls'][forward * 28:(forward + 1) * 28]
        numerical = model_forward(util, base_calls, calls, baseline['result']['forward_results'][0],
                                  observed['result']['forward_results'][forward], changed)
        # Preserve the independently computed numerical result even if a host/trace
        # document is malformed. Failed controls never erase or supply its first.
        try:
            host = verify_host(util, schema, observed, forward, mode, changed)
        except Exception as exc:
            host = {'status': 'FAIL_PUBLIC_CONTEXT_AND_CONTROL', 'errors': [type(exc).__name__ + ': ' + str(exc)]}
        try:
            trace = trace_pair(oracle, baseline['events'][0], observed['events'][forward], changed)
        except Exception as exc:
            trace = {'status': 'FAIL_SOURCE_DERIVED_LEGAL_REDUCTION', 'errors': [type(exc).__name__ + ': ' + str(exc)]}
        raw_rows, raw_errors = [], []
        for layer in range(28):
            base, actual = baseline['profiles'][layer], observed['profiles'][forward * 28 + layer]
            equal = base['_raw'] == actual['_raw']
            geometry_equal = profile_projection(base) == profile_projection(actual)
            should_change = changed and layer == 1
            mark(raw_errors, equal != should_change, 'Actual native raw change scope differs at layer ' + str(layer))
            mark(raw_errors, geometry_equal, 'Native route/key/block/workspace-count differs at layer ' + str(layer))
            raw_rows.append({'layer_index': layer, 'raw_equal_to_off': equal, 'raw_expected_to_change': should_change,
                             'baseline_sha256': util.sha_bytes(base['_raw']), 'observed_sha256': util.sha_bytes(actual['_raw']),
                             'native_route_equal': geometry_equal})
        repeated_errors = []
        if forward == 1 and modes[0] == modes[1]:
            first = observed['calls'][:28]
            mark(repeated_errors, all(util.compare_calls(a, b)['target_inputs_and_output_equal'] for a, b in zip(first, calls)),
                 'Same mode repeated forward target values differ')
            mark(repeated_errors, compare_logits(util, observed['result']['forward_results'][0],
                                                observed['result']['forward_results'][forward])['sha_equal'],
                 'Same mode repeated forward logits differ')
        if on_reference is not None and mode == 'on':
            ref = on_reference['calls'][:28]
            mark(repeated_errors, all(util.compare_calls(a, b)['target_inputs_and_output_equal'] for a, b in zip(ref, calls)),
                 'Recovery initial ON differs from independent ON run')
            mark(repeated_errors, compare_logits(util, on_reference['result']['forward_results'][0],
                                                observed['result']['forward_results'][forward])['sha_equal'],
                 'Recovery initial ON logits differ from independent ON run')
        errors = numerical['errors'] + host['errors'] + trace['errors'] + raw_errors + repeated_errors
        result['forwards'].append({'forward_index': forward, 'status': 'PASS' if not errors else 'FAIL',
                                   'errors': errors, 'numerical': numerical, 'public_host_control': host,
                                   'legal_reduction': trace, 'all_layer_actual_raw_comparisons': raw_rows,
                                   'repeat_or_recovery_consistency_errors': repeated_errors})
        result['errors'].extend('forward' + str(forward) + ': ' + e for e in errors)
    if not result['errors']:
        result['status'] = 'PASS_NATIVE_FA_EFFECT_DIAGNOSTICS'
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True, type=Path)
    ap.add_argument('--off-run', default='private_off_compat_a2')
    ap.add_argument('--on-run', default='private_on_validation_a2')
    ap.add_argument('--stock-off-comparator', required=True, type=Path)
    ap.add_argument('--extractor', required=True, type=Path)
    ap.add_argument('--trace-oracle', required=True, type=Path)
    ap.add_argument('--source-root', required=True, type=Path)
    ap.add_argument('--schema', required=True, type=Path)
    ap.add_argument('--vendor-manifest', required=True, type=Path)
    ap.add_argument('--off-compat-report', required=True, type=Path)
    ap.add_argument('--output', default='-')
    args = ap.parse_args()
    util = module(args.stock_off_comparator, 'stock_off_util')
    extractor = module(args.extractor, 'native_fa_extractor')
    oracle = module(args.trace_oracle, 'native_fa_trace_oracle')
    paths = (args.stock_off_comparator, args.extractor, args.trace_oracle, args.schema,
             args.vendor_manifest, args.off_compat_report)
    evidence = {str(p): util.sha_file(p) for p in paths}
    if Path(__file__).is_file():
        evidence[str(Path(__file__).resolve())] = util.sha_file(Path(__file__))
    report = {'schema': 'fault28.native_fa.effect_diagnostics.v1', 'status': 'FAIL_NATIVE_FA_EFFECT_DIAGNOSTICS',
              'mechanism_accepted': False, 'msprobe_used': False, 'errors': [], 'cases': [],
              'source_evidence_sha256': evidence,
              'limits': ['This diagnostic is separate from independent native msprobe first-difference inference.',
                         'Source-derived ownership/key coverage is not a hardware execution trace or rounding oracle.',
                         'Logical hashes witness captured target values; private raw tensors are not loaded here.',
                         'No claim of historical 38B fault reproduction or all-input generality.']}
    try:
        compatibility = util.read_json(args.off_compat_report)
        require(compatibility['status'] == 'PASS_STOCK_OFF_COMPATIBILITY', 'Prerequisite strict stock/OFF compatibility failed')
        for relative, expected in oracle.SOURCE_FILES.items():
            path = args.source_root / relative
            require(util.sha_file(path) == expected, 'Source-derived oracle kernel source mismatch: ' + relative)
            evidence[str(path)] = expected
        schema = util.read_json(args.schema)
        baselines = {b: load_case(util, args.root, args.off_run, 'fa_off_b' + str(b), evidence) for b in (1, 8)}
        cases = {name: load_case(util, args.root, args.on_run, name, evidence) for name in
                 ('fa_on_b1_guard', 'fa_on_b8', 'fa_recover_b8')}
        for case in list(baselines.values()) + list(cases.values()):
            native_profiles(util, extractor, case, evidence)
            actual_log = util.scan_native_log(case['folder'].parent / (case['result']['case_id'] + '.log'),
                                              case['calls'], case['result']['private_vendor']['approved_custom_opp_path'])
            case['vendor_verification'] = util.verify_vendor(args.root, case['folder'], case['result'],
                                                             util.read_json(args.vendor_manifest), args.vendor_manifest, actual_log)
        tokens = [e['token'] for case in cases.values() for e in case['events']]
        require(len(tokens) == len(set(tokens)) == 6, 'Expected six distinct actual ON/guard/recovery callback scopes')
        # Bind prerequisite report to these actual baseline files, not just a status string.
        for baseline in baselines.values():
            path = str(baseline['folder'] / 'result.json')
            require(compatibility['source_evidence_sha256'].get(path) == evidence[path], 'Compatibility report refers to another OFF result')
            for forward in range(baseline['result']['forward_count']):
                check = verify_host(util, schema, baseline, forward, 'off', False)
                require(not check['errors'], 'OFF reference public context/control gate failed')
            if baseline['result']['forward_count'] == 2:
                check = model_forward(util, baseline['calls'][:28], baseline['calls'][28:],
                                      baseline['result']['forward_results'][0], baseline['result']['forward_results'][1], False)
                require(not check['errors'], 'OFF repeated forward is unstable')
        specifications = [('fa_on_b1_guard', 1, ['on', 'on'], None),
                          ('fa_on_b8', 8, ['on', 'on'], None),
                          ('fa_recover_b8', 8, ['on', 'off'], cases['fa_on_b8'])]
        for name, batch, modes, reference in specifications:
            try:
                result = analyze_case(util, oracle, schema, baselines[batch], cases[name], modes, reference)
                result['vendor_verification'] = cases[name]['vendor_verification']
                report['cases'].append(result)
            except Exception as exc:
                report['errors'].append({'case_id': name, 'error_type': type(exc).__name__, 'message': str(exc)})
    except Exception as exc:
        report['errors'].append({'error_type': type(exc).__name__, 'message': str(exc)})
    if not report['errors'] and len(report['cases']) == 3 and all(c['status'] == 'PASS_NATIVE_FA_EFFECT_DIAGNOSTICS' for c in report['cases']):
        report['status'] = 'PASS_NATIVE_FA_EFFECT_DIAGNOSTICS'
    payload = json.dumps(report, ensure_ascii=True, indent=2) + '\n'
    if args.output == '-':
        sys.stdout.write(payload)
    else:
        require(not Path(args.output).exists(), 'Output must be fresh')
        Path(args.output).write_text(payload, encoding='utf8')
    return 0 if report['status'] == 'PASS_NATIVE_FA_EFFECT_DIAGNOSTICS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
