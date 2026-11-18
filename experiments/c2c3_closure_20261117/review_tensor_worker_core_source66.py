"""Read-only public Core8/18 original receipts; standard library only.

Reuses the independent saved-coverage auditor and MoE producer join, while
checking the new actual-request protocol and original live C2 receipt chain.
"""
import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import struct


SOURCE_COMMIT = '239096ef14a3f5f4be15311d879aa8e67f77def4'
SOURCE_MANIFEST = 'e7b238d9554ae6095a1dc09f94db64a298d80b31e04fcb841d6d6fcbba20dc3c'
HELPER_SHA = '8151f93dca35ead3b32b98d66881e5bc23432884bf5af8c1c6694f1b59fb8b5d'
MOE_SHA = '790a2ece83ed24ac40646692b701392865b183cec008c7aa5462ae292c277654'
AUDIT_SHA = '55af44a8acd8e2530209d6c75a941172b5201fa1088796da033f9a0bb45feca4'
PHASES = ('native_baseline', 'baseline', 'baseline_repeat', 'lossless', 'c2_fault', 'c2_off',
          'c3_fault_0', 'c3_fault_1', 'c3_off', 'cleanup_healthy')
FAULTS = (4, 6, 7)
ACTIONS = ['attach', 'prepare', 'register', 'check', 'enable'] + ['advance', 'query', 'advance']*10 + ['finish', 'disable', 'unregister', 'detach']
CORE = 'src/demoncat/runtime/vllm_ascend/worker_extension/'
COMMON = [CORE + 'providers/tensor_natural.py'] + ['fault_injection/' + name + '.py' for name in (
    'rmsnorm_return', 'index22', 'tensor_observer', 'tensor_callable', '_scoped_dispatch', 'module_faults',
    'route8', 'moe_census', 'moe_restore', 'conv3d_natural')] + [CORE + 'serving_tensor.py', CORE + 'serving_moe.py']
BINDINGS = [CORE + p + '.py' for p in ('providers/tensor_serving', 'serving_tensor_binding', 'serving_tensor_request', 'registry')]
SOURCES = COMMON + BINDINGS + [CORE + p + '.py' for p in ('config', 'host', 'runtime', 'extension', 'identity',
    'providers/tensor_modules', 'providers/_scope', 'serving_context', 'call_binding')] + [
    'fault_injection/coverage.py', 'fault_injection/coverage_session.py', 'fault_injection/coverage_audit.py',
    'fault_injection/runtime_discovery.py', 'experiments/c2c3_closure_20261117/natural_tensor_worker_core_probe.py']
COVERAGE = ('schema session_id scope selection trigger owner generation budget discovery observation '
    'planned_targets planned_targets_hit observed_calls observed_eligible_calls hits missing_outcomes '
    'propagation_added_calls calls recovery contract_satisfied rejection_reasons real_worker_evidence '
    'real_pretrained_model_evidence public_qualification_granted').split()
NATIVE_RMS = {namespace + '.' + name + '.default' for namespace in ('npu', '_C_ascend')
              for name in ('npu_rms_norm', 'npu_add_rms_norm', 'npu_add_rms_norm_bias')}


def load_module(path, name, expected):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        raise ValueError('REVIEWED_READER_CHANGED:' + name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


H = load_module(Path(__file__).with_name('review_native26_single_source54.py'), '_tensor_artifacts', HELPER_SHA)
Review, file_sha, read, under, valid_sha, positive = H.Review, H.file_sha, H.read, H.resolve_under, H.valid_sha, H.positive


def sha(value):
    h = hashlib.sha256()
    for block in json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False).iterencode(value):
        h.update(block.encode())
    return h.hexdigest()


def require(value, reason):
    if not value:
        raise ValueError(reason)


def moe_checks():
    """Reuse only the frozen pure-JSON functions, not the historical CLI imports."""
    path = Path(__file__).with_name('review_moe8_group_124.py'); raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == MOE_SHA, 'REVIEWED_MOE_JOIN_CHANGED')
    tree = ast.parse(raw, filename=str(path))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in ('check_native', 'check_producer')]
    require(len(selected) == 2, 'MISSING_FROZEN_MOE_FUNCTION')
    namespace = {'require':require}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def signature(event, mechanism):
    value = {k:event[k] for k in ('path', 'ordinal', 'output_shape', 'dtype', 'input_shape', 'weight_shape', 'boundary') if k in event}
    if mechanism == '8.restore_offset':
        value.update(token_groups=event['token_groups'], topk=event['topk'])
    return value


def source_review(a, root):
    p = under(root, 'source_manifest.json'); manifest = read(p)
    a.check(file_sha(p) == SOURCE_MANIFEST and manifest['base_commit'] == SOURCE_COMMIT, 'FROZEN_SOURCE66_MANIFEST')
    files = {r['path']:r['sha256'] for r in manifest['files']}
    a.check(len(files) == len(manifest['files']), 'UNIQUE_SOURCE_MANIFEST')
    for name in SOURCES:
        a.check(file_sha(under(root, name)) == files[name], 'FROZEN_SOURCE_REHASH', name)
    common = sha({Path(name).name:files[name] for name in COMMON})
    bindings = {Path(name).name:files[name] for name in BINDINGS}
    implementation = sha(dict(group=common, worker_rpc_binding=bindings))
    auditor = load_module(under(root, 'fault_injection/coverage_audit.py'), '_saved_tensor_coverage', AUDIT_SHA)
    return dict(implementation_sha256=implementation, common_implementation_sha256=common, binding_sources=bindings,
                probe_sha256=files[SOURCES[-1]], files=files), auditor


def load_artifacts(a, path, expected):
    a.check(file_sha(path) == expected and valid_sha(expected), 'ORIGINAL_RESULT_SHA256')
    require(not a.errors, 'ORIGINAL_RESULT_CHANGED')
    d = read(path); core = d['worker_core']; root = Path(path).parent
    host_ref = core['final_host_receipt']; host_path = under(root, host_ref['path'])
    a.check(host_ref['path'] == 'host_receipt.json' and file_sha(host_path) == host_ref['sha256'], 'ORIGINAL_HOST_FILE_SHA256')
    require(not a.errors, 'ORIGINAL_HOST_CHANGED')
    # Host may be larger than the bounded per-result helper limit. Read it once,
    # and compare each standalone RPC in turn without copying the full history.
    def invalid(value): raise ValueError('NONFINITE_JSON:' + value)
    with host_path.open(encoding='utf-8') as stream:
        host = json.load(stream, object_pairs_hook=H.unique_pairs, parse_constant=invalid)
    refs, history = core['receipts'], host['history']
    a.check(len(refs) == len(history), 'ALL_ORIGINAL_RPC_FILES_RETAINED')
    for index, (ref, row) in enumerate(zip(refs, history)):
        name = 'worker_core_rpc/' + str(index).zfill(3) + '-' + row['action'] + '.json'
        a.check(ref['path'] == name and ref['action'] == row['action'], 'ORIGINAL_RPC_PATH_ORDER', index)
        p = under(root, ref['path'])
        a.check(file_sha(p) == ref['sha256'] and read(p) == row, 'ORIGINAL_RPC_BYTES_AND_HOST_HISTORY', index)
    a.check(core['last_scope'] == {k:v for k,v in host.items() if k not in ('history', 'response', 'capability_enumeration')},
            'ORIGINAL_FINAL_HOST_SURFACE')
    return d, host


def scheduler(a, item, baseline, natural):
    row, mapping = item['worker'], item['mapping']; request = row['request_id']
    a.check(mapping['internal_request_id'] == request and positive(mapping['request_state_object_id'])
        and isinstance(mapping['external_request_id'], str) and bool(mapping['external_request_id'])
        and mapping['source'] == 'ACTUAL_PUBLIC_ENQUEUE_AND_OUTPUT_PROCESSOR', 'ACTUAL_DRIVER_REQUEST_MAP', item['phase'])
    if natural:
        s = row['scheduler']; b = baseline['scheduler']; n = len(b['token_ids'])
        a.check(s['internal_request_id'] == request and s['num_requests'] == 1 and s['num_computed_tokens'] == 0
            and s['num_prompt_tokens'] == s['num_scheduled_tokens'] == n >= 2
            and s['token_ids'] == b['token_ids'] and s['positions'] == list(range(n)), 'ACTUAL_FRESH_PREFILL_SCHEDULER', item['phase'])
        a.check(positive(s['request_object_id']) and s['token_ids_sha256'] == sha(s['token_ids'])
            and len(set(s['block_ids'])) == len(s['block_ids']) and positive(s['block_size'])
            and s['slot_mapping'] == [s['block_ids'][pos//s['block_size']]*s['block_size']+pos%s['block_size']
                for pos in s['positions']], 'ACTUAL_SCHEDULER_TOKEN_AND_SLOT_MAPPING', item['phase'])
        a.check(row['input'][0] == [] and {'input_ids', 'positions'}.issubset(row['input'][1])
            and set(row['input'][1]).issubset({'input_ids', 'positions', 'intermediate_tensors', 'inputs_embeds'})
            and row['input'][1].get('intermediate_tensors') is row['input'][1].get('inputs_embeds') is None,
            'ACTUAL_TEXT_ONLY_INPUT_STRUCTURE', item['phase'])
        for key, values, dtype, code in (('input_ids', s['token_ids'], 'torch.int32', 'i'),
                                       ('positions', s['positions'], 'torch.int64', 'q')):
            meta, captured = s['input_tensor_metadata'][key], row['input'][1][key]
            a.check(meta['shape'] == captured['shape'] == [n] and meta['stride'] == [1]
                and meta['dtype'] == captured['dtype'] == dtype and meta['device'] == 'npu:0'
                and positive(meta['object_id']) and positive(meta['storage_identity'])
                and type(meta['storage_offset']) is int and meta['storage_offset'] >= 0
                and all(type(v) is int for v in values),
                'ACTUAL_INPUT_TENSOR_METADATA:' + key, item['phase'])
            # tensor_modules._identity hashes the complete contiguous bytes,
            # without serving.complete_tensor's separate dtype/shape header.
            a.check(captured['finite'] is True and captured['sha256'] ==
                hashlib.sha256(struct.pack('<' + code*len(values), *values)).hexdigest(),
                'RECOMPUTED_COMPLETE_DEVICE_INPUT_DIGEST:' + key, item['phase'])
        if 'independent' in row:
            observed = row['independent']['forwards'][0]['input']
            a.check(len(observed) == 2, 'ROOT_OBSERVER_ACTUAL_INPUT_COUNT', item['phase'])
            for observed_meta, key in zip(observed, ('input_ids', 'positions')):
                meta = s['input_tensor_metadata'][key]
                a.check(all(observed_meta[k] == meta[k] for k in ('shape', 'stride', 'dtype', 'device', 'object_id'))
                    and observed_meta['storage_id'] == meta['storage_identity']
                    and observed_meta['offset'] == meta['storage_offset'], 'SCHEDULER_ROOT_OBSERVER_SAME_INPUT_OBJECT', item['phase'])
    else:
        a.check(row['scheduler']['request_id'] == request, 'CPU_ACTUAL_REQUEST_MAP', item['phase'])


def phase(a, item, index, baseline, candidates, mechanism, natural, moe):
    row = item['worker']; name = PHASES[index]; fault = index in FAULTS
    a.check(row['phase'] == item['phase'] == name and row['input'] == baseline['input']
        and row['logits_boundary_returned'] is True, 'FULL_ACTUAL_REQUEST_INPUT_AND_LOGITS', name)
    scheduler(a, item, baseline, natural)
    for k in ('hidden', 'logits'):
        value = row['output'][k]; b = baseline['output'][k]
        a.check(value['shape'] == b['shape'] and value['dtype'] == b['dtype'] and value['finite'] is True
            and valid_sha(value['sha256']) and ((value == b) is (not fault)), 'COMPLETE_MODEL_EFFECT_OR_EXACT_OFF:' + k, name)
    independent = row['independent']; events = (row['mechanism'] or {}).get('events', [])
    a.check(independent['complete'] is True and independent['dropped_calls'] == independent['observer_hooks']
        == independent['inflight'] == independent['tensor_mutations'] == 0
        and len(independent['forwards']) == 1 and independent['forwards'][0]['status'] == 'returned', 'INDEPENDENT_CENSUS_CLOSED', name)
    root_output = independent['forwards'][0]['output']
    a.check(len(root_output) == 1 and all(root_output[0][k] == row['output']['hidden'][k] for k in ('shape', 'dtype')),
            'INDEPENDENT_ACTUAL_ROOT_OUTPUT_GEOMETRY', name)
    keys = [list(k) for k in candidates]
    a.check([c['key'] for c in row['calls']] == keys and len(events) == (0 if index in (0, 5, 8) else len(keys)),
        'EVERY_NATURAL_CALL_AND_MECHANISM_EVENT', name)
    expected_events = [r for r in independent['events'] if r['kind'] == 'operator' and r['operator'] in NATIVE_RMS]
    if mechanism == '18.api_half':
        native_paths = set()
        for e in expected_events:
            a.check(len(e['weight_owner_paths']) == 1 and len(e['weight_owner_sources']) == 1
                and e['original_returned'] is True, 'INDEPENDENT_NATIVE_RMS_WEIGHT_OWNER', name)
            native_paths.update(e['weight_owner_paths'])
        module_events = [r for r in independent['events'] if r['kind'] == 'module_return'
            and r['role'] == 'rmsnorm_source_candidate' and r['path'] not in native_paths]
        expected = {(e['weight_owner_paths'][0], e['operator']):e for e in expected_events}
        expected.update({(e['path'], 'RMSNorm.forward'):e for e in module_events})
        supported_paths = {k[0] for k in candidates}
        actual = {(c['path'], c['operator']) for c in row['calls']}
        native_keys = {(e['weight_owner_paths'][0], e['operator']) for e in expected_events}
        a.check(native_keys.issubset(actual) and len(native_keys) == len(expected_events)
            and actual == {k for k in expected if k[0] in supported_paths}
            and len(actual) == len(row['calls']), 'INDEPENDENT_RMS_ACTUAL_DENOMINATOR', name)
        a.check(all(independent['operator_counts'].get(op, 0) == sum(e['operator'] == op for e in expected_events)
            for op in NATIVE_RMS), 'ALL_INDEPENDENT_NATIVE_RMS_OPERATORS_RETAINED', name)
        for call in row['calls']:
            e = expected[(call['path'], call['operator'])]
            a.check(call['metadata'] == e['input'] and e['forward'] == 0, 'INDEPENDENT_RMS_CALL_INPUT_JOIN', name)
            target = candidates[tuple(call['key'])]
            identities = e['weight_owner_sources'] if call['operator'] in NATIVE_RMS else [m for m in independent['module_inventory'] if m['path'] == call['path']]
            a.check(len(identities) == 1 and identities[0]['module_object_id'] == target['module_object_identity'],
                'ACTUAL_RMS_WEIGHT_OWNER_MODULE_IDENTITY', name)
        for event in events:
            a.check(event['gain'] == (0.5 if fault else 1.0) and event['output_half_proxy'] is True
                and event['native_reduce_error_claimed'] is False and event['auxiliary_return_storage_unchanged'] is True,
                'ACTUAL_RMS_RETURN_GAIN_SEMANTICS', name)
            if event['operator'] != 'RMSNorm.forward':
                e = expected[(event['path'], event['operator'])]
                a.check(event['native_calls'] == 1 and event['native_operator'] == e['operator'] and event['schema'] == e['schema']
                    and any(v['object_id'] == event['weight_object_id'] for v in e['input'])
                    and event['native_output_storage'] == e['output'][0]['storage_id'], 'RMS_ORIGINAL_NATIVE_RETURN_JOIN', name)
    else:
        trace = independent['moe_trace']; paths = trace['structural_moe_paths']
        natives = [r for r in trace['native_operators'] if r['operator'] == 'npu.npu_moe_token_unpermute.default']
        a.check(trace['complete'] is True and not trace['errors'] and trace['simulation'] is (not natural)
            and trace['model_methods_replaced'] is False and trace['operator_reexecution'] is False
            and len(paths) == len(set(paths)) == trace['observed_unpermute_calls'] == len(natives) == len(keys)
            and set(paths) == {k[0] for k in candidates} == {r['model_path'] for r in natives}, 'ALL_STRUCTURAL_AND_ACTUAL_MOE_BOUNDARIES', name)
        a.check([r['model_path'] for r in natives] == [c['path'] for c in row['calls']], 'INDEPENDENT_MOE_CALL_ORDER', name)
        a.check([r['sequence'] for r in trace['python_boundaries']] == list(range(len(trace['python_boundaries'])))
            and [r['sequence'] for r in trace['native_operators']] == list(range(len(trace['native_operators'])))
            and all(r['request_id'] == row['request_id'] and r['returned'] is True for r in trace['python_boundaries'])
            and trace['observed_unpermute_per_request'] == {row['request_id']:len(natives)},
            'ALL_MOE_ORIGINAL_SEQUENCE_AND_REQUEST_IDENTITIES', name)
        for native, call in zip(natives, row['calls']):
            a.check(native['input'] == call['metadata'] and native['request_id'] == row['request_id']
                and native['original_calls'] == 1 and native['original_returned'] is True, 'ACTUAL_MOE_ORIGINAL_CALL_AND_INPUT', name)
            if natural:
                moe['check_native'](native, fault)
            else:
                original, consumed = native['arguments'], native['consumed_arguments']
                a.check(original['permuted_tokens'] == consumed['permuted_tokens']
                    and all((original[k]['object_id'] == consumed[k]['object_id']) is (not fault)
                        for k in ('sorted_indices', 'probs')), 'CPU_MOE_CONSUMED_OBJECT_JOIN', name)
            proof = native['producer_proof']; sequences = proof['source_sequences']
            a.check(len(sequences) == len(set(sequences)) == 5
                and all(type(v) is int and 0 <= v < len(trace['python_boundaries']) for v in sequences)
                and proof['path'] == native['model_path'] and proof['request_id'] == row['request_id']
                and proof['combine_sequence'] == native['combine_sequence'] == sequences[-1]
                and proof['pipeline_sequence'] == sequences[1] and proof in trace['producer_joins'],
                'CURRENT_MOE_PRODUCER_SEQUENCES_NOT_REUSED', name)
            moe['check_producer'](native, trace, row['request_id'])
            if events:
                event = next(e for e in events if e['path'] == native['model_path'])
                n = event['token_groups']; groups = list(range(1, n)) + [n-1] if fault else list(range(n))
                a.check(event['actual_producer_join'] == native['producer_proof']
                    and event['native_calls'] == event['observed_original_native_calls'] == 1
                    and event['source_groups'] == groups and event['topk'] == native['arguments']['probs']['shape'][1]
                    and all(event[k] is True for k in ('paired_group_readback_exact', 'input_storage_unchanged',
                        'consumer_returned', 'complete_permutation_value_verified', 'complete_native_output_oracle_exact')),
                    'COMPLETE_MOE_GROUP_VALUES_CURRENT_PRODUCER_AND_ORACLE', name)
    for event, call in zip(events, row['calls']):
        key = (event['path'], event['ordinal']); target = candidates[key]
        a.check(event['path'] == call['path'] and event['ordinal'] == call['ordinal'] and list(key) == call['key']
            and event['operator'] == call['operator'] == target['operator_api']
            and signature(event, mechanism) == json.loads(target['tensor_signature']), 'CURRENT_TARGET_SIGNATURE', name)
        a.check(event['original_execution_verified'] is True and event['applied'] is fault
            and event['return_readback_exact'] is True and event['output_changed'] is fault
            and bool(event['consumers']) and all(c['returned'] is True for c in event['consumers']), 'EVERY_TARGET_EFFECT_READBACK_AND_CONSUMER', name)
        if mechanism == '18.api_half':
            a.check(all(c['actual_return_lineage'] is True for c in event['consumers']), 'ACTUAL_RMS_RETURN_CONSUMPTION', name)
    cleanup = row['cleanup']
    a.check(cleanup['cleanup_confirmed'] is True and cleanup['hook_owned'] is False
        and cleanup['inflight'] == cleanup['leases'] == 0, 'PHASE_RESOURCES_REMOVED', name)


def canonical_index(d):
    """Rehash exact recorded JSON objects; no synthetic execution or tensor bytes."""
    result = {}; rows = [item['worker'] for item in d['requests']]
    def add(value, pointer):
        result.setdefault(sha(value), []).append({'recorded_json_pointer':pointer})
    add(dict(baseline=rows[1], repeat=rows[2], lossless=rows[3]), '/requests/{1,2,3}/worker')
    for name in ('c2', 'c3'): add(d[name]['phase_rows'], '/' + name + '/phase_rows')
    for i, row in enumerate(rows):
        add(row, f'/requests/{i}/worker')
        for j, event in enumerate((row['mechanism'] or {}).get('events', [])):
            add(event, f'/requests/{i}/worker/mechanism/events/{j}')
            add(dict(event=event, model_output=row['output']), f'/requests/{i}/worker/event+output/{j}')
        for call in row['calls']: add(dict(call=call['key'], output=row['output']), f'/requests/{i}/worker/call+output')
    return result


def coverage(a, d, candidates, auditor):
    references = canonical_index(d); n = len(candidates); summaries = {}
    for name, start, end, repeat in (('c2', 4, 6, 1), ('c3', 6, 9, 2)):
        group = d[name]; rows = [item['worker'] for item in d['requests'][start:end]]
        a.check(group['selection'] == 'all_eligible' and group['trigger'] == ('once' if repeat == 1 else 'every_match')
            and group['budget'] == dict(max_calls=n*repeat, max_seconds=600, max_concurrency=n), 'BOUNDED_ORIGINAL_GROUP_POLICY', name)
        a.check(group['phase_rows'] == rows and group['observation']['artifact_sha256'] == sha(rows), 'ACTUAL_GROUP_REQUEST_ROWS', name)
        checked = auditor.audit_saved_coverage(group, artifact_index=references, c2_report=d['c2'] if name == 'c3' else None)
        a.check(checked['report_consistent'] is True and not checked['errors'] and not checked['missing_artifact_sha256'],
                'INDEPENDENT_COVERAGE_LEDGER:' + ','.join(checked['errors']), name)
        calls = group['calls']; a.check(len(calls) == n*(repeat+1), 'ACTUAL_GROUP_CALL_COUNT', name)
        hits = Counter()
        for index, call in enumerate(calls):
            phase_index, ordinal = divmod(index, n); row = rows[phase_index]; actual = row['calls'][ordinal]
            target = candidates[tuple(actual['key'])]; injecting = phase_index < repeat
            a.check(call['call_id'] == row['invocation'] + ':' + str(ordinal) and call['forward_id'] == row['invocation']
                and call['request_ids'] == [row['request_id']] and call['target_id'] == sha(target)
                and call['decision']['ticket']['call']['target'] == target and call['operator'] == actual['operator'], 'ACTUAL_REQUEST_TO_COVERAGE_CALL', name)
            receipt = call['receipt']
            if injecting:
                event = row['mechanism']['events'][ordinal]
                a.check(receipt['outcome'] == 'HIT' and receipt['native_applied'] is True
                    and call['decision']['admitted'] is True and receipt['execution_sha256'] == sha(event)
                    and receipt['effect_sha256'] == sha(dict(event=event, model_output=row['output'])), 'ACTUAL_EVENT_EFFECT_RECEIPT_HASH', name)
                hits[sha(target)] += 1
            else:
                a.check(call['phase'] == 'OFF_RECOVERY' and receipt['outcome'] == 'PASSTHROUGH'
                    and receipt['native_applied'] is False and call['decision']['admitted'] is False
                    and receipt['execution_sha256'] == sha(dict(call=actual['key'], output=row['output'])), 'OFF_PASSTHROUGH_RECEIPT', name)
        a.check(set(hits) == {sha(t) for t in candidates.values()} and all(v == repeat for v in hits.values()), 'EACH_TARGET_EXACT_ONCE_OR_TWICE', name)
        a.check(group['discovery']['candidates_sha256'] == sha(d['qualification']['candidate_set'])
            and group['discovery']['artifact_sha256'] == d['qualification']['independent_discovery_sha256']
            == sha(dict(baseline=d['requests'][1]['worker'], repeat=d['requests'][2]['worker'], lossless=d['requests'][3]['worker'])),
            'RECOMPUTED_ORIGINAL_HEALTHY_DISCOVERY', name)
        for recovered in group['recovery']['targets']:
            a.check(recovered['proof_sha256'] == sha(rows[-1]) and recovered['off_output_sha256'] == sha(rows[-1]['output'])
                and recovered['lease']['device_bytes'] == recovered['lease']['host_bytes'] == 0, 'TARGET_EXACT_OFF_PROOF_AND_ZERO_PERSISTENT_BYTES', name)
        summaries[name] = dict(targets=n, actual_calls=len(calls), recomputed_hits=sum(hits.values()), generation=group['generation'])
    return summaries, len(references)


def live_proof(a, d, worker):
    c2, c3 = d['c2'], d['c3']; x, y = c2['live_c2_proof'], c3['live_c2_proof']
    a.check(len(x['issued']) == 1 and x['consumed'] is None and y['issued'] == [], 'ONE_ORIGINAL_C2_ISSUANCE')
    issued = x['issued'][0]; used = y['consumed']
    expected = dict(scope=c2['scope'], candidates_sha256=sha(d['qualification']['candidate_set']),
        target_ids=sorted(c2['discovery']['observed_target_ids']), owner=c2['owner'], generation=c2['generation'],
        session_id=c2['session_id'], report_sha256=sha({k:c2[k] for k in COVERAGE}),
        discovery_sha256=c2['discovery']['artifact_sha256'], observation_sha256=c2['observation']['artifact_sha256'], recovery_sha256=sha(c2['recovery']))
    a.check(issued['proof'] == expected and issued['proof_sha256'] == sha(expected), 'C2_PROOF_BOUND_TO_COMPLETE_ORIGINAL_REPORT')
    consumer = dict(session_id=c3['session_id'], scope=c3['scope'], owner=c3['owner'], generation=c3['generation'],
        candidates_sha256=c3['discovery']['candidates_sha256'], target_ids=sorted(c3['discovery']['observed_target_ids']))
    a.check(used == dict(issued, consumer=consumer) and c3['scope'] == c2['scope'] and c3['owner'] == c2['owner']
        and c3['generation'] == c2['generation']+1 and c3['discovery']['candidates_sha256'] == c2['discovery']['candidates_sha256'],
        'SAME_ORIGINAL_LIVE_PROOF_OBJECT_CONSUMED_BY_C3')
    a.check(issued['exact_issued_object_verified'] is True and issued['object_binding']['pid'] == worker['pid']
        and positive(issued['object_binding']['python_object_id']) and issued['report_hash_scope'] == 'coverage_report_without_live_c2_proof'
        and issued['serializable_authority'] is False and issued['qualification_granted'] is False, 'LIVE_OBJECT_PROCESS_AND_AUTHORITY_BOUNDARY')


def final_cleanup(a, d, host):
    """Report release only after an original zero-resource unregister reply."""
    released = host.get('ownership_retained') is False
    if not released:
        return dict(resource_release_confirmed=False, ownership_retained=True)
    rows = [r['response'] for r in host['history'] if r['action'] == 'unregister'
            and r['response']['status'] == 'ACTION_COMPLETED']
    a.check(len(rows) == 1, 'SUCCESSFUL_ORIGINAL_UNREGISTER_REQUIRED_FOR_RELEASE')
    if len(rows) != 1:
        return dict(resource_release_confirmed=False, ownership_retained=False)
    reply = rows[0]; state = reply['provider_state']
    zero = not any(state[k] for k in ('hook_owned', 'enabled', 'inflight', 'leases',
                                    'registered', 'binding_reserved', 'replay_pending'))
    if d['mechanism'] == '8.restore_offset':
        failed_traces = [r.get('failed_phase', {}).get('independent', {}).get('moe_trace', {})
                         for r in state.get('serving', {}).get('failures', [])]
        no_unproven_mode_error = not any(t.get('errors') for t in failed_traces)
        a.check(no_unproven_mode_error, 'SOURCE66_MOE_ERROR_PATH_MODE_CLEANUP_NOT_PROVEN')
        zero = zero and no_unproven_mode_error
    a.check(zero and reply['recovery_required'] is False and state['native_registered'] is False,
            'ZERO_OWNED_RESOURCES_BEFORE_ORIGINAL_OWNER_RELEASE')
    detached = (host['history'][-1]['action'] == 'detach' and host['history'][-1]['response']['status'] == 'DETACHED'
        and host['ticket'] is None and host['attachment'] is None and d['worker_core']['ownership_retained'] is False
        and d.get('binding_removal', {}).get('status') == 'SERVING_DEPLOYMENT_REMOVED')
    a.check(detached, 'ORIGINAL_DETACH_AND_DEPLOYMENT_REMOVAL_AFTER_RELEASE')
    return dict(resource_release_confirmed=zero and detached, ownership_retained=False,
                confirmation_scope='ORIGINAL_SOURCE_BOUND_RESOURCE_RECEIPTS_NOT_INDEPENDENT_STACK_MEASUREMENT')


def rpc(a, d, host, source, natural):
    history = host['history']; worker = host['worker']; ticket = history[1]['response']['ticket']; provider_owner = d['c2']['owner']
    a.check([r['action'] for r in history] == ACTIONS, 'ALL_39_ORIGINAL_PUBLIC_RPC_ACTIONS')
    a.check(host['config'] == d['worker_core']['config'] and ticket['worker'] == worker
        and ticket['owner'] == host['owner'] and ticket['tenant_id'] == host['tenant_id']
        and ticket['mechanism'] == d['mechanism'] and ticket['generation'] == d['c2']['generation'], 'HOST_RUNTIME_COVERAGE_IDENTITY_JOIN')
    attach = history[0]['response']; a.check(attach['status'] == 'ATTACHED'
        and attach['ticket'] == {k:ticket[k] for k in ('worker', 'tenant_id', 'attach_nonce')}, 'ORIGINAL_ATTACHMENT')
    for index, entry in enumerate(history[1:-1], 1):
        reply = entry['response']; state = reply['provider_state']; serving = state['serving']; cov = state['coverage']
        a.check(reply['ticket'] == ticket and reply['status'] == ('QUERY_OK' if entry['action'] == 'query' else 'ACTION_COMPLETED')
            and reply['recovery_required'] is False and reply['native'] is None, 'ORIGINAL_RPC_EXACT_TICKET_AND_RETURN', index)
        a.check(state['owner'] == provider_owner and state['generation'] == ticket['generation']
            and state['execution_boundary'] == 'python_hook' and state['state_source'] == 'OWNED_PYTHON_HOOKS'
            and state['hook_state_available'] is True and state['native_loaded'] is state['native_registered'] is False
            and state['inflight'] == state['leases'] == 0 and state['replay_failed'] is False
            and state['constructed_replay_fallback'] is False and not serving['error'] and serving['failures'] == [], 'ACTUAL_PROVIDER_IDENTITY_AND_RESOURCE_STATE', index)
        a.check(state['registered'] is (index not in (1, 37)) and state['binding_reserved'] is (index not in (1, 37)), 'BINDING_RESERVED_THROUGH_UNREGISTER', index)
        if index >= 4:
            deployment = cov['deployment']
            a.check(deployment['worker_rpc_owner'] == provider_owner and deployment['worker_rpc_generation'] == ticket['generation']
                and deployment['simulation'] is (not natural) and deployment['historical_scope_inherited'] is False
                and deployment['constructed_replay_fallback'] is False and serving['provenance'] == d['setup']['provenance'], 'NEW_PUBLIC_DEPLOYMENT_NO_HISTORICAL_AUTHORITY', index)
            if source:
                a.check(deployment['binding_sources'] == source['binding_sources'], 'FROZEN_PUBLIC_BINDING_SOURCES', index)
        for key, first in (('discovery', 16), ('C2', 22), ('C3', 31), ('same_worker_healthy', 33)):
            value = d['qualification'] if key == 'discovery' else d['health_request']['worker'] if key == 'same_worker_healthy' else d[key.lower()]
            a.check((key in cov) is (index >= first) and (key not in cov or cov[key] == value), 'ORIGINAL_RPC_COVERAGE_RECEIPT_CHAIN:' + key, index)
    for i, name in enumerate(PHASES):
        advance, query, collect = [r['response']['provider_state']['serving'] for r in history[5+3*i:8+3*i]]
        item = d['health_request'] if i == 9 else d['requests'][i]; rid = item['mapping']['internal_request_id']
        a.check(advance['next_phase'] == query['next_phase'] == name and advance['pending_request'] == query['pending_request'] == [rid, name]
            and collect['pending_request'] is None and collect['next_phase'] == (PHASES[i+1] if i < 9 else 'complete'), 'QUERY_IS_READ_ONLY_EXPLICIT_COLLECT_ADVANCES', name)
        if i < 9:
            a.check(query['last_request'] == item['worker'], 'ACTUAL_REQUEST_ORIGINAL_QUERY_ROW', name)
        if i in (4, 6):
            a.check(advance['group'] == d['c2_open' if i == 4 else 'c3_open'], 'ACTUAL_OPEN_GROUP_RECEIPT', name)
    state = history[-2]['response']['provider_state']
    a.check(not any(state[k] for k in ('hook_owned', 'enabled', 'inflight', 'leases', 'registered', 'binding_reserved', 'replay_pending'))
        and state['serving']['complete'] is True and state['serving']['recovered'] is True
        and history[-1]['response']['status'] == 'DETACHED' and host['ticket'] is None and host['attachment'] is None
        and host['ownership_retained'] is False and d['worker_core']['ownership_retained'] is False
        and d['binding_removal']['status'] == 'SERVING_DEPLOYMENT_REMOVED', 'FINAL_UNREGISTER_DETACH_BINDING_REMOVAL')


def review(d, host, auditor, source=None, *, natural=True):
    a = Review(); mechanism = d['mechanism']; moe = moe_checks()
    a.check(mechanism in ('8.restore_offset', '18.api_half'), 'REVIEWED_MECHANISM')
    complete = len(d.get('requests', [])) == 9 and 'health_request' in d and all(k in d for k in ('c2', 'c3', 'qualification'))
    if not complete:
        a.check(False, 'REFUSED_OR_INCOMPLETE_ACTUAL_PROTOCOL', d.get('error', 'missing actual phase rows'))
        cleanup = a.section('partial_cleanup', final_cleanup, a, d, host)
        last = host['history'][-1]['response']
        return dict(**a.summary(), source_bound_public_c2_c3_receipts_reconciled=False,
            original_error=d.get('error'), completed_requests=len(d.get('requests', [])),
            ownership_retained=host['ownership_retained'], cleanup=cleanup, last_rpc_status=last['status'])
    rows = [v['worker'] for v in d['requests']]; baseline = rows[0]
    a.check([v['phase'] for v in d['requests']] == list(PHASES[:9])
        and len({v['worker']['request_id'] for v in d['requests']+[d['health_request']]}) == 10
        and len({r['invocation'] for r in rows}) == 9, 'TEN_DISTINCT_ACTUAL_REQUESTS')
    found = d['qualification']['candidate_set']; candidates = {}
    for c in found['candidates']:
        if c['eligibility']['status'] != 'SUPPORTED':
            a.check(c['identity'] is None and bool(c['eligibility']['reasons']), 'UNSUPPORTED_CANDIDATE_RETAINED'); continue
        t = c['identity']; sig = json.loads(t['tensor_signature']); key = (sig['path'], sig['ordinal'])
        a.check(key not in candidates and c['candidate_id'] == key[0]+':'+str(key[1]) and not c['eligibility']['reasons'], 'UNIQUE_ACTUAL_ELIGIBLE_TARGET')
        candidates[key] = t
        w = host['worker']
        a.check(t['scope_kind'] == 'INITIALIZED_WORKER' and t['process_id'] == w['pid'] and t['rank'] == w['rank']
            and t['worker_id'] == w['worker_start_identity'] and t['process_start_identity'] == w['process_start_identity']
            and t['model_instance_id'] == w['model_instance_id'] and positive(t['model_object_identity'])
            and positive(t['module_object_identity']) and t['boundary']['kind'] == 'python_hook'
            and t['boundary']['call_ordinal'] == key[1] and t['provider_identity'] == 'natural_tensor:' + mechanism,
            'ACTUAL_WORKER_MODEL_TARGET_IDENTITY', key)
        if natural:
            a.check(t['device_type'] == 'npu' and t['logical_device'] == w['logical_device'] == 0 and w['rank'] == 0, 'ACTUAL_TP1_NPU_TARGET', key)
        dispatch = d['qualification']['actual_dispatch'][key[0]]
        a.check(dispatch['module_object_id'] == t['module_object_identity']
            and dispatch['callable_identity'] == t['boundary']['callable_identity']
            and dispatch['source_sha256'] == t['boundary']['callable_source_sha256'] == sha(dispatch['methods'])
            and all(valid_sha(m['source_sha256']) and valid_sha(m['code_sha256']) for m in dispatch['methods']),
            'ACTUAL_MODULE_DISPATCH_AND_METHOD_SOURCE_JOIN', key)
    a.check(found['eligible'] == len(candidates) > 0 and found['discovered'] == len(found['candidates']), 'COMPLETE_ELIGIBLE_AND_UNRESOLVED_DENOMINATOR')
    a.check(len({t['model_object_identity'] for t in candidates.values()}) == 1
        and set(d['qualification']['actual_dispatch']) == {key[0] for key in candidates}, 'ONE_MODEL_AND_COMPLETE_TARGET_CALLABLE_SET')
    if mechanism == '18.api_half':
        supported = {key[0] for key in candidates}
        expected_unsupported = {m['path'] + ':unreviewed_rms_source' for m in rows[1]['independent']['module_inventory']
            if m['role'] == 'rmsnorm_source_candidate' and m['path'] not in supported}
        a.check({c['candidate_id'] for c in found['candidates'] if c['eligibility']['status'] != 'SUPPORTED'} == expected_unsupported,
            'ALL_UNSUPPORTED_STRUCTURAL_RMS_CANDIDATES_RETAINED')
    for index, item in enumerate(d['requests']):
        a.section(item['phase'], phase, a, item, index, baseline, candidates, mechanism, natural, moe)
    a.check(rows[4]['output'] == rows[6]['output'] == rows[7]['output'], 'CONSECUTIVE_TWO_FULL_FAULT_OUTPUTS_REPEAT')
    health = d['health_request']; h = health['worker']
    a.section('health_scheduler', scheduler, a, health, baseline, natural)
    a.check(health['phase'] == 'cleanup_healthy' and h['worker'] == host['worker'] and h['input'] == baseline['input']
        and h['output'] == baseline['output'] and h['cleanup_before_request'] is True and h['fault_hooks'] == 0
        and h['full_hidden_logits_exact'] is True and health['output_tokens'] == d['healthy_before'] == d['healthy_after'], 'SAME_WORKER_CLEANUP_HEALTH_FULL_OUTPUT')
    for c in (d['cleanup'], d['worker_core']['final_cleanup']):
        a.check(c['owned_serving_hooks'] == 0 and c['active_adapter'] is False and c['independent_observer'] is False,
                'OWNED_SERVING_RESOURCES_ZERO')
    groups = a.section('coverage', coverage, a, d, candidates, auditor)
    a.section('live_proof', live_proof, a, d, host['worker'])
    a.section('public_rpc', rpc, a, d, host, source, natural)
    cleanup = a.section('final_cleanup', final_cleanup, a, d, host)
    provenance = d['setup']['provenance']; scope = d['c2']['scope']
    a.check(scope == d['c3']['scope'] and scope['mechanism'] == mechanism
        and scope['evidence_kind'] == ('natural_worker' if natural else 'cpu_contract')
        and provenance['worker_identity']['worker'] == host['worker'] and provenance['simulation'] is (not natural)
        and scope['model_weights_sha256'] == sha(provenance['weights'])
        and scope['model_source_sha256'] == sha(provenance['model_sources']), 'NEW_SCOPE_MODEL_SOURCE_WEIGHT_IDENTITIES')
    for row in rows:
        a.check(row['independent']['module_inventory'] == rows[0]['independent']['module_inventory'], 'SAME_MODEL_MODULE_CENSUS_SOURCE', row['phase'])
        if mechanism == '8.restore_offset':
            trace = row['independent']['moe_trace']
            a.check(trace['sources'] == provenance['moe_sources']
                and trace['structural_moe_paths'] == provenance['structural_moe_paths'], 'SAME_MOE_ACTUAL_SOURCE_AND_STRUCTURAL_SET', row['phase'])
        if source:
            a.check(row['independent']['implementation_sha256'] == source['files']['fault_injection/tensor_observer.py'],
                    'FROZEN_INDEPENDENT_OBSERVER_SOURCE', row['phase'])
    if natural:
        model = d['model_identity']; config = {k:v for k,v in model.items() if k != 'weights'}
        a.check(model['weights'] == provenance['weights'] and config['routing_modified'] is False
            and scope['environment_sha256'] == H.digest(d['setup']['environment'])
            and scope['workload_sha256'] == H.digest(dict(weights=model['weights'], tokens=baseline['scheduler']['token_ids'], model_config=config)),
            'ACTUAL_NEW_WORKLOAD_ENVIRONMENT_HASHES')
    if source:
        a.check(scope['implementation_sha256'] == source['implementation_sha256']
            and d['probe_sha256'] == source['probe_sha256'], 'FROZEN_NEW_ENTRY_IMPLEMENTATION_SCOPE')
        a.check(all(g['discovery']['implementation_sha256'] == source['common_implementation_sha256'] for g in (d['c2'], d['c3']))
            and all(dict(t['provider_proof']) == {'implementation_sha256':source['common_implementation_sha256']}
                for t in candidates.values()), 'FROZEN_GROUP_OBSERVER_AND_TARGET_PROVIDER_SOURCE')
    return dict(**a.summary(), source_bound_public_c2_c3_receipts_reconciled=not a.errors,
        mechanism=mechanism, actual_requests=10, eligible_targets=len(candidates),
        unsupported_candidates=len(found['candidates'])-len(candidates), groups=groups[0] if groups else None,
        canonical_reference_objects_recomputed=groups[1] if groups else None, cleanup=cleanup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--result-sha256', required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args(); artifacts = Review(); sources = Review()
    report = dict(schema='demoncat.review.public_tensor.source66.v1', E5='NOT_ASSESSED', C1='NOT_SEPARATELY_ASSESSED',
        historical_qualification_inherited=False, live_authority_reconstructed=False,
        limitations=['Complete input/hidden/logits and per-target effects remain source-bound live readback receipts; private tensor payloads were not saved for independent byte recomputation.',
            'Canonical JSON references are independently recomputed from original rows, not invented sidecar execution evidence.',
            'MoE producer/object/native-consumer joins are independently reconciled. RMS return-to-consumer liveness is source-bound where the saved consumer lacks full tensor IDs.',
            'Only this initialized TP1 eager fresh-prefill Worker scope is reviewed. No E5, statistical localization, generic model or Native reduction-error qualification is inherited.',
            'Source66 has a separately reproduced MoE nested-mode cleanup defect on exit exceptions (fixed in a5254327 for a new source scope). Normal complete original traces are reviewed here; this does not certify that failure branch or independently measure the old process dispatch stack.',
            'External model weights and installed model-method files are hash-bound by the original scope; this CLI does not independently reread those external files.'])
    original = artifacts.section('artifacts', load_artifacts, artifacts, args.result, args.result_sha256)
    source = sources.section('source', source_review, sources, args.source_root)
    report.update(artifacts=artifacts.summary(), provenance=sources.summary())
    if original and source:
        d, host = original; details, auditor = source; a = Review()
        evidence = a.section('evidence', review, d, host, auditor, details)
        report['evidence'] = evidence if evidence is not None else a.summary()
    good = all(report.get(k, {}).get('consistent') is True for k in ('artifacts', 'provenance', 'evidence'))
    report.update(source_bound_public_c2_c3_receipts_reconciled=good,
        status='SOURCE_BOUND_PUBLIC_TENSOR_C2_C3_RECEIPTS_RECONCILED_NOT_E5' if good else 'REFUSED_INCOMPLETE_OR_CONTRADICTORY_EVIDENCE')
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if good else 2


if __name__ == '__main__':
    raise SystemExit(main())
