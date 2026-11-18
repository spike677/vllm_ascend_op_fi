"""Offline public Worker output39 C1/refusal review; no runtime imports.

Reuse source51's projection/consumer checks and Core41's strict JSON, artifact
and audit utilities. No success flag, old MODEL_LOCAL result or hit total grants
qualification. A correctly recorded refusal is distinct from C1 and cleanup.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys


NATIVE_SHA = '8876f2b99e70a9df8201b6ae0ace2cc4be8702afcf6703f9b72d1dda3bc26622'
HELPERS = {
    'projection': ('review39_output_source51.py', 'fcd233febaab54f5cc209f7af3b05beb33e7a36a327d5121bb3bbc06df384372'),
    'core': ('review41_worker_core_result.py', '4801d84dadc0ac8fc195a409cc80f2584554fa1c62ebbbac4af250c56dc50c58')}
PHASES = ['observe', 'observe_repeat', 'baseline', 'fault', 'off', 'health']
ACTIONS = ['attach', 'prepare', 'register', 'check', 'enable'] + ['advance', 'query']*6 + ['finish', 'disable', 'unregister', 'detach']
CONFIG = dict(fault_id=39, variant='output_proxy', implementation_version='1')
LEGACY_PROFILE = 'source65-aten-v1'
CUSTOM_PROFILE = 'reviewed-custom-gemm-v1'
CUSTOM_HELPER = ('review39_custom_gemm.py', 'd82143ac7c3e99f10f700064fd6fd40e7a573251e97e7f4f7f186356e24fa156')
CORE = 'src/demoncat/runtime/vllm_ascend/worker_extension/'
MODULES = ['fault_injection/deployments/'+name+'.py' for name in
    ('postwrite', 'postwrite_boundaries', 'postwrite_inventory', 'postwrite_observed')] + [CORE+name+'.py' for name in
    ('serving39_output', 'providers/fault39_output', 'serving', 'serving41', 'serving_context',
     'call_binding', 'host', 'config', 'runtime', 'identity', 'registry')]
SOURCES = MODULES + ['experiments/c2c3_postwrite_20261117/'+name+'.py' for name in
    ('natural39_worker_core_probe', 'prepare39_worker_core', 'natural41_probe', 'natural41_coverage_probe', 'model_identity')]


def helpers(profile=LEGACY_PROFILE):
    if profile not in (LEGACY_PROFILE, CUSTOM_PROFILE): raise ValueError('UNKNOWN_PROJECTION_PROFILE')
    result = {}
    for key, (name, expected) in HELPERS.items():
        path = Path(__file__).with_name(name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('REVIEWED_HELPER_CHANGED:' + name)
        spec = importlib.util.spec_from_file_location('_independent_output39_' + key, path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        result[key] = module
    projection = result['projection']
    if profile == CUSTOM_PROFILE:
        name, expected = CUSTOM_HELPER; path = Path(__file__).with_name(name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('REVIEWED_HELPER_CHANGED:' + name)
        spec = importlib.util.spec_from_file_location('_independent_output39_custom', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        projection = module.ProjectionProfile(projection)
        projection.custom = module
    return result['core'], projection


def action(record):
    method, args = record['method'], record['args']
    if method == 'demoncat_fault_rpc': return args[0], args[1]
    names = {'demoncat_attach':'attach', 'demoncat_fault_query':'query', 'demoncat_detach':'detach'}
    return (names[method], args[0]) if method in names else (None, None)


def load_artifacts(c, folder, document):
    a = c.Audit(); reference = document['host_receipt']; path = c.under(folder, reference['path'])
    a.check(reference['path'] == 'host_receipt.json' and c.file_sha(path) == reference['sha256'], 'ORIGINAL_HOST_FILE_SHA')
    host = c.read(path); records = []
    for index, item in enumerate(document['rpc_receipts']):
        relative = 'worker_core_rpc/' + item['path']
        path = c.under(folder, relative)
        a.check(item['path'] == str(index).zfill(3)+'.json' and c.file_sha(path) == item['sha256'], 'ORIGINAL_RPC_FILE_SHA', index)
        record = c.read(path)
        a.check(record['method'] == item['method'] and ('error' in record) != ('responses' in record), 'ORIGINAL_RPC_RESULT_OR_EXCEPTION', index)
        record['_reference'] = item
        records.append(record)
    return host, records, a.report()


def scope(c, a, document, host, records, natural):
    worker = host['worker']; history = host['history']
    a.check(host['config'] == CONFIG and host['mechanism'] == '39.output_proxy'
        and host['capability_enumeration']['worker'] == worker, 'PUBLIC_CONFIGURATION_AND_WORKER_ROUTE')
    a.check(worker['rank'] == 0 and c.positive(worker['pid']) and worker['model_instance_id']
        and worker['process_start_identity'] and worker['worker_start_identity']
        and worker['logical_device'] == worker['configured_logical_device'] == worker['observed_logical_device'], 'ONE_HELD_WORKER_LIFETIME')
    if natural:
        a.check(bool(worker['vllm_version']) and bool(worker['vllm_ascend_version']), 'REAL_VLLM_WORKER_VERSIONS')
    calls = [r for r in records if action(r)[0]]; cursor = 0; states = []
    ticket = None; attachment = None; internal = None
    for index, row in enumerate(history):
        name, reply = row['action'], row['response']; loc = str(index) + ':' + name
        if reply['status'] == 'HOST_REJECTED' or name == 'prepare_binding':
            a.check(reply['status'] in ('HOST_REJECTED', 'RPC_UNCERTAIN'), 'HOST_ONLY_REFUSAL_HAS_NO_NATIVE_SUCCESS', loc)
            continue
        raw = calls[cursor]; cursor += 1
        call_action, payload = action(raw)
        a.check(name == call_action, 'RAW_HOST_RPC_ACTION_ORDER', loc)
        if 'responses' in raw:
            a.check(len(raw['responses']) == 1 and (raw['responses'][0] == reply
                or reply.get('observed_response') == raw['responses'][0] and reply['status'] == 'RPC_UNCERTAIN'),
                'RAW_WORKER_REPLY_EQUALS_HOST_HISTORY', loc)
        else:
            a.check(reply['status'] == 'RPC_UNCERTAIN' and bool(raw['error']) and reply.get('error_type'),
                'UNCERTAIN_TRANSPORT_REPLY_RETAINED', loc)
        if name == 'attach' and reply['status'] == 'ATTACHED':
            attachment = reply['ticket']
            a.check(attachment['worker'] == worker and attachment['tenant_id'] == host['tenant_id']
                and bool(attachment['attach_nonce']), 'ACTUAL_ATTACHMENT_IDENTITY', loc)
        else:
            a.check(attachment is not None and all(payload.get(k) == v for k,v in attachment.items()), 'RPC_ATTACHMENT_NEVER_CHANGED', loc)
        supplied = reply.get('ticket')
        if supplied and 'generation' in supplied:
            if ticket is None: ticket = supplied
            a.check(supplied == ticket and ticket['owner'] == host['owner'] and ticket['worker'] == worker
                and ticket['mechanism'] == '39.output_proxy' and ticket['implementation_version'] == '1'
                and c.positive(ticket['generation']), 'PUBLIC_TICKET_OWNER_GENERATION', loc)
        if name not in ('attach', 'prepare', 'detach') and ticket:
            a.check(all(payload.get(k) == v for k,v in ticket.items()), 'ACTUAL_RPC_TICKET_INPUT', loc)
        a.check(payload.get('settings', {}) == ({'request_id':payload['settings']['request_id']} if name == 'advance' else {}),
            'NO_USER_TARGET_OR_NATIVE_ARGUMENTS', loc)
        state = reply.get('provider_state')
        if state is None: continue
        if internal is None: internal = state['owner']
        a.check(ticket is not None and state['owner'] == internal and state['generation'] == ticket['generation']
            and state['evidence']['rpc_owner'] == internal and state['evidence']['rpc_generation'] == ticket['generation']
            and state['evidence']['worker_identity'] == worker and reply.get('native') == state,
            'HOST_RUNTIME_PROVIDER_OWNER_JOIN', loc)
        a.check(state['execution_boundary'] == 'native' and state['native_loaded'] is True
            and state['native_registered'] is False and state['native_registration'] == 'NOT_APPLICABLE_DIRECT_ABI_LEASES'
            and state['native_global_owner_table_observed'] is False and state['native']['abi'] == 1,
            'ACTUAL_DIRECT_NATIVE_ABI_NOT_GLOBAL_REGISTRATION', loc)
        if natural:
            a.check(state['native']['sha256'] == NATIVE_SHA and state['native']['simulation'] is False
                and state['native']['loaded_mapping_verified'] is True and state['native']['read_span_available'] is True,
                'ACTUAL_NATIVE_MAPPING_AND_READBACK', loc)
        a.check(state['leases'] == len(state['native_leases']) and state['enabled'] is any(v['native_live'] for v in state['native_leases']),
            'FRESH_OWNED_LEASE_READBACK', loc)
        states.append(dict(action=name, response=reply, state=state, raw=raw, index=index))
    a.check(cursor == len(calls), 'EVERY_RECORDED_CORE_RPC_IN_HOST_HISTORY')
    for raw in records:
        if raw['method'] == 'demoncat_worker_identity' and 'responses' in raw:
            a.check(len(raw['responses']) == 1 and raw['responses'][0]['worker'] == worker, 'ACTUAL_WORKER_IDENTITY_RPC')
        elif raw['method'].endswith('.setup') and 'responses' in raw:
            a.check(raw['responses'] == [document['binding']], 'ORIGINAL_TRUSTED_SETUP_REPLY')
        elif raw['method'].endswith('.remove_binding') and 'responses' in raw:
            a.check(raw['responses'] == document['binding_removal'], 'ORIGINAL_BINDING_REMOVAL_REPLY')
    return dict(ticket=ticket, states=states, owner=internal, history_actions=[v['action'] for v in history])


def scheduler(c, a, row, rid, serving_row, natural):
    snap = row['snapshot']
    a.check(snap['internal_request_id'] == rid and snap['device_input_bytes_equality'] == 'COMPLETE_DEVICE_READBACK_VERIFIED'
        and serving_row['snapshot'] == snap and serving_row['actual_internal_request_id'] == rid
        and serving_row['root_returned'] is True and serving_row['model_output_sha256'] == row['output_sha256']
        and serving_row['model_output_finite'] == row['output_finite'], 'ACTUAL_FINAL_ROOT_LOGITS_REQUEST_BINDING', rid)
    if not natural: return None
    tokens = snap['token_ids']; positions = list(range(len(tokens))); n = len(tokens)
    a.check(snap['status'] == 'READ_ONLY_SCHEDULER_SNAPSHOT_CORROBORATED_NOT_NPU_ACCEPTANCE'
        and 2 <= n <= 4096 and all(type(t) is int and t >= 0 for t in tokens) and snap['positions'] == positions
        and snap['num_requests'] == 1 and snap['num_computed_tokens'] == 0
        and snap['num_prompt_tokens'] == snap['num_scheduled_tokens'] == n and c.positive(snap['request_object_id'])
        and snap['token_ids_sha256'] == hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest(),
        'FULL_FRESH_PREFILL_TOKENS_AND_SCHEDULER', rid)
    hashes = []
    for name, dtype, values, code in (('input_ids', 'torch.int32', tokens, 'i'), ('positions', 'torch.int64', positions, 'q')):
        meta = snap['input_tensor_metadata'][name]
        a.check(meta['shape'] == [n] and meta['stride'] == [1] and meta['storage_offset'] == 0
            and meta['dtype'] == dtype and meta['device'] == 'npu:0'
            and c.positive(meta['object_id']) and c.positive(meta['storage_identity']), 'FULL_CURRENT_DEVICE_INPUT_FORMALS', rid+':'+name)
        h = hashlib.sha256(json.dumps([dtype, [n]]).encode()); h.update(struct.pack('<'+code*n, *values)); hashes.append(h.hexdigest())
    a.check(serving_row['input_sha256'] == hashlib.sha256(json.dumps(hashes).encode()).hexdigest(), 'COMPLETE_INPUT_HASH_RECOMPUTED', rid)
    a.check(c.positive(snap['block_size']) and bool(snap['attention_metadata_keys'])
        and len(set(snap['attention_metadata_keys'])) == len(snap['attention_metadata_keys'])
        and snap['slot_mapping'] == [snap['block_ids'][i//snap['block_size']]*snap['block_size']+i%snap['block_size'] for i in positions]
        and len(set(snap['slot_mapping'])) == n and set(snap['attention_metadata_keys']) == set(snap['attention_metadata_evidence']),
        'ACTUAL_PREFILL_SLOT_AND_ATTENTION_MAPPING', rid)
    for name, value in snap['attention_metadata_evidence'].items():
        a.check(value == dict(adapter='DENSE_EXPLICIT_LENGTHS_AND_CAUSAL', seq_lens_list=[n], actual_seq_lengths_q=[n], causal=True),
            'CURRENT_DENSE_ATTENTION_LENGTHS', rid+':'+name)
    return tokens


def target(c, p, a, report, observed, owner, generation, worker, model_id, device, where):
    t = report['actual_target']; proof = t['producer_proof']; geom = proof['returned_tensor']
    a.check(t['pid'] == worker['pid'] and t['model_instance'] == model_id and t['owner'] == owner
        and t['generation'] == generation and t['operator'] == observed['operator'] and t['module_path'] == observed['module_path'],
        'ACTUAL_SELECTED_MODULE_OWNER_GENERATION', where)
    a.check(proof['operator'] == t['operator'] and proof['schema'] == observed['schema']
        and proof['schema_sha256'] == hashlib.sha256(proof['schema'].encode()).hexdigest()
        and proof['original_return_storage'] is True and proof['expected_output_shape'] == p.producer_geometry(proof) == t['shape']
        and geom == observed['output_geometry'] and p.geometry_valid(geom, exclusive=True, device=device)
        and all(t[k] == geom[k] for k in ('shape', 'stride', 'storage_offset', 'dtype')) and t['logical_indices'] == [0]
        and t['storage_identity'] == [geom[k] for k in ('device','storage_identity','data_ptr','storage_bytes')]
        and all(p.geometry_valid(value, device=device) and value['storage_identity'] != geom['storage_identity'] for value in proof['operands'].values()),
        'LITERAL_ORIGINAL_PRODUCER_STORAGE_AND_OPERANDS', where)
    p.consumers(a, t['consumer_observations'], geom, [0], t['consumer_calls'], where, device)
    a.check(t['consumer_calls'] == report['consumers'] == observed['consumers'], 'SELECTED_ACTUAL_VALUE_CONSUMERS', where)


def actual_requests(c, p, a, document, host, context, serving, natural):
    states = context['states']; latest = states[-1]['state']['evidence'] if states else None
    if latest is None: return dict(completed=0, partial=[], selection=None, latest=None)
    rows = latest['actual_requests']; aliases = latest['module_aliases']; device = 'npu' if natural else 'cpu'
    a.check(latest['mechanism'] == '39.output_proxy' and latest['scope'] == 'INITIALIZED_WORKER_SINGLE_SITE_C1_FIXED_FRESH_PREFILL'
        and latest['evidence_kind'] == ('natural_worker' if natural else 'cpu_contract')
        and latest['boundary'] == 'actual_public_projection_return_then_value_consumer'
        and latest['model_local_qualification_inherited'] is False and latest['native_out_of_bounds_claim'] is False,
        'NEW_STANDARD_WORKER_C1_SCOPE_NOT_INHERITED')
    a.check(len({v['path'] for v in aliases}) == len(aliases) and all(c.positive(v['object_identity'])
        and c.positive(v['forward_identity']) for v in aliases), 'ACTUAL_HELD_MODULE_ALIAS_IDENTITIES')
    alias_pairs = {(v['path'],v['object_identity']) for v in aliases}
    native_owner = int.from_bytes(hashlib.sha256(context['owner'].encode()).digest()[:8], 'little') % (2**63-1)+1
    all_driver = document.get('requests', []) + document.get('recovery_requests', [])
    mapping_by_id = {r['mapping']['internal_request_id']:r for r in all_driver}
    a.check(len(mapping_by_id) == len(all_driver), 'NO_REPEATED_DRIVER_REQUEST_ID')
    if rows: a.check(serving is not None and len(serving['requests']) == len(rows), 'ORIGINAL_SERVING_RECEIPT_COMPLETE')
    if serving and natural:
        boundary = serving['boundary']
        a.check(c.positive(boundary['vocab_size']) and boundary['output_contract'] == [1,boundary['vocab_size']]
            and boundary['processor_path'] != boundary['lm_head_path']
            and all(any(path == boundary[name] for path,_ in alias_pairs) for name in ('processor_path','lm_head_path')),
            'SOURCE_BOUND_FULL_VOCAB_RETURN_CONTRACT')
        for name in ('compute_logits_source','get_logits_source','ascend_dispatch_source'):
            value = boundary[name]
            a.check(value is not None and c.valid_sha(value['source_sha256']) and c.positive(value['first_line'])
                and isinstance(value['file'],str) and value['file'].startswith('/'), 'ACTUAL_LOGITS_METHOD_SOURCE_RECEIPT', name)
    inventories, prompts = [], []
    for i, row in enumerate(rows):
        rid = row['actual_internal_request_id']; driver = mapping_by_id[rid]; mapping = driver['mapping']
        a.check(row['phase'] == driver['phase'] and mapping['source'] == 'ACTUAL_PUBLIC_ENQUEUE_AND_OUTPUT_PROCESSOR'
            and mapping['proof_scope'] == 'DRIVER_MAPPING_ONLY_REQUIRES_REAL_MODEL_PREHOOK_CORROBORATION'
            and c.positive(mapping['request_state_object_id']) and driver['output']['request_id'] == mapping['external_request_id'],
            'REAL_DRIVER_COMPLETED_REQUEST_JOIN', rid)
        queried = [s for s in states if s['raw']['_reference'] == driver['worker_observation_receipt']]
        a.check(len(queried) == 1 and queried[0]['action'] == 'query' and queried[0]['response']['status'] == 'QUERY_OK'
            and queried[0]['state']['evidence']['actual_requests'] == rows[:i+1], 'ORIGINAL_QUERY_COMPLETION_PREFIX', rid)
        advances = [s for s in states if s['action'] == 'advance' and action(s['raw'])[1]['settings']['request_id'] == rid]
        a.check(len(advances) == 1 and advances[0]['index'] < queried[0]['index']
            and advances[0]['state']['serving']['expected'] == rid
            and advances[0]['state']['evidence']['actual_requests'] == rows[:i], 'ADVANCE_DOES_NOT_EXECUTE_REQUEST', rid)
        a.check(row['native_owner'] == native_owner and row['native_leases'] == 0 and c.valid_sha(row['output_sha256'])
            and (row['output_finite'] is True or row['phase'] == 'fault'), 'CURRENT_NATIVE_OWNER_AND_COMPLETE_LOGITS_HASH', rid)
        if serving:
            prompts.append(scheduler(c, a, row, rid, serving['requests'][i], natural))
        if row['complete_inventory'] is None:
            a.check(row['phase'] == 'health' and row['report'] is None, 'ONLY_HEALTH_HAS_NO_INJECTION_OBSERVER', rid)
            continue
        item = row['complete_inventory']; p.inventory(a, item, rid, device); inventories.append(item)
        a.check(all((m['path'],m['object_identity']) in alias_pairs for m in item['modules']), 'ACTUAL_MODULES_JOIN_HELD_ALIASES', rid)
        report = row['report']; eligible = [v for v in item['calls'] if v['eligible']]
        a.check(report['phase'] == ('observe' if row['phase'] == 'observe_repeat' else row['phase'])
            and report['mechanism'] == '39.output_proxy' and report['transport'] == ('public_acl_native' if natural else 'cpu_test')
            and report['original_operator_calls'] == sum(item['operator_counts'].values())
            and report['discovery_diagnostics']['operator_counts'] == item['operator_counts']
            and report['eligible_with_consumers'] == len(eligible), 'FULL_OBSERVED_PROJECTION_AND_OPERATOR_CENSUS', rid)
        reasons = {v['reason'] for v in report['discovery_diagnostics']['transport_rejections']}
        a.check(all(reason in reasons for reason in report['rejections']), 'CUMULATIVE_UNSELECTED_REFUSALS_NOT_SELECTED_FAILURES', rid)
    if prompts and natural: a.check(all(value == prompts[0] for value in prompts), 'SAME_COMPLETE_FRESH_PROMPT')
    partial = []; failures = latest['failures']
    if serving:
        saved = serving['provider']
        a.check(all(saved[k] == latest[k] for k in ('worker_identity','rpc_owner','rpc_generation','module_aliases','scope','boundary'))
            and saved['failures'][:len(failures)] == failures, 'ORIGINAL_SERVING_FAILURE_SCOPE_AND_PREFIX')
        failures = saved['failures']
    for i, failure in enumerate(failures):
        if failure['partial_inventory'] is not None:
            p.inventory(a, failure['partial_inventory'], 'partial:'+str(i), device)
            item = failure['partial_inventory']
            partial.append(dict(phase=failure['phase'], projection_calls=len(item['calls']), eligible=sum(v['eligible'] for v in item['calls']),
                operator_counts=item['operator_counts'], error=failure['error']))
    return dict(completed=len(rows), partial=partial, inventories=inventories, selection=latest['selection'], latest=latest,
        native_owner=native_owner, rows=rows, serving=serving)


def c1(c, p, a, document, host, context, details, natural):
    rows = details['rows']; evidence = details['latest']; device = 'npu' if natural else 'cpu'
    a.check(context['history_actions'] == ACTIONS and [r['phase'] for r in rows] == PHASES
        and [r['phase'] for r in document['requests']] == PHASES and not document.get('recovery_requests')
        and not evidence['failures'] and not any(document.get(k) for k in ('error','recovery_error','abort_error')),
        'SIX_ORIGINAL_REQUESTS_WITHOUT_FAILURE_OR_RETRY')
    a.check(document['c1'] == evidence and details['serving']['provider'] == evidence, 'FINAL_UNCHANGED_C1_EVIDENCE')
    for row in host['history']:
        name, reply = row['action'], row['response']
        expected = {'attach':'ATTACHED', 'query':'QUERY_OK', 'detach':'DETACHED'}.get(name,'ACTION_COMPLETED')
        a.check(reply['status'] == expected and (name in ('attach','detach') or reply['recovery_required'] is False),
            'EVERY_C1_PROTOCOL_ACTION_ACTUALLY_SUCCEEDED', name)
        if name not in ('attach','detach','query'):
            a.check(reply['provider'] == reply['provider_state'], 'ACTION_AND_FRESH_PROVIDER_READBACK_MATCH', name)
    first, second = details['inventories'][:2]
    a.check([p.key(v) for v in first['calls']] == [p.key(v) for v in second['calls']]
        and first['modules'] == second['modules'] and first['operator_counts'] == second['operator_counts'], 'TWO_HEALTHY_COMPLETE_DENOMINATORS')
    for left,right in zip(first['calls'],second['calls']):
        a.check(p.geom(left) == p.geom(right) and left['eligible'] == right['eligible']
            and left['schema'] == right['schema'] and left['selected_indices'] == right['selected_indices'],
            'HEALTHY_PROJECTION_SIGNATURE_AND_ELIGIBILITY_UNCHANGED', left['module_path'])
    eligible = [v for v in first['calls'] if v['eligible']]; selected = eligible[0]; q = details['selection']
    a.check([p.key(v) for v in eligible] == [p.key(v) for v in second['calls'] if v['eligible']]
        and q['count'] == len(eligible) and q['selection'] == 'first eligible in stable actual execution order'
        and q['operator'] == selected['operator'] and q['module_path'] == selected['module_path']
        and q['producer_proof']['returned_tensor'] == selected['output_geometry'] and c.valid_sha(q['input_sha256']),
        'FIRST_ACTUAL_ELIGIBLE_PAIR_NOT_NAME_OR_HIT_SELECTION')
    p.consumers(a, q['consumer_observations'], selected['output_geometry'], [0], q['consumers'], 'selection', device)
    model_id = next(v['object_identity'] for v in evidence['module_aliases'] if v['path'] == '')
    g = context['ticket']['generation']
    a.check([r['native_generation'] for r in rows] == [g,g,g+1,g+2,g+3,g+3], 'SAME_OWNER_FRESH_NATIVE_GENERATIONS')
    baseline = rows[0]['output_sha256']
    for index, row in enumerate(rows):
        if index != 3: a.check(row['output_sha256'] == baseline and row['output_finite'] is True, 'LOSSLESS_OFF_HEALTH_COMPLETE_LOGITS_HASH', row['phase'])
        if index < 2:
            a.check(row['report']['intervention'] is None, 'HEALTHY_OBSERVATIONS_HAVE_NO_NATIVE_APPLICATION')
        if index not in (2,3,4): continue
        report, inventory = row['report'], row['complete_inventory']; observed = [v for v in inventory['calls'] if p.key(v) == p.key(selected)]
        a.check(len(observed) == 1 and observed[0]['eligible'], 'SAME_ACTUAL_SELECTED_OCCURRENCE', row['phase'])
        target(c, p, a, report, observed[0], details['native_owner'], row['native_generation'], host['worker'], model_id, device, row['phase'])
        a.check(report['recovery_confirmed'] is True and report['new_fault_hits'] == (1 if index == 3 else 0)
            and (index == 3 or report['exact_baseline_output'] is True), 'SOURCE_BOUND_FULL_RETURN_AND_NATIVE_RESTORE_GATE', row['phase'])
        n = report['intervention']
        if index == 4: a.check(n is None, 'OFF_HAS_NO_APPLICATION')
        else:
            a.check(n['original_executed'] is True and n['readback_exact'] is True and n['selected_elements'] == 1
                and n['changed_elements'] == n['new_fault_hits'] == (1 if index == 3 else 0)
                and n['operator'] == observed[0]['operator'] and n['module_path'] == observed[0]['module_path']
                and n['boundary'] == 'post_operator_tensor_output', 'ACTUAL_ORIGINAL_NATIVE_EFFECT_READBACK', row['phase'])
    return dict(observed_projection_calls=len(first['calls']), eligible_pairs=len(eligible), selected_key=selected['key'],
        selected_operator=selected['operator'], native_fault_invocations=1, logits_changed=rows[3]['output_sha256'] != baseline)


def cleanup(c, a, document, host, context, natural):
    released = [s for s in context['states'] if s['action'] == 'unregister' and s['response']['status'] == 'ACTION_COMPLETED']
    a.check(len(released) == 1, 'EXPLICIT_PUBLIC_UNREGISTER_WITH_NATIVE_READBACK')
    if released:
        state = released[0]['state']; serving = state['serving']
        a.check(state['leases'] == state['inflight'] == state['owned_hooks'] == 0 and state['native_leases'] == []
            and state['enabled'] is state['registered'] is state['replay_pending'] is state['replay_failed'] is False
            and state['binding_reserved'] is False, 'FINAL_OWNED_NATIVE_HOOK_AND_RESERVATION_ZERO')
        a.check(serving is None or serving == dict(installed=False, owned_hooks=0, expected=None, inflight=0, pending=False, observer=None),
            'FINAL_OBSERVER_MODE_LOCK_AND_REQUEST_CLOSED')
    a.check(host['history'][-1] == dict(action='detach', response=dict(status='DETACHED', recoveries=[]))
        and host['status'] == 'DETACHED' and host['attachment'] is host['ticket'] is host['census_ticket'] is None
        and host['ownership_retained'] is document['ownership_retained'] is False, 'ACTUAL_PUBLIC_OWNER_RELEASE_AND_DETACH')
    if natural: a.check(document.get('binding_removal') == [dict(status='SERVING_DEPLOYMENT_REMOVED')], 'TRUSTED_DEPLOYMENT_BINDING_REMOVED')
    health = document.get('post_unregister_health')
    if health:
        a.check(health['worker'] == host['worker'] and health['output']['request_id'] == health['mapping']['external_request_id']
            and all(health['output'][k] == document['healthy_before'][k] for k in ('tokens','text'))
            and health['mapping']['internal_request_id'] not in {r['mapping']['internal_request_id'] for r in document['requests']}
            and health['mapping']['source'] == 'ACTUAL_PUBLIC_ENQUEUE_AND_OUTPUT_PROCESSOR'
            and c.positive(health['mapping']['request_state_object_id']), 'ORDINARY_SAME_WORKER_POST_UNREGISTER_TOKENS_TEXT')
    return dict(post_unregister_health_observed=health is not None, process_global_native_owner_table_observed=False)


def review_evidence(document, host, records, serving=None, *, natural=True, profile=LEGACY_PROFILE):
    c,p = helpers(profile); a = c.Audit(); sections = {}; results = {}
    def run(name, function, *args):
        checks, failures, issues = a.checks, a.failures, len(a.issues)
        result = a.section(name, function, *args)
        sections[name] = dict(consistent=a.failures == failures, checks=a.checks-checks,
            failures=a.failures-failures, issues=a.issues[issues:])
        return result
    context = run('rpc', scope, c, a, document, host, records, natural)
    details = None
    if context:
        details = run('actual_observations', actual_requests, c,p,a,document,host,context,serving,natural)
        results['cleanup'] = run('cleanup', cleanup, c,a,document,host,context,natural)
    candidate = bool(details and details['completed'] == 6 and details['latest'] and not details['latest']['failures']
        and not document.get('error') and not document.get('recovery_requests'))
    if candidate:
        results['c1'] = run('c1', c1,c,p,a,document,host,context,details,natural)
        if profile == CUSTOM_PROFILE:
            results['custom_deployment'] = run('custom_deployment', p.custom.deployment, a,document,host,records)
            results['custom_c1'] = run('custom_c1', p.custom.completed_c1, a,p,details)
        a.check(results['cleanup'] and results['cleanup']['post_unregister_health_observed'], 'ORDINARY_POST_UNREGISTER_HEALTH_REQUIRED_FOR_C1')
    if details:
        results['observations'] = dict(completed_requests=details['completed'], partial_failures=details['partial'],
            per_completed_projection_counts=[len(i['calls']) for i in details.get('inventories', [])],
            per_completed_eligible_counts=[sum(v['eligible'] for v in i['calls']) for i in details.get('inventories', [])],
            recorded_first_failure=next(iter((details.get('latest') or {}).get('failures', [])), {}).get('error'))
    good = not a.failures; c1_good = good and candidate
    no_eligible = bool(details and details['partial'] and any('NO_ELIGIBLE_WRITER_WITH_OBSERVED_VALUE_CONSUMER' in r['error']
        and r['eligible'] == 0 for r in details['partial']) and not details['selection'])
    return dict(**a.report(), sections=sections, details=results, projection_profile=profile,
        source_bound_c1_receipts_reconciled=c1_good, cleanup_reconciled=sections.get('cleanup',{}).get('consistent',False),
        classification='SOURCE_BOUND_SINGLE_C1_RECEIPTS' if c1_good else 'NO_ELIGIBLE_OBSERVED_PAIR' if no_eligible else (
            'PRE_C1_REFUSAL' if details and details['completed'] == 0 else 'REFUSED_OR_INCOMPLETE'),
        original_driver_error=document.get('error'), original_recovery_error=document.get('recovery_error'),
        public_qualification_granted=False, c2=False, c3=False, e5=False,
        limits=['Complete logits hashes, literal producer/consumer metadata and Native readback/recovery are saved source-bound receipts, not newly observed device execution.',
            'Complete hidden+logits lossless/OFF equality is a frozen runtime gate; full hidden hashes and raw tensor bins are absent and are not independently rehashed.',
            ('The explicit custom profile checks actual registered GEMM and RMS/Swi receipts; schema examples do not establish runtime geometry.'
             if profile == CUSTOM_PROFILE else 'The denominator contains every observed supported projection and all operator counts; opaque custom GEMM entries remain unqualified.'),
            'The seventh ordinary request proves same Worker tokens/text only; no post-unregister full tensor is captured.',
            'No process-global Native owner table, MODEL_LOCAL inheritance, matmul_range, C2/C3, E5 or portable device qualification is granted.'])


def provenance(c, document, host, source_root, manifest_sha, commit, native_library, *, profile=LEGACY_PROFILE):
    _,projection = helpers(profile)
    modules = MODULES + (projection.custom.EXTRA_MODULES if profile == CUSTOM_PROFILE else [])
    sources = SOURCES + (projection.custom.EXTRA_MODULES if profile == CUSTOM_PROFILE else [])
    a = c.Audit(); path = c.under(source_root, 'source_manifest.json'); manifest = c.read(path)
    a.check(c.valid_sha(manifest_sha) and c.file_sha(path) == manifest_sha and manifest['base_commit'] == commit, 'FROZEN_SOURCE_IDENTITY')
    files = {r['path']:r for r in manifest['files']}; hashes = document['sources']
    a.check(len(files) == len(manifest['files']) and set(hashes) == set(sources), 'COMPLETE_SOURCE_AND_MANIFEST_SET')
    for name, expected in hashes.items():
        path = c.under(source_root,name)
        a.check(c.file_sha(path) == expected == files[name]['sha256'] and path.stat().st_size == files[name]['bytes'], 'SOURCE_FILE_BYTES', name)
    a.check(document['schema'] == 'demoncat.output39.worker_core_probe.v1' and document['mechanism'] == '39.output_proxy'
        and document['standard_worker'] is True and c.file_sha(native_library) == document['native_sha256'] == NATIVE_SHA
        and c.valid_sha(document['build_fingerprint']), 'NEW_WORKER_ENTRY_AND_EXACT_NATIVE_DSO')
    binding = document.get('binding')
    if profile == CUSTOM_PROFILE:
        a.check(binding is not None or not document.get('requests'), 'CUSTOM_REQUESTS_REQUIRE_ORIGINAL_DEPLOYMENT')
    if binding:
        a.check(binding['worker'] == host['worker'] and set(binding['loaded_sources']) == set(modules), 'LOADED_SOURCE_AND_WORKER_SET')
        for name, row in binding['loaded_sources'].items():
            allowed = {'/source/'+name, '/installed/'+(name[4:] if name.startswith('src/') else name)}
            a.check(row['sha256'] == hashes[name] and row['path'] in allowed, 'ACTUAL_FROZEN_LOADED_MODULE_HASH', name)
        value = binding['binding']
        a.check(value['workers_created'] == value['hooks_installed'] == value['native_leases_opened'] == 0
            and value['native']['sha256'] == NATIVE_SHA and value['native']['loaded_mapping_verified'] is True,
            'INERT_TRUSTED_NATIVE_DEPLOYMENT')
        a.check(c.digest(binding['model_source']['manifest']) == binding['model_source']['source_sha256'], 'SAVED_ACTUAL_MODEL_SOURCE_MANIFEST')
    weights = document['weights']
    a.check(c.digest(weights['manifest']) == weights['weights_sha256'] and weights['manifest']['format'] == 'safetensors'
        and weights['manifest']['files'] and all(c.valid_sha(r['sha256']) and c.positive(r['bytes']) for r in weights['manifest']['files'].values()),
        'SAVED_SELECTED_WEIGHT_MANIFEST')
    return dict(**a.report(), source_files_rehashed=len(hashes), native_dso_rehashed=True,
        build_fingerprint_receipt_rehashed=False, external_model_weights_and_sources_rehashed=False)


def deny_runtime(event, args):
    if event in ('ctypes.dlopen', 'subprocess.Popen', 'socket.connect', 'os.system') or (
            event == 'import' and args[0].split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend',
                'fault_injection','fault39','demoncat')):
        raise RuntimeError('OFFLINE_REVIEW_FORBIDS_RUNTIME:' + event)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('result','source-root','native-library'): parser.add_argument('--'+name,type=Path,required=True)
    for name in ('result-sha256','source-manifest-sha256','source-commit'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--serving-receipt',type=Path); parser.add_argument('--serving-sha256')
    parser.add_argument('--projection-profile', choices=(LEGACY_PROFILE,CUSTOM_PROFILE), default=LEGACY_PROFILE)
    args = parser.parse_args(argv); sys.addaudithook(deny_runtime)
    report = dict(schema='demoncat.output39.public_core.independent_review.v1', read_only=True,
        projection_profile=args.projection_profile, public_qualification_granted=False,
        c2=False,c3=False,e5=False, source_bound_c1_receipts_reconciled=False)
    try:
        c,_ = helpers(args.projection_profile)
        if not c.valid_sha(args.result_sha256) or c.file_sha(args.result) != args.result_sha256: raise ValueError('ORIGINAL_RESULT_SHA_MISMATCH')
        document = c.read(args.result); host, records, report['artifacts'] = load_artifacts(c,args.result.parent,document)
        serving = None
        if args.serving_receipt:
            if not c.valid_sha(args.serving_sha256) or c.file_sha(args.serving_receipt) != args.serving_sha256: raise ValueError('ORIGINAL_SERVING_SHA_MISMATCH')
            serving = c.read(args.serving_receipt)
            report['serving_original'] = dict(path=str(args.serving_receipt),sha256=args.serving_sha256,bytes=args.serving_receipt.stat().st_size)
        report['original'] = dict(path=str(args.result),sha256=args.result_sha256,bytes=args.result.stat().st_size)
        report['provenance'] = provenance(c,document,host,args.source_root,args.source_manifest_sha256,args.source_commit,args.native_library,
            profile=args.projection_profile)
        report['evidence'] = review_evidence(document,host,records,serving,profile=args.projection_profile)
        good = all(report[k]['consistent'] for k in ('artifacts','provenance','evidence'))
        report.update(review_consistent=good, source_bound_c1_receipts_reconciled=good and report['evidence']['source_bound_c1_receipts_reconciled'],
            status=report['evidence']['classification'] if good else 'EVIDENCE_GAP_OR_CLEANUP_UNCONFIRMED')
    except (OSError,ValueError,TypeError,KeyError,IndexError,RuntimeError,AttributeError) as error:
        report.update(status='EVIDENCE_GAP_OR_CLEANUP_UNCONFIRMED', review_consistent=False,error=type(error).__name__+':'+str(error))
    print(json.dumps(report,indent=2,allow_nan=False))
    return 0 if report.get('review_consistent') else 2


if __name__ == '__main__': raise SystemExit(main())
