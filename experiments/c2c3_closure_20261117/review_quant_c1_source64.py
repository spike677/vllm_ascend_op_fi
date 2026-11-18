"""Offline source64 quant C1 reconciliation. Never imports torch or a runtime/DSO.

Hashes authenticate saved evidence; they cannot recreate a live input or lease.
Every conclusion below is explicitly source-bound, single-site, and not C2/C3/E5.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import struct


SOURCE_COMMIT = '46cd8ad153ad07ce1057cfcbac73d7d9fb76e226'
SOURCE_MANIFEST = 'c8a63053271e169f8317e76c5196348ee4b33eb48c591d07bc50eec3c3e018e6'
BUILD_SHA = 'b4ac9a2b68ff58c3a5f6b54c1f052170ce5d1f6d1f8c53a6a571d175ed9328ac'
HELPER_SHA = '8151f93dca35ead3b32b98d66881e5bc23432884bf5af8c1c6694f1b59fb8b5d'
PINS = dict(result='261acddcb5cbbb34b879e5abbc27c1795ada5a0153fa77aa4bf505abe8a93d9f',
    host='8338501d3d5de42cef4144d23e33816899151acabe5f9f3fe5599160a4beb509',
    serving='ccc06760834e8aed90dfa6d09383c214cc4bd4c3e60181b6d89216a2cfb4aae0',
    discovery='8473897611bd357a0e5fca232d84b58bb7d282a9e014a834fb80add1ead9ce28',
    inventory='7a7100bf10c0755dc73911b69b8a9448c8f093f573acf69763d05a0a36a8103c')
FILES = dict(result='result.json', host='host-core-receipts.json', serving='quant-c1/runtime/serving.json',
    discovery='quant-c1/runtime/discovery.json', inventory='quant-c1/runtime/inventory/shared_scale_inventory.json')
PHASES = ('discovery_1', 'discovery_2', 'reference', 'lossless', 'qualification_off', 'fault', 'off')
MODES = dict(reference=1, lossless=2, qualification_off=4, fault=3)
SCOPE = 'ONE_AUTOMATICALLY_PROVEN_NATURAL_B8_SITE_ONCE_NOT_ALL_ELIGIBLE'
GEOMETRY = ('bytes rows columns stride0 stride1 offset dtype format dimensions reserved').split()
PROJECTION = ('path module_object_identity quant_method_object_identity method_source_sha256 weight_object_identity').split()
ZERO_CORE = ('installed targets pending_parent device_bytes host_bytes owned_descriptors owned_workspace_bytes '
             'borrowed_workspace_bytes quarantined_bytes error').split()
ZERO_RUNTIME = ('public_active native_stack_active callback_active completion_active quarantined overlap_entries '
                'rejected_calls unreturned_public_scope orphan_descriptors abi_exceptions last_error reserved').split()
ZERO_OWNED = ('owned_hooks owned_method_patches owned_api_patches owned_row_dispatch_modes '
              'owned_source_profile_callbacks owned_early_dispatch_modes owned_nested_dispatch_modes '
              'owned_live_metadata_records observer_cleanup_pending installed native_owner_installed group_owner_installed').split()
ABI = dict(view=80, call=280, config=72, lease=392, event=360, binding=48, journal=104,
           plan=264, child=408, receipt=1032, state=160, public_input=104, completion=1368, runtime=416, raw=704)
CORE = 'src/demoncat/runtime/vllm_ascend/worker_extension/'
SETUP = [CORE + name + '.py' for name in ('serving_quant_group_c1', 'shared_scale_group_inventory',
    'serving_shared_scale_group_inventory', 'quant_row_provenance', 'quant_source_rows', 'quant_mla_metadata_rows',
    'serving_quant', 'host', 'config', 'providers/quant')] + [
    'fault_injection/quant_group_native.py', 'fault_injection/quant_native_isolation.py',
    'fault_injection/deployments/quant_group_c1.py', 'fault_injection/deployments/quant_group_inventory.py']
SOURCES = SETUP + [CORE + name + '.py' for name in ('runtime', 'extension', 'identity', 'quant_inventory',
    'shared_scale_inventory', 'serving_shared_scale_inventory', 'batch_context', 'serving_context', 'serving',
    'serving20', 'serving20_qmm')] + ['experiments/c2c3_closure_20261117/providers_quant_c1_probe.py']
SOURCES += ['fault_injection/native_adapters/' + name for name in (
    'batch_scale_group_core.h', 'batch_scale_group_runtime.h', 'batch_scale_group_runtime.cpp')]


def helpers():
    path = Path(__file__).with_name('review_native26_single_source54.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != HELPER_SHA:
        raise ValueError('REVIEWED_STDLIB_HELPER_CHANGED')
    spec = importlib.util.spec_from_file_location('_quant_offline_artifacts', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = helpers()
digest, file_sha, read, under = H.digest, H.file_sha, H.read, H.resolve_under
Review, positive, valid_sha = H.Review, H.positive, H.valid_sha


def fields(value, names):
    return {name: value[name] for name in names}


def words(sha):
    return list(struct.unpack('<4Q', bytes.fromhex(sha)))


def signature(rows):
    return fields(rows, ('logical_ids', 'row_token_positions', 'prefixes', 'input_ids'))


def abi(a, value, kind, loc):
    a.check(value['version'] == 1 and value['size'] == ABI[kind] and value.get('reserved', 0) == 0,
            'NATIVE_ABI', loc)


def sequence(row):
    return row['parent']['token'] if row['kind'] == 1 else row['native']['sequence']


def zero_state(a, state, loc):
    abi(a, state, 'runtime', loc)
    a.check(all(state['core'][k] == 0 for k in ZERO_CORE) and all(state[k] == 0 for k in ZERO_RUNTIME)
        and all(state['census'][k] == 0 for k in ('preparing', 'pending', 'executing', 'overflow', 'errors')),
        'ZERO_NATIVE_RESOURCES', loc)


def view(a, native, public, role, loc, natural):
    shape, stride = public['shape'], public['stride']
    a.check(public['device'] == ('npu:0' if natural else 'cpu') and positive(public['object_id'])
        and positive(public['storage_identity']) and positive(public['storage_addr']), 'ACTUAL_PUBLIC_STORAGE', loc)
    expected = dict(address=public['storage_addr'] + public['storage_offset'] * public['element_size'],
        bytes=public['element_size'] * shape[0] * (shape[1] if len(shape) == 2 else 1), rows=shape[0],
        columns=shape[1] if len(shape) == 2 else 0, stride0=stride[0], stride1=stride[1] if len(stride) == 2 else 0,
        offset=public['storage_offset'], dtype={'input':27, 'quant':2, 'scale':0}[role],
        format=public['actual_format'], dimensions=len(shape), reserved=0)
    a.check(all(native[k] == v for k, v in expected.items()) and positive(native['descriptor'])
        and public['storage_bytes'] >= expected['offset'] * public['element_size'] + expected['bytes'],
        'PUBLIC_NATIVE_FULL_VIEW_JOIN', loc)


def tensor_join(a, graph, public, loc):
    a.check(all(graph[k] == public[k] for k in ('object_id', 'storage_identity', 'storage_bytes', 'shape', 'stride', 'dtype', 'device'))
        and graph['storage_address'] == public['storage_addr'] and graph['offset'] == public['storage_offset'],
        'EXACT_TENSOR_OBJECT_AND_STORAGE', loc)


def raw_window(a, before, after, rows, loc):
    a.check([sequence(row) for row in rows] == list(range(before['last_sequence']+1, after['last_sequence']+1)),
            'COMPLETE_RAW_SEQUENCE_WINDOW', loc)
    counts = Counter()
    for row in rows:
        abi(a, row, 'raw', loc)
        a.check(row['kind'] in (1, 2), 'RAW_KIND', loc)
        if row['kind'] == 1:
            abi(a, row['parent'], 'call', loc)
            counts['model_parents'] += 1
        else:
            kind = row['native']['kind']
            a.check(kind in (1, 2, 3, 4), 'NATIVE_KIND', loc)
            counts['native_prepares'] += int(kind == 1)
            counts['native_executes'] += int(kind == 3)
            if kind == 1:
                counts[{0:'unclassified_prepares', 1:'helper_prepares', 2:'model_prepares'}.get(
                    row['binding']['role'], 'invalid')] += 1
    for state in (before, after):
        abi(a, state, 'runtime', loc)
        c = state['census']
        a.check(c['total'] == state['last_sequence'] and c['capacity'] == 32768
            and all(c[k] == 0 for k in ('overflow', 'errors', 'preparing', 'pending', 'executing'))
            and all(state[k] == 0 for k in ZERO_RUNTIME if k not in ('public_active',)), 'RAW_CENSUS_HEALTH', loc)
    for key in ('model_parents', 'native_prepares', 'native_executes', 'unclassified_prepares', 'helper_prepares', 'model_prepares'):
        a.check(counts[key] == after['census'][key] - before['census'][key], 'RAW_COUNTER_RECONCILIATION:' + key, loc)
    return counts


def event(a, e, parent, plan, child, kind, loc):
    a.check(e['kind'] == kind and e['thread'] == parent['thread'] and e['device'] == parent['device']
        and e['reserved'] == e['status'] == 0 and e['original_calls'] == int(kind in (2, 4))
        and all(e[k] == 0 for k in ('smooth_tensor', 'group_index_tensor', 'offset_tensor'))
        and e['destination_dtype'] == 2, 'ACTUAL_NATIVE_ORIGINAL_CALL_AND_FORMALS', loc)
    a.check(all(e[name] == plan[name] for name in ('input', 'quant', 'scale')),
            'CHILD_COMPLETE_TYPED_ARGUMENTS', loc)
    a.check(e['registration'] == child['registration'] and e['executor'] == (0 if kind == 1 else child['executor']),
            'ACTUAL_GWS_EXECUTOR_JOIN', loc)
    if kind >= 2:
        a.check(e['workspace_required'] == child['workspace_required'], 'ACTUAL_WORKSPACE_REQUIREMENT', loc)
    if kind >= 3:
        a.check(all(e[k] == child[k] for k in ('stream', 'workspace', 'workspace_capacity')), 'ACTUAL_EXECUTE_STREAM_WORKSPACE', loc)


def passthrough(a, rows, loc):
    a.check(len(rows) == 5 and [row['kind'] for row in rows] == [1, 2, 2, 2, 2], 'ONE_UNADMITTED_PARENT_FOUR_EVENTS', loc)
    p, events = rows[0]['parent'], [r['native'] for r in rows[1:]]
    child = dict(events[2], registration=events[0]['sequence'], workspace_required=events[1]['workspace_required'])
    a.check([e['sequence'] for e in events] == list(range(p['token']+1, p['token']+5))
        and positive(child['executor']) and positive(child['stream'])
        and child['workspace_capacity'] >= child['workspace_required']
        and bool(child['workspace']) == bool(child['workspace_capacity']), 'PASSTHROUGH_EXECUTION_LIFETIME', loc)
    for kind, (row, e) in enumerate(zip(rows[1:], events), 1):
        a.check(not any(row['binding'].values()), 'NON_TARGET_HAS_NO_NATIVE_LEASE', loc)
        event(a, e, p, p, child, kind, loc)


def receipt(a, packet, owner, generation, target, lifetime, loc):
    a.check(digest({k:v for k, v in packet.items() if k != 'artifact_sha256'}) == packet['artifact_sha256'],
            'RECEIPT_ARTIFACT_DIGEST', loc)
    lease, done, rows = packet['lease'], packet['completion'], packet['raw_events']
    p, core, mode = done['call'], done['core'], MODES[packet['phase']]
    for value, kind in ((lease, 'lease'), (done, 'completion'), (p, 'call'), (core, 'receipt')):
        abi(a, value, kind, loc)
    native_owner = words(digest(dict(owner=owner, worker=lifetime)))[:2]
    a.check(lease['expected'] == p and done['admitted'] == 1 and done['status'] == 0
        and positive(p['lifetime_token']) and p['thread'] == lifetime['native_thread']
        and p['device'] == lifetime['device'], 'CURRENT_ACTUAL_PARENT_LIFETIME', loc)
    a.check([lease['owner_hi'], lease['owner_lo']] == native_owner and lease['generation'] == generation
        and lease['thread'] == p['thread'] and lease['target'] == words(target)
        and lease['mode'] == core['mode'] == mode, 'SAME_NATIVE_OWNER_GENERATION_TARGET', loc)
    a.check(all(core[k] == lease[k] for k in ('owner_hi', 'owner_lo', 'generation', 'target', 'actual_row_proof'))
        and core['parent_token'] == p['token'] and core['lifetime_token'] == p['lifetime_token'], 'LEASE_COMPLETION_JOIN', loc)
    a.check(all(core[k] == 0 for k in ('error', 'last_error', 'unsafe_allocation_address', 'unsafe_allocation_bytes', 'public_bytes_restored'))
        and all(core[k] == 1 for k in ('input_before_captured', 'input_unchanged', 'published', 'publication_exact', 'released')),
        'PUBLICATION_READBACK_RELEASE_WITHOUT_LATE_PUBLIC_RESTORE', loc)
    roles = (1, 2) if mode == 3 else (2,)
    a.check(len(rows) == 1+4*len(roles) and rows[0]['kind'] == 1 and rows[0]['parent'] == p
        and [sequence(r) for r in rows] == list(range(p['token'], p['token']+len(rows))), 'ALL_HELPER_AND_EFFECT_RAW_EVENTS', loc)
    by_seq = {sequence(r):r for r in rows[1:]}
    owned = []
    for role in roles:
        child = core['reference' if role == 1 else 'effect']; plan = child['plan']
        a.check(plan['role'] == role and plan['reserved'] == 0 and plan['parent_token'] == p['token']
            and plan['child_lease'] == 4*p['token']+role, 'REFERENCE_EFFECT_CHILD_ROLE', loc)
        ids = [child[k] for k in ('prepare_enter', 'prepare_exit', 'execute_enter', 'execute_exit')]
        a.check(all(x < y for x, y in zip([p['token']] + ids, ids)) and child['registration'] == ids[0]
            and positive(child['executor']) and positive(child['stream']), 'REAL_CHILD_GWS_EXECUTE_ORDER', loc)
        a.check(all(child[k] == 1 for k in ('prepare_original_calls', 'execute_original_calls', 'prepared', 'returned', 'drained', 'readback'))
            and child['prepare_status'] == child['execute_status'] == 0, 'EXACTLY_ONCE_CHILD_DRAIN_READBACK', loc)
        a.check(child['workspace_capacity'] >= child['workspace_required']
            and bool(child['workspace']) == bool(child['workspace_capacity'])
            and child['workspace_group_owned'] == int(role == 1 and child['workspace_required'] > 0), 'WORKSPACE_OWNERSHIP', loc)
        for kind, seq in enumerate(ids, 1):
            row = by_seq[seq]
            a.check(row['kind'] == 2 and row['binding'] == dict(parent_token=p['token'], child_lease=plan['child_lease'],
                owner_hi=lease['owner_hi'], owner_lo=lease['owner_lo'], generation=generation, role=role, reserved=0),
                'EXACT_CHILD_OWNER_BINDING', loc)
            event(a, row['native'], p, plan, child, kind, loc)
        for name in ('input', 'quant', 'scale'):
            v = plan[name]; expected = fields(p[name], GEOMETRY)
            if mode == 3 and role == 2:
                expected['rows'] = 1
                if name in ('input', 'quant'):
                    expected['columns'] = expected['stride0'] = p['input']['rows'] * p['input']['columns']
                else:
                    expected['bytes'] = 4
            a.check(fields(v, GEOMETRY) == expected and positive(v['descriptor']), 'ACTUAL_REFERENCE_OR_FLAT_EFFECT_GEOMETRY', loc)
            if name == 'input' or mode in (1, 4):
                a.check(v['address'] == p[name]['address'], 'CURRENT_INPUT_OR_BORROWED_PUBLIC_OUTPUT', loc)
            else:
                span = (v['address'], v['address'] + p[name]['bytes'])
                a.check(positive(span[0]) and all(span[1] <= p[k]['address'] or p[k]['address']+p[k]['bytes'] <= span[0]
                    for k in ('input', 'quant', 'scale')), 'OWNED_OUTPUT_SEPARATE_FROM_BORROWED_STORAGE', loc)
                owned.append(span)
    a.check(all(x[1] <= y[0] or y[1] <= x[0] for i, x in enumerate(owned) for y in owned[i+1:]), 'DISTINCT_OWNED_OUTPUT_SPANS', loc)
    if mode == 3:
        a.check(core['reference']['execute_exit'] < core['effect']['execute_enter']
            and core['reference']['stream'] == core['effect']['stream'] and core['reference_effect_compared'] == 1
            and 0 <= core['changed_quant_bytes'] <= p['quant']['bytes'] and 0 <= core['changed_scale_bytes'] <= p['scale']['bytes']
            and core['changed_quant_bytes'] + core['changed_scale_bytes'] > 0, 'CURRENT_INPUT_REFERENCE_EFFECT_BYTES', loc)
    else:
        a.check(core['reference_effect_compared'] == core['changed_quant_bytes'] == core['changed_scale_bytes'] == 0
            and core['reference']['prepare_enter'] == 0, 'NO_FAULT_IN_REFERENCE_LOSSLESS_OFF', loc)
    b = packet['binding']
    a.check(b['call'] == p['token'] and b['native_thread'] == p['thread'] and b['target'] == lease['target']
        and b['actual_row_proof'] == lease['actual_row_proof'] and b['original_model_calls'] == 1
        and b['original_prepare_calls'] == b['original_execute_calls'] == len(roles)
        and b['helper_calls'] == int(mode == 3), 'DERIVED_BINDING_COUNTS_MATCH_RAW', loc)


def row_proof(a, proof, trace, rows, loc):
    a.check(proof['verified'] is True and proof['source'] == 'ACTUAL_DISPATCHER_ROW_PRESERVING_CHAIN'
        and proof['forward_id'] == trace['forward_id'] and proof['actual_rows'] == rows, 'ACTUAL_REQUEST_ROW_PROOF', loc)
    p = proof['observation_prefix']
    a.check(p['forward_id'] == trace['forward_id'] and positive(p['observer_id'])
        and positive(p['observed_at_monotonic_ns']) and p['actual_calls'] == p['retained_calls'] <= trace['actual_calls']
        and all(p[k] == 0 for k in ('missing_calls', 'errors', 'source_errors')), 'HEALTHY_CURRENT_OBSERVATION_PREFIX', loc)
    nodes = trace['nodes']; node = nodes[proof['node_id']-1]
    a.check(node['node_id'] == proof['node_id'] and node['metadata'] == proof['tensor'] and node['row_axis'] == 0
        and node['write_epoch'] == proof['observed_write_epoch'] and node['rows_sha256'] == digest(rows), 'PROOF_NODE_STORAGE_WRITE_EPOCH', loc)
    todo, seen = [node['node_id']], set()
    while todo:
        index = todo.pop()
        if index in seen:
            continue
        seen.add(index); n = nodes[index-1]
        a.check(n['node_id'] == index and n['rows_sha256'] == digest(rows)
            and all(type(parent) is int and 0 < parent < index for parent in n['parents']), 'ROW_ANCESTRY_DAG', loc)
        if not n['parents']:
            a.check(n['operator'] == 'ACTUAL_SCHEDULER_EMBEDDING_RMSNORM'
                and n['evidence']['actual_rows'] == rows, 'ACTUAL_SCHEDULER_ROW_SEED', loc)
            tensor_join(a, n['metadata'], n['evidence']['norm']['normalized'], loc)
        todo.extend(parent for parent in n['parents'] if type(parent) is int and 0 < parent < index)
    source = trace['source_boundaries']
    for seq in p['parent_source_sequences']:
        joined = [r for r in source['boundaries'] if r['sequence'] == seq]
        a.check(len(joined) == 1 and joined[0]['status'] == 'OBSERVED', 'ACTUAL_SOURCE_PARENT_RETURNED', loc)


def projection(a, public, projection_row, forward, inventory, loc, natural):
    calls = [inventory['public_calls'][index] for index in projection_row['api_call_indices']]
    a.check(len(calls) == 2 and [v['operator'] for v in calls] == ['npu_dynamic_quant', 'npu_quant_matmul']
        and calls[0] is public and projection_row['dynamic_calls'] == projection_row['qmm_calls'] == 1
        and not projection_row['errors'] and all(v['original_calls'] == 1 and not v['errors'] for v in calls),
        'ONE_REAL_PRODUCER_AND_CONSUMER', loc)
    qmm = calls[1]
    a.check(qmm['formals']['x1'] == public['quantized_output'] and qmm['formals']['pertoken_scale'] == public['per_token_scale'],
        'CONSUMER_READS_ACTUAL_Q_AND_SCALE_OBJECTS', loc)
    a.check(all(v['forward_id'] == forward['forward_id'] and v['request_ids'] == projection_row['request_ids']
        and v['module_object_identity'] == projection_row['module_object_identity'] for v in calls)
        and projection_row['rank'] == 0, 'ACTUAL_PROJECTION_REQUEST_OWNER', loc)
    trace = forward['row_provenance']; left = public['request_row_lineage']; right = projection_row['output_row_lineage']
    a.check(left['actual_rows']['request_ids'] == projection_row['request_ids'], 'CONSUMER_ACTUAL_REQUEST_ROWS', loc)
    for proof in (left, right):
        row_proof(a, proof, trace, left['actual_rows'], loc)
    tensor_join(a, left['tensor'], public['float_input'], loc)
    tensor_join(a, right['tensor'], qmm['actual_output'], loc)
    output_node = trace['nodes'][right['node_id']-1]
    e = output_node['evidence']
    a.check(output_node['operator'] == 'ACTUAL_NATIVE_DYNAMIC_QUANT_TO_OWNED_PUBLIC_QMM'
        and output_node['parents'] == [left['node_id']]
        and e['public_api_indices'] == projection_row['api_call_indices']
        and e['module_object_identity'] == projection_row['module_object_identity']
        and e['weight_object_identity'] == projection_row['weight_object_identity']
        and e['method_source_sha256'] == projection_row['method_source_sha256']
        and e['actual_native_dynamic_call'] == public['independent_native']['binding']
        and e['native_build'] == inventory['native_build'], 'ACTUAL_NATIVE_TO_QMM_OUTPUT_NODE', loc)
    lp, rp = left['observation_prefix'], right['observation_prefix']
    a.check(lp['observer_id'] == rp['observer_id'] and lp['actual_calls'] <= rp['actual_calls']
        and lp['observed_at_monotonic_ns'] <= rp['observed_at_monotonic_ns'], 'INPUT_TO_CONSUMER_CLOSED_PREFIX', loc)
    a.check(not trace['active'] and not trace['cleanup_pending'] and trace['owned_dispatch_modes'] == 0
        and (trace['source_boundaries'] is None or all(not trace['source_boundaries'][k]
            for k in ('active', 'owned_nested_dispatch_modes', 'owned_profile_callbacks'))), 'CLOSED_PREFIX_OBSERVERS_RELEASED', loc)
    return dict(input=left, consumer=right, complete_at_consumer=True, full_forward_row_complete=trace['complete'])


def inventory_review(a, inventory, count, natural):
    public, forwards, native = inventory['public_calls'], inventory['forwards'], inventory['native_forwards']
    a.check(len(forwards) == len(native) and inventory['actual_public_api_calls'] == inventory['retained_public_api_calls'] == len(public)
        and inventory['missing_public_call_details'] == 0 and not inventory['failure'] and inventory['unowned_public_calls'] == 0,
        'COMPLETE_PUBLIC_DENOMINATOR')
    structural = inventory['structural_candidates']; ids = [r['module_object_identity'] for r in structural]
    a.check(len(ids) == len(set(ids)) == inventory['structural_candidate_count'], 'STRUCTURAL_UNIVERSE_RETAINED')
    selected_public = []
    total = Counter(); prior_end = 0
    for index, (f, nf) in enumerate(zip(forwards, native)):
        loc = f'forward[{index}]'
        a.check(f['forward_id'] == nf['forward_id'] == f['request_ids'][0] and f['public_start'] == prior_end,
            'ACTUAL_FORWARD_PUBLIC_WINDOW', loc)
        prior_end = f['public_end']; local = public[f['public_start']:prior_end]
        a.check(len(f['projections']) == count and len(local) == 2*count, 'ALL_NATURAL_PROJECTIONS_PUBLIC_CALLS', loc)
        indices = [n for p in f['projections'] for n in p['api_call_indices']]
        a.check(sorted(indices) == list(range(f['public_start'], prior_end))
            and all(p['dynamic_calls'] == p['qmm_calls'] == 1 and not p['errors'] for p in f['projections'])
            and all(p['original_calls'] == 1 and not p['errors'] for p in local), 'EVERY_PUBLIC_CALL_HAS_ONE_ACTUAL_PROJECTION', loc)
        total.update(raw_window(a, nf['before'], nf['after'], nf['native_events'], loc))
        parents = [r for r in nf['native_events'] if r['kind'] == 1]
        a.check(len(parents) == len(nf['candidates']) == count and not nf['unowned_native_events'], 'FULL_NATURAL_PARENT_DENOMINATOR', loc)
        raw_by_seq = {sequence(r):r for r in nf['native_events']}; seen = []
        for candidate, parent in zip(nf['candidates'], parents):
            a.check(candidate['native'] == parent, 'CANDIDATE_IS_ACTUAL_NATIVE_PARENT', loc)
            p = public[candidate['public_api_index']]; n = p['independent_native']; rows = n['native_events']
            a.check(p['operator'] == 'npu_dynamic_quant' and p['original_calls'] == 1 and not p['errors'] and not n['errors']
                and p['forward_id'] == f['forward_id'] and p['request_ids'] == f['request_ids']
                and p['module_object_identity'] == candidate['module_object_identity'] in ids
                and rows[0] == parent and all(raw_by_seq[sequence(r)] == r for r in rows), 'ACTUAL_PUBLIC_NATIVE_PARENT_JOIN', loc)
            seen.extend(sequence(r) for r in rows)
            raw_window(a, n['native_before'], n['native_after'], rows, loc)
            for role, key in (('input', 'float_input'), ('quant', 'quantized_output'), ('scale', 'per_token_scale')):
                view(a, parent['parent'][role], p[key], role, loc, natural)
            if 'completion' not in n:
                passthrough(a, rows, loc)
            else:
                selected_public.append(p)
        a.check(sorted(seen) == sorted(raw_by_seq) and len(seen) == len(set(seen)), 'EVERY_RAW_EVENT_HAS_ONE_PUBLIC_OWNER', loc)
        a.check(all(r['module_object_identity'] in ids for r in f['projections']), 'STRUCTURAL_ACTUAL_PROJECTION_JOIN', loc)
    a.check(prior_end == len(public) and total['model_parents'] == inventory['actual_native_calls']
        and total['native_prepares'] == inventory['actual_native_prepare_calls']
        and total['native_executes'] == inventory['actual_native_execute_calls'], 'FULL_NATIVE_COUNTER_TOTALS')
    return dict(forwards=len(forwards), parents=total['model_parents'], native_prepares=total['native_prepares'],
                native_executes=total['native_executes'], retained_unresolved=[sum(bool(c['reasons']) for c in n['candidates']) for n in native])


def selection(a, serving, discovery, inventory, natural):
    report = discovery['report']
    a.check(digest(report) == discovery['sha256'] == serving['discovery_sha256'], 'FROZEN_DISCOVERY_DIGEST')
    a.check(len(report['forwards']) == len(report['native_forwards']) == 2
        and report['forwards'] == inventory['forwards'][:2] and report['native_forwards'] == inventory['native_forwards'][:2]
        and report['public_calls'] == inventory['public_calls'][:report['forwards'][-1]['public_end']], 'FROZEN_TWO_HEALTHY_CENSUSES')
    chosen = None
    for candidate in report['native_forwards'][0]['candidates']:
        module = candidate['module_object_identity']; pairs = []
        for i in range(2):
            matches = [c for c in report['native_forwards'][i]['candidates'] if c['module_object_identity'] == module]
            prs = [p for p in report['forwards'][i]['projections'] if p['module_object_identity'] == module]
            if len(matches) != 1 or len(prs) != 1 or any(r != 'COMPLETE_REQUEST_ROW_OPERATOR_OBSERVATION_UNPROVEN' for r in matches[0]['reasons']):
                break
            p = report['public_calls'][matches[0]['public_api_index']]
            local = Review(); prefix = local.section('candidate', projection, local, p, prs[0], report['native_forwards'][i], report, str(module), natural)
            if local.errors:
                break
            pairs.append((matches[0], p, prs[0], prefix))
        if len(pairs) != 2:
            continue
        c0, p0, pr0, prefix0 = pairs[0]; c1, p1, pr1, prefix1 = pairs[1]
        if (fields(pr0, PROJECTION) != fields(pr1, PROJECTION) or p0['healthy_formals'] != p1['healthy_formals']
                or not all(v['finite'] for v in p0['healthy_formals'].values())
                or signature(p0['request_row_lineage']['actual_rows']) != signature(p1['request_row_lineage']['actual_rows'])
                or any(fields(c0['native']['parent'][k], GEOMETRY) != fields(c1['native']['parent'][k], GEOMETRY) for k in ('input', 'quant', 'scale'))):
            continue
        chosen = (c0, p0, pr0, [prefix0, prefix1]); break
    a.check(chosen is not None, 'AUTOMATIC_FIRST_FULLY_PROVEN_SINGLE_SITE_EXISTS')
    if chosen is None:
        return None
    c, p, pr, prefixes = chosen; target = serving['selected_site']; identity = target['identity']
    a.check(target['sha256'] == digest(identity) and identity['scope'] == SCOPE
        and identity['lifetime'] == inventory['lifetime'] and identity['native_build'] == serving['native_build'] == inventory['native_build']
        and identity['projection'] == fields(pr, PROJECTION) and identity['geometry'] == {k:fields(c['native']['parent'][k], GEOMETRY) for k in ('input', 'quant', 'scale')}
        and identity['rows'] == signature(p['request_row_lineage']['actual_rows'])
        and serving['selected_discovery_prefixes'] == prefixes, 'AUTOMATIC_SELECTION_IDENTITY_AND_PREFIXES')
    return p


def evidence(serving, inventory, discovery, *, natural=True, count=135):
    a = Review(); summary = a.section('inventory', inventory_review, a, inventory, count, natural)
    healthy = a.section('selection', selection, a, serving, discovery, inventory, natural)
    a.check([r['phase'] for r in serving['rows']] == list(PHASES) and len(inventory['forwards']) == 7, 'SEVEN_NATURAL_PHASES')
    a.check([r['phase'] for r in serving['receipts']] == list(MODES), 'FOUR_SINGLE_SITE_NATIVE_OUTCOMES')
    a.check(not serving['error'] and not serving['collection_errors'] and all(not serving[k] for k in ZERO_OWNED)
        and not inventory['active'] and not inventory['installed'] and inventory['owned_method_patches'] == inventory['owned_api_patches'] == 0,
        'ALL_OWNED_OBSERVERS_AND_PATCHES_RELEASED')
    metadata = serving['metadata_producers']
    a.check(metadata is not None and all(not metadata[k] for k in ('active', 'owned_profile_callbacks',
        'owned_dispatch_modes', 'cleanup_pending', 'live_producer_records')), 'EARLY_METADATA_OWNERS_RELEASED')
    target = serving['selected_site']; module = target['identity']['projection']['module_object_identity']; requests = set()
    for index, phase in enumerate(PHASES):
        def phase_review():
            row, f, nf = serving['rows'][index], inventory['forwards'][index], inventory['native_forwards'][index]
            scheduler = row['decode']['decode']; ids = scheduler['request_ids']
            a.check(len(ids) == len(set(ids)) == 8 and not requests.intersection(ids)
                and ids == f['request_ids'] and scheduler == f['row_mapping'], 'ACTUAL_FRESH_B8_REQUEST_MAP', phase)
            requests.update(ids)
            a.check(signature(scheduler) == target['identity']['rows'], 'FROZEN_PROMPTS_FIRST_DECODE_ROWS', phase)
            if natural:
                a.check(scheduler['counts'] == [1]*8 and scheduler['row_request_indices'] == list(range(8))
                    and scheduler['request_count'] == scheduler['token_row_count'] == scheduler['scheduler_capacity'] == 8
                    and scheduler['pure_decode'] is True and sorted(scheduler['logical_ids']) == list(range(8)), 'ACTUAL_FIRST_DECODE_SCHEDULER', phase)
                a.check(all(pos == len(prefix['prompt']) and len(prefix['output']) == 1 and token == prefix['output'][0]
                    for pos, prefix, token in zip(scheduler['row_token_positions'], scheduler['prefixes'], scheduler['input_ids'])), 'ACTUAL_PREFIX_TOKEN_POSITION', phase)
                ih = hashlib.sha256(struct.pack('<8i', *scheduler['input_ids'])).hexdigest()
                ph = hashlib.sha256(struct.pack('<8q', *scheduler['row_token_positions'])).hexdigest()
                a.check(f['input_sha256'] == hashlib.sha256(json.dumps([ih, ph]).encode()).hexdigest(), 'RECOMPUTED_ACTUAL_INPUT_HASH', phase)
            for stage in ('prefill', 'decode'):
                value = row[stage]; base = serving['rows'][0][stage]
                a.check(value['returned'] is True and value['logits_returned'] is True
                    and valid_sha(value['hidden_sha256']) and valid_sha(value['logits_sha256']) and len(value['top1']) == 8,
                    'COMPLETE_MODEL_OUTPUT_RECEIPTS', phase + '.' + stage)
                if stage == 'prefill' or phase != 'fault':
                    a.check(value['hidden_finite'] is True and value['logits_finite'] is True
                        and all(value[k] == base[k] for k in ('hidden_sha256', 'logits_sha256')), 'FULL_HEALTHY_MODEL_OUTPUT_RESTORED', phase + '.' + stage)
            a.check(f['model_output_sha256'] == row['decode']['logits_sha256']
                and f['model_output_finite'] == row['decode']['logits_finite'], 'INVENTORY_FULL_LOGITS_JOIN', phase)
            selected = [p for p in f['projections'] if p['module_object_identity'] == module]
            a.check(len(selected) == 1 and fields(selected[0], PROJECTION) == target['identity']['projection'], 'SAME_ACTUAL_SELECTED_PROJECTION', phase)
            pr = selected[0]; public = inventory['public_calls'][pr['api_call_indices'][0]]
            prefix = projection(a, public, pr, nf, inventory, phase, natural)
            weight = target['identity']['weight']; operand = inventory['public_calls'][pr['api_call_indices'][1]]['formals']['x2']
            a.check(weight['object_id'] == pr['weight_object_identity']
                and all(weight[k] == operand[k] for k in ('storage_identity', 'storage_bytes', 'dtype', 'device'))
                and weight['storage_address'] == operand['storage_addr'], 'FROZEN_PROJECTION_WEIGHT_STORAGE', phase)
            if phase in (*MODES, 'off'):
                a.check(serving['completed_projection_prefixes'][phase] == prefix, 'SAVED_COMPLETED_PREFIX_MATCH', phase)
            if phase != 'fault':
                a.check(public['healthy_formals'] == healthy['healthy_formals'], 'FULL_INPUT_Q_SCALE_HEALTHY_HASHES', phase)
            else:
                a.check(public['healthy_formals']['input'] == healthy['healthy_formals']['input']
                    and any(row['decode'][k] != serving['rows'][0]['decode'][k] for k in ('hidden_sha256', 'logits_sha256')),
                    'SAME_CURRENT_INPUT_WITH_COMPLETE_MODEL_EFFECT', phase)
            if phase in MODES:
                packet = next(p for p in serving['receipts'] if p['phase'] == phase)
                receipt(a, packet, serving['owner'], serving['generation'], target['sha256'], inventory['lifetime'], phase)
                n = public['independent_native']
                a.check(packet['completion'] == n['completion'] and packet['raw_events'] == n['native_events']
                    and packet['row_proof'] == n['row_proof'] and packet['admission_row_proof'] == n['admission_row_proof'], 'SAVED_PUBLIC_RECEIPT_JOIN', phase)
                proof, fresh = packet['row_proof'], packet['admission_row_proof']
                for proof_row in (proof, fresh):
                    row_proof(a, proof_row, nf['row_provenance'], scheduler, phase)
                    tensor_join(a, proof_row['tensor'], public['float_input'], phase)
                a.check(all(fresh[k] == proof[k] for k in ('forward_id', 'node_id', 'actual_rows', 'tensor', 'observed_write_epoch'))
                    and fresh['observation_prefix']['observer_id'] == proof['observation_prefix']['observer_id']
                    and fresh['observation_prefix']['parent_source_sequences'] == proof['observation_prefix']['parent_source_sequences']
                    and fresh['observation_prefix']['actual_calls'] >= proof['observation_prefix']['actual_calls']
                    and fresh['observation_prefix']['observed_at_monotonic_ns'] >= proof['observation_prefix']['observed_at_monotonic_ns'],
                    'FRESH_NATIVE_INPUT_PROOF_NOT_REUSED', phase)
                dispatch = fresh['native_input_dispatch']
                if dispatch is not None:
                    ev = nf['row_provenance']['events'][dispatch['sequence']-1]
                    a.check(ev['sequence'] == dispatch['sequence'] and ev['original_calls'] == 1 and not ev['mutable']
                        and ev['status'] == 'OBSERVED' and digest(ev['schema']) == dispatch['schema_sha256']
                        and ev['inputs'] == [dispatch['actual_input']] == [fresh['tensor']]
                        and dispatch['sequence'] > proof['observation_prefix']['actual_calls']
                        and (dispatch['operator'] == 'npu.npu_dynamic_quant.default' or not natural), 'ACTUAL_CURRENT_NATIVE_DISPATCH_FRAME', phase)
                proof_sha = digest(dict(proof=fresh, public_proof=proof, actual_call=packet['completion']['call'],
                    target=target['sha256'], owner=serving['owner'], generation=serving['generation']))
                a.check(packet['lease']['actual_row_proof'] == words(proof_sha), 'LEASE_BINDS_FRESH_ACTUAL_INPUT_PROOF', phase)
                if phase == 'fault':
                    for role in ('quant', 'scale'):
                        changed = packet['completion']['core']['changed_' + role + '_bytes']
                        a.check((public['healthy_formals'][role]['sha256'] != healthy['healthy_formals'][role]['sha256']) == bool(changed),
                            'PUBLICATION_HASH_AGREES_WITH_NATIVE_CHANGE:' + role, phase)
        a.section(phase, phase_review)
    callbacks = serving['observed_owner_callbacks']; tokens = [v['call']['token'] for v in callbacks]
    a.check(len(tokens) == len(set(tokens)) and all(v['completed'] is True and v['status'] == 0
        and v['admitted'] == int(v['selected']) for v in callbacks), 'ALL_OWNER_CALLBACKS_COMPLETED_WITHOUT_FOREIGN_ADMISSION')
    a.check([v['phase'] for v in callbacks if v['selected']] == list(MODES), 'ONLY_FOUR_SELECTED_ADMISSIONS')
    for packet in serving['receipts']:
        joined = [v for v in callbacks if v['call'] == packet['completion']['call']]
        a.check(len(joined) == 1 and joined[0]['lease'] == packet['lease']
            and joined[0]['admission_row_proof'] == packet['admission_row_proof'], 'CALLBACK_LEASE_CURRENT_INPUT_JOIN', packet['phase'])
    a.check(len(serving['native_recovery']) == 1, 'ONE_OWNER_RECOVERY')
    for r in serving['native_recovery']:
        a.section('before_remove', zero_state, a, r['before_remove'], 'before_remove')
        a.section('after_remove', zero_state, a, r['after_remove'], 'after_remove')
        a.check(r['worker'] == inventory['worker_identity'] and r['actual_global_device_synchronize'] is True
            and r['late_public_restore_requested'] is False and r['empty_owner_uses_actual_native_observed_stream_drain'] is False,
            'ACTUAL_DEVICE_DRAIN_BEFORE_OWNER_REMOVAL')
        a.check(r['after_remove']['last_sequence'] < inventory['native_forwards'][-1]['before']['last_sequence']
            and r['before_remove']['core']['admitted_model_calls'] == 4, 'RECOVERY_BEFORE_FRESH_HEALTHY_OFF')
        first = serving['receipts'][0]['lease']
        a.check(all(s['core']['stopped'] == s['core']['sealed'] == 1
            and all(s['core'][key] == first[key] for key in ('owner_hi', 'owner_lo', 'generation'))
            for s in (r['before_remove'], r['after_remove'])), 'STOP_DRAIN_RECOVER_SAME_NATIVE_OWNER')
    return dict(**a.summary(), denominator=summary, selected_site_sha256=target['sha256'],
        source_bound_single_site_c1=not a.errors, complete_arrays_independently_rehashed=False)


def host_review(a, result, host, serving):
    actions = ['attach', 'prepare', 'register', 'check', 'execute'] + ['advance', 'advance', 'advance', 'query']*7 + ['finish', 'unregister', 'detach']
    history = host['history']; a.check([r['action'] for r in history] == actions, 'PUBLIC_CORE_PROTOCOL_ORDER')
    worker = result['deployment']['worker']['worker']; ticket = history[1]['response']['ticket']
    a.check(host['owner'] == ticket['owner'] and ticket['generation'] == serving['generation']
        and host['worker'] == ticket['worker'] == worker and ticket['tenant_id'] == host['tenant_id'], 'HOST_RUNTIME_NATIVE_OWNER_JOIN')
    a.check(isinstance(serving['owner'], str) and len(serving['owner']) == 32
        and all(c in '0123456789abcdef' for c in serving['owner']), 'OPAQUE_RUNTIME_PROVIDER_OWNER')
    a.check(history[0]['response']['ticket'] == fields(ticket, ('worker', 'tenant_id', 'attach_nonce')), 'ACTUAL_ATTACHMENT_JOIN')
    for index, row in enumerate(history[1:-1], 1):
        reply = row['response']; action = row['action']; state = reply['provider_state']
        a.check(reply['ticket'] == ticket and reply['status'] == ('QUERY_OK' if action == 'query' else 'ACTION_COMPLETED')
            and reply['recovery_required'] is False, 'PUBLIC_RPC_EXACT_TICKET_SUCCESS', index)
        a.check(state['inflight'] == state['leases'] == 0, 'RPC_HAS_NO_ACTIVE_CALL_LEASES', index)
        a.check(state['native_registered'] == bool(state['native']['owned'])
            and state['native']['pending'] == state['native']['executing'] == 0, 'NATIVE_REGISTERED_DISTINCT_FROM_ACTIVE_LEASES', index)
        a.check(state['serving']['owner'] == serving['owner'] and state['serving']['generation'] == serving['generation'], 'PROVIDER_OWNER_GENERATION', index)
    for i, phase in enumerate(PHASES):
        original = result['phases'][i]; scheduler = serving['rows'][i]['decode']['decode']
        a.check(original['phase'] == phase and original['enqueued'] == scheduler['request_ids']
            and {r['internal_request_id'] for r in original['mapping']} == set(original['enqueued'])
            and len({r['external_request_id'] for r in original['mapping']}) == 8
            and all(positive(r['request_state_object_id']) for r in original['mapping']), 'ACTUAL_HOST_ENQUEUE_MAPPING', phase)
        group = history[5+4*i:9+4*i]
        a.check(group[0]['response']['provider_state']['serving']['next_phase'] == phase
            and group[-1]['response']['provider_state']['serving']['rows'] == serving['rows'][:i+1], 'REQUEST_PHASE_COLLECTION_JOIN', phase)
        a.check(len(original['output_tokens']) == 8 and all(len(v) == 2 for v in original['output_tokens']), 'ACTUAL_BATCH_COMPLETION_TOKENS', phase)
    a.check(history[-3]['response']['provider_state']['serving'] == result['serving_final'] == serving
        and history[-2]['response']['provider_state']['registered'] is False
        and history[-1]['response']['status'] == 'DETACHED' and host['ticket'] is None and host['attachment'] is None
        and host['ownership_retained'] is False, 'FINISH_UNREGISTER_DETACH_ZERO_OWNER')
    a.check(result['remove']['status'] == 'DEPLOYMENT_REMOVED'
        and result['worker_after']['worker'] == worker and result['worker_after']['quarantined'] is False
        and result['healthy_before'] == result['healthy_after'] and len(result['healthy_after']) == 8
        and all(len(v) == 2 for v in result['healthy_after']), 'SAME_WORKER_POST_UNREGISTER_ORDINARY_HEALTH')


def provenance(a, result, serving, source_root, native_root):
    manifest_path = under(source_root, 'source_manifest.json'); manifest = read(manifest_path)
    a.check(file_sha(manifest_path) == SOURCE_MANIFEST and manifest['base_commit'] == SOURCE_COMMIT, 'FROZEN_SOURCE64_MANIFEST')
    entries = {v['path']:v['sha256'] for v in manifest['files']}
    a.check(len(entries) == len(manifest['files']), 'UNIQUE_SOURCE_MANIFEST_PATHS')
    for name in SOURCES:
        a.check(file_sha(under(source_root, name)) == entries[name], 'SOURCE_FILE_REHASH', name)
    setup = result['deployment']['sources']
    a.check(len(setup) == len(SETUP), 'COMPLETE_INSTALLED_SETUP_SOURCE_SET')
    for row, name in zip(setup, SETUP):
        expected_path = '/installed/' + (name[4:] if name.startswith('src/') else name)
        a.check(row['path'] == expected_path and row['sha256'] == entries[name], 'ACTUAL_INSTALLED_SOURCE_JOIN', name)
    path = under(native_root, 'build.json'); build = read(path); identity = serving['native_build']
    a.check(file_sha(path) == BUILD_SHA == identity['build_sha256'] and build['schema'] == 'demoncat.quant.group_deployment.v1'
        and build['cann'] == '8.5.1' and build['actual_npu_calls'] == 0 and build['legacy_c1_library_present'] is False,
        'NATIVE_A2_BUILD_IDENTITY')
    for relative, sha in build['files'].items():
        a.check(file_sha(under(native_root, relative)) == sha, 'NATIVE_BUILD_FILE_REHASH', relative)
    for name in SOURCES[-3:]:
        a.check(build['sources'][name] == entries[name] == build['files'][Path(name).name], 'NATIVE_INSTALLED_COMPILE_SOURCE_JOIN', name)
    a.check(identity['sdk_abi_sizes'] == build['sdk_identity']['abi_sizes']
        and all(identity['sdk_abi_sizes'][name] == size for name, size in ABI.items())
        and identity['actual_image_id'] == build['actual_image_id'] and identity['source_commit'] == build['source_commit']
        and identity['library_sha256'] == build['files'][build['library']], 'ACTUAL_LOADED_BUILD_ABI_LIBRARY_JOIN')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, required=True)
    args = parser.parse_args(); artifacts = Review(); values = {}
    for key, relative in FILES.items():
        def load():
            path = under(args.evidence_root, relative)
            if not artifacts.check(file_sha(path) == PINS[key], 'ORIGINAL_ARTIFACT_SHA256', relative):
                return None
            return read(path)
        values[key] = artifacts.section(relative, load)
    report = dict(schema='demoncat.review.quant.single_c1.source64.v1', c2=False, c3=False, e5='NOT_ASSESSED',
        scope=SCOPE, artifacts=artifacts.summary(), original_artifacts=PINS,
        limitations=['All tensor values are saved full-value hashes and Native readback receipts; complete arrays were not saved and are not independently rehashed.',
            'Runtime object identity, current-input liveness, synchronization and Native byte changes are source-bound receipts, not new live observations.',
            'The complete 135-per-decode census remains retained; qualification is limited to one automatically selected site and one F.',
            'The post-unregister ordinary health check covers same-Worker token output; it does not capture post-unregister full logits.',
            'No model-weight file or external SDK binary is independently rehashed by this review. No E5 statistics or fault-only localization is assessed.'])
    if all(v is not None for v in values.values()):
        prov, host = Review(), Review()
        prov.section('provenance', provenance, prov, values['result'], values['serving'], args.source_root, args.native_root)
        report['provenance'] = prov.summary()
        host.section('public_core', host_review, host, values['result'], values['host'], values['serving'])
        report['public_core'] = host.summary()
        ev = Review()
        result = ev.section('evidence', evidence, values['serving'], values['inventory'], values['discovery'])
        report['evidence'] = result if result is not None else ev.summary()
    good = all(report.get(k, {}).get('consistent') is True for k in ('artifacts', 'provenance', 'public_core', 'evidence'))
    report.update(source_bound_single_site_c1_receipts_reconciled=good,
        status='SOURCE_BOUND_SINGLE_SITE_QUANT_C1_RECEIPTS_RECONCILED_NOT_C2_C3_E5' if good else 'REFUSED_INCOMPLETE_OR_CONTRADICTORY_EVIDENCE')
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if good else 2


if __name__ == '__main__':
    raise SystemExit(main())
