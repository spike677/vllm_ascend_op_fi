"""Read-only public Core41 receipt review; stdlib only, no Worker or Native load.

The existing KV41 reviewer checks complete logical prefixes and recovery. This
layer joins those records to the actual public protocol and original live-proof
audit receipts. Saved JSON is evidence, never a reconstructible C2 capability.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys


NATIVE_SHA = '8876f2b99e70a9df8201b6ae0ace2cc4be8702afcf6703f9b72d1dda3bc26622'
SOURCE_COMMIT = '62095b0a5fc4c6eb7450f666cdf5f8a095ba7b18'
SOURCE_MANIFEST = '6f8ec1672a75b89a7dc4884b59d4285790482a05088e946d2ebd11aadd71785f'
HELPER_SHA = '8d479a7c91239e027c21b987c163e1cfc0c44a42f2418d982118343c5521731f'
PHASES = ('c1_observe_0', 'c1_observe_1', 'c1_lossless', 'c1_fault', 'c1_off',
          'group_observe_0', 'group_observe_1', 'c2_lossless', 'c2_fault_0', 'c2_fault_1',
          'c2_off', 'c2_health', 'c3_lossless', 'c3_fault_0', 'c3_fault_1', 'c3_off', 'c3_health')
INNER = ('observe', 'observe', 'lossless', 'fault', 'off', 'group_observe', 'group_observe',
         'group_lossless', 'group_fault', 'group_fault', 'group_off', 'health',
         'group_lossless', 'group_fault', 'group_fault', 'group_off', 'health')
ACTIONS = ['attach', 'prepare', 'register', 'check', 'enable'] + ['advance', 'query'] * 17 + [
    'finish', 'disable', 'unregister', 'detach']
COVERAGE_FIELDS = ('schema session_id scope selection trigger owner generation budget discovery observation '
    'planned_targets planned_targets_hit observed_calls observed_eligible_calls hits missing_outcomes '
    'propagation_added_calls calls recovery contract_satisfied rejection_reasons real_worker_evidence '
    'real_pretrained_model_evidence live_c2_proof public_qualification_granted').split()
CORE_PREFIX = 'src/demoncat/runtime/vllm_ascend/worker_extension/'
REQUIRED_SOURCES = {
    'fault_injection/coverage.py', 'fault_injection/coverage_session.py', 'fault_injection/runtime_discovery.py',
    'fault_injection/deployments/postwrite.py', 'fault_injection/deployments/postwrite_kv.py',
    'fault_injection/deployments/postwrite_evidence.py', 'fault_injection/deployments/postwrite_kv_inventory.py',
    'experiments/c2c3_postwrite_20261117/natural41_worker_core_probe.py',
    *(CORE_PREFIX + name for name in ('config.py', 'host.py', 'runtime.py', 'extension.py', 'identity.py',
        'registry.py', 'serving41.py', 'providers/fault41.py', 'postwrite41.py', 'postwrite41_coverage.py',
        'postwrite_collector.py', 'paged33_audit.py', 'batch_context.py', 'serving_context.py'))}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def positive(value):
    return type(value) is int and value > 0


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('DUPLICATE_JSON_KEY:' + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('NONFINITE_JSON:' + value)
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            nonfinite(value)
        return result
    with Path(path).open('r', encoding='utf-8') as stream:
        return json.load(stream, object_pairs_hook=pairs, parse_constant=nonfinite, parse_float=finite_float)


def under(root, name):
    root = Path(root).resolve(strict=True)
    name = str(name)
    if not name or '\\' in name or ':' in name or Path(name).is_absolute() or any(p in ('', '.', '..') for p in name.split('/')):
        raise ValueError('UNSAFE_ARTIFACT_PATH')
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('ARTIFACT_OUTSIDE_ROOT')
    return path


def helper():
    path = Path(__file__).with_name('review41_group_receipts.py')
    if file_sha(path) != HELPER_SHA:
        raise ValueError('REVIEWED_KV41_HELPER_CHANGED')
    spec = importlib.util.spec_from_file_location('_reviewed_kv41_receipts', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Audit:
    def __init__(self):
        self.checks, self.failures, self.issues = 0, 0, []

    def check(self, condition, code, location=''):
        self.checks += 1
        if not condition:
            self.failures += 1
            if len(self.issues) < 100:
                self.issues.append(dict(code=code, location=str(location)))
        return bool(condition)

    def section(self, name, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (KeyError, TypeError, ValueError, IndexError, AttributeError, ZeroDivisionError) as error:
            self.check(False, 'MISSING_OR_INVALID_SCHEMA:' + type(error).__name__, name)
            return None

    def report(self):
        return dict(consistent=not self.failures, checks=self.checks, failures=self.failures, issues=self.issues)


def subset(left, right):
    return isinstance(right, dict) and all(key in right and right[key] == value for key, value in left.items())


def worker_key(target):
    return [target[k] for k in ('process_id', 'process_start_identity', 'worker_id', 'rank', 'device_type', 'logical_device')]


def scheduler(a, value, request_id, prefix, location, natural):
    a.check(value['request_ids'] == [request_id] and value['counts'] == [1]
        and value['request_count'] == value['token_row_count'] == 1 and value['prefixes'] == prefix
        and value['row_token_positions'] == [len(prefix[0]['prompt'])], 'ACTUAL_FIRST_DECODE_REQUEST', location)
    a.check(len(prefix) == 1 and len(prefix[0]['output']) == 1 and len(prefix[0]['prompt']) >= 2
        and all(type(x) is int and x >= 0 for x in prefix[0]['prompt'] + prefix[0]['output']), 'EXPLICIT_TOKEN_PREFIX', location)
    if natural:
        a.check(value['pure_decode'] is True and value['scheduler_capacity'] == 1
            and value['row_request_indices'] == [0] and value['logical_ids'] == [0]
            and value['input_ids'] == prefix[0]['output'], 'ACTUAL_SCHEDULER_ROW_MAPPING', location)
        for name, dtype in (('actual_input_metadata', 'torch.int32'), ('actual_position_metadata', 'torch.int64')):
            meta = value[name]
            a.check(meta['shape'] == [1] and meta['stride'] == [1] and meta['dtype'] == dtype
                and str(meta['device']).startswith('npu') and positive(meta['object_id']) and positive(meta['storage_identity'])
                and type(meta['storage_offset']) is int and meta['storage_offset'] >= 0,
                'ACTUAL_MODEL_INPUT_BUFFER', location + '/' + name)


def contiguous(shape):
    result, n = [], 1
    for value in reversed(shape):
        if not positive(value):
            return None
        result.insert(0, n)
        n *= value
    return result


def boundary(a, proof, key, writer, location):
    """Join actual writer objects, both legal FIA views, and exact tail indices."""
    meta, cache = proof['observational_metadata'], proof['cache']
    blocks, page, heads, columns = cache['shape']
    width, elements = heads * columns, math.prod(cache['shape'])
    length, = meta['sequence_lengths']
    pages, = meta['used_pages']
    expected_slot = pages[(length - 1) // page] * page + (length - 1) % page
    indices = [h * columns + c for h in range(heads) for c in range(columns - 2, columns)]
    a.check(proof['boundary'] == 'actual_kv_writer_to_live_value_slot_to_tnd_fia'
        and proof['cache_kind'] == 'value' and proof['controlled_legal_address_proxy'] is True
        and proof['native_out_of_bounds_claim'] is False, 'ACTUAL_KV41_BOUNDARY', location)
    a.check(meta['actual_operator'] == 'torch_npu.npu_fused_infer_attention_score' and meta['layout'] == 'TND'
        and meta['metadata_binding'] == 'VERIFIED' and meta['read_only'] is True
        and meta['writer'] == writer and writer['returned'] is True and writer['module_path'] == key[0],
        'ACTUAL_WRITER_TO_CONSUMER_JOIN', location)
    a.check(columns >= 2 and proof['slot'] == expected_slot and meta['slots'] == meta['expected_slots'] == [expected_slot]
        and writer['slots'] == [expected_slot] and proof['span_element_offset'] == expected_slot * width
        and proof['span_elements'] == width and proof['relative_indices'] == indices
        and proof['allocation_element_indices'] == [cache['storage_offset'] + expected_slot * width + i for i in indices],
        'EXACT_LEGAL_TAIL_SPAN', location)
    a.check(key[4] == proof['storage'] and key[5:] == [cache['storage_offset'], cache['shape'], cache['stride']]
        and cache['dtype'] == 'torch.bfloat16' and cache['stride'] == contiguous(cache['shape'])
        and cache['logical_bytes'] == elements * 2 and cache['storage_offset'] >= 0
        and (cache['storage_offset'] + elements) * 2 <= cache['storage_bytes']
        and proof['storage'][0] == cache['device'] and proof['storage'][1] == cache['storage_identity']
        and proof['storage'][3] == cache['storage_bytes'], 'LIVE_STORAGE_AND_SPAN_BOUNDS', location)
    for name in ('key', 'value'):
        view, live, actual_writer = meta[name], meta['live_' + name + '_cache'], writer[name + '_cache']
        a.check(live == actual_writer and positive(live['object_id']) and positive(view['object_id'])
            and live['shape'] == cache['shape'] and view['shape'] == [blocks, page, width]
            and live['stride'] == contiguous(live['shape']) and view['stride'] == contiguous(view['shape'])
            and all(view[k] == live[k] for k in ('storage_identity', 'storage_offset', 'storage_nbytes', 'dtype', 'device'))
            and view['dtype'] == 'torch.bfloat16' and view['storage_offset'] >= 0
            and (view['storage_offset'] + elements) * 2 <= view['storage_nbytes'], 'ACTUAL_3D_TO_4D_STORAGE_VIEW', location + '/' + name)
    a.check(all(cache[k] == meta['value'][k] for k in ('storage_identity', 'storage_offset', 'dtype', 'device')),
        'MUTATED_VALUE_IS_ACTUAL_CONSUMER_VIEW', location)
    for name in ('key', 'query'):
        other, value = meta[name], meta['value']
        if other['storage_identity'] == value['storage_identity']:
            left, right = other['storage_offset'], value['storage_offset']
            a.check(left + math.prod(other['shape']) <= right or right + elements <= left,
                'NON_TARGET_STORAGE_DISJOINT', location + '/' + name)
    return len(indices)


def native(a, row, proof, owner, generation, applied, location):
    a.check(row['owner'] == owner and row['generation'] == generation and row['boundary'] == proof
        and row['original_calls'] == 1 and row['native_applied'] is applied
        and all(row[k] is True for k in ('consumer_returned', 'restored', 'released', 'readback_exact')),
        'ACTUAL_NATIVE_OWNER_CONSUMER_RESTORE', location)
    a.check(row['selected_elements'] == len(proof['relative_indices'])
        and row['changed_elements'] == (row['selected_elements'] if applied else 0), 'ACTUAL_CHANGED_SPAN', location)


def inventory(a, frame, run, request_id, keys, location):
    calls, writers = frame['calls'], frame['writers']
    n = len(keys)
    modules = {str(k[1]): 1 for k in keys}
    a.check(frame['complete'] is True and frame['request_ids'] == [request_id]
        and frame['forward_id'] == run['forward_id'] and frame['scheduler'] == run['actual_scheduler']
        and frame['output_sha256'] == run['model_output_sha256'] and frame['module_calls'] == modules
        and len(calls) == len(writers) == len(run['rows']) == n, 'FULL_ACTUAL_MODULE_WRITER_CONSUMER_DENOMINATOR', location)
    a.check([c['target_key'] for c in calls] == keys and len({c['call_id'] for c in calls}) == n,
        'ACTUAL_ORDERED_TARGET_SET', location)
    for index, (call, written, row) in enumerate(zip(calls, writers, run['rows'])):
        loc = location + '/' + str(index)
        a.check(call['module_path'] == keys[index][0] and call['module_object_id'] == keys[index][1]
            and call['original_calls'] == 1 and call['original_returned'] is True
            and call['eligible'] is True and call['rejection'] is None
            and call['operator'] == row['operator'] == 'npu_fused_infer_attention_score'
            and call['request_ids'] == [request_id] and call['forward_id'] == frame['forward_id']
            and (('call_id' not in row) if run['phase'].startswith('group_') else call['call_id'] == row['call_id'])
            and row['target_key'] == call['target_key']
            and row['consumer_returned'] is True and row['eligible'] is True
            and written['module_object_id'] == call['module_object_id'] and written['original_returned'] is True,
            'INDEPENDENT_ORIGINAL_CALL_AND_WRITER', loc)
        boundary(a, row['boundary'], keys[index], written['receipt'], loc)
        a.check(row['boundary']['observational_metadata']['scheduler'] == run['actual_scheduler'],
            'BOUNDARY_ACTUAL_SCHEDULER_JOIN', loc)


def requests(a, d, host, natural):
    rows, actual = d['requests'], d['worker_core']['serving']['actual_requests']
    a.check([r['phase'] for r in rows] == list(PHASES) and len(actual) == 17, 'EXACT_17_REQUEST_PROTOCOL')
    keys = [v['target_key'] for v in rows[0]['worker']['independent_inventory']['calls']]
    a.check(len(keys) > 0 and len({digest(k) for k in keys}) == len(keys), 'NONEMPTY_ACTUAL_TARGET_UNIVERSE')
    prefix = rows[0]['worker']['actual_scheduler']['prefixes']
    model, baseline = rows[0]['worker']['model_instance'], rows[0]['worker']['model_output_sha256']
    process_start = rows[0]['worker']['process_start']
    seen, forwards = set(), set()
    for i, (row, recorded) in enumerate(zip(rows, actual)):
        loc, mapping, run = PHASES[i], row['mapping'], row['worker']
        rid = mapping['internal_request_id']
        a.check(isinstance(rid, str) and rid and rid not in seen and isinstance(mapping['external_request_id'], str)
            and mapping['external_request_id'] and positive(mapping['request_state_object_id'])
            and mapping['source'] == 'ACTUAL_PUBLIC_ENQUEUE_AND_OUTPUT_PROCESSOR'
            and mapping['proof_scope'] == 'DRIVER_MAPPING_ONLY_REQUIRES_REAL_MODEL_PREHOOK_CORROBORATION',
            'REAL_UNIQUE_ENQUEUE_MAPPING', loc)
        seen.add(rid)
        a.check(row['output']['request_id'] == mapping['external_request_id'], 'DRIVER_OUTPUT_REQUEST_MAPPING', loc)
        a.check(recorded['request_id'] == rid and recorded['phase'] == loc, 'COMPLETED_REQUEST_LEDGER', loc)
        if INNER[i] == 'health':
            a.check(recorded['actual']['health'] == run and run['actual_model_object_id'] == model
                and run['process_start'] == process_start and run['model_output_sha256'] == baseline,
                'ACTUAL_UNINSTALLED_HEALTH_FULL_ROOT', loc)
            scheduler(a, run['request'], rid, prefix, loc, natural)
        else:
            a.check(subset(run, recorded['actual']) and recorded['actual']['recovery_confirmed'] is True
                and run['phase'] == INNER[i] and run['request_ids'] == [rid]
                and run['pid'] == host['worker']['pid'] and run['process_start'] == process_start
                and run['model_instance'] == model and not run['errors'] and run['native_live_leases'] == 0
                and run['model_output_snapshot'] == 'FULL_ROOT_OUTPUT_AFTER_CAPTURE_STOP'
                and valid_sha(run['model_output_sha256']), 'ACTUAL_COMPLETED_FULL_ROOT_CAPTURE', loc)
            a.check(run['forward_id'] not in forwards, 'UNIQUE_ACTUAL_FORWARD', loc)
            forwards.add(run['forward_id'])
            scheduler(a, run['actual_scheduler'], rid, prefix, loc, natural)
            inventory(a, run['independent_inventory'], run, rid, keys, loc)
            expected_hits = 1 if i == 3 else len(keys) if i in (8, 13, 14) else 0
            a.check(run['new_fault_hits'] == expected_hits, 'ACTUAL_REQUEST_HIT_DISTRIBUTION', loc)
            if i not in (3, 8, 13, 14):
                a.check(run['model_output_sha256'] == baseline, 'LOSSLESS_OFF_EXHAUSTED_FULL_ROOT_EXACT', loc)
            elif i == 3:  # Existing same-process C1 gate; group effects need not change generated output.
                a.check(run['model_output_sha256'] != baseline, 'FAULT_FULL_ROOT_DIFFERS', loc)
        if i not in (3, 8, 13, 14):
            a.check(all(row['output'][k] == d['healthy_before'][k] for k in ('tokens', 'text')), 'NORMAL_TOKEN_AND_TEXT_OUTPUT', loc)
    a.check(d['c1'] == [r['actual'] for r in actual[:5]], 'C1_ORIGINAL_REQUEST_LEDGER')
    for i in range(5):
        interventions = [r for r in rows[i]['worker']['rows'] if 'intervention' in r]
        a.check(len(interventions) == (1 if i in (2, 3) else 0), 'C1_ONE_SELECTED_NATIVE_CALL', PHASES[i])
        for row in interventions:
            a.check(row['target_key'] == keys[0], 'C1_FIRST_ACTUAL_TARGET', PHASES[i])
            native(a, row['intervention'], row['boundary'], d['deployment']['native_owner'], i + 1, i == 3, PHASES[i])
    a.check(d['worker_core']['final_cleanup']['provider']['candidate_count'] == len(keys), 'STRUCTURAL_CANDIDATE_COUNT_JOIN')
    return dict(targets=len(keys), keys=keys, prefix=prefix, model=model, baseline=baseline,
                process_start=process_start, request_ids=seen)


def group(a, d, name, context, helper_module):
    g, n = d[name], context['targets']
    start, health_index = (7, 11) if name == 'c2' else (12, 16)
    view = dict(arm='fault', groups={name:dict(coverage=g,
        phases=[dict(provider=d['requests'][start]['worker'])], health=dict(mapping=d['requests'][health_index]['mapping']))},
        cleanup=dict(provider=d['worker_core']['final_cleanup']['provider']),
        source_sha256=d.get('source_sha256', {}), setup=dict(library_sha256=d['deployment']['native_sha256']))
    _, inherited = helper_module.review_document(view, name, require_valid_prefix=True, native_owner=d['deployment']['native_owner'])
    for key, value in inherited['checks'].items():
        a.check(value, 'KV41_EXISTING_' + key.upper(), name)
    a.check(g['selection'] == 'all_eligible' and g['planned_targets'] == g['planned_targets_hit'] == n
        and g['owner'] == d['deployment']['worker_rpc_owner'] and positive(g['generation'])
        and not g['failures'] and not g['rejection_reasons'] and not g['missing_outcomes']
        and not g['propagation_added_calls'] and g['baseline_control'] is False,
        'GROUP_SCOPE_AND_NO_OMITTED_FAILURES', name)
    recovery = g['recovery']
    targets = {digest(r['lease']['target']):r['lease']['target'] for r in recovery['targets']}
    leases = {digest(r['lease']['target']):r['lease'] for r in recovery['targets']}
    planned = sorted(targets)
    a.check(len(targets) == n and planned == g['discovery']['observed_target_ids'], 'FULL_TARGET_IDENTITIES', name)
    by_key = {digest(v['target_key']):v['target_id'] for v in g['per_target_lossless']}
    candidates = []
    for key in context['keys']:
        target = targets[by_key[digest(key)]]
        a.check(target['model_object_identity'] == context['model'] and target['model_instance_id'] == str(context['model'])
            and target['module_object_identity'] == key[1] and target['process_start_identity'] == context['process_start']
            and target['module_roles'] == ['attention'] and target['operator_api'] == 'torch_npu.npu_fused_infer_attention_score'
            and target['provider_identity'] == '41.tail_column:live-kv-group-v1'
            and target['tensor_signature'] == digest(dict(target_key=key, shape=key[6], dtype='torch.bfloat16'))
            and target['boundary']['call_token'] == 'live-kv-target:' + digest(key)
            and target['boundary']['implementation_sha256'] == g['scope']['implementation_sha256'], 'ACTUAL_TARGET_IDENTITY', name)
        a.check(dict(target['provider_proof']) == dict(native_library_sha256=d['deployment']['native_sha256'],
            writer_consumer_source_sha256=digest(d['worker_core']['final_cleanup']['provider']['source'])),
            'NATIVE_AND_WRITER_SOURCE_TARGET_BINDING', name)
        candidates.append(dict(candidate_id='live-kv:' + digest(key), eligibility=dict(status='SUPPORTED', reasons=[]), identity=target))
    candidate_report = dict(candidates=candidates, discovery_complete=True, eligible=n, discovered=n,
        eligibility_counts=dict(SUPPORTED=n, UNSUPPORTED=0, UNVERIFIED=0, AMBIGUOUS=0, NOT_OBSERVED=0, RESOURCE_LIMITED=0),
        execution_authorized=False)
    workers = sorted({tuple(worker_key(t)) for t in targets.values()})
    a.check(g['discovery']['candidates_sha256'] == digest(candidate_report)
        and g['discovery']['artifact_sha256'] == digest(g['independent_discovery'])
        and g['discovery']['expected_workers'] == g['discovery']['observed_workers'] == [list(w) for w in workers]
        and g['discovery']['workload_sha256'] == g['scope']['workload_sha256']
        and g['discovery']['complete'] is True and g['discovery']['source'] == 'independent_workload_observer',
        'RECOMPUTED_FULL_DISCOVERY', name)
    records = g['completed_requests']
    a.check([r['phase'] for r in records] == ['group_lossless', 'group_fault', 'group_fault', 'group_off'], 'EXACT_GROUP_PHASES', name)
    references = {r['target_id']:r for r in g['independent_discovery'][0]['byte_evidence']}
    all_records = g['independent_discovery'] + records
    for ri, record in enumerate(all_records):
        request_index = ri + 5 if ri < 2 else start + ri - 2
        run = d['requests'][request_index]['worker']
        a.check(record['inventory'] == run['independent_inventory'], 'ORIGINAL_REQUEST_INVENTORY_JOIN', name + '/' + str(ri))
        evidence = record['byte_evidence']
        a.check(len(evidence) == n and {r['target_id'] for r in evidence} == set(planned), 'COMPLETE_TARGET_BYTE_EVIDENCE', name + '/' + str(ri))
        for j, row in enumerate(evidence):
            loc = name + '/' + str(ri) + '/' + str(j)
            independent = record['inventory']['calls'][j]
            a.check(row['target_id'] == by_key[digest(row['target_key'])]
                and row['target_key'] == independent['target_key'] and row['call_id'] == independent['call_id']
                and row['forward_id'] == independent['forward_id'] and row['request_ids'] == independent['request_ids']
                and row['boundary'] == run['rows'][j]['boundary'] and row['original_calls'] == 1
                and row['original_returned'] is True and row['baseline_control'] is False
                and valid_sha(row['consumer_output_sha256']), 'ORIGINAL_FULL_BYTE_RECORD_JOIN', loc)
            a.check(helper_module.valid_prefix(row), 'EVERY_DISCOVERY_AND_EXECUTION_FULL_PREFIX', loc)
            if ri < 3 or ri == 5 or (name == 'c2' and ri == 4):
                a.check(row['input_before'] == references[row['target_id']]['input_before']
                    and row['consumer_output_sha256'] == references[row['target_id']]['consumer_output_sha256'],
                    'HEALTHY_LOSSLESS_OFF_PREFIX_AND_OUTPUT_EXACT', loc)
            has_native = ri == 2 or ri == 3 or (name == 'c3' and ri == 4)
            applied = ri == 3 or (name == 'c3' and ri == 4)
            a.check(('native' in row) is has_native and row['native_applied'] is applied, 'ACTUAL_NATIVE_PHASE_ADMISSION', loc)
            if has_native:
                native(a, row['native'], row['boundary'], d['deployment']['native_owner'], g['generation'], applied, loc)
                a.check(valid_sha(row['slot_during_sha256']) and
                    ((row['slot_during_sha256'] != row['input_before']['slot_sha256']) is applied), 'ACTUAL_SLOT_EFFECT_DIGEST', loc)
    a.check(g['per_target_lossless'] == records[0]['byte_evidence'], 'ORIGINAL_ALL_TARGET_LOSSLESS', name)
    expected_executions = records[1]['byte_evidence'] + (records[2]['byte_evidence'] if name == 'c3' else [])
    a.check(g['executions'] == expected_executions, 'ORIGINAL_NATIVE_EXECUTION_ROWS', name)
    tickets = []
    for sequence, call in enumerate(g['calls']):
        window, index = divmod(sequence, n)
        row = records[window + 1]['byte_evidence'][index]
        tid = row['target_id']; target = targets[tid]
        applied = window == 0 or (name == 'c3' and window == 1)
        phase = 'OFF_RECOVERY' if window == 2 else ('POST_FIRST_ANOMALY' if applied and index else 'PRE_FAULT_PROPAGATION')
        ticket = dict(observer_id=g['discovery']['observer_id'], sequence=sequence,
            call=dict(target=target, call_id=row['call_id'], forward_context=row['forward_id'], request_ids=row['request_ids'], phase=phase),
            eligible=True, propagation_added=False)
        decision = dict(session_id=g['session_id'], ticket=ticket, lease=leases[tid], admitted=applied,
            reason='ADMITTED' if applied else 'OFF_RECOVERY' if window == 2 else 'TARGET_ALREADY_ATTEMPTED')
        receipt = dict(decision=decision, outcome='HIT' if applied else 'PASSTHROUGH', execution_sha256=digest(row),
            effect_sha256=row.get('slot_during_sha256'), native_applied=applied, original_calls=1)
        expected = dict(call_id=row['call_id'], forward_id=row['forward_id'], request_ids=row['request_ids'],
            model_instance=target['model_instance_id'], operator=target['operator_api'], target_id=tid,
            rank=target['rank'], device=target['logical_device'], worker=worker_key(target), owner=g['owner'],
            generation=g['generation'], phase=phase, eligible=True, propagation_added=False, decision=decision, receipt=receipt)
        a.check(call == expected, 'RECOMPUTED_INDEPENDENT_TICKET_ADMISSION_EFFECT_RECEIPT', name + '/' + str(sequence))
        tickets.append(ticket)
    a.check(len(tickets) == 3*n and g['observation']['calls_sha256'] == digest(tickets)
        and g['observation']['artifact_sha256'] == digest(records) and g['observation']['complete'] is True
        and g['observation']['dropped_calls'] == 0 and g['observation']['observed_calls'] == 3*n,
        'RECOMPUTED_FULL_OBSERVATION', name)
    for row in recovery['targets']:
        lease = row['lease']
        a.check(row['drained'] is True and row['released'] is True and row['off_exact'] is True and row['pending_reason'] is None
            and lease['owner'] == g['owner'] and lease['generation'] == g['generation']
            and lease['device_bytes'] == 0 and lease['host_bytes'] > 0, 'ACTUAL_TARGET_DRAIN_RELEASE_OFF', name)
    for row in recovery['workers']:
        a.check(row['owner'] == g['owner'] and row['generation'] == g['generation']
            and row['worker'] in [list(w) for w in workers] and row['baseline_output_sha256'] == row['off_output_sha256'] == context['baseline'],
            'SAME_OWNER_WORKER_FULL_OUTPUT_RECOVERY', name)
    return dict(planned_targets=n, full_prefix_rows=sum(len(r['byte_evidence']) for r in all_records),
                recomputed_hits=len(expected_executions), actual_calls=len(tickets), generation=g['generation'],
                fault_full_root_changed=[r['model_output_sha256'] != context['baseline'] for r in records[1:3]])


def proof(a, d, host):
    c2, c3 = d['c2'], d['c3']
    issued, = c2['live_c2_proof']['issued']
    consumed = c3['live_c2_proof']['consumed']
    body = {k:c2[k] for k in COVERAGE_FIELDS if k != 'live_c2_proof'}
    expected = dict(scope=c2['scope'], candidates_sha256=c2['discovery']['candidates_sha256'],
        target_ids=sorted(c2['discovery']['observed_target_ids']), owner=c2['owner'], generation=c2['generation'],
        session_id=c2['session_id'], report_sha256=digest(body), discovery_sha256=c2['discovery']['artifact_sha256'],
        observation_sha256=c2['observation']['artifact_sha256'], recovery_sha256=digest(c2['recovery']))
    a.check(issued['proof'] == expected and issued['proof_sha256'] == digest(expected), 'ORIGINAL_C2_PROOF_ALL_HASHES')
    a.check(issued['exact_issued_object_verified'] is True and issued['object_binding']['pid'] == host['worker']['pid']
        and positive(issued['object_binding']['python_object_id']) and issued['serializable_authority'] is False
        and issued['qualification_granted'] is False and issued['report_hash_scope'] == 'coverage_report_without_live_c2_proof',
        'ORIGINAL_C2_LIVE_OBJECT_ISSUANCE_RECEIPT')
    expected_consumer = dict(session_id=c3['session_id'], scope=c3['scope'], owner=c3['owner'], generation=c3['generation'],
        candidates_sha256=c3['discovery']['candidates_sha256'], target_ids=sorted(c3['discovery']['observed_target_ids']))
    a.check(consumed == dict(issued, consumer=expected_consumer) and c3['generation'] > c2['generation']
        and c3['owner'] == c2['owner'] and c3['scope'] == c2['scope']
        and c3['discovery']['candidates_sha256'] == c2['discovery']['candidates_sha256']
        and c3['discovery']['observed_target_ids'] == c2['discovery']['observed_target_ids']
        and c2['live_c2_proof']['consumed'] is None and c3['live_c2_proof']['issued'] == [], 'C3_USED_SAME_ACTUAL_C2_OBJECT_AND_SCOPE')
    return dict(proof_sha256=issued['proof_sha256'], object_binding=issued['object_binding'],
                serialized_receipt_is_capability=False)


def zero(a, state, location):
    a.check(state['leases'] == state['inflight'] == state['owned_hooks'] == state['coverage_held_leases'] == 0
        and state['owned_api_state']['owned_links_present'] == 0 and state['owned_api_state']['read_only'] is True
        and state['native_lease'] is None and state['enabled'] is False and state['replay_pending'] is False
        and state['replay_failed'] is False and state['serving']['pending_request'] is None
        and state['serving']['error'] is None and not state['serving']['failures'], 'ACTUAL_PUBLIC_RESOURCES_ZERO', location)
    p = state['provider']
    a.check(p['installed'] is False and p['pending'] is None and p['owned_hooks'] == p['owned_api_patches'] == p['native_live_leases'] == 0,
        'ACTUAL_PROVIDER_RESOURCES_ZERO', location)


def rpc(a, d, host, receipts, natural):
    a.check([r['action'] for r in receipts] == ACTIONS and receipts == host['history'], 'ORIGINAL_43_RPC_HISTORY')
    ticket = receipts[1]['response']['ticket']
    attachment = {k:ticket[k] for k in ('attach_nonce', 'tenant_id', 'worker')}
    a.check(host['config'] == d['worker_core']['config'] == dict(fault_id=41, variant='tail_column', implementation_version='1')
        and host['mechanism'] == '41.tail_column' and host['capability_enumeration']['worker'] == host['worker'],
        'ACTUAL_PUBLIC_CONFIGURATION_AND_ROUTE')
    a.check(receipts[0]['response'] == dict(status='ATTACHED', ticket=attachment)
        and ticket['owner'] == host['owner'] and ticket['tenant_id'] == host['tenant_id']
        and ticket['worker'] == host['worker'] and positive(ticket['generation'])
        and ticket['mechanism'] == '41.tail_column' and ticket['implementation_version'] == '1', 'ACTUAL_ATTACHMENT_TICKET_OWNER')
    internal_owner = d['deployment']['worker_rpc_owner']
    a.check(d['deployment']['worker_rpc_generation'] == ticket['generation'] and isinstance(internal_owner, str) and internal_owner
        and positive(d['deployment']['native_owner']), 'DISTINCT_RUNTIME_AND_NATIVE_OWNER_JOIN')
    for index, record in enumerate(receipts[1:-1], 1):
        reply, action = record['response'], record['action']
        state = reply['provider_state']; location = str(index) + ':' + action
        a.check(reply['ticket'] == ticket and reply['status'] == ('QUERY_OK' if action == 'query' else 'ACTION_COMPLETED')
            and reply['recovery_required'] is False and (action == 'query' or reply['action'] == action), 'PUBLIC_RPC_SCOPE_AND_SUCCESS', location)
        a.check(reply['native'] == state and (action == 'query' or reply['provider'] == state)
            and state['owner'] == internal_owner and state['generation'] == ticket['generation']
            and state['execution_boundary'] == 'native' and state['state_source'] == 'NATIVE_ABI_OWNER_LEASE_AND_OWNED_HOST_READBACK'
            and state['native_loaded'] is True and state['native_registered'] is False
            and state['native_registration'] == 'NOT_APPLICABLE_DIRECT_ABI_LEASES'
            and state['mechanism'] == '41.tail_column' and state['native_out_of_bounds_claim'] is False
            and state['native']['sha256'] == d['deployment']['native_sha256'] and state['native']['abi'] == 1,
            'RAW_RUNTIME_PROVIDER_NATIVE_READBACK', location)
        if natural:
            a.check(state['native']['simulation'] is False and state['native']['loaded_mapping_verified'] is True
                and state['native']['read_span_available'] is True, 'ACTUAL_NATIVE_MAPPING_AND_READ_SPAN', location)
        published = state['coverage']
        for label, first in (('deployment', 4), ('C1', 15), ('C2', 29), ('C3', 39)):
            a.check((label in published) is (index >= first), 'COVERAGE_PUBLISHED_AFTER_REAL_COMPLETION', location + '/' + label)
            if index >= first:
                a.check(published[label] == d[label.lower()], 'ORIGINAL_PUBLISHED_REPORT_UNCHANGED', location + '/' + label)
        ledger = (state['group'] or {}).get('live_c2_proof', {})
        a.check(ledger.get('issued', []) == (d['c2']['live_c2_proof']['issued'] if index in (29, 30) else [])
            and ledger.get('consumed') == (d['c3']['live_c2_proof']['consumed'] if index >= 31 else None),
            'LIVE_PROOF_ISSUANCE_USE_TEMPORAL_ORDER', location)
    final_actual = d['worker_core']['serving']['actual_requests']
    for i, row in enumerate(d['requests']):
        states = [receipts[j]['response']['provider_state'] for j in (5 + 2*i, 6 + 2*i)]
        for state in states:
            a.check(state['serving']['pending_request'] == [row['mapping']['internal_request_id'], PHASES[i], INNER[i]]
                and state['serving']['actual_requests'] == final_actual[:i], 'NO_FABRICATED_OR_REPLAYED_REQUEST_COMPLETION', PHASES[i])
        selected = states[1]['group']['health'] if INNER[i] == 'health' else states[1]['provider']['pending']
        a.check(selected == row['worker'], 'RAW_QUERY_ACTUAL_RETURNED_REQUEST', PHASES[i])
    for index in (39, 40, 41):
        state = receipts[index]['response']['provider_state']
        zero(a, state, ACTIONS[index])
        a.check(state['serving']['actual_requests'] == final_actual and state['serving']['complete'] is True
            and state['registered'] is (index != 41), 'COMPLETION_BEFORE_UNREGISTER', ACTIONS[index])
    a.check(d['worker_core']['final_cleanup'] == receipts[39]['response']['provider_state'], 'ORIGINAL_FINISH_CLEANUP_READBACK')
    a.check(receipts[-1]['response'] == dict(status='DETACHED', recoveries=[]) and host['status'] == 'DETACHED'
        and host['attachment'] is host['ticket'] is host['census_ticket'] is None and host['ownership_retained'] is False
        and d['worker_core']['ownership_retained'] is False, 'PUBLIC_OWNER_UNREGISTER_AND_DETACH')
    a.check(d['worker_core']['last_scope'] == {k:v for k,v in host.items() if k not in ('history', 'response', 'capability_enumeration')},
        'FINAL_HOST_SCOPE_UNCHANGED')
    health = d['post_unregister_health']
    a.check(health['worker'] == host['worker'] and health['after_public_unregister'] is True
        and all(health['output'][k] == d['healthy_before'][k] for k in ('tokens', 'text'))
        and health['output']['request_id'] == health['mapping']['external_request_id']
        and d['post_unregister_health_attempts'] == [health]
        and health['mapping']['internal_request_id'] not in {r['mapping']['internal_request_id'] for r in d['requests']}
        and health['mapping']['source'] == 'ACTUAL_PUBLIC_ENQUEUE_AND_OUTPUT_PROCESSOR'
        and positive(health['mapping']['request_state_object_id']), 'EIGHTEENTH_ORDINARY_SAME_WORKER_REQUEST')
    if natural:
        a.check(d['binding_removal'] == dict(status='SERVING_DEPLOYMENT_REMOVED'), 'TRUSTED_BINDING_REMOVED_AFTER_OWNER_RELEASE')
    return dict(rpc_actions=len(receipts), completed_protocol_requests=len(final_actual), ordinary_post_unregister_requests=1,
                host_owner=ticket['owner'], runtime_owner=internal_owner, native_owner=d['deployment']['native_owner'])


def review_evidence(d, host, receipts, *, natural=True):
    a = Audit(); details, scopes = {}, {}
    def section(name, function, *args):
        checks, failures, issue_count = a.checks, a.failures, len(a.issues)
        result = a.section(name, function, *args)
        scopes[name] = dict(consistent=a.failures == failures, checks=a.checks-checks,
            failures=a.failures-failures, issues=a.issues[issue_count:])
        return result
    old = helper()
    details['rpc'] = section('public_rpc', rpc, a, d, host, receipts, natural)
    context = section('requests', requests, a, d, host, natural)
    if context:
        for name in ('c2', 'c3'):
            details[name] = section(name + '_execution_and_recovery', group, a, d, name, context, old)
    details['live_c2_proof'] = section('live_c2_proof', proof, a, d, host)
    return dict(**a.report(), details=details, scopes=scopes, public_qualification_granted=False,
        evidence_kind='natural_worker_source_bound_receipts' if natural else 'CPU_CONTRACT_ONLY',
        limits=[
            'Complete logical K/V prefix and full root-output SHA receipts are reconciled; tensor byte arrays are not saved here and are not independently rehashed.',
            'The denominator is every observed writer/consumer in the frozen source-checked structural attention universe; unused cache storage is outside the valid-prefix proof.',
            'Object identity, native synchronization and original-call counts remain source-bound runtime receipts, not new live observations.',
            'Post-unregister ordinary request preserves same Worker identity, tokens and text; its full root tensor is not captured after observer removal.',
            'Native KV tail overwrite is a controlled legal-address proxy; no out-of-bounds, E5, other mechanism or portable model/device qualification is granted.'])


def load_artifacts(folder, d):
    a = Audit()
    item = d['worker_core']['final_host_receipt']; path = under(folder, item['path'])
    a.check(item['path'] == 'host_receipt.json' and file_sha(path) == item['sha256'], 'ORIGINAL_HOST_FILE_SHA')
    host = read(path)
    receipts = host['history']
    a.check(len(receipts) == len(d['worker_core']['receipts']), 'ORIGINAL_RPC_FILE_COUNT')
    for index, item in enumerate(d['worker_core']['receipts']):
        # Path.relative_to is serialized with the producing host's separator.
        # Normalize only for resolution; under still rejects traversal/drives.
        relative = item['path'].replace('\\', '/')
        path = under(folder, relative)
        a.check(relative == 'worker_core_rpc/' + str(index).zfill(3) + '-' + item['action'] + '.json'
            and file_sha(path) == item['sha256'], 'ORIGINAL_RPC_FILE_SHA', index)
        # Retain only the host ledger; avoid a second multi-gigabyte copy.
        a.check(index < len(receipts) and read(path) == receipts[index], 'ORIGINAL_HOST_AND_RPC_FILES_IDENTICAL', index)
    return host, receipts, a.report()


def provenance(d, host, source_root, source_sha, source_commit, native_path):
    a = Audit(); manifest_path = under(source_root, 'source_manifest.json'); manifest = read(manifest_path)
    a.check(file_sha(manifest_path) == source_sha == SOURCE_MANIFEST and manifest['base_commit'] == source_commit == SOURCE_COMMIT,
        'FROZEN_SOURCE_MANIFEST')
    entries = {r['path']:r for r in manifest['files']}
    hashes = d['source_sha256']
    a.check(len(entries) == len(manifest['files']) and REQUIRED_SOURCES.issubset(hashes), 'COMPLETE_REVIEWED_SOURCE_SET')
    for name, sha in hashes.items():
        a.check(file_sha(under(source_root, name)) == sha == entries[name]['sha256'], 'SOURCE_FILE_REHASH', name)
    a.check(d['probe_sha256'] == hashes['experiments/c2c3_postwrite_20261117/natural41_worker_core_probe.py']
        and file_sha(native_path) == d['native_sha256'] == d['deployment']['native_sha256'] == NATIVE_SHA,
        'ACTUAL_PROBE_AND_NATIVE_LIBRARY_REHASH')
    a.check(d['schema'] == 'demoncat.kv41.worker_core_probe.v1' and d['mechanism'] == '41.tail_column'
        and d['standard_vllm_worker'] is True and d['native_out_of_bounds_claim'] is False
        and d['public_entrypoint'] == 'WorkerFaultClient+FaultConfig(41,tail_column)'
        and d['deployment']['simulation'] is False, 'STANDARD_PUBLIC_CORE41_SOURCE_SCOPE')
    setup, worker = d['setup'], host['worker']
    a.check(setup['worker'] == worker == setup['environment']['worker'] and worker['rank'] == 0
        and worker['logical_device'] == worker['configured_logical_device'] == worker['observed_logical_device']
        and positive(worker['pid']) and worker['vllm_version'] and worker['vllm_ascend_version']
        and worker['model_instance_id'] and worker['worker_start_identity'] and worker['process_start_identity'], 'ACTUAL_STANDARD_WORKER_IDENTITY')
    binding = setup['binding']
    a.check(binding['status'] == 'SERVING_DEPLOYMENT_BOUND_NOT_QUALIFIED' and binding['simulation'] is False
        and binding['workers_created'] == binding['hooks_installed'] == binding['native_leases_opened'] == 0
        and binding['native'] == host['history'][1]['response']['provider_state']['native'], 'TRUSTED_INERT_DEPLOYMENT_AND_LOADED_ABI')
    sources = d['deployment']['binding_sources']
    a.check(sources == {name:hashes[CORE_PREFIX + ('providers/' if name == 'fault41.py' else '') + name]
        for name in ('fault41.py', 'serving41.py')}, 'ACTUAL_RUNTIME_BINDING_SOURCE_HASHES')
    prefix = d['requests'][0]['worker']['actual_scheduler']['prefixes'][0]['prompt']
    expected_scope = dict(mechanism='41.tail_column', workload_sha256=digest(dict(prompt=prefix, max_tokens=2, requests=2)),
        environment_sha256=digest(setup['environment']), implementation_sha256=digest(dict(
            reviewed_implementation=digest(hashes), worker_rpc_binding=sources)), evidence_kind='natural_worker',
        model_weights_sha256=d['weights']['weights_sha256'], model_source_sha256=setup['model_source']['source_sha256'])
    a.check(d['c2']['scope'] == d['c3']['scope'] == expected_scope and valid_sha(expected_scope['model_weights_sha256'])
        and valid_sha(expected_scope['model_source_sha256']), 'EXACT_SOURCE_MODEL_WORKLOAD_ENVIRONMENT_SCOPE')
    a.check(digest(d['weights']['manifest']) == d['weights']['weights_sha256']
        and d['weights']['manifest']['format'] == 'safetensors'
        and d['weights']['manifest']['loader_selection'] == "vllm.LLM(load_format='safetensors', trust_remote_code=False)"
        and d['weights']['manifest']['files']
        and all(valid_sha(row['sha256']) and positive(row['bytes']) for row in d['weights']['manifest']['files'].values())
        and digest(setup['model_source']['manifest']) == setup['model_source']['source_sha256'], 'SAVED_COMPLETE_MODEL_MANIFEST_HASHES')
    psource = d['worker_core']['final_cleanup']['provider']['source']
    a.check({k:v['sha256'] for k,v in psource.items()} == dict(
        backend='bbe69c3015ee15789dea0d6ec9795243458b68494294c4c023ed5a5d665c550b',
        layer='990ead07c828e26192ca46b4b8abae7a6792f67c27510705ad7b483035f3631c'), 'FROZEN_ACTUAL_ATTENTION_SOURCE_RECEIPTS')
    for group_name in ('c2', 'c3'):
        g = d[group_name]
        a.check(g['discovery']['implementation_sha256'] == hashes['fault_injection/deployments/postwrite_kv_inventory.py'],
            'INDEPENDENT_OBSERVER_SOURCE_HASH', group_name)
        for row in g['recovery']['targets']:
            target = row['lease']['target']
            a.check(target['scope_kind'] == 'INITIALIZED_WORKER' and target['device_type'] == 'npu'
                and target['rank'] == worker['rank'] and target['process_id'] == worker['pid']
                and target['logical_device'] == worker['logical_device'], 'ACTUAL_NATIVE_TARGET_WORKER_SCOPE', group_name)
    return dict(**a.report(), source_files_rehashed=len(hashes), native_library_rehashed=True,
        external_model_weights_and_method_sources_rehashed=False, source_manifest_sha256=file_sha(manifest_path))


def deny_runtime(event, args):
    if event in ('subprocess.Popen', 'os.system', 'ctypes.dlopen', 'socket.connect') or event.startswith('torch.'):
        raise RuntimeError('OFFLINE_REVIEW_FORBIDS_RUNTIME_EXECUTION:' + event)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--result-sha256', required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--source-manifest-sha256', default=SOURCE_MANIFEST)
    parser.add_argument('--source-commit', default=SOURCE_COMMIT)
    parser.add_argument('--native-library', type=Path, required=True)
    args = parser.parse_args(argv)
    sys.addaudithook(deny_runtime)
    report = dict(schema='demoncat.kv41.public_core.independent_review.v1', public_qualification_granted=False,
        E5='NOT_ASSESSED_SEPARATE_RAW_BF_CAPTURE_REQUIRED', other_mechanisms_qualified=False)
    try:
        actual_sha = file_sha(args.result)
        if not valid_sha(args.result_sha256) or actual_sha != args.result_sha256:
            raise ValueError('ORIGINAL_RESULT_SHA_MISMATCH')
        d = read(args.result)
        host, receipts, report['artifacts'] = load_artifacts(args.result.parent, d)
        report.update(result=dict(path=str(args.result), sha256=actual_sha, bytes=args.result.stat().st_size),
            provenance=provenance(d, host, args.source_root, args.source_manifest_sha256, args.source_commit, args.native_library),
            evidence=review_evidence(d, host, receipts))
        good = all(report[k]['consistent'] for k in ('artifacts', 'provenance', 'evidence'))
        report.update(source_bound_core41_c2_c3_receipts_reconciled=good,
            status='SOURCE_BOUND_CORE41_C2_C3_RECONCILED_NOT_E5_OR_GENERAL_QUALIFICATION' if good else 'EVIDENCE_GAP_OR_CONTRADICTION')
    except (OSError, ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
        report.update(status='EVIDENCE_GAP_OR_CONTRADICTION', source_bound_core41_c2_c3_receipts_reconciled=False,
            error=type(error).__name__ + ':' + str(error))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report.get('source_bound_core41_c2_c3_receipts_reconciled') else 2


if __name__ == '__main__':
    raise SystemExit(main())
