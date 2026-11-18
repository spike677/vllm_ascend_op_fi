"""Read-only stdlib review of the new public Native26 single-site protocol.

Saved receipts are evidence, not live capabilities. This does not call a model,
load a Native library, or inherit the previous direct-experiment qualification.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys


HELPERS = {
    'review_native26_single_source54.py': '8151f93dca35ead3b32b98d66881e5bc23432884bf5af8c1c6694f1b59fb8b5d',
    'review_native26_fia_single_source62.py': 'e6f9e2885ed2b707e45d492d29ec4b56db0f3b267995eccdd267c1a8c6934393',
}
PHASES = ('inventory_1', 'inventory_2', 'single_site_reference', 'single_site_lossless',
          'single_site_qualification_off', 'single_site_fault', 'single_site_recovery_off', 'health')
MODES = (None, None, 'reference', 'lossless', 'off', 'fault', 'off', None)
ACTIONS = ['attach', 'prepare', 'register', 'check', 'enable'] + ['advance', 'query', 'advance'] * 8 + [
    'finish', 'disable', 'unregister', 'detach']
SUCCESS = 'STANDARD_WORKER_NATIVE26_SINGLE_SITE_C1_EVIDENCE_READY_FOR_REVIEW_NOT_C2'
SCOPES = {
    '26.residual_stomp': 'ONE_ACTUAL_QMM_SITE_FIXED_FRESH_PREFILL_NOT_ALL_ELIGIBLE_NOT_C2_C3',
    '26.fia_destination': 'ONE_ACTUAL_TYPED_FIA_SITE_FIXED_FRESH_PREFILL_NOT_ALL_ELIGIBLE_NOT_C2_C3',
}
PREFIX = 'src/demoncat/runtime/vllm_ascend/worker_extension/'
LOADED = {'fault_injection/deployments/' + n + '.py' for n in (
    'writer26_loader', 'writer26_group', 'writer26_inventory', 'writer26_experiment',
    'writer26_public', 'writer26_fia_typed', 'residual26')} | {
    'fault_injection/metadata_arrays.py', 'fault_injection/providers.py'} | {
    PREFIX + n.replace('.', '/') + '.py' for n in ('serving26_core', 'providers.fault26_core',
    'providers.fault26', 'providers._scope', 'serving26_group', 'serving26_inventory', 'serving',
    'serving_context', 'call_binding', 'host', 'config', 'runtime', 'identity', 'extension', 'registry',
    'providers.__init__')}
REQUIRED_SOURCES = LOADED | {
    'experiments/c2c3_closure_20261117/providers26_worker_core_probe.py',
    'experiments/c2c3_closure_20261117/providers26_prepare_group_cases.py',
    'experiments/c2c3_closure_20261117/providers26_group_probe.py',
    'experiments/c2c3_postwrite_20261117/model_identity.py',
}


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('DUPLICATE_JSON_KEY:' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('NONFINITE_JSON:' + value)
    def finite(value):
        result = float(value)
        return result if math.isfinite(result) else invalid(value)
    with Path(path).open('r', encoding='utf-8') as stream:
        return json.load(stream, object_pairs_hook=pairs, parse_constant=invalid, parse_float=finite)


def under(root, name):
    root, name = Path(root).resolve(strict=True), str(name)
    if (not name or '\\' in name or ':' in name or PurePosixPath(name).is_absolute()
            or any(p in ('', '.', '..') for p in name.split('/'))):
        raise ValueError('UNSAFE_ARTIFACT_PATH')
    result = (root / name).resolve(strict=True)
    if not result.is_relative_to(root) or not result.is_file():
        raise ValueError('ARTIFACT_OUTSIDE_ROOT')
    return result


def helper(name):
    path = Path(__file__).with_name(name)
    if file_sha(path) != HELPERS[name]:
        raise ValueError('REVIEWED_ATOMIC_HELPER_CHANGED:' + name)
    spec = importlib.util.spec_from_file_location('_core26_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Originals:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.checked = {}

    def load(self, ref, *, folder='', absolute=False):
        name = ref['path']
        if absolute:
            if not isinstance(name, str) or not name.startswith('/evidence/'):
                raise ValueError('ACTUAL_EVIDENCE_MOUNT_REQUIRED')
            name = name[len('/evidence/'):]
        elif folder:
            name = folder + '/' + name
        path = under(self.root, name)
        if not re.fullmatch('[0-9a-f]{64}', ref['sha256']) or file_sha(path) != ref['sha256']:
            raise ValueError('ORIGINAL_ARTIFACT_SHA_CHANGED:' + name)
        if 'bytes' in ref and (type(ref['bytes']) is not int or path.stat().st_size != ref['bytes']):
            raise ValueError('ORIGINAL_ARTIFACT_SIZE_CHANGED:' + name)
        if name in self.checked and self.checked[name] != ref['sha256']:
            raise ValueError('ORIGINAL_PATH_REUSED_WITH_CHANGED_BYTES')
        self.checked[name] = ref['sha256']
        return read(path)


def project_inventory(inventory):
    """Rebuild the provider's compact view from saved complete originals."""
    omitted = {'rows', 'candidates', 'outcomes', 'public_output_observer',
               'metadata_arrays_initial', 'metadata_arrays_latest'}
    result = {k:v for k, v in inventory.items() if k not in omitted}
    result['request_index'] = [dict(request_id=r['request_id'], phase=r['phase'],
        native_count=len(r['candidates']), output_sha256=r['output_sha256'], admissions=r['admissions'])
        for r in inventory['rows']]
    result['candidate_index'] = [dict(index=r['index'], target_key=r['target_key'], status=r['status'],
        reasons=r['reasons'], module_object_identity=(r['callback'] or {}).get('module_object_identity'),
        module_path=(r['callback'] or {}).get('module_path')) for r in inventory['candidates']]
    return result


def original(r, originals, ref, state, worker, where):
    value = originals.load(ref, absolute=True)
    r.check(value['schema'] == 'demoncat.writer26.core_observations.v1'
        and value['action'] == ref['action'] and value['worker'] == worker
        and value['simulation'] is False and value['qualification_granted'] is False
        and value['native_owner'] == state['owner'] and value['generation'] == state['generation'],
        'ACTUAL_CORE_ORIGINAL_SCOPE', where)
    inventory = value['inventory']
    r.check(ref['complete_requests'] == len(inventory['rows']) == len(value['serving_requests']),
            'ORIGINAL_COMPLETE_REQUEST_COUNT', where)
    compact = {k:v for k,v in state['evidence'].items() if k not in ('public_core', 'original_observations')}
    r.check(compact == project_inventory(inventory) and state['native'] == inventory['native_state']
        and value['serving'] == state['serving'], 'COMPLETE_ORIGINAL_TO_PUBLIC_READBACK_JOIN', where)
    return value


def protocol(b, r, d, originals, *, mechanism):
    host = originals.load(d['host_receipt'], folder='probe')
    r.check(d['host_receipt']['path'] == 'host_receipt.json'
        and host['schema'] == 'demoncat.worker_core.host_receipt.v1', 'ORIGINAL_HOST_RECEIPT')
    raw, linked, routes, setups, removals, capabilities = [], [], [], [], [], []
    for index, ref in enumerate(d['rpc_receipts']):
        r.check(ref['path'] == str(index).zfill(3)+'.json', 'COMPLETE_ORDERED_RPC_FILES', str(index))
        item = originals.load(ref, folder='probe/worker_core_rpc')
        r.check(item['method'] == ref['method'] and 'error' not in item and len(item['responses']) == 1,
                'ONE_ACTUAL_RPC_RESPONSE', str(index))
        reply, method, args = item['responses'][0], item['method'], item['args']
        if method in ('demoncat_attach', 'demoncat_detach', 'demoncat_fault_query', 'demoncat_fault_rpc'):
            action = {'demoncat_attach':'attach', 'demoncat_detach':'detach', 'demoncat_fault_query':'query'}.get(method)
            if method == 'demoncat_fault_rpc':
                action = args[0]
            linked.append(dict(action=action, response=reply))
            raw.append((item, ref))
        elif method == 'demoncat_worker_identity':
            routes.append((len(linked), reply))
        elif method == 'demoncat_capabilities':
            capabilities.append(reply)
        elif method in ('__main__.setup', 'experiments.c2c3_closure_20261117.providers26_worker_core_probe.setup'):
            setups.append((len(linked), item))
        elif method in ('__main__.remove', 'experiments.c2c3_closure_20261117.providers26_worker_core_probe.remove'):
            removals.append((len(linked), item))
        else:
            r.check(False, 'UNREVIEWED_RPC_EXECUTION', method)
    r.check([row['action'] for row in linked] == ACTIONS and host['history'] == linked,
            'PUBLIC_HOST_EXACT_PROTOCOL_AND_ORIGINAL_RPC_JOIN')
    ticket = linked[1]['response']['ticket']
    worker = host['worker']
    attach = {k:ticket[k] for k in ('attach_nonce', 'tenant_id', 'worker')}
    r.check(host['config'] == dict(fault_id=26, variant=mechanism.split('.', 1)[1], implementation_version='1')
        and host['mechanism'] == mechanism and ticket['mechanism'] == mechanism
        and ticket['implementation_version'] == '1' and b.positive(ticket['generation'])
        and ticket['owner'] == host['owner'] and ticket['tenant_id'] == host['tenant_id']
        and ticket['worker'] == worker == d['worker'] and linked[0]['response'] == dict(status='ATTACHED', ticket=attach),
        'ACTUAL_PUBLIC_CONFIG_WORKER_AND_OWNER')
    r.check(len(routes) == 2 and [x[0] for x in routes] == [0, len(ACTIONS)-1]
        and all(v['status'] == 'WORKER_IDENTIFIED' and v['worker'] == worker and v['quarantined'] is False for _,v in routes),
        'SAME_ACTUAL_WORKER_BEFORE_AND_AFTER_RELEASE')
    r.check(capabilities == [host['capability_enumeration']] and capabilities[0]['worker'] == worker
        and any(v['mechanism'] == mechanism and v['implementation_version'] == '1'
                for v in capabilities[0]['providers']), 'ACTUAL_PROVIDER_ENUMERATION')
    r.check(len(setups) == 1 and setups[0][0] == 1 and setups[0][1]['responses'] == [d['setup']]
        and setups[0][1]['args'] == [worker, mechanism, d['actual_prompt_token_ids'], d['manifest_sha256'], d['sources']],
        'TRUSTED_DEPLOYMENT_BEFORE_PREPARE')
    bound = d['setup']['binding']
    r.check(d['setup']['worker'] == worker and bound == dict(status='DEPLOYMENT_BOUND_NOT_ACCEPTED',
        mechanism=mechanism, stage='c1_single_effects', hooks_installed=0, native_owner_installed=False,
        workers_created=0, qualification_granted=False, c2=False, c3=False), 'INERT_BOUND_DEPLOYMENT')
    owner = linked[1]['response']['provider_state']['owner']
    for index, record in enumerate(linked[1:-1], 1):
        action, reply = record['action'], record['response']
        state = reply['provider_state']
        item = raw[index][0]
        payload = item['args'][-1]
        expected_payload = dict(ticket)
        if action == 'prepare':
            expected_payload.pop('generation')
        if 5 <= index < 29 and (index-5) % 3 != 1:
            ordinal = (index-5)//3
            expected_payload['settings'] = ({'request_id':d['requests'][ordinal]['mapping']['internal_request_id']}
                if (index-5) % 3 == 0 else {'collect':True})
        r.check(payload == expected_payload and reply['ticket'] == ticket and reply['recovery_required'] is False
            and reply['status'] == ('QUERY_OK' if action == 'query' else 'ACTION_COMPLETED')
            and (action == 'query' or reply['action'] == action), 'EXACT_PUBLIC_ACTION_AND_SCOPE', str(index))
        r.check(reply['native'] == state and (action == 'query' or reply['provider'] == state)
            and state['execution_boundary'] == 'native' and state['native_loaded'] is True
            and state['owner'] == owner and state['generation'] == ticket['generation']
            and state['query_read_only'] is True and state['enabled'] is False
            and type(state['inflight']) is type(state['leases']) is int
            and state['inflight'] == state['leases'] == 0 and state['replay_failed'] is False
            and state['native_registered'] is bool(state['native']['core']['installed'])
            and state['registered'] is (2 <= index <= 30) and state['binding_reserved'] is (2 <= index <= 30)
            and state['replay_pending'] is (4 <= index <= 28)
            and state['recovery_validation_pending'] is (4 <= index <= 28),
            'CURRENT_NATIVE_OWNER_AND_QUIESCENT_ACTION_READBACK', str(index))
        if state['evidence'] is not None:
            core = state['evidence']['public_core']
            r.check(core == dict(worker_identity=worker, owner=owner, generation=ticket['generation'],
                native_owner=b.words(hashlib.sha256(owner.encode()).hexdigest())[:2],
                native_generation=ticket['generation'], simulation=False, qualification_granted=False,
                c2=False, c3=False, owner_kind='runtime_lease_native_namespace', query_read_only=True,
                legacy_qualification_inherited=False), 'RUNTIME_PRIVATE_OWNER_AND_NO_INHERITED_AUTHORITY', str(index))
    r.check(raw[0][0]['args'] == [dict(tenant_id=host['tenant_id'], worker=worker)]
        and raw[-1][0]['args'] == [attach], 'EXACT_ATTACH_DETACH_PAYLOADS')
    r.check(host['status'] == 'DETACHED' and host['ownership_retained'] is d['ownership_retained'] is False
        and host['attachment'] is host['ticket'] is host['census_ticket'] is None
        and linked[-1]['response'] == host['response'] == dict(status='DETACHED', recoveries=[]), 'PUBLIC_OWNER_RELEASED')
    r.check(len(removals) == 1 and removals[0][0] == len(ACTIONS)
        and removals[0][1]['args'] == [worker, mechanism]
        and removals[0][1]['responses'] == [d['deployment_removed']]
        and d['deployment_removed'] == dict(status='WRITER26_CORE_DEPLOYMENT_REMOVED'), 'INERT_BINDING_REMOVED_LAST')
    seen_refs, collected = [], []
    for index, request in enumerate(d['requests']):
        armed, queried, saved = [linked[j]['response']['provider_state'] for j in range(5+index*3, 8+index*3)]
        pre = queried['evidence']['original_observations']
        evidence = saved['evidence']['original_observations']
        r.check(request['pre_collect_query'] == raw[6+index*3][1]
            and pre['originals'] == seen_refs and queried['serving']['needs_collect'] is True
            and evidence['status'] == 'CURRENT_COLLECTED' and saved['serving']['needs_collect'] is False
            and evidence['originals'][:-1] == seen_refs and len(evidence['originals']) == index+1
            and request['original_observations'] == evidence['originals'][-1],
            'READ_ONLY_QUERY_THEN_ONE_EXPLICIT_COLLECTION', PHASES[index])
        ref = request['original_observations']
        r.check(ref['action'] == 'collect', 'COLLECT_ACTION_ORIGINAL', PHASES[index])
        value = original(r, originals, ref, saved, worker, PHASES[index])
        r.check(len(armed['evidence']['request_index']) == index
            and armed['serving']['expected'] == request['mapping']['internal_request_id']
            and not armed['serving']['needs_collect']
            and len(queried['evidence']['request_index']) == index+1
            and queried['evidence']['request_index'] == saved['evidence']['request_index'],
            'DEFERRED_ENQUEUE_NOT_RPC_REPLAY', PHASES[index])
        if collected:
            r.check(value['inventory']['rows'][:-1] == collected[-1]['inventory']['rows']
                and value['serving_requests'][:-1] == collected[-1]['serving_requests'], 'IMMUTABLE_PRIOR_REQUEST_PREFIX', PHASES[index])
        seen_refs = evidence['originals']
        collected.append(value)
    final_state = linked[29]['response']['provider_state']
    final_refs = final_state['evidence']['original_observations']
    r.check(final_refs['status'] == 'CURRENT_COLLECTED' and final_refs['originals'][:-1] == seen_refs
        and len(final_refs['originals']) == 9 and final_refs['originals'][-1]['action'] == 'cleanup',
        'FINISH_SAVES_FULL_CLEANUP_ORIGINAL')
    final = original(r, originals, final_refs['originals'][-1], final_state, worker, 'cleanup')
    r.check(final['inventory']['rows'] == collected[-1]['inventory']['rows']
        and final['serving_requests'] == collected[-1]['serving_requests']
        and d['c1'] == final_state['evidence'], 'FINISH_TO_DRIVER_ORIGINAL_JOIN')
    for index in (29, 30, 31):
        state = linked[index]['response']['provider_state']
        s, inv, native = state['serving'], state['evidence'], state['native']
        r.check(not any(native['core'][k] for k in ('installed', 'pending', 'executing', 'device_bytes', 'host_bytes', 'descriptor_pending', 'error'))
            and not any(native[k] for k in ('callback_active', 'completion_active', 'descriptor', 'prepared_fia', 'foreign_calls', 'rejected_calls'))
            and state['native_registered'] is False and not state['replay_pending'] and not state['recovery_validation_pending']
            and not s['installed'] and s['owned_hooks'] == inv['owned_hooks'] == inv['owned_public_api_wrappers'] == 0
            and not s['needs_collect'] and s['expected'] is s['current'] is None
            and not s['root_running'] and not s['logits_running']
            and s['complete_requests'] == 8 and state['registered'] is (index != 31)
            and state['binding_reserved'] is (index != 31), 'COMPLETE_OWNED_CLEANUP_BEFORE_RELEASE', ACTIONS[index])
        r.check(state['evidence'] == d['c1'] and s == final['serving'], 'FINAL_CLEANUP_STATE_PRESERVED', ACTIONS[index])
    return dict(host=host, linked=linked, raw=raw, final=final, collected=collected, owner=owner,
                generation=ticket['generation'], originals_rehashed=len(originals.checked))


def requests(b, r, d, context):
    inv = context['final']['inventory']
    rows, observed = inv['rows'], context['final']['serving_requests']
    r.check(len(rows) == len(observed) == len(d['requests']) == 8, 'EXACT_EIGHT_ACTUAL_REQUESTS')
    tokens, baseline = d['actual_prompt_token_ids'], rows[0]['output_sha256']
    r.check(len(tokens) > 1 and all(type(v) is int and v >= 0 for v in tokens), 'ACTUAL_PROMPT_TOKENS')
    internal, external, shapes = set(), set(), []
    for index, (row, obs, driver) in enumerate(zip(rows, observed, d['requests'])):
        name, mapping, snapshot = PHASES[index], driver['mapping'], obs['snapshot']
        rid = mapping['internal_request_id']
        r.check(driver['phase'] == row['phase'] == obs['phase'] == name
            and rid == row['request_id'] == obs['actual_internal_request_id'] == snapshot['internal_request_id']
            and isinstance(rid, str) and rid and rid not in internal
            and isinstance(mapping['external_request_id'], str) and mapping['external_request_id']
            and mapping['external_request_id'] not in external and b.positive(mapping['request_state_object_id'])
            and mapping['source'] == 'ACTUAL_PUBLIC_ENQUEUE_AND_OUTPUT_PROCESSOR'
            and mapping['proof_scope'] == 'DRIVER_MAPPING_ONLY_REQUIRES_REAL_MODEL_PREHOOK_CORROBORATION',
            'ACTUAL_DRIVER_TO_MODEL_REQUEST', name)
        internal.add(rid); external.add(mapping['external_request_id'])
        r.check(b.positive(snapshot['request_object_id']) and snapshot['num_requests'] == 1
            and snapshot['num_computed_tokens'] == 0
            and snapshot['num_prompt_tokens'] == snapshot['num_scheduled_tokens'] == len(tokens)
            and snapshot['token_ids'] == tokens and snapshot['positions'] == list(range(len(tokens)))
            and snapshot['token_ids_sha256'] == hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()
            and snapshot['device_input_bytes_equality'] == 'COMPLETE_DEVICE_READBACK_VERIFIED'
            and snapshot['attention_metadata_keys'], 'COMPLETE_FRESH_PREFILL_SCHEDULER_INPUT', name)
        r.check(row['input_sha256'] == obs['input_sha256'] == rows[0]['input_sha256']
            == b.input_hash(tokens, snapshot['positions'], snapshot['input_tensor_metadata']),
            'ACTUAL_INPUT_DIGEST_RECOMPUTATION', name)
        shape = obs['output_shape']; shapes.append(shape)
        r.check(obs['root_returned'] is True and len(shape) == 2 and shape[0] == 1 and b.positive(shape[1])
            and obs['model_output_sha256'] == row['output_sha256'] and b.valid_sha(row['output_sha256'])
            and type(obs['model_output_finite']) is bool and obs['model_output_finite'] == row['output_finite']
            and type(obs['top1']) is int and 0 <= obs['top1'] < shape[1]
            and driver['output']['tokens'] == [obs['top1']], 'ACTUAL_FULL_LOGITS_AND_GENERATED_TOKEN', name)
        if index != 5:
            r.check(obs['model_output_finite'] is True and row['output_sha256'] == baseline
                and driver['output'] == d['healthy_before'], 'FULL_HEALTHY_LOSSLESS_OFF_OUTPUT_EQUAL', name)
    health = d['post_unregister_health']
    r.check(all(s == shapes[0] for s in shapes) and health['worker'] == context['host']['worker']
        and health['output'] == d['healthy_before']
        and health['mapping']['internal_request_id'] not in internal
        and health['mapping']['external_request_id'] not in external
        and b.positive(health['mapping']['request_state_object_id'])
        and health['mapping']['source'] == 'ACTUAL_PUBLIC_ENQUEUE_AND_OUTPUT_PROCESSOR'
        and health['mapping']['proof_scope'] == 'DRIVER_MAPPING_ONLY_REQUIRES_REAL_MODEL_PREHOOK_CORROBORATION',
        'ORDINARY_SAME_WORKER_HEALTH_AFTER_UNREGISTER')
    return dict(requests=8, ordinary_post_unregister_requests=1, complete_output_shape=shapes[0],
        healthy_logits_sha256=baseline, fault_logits_sha256=rows[5]['output_sha256'],
        fault_logits_finite=rows[5]['output_finite'], whole_request_output_changed=rows[5]['output_sha256'] != baseline,
        post_native_remove_full_logits=True, post_python_remove_full_logits=False)


def native_chain(b, r, d, context, artifacts, *, mechanism, expected_count):
    inv = context['final']['inventory']
    rows, lifetime, aliases = inv['rows'], inv['lifetime'], inv['alias_inventory']
    pairs = {(v['path'], v['object_identity']) for v in aliases}
    scope = SCOPES[mechanism]
    number = 1 if mechanism == '26.residual_stomp' else 2
    r.check(inv['schema'] == 'demoncat.writer26.candidate_experiment.v2'
        and inv['stage'] == d['stage'] == 'c1_single_effects' and inv['mechanism'] == d['mechanism'] == mechanism
        and inv['scope'] == d['scope'] == scope and inv['build'] == d['manifest_sha256'], 'CURRENT_SINGLE_SITE_BUILD_SCOPE')
    single = inv['single_site_c1']
    r.check(inv['c1'] is inv['c2'] is inv['c3'] is False and single['qualification_granted'] is False
        and single['all_eligible_qualification'] is single['serializable_authority'] is single['fallback_attempted'] is False
        and single['evidence_complete'] is True and single['selected_native_admissions'] == 5
        and single['actual_fault_hits'] == 1 and single['complete_native_counts'] == [expected_count]*8,
        'SINGLE_SITE_REPORT_CONSISTENT_WITH_BOUNDED_SCOPE')
    worker = lifetime['worker']
    r.check(worker == context['host']['worker'] == d['setup']['worker'] and worker['pid'] == lifetime['pid']
        and worker['process_start_identity'] == lifetime['process_start_identity'] and b.positive(lifetime['pid'])
        and b.positive(lifetime['native_thread']) and worker['rank'] == 0
        and worker['logical_device'] == worker['configured_logical_device'] == worker['observed_logical_device'] == 0
        and worker['worker_start_identity'] and worker['model_instance_id'], 'EXACT_TP1_WORKER_LIFETIME')
    r.check(len({v['path'] for v in aliases}) == len(aliases) and all(b.positive(v['object_identity'])
        and isinstance(v['path'], str) and v['path'] for v in aliases)
        and ('<root>', lifetime['model_object_identity']) in pairs, 'COMPLETE_ACTUAL_MODULE_ALIAS_SET')
    r.check(inv['failure'] is inv['first_failure'] is None and inv['failure_events'] == [] and inv['blockers'] == [],
            'FIRST_FAILURE_AND_UNRESOLVED_RECOVERY_RETAINED')
    rebuilt, seen_calls, seen_tokens = [], set(), set()
    callbacks = producers = 0
    for index, row in enumerate(rows):
        name = row['phase']
        b.trace(r, row, lifetime, name, mechanism=number)
        if index:
            previous, current = rows[index-1]['native_after'], row['native_before']
            delta = current['all_runs'] - previous['all_runs']
            r.check(delta >= 0 and delta == current['non_target_runs'] - previous['non_target_runs']
                and all(current[k] == previous[k] for k in current if k not in ('all_runs', 'non_target_runs')),
                'NO_HIDDEN_BETWEEN_REQUEST_WRITER_CALL', name)
        r.check(row['callback_start'] == callbacks and row['producer_start'] == producers, 'COMPLETE_PUBLIC_CALLBACK_OFFSETS', name)
        callbacks += len(row['callbacks']); producers += len(row['producers'])
        actual = b.correlate(row, mechanism=number); rebuilt.append(actual)
        r.check(actual == row['candidates'] and len(actual) == expected_count, 'REBUILD_COMPLETE_NATIVE_CANDIDATE_DENOMINATOR', name)
        calls = [v['call'] for v in row['native_events'] if v['kind_name'] == ('qmm_enter' if number == 1 else 'fia_before')]
        tokens = [v['native_packet']['token'] for v in row['callbacks']]
        callback_calls = [v['independent_call'] for v in row['callbacks']]
        r.check(len(set(calls)) == len(calls) and not seen_calls.intersection(calls)
            and len(set(tokens)) == len(tokens) and not seen_tokens.intersection(tokens)
            and len(set(callback_calls)) == len(callback_calls) and set(callback_calls).issubset(calls)
            and (not row['callbacks'] if index == 7 else bool(row['callbacks'])), 'UNIQUE_ACTUAL_CALL_AND_CALLBACK_LIFETIMES', name)
        seen_calls.update(calls); seen_tokens.update(tokens)
        for value in row['producers']:
            b.public_producer(r, value, row, pairs, name)
        for ordinal, candidate in enumerate(actual):
            b.callback_proof(r, candidate, row, lifetime, pairs, artifacts, name+':'+str(ordinal), mechanism=number)
        complete = row['complete_inventory']
        r.check(complete['native_writer_count'] == len(calls) == expected_count
            and complete['native_events_sha256'] == b.digest(row['native_events'])
            and complete['candidates_sha256'] == b.digest(actual)
            and complete['actual_aliases_sha256'] == b.digest(aliases), 'FULL_UNFILTERED_INVENTORY_DIGESTS', name)
    frozen = b.freeze(rebuilt[0], rebuilt[1])
    r.check(frozen == inv['candidates'], 'TWO_HEALTHY_CENSUS_CLASSIFICATION_REBUILT')
    resolved = {v['target_key']:v for v in frozen if v['status'] == 'OBSERVED_NOT_QUALIFIED'}
    r.check(bool(resolved) and len(resolved) == inv['resolved_candidate_count'], 'COMPLETE_RESOLVED_SET_PRESERVED')
    selected, q = inv['single_site_c1']['selection'], inv['single_site_c1']['reference_lossless_off']
    key = min(resolved)
    r.check(selected['target_key'] == key and selected['inventory_index'] == resolved[key]['index']
        and selected['policy'] == 'MIN_ACTUAL_HEALTHY_RESOLVED_TARGET_KEY_NO_FALLBACK'
        and selected['complete_native_count'] == len(frozen) and selected['complete_resolved_count'] == len(resolved)
        and selected['inventory_sha256'] == b.digest(frozen) and selected['aliases_sha256'] == b.digest(aliases)
        and selected['planned_target_count'] == 1 and selected['max_targets_is_selector'] is False,
        'AUTOMATIC_STABLE_SINGLE_TARGET_NO_FALLBACK')
    outcomes = inv['outcomes']
    r.check(len(outcomes) == 5 and [v['mode'] for v in outcomes] == ['reference', 'lossless', 'off', 'fault', 'off'],
            'EXACT_FIVE_NATIVE_OUTCOMES')
    r.check(q['target_key'] == key and q['request_ids'] == [v['request_id'] for v in outcomes[:3]]
        and q['outcomes_sha256'] == b.digest(outcomes[:3]) and q['lifetime_sha256'] == b.digest(lifetime)
        and q['build'] == inv['build'] and q['baseline_output_sha256'] == rows[0]['output_sha256']
        and q['scope'] == scope and q['owner'] == context['owner'] and q['generation'] == context['generation']
        and q['native_owner'] == b.words(hashlib.sha256(q['owner'].encode()).hexdigest())[:2]
        and q['qualification_granted'] is q['serializable_authority'] is False,
        'FRESH_PUBLIC_OWNER_REFERENCE_LOSSLESS_OFF_BEFORE_F')
    callbacks = admissions = 0
    for index, row in enumerate(rows):
        name, mode = PHASES[index], MODES[index]
        r.check(row['plan'] == dict(phase=name, mode=mode, target=key if mode else None)
            and row['admissions'] == row['completed'] == int(mode is not None), 'ONE_ACTUAL_SELECTED_CALL_PER_PLAN', name)
        claimed = [v for v in row['callbacks'] if v['claimed'] == 1]
        r.check(len(claimed) == int(mode is not None) and all(v['target_key'] == key for v in claimed),
                'UNSELECTED_WRITERS_UNMODIFIED', name)
        if mode:
            candidate = row['candidates'][selected['inventory_index']]
            r.check(b.classify(candidate) == ('OBSERVED_NOT_QUALIFIED', []) and candidate['callback']['target_key'] == key
                and all(candidate['native'][k] == rows[1]['candidates'][selected['inventory_index']]['native'][k]
                        for k in b.STABLE_NATIVE), 'CURRENT_SELECTED_OCCURRENCE_REVALIDATED', name)
            b.outcome(r, outcomes[index-2], candidate, row, q, name, mechanism=number)
        advance = context['linked'][5+index*3]['response']['provider_state']['native']
        after = context['linked'][6+index*3]['response']['provider_state']['native']
        b.runtime_state(r, advance, q, lifetime['native_thread'], callbacks, admissions,
            removed=False, sealed=index >= 5, where=name+':advance', mechanism=number)
        callbacks += len(row['callbacks']); admissions += int(mode is not None)
        b.runtime_state(r, after, q, lifetime['native_thread'], callbacks, admissions,
            removed=index == 7, sealed=index >= 4, where=name+':query', mechanism=number)
    b.runtime_state(r, inv['native_state'], q, lifetime['native_thread'], callbacks, admissions,
        removed=True, sealed=True, where='finish', mechanism=number)
    r.check(inv['qualified_candidate_count'] == 1 and len({v['target'] for v in outcomes}) == 1
        and len({v['writer_output_sha256'] for v in outcomes}) == 1
        and len({v['residual_before_sha256'] for v in outcomes}) == 1
        and inv['last_native_receipt'] == outcomes[-1]['observed_consumer_receipt'], 'ONE_EFFECT_TARGET_AND_FULL_RECOVERY_BASELINES')
    r.check(inv['phase'] == 'complete' and inv['installed'] is False and inv['native_removed'] is True
        and inv['owned_hooks'] == inv['owned_public_api_wrappers'] == inv['single_site_c1']['owned_root_logits_hooks'] == 0
        and all(inv[k] == 0 for k in ('pending_output_tensors', 'pending_residual_tensors', 'pending_public_storage_owners'))
        and inv['pending_per_call_leases'] == inv['pending_outcomes'] == []
        and (inv['public_output_observer'] is None if number == 1 else
            inv['public_output_observer']['owned_api_wrappers'] == inv['public_output_observer']['inflight_public_calls'] == 0),
        'NO_RETAINED_NATIVE_PUBLIC_TENSOR_OR_HOOK_RESOURCES')
    snapshots = [('initial', inv['metadata_arrays_initial'])] + [(v['phase'], v['metadata_arrays']) for v in rows]
    snapshots.append(('finish', inv['metadata_arrays_latest']))
    array_report = b.metadata(r, snapshots, lifetime, inv['build'])
    return dict(natural_calls=len(seen_calls), calls_per_request=[len(v['candidates']) for v in rows],
        classification=dict(Counter(v['status'] for v in frozen)),
        reasons=dict(Counter(reason for v in frozen for reason in v['reasons'])),
        resolved_targets=len(resolved), selected_key=key, selected_index=selected['inventory_index'],
        fault_invocations=1, fault_changed_residual_bytes=outcomes[3]['consumer_receipt']['changed_residual_bytes'],
        metadata_arrays=array_report, allocator_artifacts_rehashed=len(artifacts.cache))


def review_document(d, originals, *, mechanism, expected_count):
    b = helper('review_native26_single_source54.py')
    r, details = b.Review(), {}
    r.check(d['schema'] == 'demoncat.writer26.worker_core_probe.v1' and d['status'] == SUCCESS
        and d['mechanism'] == mechanism and d['c2'] is d['c3'] is d['qualification_granted'] is False
        and d['previous_direct_entry_qualification_inherited'] is False
        and d['E5'] == 'UNADAPTED_COLLECTOR_BOUNDARY'
        and not any(d.get(k) for k in ('error', 'cleanup_error', 'abort_error', 'owned_request_abort')),
        'NEW_PUBLIC_CORE_ATTEMPT_WITH_NO_HIDDEN_FAILURE')
    context = r.section('public_protocol', protocol, b, r, d, originals, mechanism=mechanism)
    if context is not None:
        details['requests'] = r.section('actual_requests', requests, b, r, d, context)
        details['native'] = r.section('native_chain', native_chain, b, r, d, context, b.Artifacts(originals.root),
            mechanism=mechanism, expected_count=expected_count)
        if mechanism == '26.fia_destination':
            typed = helper('review_native26_fia_single_source62.py')
            details['typed_fia'] = r.section('typed_fia', typed.typed_review, b, context['final']['inventory'],
                typed.FiaArtifacts(b, originals.root), expected_count=expected_count)
            r.check(details['typed_fia'] is not None and details['typed_fia']['consistent'], 'COMPLETE_TYPED_FIA_STOCK_BRIDGE')
    return dict(**r.summary(), details=details, mechanism=mechanism,
        public_protocol_reconciled=not r.errors, public_qualification_granted=False, c2=False, c3=False, e5=False,
        originals_rehashed=len(originals.checked), evidence_kind='saved_source_bound_natural_worker_receipts',
        limits=[
            'This independently rebuilds one selected Native writer/consumer/recovery chain, not all eligible targets or any live C2/C3 authority.',
            'F functional effect is the actual changed residual and complete consumer receipt. Full-logit change/finite status is reported separately; no E5 attribution is inferred.',
            'Full inputs, residuals and logits are frozen-runtime hash/readback receipts, not independently rehashed tensor files. Typed FIA additionally rehashes saved stock/writer files.',
            'Actual pointer ownership, synchronization, original-call counts and module/dispatcher identity remain source-bound receipts, not new live observations.',
            'The immutable Core files do not save the logits boundary object; complete per-request output shapes join the reviewed final-logits source guard.',
            'The ordinary post-unregister request checks same Worker, tokens and text; full logits are only captured during the preceding post-Native-release health request.'])


def provenance(d, final, source_root, native_root, *, source_commit, source_sha, build_sha):
    b = helper('review_native26_single_source54.py')
    r = b.Review()
    manifest_path = under(source_root, 'source_manifest.json')
    manifest = read(manifest_path)
    r.check(file_sha(manifest_path) == source_sha and manifest['base_commit'] == source_commit
        and d['source_manifest'] == dict(sha256=source_sha, base_commit=source_commit), 'EXACT_REVIEWED_SOURCE_SNAPSHOT')
    files = {v['path']:v for v in manifest['files']}
    r.check(len(files) == len(manifest['files']) and REQUIRED_SOURCES.issubset(d['sources']), 'COMPLETE_CURRENT_CORE_SOURCE_SET')
    for name, expected in d['sources'].items():
        path = under(source_root, name)
        r.check(b.valid_sha(expected) and file_sha(path) == expected == files[name]['sha256']
            and path.stat().st_size == files[name]['bytes'], 'SOURCE_BYTES_AND_FROZEN_MANIFEST', name)
    loaded = d['setup']['loaded_sources']
    r.check(set(loaded) == LOADED, 'COMPLETE_ACTUAL_INSTALLED_MODULE_SET')
    for name, value in loaded.items():
        installed = '/installed/' + (name[4:] if name.startswith('src/') else name)
        r.check(value['path'] == installed and value['sha256'] == d['sources'][name], 'ACTUAL_LOADED_SOURCE_SHA', name)
    build_path = under(native_root, 'build.json'); build = read(build_path)
    inv = final['inventory']
    r.check(file_sha(build_path) == build_sha == d['manifest_sha256'] == inv['build']
        and build['schema'] == 'demoncat.writer26.group_deployment.v2'
        and build['legacy_writers_absent'] is True and build['actual_npu_calls'] == 0, 'CURRENT_NATIVE_BUILD_ORIGINAL')
    for name, expected in build['files'].items():
        r.check(b.valid_sha(expected) and file_sha(under(native_root, name)) == expected, 'NATIVE_FILE_BYTES', name)
    receipt_path = under(native_root, 'build-receipt.json'); receipt = read(receipt_path)
    r.check(file_sha(receipt_path) == build['build_receipt_sha256'] and receipt['schema'] == 'demoncat.writer26.group_build.v2'
        and receipt['actual_npu_calls'] == 0 and receipt['device_mounts'] == [] and receipt['sources'] == build['sources'],
        'CURRENT_NATIVE_SDK_BUILD_RECEIPT')
    r.check({'fault_injection/native_adapters/writer26_group_core.h',
        'fault_injection/native_adapters/writer26_group_runtime.cpp'}.issubset(build['sources']),
            'ACTUAL_NATIVE_CORE_INPUTS_REQUIRED')
    for name, expected in build['sources'].items():
        r.check(b.valid_sha(expected) and file_sha(under(source_root, name)) == expected == files[name]['sha256'],
                'NATIVE_COMPILE_SOURCE_BYTES', name)
    r.check(all(not (Path(native_root)/name).exists() for name in (
        'libfault26_qmm_output_underflow_abi0.so', 'libfault26_fia_writer_proxy_abi0.so')), 'LEGACY_WRITER_DSO_ABSENT')
    spec = build['metadata_array_ownership']
    r.check(spec['schema'] == 'demoncat.metadata_arrays.v1' and spec['state_version'] == 1 and spec['images'] == b.SYMBOLS
        and spec['reset_supported'] is False and spec['runtime_quiescent_readback_required'] is True
        and spec['parent_receipt_sha256'] == build['files']['metadata-array-ownership.json'], 'SDK_METADATA_OWNERSHIP_PROFILE')
    r.check(build['files']['libfault26_readonly_trace_abi0.so'] == inv['lifetime']['native_inventory_sha256']
        and build['files']['libdcat26_group_v2.so'] == inv['lifetime']['native_writer_sha256'], 'ACTUAL_TRACE_AND_WRITER_BUILD_JOIN')
    for snapshot in (inv['metadata_arrays_initial'], inv['metadata_arrays_latest']):
        for value in snapshot['images']:
            r.check(value['sha256'] == build['files'][value['name']] and value['path'] == '/native/'+value['name'],
                    'ACTUAL_MAPPED_METADATA_IMAGE_SHA', value['name'])
    startup = d['metadata_startup']
    r.check(startup['schema'] == 'demoncat.writer26.startup_dependencies.v1' and startup['pid'] == inv['lifetime']['pid']
        and startup['manifest_sha256'] == build_sha
        and startup['implementation_sha256'] == d['sources']['fault_injection/deployments/writer26_loader.py']
        and startup['model_runtime_imports_before'] == [] and startup['callback_installed'] is False
        and startup['device_api_calls'] == 0 and startup['global_symbol_scope_changed'] is False, 'PRE_MODEL_DEPENDENCY_BOOTSTRAP')
    b.metadata(r, [('startup', startup['metadata_arrays']), ('initial', inv['metadata_arrays_initial']),
        ('finish', inv['metadata_arrays_latest'])], inv['lifetime'], build_sha)
    weights = d['weights']; w = weights['manifest']
    r.check(weights['path'] == '/model' and weights['weights_sha256'] == digest(w)
        and w['format'] == 'safetensors' and w['files'] and b.valid_sha(w['config_sha256'])
        and all('/' not in name and '\\' not in name and name.endswith('.safetensors')
                and b.positive(v['bytes']) and b.valid_sha(v['sha256']) for name,v in w['files'].items()), 'FULL_LOCAL_PRETRAINED_WEIGHT_RECEIPT')
    r.check(w['loader_selection'] == "vllm.LLM(load_format='safetensors', trust_remote_code=False)",
            'EXPLICIT_VLLM_SAFETENSORS_LOADER')
    r.check((w['index'] is None and set(w['files']) == {'model.safetensors'}) or (
        isinstance(w['index'], dict) and '/' not in w['index']['file'] and '\\' not in w['index']['file']
        and w['index']['file'].endswith('.safetensors.index.json') and b.valid_sha(w['index']['sha256'])),
        'UNAMBIGUOUS_SAVED_WEIGHT_INDEX')
    if 'quantization_description' in w:
        quant = w['quantization_description']
        r.check(quant['file'] == 'quant_model_description.json' and b.positive(quant['bytes'])
            and b.valid_sha(quant['sha256']), 'ACTUAL_QUANTIZATION_DESCRIPTION_RECEIPT')
    source = d['setup']['model_source']
    r.check(source['source_sha256'] == digest(source['manifest']) and source['manifest']['files']
        and all(b.valid_sha(v) for v in source['manifest']['files'].values())
        and all(v['source'] in source['manifest']['files'] for v in source['manifest']['classes']), 'ACTUAL_MODEL_SOURCE_RECEIPT')
    if d['mechanism'] == '26.fia_destination':
        typed = helper('review_native26_fia_single_source62.py')
        r.check(typed.build_profile(b, build)['consistent'], 'CURRENT_TYPED_FIA_BUILD_PROFILE')
    return dict(**r.summary(), source_files_rehashed=len(d['sources']), native_files_rehashed=len(build['files'])+1,
        native_compile_sources_rehashed=len(build['sources']), external_model_and_weight_files_rehashed=False,
        limit='Weight/config and external model source digests are original frozen-runtime receipts; those large/external files are not re-read by this CLI.')


def forbid_runtime(event, args):
    if event in ('ctypes.dlopen', 'subprocess.Popen', 'os.system', 'socket.connect'):
        raise RuntimeError('READ_ONLY_REVIEW:' + event)
    if event == 'import' and args[0].split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend', 'demoncat', 'fault_injection'):
        raise RuntimeError('NO_RUNTIME_IMPORT_IN_REVIEW')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-root', type=Path, required=True)
    p.add_argument('--result-sha256', required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--source-manifest-sha256', required=True)
    p.add_argument('--native-root', type=Path, required=True)
    p.add_argument('--native-build-sha256', required=True)
    p.add_argument('--mechanism', choices=tuple(SCOPES), required=True)
    p.add_argument('--expected-writers', type=int, required=True)
    args = p.parse_args(argv)
    if (not 1 <= args.expected_writers <= 4096 or not re.fullmatch('[0-9a-f]{40}', args.source_commit)
            or any(not re.fullmatch('[0-9a-f]{64}', value) for value in (
                args.result_sha256, args.source_manifest_sha256, args.native_build_sha256))):
        p.error('Exact SHA pins, source commit and bounded complete writer count required')
    sys.dont_write_bytecode = True
    sys.addaudithook(forbid_runtime)
    try:
        originals = Originals(args.evidence_root)
        d = originals.load(dict(path='probe/result.json', sha256=args.result_sha256))
        result = review_document(d, originals, mechanism=args.mechanism, expected_count=args.expected_writers)
        b, guard = helper('review_native26_single_source54.py'), None
        guard = b.Review()
        final_ref = d['c1']['original_observations']['originals'][-1]
        final = originals.load(final_ref, absolute=True)
        proof = guard.section('source_and_build', provenance, d, final, args.source_root, args.native_root,
            source_commit=args.source_commit, source_sha=args.source_manifest_sha256, build_sha=args.native_build_sha256)
        result['provenance'], result['provenance_schema'] = proof, guard.summary()
        result['consistent'] = bool(result['consistent'] and proof and proof['consistent'] and not guard.errors)
        result['public_protocol_reconciled'] = result['consistent']
        result['result_sha256'] = args.result_sha256
    except (KeyError, TypeError, ValueError, OSError, IndexError, OverflowError, AttributeError) as error:
        result = dict(consistent=False, error=type(error).__name__+':'+str(error), public_qualification_granted=False,
                      c2=False, c3=False, e5=False)
    result.update(schema='demoncat.writer26.public_core.independent_review.v1',
        reviewer_sha256=file_sha(__file__), read_only=True, atomic_helpers=HELPERS)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['consistent'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
