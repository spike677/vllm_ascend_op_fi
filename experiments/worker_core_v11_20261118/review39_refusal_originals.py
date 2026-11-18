"""Offline audit of the bounded Fault39 refusal's full original evidence.

No provider imports, new execution, edited receipts, or live qualifications.
The exact two experiment helpers are audited before the Core-only RPC partition
is given to the existing group reader. Native build provenance is separate.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
import importlib.util
_spec = importlib.util.spec_from_file_location('_refusal_group', HERE/'review39_group_originals.py')
GROUP = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(GROUP)
C, OLD = GROUP.C, GROUP.OLD
NARROW = C.pure_module(HERE/'review39_refusal_diagnostic.py', '_narrow_refusal')
HELPER = 'experiments.worker_core_v11_20261118.output39_refusal_probe.'
INSTALL_METHODS = (HELPER+'install_diagnostic','__main__.install_diagnostic')
FINISH_METHODS = (HELPER+'finish_diagnostic','__main__.finish_diagnostic')
SELF = 'experiments/worker_core_v11_20261118/output39_refusal_probe.py'
PLAN = dict(predecessor_module='model.layers.0.mlp.gate_up_proj',
    refused_module='model.layers.1.mlp.gate_up_proj', operator='vllm.unquantized_gemm.default',
    call_ordinal=0, refused_shape=[14,6144], expected_all_calls=113, expected_eligible=28,
    maximum_group_requests=42, maximum_observations=5)
PIN_FILES = ('fault_injection/deployments/postwrite.py',
    'fault_injection/deployments/postwrite_coverage.py',
    'fault_injection/deployments/postwrite_boundaries.py',
    'fault_injection/deployments/postwrite_inventory.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/providers/fault39_output_group.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/serving39_output_group.py',
    'experiments/worker_core_v11_20261118/output39_refusal_diagnostic.py',
    'fault_injection/native_adapters/postwrite/postwrite.cpp',
    'fault_injection/native_adapters/postwrite/postwrite.h')


def helper_partition(a, d, host, records, diagnostic, *, natural):
    installs = [(i,r) for i,r in enumerate(records) if r['method'] in INSTALL_METHODS]
    finishes = [(i,r) for i,r in enumerate(records) if r['method'] in FINISH_METHODS]
    a.check(len(installs) == len(finishes) == 1, 'EXACT_TWO_ORIGINAL_HELPER_RPCS')
    if len(installs) != 1 or len(finishes) != 1:
        return None
    ii, install = installs[0]; fi, finish = finishes[0]
    # The actual queue executes the pinned file as __main__; imports in CPU
    # contracts use its package name. Neither name alone authenticates a helper:
    # the full source arguments, SHA, exact Worker/ticket and response are below.
    a.check(install['method'].rsplit('.',1)[0] == finish['method'].rsplit('.',1)[0], 'SAME_PINNED_HELPER_ENTRY_NAMESPACE')
    a.check(ii < fi, 'HELPER_INSTALL_BEFORE_FINISH')
    a.check(install.get('responses') == [d.get('diagnostic_install')] and
            finish.get('responses') == [d.get('diagnostic_original')], 'EXACT_HELPER_RESPONSE_ORIGINALS')
    identity,ticket,source,pins,output,plan,cpu = install['args']
    a.check(identity == diagnostic['worker_identity'] == host['worker'] == d['worker'] and
            finish['args'] == [identity,ticket] and diagnostic['helper']['actual_public_rpc_ticket'] == ticket,
            'HELPER_SAME_ACTUAL_WORKER_AND_PUBLIC_TICKET')
    a.check(type(cpu) is bool and cpu is (not natural) and diagnostic['explicit_cpu_contract'] is cpu,
            'HELPER_CPU_AND_NATURAL_SCOPES_NOT_INTERCHANGEABLE')
    a.check(pins == d['sources'] and diagnostic['source_pins'] == {p:pins[p] for p in PIN_FILES} and
            pins[SELF] == diagnostic['helper']['source_sha256'] == d['diagnostic_install']['source_sha256'],
            'HELPER_EXACT_FROZEN_SOURCE_ARGUMENTS')
    a.check(output == d['diagnostic_original']['path'] and plan == diagnostic['plan'] == d['plan'],
            'HELPER_EXACT_OUTPUT_AND_PLAN')
    if natural:
        a.check(source == '/source' and plan == PLAN, 'FROZEN_PRODUCTION_ENTRY_AND_PLAN')
    count = plan['expected_eligible']
    full = GROUP.PREFIX+['c2_lossless']+['c2_fault']*count+['c2_off','c2_health']+[
        'c3_lossless','c3_fault','c3_fault','c3_off','c3_health']
    prefix = full[:full.index('c3_fault')+1]
    expected_budget = dict(derivation='ACTUAL_PROVIDER_PLAN_AFTER_ORIGINAL_PREFIX_COLLECTION',
        original_full_group_schedule=full, diagnostic_normal_prefix=prefix,
        diagnostic_normal_requests=len(prefix), bounded_recovery_schedule=['group_recovery_off','group_recovery_health'],
        initial_ordinary_health=1,post_unregister_ordinary_health=1,
        maximum_user_requests=len(prefix)+4,original_schedule_modified=False)
    a.check(d['diagnostic_install']['phase_budget'] == expected_budget and len(prefix)+4 <= 43,
            'ORIGINAL_PHASE_BUDGET_NO_REPLACED_PROTOCOL')
    a.check(d['diagnostic_install']['actual_completed_requests'] == 6 and
            d['diagnostic_install']['status'] == 'DIAGNOSTIC_INSTALLED_AFTER_ORIGINAL_PREFIX' and
            C.positive(diagnostic['helper']['thread']) and
            d['diagnostic_install']['thread'] == diagnostic['helper']['thread'], 'SAME_HELPER_THREAD_AFTER_SIX')
    before = [r for r in records[:ii] if C.rpc_action(r)[0] == 'advance' and
              C.rpc_action(r)[1].get('settings') == {'collect':True}]
    a.check(len(before) == 6, 'SIX_ACTUAL_COLLECT_RPCS_BEFORE_INSTALL')
    state = before[-1]['responses'][0]['provider_state'] if before else {}
    a.check(state.get('serving',{}).get('next_phase', state.get('next_phase')) == 'c2_lossless' or
            state.get('evidence',{}).get('next_phase') == 'c2_lossless', 'INSTALL_AT_ORIGINAL_C2_BOUNDARY')
    a.check(diagnostic['helper']['actual_provider_ticket'] == [state['owner'],state['generation']] and
            before[-1]['responses'][0]['ticket'] == ticket, 'EXACT_PUBLIC_TO_PRIVATE_PROVIDER_TICKET')
    public = [r for r in records if C.rpc_action(r)[0]]
    first_recover = next((i for i,r in enumerate(records) if C.rpc_action(r)[0] == 'recover'),None)
    a.check(first_recover is not None and fi < first_recover, 'TRACE_CLOSED_BEFORE_ORIGINAL_RECOVERY')
    finish_result = d['diagnostic_original']
    a.check(finish_result['status'] == 'DIAGNOSTIC_CLOSED' and finish_result['trace_removed'] is True and
            finish_result['native_qualification'] is False and finish_result['observations'] == len(diagnostic['records']) and
            finish_result['errors'] == diagnostic['errors'] == [] and
            finish_result['original_core_profile_managed_by_original_recovery'] is True and
            diagnostic['helper']['public_core_modified'] is False and not d.get('diagnostic_close_error'),
            'OWN_EXPERIMENT_CLEANUP_AND_ORIGINAL_RECOVERY_DISTINCT')
    selected = {ii,fi}; core = [r for i,r in enumerate(records) if i not in selected]
    a.check(len(core)+2 == len(records) and len(public) == len(host['history']), 'COMPLETE_RPC_PARTITION_NO_OMISSIONS')
    return core


def all_originals(a, originals, d, records):
    """Load every reference, including collect_failed not present in requests."""
    refs = {}
    def add(ref):
        old = refs.get(ref['path'])
        a.check(old is None or old == ref, 'ORIGINAL_REFERENCE_NEVER_REBOUND', ref['path'])
        refs[ref['path']] = ref
    for row in d.get('requests',[])+d.get('recovery_requests',[]):
        if row.get('original_observations'):
            add(row['original_observations'])
    for raw in records:
        for reply in raw.get('responses',[]):
            if not isinstance(reply,dict):
                continue
            state = reply.get('provider_state',{})
            for ref in state.get('evidence',{}).get('group_originals',[]):
                add(ref)
    loaded = [originals.load(ref,mounted=True) for ref in refs.values()]
    a.check(bool(loaded), 'COMPLETE_ORIGINALS_REQUIRED')
    return loaded


def native_universe(a, snapshots, final, *, natural, expected_unique=None):
    """Deduplicate immutable copies, not actual Native invocations."""
    unique = {}; sightings = 0
    def walk(value):
        nonlocal sightings
        if isinstance(value,dict):
            if value.get('schema') == 'demoncat.postwrite.lease_receipt.v1':
                sightings += 1
                key = (value['owner'],value['generation'],value['handle'])
                prior = unique.get(key)
                if prior is not None:
                    n = min(len(prior['events']),len(value['events']))
                    a.check(prior['events'][:n] == value['events'][:n] and all(
                        prior[k] == value[k] for k in ('geometry','elements','element_offset','thread','owner','generation','handle')),
                        'NATIVE_DUPLICATE_OR_PREFIX_EXACT', str(key))
                if prior is None or len(value['events']) > len(prior['events']):
                    unique[key] = value
                return
            for child in value.values(): walk(child)
        elif isinstance(value,list):
            for child in value: walk(child)
    for saved in snapshots: walk(saved)
    final_keys = set()
    def keys(value):
        if isinstance(value,dict):
            if value.get('schema') == 'demoncat.postwrite.lease_receipt.v1':
                final_keys.add((value['owner'],value['generation'],value['handle'])); return
            for child in value.values(): keys(child)
        elif isinstance(value,list):
            for child in value: keys(child)
    keys(final)
    a.check(set(unique) == final_keys, 'FINAL_ORIGINAL_RETAINS_ALL_NATIVE_LEASES')
    for key,r in unique.items():
        where = str(key); events = r['events']; geom = r['geometry']
        a.check([e['operation'] for e in events] == ['open','apply','restore','release','live'] and
                [e['sequence'] for e in events] == [1,2,3,4,5] and r['closed'] is True and
                r['open'] == events[0] and r['applications'] == [events[1]] and r['restore'] == events[2] and
                r['release'] == events[3] and r['live_readbacks'] == [events[4]], 'COMPLETE_NATIVE_EVENT_LIFECYCLE',where)
        for e in events:
            a.check(all(type(e[k]) is int and e[k] == r[k] for k in ('owner','generation','handle')) and
                    e['returned'] is True and e['error'] is None and type(e['return_code']) is int,
                    'NATIVE_EVENT_EXACT_TYPED_IDENTITY',where)
        a.check(all(events[i]['return_code'] == 0 for i in (0,2,3,4)) and events[-1]['native_live'] is False,
                'EVERY_NATIVE_LEASE_RESTORED_RELEASED_AND_NOT_LIVE',where)
        application = events[1]; rc = application['return_code']
        a.check(type(application['mode']) is int and application['mode'] in (0,1) and application['indices'] == [0] and
                type(application['indices'][0]) is int and type(application['changed_elements']) is int and
                application['changed_elements'] == (application['mode'] if rc == 0 else 0) and
                (r['first_failure'] is None if rc == 0 else r['first_failure'] == application),
                'ORIGINAL_APPLY_AND_FIRST_FAILURE_NOT_RELABELED',where)
        a.check(r['element_offset'] == geom['storage_offset'] == 0 and r['elements'] == math.prod(geom['shape']) and
                geom['dtype'] == 'torch.bfloat16' and events[0]['bytes'] == geom['logical_bytes'] == geom['storage_bytes'] == 2*r['elements'] and
                events[0]['data_ptr'] == geom['data_ptr'], 'NATIVE_COMPLETE_BF16_SPAN',where)
        device = geom['device']
        a.check((device.startswith('npu:') and device[4:].isdecimal() and type(events[0]['device']) is int and
                 events[0]['device'] == int(device[4:])) if natural else (device == 'cpu' and events[0]['device'] is None),
                'NATIVE_DEVICE_SCOPE',where)
    count = final['finished_groups']['c2']['eligible_target_count']
    # Initial C1 B/F; C2 lossless + independent F; C3 lossless + two attempted leases.
    expected_unique = 2+3*count+2 if expected_unique is None else expected_unique
    a.check(C.positive(expected_unique) and len(unique) == expected_unique,
            'COMPLETE_NATIVE_LEASE_DENOMINATOR_FROM_ORIGINAL_SCHEDULE')
    return unique, dict(unique_leases=len(unique),snapshot_occurrences=sightings,
        repeated_originals_not_counted_as_new_calls=True,
        apply_return_counts={str(k):v for k,v in Counter(r['events'][1]['return_code'] for r in unique.values()).items()})


def span_tensor_digest(row):
    g = row['binding']['geometry']; s = row['span']
    return C.digest(dict(shape=g['shape'],stride=g['stride'],dtype=g['dtype'],offset=0,
                         elements=s['elements'],bytes_sha256=s['sha256']))


def causal_joins(a, d, diagnostic, final, unique):
    groups = final['finished_groups']; c2,c3 = groups['c2'],groups['c3']
    rows = {(r['arm'],r['target']):r for r in diagnostic['records']}
    initial = final['actual_requests'][0]['complete_inventory']['calls']
    all_keys = [x['key'] for x in initial]; eligible = [x for x in initial if x['eligible']]
    plan = diagnostic['plan']
    a.check(len(all_keys) == plan['expected_all_calls'] and len(eligible) == plan['expected_eligible'] and
            [x['module_path'] for x in eligible[:2]] == [plan['predecessor_module'],plan['refused_module']],
            'FROZEN_TWO_TARGETS_FROM_COMPLETE_HEALTHY_UNIVERSE')
    expected = {'healthy':0,'single_predecessor_fault':1,'cumulative_fault':2}; summaries = {}
    for arm,expected_apps in expected.items():
        selected = [r for r in diagnostic['records'] if r['arm'] == arm]
        rid = selected[0]['binding']['request_id']; group = c3 if arm == 'cumulative_fault' else c2
        forwards = [f for f in group['forward_artifacts'] if f['request_id'] == rid]
        a.check(len(forwards) == 1, 'OBSERVATION_JOINS_ONE_ACTUAL_FORWARD',arm)
        if len(forwards) != 1: continue
        forward = forwards[0]; inv = forward['inventory']; calls = inv['calls']
        expected_keys = all_keys[:all_keys.index(eligible[1]['key'])+1] if arm == 'cumulative_fault' else all_keys
        a.check([c['key'] for c in calls] == expected_keys and inv['dropped_calls'] == 0,
                'COMPLETE_REQUEST_OR_FAILED_PREFIX_CALL_ORDER',arm)
        applications = [r for r in group['native_lease_receipts'] if r['actual_request_id'] == rid]
        executions = [e for e in group['execution_artifacts'] if e['request_id'] == rid]
        a.check(len(applications) == len(executions) == expected_apps, 'ALL_ACTUAL_APPLICATIONS_IN_ARM',arm)
        actual_keys = [r['target_key'] for r in applications]
        a.check(actual_keys == [x['key'] for x in eligible[:expected_apps]], 'EXACT_PREDECESSOR_THEN_REFUSED_APPLICATION_ORDER',arm)
        ledger = [r for r in group['coverage']['calls'] if r['request_ids'] == [rid]]
        if arm != 'healthy':
            a.check(len(ledger) == len(calls) and [r['call_id'] for r in ledger] == [forward['forward_id']+':'+str(i+1) for i in range(len(calls))],
                    'COMPLETE_ARM_LEDGER_INCLUDES_NONTARGETS',arm)
            a.check(sum(r['decision']['admitted'] is True for r in ledger) == expected_apps,
                    'ONLY_EXPECTED_ACTUAL_ADMISSIONS',arm)
        for row in selected:
            b = row['binding']; key = b['target_key']; target = row['target']; where = arm+':'+target
            matched = [c for c in calls if c['key'] == key]
            a.check(len(matched) == 1, 'ONE_ACTUAL_PRODUCER_IN_COMPLETE_INVENTORY',where)
            if len(matched) != 1: continue
            call = matched[0]
            a.check(b['forward_id'] == forward['forward_id'] and b['geometry'] == call['output_geometry'] and
                    b['producer_proof'] == call['producer_proof'] and b['module_path'] == call['module_path'] and
                    b['module_object_identity'] == call['module_object_identity'] and b['operator'] == call['operator'] and
                    call['original_calls'] == 1 and row['consumer_observations'] == call['consumer_observations'],
                    'EXACT_PRODUCER_STORAGE_AND_ACTUAL_CONSUMER_JOIN',where)
            driver = [r for r in d['requests'] if r['mapping']['internal_request_id'] == rid]
            a.check(len(driver) == 1 and driver[0]['phase'] == b['phase'], 'ORIGINAL_DRIVER_PHASE_AND_REQUEST_JOIN',where)
            completed = [r for r in final['actual_requests'] if r['actual_internal_request_id'] == rid]
            if arm != 'cumulative_fault':
                a.check(len(completed) == 1 and completed[0]['snapshot'] == b['request_mapping']['snapshot'],
                        'ACTUAL_INPUT_SNAPSHOT_JOIN',where)
            else:
                a.check(not completed and rid in d['owned_request_abort'] and forward['complete'] is False,
                        'FAILED_REQUEST_ABORT_NOT_INVENTED_COMPLETION',where)
            if row['origin'] == 'SAME_NATIVE_LEASE_ORIGINAL':
                app = [r for r in applications if r['target_key'] == key]
                a.check(len(app) == 1, 'DIAGNOSTIC_NATIVE_LEASE_PRESENT_IN_FULL_ORIGINAL',where)
                if len(app) != 1: continue
                receipt = app[0]['receipt']; eventkey = tuple(receipt[k] for k in ('owner','generation','handle'))
                a.check(unique.get(eventkey) == receipt and row['open_event'] == receipt['open'] and
                        row['apply_event'] == receipt['applications'][0] and receipt['thread'] == diagnostic['helper']['thread'] and
                        receipt['owner'] == b['native_owner'] and receipt['generation'] == b['native_generation'] and
                        app[0]['forward_id'] == b['forward_id'], 'SAME_ORIGINAL_NATIVE_EVENTS_AND_THREAD',where)
                ex = [e for e in executions if e['target_id'] == app[0]['target_id']]
                a.check(len(ex) == 1 and ex[0]['original_output_sha256'] == span_tensor_digest(row) and
                        ex[0]['native_generation'] == b['native_generation'] and ex[0]['original_calls'] == 1 and
                        ex[0]['application_returned'] is (row['apply_event']['return_code'] == 0),
                        'COMPLETE_RAW_SPAN_SHA_REBUILDS_ORIGINAL_TENSOR_DIGEST',where)
            elif arm == 'healthy':
                controls = [r for r in group['per_target_lossless'] if r['key'] == key]
                a.check(len(controls) == 1 and controls[0]['before'] == controls[0]['after'] == span_tensor_digest(row),
                        'HEALTHY_FRESH_SPAN_EQUALS_ORIGINAL_LOSSLESS_BYTES',where)
        summaries[arm] = dict(actual_calls=len(calls),native_fault_applications=len(applications),
            actual_return_codes=[x['receipt']['applications'][0]['return_code'] for x in applications],
            full_call_order_sha256=C.digest([c['key'] for c in calls]))
    a.check(summaries.get('single_predecessor_fault',{}).get('actual_return_codes') == [0] and
            summaries.get('cumulative_fault',{}).get('actual_return_codes') == [0,4], 'ACTUAL_ONE_WRITE_THEN_GUARD_REFUSAL')
    return summaries


def recovery(a, d, host, diagnostic, final, *, natural, stage='c3', expected_budget=None):
    group = final['finished_groups'][stage]; recovery = group['coverage']['recovery']
    baseline = group['forward_artifacts'][0]; forwards = group['forward_artifacts']
    a.check(recovery['state'] == 'DISABLED_CONFIRMED' and recovery['errors'] == [] and
            recovery['held_leases'] == recovery['pending_leases'] == recovery['pending_drains'] == [] and
            type(recovery['inflight']) is int and recovery['inflight'] == 0, 'FAILED_C3_ALL_OWNED_RESOURCES_DRAINED')
    targets = recovery['targets']; a.check(len(targets) == diagnostic['plan']['expected_eligible'], 'RECOVERY_ALL_LOGICAL_TARGETS')
    for target in targets:
        a.check(target['released'] is target['drained'] is target['off_exact'] is True and
                target['pending_reason'] is None and target['off_output_sha256'] == baseline['output_sha256'],
                'RECOVERY_TARGET_FULL_BUNDLE_EQUAL',C.digest(target['lease']['target']))
    off = [r for r in forwards if r['phase'] == 'recovery_retry']; health = group['actual_health']
    a.check(len(off) == 1 and off[0]['output_digest'] == baseline['output_digest'] == health['output_digest'] and
            off[0]['output_sha256'] == baseline['output_sha256'] == health['output_sha256'],
            'BUNDLE_BASELINE_OFF_HEALTH_EXACT_WITHIN_SAME_DIGEST_DOMAIN')
    driver_recovery = d.get('recovery_requests',[])
    a.check([r['phase'] for r in driver_recovery] == ['group_recovery_off','group_recovery_health'] and
            [r['mapping']['internal_request_id'] for r in driver_recovery] == [off[0]['request_id'],health['actual_request_id']],
            'REAL_TWO_RECOVERY_REQUESTS_JOIN_BUNDLE')
    a.check(not final['cleanup_failures'] and final['collection_failure'] is None and
            final['first_collection_failure'] is None and not d.get('recovery_error') and not d.get('runner_error'),
            'NO_HIDDEN_CLEANUP_OR_COLLECTION_FAILURE')
    budget = d['diagnostic_budget']; count = diagnostic['plan']['expected_eligible']
    expected_budget = dict(normal_requests=11+count,recovery_requests=2,maximum_normal_requests=11+count,
        maximum_recovery_requests=2,cumulative_started=True,installed=True,closed=True) if expected_budget is None else expected_budget
    a.check(budget == expected_budget, 'BOUNDED_ACTUAL_REQUEST_COUNTS')
    if natural:
        health = d['post_unregister_health']
        allids = {r['mapping']['internal_request_id'] for r in d['requests']+driver_recovery}
        a.check(health['worker'] == host['worker'] and health['mapping']['internal_request_id'] not in allids and
            health['mapping']['source'] == 'ACTUAL_PUBLIC_ENQUEUE_AND_OUTPUT_PROCESSOR' and
            health['output']['request_id'] == health['mapping']['external_request_id'] and
            all(health['output'][k] == d['healthy_before'][k] for k in ('tokens','text')),
            'ORDINARY_POST_UNREGISTER_HEALTH_SAME_WORKER')
        a.check(d['binding_removal'] == [dict(status='SERVING_DEPLOYMENT_REMOVED')],
                'ORDINARY_BINDING_REMOVED_AFTER_NO_OWNER')
    lossless = {r['target_id']:r for r in group['per_target_lossless']}
    for target in targets:
        tid = C.digest(target['lease']['target'])
        proof = dict(target_id=tid,off=baseline['output_sha256'],native_live_leases=0,
                     off_targets={key:value['before'] for key,value in lossless.items()})
        a.check(target['proof_sha256'] == C.digest(proof) and target['lease']['owner'] == recovery['owner'] and
                target['lease']['generation'] == recovery['generation'], 'ORIGINAL_RECOVERY_PROOF_REBUILT',tid)
    workers = recovery['workers']; original_health = next(r for r in final['actual_requests'] if r['phase']=='group_recovery_health')
    a.check(len(workers)==1 and group['actual_health']['snapshot'] == original_health['snapshot'], 'ACTUAL_RECOVERY_WORKER_HEALTH')
    for worker in workers:
        _,projection=OLD.helpers(OLD.CUSTOM_PROFILE)
        expected_worker=projection.worker(targets[0]['lease']['target'])
        a.check(worker['proof_sha256'] == C.digest(group['actual_health']) and worker['owner'] == recovery['owner'] and
            worker['generation'] == recovery['generation'] and worker['baseline_output_sha256'] == worker['off_output_sha256'] == baseline['output_sha256'] and
            worker['worker'] == expected_worker and
            all(type(worker[k]) is int and worker[k]==0 for k in ('active_leases','device_bytes','host_bytes','inflight_calls','owned_hooks')),
            'ACTUAL_RECOVERY_WORKER_ZERO_RESOURCE_READBACK')


def review(d, host, records, originals, diagnostic, *, natural=True):
    a = C.Audit(); detail = {}
    a.check(d.get('schema') == 'demoncat.output39.fixed_c3_refusal_probe.v1' and
            d.get('mechanism') == '39.output_proxy' and d.get('stage') == 'c2_c3' and
            all(d.get(k) is False for k in ('c2','c3','e5','qualification_granted')), 'EXACT_DIAGNOSTIC_ENTRY_NO_AUTHORITY')
    a.check(d.get('status') == 'REFUSED_OR_FAILED' and 'POSTWRITE_APPLY_4' in str(d.get('error')),
            'ORIGINAL_GUARD_FAILURE_RETAINED_NOT_SUCCESS')
    partition = a.section('helpers',helper_partition,a,d,host,records,diagnostic,natural=natural)
    narrow = NARROW.review(diagnostic); detail['five_record_pattern'] = narrow
    a.check(not narrow['issues'], 'NARROW_FIVE_ORIGINAL_SPANS_RECONCILED')
    snapshots = a.section('all_references',all_originals,a,originals,d,records)
    if not snapshots:
        return dict(**a.report(),details=detail,classification='PARTIAL_OR_UNRESOLVED',causal_qualification_granted=False)
    refs = [r['original_observations'] for r in d.get('requests',[])+d.get('recovery_requests',[]) if r.get('original_observations')]
    final = originals.load(refs[-1],mounted=True)
    if partition is None:
        # Keep already completed Core facts visible on missing/failed helpers.
        # The helper error above remains an issue; this partition grants nothing.
        partition = [r for r in records if r['method'] not in INSTALL_METHODS+FINISH_METHODS]
    detail['core_originals'] = a.section('unchanged_core_originals',GROUP.review,d,host,partition,originals,natural=natural)
    a.check(detail['core_originals'] is not None and detail['core_originals']['consistent'] is True,
            'FULL_CORE_AND_COMPLETED_C2_ORIGINALS_RECONCILED')
    result = a.section('native_universe',native_universe,a,snapshots,final,natural=natural)
    if result:
        unique,detail['native_universe'] = result
        detail['counterfactual_full_joins'] = a.section('counterfactual_joins',causal_joins,a,d,diagnostic,final,unique)
    a.section('failed_c3_recovery',recovery,a,d,host,diagnostic,final,natural=natural)
    if natural:
        removals = [r for r in records if r['method']=='experiments.c2c3_postwrite_20261117.natural39_worker_core_probe.remove_binding']
        a.check(len(removals)==1 and removals[0]['args']==[host['worker']] and
                removals[0].get('responses')==d.get('binding_removal') and
                records.index(removals[0])>max(i for i,r in enumerate(records) if C.rpc_action(r)[0]=='detach'),
                'BINDING_REMOVAL_ACTUAL_RPC_AFTER_DETACH')
    return dict(**a.report(),details=detail,classification='BOUNDED_GUARD_REFUSAL_ORIGINALS' if not a.issues else 'PARTIAL_OR_UNRESOLVED',
        original_error=d.get('error'),original_status=d.get('status'),
        counterfactual_pattern_and_original_joins_consistent=not a.issues,
        causal_qualification_granted=False, native_build_provenance_required_separately=True,
        scope='ACTUAL_STANDARD_WORKER_SAVED_ORIGINALS' if natural else 'EXPLICIT_CPU_ABI_CONTRACT_ONLY',
        limits=['No live C2 capability, C3 completion or statistics E5 is issued.',
            'Metadata hashes bind the observed full spans; private arrays are neither downloaded nor reconstructed.',
            'Native build, SDK/compiler and actual mapped binary provenance require the separate rehashed build chain.',
            'This attempt does not reconstruct the absent historical source84 original arrays.',
            'Failed cumulative forward outcomes are retained, including unreached downstream consumers.'])


def source_review(a, d, host, diagnostic, root, manifest_sha, commit, native_sha):
    path = C.under(root,'source_manifest.json'); manifest = C.read(path)
    a.check(C.sha(manifest_sha) and C.file_sha(path) == manifest_sha and manifest['base_commit'] == commit,
            'PINNED_ACTUAL_SOURCE_MANIFEST')
    files = {r['path']:r for r in manifest['files']}; sources = d['sources']
    _,projection = OLD.helpers(OLD.CUSTOM_PROFILE)
    modules = set(OLD.MODULES+GROUP.EXTRA_MODULES+projection.custom.EXTRA_MODULES)
    expected = set(OLD.SOURCES+GROUP.EXTRA_MODULES+projection.custom.EXTRA_MODULES+list(PIN_FILES)+[SELF,
        'experiments/worker_core_v11_20261118/review39_refusal_diagnostic.py',
        'experiments/worker_core_v11_20261118/prepare_output39_refusal.py'])
    a.check(len(files) == len(manifest['files']) and set(sources) == expected, 'EXACT_DIAGNOSTIC_SOURCE_DEPENDENCY_SET')
    for name,sha in sources.items():
        path = C.under(root,name)
        a.check(C.sha(sha) and C.file_sha(path) == sha == files[name]['sha256'] and path.stat().st_size == files[name]['bytes'],
                'SOURCE_ORIGINAL_REHASH',name)
    binding = d['binding']; value = binding['binding']; native = value['native']
    a.check(binding['worker'] == host['worker'] and set(binding['loaded_sources']) == modules, 'EXACT_ACTUAL_LOADED_MODULE_SET')
    for name,row in binding['loaded_sources'].items():
        allowed = {'/source/'+name,'/installed/'+(name[4:] if name.startswith('src/') else name)}
        a.check(row['sha256'] == sources[name] and row['path'] in allowed, 'LOADED_SOURCE_PIN',name)
    a.check(value['mechanism'] == '39.output_proxy' and value['status'] == 'SERVING_DEPLOYMENT_BOUND_NOT_QUALIFIED' and
            all(type(value[k]) is int and value[k] == 0 for k in ('workers_created','hooks_installed','native_leases_opened')),
            'ORIGINAL_INERT_DEPLOYMENT')
    a.check(C.sha(native_sha) and native['sha256'] == d['native_sha256'] == diagnostic['native_identity']['sha256'] == native_sha and
            type(native['abi']) is int and native['abi'] == 1 and native['loaded_mapping_verified'] is True and
            native['read_span_available'] is True and native['simulation'] is False,
            'EXPLICIT_EXPECTED_NATIVE_SHA_REPORTED_BY_BOTH_OBSERVERS')
    identity=diagnostic['native_identity']
    a.check(identity['scope']=='ACTUAL_LOADED_NATIVE_DSO' and identity['export']=='dcat_pw_copy_original_v1' and
            C.positive(identity['copy_symbol_address']) and identity['build_provenance_review_required'] is True,
            'ADDITIVE_COPY_EXPORT_ACTUAL_MAPPING_RECEIPT')
    codefiles={'apply':('fault_injection/deployments/postwrite.py','_NativeLease._native_call'),
        'projection':('fault_injection/deployments/postwrite_coverage.py','ProjectionCoverage.projection'),
        'end':('fault_injection/deployments/postwrite_coverage.py','ProjectionCoverage.end_observed_forward')}
    a.check(set(diagnostic['loaded_code_identity'])==set(codefiles), 'EXACT_THREE_READONLY_TRACE_BOUNDARIES')
    for label,(name,qualified) in codefiles.items():
        row=diagnostic['loaded_code_identity'][label];tree=ast.parse(C.under(root,name).read_text(encoding='utf-8'))
        cls,method=qualified.split('.')
        nodes=[f for node in tree.body if isinstance(node,ast.ClassDef) and node.name==cls
               for f in node.body if isinstance(f,ast.FunctionDef) and f.name==method]
        a.check(len(nodes)==1 and row['qualified_name']==qualified and row['file']=='/source/'+name and
                row['first_line']==min([nodes[0].lineno]+[d.lineno for d in nodes[0].decorator_list]) and
                C.sha(row['marshalled_code_sha256']), 'TRACE_CODE_EXACT_SOURCE_LOCATION',label)
    frozen={'dcat_pw_open':'aa413569aca4fd897dad3f4de49dd495d95ff7d1eb9ac4bfa90eb5579eb9abc8',
        'dcat_pw_apply':'e4e7711b4367e02852ebd79abe14b7c35c8d8d85230dc00d2a713288b5225b76',
        'dcat_pw_restore':'f4f049d7c185ed9ecaffe93dcee0277652fc4368cc7f30a0fc115720931a56c3',
        'dcat_pw_release':'6738457833024689e4633bf5843eb4ba9baeca459bd6f8e40e817feb7a02d602',
        'dcat_pw_live':'8dbaef24f2bb34cd0187d6c9e9ec0420387a153533f01ef2b94a20b46030c485',
        'dcat_pw_read_span':'1d66f093dd5e33758de23687e8f0580c546e29b448e2d24c475fd2323591db96'}
    cpp=C.under(root,'fault_injection/native_adapters/postwrite/postwrite.cpp').read_text(encoding='utf-8')
    for name,sha in frozen.items():
        begin=cpp.index('extern "C" int '+name+'(');end=cpp.index('\n}',begin)+2
        a.check(hashlib.sha256(cpp[begin:end].encode()).hexdigest()==sha,'UNCHANGED_ORIGINAL_NATIVE_GUARD_AND_MUTATION_BODY',name)
    a.check(C.digest(binding['model_source']['manifest']) == binding['model_source']['source_sha256'], 'ORIGINAL_MODEL_SOURCE_MANIFEST')
    weights = d['weights']
    a.check(C.digest(weights['manifest']) == weights['weights_sha256'] and weights['manifest']['format'] == 'safetensors' and
            bool(weights['manifest']['files']) and all(C.sha(r['sha256']) and C.positive(r['bytes']) for r in weights['manifest']['files'].values()),
            'ORIGINAL_WEIGHT_FILE_MANIFEST')
    return dict(source_files_rehashed=len(sources),source_manifest_sha256=manifest_sha,source_commit=commit,
                actual_native_dso_rehashed_by_this_reader=False,external_model_files_rehashed=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-root',type=Path,required=True);p.add_argument('--result',default='probe/result.json')
    p.add_argument('--result-sha256',required=True);p.add_argument('--source-root',type=Path,required=True)
    for name in ('source-manifest-sha256','source-commit','native-sha256'): p.add_argument('--'+name,required=True)
    args=p.parse_args(argv);sys.addaudithook(C.deny_runtime);a=C.Audit()
    try:
        path=C.under(args.evidence_root,args.result)
        if not C.sha(args.result_sha256) or C.file_sha(path)!=args.result_sha256:
            raise ValueError('ORIGINAL_RESULT_SHA_MISMATCH')
        d=C.read(path);originals=C.Originals(args.evidence_root)
        host,records=C.load_rpc(originals,d,prefix=Path(args.result).parent.as_posix())
        diagnostic=originals.load(d['diagnostic_original'],mounted=True)
        result=review(d,host,records,originals,diagnostic)
        details=a.section('source_review',source_review,a,d,host,diagnostic,args.source_root,args.source_manifest_sha256,args.source_commit,args.native_sha256)
        result['source_review']=dict(**a.report(),details=details)
        result['consistent']=result['consistent'] and not a.issues
        result['counterfactual_pattern_and_original_joins_consistent'] = result.get('counterfactual_pattern_and_original_joins_consistent',False) and result['consistent']
        result['original_files_rehashed']=len(originals.checked)
    except (ValueError,TypeError,KeyError,OSError,IndexError) as error:
        a.check(False,'ORIGINAL_REVIEW_REFUSED',repr(error));result=a.report()
    result.update(schema='demoncat.output39.refusal_full_originals.review.v1',reviewer_sha256=C.file_sha(__file__),
        input_result_sha256=args.result_sha256,runtime_executed=False,read_only=True,
        independent_helpers={name:C.file_sha(HERE/name) for name in ('review39_refusal_diagnostic.py',
            'review39_group_originals.py','review39_forward_originals.py','group_review_common.py')})
    print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False));return 0 if result['consistent'] else 2


if __name__=='__main__':
    raise SystemExit(main())
