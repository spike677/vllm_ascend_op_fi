"""Independent saved-original output39 Worker C2/C3 audit; never a capability.

Inputs are the actual public RPC files and immutable group originals. No probe
status, contract boolean, historical model-local result or hit total is used as
the acceptance oracle. New private tensor/statistics files are not published.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys


def support():
    import importlib.util
    path = Path(__file__).with_name('group_review_common.py')
    spec = importlib.util.spec_from_file_location('_group_common', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    old = module.pure_module(Path(__file__).parents[1] / 'c2c3_postwrite_20261117/review39_worker_core_result.py', '_old39')
    return module, old


C, OLD = support()
FORWARD = C.pure_module(Path(__file__).with_name('review39_forward_originals.py'), '_frames39')
PREFIX = ['observe', 'observe_repeat', 'baseline', 'fault', 'off', 'health']
EXTRA_MODULES = ['fault_injection/deployments/'+n+'.py' for n in
    ('postwrite_coverage', 'postwrite_structure', 'postwrite_evidence')] + [
    'fault_injection/'+n+'.py' for n in ('coverage', 'coverage_session', 'runtime_discovery', 'environment')] + [
    OLD.CORE+n+'.py' for n in ('serving39_output_group', 'serving39_output_discovery', 'providers/fault39_output_group')]


def process_start_matches(target_start, worker, *, natural):
    """Join the saved CPU representation without weakening a real Worker join.

    Older CPU originals record /proc start ticks or a CPU-contract stamp.
    Current producers retain the exact Worker proc or process_local identity.
    Natural mode requires a complete matching proc identity, never a CPU form.
    """
    worker_start=worker.get('process_start_identity');pid=worker.get('pid')
    if not isinstance(target_start,str) or not isinstance(worker_start,str) or not C.positive(pid):
        return False
    linux=re.fullmatch(r'proc:([^:]+):([1-9][0-9]*):([0-9]+)',worker_start)
    valid_linux=linux is not None and int(linux[2])==pid
    if natural:
        return valid_linux and target_start==worker_start
    local=re.fullmatch(r'process_local:([1-9][0-9]*):([0-9a-f]{32})',worker_start)
    if local is not None and int(local[1])==pid and target_start==worker_start:
        return True
    fallback=re.fullmatch(r'cpu-contract:([1-9][0-9]*):([0-9]+)',target_start)
    if fallback is not None:
        return int(fallback[1])==pid
    return valid_linux and target_start in (worker_start,linux[3])


def native_identity_guards(a, d, host, expected):
    """An explicit external pin extends the group reader, not the old C1 oracle."""
    a.check(C.sha(expected) and d.get('native_sha256') == expected, 'EXPLICIT_NATIVE_LIBRARY_PIN')
    a.check(d.get('schema') == 'demoncat.output39.worker_core_probe.v1' and
        d.get('mechanism') == '39.output_proxy' and d.get('standard_worker') is True and
        d.get('stage') in ('c2_once','c2_c3') and d.get('status') in
            ('STANDARD_WORKER_OUTPUT39_GROUP_EVIDENCE_READY_FOR_REVIEW','REFUSED_OR_FAILED'),
        'ACTUAL_GROUP_ENTRY_SCHEMA_STAGE_AND_STATUS')
    binding = d.get('binding')
    a.check(isinstance(binding, dict), 'ORIGINAL_GROUP_DEPLOYMENT_REQUIRED')
    if not isinstance(binding, dict):
        return
    value = binding['binding']; native = value['native']
    a.check(binding['worker'] == host['worker'] and value['mechanism'] == '39.output_proxy' and
        value['status'] == 'SERVING_DEPLOYMENT_BOUND_NOT_QUALIFIED' and all(
            type(value[k]) is int and value[k] == 0 for k in ('workers_created','hooks_installed','native_leases_opened')) and
        all(value[k] is False for k in ('qualification_granted','c2','c3')),
        'TYPED_INERT_GROUP_DEPLOYMENT_NOT_ACCEPTANCE')
    a.check(type(native['abi']) is int and native['abi'] == 1 and native['sha256'] == expected and
        native['loaded_mapping_verified'] is True and native['read_span_available'] is True and
        native['simulation'] is False and native['qualification_granted'] is False,
        'EXACT_PINNED_NONSIMULATED_NATIVE_ABI_AND_MAPPING')


def provenance(d, host, final, source_root, manifest_sha, commit, native_library, *, profile,
               native_sha256=None):
    c, p = OLD.helpers(profile)
    saved_modules, saved_sources, saved_native_sha = OLD.MODULES, OLD.SOURCES, OLD.NATIVE_SHA
    expected_native = saved_native_sha if native_sha256 is None else native_sha256
    guard = C.Audit(); native_identity_guards(guard, d, host, expected_native)
    # Extend the *reviewer's expected file set*, never alter source or receipts.
    # Preserve every legacy source/file/loaded-mapping comparison. The new
    # expected DSO comes only from an explicit caller pin, never the probe field.
    try:
        OLD.MODULES = saved_modules + EXTRA_MODULES
        OLD.SOURCES = saved_sources + EXTRA_MODULES
        OLD.NATIVE_SHA = expected_native
        result = OLD.provenance(c, d, host, source_root, manifest_sha, commit, native_library, profile=profile)
    finally:
        OLD.MODULES, OLD.SOURCES, OLD.NATIVE_SHA = saved_modules, saved_sources, saved_native_sha
    a = C.Audit(); names = {Path(name).name: name for name in d['sources']}
    pins = [OLD.CORE+n+'.py' for n in ('serving39_output_group', 'providers/fault39_output_group',
        'serving39_output', 'serving39_output_discovery', 'providers/fault39_output', 'gemm_lineage', 'call_binding', 'identity')] + [
        'fault_injection/deployments/'+n+'.py' for n in ('postwrite_coverage', 'postwrite', 'postwrite_boundaries',
            'postwrite_npu_consumers', 'postwrite_observed', 'postwrite_inventory', 'postwrite_structure')] + ['fault_injection/environment.py']
    expected = {Path(name).name: d['sources'][name] for name in pins}
    for label, group in final['finished_groups'].items():
        a.check(group['source_pins'] == expected and group['scope']['implementation_sha256'] == C.digest(expected),
                'GROUP_IMPLEMENTATION_BINDS_ACTUAL_LOADED_SOURCE', label)
        for row in group['coverage']['recovery']['targets']:
            a.check(dict(row['lease']['target']['provider_proof'])['native_library_sha256'] == d['native_sha256'],
                    'GROUP_TARGET_BINDS_REHASHED_NATIVE_DSO', label)
    return dict(consistent=result['consistent'] and not a.issues and not guard.issues, loaded_source_review=result,
                native_identity_review=guard.report(), expected_native_sha256=expected_native,
                native_pin_basis='EXPLICIT_REVIEW_ARGUMENT' if native_sha256 is not None else 'LEGACY_C1_DEFAULT_PIN',
                group_source_review=a.report(), source_manifest_sha256=manifest_sha,
                limits=['External installed model and weight files remain original runtime SHA receipts; not rehashed by this audit.',
                        'A build fingerprint string is not an independently rehashed compiler/build receipt.'])


def single_prerequisite(a, c, p, rows, aliases, worker, *, natural):
    """Independently inspect the initial six requests, not their success flag."""
    device = 'npu' if natural else 'cpu'; first, second = [x['complete_inventory'] for x in rows[:2]]
    order = p.inventory(a, first, 'initial:0', device)
    a.check(p.inventory(a, second, 'initial:1', device) == order and first['modules'] == second['modules'],
            'INITIAL_COMPLETE_PROJECTION_DENOMINATOR')
    eligible = [x for x in first['calls'] if x['eligible']]; selected = eligible[0]
    model_id = next(x['object_identity'] for x in aliases if x['path'] == '')
    for i, row in enumerate(rows[:6]):
        if i in (0,1):
            a.check(row['report']['intervention'] is None, 'INITIAL_DISCOVERY_NO_APPLICATION')
        if i not in (2,3,4):
            continue
        inventory, report = row['complete_inventory'], row['report']
        p.inventory(a, inventory, 'initial:'+row['phase'], device)
        actual = [x for x in inventory['calls'] if tuple(x['key']) == tuple(selected['key'])]
        a.check(len(actual) == 1 and actual[0]['eligible'], 'INITIAL_SAME_ELIGIBLE_TARGET')
        OLD.target(c,p,a,report,actual[0],row['native_owner'],row['native_generation'],worker,model_id,device,row['phase'])
        if i == 4:
            a.check(report['intervention'] is None and report.get('native_lease_receipt') is None, 'INITIAL_OFF_NO_APPLICATION')
        else:
            item=report['intervention']; fault=i==3
            a.check(item['original_executed'] is item['readback_exact'] is True and
                    item['selected_elements']==1 and item['changed_elements']==item['new_fault_hits']==int(fault),
                    'INITIAL_NATIVE_VALUE_EFFECT')
            check_native_receipt(a, report.get('native_lease_receipt'), actual[0]['output_geometry'],row['native_owner'],
                row['native_generation'],fault,natural=natural,where='initial:'+row['phase'])
        a.check(report['recovery_confirmed'] is True and (i == 3 or report['exact_baseline_output'] is True),
                'INITIAL_LOSSLESS_OFF_RESTORED')


def original_join(a, originals, d, host, context):
    rows, collected, prior = [], [], []
    states = context['states']
    for request in d.get('requests', []) + d.get('recovery_requests', []):
        index = len(prior)
        rid = request['mapping']['internal_request_id']
        queries = [x for x in states if x['raw']['_reference'] == request.get('worker_observation_receipt')]
        armed = [x for x in states if x['action'] == 'advance' and
                 C.rpc_action(x['raw'])[1].get('settings') == {'request_id': rid}]
        if 'original_observations' not in request:
            a.check(bool(d.get('error')), 'MISSING_COMPLETE_REQUEST_ORIGINAL', rid)
            continue
        ref = request['original_observations']; raw = originals.load(ref, mounted=True)
        a.check(raw['schema'] == 'demoncat.output39.worker_group_original.v1' and raw['worker'] == host['worker'] and
                raw['rpc_owner'] == context['provider_owner'] and raw['rpc_generation'] == context['ticket']['generation'] and
                raw['qualification_granted'] is False and raw['action'] == ref['action'] == 'collect', 'ORIGINAL_GROUP_SCOPE', rid)
        a.check(len(raw['actual_requests']) == index + 1 == ref['actual_requests'] and
                raw['actual_requests'][:-1] == prior, 'IMMUTABLE_COMPLETE_REQUEST_PREFIX', rid)
        a.check(len(armed) == len(queries) == 1 and armed[0]['index'] < queries[0]['index'] and
                armed[0]['state']['serving']['expected'] == rid and
                len(armed[0]['state']['evidence']['actual_requests']) == index and
                len(queries[0]['state']['evidence']['actual_requests']) == index + 1, 'RPC_ARM_DOES_NOT_EXECUTE_FORWARD', rid)
        following = [x for x in states if x['index'] > queries[0]['index'] and x['action'] == 'advance'] if queries else []
        a.check(following and C.rpc_action(following[0]['raw'])[1].get('settings') == {'collect': True} and
                following[0]['state']['evidence']['group_originals'][-1] == ref and
                following[0]['state']['evidence']['collection_pending'] is False, 'EXPLICIT_COLLECT_AFTER_REAL_QUERY', rid)
        prior = raw['actual_requests']; collected.append(raw); rows.append(raw['actual_requests'][-1])
    return rows, collected


def check_native_receipt(a, receipt, geom, owner, generation, fault, *, natural, where):
    if not natural and receipt is None:
        return
    # Check original ABI events directly. Do not synthesize intervention or
    # readback fields merely to fit the earlier single-site report schema.
    a.check(receipt['schema'] == 'demoncat.postwrite.lease_receipt.v1' and receipt['owner'] == owner and
            receipt['generation'] == generation and C.positive(receipt['handle']) and C.positive(receipt['thread']) and
            receipt['geometry'] == geom and type(receipt['element_offset']) is int and receipt['element_offset'] == 0 and
            type(receipt['elements']) is int and receipt['elements'] == math.prod(geom['shape']) and
            receipt['closed'] is True and receipt['first_failure'] is None, 'ACTUAL_NATIVE_SPAN_OWNER_CLOSE', where)
    events = receipt['events']; opened = receipt['open']; applied = receipt['applications'][0]
    a.check([x['operation'] for x in events] == ['open', 'apply', 'restore', 'release', 'live'] and
            [x['sequence'] for x in events] == list(range(1,6)) and opened == events[0] and
            receipt['applications'] == [events[1]] and receipt['restore'] == events[2] and
            receipt['release'] == events[3] and receipt['live_readbacks'] == [events[4]], 'EXACT_NATIVE_EVENT_SEQUENCE', where)
    for event in events:
        a.check(event['handle'] == receipt['handle'] and event['owner'] == owner and event['generation'] == generation and
                event['returned'] is True and type(event['return_code']) is int and event['return_code'] == 0 and
                event['error'] is None, 'ACTUAL_NATIVE_RETURN', where)
    device, sep, index = geom['device'].rpartition(':')
    actual_device = (device == 'npu' and sep == ':' and index.isdecimal() and type(opened['device']) is int and
                     opened['device'] == int(index)) if natural else (geom['device'] == 'cpu' and opened['device'] is None)
    a.check(actual_device and opened['data_ptr'] == geom['data_ptr'] and
            opened['bytes'] == geom['logical_bytes'] == geom['storage_bytes'] == receipt['elements']*2,
            'EXACT_NATIVE_DEVICE_AND_STORAGE', where)
    a.check(type(applied['mode']) is int and applied['mode'] == int(fault) and applied['indices'] == [0] and
            type(applied['changed_elements']) is int and applied['changed_elements'] == int(fault) and
            events[-1]['native_live'] is False, 'NATIVE_EFFECT_AND_RELEASE_READBACK', where)


def check_group(a, p, group, name, original_rows, worker, *, natural, aliases=None,
                model_source=None, direct_cpu_profile=None):
    coverage = group['coverage']; count = group['eligible_target_count']; device = 'npu' if natural else 'cpu'
    a.check(C.positive(count) and count <= 256 and group['worker'] == worker and
            group['scope'] == coverage['scope'] and group['trigger'] == coverage['trigger'] ==
            ('once' if name == 'c2' else 'every_match') and group['qualification_granted'] is False and
            group['serializable_authority'] is False and group['failed_attempt'] is None, 'GROUP_SCOPE_AND_FAILURE', name)
    a.check(group['execution_scope'] == ('INDEPENDENT_TARGET_REPLAY' if name == 'c2' else
            'SAME_REQUEST_EVERY_MATCH_MULTIPLE_LIVE_LEASES'), 'DISTINCT_C2_C3_EXECUTION_SCOPE', name)
    a.check(group['scope']['evidence_kind'] == ('natural_worker' if natural else 'cpu_contract') and
            coverage['budget']['max_concurrency'] == (1 if name == 'c2' else count), 'EXACT_EVIDENCE_SCOPE_AND_BUDGET', name)
    healthy = group['independent_discovery']; a.check(len(healthy) == 2, 'TWO_ORIGINAL_DISCOVERY_FORWARDS', name)
    first, second = healthy; order = p.inventory(a, first, name + ':healthy0', device)
    a.check(p.inventory(a, second, name + ':healthy1', device) == order and first['modules'] == second['modules'] and
            first['operator_counts'] == second['operator_counts'], 'HEALTHY_COMPLETE_UNIVERSE_MATCH', name)
    ref = {tuple(x['key']): x for x in first['calls']}; eligible = {k for k, x in ref.items() if x['eligible']}
    a.check(len(eligible) == count and len(order) == group['all_projection_count'], 'COMPLETE_GROUP_DENOMINATOR', name)
    for row in second['calls']:
        old = ref[tuple(row['key'])]
        a.check(p.geom(row) == p.geom(old) and row['eligible'] is old['eligible'] and
                row['selected_indices'] == old['selected_indices'] and row['schema'] == old['schema'], 'HEALTHY_SIGNATURE_DRIFT', name)
    diagnostics = group['discovery']; structural = diagnostics['structural_noncall_inventory']
    actual_modules = {x['object_identity'] for x in first['modules']}
    # Old unit fixtures without direct frames can supply their single-path
    # aliases. A bounded helper proof always needs the full held alias census.
    if aliases is None and not any(x.get('bounded_noncall_proofs') for x in healthy):
        aliases = [dict(path=x['path'], object_identity=x['object_identity']) for x in first['modules']+structural]
    direct = a.section(name+':actual_frames', FORWARD.check, a, C, group, original_rows, worker, aliases,
        natural=natural, model_source=model_source, cpu_profile=direct_cpu_profile)
    a.check(diagnostics['actual_module_calls'] == first['modules'] and
            diagnostics['structural_modules'] == len(actual_modules | {x['object_identity'] for x in structural}) and
            diagnostics['independent_forwards'] == healthy, 'FULL_STRUCTURAL_AND_ACTUAL_UNIVERSE', name)
    for row in structural:
        container = (row['reason'] == 'EXACT_MODULELIST_UNIMPLEMENTED_FORWARD_WITH_OBSERVED_MEMBERS' and
            row['class_name'] == 'torch.nn.modules.container.ModuleList' and row['parent_identity'] in actual_modules and
            bool(row['direct_children']) and set(row['direct_children']).issubset(actual_modules))
        borrowed = (row['reason'] == 'ACTUAL_LOGITS_ARGUMENT_PARAMETER_OWNER_NOT_MODULE_FORWARD' and
            bool(row.get('actual_parameter_consumption')) and all(x in group['actual_parameter_consumption'][0]
                and x['module_object_identity'] == row['object_identity'] and x['parameter_name'] == 'weight' and
                x['original_returned'] is True and x['module_forward_observed'] is False
                for x in row.get('actual_parameter_consumption', [])))
        a.check(row['structural_noncall'] is True and row['actual_forward_calls'] == 0 and
                row['object_identity'] not in actual_modules and (container or borrowed or
                    direct is not None and row['object_identity'] in direct['rotary_children']), 'STRUCTURAL_NONCALL_PROOF', name)
    a.check(len(group['actual_parameter_consumption']) == 2 and group['actual_parameter_consumption'][0] ==
            group['actual_parameter_consumption'][1], 'BORROWED_PARAMETER_REPEAT_EXACT', name)
    recovered = coverage['recovery']['targets']; leases = {C.digest(x['lease']['target']): x['lease'] for x in recovered}
    a.check(len(leases) == len(recovered) == count, 'RECOVERY_TARGET_DENOMINATOR', name)
    prototype = recovered[0]['lease']['target']
    a.check(prototype['scope_kind'] == 'INITIALIZED_WORKER' and prototype['worker_id'] == worker['worker_start_identity'] and
            prototype['process_id'] == worker['pid'] and process_start_matches(
                prototype['process_start_identity'],worker,natural=natural) and
            prototype['model_instance_id'] == worker['model_instance_id'] and prototype['device_type'] == device and
            prototype['logical_device'] == (0 if natural else None), 'ACTUAL_WORKER_TARGET_LIFETIME', name)
    native_sha = dict(prototype['provider_proof'])['native_library_sha256']
    targets = {}
    for key, row in ref.items():
        target = p.reconstructed_target(row, prototype, group['scope'], native_sha)
        target['provider_identity'] = '39.output_proxy:worker-core-group-v1'
        targets[key] = target
    ids = {k: C.digest(x) for k, x in targets.items()}; by_id = {v: k for k, v in ids.items()}; planned = {ids[k] for k in eligible}
    a.check(set(leases) == planned and all(leases[ids[k]]['target'] == targets[k] for k in eligible) and
            coverage['discovery']['observed_target_ids'] == sorted(ids.values()), 'ALL_TARGET_IDENTITIES_REBUILT', name)
    candidate_rows = [dict(candidate_id='projection:' + C.digest(row['key']), identity=targets[k], eligibility=dict(
        status='SUPPORTED' if k in eligible else 'UNSUPPORTED', reasons=[] if k in eligible else
        [row['rejection'] or 'NO_SELECTED_ELEMENT_VALUE_CONSUMER'])) for k, row in ref.items()]
    seen = {x['module_object_identity'] for x in ref.values()}
    candidate_rows += [dict(candidate_id='module:' + str(x['object_identity']), identity=None, eligibility=dict(
        status='UNSUPPORTED', reasons=['COMPLETE_ACTUAL_MODULE_CALL_HAS_NO_ADAPTED_PROJECTION']))
        for x in first['modules'] if x['object_identity'] not in seen]
    candidate_rows += [dict(candidate_id='structural-noncall:' + str(x['object_identity']), identity=None,
        eligibility=dict(status='UNSUPPORTED', reasons=[x['reason']])) for x in structural]
    candidate_report = dict(candidates=candidate_rows, discovery_complete=True, eligible=count, discovered=len(candidate_rows),
        eligibility_counts={s: sum(x['eligibility']['status'] == s for x in candidate_rows) for s in
            ('SUPPORTED', 'UNSUPPORTED', 'UNVERIFIED', 'AMBIGUOUS', 'NOT_OBSERVED', 'RESOURCE_LIMITED')}, execution_authorized=False)
    a.check(diagnostics['candidates'] == candidate_report and coverage['discovery']['candidates_sha256'] == C.digest(candidate_report),
            'CANDIDATE_SET_NOT_A_TRUNCATED_SUCCESS_SUBSET', name)
    a.check(coverage['discovery']['artifact_sha256'] == C.digest(dict(forwards=healthy,
            structural_noncall_inventory=[structural, structural])), 'DISCOVERY_REFERENCE_HASH', name)
    forwards = group['forward_artifacts']; repeat = count if name == 'c2' else 2
    a.check([x['phase'] for x in forwards] == ['lossless'] + ['fault'] * repeat + ['off'], 'ACTUAL_GROUP_PHASE_SEQUENCE', name)
    source_rows = {x['actual_internal_request_id']: x for x in original_rows}; base = forwards[0]['output_sha256']
    frames, inventories = {}, {}
    for index, frame in enumerate(forwards):
        fid = frame['forward_id']; rid = frame['request_id']; row = source_rows[rid]
        a.check(fid not in frames and row['phase'] == name + '_' + frame['phase'] and
                C.sha(row['output_sha256']) and C.sha(frame['output_sha256']) and row['complete_inventory'] == frame['inventory'],
                'ACTUAL_GROUP_FORWARD_REQUEST_JOIN', rid)
        frames[fid] = frame; inv = frame['inventory']; inventories[fid] = {tuple(x['key']): x for x in inv['calls']}
        a.check(p.inventory(a, inv, rid, device) == order and inv['modules'] == first['modules'], 'NO_HIDDEN_PROJECTION_CALL', rid)
        for key, current in inventories[fid].items():
            a.check(p.geom(current) == p.geom(ref[key]) and current['schema'] == ref[key]['schema'], 'ACTUAL_TARGET_GEOMETRY_CHANGED', rid)
            if frame['phase'] != 'fault':
                a.check(current['eligible'] is ref[key]['eligible'], 'OFF_LOSSLESS_ELIGIBLE_DENOMINATOR', rid)
        if frame['phase'] != 'fault':
            a.check(frame['output_sha256'] == base and row['output_finite'] is True, 'LOSSLESS_OFF_FULL_LOGITS_EXACT', rid)
    lossless = {x['target_id']: x for x in group['per_target_lossless']}
    a.check(len(lossless) == len(group['per_target_lossless']) == count and set(lossless) == planned, 'ALL_TARGET_LOSSLESS', name)
    for tid, item in lossless.items():
        a.check(tuple(item['key']) == by_id[tid] and C.sha(item['before']) and item['before'] == item['after'] and
                item['original_calls'] == 1 and item['temporary_readback_verified'] is True and item['released'] is True,
                'COMPLETE_LOSSLESS_VALUE_AND_RELEASE', tid)
        geom = inventories[forwards[0]['forward_id']][by_id[tid]]['output_geometry']
        check_native_receipt(a, item['native_lease_receipt'], geom, group['native_owner'], item['native_generation'],
                             False, natural=natural, where=tid)
    executions = {x['call_id']: x for x in group['execution_artifacts']}
    a.check(len(executions) == len(group['execution_artifacts']) == count * (1 if name == 'c2' else 2), 'EVERY_HIT_HAS_ONE_EFFECT', name)
    expected_calls = [(frame, key) for frame in forwards[1:] for key in order]
    calls = coverage['calls']; a.check(len(calls) == len(expected_calls), 'FULL_LEDGER_NOT_ONLY_HITS', name)
    attempted = set(); active = 0; previous = None; hits = 0; tickets = []; issued_worker = p.worker(prototype)
    native_receipts = {(x['actual_request_id'], x['target_id']): x for x in group['native_lease_receipts']}
    a.check(len(native_receipts) == len(group['native_lease_receipts']) == len(executions), 'NATIVE_LEASE_RECEIPT_DENOMINATOR', name)
    for sequence, (call, (frame, key)) in enumerate(zip(calls, expected_calls)):
        fid = frame['forward_id']; tid = ids[key]; off = frame['phase'] == 'off'; decision = call['decision']; receipt = call['receipt']
        if fid != previous:
            active = 0; previous = fid
        reason = ('OFF_RECOVERY' if off else 'NON_TARGET' if key not in eligible else
                  'TARGET_ALREADY_ATTEMPTED' if name == 'c2' and tid in attempted else
                  'CONCURRENCY_BUDGET_EXHAUSTED' if active >= coverage['budget']['max_concurrency'] else 'ADMITTED')
        admitted = reason == 'ADMITTED'; phase = 'OFF_RECOVERY' if off else 'POST_FIRST_ANOMALY' if active else 'PRE_FAULT_PROPAGATION'
        a.check(call['target_id'] == tid and call['call_id'] == fid + ':' + str(order.index(key)+1) and
                call['forward_id'] == fid and call['request_ids'] == [frame['request_id']] and
                call['worker'] == issued_worker and call['owner'] == coverage['owner'] and call['generation'] == coverage['generation'] and
                call['eligible'] is (key in eligible) and call['phase'] == phase and call['propagation_added'] is False,
                'CALL_ACTUAL_TARGET_REQUEST_OWNER', call['call_id'])
        ticket = dict(observer_id=coverage['discovery']['observer_id'], sequence=sequence, eligible=key in eligible,
            propagation_added=False, call=dict(target=targets[key], call_id=call['call_id'], forward_context=fid,
                                              request_ids=[frame['request_id']], phase=phase))
        tickets.append(ticket)
        a.check(decision['ticket'] == ticket and decision['session_id'] == coverage['session_id'] and
                decision['lease'] == leases.get(tid) and decision['admitted'] is admitted and decision['reason'] == reason and
                receipt['decision'] == decision and receipt['original_calls'] == 1 and receipt['native_applied'] is admitted and
                receipt['outcome'] == ('HIT' if admitted else 'RESOURCE_LIMITED' if reason == 'CONCURRENCY_BUDGET_EXHAUSTED' else 'PASSTHROUGH'),
                'INDEPENDENT_ADMISSION_AND_OUTCOME_RECOMPUTED', call['call_id'])
        if not admitted:
            a.check(call['call_id'] not in executions and receipt['execution_sha256'] is receipt['effect_sha256'] is None,
                    'NON_TARGET_OR_OFF_NOT_INJECTED', call['call_id'])
            continue
        active += 1; hits += 1; attempted.add(tid); effect = executions[call['call_id']]
        a.check(all(effect[k] is True for k in ('application_attempted', 'application_returned', 'native_applied',
                'readback_exact', 'native_recovered')) and effect['baseline_control'] is False and
                effect['target_id'] == tid and effect['forward_id'] == fid and effect['request_id'] == frame['request_id'] and
                effect['original_calls'] == effect['selected_elements'] == effect['changed_elements'] == 1 and
                effect['full_model_output_sha256'] == frame['output_sha256'] and
                effect['native_generation'] == source_rows[frame['request_id']]['native_generation'] and
                C.sha(effect['original_output_sha256']) and C.sha(effect['mutated_output_sha256']) and
                effect['original_output_sha256'] != effect['mutated_output_sha256'], 'PER_TARGET_NATIVE_EFFECT_AND_RECOVERY', call['call_id'])
        if name == 'c2':
            a.check(effect['original_output_sha256'] == lossless[tid]['before'], 'C2_INDEPENDENT_TARGET_REFERENCE', call['call_id'])
        a.check(receipt['execution_sha256'] == C.digest(effect) and receipt['effect_sha256'] == effect['mutated_output_sha256'],
                'EFFECT_REFERENCES_EXACT_ORIGINAL_OBJECT', call['call_id'])
        actual = inventories[fid][key]; geom = actual['output_geometry']; proof = actual['producer_proof']
        a.check(proof['returned_tensor'] == geom and proof['original_return_storage'] is True and
                proof['expected_output_shape'] == p.producer_geometry(proof) == geom['shape'], 'ACTUAL_WRITER_RETURN_NOT_CLONE', call['call_id'])
        p.consumers(a, effect['consumer_observations'], geom, ref[key]['selected_indices'], effect['consumers'], call['call_id'], device)
        native = native_receipts[(frame['request_id'], tid)]
        # A CPU ABI double can exercise the real lease receipt path. Its full
        # device='cpu'/device=None events are checked below; never accept that
        # representation as natural NPU evidence.
        native_kind = native['explicit_cpu_mock'] is False if natural else (
            native['explicit_cpu_mock'] is True or native['explicit_cpu_mock'] is False and
            isinstance(native.get('receipt'), dict) and native['receipt']['geometry']['device'] == 'cpu')
        a.check(native['forward_id'] == fid and tuple(native['target_key']) == key and native_kind,
                'NATIVE_LEASE_TO_ACTUAL_TARGET_JOIN', call['call_id'])
        check_native_receipt(a, native['receipt'], geom, group['native_owner'], effect['native_generation'], True,
                             natural=natural, where=call['call_id'])
    a.check(hits == coverage['hits'] == count * (1 if name == 'c2' else 2) and coverage['planned_targets_hit'] == count and
            coverage['planned_targets'] == count and coverage['observed_calls'] == len(calls) and
            coverage['observed_eligible_calls'] == repeat * count and not coverage['missing_outcomes'] and
            not coverage['propagation_added_calls'], 'RECOMPUTED_COVERAGE_TOTALS', name)
    observation = dict(discovery=healthy, forwards=forwards, tickets=tickets, native_receipts=group['native_lease_receipts'])
    a.check(coverage['observation']['complete'] is True and coverage['observation']['dropped_calls'] == 0 and
            coverage['observation']['calls_sha256'] == C.digest(tickets) and
            coverage['observation']['artifact_sha256'] == C.digest(observation), 'EXACT_INDEPENDENT_OBSERVATION_REFERENCE', name)
    recovery = coverage['recovery']; a.check(recovery['state'] == 'DISABLED_CONFIRMED' and recovery['inflight'] == 0 and
            not any(recovery[k] for k in ('held_leases', 'pending_drains', 'pending_leases', 'errors')), 'GROUP_RECOVERY_ZERO', name)
    for row in recovered:
        tid = C.digest(row['lease']['target']); proof = dict(target_id=tid, off=base, native_live_leases=0,
            off_targets={x: v['before'] for x, v in lossless.items()})
        a.check(row['lease']['owner'] == coverage['owner'] and row['lease']['generation'] == coverage['generation'] and
                row['drained'] is row['released'] is row['off_exact'] is True and row['pending_reason'] is None and
                row['off_output_sha256'] == base and row['proof_sha256'] == C.digest(proof), 'ALL_TARGET_OFF_REFERENCE', tid)
    health = group['actual_health']; original_health = source_rows[health['actual_request_id']]
    a.check(original_health['phase'] == name + '_health' and health['output_sha256'] == base and
            original_health['output_sha256'] == source_rows[forwards[0]['request_id']]['output_sha256'] and
            health['snapshot'] == original_health['snapshot'] and len(recovery['workers']) == 1, 'REAL_POST_RELEASE_HEALTH_REQUEST', name)
    for row in recovery['workers']:
        a.check(row['worker'] == issued_worker and row['owner'] == coverage['owner'] and row['generation'] == coverage['generation'] and
                row['proof_sha256'] == C.digest(health) and row['baseline_output_sha256'] == row['off_output_sha256'] == base and
                all(type(row[k]) is int and row[k] == 0 for k in
                    ('active_leases', 'device_bytes', 'host_bytes', 'inflight_calls', 'owned_hooks')), 'REAL_HEALTH_ZERO_RESOURCES', name)
    return dict(eligible=count, projections=len(order), ledger_calls=len(calls), recomputed_hits=hits,
                fault_forwards=repeat, native_receipts=len(native_receipts), contract_boolean_used_as_verdict=False,
                direct_discovery=None if direct is None else {k:v for k,v in direct.items() if k!='rotary_children'})


def check_live_proof(a, groups):
    """An issuance/use audit remains historical even when the subsequent C3 fails."""
    if 'c2' not in groups:
        return None
    b = groups['c2']['coverage']; issued = b['live_c2_proof']['issued']
    a.check(len(issued) == 1 and b['live_c2_proof']['consumed'] is None, 'ONE_C2_ORIGINAL_ISSUANCE')
    if len(issued) != 1:
        return None
    entry = issued[0]; proof = entry['proof']; stripped = {k:v for k,v in b.items() if k!='live_c2_proof'}
    a.check(entry['proof_sha256'] == C.digest(proof) and entry['exact_issued_object_verified'] is True and
        entry['qualification_granted'] is entry['serializable_authority'] is False and
        all(proof[k] == b[k] for k in ('scope','owner','generation','session_id')) and
        proof['report_sha256'] == C.digest(stripped) and proof['recovery_sha256'] == C.digest(b['recovery']) and
        proof['candidates_sha256'] == b['discovery']['candidates_sha256'] and
        proof['discovery_sha256'] == b['discovery']['artifact_sha256'] and
        proof['observation_sha256'] == b['observation']['artifact_sha256'] and
        proof['target_ids'] == sorted(C.digest(x['lease']['target']) for x in b['recovery']['targets']) and
        entry['object_binding']['pid'] == groups['c2']['worker']['pid'] and
        C.positive(entry['object_binding']['python_object_id']), 'C2_PROOF_BINDS_COMPLETE_ORIGINAL_SCOPE')
    consumed = groups.get('c3', {}).get('coverage', {}).get('live_c2_proof', {}).get('consumed')
    if consumed is not None:
        f = groups['c3']['coverage']; consumer = consumed['consumer']
        a.check(entry['proof'] == consumed['proof'] and entry['proof_sha256'] == consumed['proof_sha256'] and
            entry['object_binding'] == consumed['object_binding'] and consumed['exact_issued_object_verified'] is True and
            consumed['qualification_granted'] is consumed['serializable_authority'] is False and
            b['scope'] == f['scope'] and b['owner'] == f['owner'] and f['generation'] == b['generation']+1 and
            all(consumer[k] == f[k] for k in ('scope','owner','generation','session_id')) and
            consumer['candidates_sha256'] == f['discovery']['candidates_sha256'] == proof['candidates_sha256'] and
            consumer['target_ids'] == proof['target_ids'], 'ORIGINAL_LIVE_C2_OBJECT_CONSUMED_BY_C3')
    return dict(issuance_recorded=True, consumption_recorded=consumed is not None,
                consumption_is_not_c3_completion=True, saved_json_authority=False)


def check_completed_requests(a, d, rows, *, natural):
    drivers = [x for x in d.get('requests', [])+d.get('recovery_requests', []) if 'original_observations' in x]
    a.check(len(rows) == len(drivers), 'COMPLETE_REQUEST_ORIGINAL_DENOMINATOR')
    baseline = rows[0]['output_sha256']; seen = set()
    for row, driver in zip(rows, drivers):
        rid = row['actual_internal_request_id']; snap = row['snapshot']; mapping = driver['mapping']
        a.check(rid not in seen and row['phase'] == driver['phase'] and
            rid == mapping['internal_request_id'] == snap['internal_request_id'] and
            mapping['source'] == 'ACTUAL_PUBLIC_ENQUEUE_AND_OUTPUT_PROCESSOR' and
            driver['output']['request_id'] == mapping['external_request_id'] and row['native_leases'] == 0,
            'ACTUAL_REQUEST_MAPPING_AND_NO_PENDING_NATIVE', rid)
        seen.add(rid)
        if natural:
            tokens = snap['token_ids']; a.check(snap['num_requests'] == 1 and snap['num_computed_tokens'] == 0 and
                snap['num_prompt_tokens'] == snap['num_scheduled_tokens'] == len(tokens) and
                snap['positions'] == list(range(len(tokens))) and snap['token_ids'] == rows[0]['snapshot']['token_ids'] and
                snap['device_input_bytes_equality'] == 'COMPLETE_DEVICE_READBACK_VERIFIED', 'FULL_FRESH_PREFILL_INPUT', rid)
        if 'fault' not in row['phase']:
            a.check(row['output_sha256'] == baseline and row['output_finite'] is True, 'REFERENCE_LOSSLESS_OFF_LOGITS', rid)
    return seen


def check_failed_native(a, d, final):
    """Join a rejected apply to its failed enqueue and original partial call.

    A return code is observed ABI evidence. This reader does not infer the
    selected tensor's numeric value or prove compiler provenance from that code.
    """
    summaries = []; seen = set()
    pending = {x['mapping']['internal_request_id']:x for x in d.get('requests', [])
               if 'original_observations' not in x}
    for label, group in final['finished_groups'].items():
        if not group.get('failed_attempt'):
            continue
        targets = {C.digest(x['lease']['target']):x['lease']['target'] for x in group['coverage']['recovery']['targets']}
        for row in group['native_lease_receipts']:
            receipt = row.get('receipt')
            if receipt is None:
                continue
            rejected = [e for e in receipt['events'] if e.get('return_code') not in (0, None)]
            if not rejected:
                continue
            key = (receipt['owner'], receipt['generation'], receipt['handle'])
            if key in seen:
                continue
            seen.add(key); rid = row['actual_request_id']; driver = pending.get(rid)
            snapshots = [x for x in final['failures'] if x['phase'] == label+'_fault' and
                x.get('partial_inventory') and any(tuple(c['key']) == tuple(row['target_key'])
                    for c in x['partial_inventory']['calls'])]
            a.check(driver is not None and driver['phase'] == label+'_fault' and 'output' not in driver and
                rid in d.get('owned_request_abort', []) and len(snapshots) == 1 and row['target_id'] in targets,
                'REJECTED_NATIVE_APPLY_JOINS_FAILED_ENQUEUE_AND_TARGET', label)
            if len(snapshots) != 1:
                continue
            inv = snapshots[0]['partial_inventory']; calls = [c for c in inv['calls'] if tuple(c['key']) == tuple(row['target_key'])]
            a.check(len(calls) == 1, 'REJECTED_NATIVE_ACTUAL_PROJECTION_UNIQUE', label)
            if len(calls) != 1:
                continue
            target = targets.get(row['target_id'], {})
            a.check(target.get('module_object_identity') == calls[0]['module_object_identity'] == row['target_key'][0] and
                target.get('operator_api') == calls[0]['operator'] == row['target_key'][1] and
                calls[0]['original_calls'] == 1, 'REJECTED_NATIVE_COMPLETE_TARGET_IDENTITY_JOIN', label)
            scope = inv['frame_observer_scope']; events = receipt['events']; applied = receipt['applications']
            a.check(receipt['schema'] == 'demoncat.postwrite.lease_receipt.v1' and
                receipt['owner'] == scope['owner'] == group['native_owner'] and
                receipt['generation'] == scope['native_generation'] and receipt['thread'] == scope['thread'] and
                receipt['geometry'] == calls[0]['output_geometry'] and C.positive(receipt['handle']) and
                receipt['element_offset'] == 0 and receipt['elements'] == math.prod(receipt['geometry']['shape']),
                'REJECTED_NATIVE_EXACT_SPAN_OWNER_GENERATION', label)
            a.check([x['operation'] for x in events] == ['open','apply','restore','release','live'] and
                [x['sequence'] for x in events] == [1,2,3,4,5] and len(rejected) == len(applied) == 1 and
                rejected[0] == applied[0] == receipt['first_failure'] == events[1] and
                receipt['open'] == events[0] and receipt['restore'] == events[2] and receipt['release'] == events[3] and
                receipt['live_readbacks'] == [events[4]], 'REJECTED_APPLY_ORIGINAL_EVENT_SEQUENCE', label)
            for event in events:
                a.check(all(event[k] == receipt[k] for k in ('owner','generation','handle')) and
                    event['returned'] is True and event['error'] is None and type(event['return_code']) is int,
                    'REJECTED_APPLY_ACTUAL_NATIVE_RETURN', label)
            a.check(events[1]['mode'] == 1 and events[1]['indices'] == [0] and
                type(events[1]['return_code']) is int and events[1]['return_code'] != 0 and
                type(events[1]['changed_elements']) is int and events[1]['changed_elements'] == 0 and
                all(events[i]['return_code'] == 0 for i in (0,2,3,4)) and
                events[4]['native_live'] is False and receipt['closed'] is True,
                'REJECTED_APPLICATION_NOT_A_HIT_AND_NATIVE_RELEASED', label)
            summaries.append(dict(stage=label, original_operator=calls[0]['operator'],
                native_apply_return_code=events[1]['return_code'], changed_elements=events[1]['changed_elements'],
                subsequent_restore_release_return_codes=[events[2]['return_code'],events[3]['return_code']],
                final_native_live=events[4]['native_live'], receipt_sha256=C.digest(receipt),
                numeric_original_value_observed=False, compiler_source_binding_inferred=False,
                qualification_granted=False))
    declared = re.search(r'POSTWRITE_APPLY_([0-9]+)', str(d.get('error', '')))
    if declared:
        a.check(any(x['native_apply_return_code'] == int(declared[1]) for x in summaries),
                'NATIVE_APPLY_ERROR_REQUIRES_ORIGINAL_NONZERO_RETURN')
    return dict(unique_rejected_native_receipts=len(summaries), receipts=summaries,
        absence_of_native_rejection_does_not_mean_no_failure=True, qualification_granted=False)


def review(d, host, records, originals, *, natural=True, profile=OLD.CUSTOM_PROFILE,
           direct_cpu_profile=None, frame_model_source=None):
    c, p = OLD.helpers(profile); a = C.Audit(); details = {}
    context = a.section('protocol', C.protocol, a, d, host, records, '39.output_proxy', natural=natural)
    if not context:
        return a.report()
    result = a.section('originals', original_join, a, originals, d, host, context)
    C.final_cleanup(a, d, host, context)
    if not result:
        return a.report()
    rows, snapshots = result; failed = bool(d.get('error') or d.get('recovery_requests'))
    if not snapshots:
        return dict(**a.report(), classification='REFUSAL_BEFORE_COMPLETE_ORIGINAL', original_error=d.get('error'))
    final = snapshots[-1]; groups = final['finished_groups']
    model_source = d.get('binding', {}).get('model_source', frame_model_source)
    seen = a.section('completed_request_join', check_completed_requests, a, d, rows, natural=natural)
    if failed:
        completed = {}
        if 'c2' in groups and groups['c2'].get('failed_attempt') is None:
            count = groups['c2']['eligible_target_count']
            plan = PREFIX + ['c2_lossless'] + ['c2_fault']*count + ['c2_off', 'c2_health']
            if d['stage'] == 'c2_c3':
                plan += ['c3_lossless', 'c3_fault', 'c3_fault', 'c3_off', 'c3_health']
            normal_phases = [x['phase'] for x in d.get('requests', [])]
            recovery_phases = [x['phase'] for x in d.get('recovery_requests', [])]
            a.check(normal_phases == plan[:len(normal_phases)] and recovery_phases in
                ([], ['group_recovery_off'], ['group_recovery_health'], ['group_recovery_off','group_recovery_health']),
                'PARTIAL_ACTUAL_PREFIX_AND_OFF_HEALTH_NOT_FAULT_RETRY')
            details['c2'] = a.section('completed_c2', check_group, a, p, groups['c2'], 'c2', rows,
                host['worker'], natural=natural, aliases=final['module_aliases'], model_source=model_source,
                direct_cpu_profile=direct_cpu_profile)
            a.section('initial_six', single_prerequisite, a,c,p,rows,final['module_aliases'],host['worker'],natural=natural)
            details['live_c2_history'] = a.section('partial_live_proof', check_live_proof, a, groups)
            completed['c2'] = details['c2'] is not None and not a.issues
        a.check(bool(d.get('error')) and bool(final['failure'] or final['failures']), 'PARTIAL_ORIGINAL_FAILURE_PRESERVED')
        details['failed_native'] = a.section('failed_native', check_failed_native, a, d, final)
        if 'c2' in completed:
            completed['c2'] = bool(completed['c2'] and not a.issues)
        return dict(**a.report(), classification='FAILED_OR_PARTIAL_NOT_QUALIFIED', original_error=d.get('error'),
            original_recovery_error=d.get('recovery_error'), completed_requests=len(rows),
            recorded_failure=final['failure'], recorded_failures=final['failures'],
            completed_stage_originals_reconciled=completed, details=details,
            failed_stage_reports={name:g.get('failed_attempt') for name,g in groups.items() if g.get('failed_attempt')},
            group_completion_reconciled=False, originals_rehashed=len(originals.checked),
            limits=['A completed C2 record does not qualify the failed C3.',
                    'Recorded live proof consumption is not successful every-match execution.',
                    'No new live authority or E5 is granted by saved originals.'])
    count = groups['c2']['eligible_target_count']; plan = PREFIX + ['c2_lossless'] + ['c2_fault']*count + ['c2_off', 'c2_health']
    if d['stage'] == 'c2_c3':
        plan += ['c3_lossless', 'c3_fault', 'c3_fault', 'c3_off', 'c3_health']
    a.check([x['phase'] for x in rows] == plan and [x['phase'] for x in d['requests']] == plan and
            context['actions'] == ['attach', 'prepare', 'register', 'check', 'enable'] +
            ['advance', 'query', 'advance']*len(plan) + ['finish', 'disable', 'unregister', 'detach'], 'EXACT_PUBLIC_PHASE_PROTOCOL')
    a.check(final['failure'] is None and not final['failures'], 'ORIGINAL_FAILURES_NOT_ERASED')
    for name in ('c2', 'c3'):
        if name not in groups:
            continue
        details[name] = a.section(name, check_group, a, p, groups[name], name, rows, host['worker'], natural=natural,
            aliases=final['module_aliases'], model_source=model_source, direct_cpu_profile=direct_cpu_profile)
    a.section('initial_six', single_prerequisite, a,c,p,rows,final['module_aliases'],host['worker'],natural=natural)
    details['live_c2_history'] = a.section('live_proof', check_live_proof, a, groups)
    if 'c3' in groups:
        a.check(bool(details['live_c2_history'] and details['live_c2_history']['consumption_recorded']),
                'COMPLETED_C3_REQUIRES_ORIGINAL_LIVE_PROOF_CONSUMPTION')
    health = d.get('post_unregister_health')
    a.check(health and health['worker'] == host['worker'] and health['mapping']['internal_request_id'] not in seen and
            all(health['output'][k] == d['healthy_before'][k] for k in ('tokens', 'text')), 'POST_UNREGISTER_ORDINARY_WORKER_HEALTH')
    return dict(**a.report(), details=details, group_completion_reconciled=not a.issues,
        classification='SAVED_STANDARD_WORKER_GROUP_RECEIPTS' if natural else 'CPU_CONTRACT_ONLY', originals_rehashed=len(originals.checked),
        limits=['No live capability or public qualification is issued.', 'E5 remains separate; full private tensor payloads are not read.',
                'Group output_sha256 hashes the hidden+logits bundle; actual_requests.output_sha256 hashes logits. Each is compared only with its own baseline.',
                'The bundle/logits join uses actual request, inventory and frozen execution source, not equality between different digest domains.',
                'Post-unregister health is token/text; preceding actual health retains full logits hashes.',
                'Actual device operations remain source-bound original receipts, not execution performed by this audit.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path, required=True); parser.add_argument('--result-sha256', required=True)
    parser.add_argument('--result', default='probe/result.json'); parser.add_argument('--profile', default=OLD.CUSTOM_PROFILE,
                        choices=(OLD.LEGACY_PROFILE, OLD.CUSTOM_PROFILE))
    for name in ('source-root', 'native-library'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--native-sha256', help='Explicit reviewed DSO byte pin; omission retains the fixed legacy C1 pin')
    for name in ('source-manifest-sha256', 'source-commit'): parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv); sys.addaudithook(C.deny_runtime)
    a = C.Audit()
    try:
        path = C.under(args.evidence_root, args.result)
        if not C.sha(args.result_sha256) or C.file_sha(path) != args.result_sha256:
            raise ValueError('ORIGINAL_RESULT_SHA_MISMATCH')
        document = C.read(path); originals = C.Originals(args.evidence_root)
        host, records = C.load_rpc(originals, document, prefix=Path(args.result).parent.as_posix())
        result = review(document, host, records, originals, profile=args.profile)
        refs = [x['original_observations'] for x in document.get('requests', []) + document.get('recovery_requests', [])
                if 'original_observations' in x]
        if refs:
            final = originals.load(refs[-1], mounted=True)
            result['provenance'] = provenance(document,host,final,args.source_root,args.source_manifest_sha256,
                args.source_commit,args.native_library,profile=args.profile,native_sha256=args.native_sha256)
            result['consistent'] = result['consistent'] and result['provenance']['consistent']
            if 'completed_stage_originals_reconciled' in result:
                result['completed_stage_originals_reconciled'] = {name:bool(ok and result['consistent'])
                    for name,ok in result['completed_stage_originals_reconciled'].items()}
        else:
            result['consistent'] = False
            result['provenance'] = {'consistent':False,'reason':'NO_COMPLETE_ORIGINAL_TO_BIND'}
        result['group_completion_reconciled'] = bool(result.get('group_completion_reconciled') and result['consistent'])
    except (ValueError, TypeError, KeyError, OSError, IndexError) as error:
        a.check(False, 'ORIGINAL_REVIEW_REFUSED', repr(error)); result = a.report()
    result.update(schema='demoncat.output39.worker_group.independent_originals.v1', reviewer_sha256=C.file_sha(__file__),
        input_result_sha256=args.result_sha256,
        independent_helpers={'review39_forward_originals.py':C.file_sha(Path(__file__).with_name('review39_forward_originals.py'))},
        runtime_executed=False, read_only=True)
    print(json.dumps(result, indent=2, allow_nan=False)); return 0 if result['consistent'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
