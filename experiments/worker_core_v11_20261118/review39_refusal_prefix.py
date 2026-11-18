"""Independent bounded-session prefix review; an incomplete C2/C3 stays incomplete.

The first saved dispatch decision and the complete call denominator identify
the budget branch. Elapsed time is inferred from the pinned source only when
the call budget cannot have fired; no unrecorded timestamp is invented.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import importlib.util

HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_refusal_originals',HERE/'review39_refusal_originals.py')
R=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(R)
C,G=R.C,R.GROUP


def ledger(a,group,rows,*,natural,worker=None):
    _,p=R.OLD.helpers(R.OLD.CUSTOM_PROFILE);device='npu' if natural else 'cpu'
    coverage=group['coverage'];healthy=group['independent_discovery'];first,second=healthy
    order=p.inventory(a,first,'healthy0',device)
    a.check(p.inventory(a,second,'healthy1',device)==order and first['modules']==second['modules'], 'TWO_COMPLETE_HEALTHY_INVENTORIES')
    ref={tuple(x['key']):x for x in first['calls']};eligible={k for k,x in ref.items() if x['eligible']};count=len(eligible)
    a.check(count==group['eligible_target_count'] and len(order)==group['all_projection_count'], 'ACTUAL_ELIGIBLE_AND_FULL_DENOMINATOR')
    recovered=coverage['recovery']['targets'];leases={C.digest(x['lease']['target']):x['lease'] for x in recovered}
    prototype=recovered[0]['lease']['target'];native_sha=dict(prototype['provider_proof'])['native_library_sha256']
    worker=group['worker'] if worker is None else worker
    a.check(group['worker']==worker and group['scope']==coverage['scope'] and group['trigger']==coverage['trigger']=='once' and
        group['execution_scope']=='INDEPENDENT_TARGET_REPLAY' and group['scope']['evidence_kind']==('natural_worker' if natural else 'cpu_contract') and
        prototype['scope_kind']=='INITIALIZED_WORKER' and prototype['process_id']==worker['pid'] and
        G.process_start_matches(prototype['process_start_identity'],worker,natural=natural) and
        prototype['worker_id']==worker['worker_start_identity'] and prototype['model_instance_id']==worker['model_instance_id'] and
        prototype['rank']==worker['rank'] and prototype['device_type']==device and prototype['logical_device']==(0 if natural else None),
        'EXACT_PARTIAL_GROUP_WORKER_LIFETIME_AND_SCOPE')
    targets={}
    for key,row in ref.items():
        target=p.reconstructed_target(row,prototype,group['scope'],native_sha)
        target['provider_identity']='39.output_proxy:worker-core-group-v1';targets[key]=target
    ids={k:C.digest(v) for k,v in targets.items()};by_id={v:k for k,v in ids.items()};planned={ids[k] for k in eligible}
    a.check(set(leases)==planned and len(leases)==len(recovered)==count and
        all(leases[ids[k]]['target']==targets[k] for k in eligible), 'COMPLETE_ORIGINAL_TARGET_IDENTITIES_REBUILT')
    source_rows={r['actual_internal_request_id']:r for r in rows};forwards=group['forward_artifacts'];base=forwards[0]['output_sha256']
    a.check(forwards[0]['phase']=='lossless' and forwards[-1]['phase']=='recovery_retry' and
            sum(f.get('complete') is False for f in forwards)==1, 'ONE_FAILED_FORWARD_WITH_LOSSLESS_AND_RECOVERY')
    inventories={};completed=[];failed=[]
    for f in forwards:
        fid=f['forward_id'];inv=f['inventory'];calls=inv['calls'];inventories[fid]={tuple(c['key']):c for c in calls}
        if f.get('complete') is False:
            failed.append(f)
            a.check(f['phase']=='fault' and f['request_id'] not in source_rows and inv['dropped_calls']==0 and
                [tuple(c['key']) for c in calls]==order[:len(calls)], 'FAILED_FORWARD_HAS_ACTUAL_ORDERED_PREFIX')
        else:
            row=source_rows[f['request_id']];completed.append(f)
            expected_phase='group_recovery_off' if f['phase']=='recovery_retry' else 'c2_'+f['phase']
            a.check(row['phase']==expected_phase and row['complete_inventory']==inv and
                p.inventory(a,inv,fid,device)==order and inv['modules']==first['modules'], 'COMPLETE_FORWARD_JOINS_ACTUAL_REQUEST')
            if f['phase']!='fault':
                a.check(f['output_sha256']==base and row['output_finite'] is True, 'COMPLETE_CONTROL_BUNDLE_EXACT')
        for key,current in inventories[fid].items():
            a.check(p.geom(current)==p.geom(ref[key]) and current['schema']==ref[key]['schema'], 'ACTUAL_PROJECTION_SIGNATURE')
    lossless={r['target_id']:r for r in group['per_target_lossless']}
    a.check(set(lossless)==planned and len(lossless)==count, 'ALL_TARGET_LOSSLESS_BEFORE_ANY_GROUP_F')
    for tid,item in lossless.items():
        a.check(tuple(item['key'])==by_id[tid] and C.sha(item['before']) and item['before']==item['after'] and
            item['original_calls']==1 and item['temporary_readback_verified'] is item['released'] is True,
            'ORIGINAL_TARGET_LOSSLESS_VALUE',tid)
        G.check_native_receipt(a,item['native_lease_receipt'],inventories[forwards[0]['forward_id']][by_id[tid]]['output_geometry'],
            group['native_owner'],item['native_generation'],False,natural=natural,where=tid)
    effects={e['call_id']:e for e in group['execution_artifacts']};native={(n['actual_request_id'],n['target_id']):n for n in group['native_lease_receipts']}
    a.check(len(effects)==len(group['execution_artifacts'])==len(native)==len(group['native_lease_receipts']), 'ALL_EFFECTS_HAVE_ORIGINAL_NATIVE_RECEIPTS')
    expected=[(f,tuple(c['key'])) for f in forwards if f['phase']=='fault' for c in f['inventory']['calls']]
    calls=coverage['calls'];a.check(len(calls)==len(expected), 'FULL_LEDGER_INCLUDES_FAILED_PREFIX_AND_NONTARGETS')
    attempted=set();previous=None;active=0;stopped=False;first_stop=None;tickets=[];applied=[]
    for sequence,(call,(f,key)) in enumerate(zip(calls,expected)):
        fid=f['forward_id'];tid=ids[key];decision=call['decision'];receipt=call['receipt']
        if previous!=fid:active=0;previous=fid
        natural_reason='NON_TARGET' if key not in eligible else 'TARGET_ALREADY_ATTEMPTED' if tid in attempted else 'CONCURRENCY_BUDGET_EXHAUSTED' if active else 'ADMITTED'
        if stopped:reason='SESSION_STOPPED'
        elif decision['reason']=='SESSION_BUDGET_EXHAUSTED':
            reason='SESSION_BUDGET_EXHAUSTED';stopped=True
            first_stop=dict(sequence=sequence,call_id=call['call_id'],phase=call['phase'],operator=call['operator'],
                module_path=ref[key]['module_path'],counted_non_off_calls=sequence+1,
                max_calls=coverage['budget']['max_calls'],max_seconds=coverage['budget']['max_seconds'],
                call_limit_exceeded=sequence+1>coverage['budget']['max_calls'],eligible=key in eligible)
        else:reason=natural_reason
        admitted=reason=='ADMITTED';phase='POST_FIRST_ANOMALY' if active else 'PRE_FAULT_PROPAGATION'
        ticket=dict(observer_id=coverage['discovery']['observer_id'],sequence=sequence,eligible=key in eligible,
            propagation_added=False,call=dict(target=targets[key],call_id=fid+':'+str(order.index(key)+1),
                forward_context=fid,request_ids=[f['request_id']],phase=phase));tickets.append(ticket)
        a.check(call['target_id']==tid and call['call_id']==ticket['call']['call_id'] and call['request_ids']==[f['request_id']] and
            call['worker']==p.worker(prototype) and call['owner']==coverage['owner'] and call['generation']==coverage['generation'] and
            call['eligible'] is (key in eligible) and call['phase']==phase and call['propagation_added'] is False,
            'FULL_CALL_TARGET_REQUEST_SCOPE',call['call_id'])
        a.check(decision['ticket']==ticket and decision['session_id']==coverage['session_id'] and decision['lease']==leases.get(tid) and
            decision['admitted'] is admitted and decision['reason']==reason and receipt['decision']==decision and
            receipt['original_calls']==1 and receipt['native_applied'] is admitted and receipt['outcome']==(
                'HIT' if admitted else 'RESOURCE_LIMITED' if reason=='CONCURRENCY_BUDGET_EXHAUSTED' else 'PASSTHROUGH'),
            'DISPATCH_REASON_REBUILT_WITH_RECORDED_FIRST_STOP',call['call_id'])
        if not admitted:
            a.check(call['call_id'] not in effects and receipt['execution_sha256'] is receipt['effect_sha256'] is None,
                'NO_NATIVE_EFFECT_ON_UNADMITTED_CALL',call['call_id']);continue
        attempted.add(tid);active+=1;effect=effects[call['call_id']];applied.append(effect)
        a.check(f.get('complete') is not False and all(effect[k] is True for k in ('application_attempted','application_returned','native_applied','readback_exact','native_recovered')) and
            effect['target_id']==tid and effect['request_id']==f['request_id'] and effect['forward_id']==fid and
            effect['original_calls']==effect['selected_elements']==effect['changed_elements']==1 and
            effect['original_output_sha256']==lossless[tid]['before']!=effect['mutated_output_sha256'] and
            effect['full_model_output_sha256']==f['output_sha256'] and effect['native_generation']==source_rows[f['request_id']]['native_generation'] and
            receipt['execution_sha256']==C.digest(effect) and receipt['effect_sha256']==effect['mutated_output_sha256'],
            'ACTUAL_INDEPENDENT_TARGET_EFFECT_AND_REFERENCE',call['call_id'])
        actual=inventories[fid][key];geom=actual['output_geometry'];proof=actual['producer_proof']
        a.check(proof['returned_tensor']==geom and proof['original_return_storage'] is True and p.producer_geometry(proof)==geom['shape'],
            'EXACT_ACTUAL_WRITER_OUTPUT',call['call_id'])
        p.consumers(a,effect['consumer_observations'],geom,ref[key]['selected_indices'],effect['consumers'],call['call_id'],device)
        n=native[f['request_id'],tid]
        a.check(n['forward_id']==fid and tuple(n['target_key'])==key, 'NATIVE_JOINS_ORIGINAL_EFFECT',call['call_id'])
        G.check_native_receipt(a,n['receipt'],geom,group['native_owner'],effect['native_generation'],True,natural=natural,where=call['call_id'])
    a.check(first_stop is not None and first_stop['sequence']<len(calls) and len(failed)==1 and
        calls[-1]['forward_id']==failed[0]['forward_id'] and calls[-1]['eligible'] is True and
        calls[-1]['decision']['reason'] in ('SESSION_BUDGET_EXHAUSTED','SESSION_STOPPED'), 'ACTUAL_FIRST_BUDGET_STOP_AND_ELIGIBLE_ABORT')
    a.check(coverage['budget']==dict(max_calls=count*len(order),max_concurrency=1,max_seconds=1200), 'FROZEN_SESSION_BUDGET')
    a.check(len(applied)==len(attempted)==len(effects)==coverage['hits']==coverage['planned_targets_hit'] and
        len(attempted)<count and coverage['planned_targets']==count and coverage['observed_calls']==len(calls) and
        not coverage['missing_outcomes'] and not coverage['propagation_added_calls'] and
        coverage['observed_eligible_calls']==sum(c['eligible'] for c in calls), 'INCOMPLETE_C2_COUNTS_REBUILT_WITHOUT_PROMOTION')
    a.check(coverage['live_c2_proof']['issued']==[] and coverage['live_c2_proof']['consumed'] is None,
            'NO_LIVE_C2_PROOF_ISSUED_OR_CONSUMED')
    observation=dict(discovery=healthy,forwards=forwards,tickets=tickets,native_receipts=group['native_lease_receipts'])
    a.check(coverage['observation']['artifact_sha256']==C.digest(observation) and
        coverage['observation']['calls_sha256']==C.digest(tickets) and coverage['observation']['dropped_calls']==0,
        'PARTIAL_OBSERVATION_REFERENCES_REBUILT')
    if first_stop:
        first_stop['cause_interpretation']='CALL_LIMIT_POSSIBLE_TIME_ALSO_POSSIBLE' if first_stop['call_limit_exceeded'] else 'ELAPSED_LIMIT_BY_PINNED_SOURCE_BRANCH_ELIMINATION'
        first_stop['exact_elapsed_seconds_observed']=False
    return dict(all_projections=len(order),eligible_targets=count,lossless_targets=len(lossless),
        completed_independent_fault_targets=len(attempted),remaining_targets=count-len(attempted),
        ledger_calls=len(calls),first_stop=first_stop,completed_fault_forwards=sum(f['phase']=='fault' for f in completed),
        failed_fault_forwards=len(failed),native_fault_leases=len(native),
        successful_target_modules=[ref[by_id[tid]]['module_path'] for tid in sorted(attempted)],
        untested_target_modules=[ref[by_id[tid]]['module_path'] for tid in sorted(planned-attempted)])


def diagnostic_prefix(a,d,diag,final):
    rows=diag['records'];a.check([(r['arm'],r['target']) for r in rows]==[
        ('healthy','refused'),('single_predecessor_fault','predecessor'),('single_predecessor_fault','refused')],
        'EXACT_THREE_OBSERVED_RECORDS_NO_CUMULATIVE_INVENTION')
    healthy,pred,downstream=rows;group=final['finished_groups']['c2']
    a.check([r['sequence'] for r in rows]==[1,2,3] and diag['errors']==[] and diag['plan_sha256']==C.digest(diag['plan']),
            'DIAGNOSTIC_ORIGINAL_PREFIX')
    for row in rows:
        b=row['binding'];span=row['span'];rid=b['request_id']
        forward=next(f for f in group['forward_artifacts'] if f['request_id']==rid)
        call=next(c for c in forward['inventory']['calls'] if c['key']==b['target_key'])
        request=next(r for r in final['actual_requests'] if r['actual_internal_request_id']==rid)
        a.check(b['forward_id']==forward['forward_id'] and b['geometry']==call['output_geometry'] and b['producer_proof']==call['producer_proof'] and
            row['consumer_observations']==call['consumer_observations'] and b['request_mapping']['snapshot']==request['snapshot'],
            'PREFIX_DIAGNOSTIC_EXACT_PRODUCER_REQUEST_CONSUMER')
        a.check(span['elements']==86016 and span['bytes']==172032 if b['geometry']['shape']==[14,6144] else
            span['elements']*2==span['bytes'], 'DIAGNOSTIC_COMPLETE_SPAN_SIZE')
        a.check(span['full_span'] is True and span['raw_payload_saved'] is False and span['bytes']==b['geometry']['logical_bytes'] and
            C.sha(span['sha256']) and sum(span['counts'].values())==span['elements'] and
            span['nonfinite_elements']==span['elements']-span['counts']['finite'], 'DIAGNOSTIC_FULL_BF16_COUNTS')
        if row['target']=='predecessor':
            native=[n for n in group['native_lease_receipts'] if n['actual_request_id']==rid]
            a.check(len(native)==1 and native[0]['target_key']==b['target_key'] and row['origin']=='SAME_NATIVE_LEASE_ORIGINAL' and
                row['apply_event']==native[0]['receipt']['applications'][0] and row['open_event']==native[0]['receipt']['open'] and
                row['apply_event']['return_code']==0 and row['apply_event']['changed_elements']==1,
                'SINGLE_PREDECESSOR_ONLY_ONE_ACTUAL_NATIVE_WRITE')
            effect=next(e for e in group['execution_artifacts'] if e['request_id']==rid)
            a.check(effect['original_output_sha256']==R.span_tensor_digest(row),'PREFIX_NATIVE_ORIGINAL_FULL_SPAN_DIGEST')
        elif row['arm']=='healthy':
            value=next(v for v in group['per_target_lossless'] if v['key']==b['target_key'])
            a.check(row['origin']=='ACTUAL_FRESH_READ_SPAN' and R.span_tensor_digest(row)==value['before']==value['after'],
                'HEALTHY_FRESH_SPAN_LOSSLESS_ORIGINAL_BYTES')
        else:a.check(row['origin']=='ACTUAL_FRESH_READ_SPAN','DOWNSTREAM_FRESH_CONTROL_NOT_NATIVE_ORIGINAL')
    a.check(healthy['span']['nonfinite_elements']==pred['span']['nonfinite_elements']==0 and downstream['span']['nonfinite_elements']>0 and
        healthy['binding']['target_key']==downstream['binding']['target_key'] and
        pred['binding']['request_id']==downstream['binding']['request_id']!=healthy['binding']['request_id'] and
        len({r['binding']['request_mapping']['input_sha256'] for r in rows})==1,'OBSERVED_SINGLE_PREDECESSOR_PROPAGATION_ONLY')
    return dict(observations=3,healthy_nonfinite=healthy['span']['nonfinite_elements'],
        predecessor_original_nonfinite=pred['span']['nonfinite_elements'],downstream_fresh_nonfinite=downstream['span']['nonfinite_elements'],
        downstream_first_nonfinite=downstream['span']['first_nonfinite'],cumulative_original_observed=False,
        native_apply4_reproduced=False,source84_same_original_causal_chain_closed=False)


def review(d,host,records,originals,diag,*,natural=True):
    a=C.Audit();details={}
    a.check(d['schema']=='demoncat.output39.fixed_c3_refusal_probe.v1' and d['stage']=='c2_c3' and
        d['status']=='REFUSED_OR_FAILED' and all(d[k] is False for k in ('c2','c3','e5','qualification_granted')),
        'ORIGINAL_FAILED_ENTRY_NO_NEW_AUTHORITY')
    partition=a.section('helpers',R.helper_partition,a,d,host,records,diag,natural=natural)
    if partition is None:return a.report()
    context=a.section('original_protocol',C.protocol,a,d,host,partition,'39.output_proxy',natural=natural)
    if context is None:return a.report()
    joined=a.section('original_prefix',G.original_join,a,originals,d,host,context)
    C.final_cleanup(a,d,host,context)
    if not joined:return a.report()
    rows,snapshots=joined;final=snapshots[-1]
    a.check(set(final['finished_groups'])=={'c2'},'C3_NOT_STARTED_OR_IMPORTED_FROM_ANOTHER_RUN')
    G.check_completed_requests(a,d,rows,natural=natural)
    c,p=R.OLD.helpers(R.OLD.CUSTOM_PROFILE)
    a.section('initial_six',G.single_prerequisite,a,c,p,rows,final['module_aliases'],host['worker'],natural=natural)
    group=final['finished_groups']['c2']
    a.check(group['coverage']['owner']==context['provider_owner'] and group['coverage']['generation']==context['ticket']['generation'],
            'PARTIAL_GROUP_AND_PUBLIC_OWNER_GENERATION_JOIN')
    details['prefix']=a.section('partial_ledger',ledger,a,group,rows,natural=natural,worker=host['worker'])
    # A derived view selects only declared complete original inventories; the
    # actual failed partial remains independently audited above, never repaired.
    complete_view=dict(group,forward_artifacts=[f for f in group['forward_artifacts'] if f.get('complete') is not False])
    details['complete_frames']=a.section('actual_complete_frames',G.FORWARD.check,a,C,complete_view,rows,host['worker'],final['module_aliases'],
        natural=natural,model_source=d.get('binding',{}).get('model_source'))
    expected=2+group['eligible_target_count']+len(group['native_lease_receipts'])
    all_saved=a.section('all_original_refs',R.all_originals,a,originals,d,records)
    universe=a.section('all_native_events',R.native_universe,a,all_saved or snapshots,final,natural=natural,expected_unique=expected)
    if universe:
        details['native_universe']=universe[1]
        a.check(all(r['events'][1]['return_code']==0 for r in universe[0].values()), 'NO_NATIVE_APPLY4_IN_THIS_ATTEMPT')
    details['three_observations']=a.section('actual_three_observations',diagnostic_prefix,a,d,diag,final)
    budget=dict(normal_requests=len(d['requests']),recovery_requests=2,maximum_normal_requests=11+group['eligible_target_count'],
        maximum_recovery_requests=2,cumulative_started=False,installed=True,closed=True)
    a.section('partial_group_recovery',R.recovery,a,d,host,diag,final,natural=natural,stage='c2',expected_budget=budget)
    fault_count=sum(r['phase']=='c2_fault' for r in d['requests'])
    a.check([r['phase'] for r in d['requests']]==G.PREFIX+['c2_lossless']+['c2_fault']*fault_count and
        len(d['requests'])==len(rows)-1 and fault_count==len(group['native_lease_receipts'])+1 and
        d['requests'][-1]['mapping']['internal_request_id'] in d['owned_request_abort'] and
        'original_observations' not in d['requests'][-1], 'ACTUAL_PARTIAL_PHASE_AND_ABORT_DENOMINATOR')
    if natural:
        removals=[r for r in records if r['method']=='experiments.c2c3_postwrite_20261117.natural39_worker_core_probe.remove_binding']
        a.check(len(removals)==1 and removals[0]['args']==[host['worker']] and removals[0].get('responses')==d.get('binding_removal') and
            records.index(removals[0])>max(i for i,r in enumerate(records) if C.rpc_action(r)[0]=='detach'),
            'ACTUAL_BINDING_REMOVAL_RPC_AFTER_PUBLIC_DETACH')
    return dict(**a.report(),details=details,classification='CONSISTENT_INCOMPLETE_C2_BUDGET_STOP' if not a.issues else 'UNRESOLVED_PARTIAL_ORIGINALS',
        original_error=d['error'],c2_completed=False,c3_started=False,c3_completed=False,native_apply4_reproduced=False,
        complete_counterfactual_chain=False,first_stop_elapsed_is_source_bound_inference=True,
        source_review_required=True,native_build_review_required=True)


def metadata_review(a,root,source,d,expected_source_manifest):
    """Rehash copied originals; external SDK/binary hashes stay runtime receipts."""
    path=C.under(root,'metadata_bundle_manifest.json');manifest=C.read(path)
    files=manifest['files'];by_path={r['path']:r for r in files};total=0
    a.check(len(files)==len(by_path) and manifest['no_tensor_or_binary_files'] is True,
            'COMPLETE_PRIVATE_METADATA_FILE_SET')
    for name,r in by_path.items():
        p=C.under(root,name);size=p.stat().st_size;total+=size
        a.check(C.sha(r['sha256']) and C.file_sha(p)==r['sha256'] and type(r['bytes']) is int and size==r['bytes'],
                'BUNDLE_ORIGINAL_FILE_REHASH',name)
    a.check(total==manifest['total_original_bytes'],'TOTAL_PRIVATE_METADATA_BYTES')
    def one(suffix):
        found=[r for r in files if r['path'].endswith(suffix)]
        if len(found)!=1:raise ValueError('EXACT_ONE_BUILD_METADATA_REQUIRED:'+suffix)
        return C.read(C.under(root,found[0]['path'])),found[0]
    inputs,iref=one('/inputs.json');receipt,rref=one('/receipt.json');log,lref=one('/build.log')
    build_key=C.digest(inputs['inputs'])
    a.check(build_key==inputs['build_key']==receipt['build_key']==log['build_key']==d['build_fingerprint'] and
        receipt['inputs_sha256']==iref['sha256'] and all(log[k]==v for k,v in receipt.items()), 'EXACT_CANONICAL_BUILD_INPUTS_AND_RECEIPTS')
    build_controller,_=one('native-v11-postwrite39-source101-a1/controller.json')
    a.check(build_controller['source_manifest_sha256']==expected_source_manifest and
        build_controller['returncode']==0 and build_controller['final_state']['ExitCode']==0 and
        build_controller['final_state']['OOMKilled'] is False and build_controller['devices']==[] and
        build_controller['actual_npu_calls']==0 and build_controller['cleanup'] is True and
        build_controller['log_sha256']==lref['sha256'], 'DEVICE_FREE_BUILD_COMPLETED_WITH_SOURCE_AND_LOG_PIN')
    build_sources=inputs['inputs']['sources']
    for name,sha in build_sources.items():
        a.check(C.file_sha(C.under(source,name))==sha,'EXACT_COMPILER_SOURCE_FILE_BYTES',name)
    key='native/libdcat_postwrite.so';image=receipt['images'][key]
    a.check(receipt['artifacts'][key]==image['sha256']==d['native_sha256']==manifest['native_library_sha256'] and
        C.positive(image['bytes']) and receipt['native_qualified'] is False and receipt['actual_npu_calls']==0,
        'BUILD_ARTIFACT_MATCHES_EXPLICIT_ACTUAL_LOADED_DSO_RECEIPT')
    controller,cref=one('v11-output39-refusal-source101-card5-a1/controller.json')
    cases,kref=one('/reviewed_cases.json');case=controller['cases'][0]
    a.check(len(cases)==len(controller['cases'])==1 and controller['case_manifest_sha256']==kref['sha256'] and
        case['review']==cases[0]['review'] and controller['status']=='CASE_FAILED' and case['exit_code']==2 and
        case['container_state']['ExitCode']==2 and case['container_state']['OOMKilled'] is False and
        case['cleanup']['status']=='OWNED_CONTAINER_REMOVED' and case['post_cleanup_gate']['blockers']==[],
        'ACTUAL_FAILURE_EXIT_AND_CONTAINER_CLEANUP_NOT_RELABELED_SUCCESS')
    a.check(case['image_id'] in build_controller['command'] and case['review']['source_manifest_sha256']==expected_source_manifest and
        case['review']['build_fingerprint']==build_key and cases[0]['timeout']==1800 and
        case['review']['resource_budget']['timeout_seconds']==1800 and
        case['review']['resource_budget']['soft_fault_window_seconds']==1500 and
        '/source/'+R.SELF in str(cases[0]['command']), 'SAME_IMAGE_FIXED_ENTRY_AND_UNEXTENDED_OUTER_BUDGET')
    return dict(metadata_files_rehashed=len(files),metadata_bytes_rehashed=total,
        bundle_manifest_sha256=C.file_sha(path),build_sources_rehashed=len(build_sources),build_key=build_key,
        native_sha256=image['sha256'],native_bytes_recorded=image['bytes'],native_elf_build_id_recorded=image['elf_build_id'],
        build_receipt_sha256=rref['sha256'],build_inputs_sha256=iref['sha256'],runtime_controller_sha256=cref['sha256'],
        image_id=case['image_id'],sdk_version_recorded=inputs['inputs']['installation']['fields']['version'],
        compiler_version_recorded=inputs['inputs']['toolchain']['g++']['version'],
        header_sha_records=len(inputs['inputs']['headers']),
        sdk_compiler_binary_and_library_bytes_rehashed_locally=False,
        dso_bytes_rehashed_locally=False,private_arrays_downloaded_or_published=False,
        same_process_recovery_basis='SEPARATE_ORIGINAL_RPC_NATIVE_AND_MODEL_REVIEW',
        container_cleanup_is_not_model_recovery=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-root',type=Path,required=True);p.add_argument('--result-sha256',required=True)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--metadata-bundle-root',type=Path)
    for name in ('source-manifest-sha256','source-commit','native-sha256'):p.add_argument('--'+name,required=True)
    args=p.parse_args(argv);sys.addaudithook(C.deny_runtime);a=C.Audit()
    try:
        path=C.under(args.evidence_root,'probe/result.json')
        if C.file_sha(path)!=args.result_sha256:raise ValueError('ORIGINAL_RESULT_SHA_MISMATCH')
        d=C.read(path);originals=C.Originals(args.evidence_root);host,records=C.load_rpc(originals,d)
        diag=originals.load(d['diagnostic_original'],mounted=True)
        result=review(d,host,records,originals,diag)
        source=a.section('source',R.source_review,a,d,host,diag,args.source_root,args.source_manifest_sha256,args.source_commit,args.native_sha256)
        result['source_review']=dict(**a.report(),details=source);result['consistent']=result['consistent'] and not a.issues
        if args.metadata_bundle_root:
            b=C.Audit();details=b.section('metadata_chain',metadata_review,b,args.metadata_bundle_root,args.source_root,d,args.source_manifest_sha256)
            result['build_and_controller_metadata_review']=dict(**b.report(),details=details)
            result['consistent']=result['consistent'] and not b.issues
        result['original_files_rehashed']=len(originals.checked)
    except (ValueError,TypeError,KeyError,OSError,IndexError) as e:
        a.check(False,'PARTIAL_ORIGINAL_REVIEW_REFUSED',repr(e));result=a.report()
    result.update(schema='demoncat.output39.refusal_prefix.review.v1',reviewer_sha256=C.file_sha(__file__),input_result_sha256=args.result_sha256,
        native_qualification_granted=False,c2=False,c3=False,e5=False,read_only=True,runtime_executed=False,
        helper_sha256={name:C.file_sha(HERE/name) for name in ('review39_refusal_originals.py','review39_group_originals.py',
            'review39_refusal_diagnostic.py','review39_forward_originals.py','group_review_common.py')})
    print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False));return 0 if result['consistent'] else 2


if __name__=='__main__':raise SystemExit(main())
