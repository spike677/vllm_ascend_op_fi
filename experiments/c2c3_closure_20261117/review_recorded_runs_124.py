"""Independent CPU review of immutable, named 124 receipts; never runs an NPU.

All outputs are fresh sidecars. Canonical exports reconstruct only objects that
the recorded implementation hashed, with exact source pointers and equality to
the referenced digest required. Original receipts/statistics are never changed.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def load(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return file_record(path)


def review_natural18(root, out, audit_saved_coverage, index_artifacts):
    path=root/'natural18-group-a2/natural-group-18/probe/result.json'
    data=load(path); phases=data['all_phases']
    preliminary={name:audit_saved_coverage(data[name], c2_report=data['c2'] if name=='c3' else None)
                 for name in ('c2','c3')}
    needed=set().union(*(set(value['missing_artifact_sha256']) for value in preliminary.values()))
    origins={}; files=[]
    directory=out/'natural18/recorded-canonical-objects'; directory.mkdir(parents=True)
    def export(value, origin):
        encoded=canonical(value); digest=hashlib.sha256(encoded).hexdigest()
        if digest not in needed:return
        if digest not in origins:
            target=directory/(digest+'.json')
            with target.open('xb') as stream:stream.write(encoded)
            files.append(target); origins[digest]=[]
        origins[digest].append(origin)
    export(dict(baseline=phases[1],repeat=phases[2],lossless=phases[3]),
           dict(derivation='tensor_natural.qualify canonical object', baseline='/all_phases/1',
                repeat='/all_phases/2', lossless='/all_phases/3'))
    for name in ('c2','c3'):export(data[name]['phase_rows'],'/'+name+'/phase_rows')
    for i,row in enumerate(phases):
        export(row,'/all_phases/'+str(i))
        for j,event in enumerate((row.get('mechanism') or {}).get('events',[])):
            origin='/all_phases/'+str(i)+'/mechanism/events/'+str(j)
            export(event,origin)
            export(dict(event=event,model_output=row['output']),dict(event=origin,model_output='/all_phases/'+str(i)+'/output'))
    index=index_artifacts(files)
    audits={name:audit_saved_coverage(data[name],artifact_index=index,c2_report=data['c2'] if name=='c3' else None)
            for name in ('c2','c3')}
    for name,result in audits.items():write(out/'natural18'/(name+'-coverage-audit.json'),result)
    manifest=dict(original_receipt=file_record(path),export_semantics='Exact recorded canonical objects only; no events or tensor values invented.',
                  origins=origins,missing_reference_sha256=sorted(needed-set(index)))
    manifest_ref=write(out/'natural18/canonical-export-manifest.json',manifest)
    summaries=[]
    for row in phases:
        independent=row['independent'];events=(row.get('mechanism') or {}).get('events',[])
        summaries.append(dict(phase=row['phase'],input_sha256=sha(row['input']),output=row['output'],
            observed_calls=len(row['calls']),events=len(events),
            interventions=sum(event.get('applied') is True for event in events),
            changed_outputs=sum(event.get('output_changed') is True for event in events),
            consumer_proven=sum(bool(event.get('consumers')) for event in events),
            exact_readback=sum(event.get('return_readback_exact') is True for event in events),
            observer_complete=independent['complete'], dropped_calls=independent['dropped_calls'],
            observer_hooks=independent['observer_hooks'],observer_inflight=independent['inflight'],cleanup=row['cleanup']))
    bare=phases[0]['output']
    healthy=['native_baseline','baseline','baseline_repeat','lossless','off','statistics_baseline','statistics_baseline_repeat','statistics_off']
    healthy_exact=all(row['output']==bare and row['input']==phases[0]['input'] for row in phases if row['phase'] in healthy)
    candidates=data['discovery']['candidate_set']
    summary=dict(original_receipt=file_record(path),status=data['status'],provenance=data['provenance'],runtime=data['runtime'],
        weights=data['weights'],model_weights_sha256=data['model_weights_sha256'],model_source_sha256=data['model_source_sha256'],
        weights_manifest_bound=sha(data['weights'])==data['model_weights_sha256'],
        source_manifest_bound=sha(data['model_sources'])==data['model_source_sha256'],
        same_process_healthy_phases_exact=healthy_exact,candidate_set_counts={k:v for k,v in candidates.items() if k!='candidates'},
        canonical_export=manifest_ref,phases=summaries,
        audits={name:{k:v for k,v in result.items() if k not in ('target_ids','scope')} for name,result in audits.items()},
        recommendation='LIMITED_MODEL_LOCAL_C2_C3_READY_FOR_ROOT_REVIEW',standard_vllm_worker=False,
        qualifications_granted=False)
    write(out/'natural18/summary.json',summary)
    return summary


def review_native39(root,out,audit_saved_coverage,index_artifacts):
    folder=root/'providers39-a5/providers39-qwen-c2c3/c2'
    path=folder/'runtime/serving_receipt.json';data=load(path)
    coverage=data['qualification']['coverage'];files=sorted((folder/'native').glob('*.json'))
    index=index_artifacts(files);audit=audit_saved_coverage(coverage,artifact_index=index)
    audit_ref=write(out/'native39/c2-coverage-audit.json',audit)
    phases=data['native_phases'];healthy=phases[0]['model_output_sha256'];references={}
    summaries=[];errors=[]
    for phase in phases:
        name=phase['phase'];rows=phase['calls'];native=[row for row in rows if row['native_delta']]
        # OpExecCache may be reused across different actual module/call sites.
        # The reviewed target identity includes both site and cache lifetime.
        keys=[(tuple(row['site']),tuple(row['native_delta'][0]['key'])) for row in native]
        if len(keys)!=len(set(keys)):errors.append(name+':DUPLICATE_NATIVE_IDENTITY')
        if not phase['independent_denominator_complete'] or sum(sum(d['count'] for d in row['native_delta']) for row in rows)!=phase['native_denominator']:
            errors.append(name+':INDEPENDENT_DENOMINATOR_MISMATCH')
        if any(row['original_calls']!=1 for row in rows):errors.append(name+':ORIGINAL_CALL_COUNT')
        if name=='discovery_1':discovered=set(keys)
        if set(keys)!=discovered:errors.append(name+':NATIVE_TARGET_SET_CHANGED')
        if name=='reference':references={(tuple(row['site']),tuple(row['native_delta'][0]['key'])):row['native']['last_observation'] for row in native}
        effects=Counter();non_target=len(rows)-len(native)
        for row in native:
            state=row.get('native')
            if state is None:continue
            obs=state['last_observation'];proof=obs['proof'];key=(tuple(row['site']),tuple(row['native_delta'][0]['key']))
            if not all(obs[k]==0 for k in ('pre_sync_status','copy_status','run_status','post_sync_status','free_status')):
                errors.append(name+':NATIVE_EXECUTION_OR_SYNCHRONIZATION_FAILED')
            if state['inflight'] or state['leases']:errors.append(name+':NATIVE_RESOURCES_PENDING')
            if obs['original_storage_sha256_before']!=obs['original_storage_sha256_after']:errors.append(name+':ORIGINAL_STORAGE_CHANGED')
            if proof['api']!='aclnnMatmul' or proof['slots']!=[0,1,2] or not proof['formal_order_proven']:
                errors.append(name+':FORMAL_ORDER_UNPROVEN')
            if name.startswith('lossless') or name=='off_observed':
                if obs['output_storage_sha256']!=references[key]['output_storage_sha256']:
                    errors.append(name+':TARGET_OUTPUT_NOT_RESTORED')
                if obs['original_storage_sha256_before']!=references[key]['original_storage_sha256_before']:
                    errors.append(name+':TARGET_INPUT_CHANGED')
            if name=='fault':
                fields=('input_device_readback_verified','self_constant_positive_verified','per_column_sign_counts_verified',
                        'theoretical_output_exact_zero','amplitude_equal')
                if not all(obs.get(k) is True for k in fields):errors.append(name+':FINITE_CONSTRUCTION_PROOF_MISSING')
                if not all(obs[n]['finite']==obs[n]['total']>0 for n in ('temporary_self','temporary_mat2')):
                    errors.append(name+':SUBSTITUTED_INPUT_NOT_FINITE')
                effects['changed']+=obs['output_storage_sha256']!=references[key]['output_storage_sha256']
                effects['original_input_nonfinite']+=any(obs[n]['finite']!=obs[n]['total'] for n in ('original_self','original_mat2'))
                effects['output_nonfinite']+=obs['output']['finite']!=obs['output']['total']
                if proof.get('input_mask')!=3:errors.append(name+':INPUT_MASK_CHANGED')
                if any((t['original_address']!=t['run_address'])!=(i<2) for i,t in enumerate(proof['tensors'])):
                    errors.append(name+':REPLACEMENT_FORMAL_MISMATCH')
            elif name.startswith('lossless'):
                if not obs.get('input_device_readback_verified') or obs['changed']!=[0,0]:errors.append(name+':LOSSLESS_DEVICE_READBACK_FAILED')
            elif name=='off_observed':
                if obs['changed']!=[0,0] or any(t['original_address']!=t['run_address'] for t in proof['tensors']):
                    errors.append(name+':OFF_SUBSTITUTION_PRESENT')
        if name!='fault' and phase['model_output_sha256']!=healthy:errors.append(name+':FULL_LOGITS_NOT_RESTORED')
        summaries.append(dict(phase=name,public_calls=len(rows),native_eligible_calls=len(native),
            distinct_native_cache_keys=len({key[1] for key in keys}),
            native_denominator=phase['native_denominator'],non_target_public_calls=non_target,
            actual_request_id=phase['request_id'],input_sha256=phase['input_sha256'],
            full_logits_sha256=phase['model_output_sha256'],model_output_finite=phase['model_output_finite'],effects=dict(effects)))
    result=dict(original_receipt=file_record(path),coverage_audit=audit_ref,
        audit_summary={k:v for k,v in audit.items() if k not in ('target_ids','scope')},
        exact_scope=coverage['scope'],phases=summaries,native_checks_passed=not errors,native_check_errors=sorted(set(errors)),
        referenced_native_file_count=len(files),c3=False,c3_failure='LOGITS_HEAD_OR_BIAS_CHANGED before C3 injection',
        recommendation='LIMITED_STANDARD_WORKER_C2_READY_FOR_ROOT_REVIEW',qualifications_granted=False)
    write(out/'native39/summary.json',result)
    return result


def statistics_reviews(root,out,compare_raw_statistics,compact_summary):
    pairs={
        'attention28_legal':(root/'attention28-launch-a3/fia-legal-baseline/run',root/'attention28-rest-a1/fia-legal-fault/run',0,0),
        'attention28_replay':(root/'attention28-rest-a1/fia-replay-baseline/run',root/'attention28-rest-a1/fia-replay-fault/run',0,0),
        'natural18':(root/'natural18-group-a2/natural-group-18/probe/statistics',root/'natural18-group-a2/natural-group-18/probe/statistics',0,2),
        'natural18_repeat':(root/'natural18-group-a2/natural-group-18/probe/statistics',root/'natural18-group-a2/natural-group-18/probe/statistics',0,1),
        'postwrite41':(root/'postwrite41-natural-a2/postwrite41worker-baseline/capture',root/'postwrite41-natural-a2/postwrite41worker-fault/capture',0,0),
        'postwrite39':(root/'postwrite-natural-a1/postwrite39hf-baseline/capture',root/'postwrite-natural-a1/postwrite39hf-fault/capture',0,0),
        'native39':(root/'providers39-a5/providers39-qwen-c2c3/c2/statistics',root/'providers39-a5/providers39-qwen-c2c3/c2/statistics',2,6),
    }
    summaries={}
    for name,(baseline,fault,bs,fs) in pairs.items():
        try:
            report=compare_raw_statistics(baseline=baseline/f'observation/step{bs}/rank0',
                fault=fault/f'observation/step{fs}/rank0',baseline_config=baseline/'config.json',fault_config=fault/'config.json',
                baseline_collector=baseline/'collector_identity.json',fault_collector=fault/'collector_identity.json')
            full=write(out/'statistics'/name/'full.json',report);summary=compact_summary(report)
            summary['full_report']=full;summary['actual_baseline_step']=bs;summary['actual_fault_step']=fs
            write(out/'statistics'/name/'summary.json',summary)
            summaries[name]={k:summary.get(k) for k in ('status','aligned','mismatched','first_exceedance','unobservable_fields_count',
                'incomparable_fields_count','baseline_self_comparable','e5_statistics_comparable','full_dump_inventory','full_report')}
        except Exception as error:
            summaries[name]=dict(status='AUDIT_REFUSED',error=repr(error))
    write(out/'statistics/summary.json',summaries)
    return summaries


def review_postwrite(root,out):
    reports={}
    path=root/'postwrite41-natural-a2/postwrite41worker-fault/result.json';data=load(path);provider=data['provider']
    phases=[];errors=[];reference=provider['history'][0]['model_output_sha256']
    lives=set();hits=0
    for row in provider['history']:
        lives.add((row['pid'],row['process_start'],row['model_instance']))
        interventions=[item['intervention'] for item in row['rows'] if item.get('intervention')]
        facts=[]
        for item in interventions:
            boundary=item['boundary'];meta=boundary['observational_metadata']
            checks=dict(actual_writer_returned=meta['writer']['returned'] is True,
                live_value_storage_matches=meta['value']['storage_identity']==meta['writer']['value_cache']['storage_identity'],
                actual_slot_matches=meta['writer']['slots']==[boundary['slot']],
                legal_selected_span=all(0<=index<boundary['span_elements'] for index in boundary['relative_indices']),
                native_consumer_returned=item['consumer_returned'] is True,
                readback_exact=item['readback_exact'] is True,restored=item['restored'] is True,released=item['released'] is True,
                original_once=item['original_calls']==1)
            if not all(checks.values()):errors.append(row['phase']+':WRITER_CONSUMER_OR_RECOVERY_CHECK')
            facts.append(dict(checks=checks,native_applied=item['native_applied'],selected_elements=item['selected_elements'],
                              changed_elements=item['changed_elements'],layout=meta['layout'],actual_slot=boundary['slot']))
        hits+=row['new_fault_hits']
        if row['errors'] or row['native_live_leases'] or not row['recovery_confirmed']:errors.append(row['phase']+':RECOVERY_FAILED')
        if row['phase']!='fault' and row['model_output_sha256']!=reference:errors.append(row['phase']+':MODEL_OFF_NOT_EXACT')
        phases.append(dict(phase=row['phase'],eligible_observed=len(row['rows']),new_fault_hits=row['new_fault_hits'],
            model_output_sha256=row['model_output_sha256'],model_output_changed=row['model_output_sha256']!=reference,
            exact_healthy_output=row['exact_healthy_output'],interventions=facts))
    if hits!=1:errors.append('EXPECTED_SINGLE_INTERVENTION')
    if len(lives)!=1:errors.append('WORKER_OR_MODEL_LIFETIME_CHANGED')
    if not data['same_worker_health'] or provider['owned_hooks'] or provider['owned_api_patches'] or provider['pending'] is not None:
        errors.append('FINAL_HEALTH_OR_OWNED_RESOURCES_FAILED')
    source_checks={rel:file_record(root/'source-a15'/rel)['sha256']==digest for rel,digest in data['source_sha256'].items()}
    if not all(source_checks.values()):errors.append('EXECUTED_SOURCE_HASH_MISMATCH')
    reports['41.tail_column']=dict(original_receipt=file_record(path),checks_passed=not errors,errors=errors,
        phases=phases,actual_source_checks=source_checks,scope=data['scope'],same_worker_lifetime_count=len(lives),
        healthy_tokens_equal=data['healthy_before']['tokens']==data['healthy_after']['tokens'],
        native_out_of_bounds_claim=False,c2=False,c3=False,
        recommendation='LIMITED_STANDARD_WORKER_C1_READY_FOR_ROOT_REVIEW')
    path=root/'postwrite-natural-a1/postwrite39hf-fault/result.json';data=load(path);phase=data['phase'];hit=phase['intervention']
    checks=dict(original_once=hit['original_executed'] is True,one_native_intervention=phase['new_fault_hits']==1,
        one_changed_element=hit['selected_elements']==hit['changed_elements']==1,readback=hit['readback_exact'] is True,
        real_consumer=bool(phase['actual_target']['consumer_observations']),exact_off=data['exact_off'] is True,
        resources_released=data['live_leases']==0,recovered=phase['recovery_confirmed'] is True)
    reports['39.output_proxy']=dict(original_receipt=file_record(path),checks=checks,checks_passed=all(checks.values()),
        scope=data['scope'],operator=hit['operator'],boundary=hit['boundary'],
        eligible_with_consumers=phase['eligible_with_consumers'],targets_selected=1,
        model_changed_elements=data['changed_elements'],model_nan_elements=data['observed_nan'],
        standard_worker=False,c2=False,c3=False,recommendation='LIMITED_NATURAL_MODEL_C1_READY_FOR_ROOT_REVIEW')
    reports['qualifications_granted']=False
    write(out/'postwrite/summary.json',reports)
    return reports


def review_identities(root,out):
    records={}
    for name,folder in [('native39','providers39-a5'),('natural18','natural18-group-a2'),('postwrite41','postwrite41-natural-a2')]:
        path=root/folder/'controller.json';data=load(path)
        records[name]=dict(controller=file_record(path),cases=[{k:row.get(k) for k in
            ('id','image_id','source_manifests','frozen_scripts','exit_code')} for row in data['cases']])
    data=load(root/'natural18-group-a2/natural-group-18/probe/result.json')
    sources=['src/demoncat/runtime/vllm_ascend/worker_extension/providers/tensor_natural.py',
             'fault_injection/rmsnorm_return.py','fault_injection/index22.py','fault_injection/tensor_observer.py']
    actual={Path(rel).name:file_record(root/'source-a17'/rel)['sha256'] for rel in sources}
    records['natural18'].update(actual_provider_source_sha256=sha(actual),
        matches_recorded_implementation=sha(actual)==data['c2']['scope']['implementation_sha256'])
    weights={name:file_record(Path('/home/gyl/models/Qwen3-0.6B')/name) for name in data['weights']}
    records['natural18']['actual_weights_verified']=all(row['sha256']==data['weights'][name]['sha256']
        and row['bytes']==data['weights'][name]['bytes'] for name,row in weights.items())
    records['natural18']['actual_weight_files']=weights
    installed=root/'packages-a4/installed'
    pairs=[('fault39/src/demoncat_fault39/coverage_adapter.py','demoncat_fault39/coverage_adapter.py'),
           ('src/demoncat/runtime/vllm_ascend/worker_extension/serving39_coverage.py','demoncat/runtime/vllm_ascend/worker_extension/serving39_coverage.py')]
    records['native39']['actual_installed_sources']={rel:dict(source=file_record(root/'source-a18'/rel),
        installed=file_record(installed/target),exact=file_record(root/'source-a18'/rel)['sha256']==file_record(installed/target)['sha256'])
        for rel,target in pairs}
    for name,cases in [('native39','providers39-a5-cases.json'),('postwrite41','postwrite41-natural-a2-cases.json')]:
        manifest=load(root/cases)
        native=next(Path(row[0]) for row in manifest[0]['mounts'] if row[1]=='/native')
        receipt_path=native.parent/'receipt.json';receipt=load(receipt_path)
        artifact_checks={rel:dict(actual=file_record(native.parent/rel),expected=expected,
            matches=file_record(native.parent/rel)['sha256']==expected) for rel,expected in receipt['artifacts'].items()}
        records[name]['native_build']=dict(receipt=file_record(receipt_path),status=receipt['status'],
            build_key=receipt['build_key'],actual_npu_calls_during_build=receipt['actual_npu_calls'],
            artifacts=artifact_checks)
    records['qualifications_granted']=False
    write(out/'identities.json',records)
    return records


def review_worker18(root,out,audit_saved_coverage,index_artifacts):
    path=root/'natural18-worker-a1/natural-worker-rms-group/probe/result.json';data=load(path)
    phases=[item['worker'] for item in data['requests']]
    preliminary={name:audit_saved_coverage(data[name],c2_report=data['c2'] if name=='c3' else None) for name in ('c2','c3')}
    wanted=set().union(*(set(value['missing_artifact_sha256']) for value in preliminary.values()))
    directory=out/'worker18/recorded-canonical-objects';directory.mkdir(parents=True)
    files=[];origins={}
    def export(value,origin):
        encoded=canonical(value);digest=hashlib.sha256(encoded).hexdigest()
        if digest not in wanted:return
        if digest not in origins:
            target=directory/(digest+'.json')
            with target.open('xb') as stream:stream.write(encoded)
            files.append(target);origins[digest]=[]
        origins[digest].append(origin)
    export(dict(baseline=phases[1],repeat=phases[2],lossless=phases[3]),
        dict(derivation='qualify_recorded canonical object',baseline='/requests/1/worker',repeat='/requests/2/worker',lossless='/requests/3/worker'))
    for name in ('c2','c3'):export(data[name]['phase_rows'],'/'+name+'/phase_rows')
    for i,row in enumerate(phases):
        origin='/requests/'+str(i)+'/worker';export(row,origin)
        for j,event in enumerate((row.get('mechanism') or {}).get('events',[])):
            ep=origin+'/mechanism/events/'+str(j);export(event,ep)
            export(dict(event=event,model_output=row['output']),dict(event=ep,model_output=origin+'/output'))
    index=index_artifacts(files)
    audits={name:audit_saved_coverage(data[name],artifact_index=index,c2_report=data['c2'] if name=='c3' else None) for name in ('c2','c3')}
    for name,audit in audits.items():write(out/'worker18'/(name+'-coverage-audit.json'),audit)
    export_ref=write(out/'worker18/canonical-export-manifest.json',dict(original_receipt=file_record(path),origins=origins,
        missing_reference_sha256=sorted(wanted-set(index)),export_semantics='Exact recorded canonical objects only.'))
    summaries=[];errors=[]
    healthy=phases[0]['output'];fault_outputs=[]
    for i,row in enumerate(phases):
        request=data['requests'][i]['mapping']['internal_request_id'];scheduler=row['scheduler']
        if row['request_id']!=request or scheduler['internal_request_id']!=request or scheduler['num_requests']!=1:
            errors.append(row['phase']+':ACTUAL_REQUEST_MAPPING_MISMATCH')
        if row['input']!=phases[0]['input']:errors.append(row['phase']+':ACTUAL_MODEL_INPUT_CHANGED')
        if row['logits_boundary_returned'] is not True:errors.append(row['phase']+':FULL_LOGITS_BOUNDARY_NOT_RETURNED')
        if not row['independent']['complete'] or row['independent']['dropped_calls']:errors.append(row['phase']+':INDEPENDENT_OBSERVER_INCOMPLETE')
        events=(row.get('mechanism') or {}).get('events',[])
        fault='fault' in row['phase']
        applied=[event for event in events if event.get('applied')]
        if fault:
            fault_outputs.append(row['output'])
            if len(applied)!=113 or any(not event.get('original_execution_verified') or not event.get('consumers')
                or not event.get('output_changed') or not event.get('return_readback_exact') for event in applied):
                errors.append(row['phase']+':PER_TARGET_EFFECT_OR_CONSUMER_UNPROVEN')
        elif applied or row['output']!=healthy:errors.append(row['phase']+':HEALTHY_OUTPUT_OR_PASSTHROUGH_CHANGED')
        summaries.append(dict(phase=row['phase'],actual_request_id=request,events=len(events),hits=len(applied),
            full_output=row['output'],observer_complete=row['independent']['complete'],cleanup=row['cleanup']))
    if not all(value==fault_outputs[0] for value in fault_outputs):errors.append('FAULT_REPEATS_NOT_EXACT')
    if (data['healthy_before']!=data['healthy_after'] or data['cleanup']['owned_serving_hooks']
            or data['cleanup']['active_adapter'] or data['cleanup']['independent_observer']):
        errors.append('SAME_WORKER_FINAL_HEALTH_OR_OWNERSHIP')
    source=root/'source-a20'
    names=['src/demoncat/runtime/vllm_ascend/worker_extension/providers/tensor_natural.py',
        'fault_injection/rmsnorm_return.py','fault_injection/index22.py','fault_injection/tensor_observer.py',
        'fault_injection/tensor_callable.py','fault_injection/_scoped_dispatch.py','fault_injection/module_faults.py']
    actual_sha=sha({Path(name).name:file_record(source/name)['sha256'] for name in names})
    if actual_sha!=data['c2']['scope']['implementation_sha256']:errors.append('ACTUAL_PROVIDER_SOURCE_HASH_MISMATCH')
    provenance=data['setup']['provenance']
    if sha(provenance['weights'])!=data['c2']['scope']['model_weights_sha256'] or sha(provenance['model_sources'])!=data['c2']['scope']['model_source_sha256']:
        errors.append('ACTUAL_MODEL_IDENTITY_BINDING_MISMATCH')
    controller=load(root/'natural18-worker-a1/controller.json')
    summary=dict(original_receipt=file_record(path),status=data['status'],scope=data['c2']['scope'],
        image_id=controller['cases'][0]['image_id'],source_manifest=controller['cases'][0]['source_manifests'],
        actual_provider_source_sha256=actual_sha,worker_identity=provenance['worker_identity'],
        actual_checks_passed=not errors,actual_check_errors=errors,canonical_export=export_ref,phases=summaries,
        audits={name:{k:v for k,v in audit.items() if k not in ('target_ids','scope')} for name,audit in audits.items()},
        statistics_status=data['statistics_status'],e5=False,hf_e5_inherited=False,
        recommendation='LIMITED_STANDARD_WORKER_C2_C3_READY_FOR_ROOT_REVIEW',qualifications_granted=False)
    write(out/'worker18/summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--only-worker18',action='store_true')
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.source))
    from fault_injection.coverage_audit import audit_saved_coverage,index_artifacts
    from fault_injection.statistics_audit import compare_raw_statistics,compact_summary
    if args.only_worker18:
        result=review_worker18(args.root,args.out,audit_saved_coverage,index_artifacts)
        print(json.dumps({k:result.get(k) for k in ('status','actual_checks_passed','actual_check_errors','statistics_status')}))
        return
    result={}
    for name,fn in [('native39',review_native39),('natural18',review_natural18),('worker18',review_worker18)]:
        try:result[name]=fn(args.root,args.out,audit_saved_coverage,index_artifacts)
        except Exception as error:result[name]=dict(status='AUDIT_REFUSED',error=repr(error))
    result['statistics']=statistics_reviews(args.root,args.out,compare_raw_statistics,compact_summary)
    for name,fn in [('postwrite',review_postwrite),('identities',review_identities)]:
        try:result[name]=fn(args.root,args.out)
        except Exception as error:result[name]=dict(status='AUDIT_REFUSED',error=repr(error))
    result['audit_implementation']=file_record(Path(__file__))
    result['comparator']=file_record(args.source/'fault_injection/statistics_audit.py')
    result['coverage_auditor']=file_record(args.source/'fault_injection/coverage_audit.py')
    result['qualifications_granted']=False
    write(args.out/'review.json',result)
    print(json.dumps({key:({k:v for k,v in value.items() if k in ('status','error','native_checks_passed')}
        if key!='statistics' else {k:{n:r.get(n) for n in ('status','first_exceedance','e5_statistics_comparable')} for k,r in value.items()})
        for key,value in result.items() if key in ('native39','natural18','worker18','statistics')}))


if __name__=='__main__':main()
