"""Read original all-target isolated C2 bytes; never load an injector or device.

Reuse the normal, public-protocol, raw ABI and coverage consistency readers.
Only one bounded request capture is loaded at a time after the three controls.
Neither a saved normal report nor a provider C2 flag is admission authority.
"""
import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import hashlib
import json
import math
import struct
from pathlib import Path
import sys
from types import SimpleNamespace

import arch22_normal_bridge_evidence as E
import review_arch22_all_normal as A
import review_arch22_normal_bridge as N
import review_arch22_s2_single as S

PHASES=('reference','lossless','qualification_off')+tuple(
    name+f'_{index:02d}' for index in range(28) for name in ('fault','off'))+('cleanup_health',)
need=E.need


def capture_file(run,reference,index,phase):
    expected='/evidence/probe/group_requests/'+f'{index:03d}-'+phase+'/runtime_result.json'
    need(type(reference) is dict and reference.get('path')==expected
        and reference.get('phase')==phase and type(reference.get('bytes')) is int
        and 0<reference['bytes']<=128*1024**2,'GROUP_COMPLETE_BOUNDED_CAPTURE_REQUIRED:'+phase)
    path=S.local(run,reference['path'])
    need(path.stat().st_size==reference['bytes'] and E.sha(path)==reference['sha256'],
         'GROUP_CAPTURE_BYTES_CHANGED:'+phase)
    return E.read(path)


def payloads(run,capture,folder,helper,normal):
    values={};prefix='/evidence/probe/'+folder+'/private/'
    for name,shape in (('hidden',[1,1024]),('logits_hidden',[1,1024]),('logits',[1,151936])):
        values[name]=S.raw_tensor(run,capture[name],shape,normal[name]['dtype'],helper,prefix+name.replace('_','-')+'.bin')
    for index,path in enumerate(sorted(N.LAYERS)):
        for kind in ('output','key','value'):
            row=capture['attention'][path][kind];shape=[1,16,128] if kind=='output' else [1025,8,128]
            values[path+'/'+kind]=S.raw_tensor(run,row,shape,'torch.float16',helper,prefix+f'layer-{index}-{kind}.bin')
    need(set(capture['attention'])==set(N.LAYERS) and len(values)==87,'GROUP_ALL_87_RAW_PAYLOADS_REQUIRED')
    A.actual_output_join(capture)
    return values


def native_call(run,folder,call,saved,ordinal,scope,context,runtime,helper,bundle,symbols,selected):
    receipt=call['native'];host=receipt['host'];token=receipt['call_id'];tile=256 if selected else 512
    need(E.read(E.child(run,folder+'/native/'+token+'.json'))==receipt and receipt['original_calls']==1
        and receipt['module_path']==host['target_path']==call['module_path']
        and receipt['module_object_id']==host['module_object_id']==call['module_object_id']
        and receipt['owner']==host['owner']==context['provider_owner']
        and receipt['generation']==host['generation']==context['generation']
        and receipt['token']==host['token']==token==saved['native_call_id']
        and receipt['target_set_sha256']==host['target_set_sha256']==scope['sha256']
        and receipt['target_ordinal']==host['target_ordinal']==ordinal
        and receipt['selection_applied'] is selected and receipt['normal_only'] is (not selected)
        and all(receipt[k] is False for k in ('api_arguments_changed','cache_repacked','output_mutated')),
        'GROUP_ACTUAL_NATIVE_CALL_OWNER_TARGET_JOIN')
    phase='fault' if selected else 'baseline' if context['phase'] in PHASES[:2] else 'off'
    need(receipt['phase']==host['phase']==phase and host['actual_observer_scope_matched'] is True
        and host['admission_sha256']==receipt['admission_sha256']
        and host['coverage_generation']==receipt['coverage_generation']==int(selected)
        and (selected or host['admission_sha256']=='0'*64),'GROUP_NATIVE_PHASE_AND_ONE_USE_ADMISSION_JOIN')
    thread=N.captured_thread_identity(saved)
    resolved=N.one_launch(run,runtime,token,saved['pid'],helper,native_tid=thread['native_tid'],
        kernel_sha=bundle['completion']['kernel_binary_sha256'],symbols=symbols,block_dim=host['block_dim'])
    launch=resolved['launches'][0]
    decoded=N.final_private_pod(runtime[launch['raw_line']-1],launch,host,bundle,helper.abi,selected_s2=tile)
    helper.arch22_s2_build_evidence.verify_scope_geometry(decoded['typed'],host,bundle['s2_cpu_matrix']['first_decode_rows'][tile][host['aic']])
    need(all(receipt['native'][key]==value for key,value in decoded.items())
        and receipt['native']['kernel_sha256']==launch['registered_binary']['sha256'],
        'GROUP_ORIGINAL_NATIVE_POD_DIFFERS_FROM_RAW')
    return dict(token=token,selected_s2=tile,raw_tiling_sha256=decoded['raw_sha256'],
        full_typed_sha256=E.digest(decoded['typed']),host=host)


def candidate_report(group):
    value=deepcopy(group['candidates']);rows=value['candidates']
    counts={key:0 for key in ('SUPPORTED','UNSUPPORTED','UNVERIFIED','AMBIGUOUS','NOT_OBSERVED','RESOURCE_LIMITED')}
    for row in rows:counts[row['eligibility']['status']]+=1
    value.update(eligible=counts['SUPPORTED'],discovered=len(rows)-counts['NOT_OBSERVED'],
                 eligibility_counts=counts,execution_authorized=False)
    return value


def receipt_signature(signature,*,simulation=False):
    # Match the source-bound receipt encoding after independently checking the
    # complete page map. These labels are not used to bypass any comparison.
    if not simulation:value=S.normalized(signature)
    else:
        # The real CPU fixture has 80 physical pages. This explicit contract
        # path cannot enter review(), whose Native geometry remains 128.
        value=deepcopy(signature);table=value['block_table_values'];cache=value['cache_geometry']
        need(all(cache[k]['shape']==[80,16,8,128] and cache[k]['device']=='cpu' for k in ('key','value'))
            and len(table)==1 and len(table[0])==80 and len(set(table[0][:65]))==65
            and all(type(p) is int and 0<=p<80 for p in table[0])
            and value['writer_consumer_slot']==[table[0][64]*16]
            and value['tensors']['block_table']['bytes']==320
            and value['tensors']['block_table']['sha256']==hashlib.sha256(struct.pack('<80i',*table[0])).hexdigest(),
            'GROUP_CPU_CONTRACT_ACTUAL_PAGE_MAP_CHANGED')
        table[0][:65]=list(range(65))
    value['tensors']['block_table']['sha256']='PROVEN_USED_PAGE_RELOCATION_ONLY'
    value['writer_consumer_slot']=['PROVEN_CURRENT_LOGICAL_POSITION_SLOT']
    return value


def resource_join(saved,context,observations,runtime_path,source,build_sha256,planner):
    """Recalculate all 56 pre-F guards from preserved counters and log bytes."""
    plan=saved['resource_plan'];need(plan==planner.plan(observations,variant=E.GROUP_VARIANT,
        source_sha256=source['manifest_sha256'],build_sha256=build_sha256,case_seconds=plan['case_seconds']),
        'GROUP_RESOURCE_PLAN_NOT_BOUND_TO_FRESH_NORMAL')
    rows=saved['resource_checks'];phases=saved['phase_resources']
    need(len(rows)==56 and len(phases)==60 and [r['phase'] for r in phases]==list(PHASES)
        and [r['phase_index'] for r in phases]==list(range(60)),'GROUP_TWO_RESOURCE_CHECKS_PER_F_AND_60_PHASES_REQUIRED')
    native=context['thread_identity'];thread=[native[k] for k in ('pid','python_thread_ident','native_tid')]
    scope=dict(variant=E.GROUP_VARIANT,source_manifest_sha256=source['manifest_sha256'],completion_sha256=build_sha256,
        normal_evidence_sha256=context['normal_evidence_sha256'],attempt_root='/evidence/probe',owner=thread,
        maximum_runtime_bytes=512*1024**2,qualification_granted=False)
    prefixes={};clock=0.;start=None;identity=None;previous=0;by_hash={}
    for row in phases:
        need('partial' not in row and type(row['started_monotonic']) in (int,float)
            and type(row['finished_monotonic']) in (int,float) and math.isfinite(row['finished_monotonic'])
            and 0<row['started_monotonic']<row['finished_monotonic'],'GROUP_COMPLETE_PHASE_RESOURCE_TIME_REQUIRED')
        for index in (row['start'],row['end']):
            need(index['group_resource_scope']==scope and index['owner']==thread
                and index['maximum_runtime_bytes']==512*1024**2,'GROUP_PHASE_RESOURCE_SCOPE_CHANGED')
            length=index['prefix_bytes'];wanted=(index['prefix_sha256'],index['parsed_events'])
            need(type(length) is int and 0<length<=512*1024**2
                and (length not in prefixes or prefixes[length]==wanted),'GROUP_PHASE_PREFIX_CONFLICT')
            prefixes[length]=wanted
    for index,item in enumerate(rows):
        row=item['receipt'];target,stage=divmod(index,2);phase_index=3+2*target
        need(item['sha256']==E.digest(row) and row['passed'] is True and 'error' not in row
            and row['phase']==f'fault_{target:02d}' and row['phase_index']==phase_index
            and row['stage']==('before_request','before_native')[stage]
            and row['owner']==context['provider_owner'] and row['public_generation']==context['generation']
            and row['scope']==scope and row['plan_sha256']==E.digest(plan), 'GROUP_ACTUAL_RESOURCE_GUARD_OWNER_PHASE_OR_SCOPE')
        need(row['completed_phase_resources_sha256']==E.digest(phases[:phase_index]),'GROUP_ACTUAL_COMPLETED_RESOURCE_PHASES_DIFFER')
        for actual in phases[:phase_index]:
            elapsed=actual['finished_monotonic']-actual['started_monotonic']
            increment=actual['end']['prefix_bytes']-actual['start']['prefix_bytes']
            bound=2*(plan['constant_phase_seconds']+plan['indexed_history_seconds']*
                max(1,actual['end']['prefix_bytes']/plan['maximum_normal_prefix_bytes']))
            need(0<=increment<=2*plan['maximum_phase_increment_bytes'] and 0<elapsed<=bound,
                 'GROUP_ACTUAL_PHASE_EXCEEDS_MEASURED_NORMAL_BOUND')
        original=row['runtime_index'];length=original['prefix_bytes'];check=row['check'];r=check['resources']
        need(type(length) is int and previous<=length<=512*1024**2 and original['owner']==thread
            and original['group_resource_scope']==scope and original['maximum_runtime_bytes']==512*1024**2
            and original['complete_old_prefix_rehashed'] is True and original['raw_log_modified'] is False,
            'GROUP_RESOURCE_LOG_PREFIX_OWNER_OR_CAP')
        previous=length
        if identity is None:identity=original['file_identity']
        need(original['file_identity']==identity and type(identity) is list and len(identity)==2,'GROUP_RESOURCE_LOG_FILE_REPLACED')
        wanted=(original['prefix_sha256'],original['parsed_events'])
        need(length not in prefixes or prefixes[length]==wanted,'GROUP_CONFLICTING_PREFIX_OBSERVATIONS');prefixes[length]=wanted
        now=row['checked_monotonic'];started=row['started_monotonic'];sample=r['sampled_monotonic']
        need(all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in (now,started,sample))
            and clock<=now and started<=sample<=now and now-sample<=5,'GROUP_RESOURCE_CLOCK_OR_FRESHNESS')
        clock=now
        if start is None:start=started
        need(started==start and [r[k] for k in ('pid','python_thread_ident','native_tid')]==thread
            and r['disk_root']=='/evidence/probe','GROUP_RESOURCE_CPU_OWNER_OR_ATTEMPT_CHANGED')
        remaining=60-phase_index;delta=plan['maximum_phase_increment_bytes']*2
        projected=length+remaining*delta+plan['recovery_runtime_reserve_bytes']
        factors=sum(max(1,(length+delta*(i+1))/plan['maximum_normal_prefix_bytes']) for i in range(remaining))
        reserve=3*(plan['constant_phase_seconds']+plan['indexed_history_seconds']*max(1,projected/plan['maximum_normal_prefix_bytes']))
        needed=remaining*plan['constant_phase_seconds']+plan['indexed_history_seconds']*factors+reserve+plan['margin_seconds']+observations['ordinary_health_seconds']
        disk=2*plan['maximum_phase_artifact_bytes']*(remaining+3)+max(0,projected-length)+2*observations['registered_binary_bytes']+64*1024**2
        memory=max(plan['estimated_additional_memory_bytes'],8*projected+12*plan['maximum_capture_json_bytes']+512*1024**2)
        expected=dict(phase_index=phase_index,elapsed_seconds=now-started,remaining_seconds=needed,runtime_bytes=length,
            projected_runtime_bytes=projected,required_free_disk_bytes=disk,required_available_memory_bytes=memory,
            resources=r,plan_sha256=E.digest(plan),fault_authorized=False,qualification_granted=False)
        need(check==expected and projected<=512*1024**2 and now-started+needed<=plan['case_seconds']<=1800
            and type(r['free_disk_bytes']) is type(r['available_memory_bytes']) is int
            and r['free_disk_bytes']>=disk and r['available_memory_bytes']>=memory,'GROUP_RECOMPUTED_F_OFF_RECOVERY_RESOURCES_INSUFFICIENT')
        need(type(r['cgroup_memory']) is list and r['cgroup_memory'],'GROUP_ACTUAL_CGROUP_COUNTERS_REQUIRED')
        for limit in r['cgroup_memory']:
            need(type(limit['current_bytes']) is int and limit['current_bytes']>=0 and
                (limit['limit_bytes'] is None or type(limit['limit_bytes']) is int and limit['limit_bytes']>0
                 and r['available_memory_bytes']<=max(0,limit['limit_bytes']-limit['current_bytes'])),
                 'GROUP_RESOURCE_CGROUP_AVAILABLE_MEMORY_DIFFERS')
        by_hash[item['sha256']]=row
    wanted=set(prefixes);offset=0;lines=0;hashed=hashlib.sha256()
    with runtime_path.open('rb') as stream:
        for line in stream:
            need(len(line)<=4*1024**2 and line.endswith(b'\n'),'GROUP_RESOURCE_ORIGINAL_LOG_LINE_BOUND')
            offset+=len(line);lines+=1;hashed.update(line)
            if offset in wanted:
                need(prefixes[offset]==(hashed.hexdigest(),lines),'GROUP_ORIGINAL_PREFIX_HASH_OR_ROW_COUNT_DIFFERS')
                wanted.remove(offset)
    need(not wanted and offset<=512*1024**2,'GROUP_RESOURCE_PREFIX_NOT_AN_ORIGINAL_COMPLETE_LINE')
    for index,entry in enumerate(saved['admissions']):
        for stage in ('before_request','before_native'):
            sha=entry['proof'][stage+'_resource_sha256'];row=by_hash[sha]
            need(row['phase']==f'fault_{index:02d}' and row['stage']==stage,'GROUP_NATIVE_ADMISSION_RESOURCE_LINK_DIFFERS')
    return dict(checks=56,independently_rehashed_prefixes=len(prefixes),maximum_runtime_bytes=512*1024**2,
        plan_sha256=E.digest(plan),source_bound_host_counter_receipts=True,qualification_granted=False)


def coverage_join(saved,context,actual,controls,references,artifacts,source,prompt,execution,auditor,*,allow_cpu_contract=False):
    """Match every ledger entry to the source-bound actual call, not totals."""
    report=saved['coverage'];group=saved['group'];candidates=candidate_report(group)
    owner=context['provider_owner'];generation=1;implementation=E.digest(source['pins'])
    simulation=context.get('simulation') is True
    need(not simulation or allow_cpu_contract,'GROUP_REAL_NATIVE_EVIDENCE_REQUIRED')
    need(group['selection']=='all_eligible' and group['trigger']=='once' and group['concurrency']==1
        and group['held_leases']==[] and len(candidates['candidates'])==56 and candidates['discovery_complete'] is True,
        'GROUP_COMPLETE_ACTUAL_DISCOVERY_REQUIRED')
    known={};leases={};base=controls[0];model=base['scope']['model_object_id'];worker=context['worker']
    for i,row in enumerate(candidates['candidates']):
        decode=i>=28;path=N.LAYERS[i%28];target=row['identity'];call=base['calls'][i]
        eligibility=dict(status='SUPPORTED',reasons=[]) if decode else dict(status='UNSUPPORTED',reasons=['ACTUAL_PREFILL_OUTSIDE_Q1_S2_SCOPE'])
        need(row['candidate_id']==path+(':decode' if decode else ':prefill') and row['eligibility']==eligibility,
            'GROUP_ACTUAL_TARGET_CLASSIFICATION_DIFFERS')
        expected=dict(process_id=worker['pid'],process_start_identity=worker['process_start_identity'],
            worker_id=worker['worker_start_identity'],scope_kind='INITIALIZED_WORKER',rank=worker['rank'],
            device_type='cpu' if simulation else 'npu',logical_device=None if simulation else worker['logical_device'],model_instance_id=worker['model_instance_id'],
            model_object_identity=model,module_object_identity=call['module_object_id'],module_roles=['attention'],
            operator_api=call['actual_operator'],tensor_signature='TND_Q1_PAGE16' if decode else 'ACTUAL_PREFILL',
            boundary=dict(schema_sha256=E.digest(dict(api=call['actual_operator'],decode=decode)),implementation_sha256=implementation,
                call_token='actual-target:'+E.digest(dict(path=path,module=call['module_object_id'],decode=decode)),
                implementation_kind='torch_dispatch',kind='operator_api'),provider_identity='33:all-target-isolated-s2',
            provider_proof=[['implementation_sha256',implementation]])
        need(target==expected,'GROUP_ACTUAL_WORKER_OBJECT_BOUNDARY_IDENTITY_DIFFERS')
        known[E.digest(target)]=(path,decode,target)
    for index,lease in enumerate(group['leases']):
        path=N.LAYERS[index];identity=E.digest(lease['target'])
        need(known[identity][:2]==(path,True) and lease['owner']==owner and lease['generation']==generation
            and lease['lease_id']=='target:'+str(index) and lease['device_bytes']==lease['host_bytes']==0,
            'GROUP_TARGET_LEASE_IDENTITY_DIFFERS')
        leases[identity]=lease
    need(len(leases)==28 and len(group['lossless'])==84,'GROUP_ALL_TARGET_LOSSLESS_DENOMINATOR_REQUIRED')
    for index,proof in enumerate(group['lossless']):
        ordinal,mode=divmod(index,3);path=N.LAYERS[ordinal];control=controls[mode];live=control['live'][path];e=live['evidence']
        expected=dict(lease=group['leases'][ordinal],mode=PHASES[mode],proof_sha256=E.digest(live),
            input_sha256_before=[E.digest(receipt_signature(e['before']['signature'],simulation=simulation))],
            input_sha256_after=[E.digest(receipt_signature(e['after']['signature'],simulation=simulation))],
            reference_output_sha256=base['payloads'][path+'/output']['sha256'],output_sha256=e['outputs'][0]['sha256'],
            original_calls=1,temporary_readback_verified=True,resources_released=True)
        need(proof==expected and e['before']==e['after'],'GROUP_ACTUAL_THREE_CONTROL_PROOF_DIFFERS')
    need(report['owner']==owner and report['generation']==generation and report['scope']==dict(mechanism='33.paged_decode',
        workload_sha256=E.digest(prompt),environment_sha256=E.digest(execution),implementation_sha256=implementation,
        evidence_kind='cpu_contract' if simulation else 'natural_worker',model_weights_sha256=None,model_source_sha256=None)
        and report['discovery']['candidates_sha256']==E.digest(candidates)
        and report['discovery']['artifact_sha256']==controls[2]['capture_sha256']
        and report['discovery']['observed_target_ids']==sorted(known)
        and report['observation']['artifact_sha256']==E.digest(references), 'GROUP_ORIGINAL_SCOPE_DISCOVERY_OR_OBSERVATION_HASH')
    need(len(actual)==len(report['calls'])==3136,'GROUP_ALL_F_OFF_CALL_ROWS_REQUIRED')
    hits=Counter()
    for row,observed in zip(report['calls'],actual):
        call=observed['call'];identity=call['coverage_target_id'];path,decode,target=known[identity]
        injecting=observed['selected'];receipt=row['receipt']
        phase=observed['phase'];selected=int(phase.split('_')[1]) if phase.startswith('fault_') else None
        expected_phase='OFF_RECOVERY' if selected is None else 'POST_FIRST_ANOMALY' if decode and N.LAYERS.index(path)>selected else 'PRE_FAULT_PROPAGATION'
        need(row['call_id']==call['coverage_call_id'] and row['target_id']==identity
            and row['request_ids']==[observed['request_id']]
            and row['forward_id']==observed['request_id']+(':decode' if decode else ':prefill')
            and row['phase']==call['coverage_phase']==expected_phase and row['eligible'] is decode
            and row['decision']['ticket']['call']['target']==target
            and call['module_path']==path and call['module_object_id']==target['module_object_identity']
            and row['decision']['admitted'] is injecting and receipt['original_calls']==1
            and receipt['native_applied'] is injecting and receipt['outcome']==('HIT' if injecting else 'PASSTHROUGH'),
            'GROUP_ACTUAL_CALL_TO_DECISION_RECEIPT_DIFFERS')
        need(receipt['execution_sha256']==observed['execution_sha256'] and receipt['effect_sha256']==observed['effect_sha256'],
             'GROUP_ACTUAL_EXECUTION_OR_EFFECT_ARTIFACT_HASH_DIFFERS')
        if injecting:hits[identity]+=1
    need(set(hits)==set(leases) and set(hits.values())=={1},'GROUP_EACH_ACTUAL_TARGET_EXACT_ONCE')
    need(len(saved['admissions'])==28,'GROUP_EXACT_28_LIVE_ADMISSIONS_REQUIRED')
    admitted=[row for row in actual if row['selected']]
    for entry,observed in zip(saved['admissions'],admitted):
        proof=entry['proof'];call=observed['call'];native=call['native'];path=call['module_path']
        normal_match=proof['normal_match']
        need(E.digest(proof)==entry['sha256']==native['admission_sha256']
            and proof['owner']==owner and proof['public_generation']==context['generation'] and proof['coverage_generation']==generation
            and proof['request_id']==observed['request_id'] and proof['phase']==observed['phase'] and proof['target_path']==path
            and proof['target_set_sha256']==base['scope']['sha256']
            and proof['coverage_call_id']==call['coverage_call_id'] and proof['coverage_target_id']==call['coverage_target_id']
            and proof['control_request_ids']==[control['request_id'] for control in controls]
            and type(proof['binding_object_id']) is int and proof['binding_object_id']>0
            and normal_match['current_signature_sha256']==E.digest(native['logical_input_binding']['evidence']['before']['signature'])
            and normal_match['module_path']==path and normal_match['normal_evidence_sha256']==context['normal_evidence_sha256']
            and entry['consumed'] is entry['returned'] is entry['attention_output_changed'] is True
            and entry['artifact']==observed['artifact'] and entry['native_call_id']==native['call_id'],
            'GROUP_LIVE_CURRENT_INPUT_ADMISSION_USE_DIFFERS')
        if not simulation:need(entry['numeric_effect']==observed['numeric_effect'],
                               'GROUP_COMPLETE_NUMERIC_EFFECT_RECEIPT_DIFFERS')
    checked=auditor.audit_saved_coverage(report,artifact_index=artifacts)
    need(checked['report_consistent'] and not checked['errors'] and not checked['missing_artifact_sha256'],
         'GROUP_INDEPENDENT_LEDGER_RECONCILIATION:'+','.join(checked['errors']))
    recovery=report['recovery'];health=references[-1]
    for row in recovery['targets']:
        path=known[E.digest(row['lease']['target'])][0]
        need(row['lease']==leases[E.digest(row['lease']['target'])] and row['proof_sha256']==E.digest(health)
            and row['off_output_sha256']==base['payloads'][path+'/output']['sha256'],'GROUP_ACTUAL_TARGET_OFF_HASH_DIFFERS')
    need(len(recovery['workers'])==1 and recovery['workers'][0]['proof_sha256']==E.digest(health)
        and recovery['workers'][0]['baseline_output_sha256']==recovery['workers'][0]['off_output_sha256']==base['payloads']['logits']['sha256'],
        'GROUP_ACTUAL_SAME_WORKER_HEALTH_HASH_DIFFERS')
    ledger=report['live_c2_proof'];need(len(ledger['issued'])==1 and ledger['consumed'] is None,'GROUP_ONE_ORIGINAL_C2_PROOF_NO_C3_USE')
    issued=ledger['issued'][0];expected=dict(scope=report['scope'],candidates_sha256=E.digest(candidates),target_ids=sorted(leases),
        owner=owner,generation=generation,session_id=report['session_id'],
        report_sha256=E.digest({k:v for k,v in report.items() if k!='live_c2_proof'}),
        discovery_sha256=report['discovery']['artifact_sha256'],observation_sha256=report['observation']['artifact_sha256'],
        recovery_sha256=E.digest(recovery))
    need(issued['proof']==expected and issued['proof_sha256']==E.digest(expected)
        and issued['exact_issued_object_verified'] is True and issued['object_binding']['pid']==worker['pid']
        and type(issued['object_binding']['python_object_id']) is int and issued['object_binding']['python_object_id']>0
        and issued['serializable_authority'] is issued['qualification_granted'] is False
        and saved['control']['live_c2_proof']==expected,'GROUP_LIVE_C2_ISSUANCE_NOT_BOUND_TO_ORIGINAL_REPORT')
    return dict(targets=28,requests=60,natural_calls=3360,coverage_calls=3136,recomputed_hits=28,
        live_c2_issuance=issued,serializable_authority=False,simulation=simulation,C2=not simulation,C3=False,E5='NOT_ASSESSED')


def review(args):
    checks=N.Checks();report=dict(schema='demoncat.paged33.public_group_c2_review.v1',C1=False,C2=False,C3=False,
        E5='NOT_ASSESSED',fault_authorized=False,original_results_modified=False,qualification_flags_used_as_proof=False)
    def perform():
        source=E.frozen_source(args.source,args.source_manifest_sha256,build_variant=E.GROUP_VARIANT)
        need(E.sha(__file__)==source['pins'][E.EXPERIMENT+Path(__file__).name],'GROUP_FROZEN_REVIEWER_DIFFERS')
        helper=E.helpers(source);bundle=E.build_bundle(args.build_root,source,helper,args.completion_sha256)
        observer=E.observer_bundle(args.observer,source,args.observer_receipt_sha256)
        normal=A.review(SimpleNamespace(**{k:getattr(args,k) for k in ('source','source_manifest_sha256','build_root',
            'completion_sha256','observer','observer_receipt_sha256','stock_run','stock_sha256','baseline_run','baseline_sha256')}),build_variant=E.GROUP_VARIANT)
        need(not normal['issues'] and normal.get('complete_payload_pairs')==435 and normal.get('per_target_input_output_pairs')==140,
             'GROUP_FRESH_SAME_PACKAGE_ALL_TARGET_NORMAL_REQUIRED')
        need(E.sha(args.normal_report)==args.normal_report_sha256 and E.sha(args.run/'result.json')==args.result_sha256,
             'GROUP_PINNED_ORIGINAL_CHANGED')
        published=E.read(args.normal_report)
        need(published['source']['manifest_sha256']==source['manifest_sha256']
            and published['build']['completion_sha256']==args.completion_sha256
            and published['arms']['stock']['result_sha256']==args.stock_sha256
            and published['arms']['baseline']['result_sha256']==args.baseline_sha256,'GROUP_PUBLISHED_NORMAL_INPUTS_DIFFER')
        data=E.read(args.run/'result.json');report['original_status']=data.get('status');report['original_failures']=N.original_failures(data)
        need(not report['original_failures'],'GROUP_ORIGINAL_FAILURES_PRESERVED')
        phases=[r.get('phase') for r in data.get('requests',[])]
        need(phases==list(PHASES),'GROUP_INCOMPLETE_PROTOCOL_PHASES:'+','.join(p for p in PHASES if p not in phases))
        context=S.protocol(args.run,data,group_phases=PHASES);report['protocol']=context
        context['normal_evidence_sha256']=data['setup']['normal_evidence_sha256']
        N.check_coexistence(data,source,bundle,context['worker']['pid'],cleanup_receipts=[data['library_coexistence_after_owned_cleanup']])
        original=E.read(args.baseline_run/'result.json');normal_capture=A.phase_capture(args.baseline_run,original['phases'][0],group=True)
        contract=data['execution_contract'];proof=data['setup']['normal_evidence'];normal_contract=original['execution_contract']
        need({k:v for k,v in contract.items() if k!='driver_sha256'}=={k:v for k,v in normal_contract.items() if k!='driver_sha256'}
            and contract['driver_sha256']==source['pins'][E.EXPERIMENT+'natural33_arch22_group_worker_core_probe.py']
            and data['prompt_token_ids']==original['warm_healthy']['prompt_token_ids']
            and E.digest(proof)==context['normal_evidence_sha256'] and proof['variant']==E.GROUP_VARIANT
            and proof['source_pins']==source['pins'] and proof['source_manifest_sha256']==source['manifest_sha256']
            and proof['completion_sha256']==args.completion_sha256 and proof['observer_receipt_sha256']==observer['sha256']
            and proof['stock_result_sha256']==args.stock_sha256 and proof['private_result_sha256']==args.baseline_sha256,
            'GROUP_FRESH_PACKAGE_MODEL_OR_NORMAL_OBSERVATION_IDENTITY_DIFFERS')
        need(proof['published_report_sha256']==args.normal_report_sha256
            and proof['resource_observations']==normal['arms']['baseline']['resource_observations'],
            'GROUP_ORIGINAL_NORMAL_REPLAY_AND_RESOURCE_INPUTS_DIFFER')
        runtime,raw=helper.boundary.read_runtime(args.run,maximum_bytes=512*1024**2);report['runtime_file']=raw;symbols={};tokens=[];controls=[];actual=[];refs=[];artifacts={};summaries=[]
        def add(value,label):artifacts.setdefault(E.digest(value),[]).append(dict(path=label))
        references=[r['worker'] for r in data['requests']]
        need(sum(r['bytes'] for r in references)<=60*128*1024**2,'GROUP_TOTAL_CAPTURE_JSON_BOUND')
        for index,request in enumerate(data['requests']):
            phase=request['phase'];context['phase']=phase;folder='group_requests/'+f'{index:03d}-'+phase
            capture=capture_file(args.run,request['worker'],index,phase);row=capture['natural_request'];rid=row['request_id']
            need(capture['audit']['source']==normal_capture['audit']['source'],'GROUP_CURRENT_MODEL_BOUNDARY_SOURCE_DIFFERS')
            lineage=S.request_lineage(dict(request,worker=row),data['prompt_token_ids'],'npu:'+str(context['worker']['logical_device']))
            prefill,decode=row['forwards'];S.decode_cache_rows(prefill['scheduler'],decode['scheduler'],decode['calls'],device='npu:0')
            calls=prefill['calls']+decode['calls'];audit={c['module_path']:c for c in decode['calls']}
            need(capture['audit']['rows']==decode['calls'],'GROUP_AUDIT_AND_ACTUAL_DECODE_ROWS_DIFFER')
            paths=A.target_set(capture,audit,expected_generation=context['generation']);scope=capture['target_scope']
            need(paths==list(N.LAYERS) and len(capture['native_calls'])==28
                and capture['native_calls']==[c['native'] for c in decode['calls']],'GROUP_ALL_28_ACTUAL_NATIVE_TARGETS_REQUIRED')
            values=payloads(args.run,capture,folder,helper,normal_capture);live={};native=[];is_fault=phase.startswith('fault_')
            selected_index=int(phase.split('_')[1]) if is_fault else None
            host_path=E.child(args.run,folder+'/native/host.jsonl');host_raw=host_path.read_bytes()
            need(host_raw.endswith(b'\n') and len(host_raw)<=8*1024**2,'GROUP_COMPLETE_BOUNDED_HOST_LOG_REQUIRED')
            hosts=[json.loads(line,object_pairs_hook=E.unique) for line in host_raw.splitlines()]
            need(len(hosts)==28,'GROUP_EXACT_28_ORIGINAL_HOST_CALLBACKS')
            for ordinal,path in enumerate(paths):
                saved=capture['all_decode_boundaries'][path];call=audit[path]
                thread=N.captured_thread_identity(saved)
                if 'thread_identity' not in context:context['thread_identity']=thread
                need(context['thread_identity']==thread,'GROUP_ACTUAL_THREAD_IDENTITY_CHANGED')
                N.check_actual_inputs(saved);N.check_live_input(saved,call,expected_generation=context['generation'])
                boundary=helper.boundary.first_boundary(args.run,dict(phase=folder,finish=dict(capture=capture)),call,
                    difference_reader=helper.differences,target_path=path,relative_folder='boundaries/'+str(ordinal),evidence_prefix='/evidence/probe')
                N.page_relocation_evidence(saved,boundary,call,target_path=path)
                live[path]=saved['logical_input_binding'];signature=live[path]['evidence']['before']['signature']
                normal_signature=normal_capture['all_decode_boundaries'][path]['logical_input_binding']['evidence']['before']['signature']
                need(proof['targets'][path]['reference_input']==normal_signature,'GROUP_TARGET_NORMAL_CATALOG_DIFFERS')
                if not is_fault or ordinal<=selected_index:
                    need(S.normalized(signature)==S.normalized(normal_signature),'GROUP_CURRENT_INPUT_DIFFERS_BEFORE_SELECTED_TARGET:'+path)
                decoded=native_call(args.run,folder,call,saved,ordinal,scope,context,runtime,helper,bundle,symbols,ordinal==selected_index)
                need(decoded['host']==hosts[ordinal],'GROUP_ORIGINAL_HOST_ROW_ORDER_DIFFERS')
                native.append(decoded);tokens.append(decoded['token'])
                if controls:need(live[path]['identity']==controls[0]['live'][path]['identity'],'GROUP_LIVE_CONTROL_OBJECT_OWNER_CHANGED')
            if controls:
                for a,b in zip(controls[0]['forwards'],row['forwards']):need(a['input']==b['input'],'GROUP_ACTUAL_ROOT_INPUT_CHANGED')
                need(all(prefill[k]==controls[0]['forwards'][0][k] for k in ('hidden','logits')),'GROUP_NON_TARGET_PREFILL_OUTPUT_CHANGED')
                need(scope==controls[0]['scope'],'GROUP_ACTUAL_TARGET_OWNER_CHANGED')
            effect=None
            if is_fault:
                selected=paths[selected_index];effect=helper.differences.raw_difference(controls[0]['payloads'][selected+'/output'],values[selected+'/output'])
                need(effect['numeric_changed_elements']>0,'GROUP_STRATEGY_SELECTED_WITHOUT_NUMERICAL_EFFECT:'+selected)
                for path in paths[:selected_index]:need(all(values[path+'/'+k]['sha256']==controls[0]['payloads'][path+'/'+k]['sha256'] for k in ('output','key','value')),
                    'GROUP_ISOLATED_TARGET_PREFIX_CHANGED')
            else:
                expected=proof['payloads']
                need({k:{x:v[x] for x in ('shape','dtype','bytes','sha256')} for k,v in values.items()}==expected,'GROUP_COMPLETE_NORMAL_OR_OFF_PAYLOAD_DIFFERS')
            compact=dict(request_id=rid,phase=phase,scope=scope,calls=calls,forwards=row['forwards'],live=live,payloads=values,
                capture_sha256=E.digest(capture))
            if index<3:controls.append(compact);add(capture,request['worker']['path'])
            if phase.startswith(('fault_','off_')):
                for ordinal,call in enumerate(calls):
                    injecting=is_fault and ordinal==28+selected_index
                    execution_hash=E.digest(call['native'] if injecting else call);effect_hash=None
                    add(call['native'] if injecting else call,request['worker']['path'])
                    if injecting:
                        name=call['module_path']+'/output'
                        def sig(value):return {k:value[k] for k in ('shape','dtype','bytes','sha256')}
                        artifact=dict(before=sig(controls[0]['payloads'][name]),after=sig(values[name]),artifact=request['worker'])
                        effect_hash=E.digest(artifact);add(artifact,request['worker']['path'])
                    projected={k:call[k] for k in ('coverage_call_id','coverage_target_id','coverage_phase','module_path','module_object_id')}
                    if injecting:projected['native']=call['native']
                    actual.append(dict(call=projected,phase=phase,request_id=rid,selected=injecting,artifact=request['worker'],
                        execution_sha256=execution_hash,effect_sha256=effect_hash,
                        numeric_effect=(dict(numeric_changed_elements=effect['numeric_changed_elements'],
                            max_abs_difference=effect['max_abs_difference'],complete_elements=2048,
                            before_sha256=controls[0]['payloads'][call['module_path']+'/output']['sha256'],
                            after_sha256=values[call['module_path']+'/output']['sha256']) if injecting else None)))
            refs.append(request['worker']);add(request['worker'],request['worker']['path'])
            summaries.append(dict(phase=phase,request_id=rid,actual_calls=56,decode_calls=28,native_scopes=28,
                full_payloads=87,operator_effect=effect,model_logits_changed=bool(controls and values['logits']['sha256']!=controls[0]['payloads']['logits']['sha256'])))
        need(len(tokens)==len(set(tokens))==1680 and [r['call_id'] for r in runtime if r['event']=='begin']==tokens
            and [r['call_id'] for r in runtime if r['event']=='end']==tokens and not any(r['event']=='capture_overflow' for r in runtime),
            'GROUP_COMPLETE_ORDERED_1680_ACTUAL_NATIVE_SCOPES_REQUIRED')
        coverage_ref=data['final_cleanup']['evidence']['coverage'];coverage_path=S.local(args.run,coverage_ref['path'])
        need(coverage_path.stat().st_size==coverage_ref['bytes']<=128*1024**2 and E.sha(coverage_path)==coverage_ref['sha256'],
             'GROUP_FINAL_COVERAGE_ARTIFACT_CHANGED')
        saved=E.read(coverage_path);need(saved['requests']==refs,'GROUP_COVERAGE_ACTUAL_REQUEST_REFERENCES_DIFFERS');add(refs,'all request references')
        audit_path=source['root']/'fault_injection/coverage_audit.py'
        spec=importlib.util.spec_from_file_location('independent_group_coverage_audit',audit_path);auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)
        report['coverage']=coverage_join(saved,context,actual,controls,refs,artifacts,source,data['prompt_token_ids'],normal_contract,auditor)
        resource_path=source['root']/'fault_injection/native_adapters/attention28/group_resource_budget.py'
        spec=importlib.util.spec_from_file_location('independent_group_resource_plan',resource_path);planner=importlib.util.module_from_spec(spec);spec.loader.exec_module(planner)
        report['resources']=resource_join(saved,context,normal['arms']['baseline']['resource_observations'],
            args.run/'runtime/runtime.jsonl',source,args.completion_sha256,planner)
        report.update(phases=summaries,registered_elfs=list(symbols.values()),complete_payloads_rehashed=5220,
            source_manifest_sha256=source['manifest_sha256'],result_sha256=args.result_sha256,normal_replay_sha256=E.digest(normal))
        need(E.sha(args.run/'result.json')==args.result_sha256 and E.sha(args.run/'runtime/runtime.jsonl')==raw['sha256'],
             'GROUP_ORIGINAL_CHANGED_DURING_REVIEW')
    checks.run('source_raw_protocol_and_coverage',perform)
    report.update(checks=checks.checks,issues=checks.issues,status='SOURCE_BOUND_ALL_28_ISOLATED_C2_RECONCILED_NOT_C3_E5' if not checks.issues else 'REFUSED_WITH_ORIGINAL_EVIDENCE',
        C2=not checks.issues,limits=['C2 is this exact model/package/workload and 28 actual Q1/page16 decode objects only.',
            'C3 propagated current input and stock-owner exact normal are not established; no saved proof is execution authority.',
            'Mask/read-before logical KV use source-bound hashes; matching complete post-consumer KV bytes were rehashed.',
            'Selected ACLNN caller owner is not observed. Post-unregister health covers tokens/text only.'])
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','build-root','observer','stock-run','baseline-run','normal-report','run','out'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('source-manifest-sha256','completion-sha256','observer-receipt-sha256','stock-sha256','baseline-sha256','normal-report-sha256','result-sha256'):p.add_argument('--'+key,required=True)
    args=p.parse_args();need(not args.out.exists(),'FRESH_GROUP_C2_REVIEW_REQUIRED');sys.dont_write_bytecode=True
    numeric=N.initialize_cpu_byte_reader();sys.addaudithook(E.prohibit_runtime)
    report=review(args);report['cpu_byte_reader']=numeric;args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=report['status'],checks=report['checks'],issues=len(report['issues']),sha256=E.sha(args.out),C3=False,E5='NOT_ASSESSED')))
    return 0 if not report['issues'] else 2


if __name__=='__main__':raise SystemExit(main())
