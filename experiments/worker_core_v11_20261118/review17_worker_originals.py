"""Read-only original Worker17 Torch-parameter audit; never a qualification issuer.

Requires the complete original result and reviewed source snapshot. It neither
imports the injector nor resolves a framework/operator. No statistics/E5 claim.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

_spec=importlib.util.spec_from_file_location('_vision17_review_common',Path(__file__).with_name('group_review_common.py'))
C=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(C)
PHASES=('native_baseline','baseline','baseline_repeat','lossless','c2_fault','c2_off',
        'c3_fault_0','c3_fault_1','c3_off','cleanup_healthy')
FAULTS=(4,6,7)
ARMS=(None,'baseline','baseline','baseline','fault',None,'fault','fault',None,None)
ALIAS={'view','reshape','_unsafe_view','flatten','transpose','permute','squeeze','unsqueeze','slice','select','alias','detach'}
COPY={'clone','_to_copy'}
READERS={'add','mul','linear','mm','matmul','addmm','bmm','convolution','conv3d','cat'}
BASE_KEYS=('schema','session_id','scope','selection','trigger','owner','generation','budget','discovery','observation',
    'planned_targets','planned_targets_hit','observed_calls','observed_eligible_calls','hits','missing_outcomes',
    'propagation_added_calls','calls','recovery','contract_satisfied','rejection_reasons','real_worker_evidence',
    'real_pretrained_model_evidence','public_qualification_granted')
SOURCE_NAMES=('vision17_group.py','vision17_request.py','vision17_worker.py','vision17_context.py',
    'vision17_trace.py','vision17_padding.py','conv3d_natural.py','_scoped_dispatch.py','tensor_callable.py')
SOURCE_MODULES=('fault_injection.coverage','fault_injection.coverage_session',
    'demoncat.runtime.vllm_ascend.worker_extension.providers.tensor_natural',
    'demoncat.runtime.vllm_ascend.worker_extension.providers.vision17_group')
PROBE='experiments/c2c3_closure_20261117/vision17_worker_group_probe.py'


def signature(row):
    return dict(path=row['path'],sequence=row['sequence'],operator=row['operator'],schema=row['schema'],
        geometry=row['actual_geometry'],inputs=row['inputs_before'],input_metadata=[
            {k:v for k,v in t.items() if k in ('shape','stride','offset','dtype','device','actual_format')}
            for t in row['actual_input_metadata']])


def outputs(row):return {k:row[k] for k in ('encoder_output','hidden','logits')}


def tensor(a,value,where,*,nullable=False):
    if nullable and value is None:return
    a.check(isinstance(value,dict) and C.sha(value['sha256']) and isinstance(value['dtype'],str) and
        isinstance(value['shape'],list) and all(type(v) is int and v>=0 for v in value['shape']),
        'COMPLETE_RECORDED_TENSOR_IDENTITY',where)


def metadata(a,value,where,*,natural):
    a.check(all(C.positive(value[k]) for k in ('object_id','storage_id')) and type(value['offset']) is int and
        value['offset']>=0 and len(value['stride'])==len(value['shape']) and
        all(type(v) is int for v in value['stride']) and all(type(v) is int and v>=0 for v in value['shape']),
        'ACTUAL_OBJECT_STORAGE_GEOMETRY',where)
    a.check(value['device']==('npu:0' if natural else 'cpu'),'RECORDED_EXECUTION_DEVICE',where)
    if natural:a.check(type(value['actual_format']) is int and value['actual_format']>=0,'OBSERVED_ACTUAL_NPU_FORMAT',where)


def operator(name):
    parts=name.split('.')
    if len(parts)<3 or parts[0]!='aten':return None
    return parts[1]


def trace(a,observed,where,*,natural):
    calls=observed['observed_calls']
    a.check(observed['simulation'] is (not natural) and observed['forwards']==[dict(index=0,returned=True)] and
        observed['hooks']==observed['held_lineage_tensors']==0 and observed['owned_mode'] is False and
        len(calls)==1 and observed['structural_candidates']==observed['resolved_candidates']==[calls[0]['path']],
        'COMPLETE_SINGLE_PROJECTION_TRACE_AND_RELEASE',where)
    a.check(all(observed[k] is False for k in ('aclnn_launch_arguments','device_payload_observed','native_qualification',
        'c1','c2','c3','e5')),'TRACE_SCOPE_REMAINS_TORCH_ONLY',where)
    row=calls[0]
    a.check(row['sequence']==row['forward']==0 and C.positive(row['module_object_id']) and row['operator']=='aten.conv3d.default' and
        row['schema'].startswith('aten::conv3d(') and row['returned'] is True and row['aclnn_launch_arguments'] is False and
        row['final_torch_native_dispatch_arguments'] is True,'ACTUAL_TORCH_CONV3D_RETURN',where)
    a.check(len(row['inputs_before'])==len(row['inputs_after'])==3 and len(row['actual_input_metadata'])==2 and
        row['inputs_before']==row['inputs_after'],'FULL_INPUT_WEIGHT_BIAS_UNCHANGED',where)
    for index,value in enumerate(row['inputs_before']):tensor(a,value,where+':input'+str(index),nullable=index==2)
    for index,value in enumerate(row['actual_input_metadata']):
        metadata(a,value,where+':metadata'+str(index),natural=natural)
        a.check(all(value[k]==row['inputs_before'][index][k] for k in ('shape','dtype')),'INPUT_VALUE_METADATA_JOIN',where)
    tensor(a,row['output'],where+':output');metadata(a,row['output_metadata'],where+':output',natural=natural)
    a.check(all(row['output'][k]==row['output_metadata'][k] for k in ('shape','dtype')),'RETURN_VALUE_METADATA_JOIN',where)
    lineage={row['output_metadata']['object_id']:row['output_metadata']}
    a.check(len(row['lineage_edges'])<=128,'BOUNDED_LINEAGE',where)
    for index,edge in enumerate(row['lineage_edges']):
        kind=operator(edge['operator']);ids=edge['actual_input_object_ids'];out=edge['output_metadata']
        a.check(edge['returned'] is True and ids and all(i in lineage for i in ids) and out and
            kind in ALIAS|COPY and edge['relation']==('ACTUAL_RETURNED_ALIAS' if kind in ALIAS else 'ACTUAL_RETURNED_COPY'),
            'EXACT_OBJECT_LINEAGE_EDGE',where+':'+str(index))
        for item in out:
            metadata(a,item,where+':edge'+str(index),natural=natural)
            if kind in ALIAS:
                a.check(any(item['storage_id']==lineage[i]['storage_id'] for i in ids if i in lineage),'ACTUAL_ALIAS_STORAGE',where)
            if item['object_id'] in lineage:a.check(item==lineage[item['object_id']],'NO_REUSED_OBJECT_ID_WITH_CHANGED_METADATA',where)
            lineage[item['object_id']]=item
    consumer=row['consumer'];positions=consumer['actual_input_positions'];consumed=consumer['actual_input_metadata']
    a.check(consumer['returned'] is True and operator(consumer['operator']) in READERS and
        consumer['binding']=='EXACT_LIVE_OBJECT_FROM_OBSERVED_RETURN_CHAIN' and
        len(consumed)==len(consumer['consumed_values'])==len(positions)>0 and
        positions==sorted(set(positions)) and all(type(p) is int and p>=0 for p in positions),
        'FIRST_NUMERICAL_CONSUMER_EXACT_OBJECT_BINDING',where)
    for item,value in zip(consumed,consumer['consumed_values']):
        tensor(a,value,where+':consumer');metadata(a,item,where+':consumer',natural=natural)
        a.check(lineage.get(item['object_id'])==item and all(value[k]==item[k] for k in ('shape','dtype')),
                'CONSUMER_MUST_READ_OBSERVED_RETURN_OBJECT',where)
    a.check(bool(consumer['outputs']),'ACTUAL_CONSUMER_RETURN_VALUES',where)
    for value in consumer['outputs']:tensor(a,value,where+':consumer_output')
    return row


def phases(a,d,*,natural):
    rows=d['phase_rows'];requests=d['requests'];resets=d['mm_cache_resets']
    a.check(len(rows)==len(requests)==len(resets)==10 and [v['phase'] for v in rows]==list(PHASES),
            'ALL_TEN_ACTUAL_REQUEST_PHASES')
    first=rows[0];binding=first['request_binding'];seen=set();summaries=[];upper=lower=None
    for index,(row,driver,reset) in enumerate(zip(rows,requests,resets)):
        phase=PHASES[index];rid=row['request_id'];mapped=driver['mapping'];current=row['request_binding']
        a.check(rid not in seen and phase==driver['phase']==reset['phase'] and
            rid==mapped['internal_request_id']==current['request_id'] and C.positive(mapped['request_state_object_id']) and
            mapped['source']==('ACTUAL_PUBLIC_ENQUEUE_AND_OUTPUT_PROCESSOR' if natural else 'EXPLICIT_CPU_RUNNER_MAPPING'),
            'REAL_FRESH_REQUEST_MAPPING',phase);seen.add(rid)
        a.check(row['encoder_observed'] is row['root_returned'] is row['logits_returned'] is True and
            current['simulation'] is (not natural) and current['single_image_request_join_proven'] is True and
            current['encoder_input_index']==0 and current['actual_visual_object_id']==binding['actual_visual_object_id'] and
            all(C.positive(current[k]) for k in ('actual_batched_object_id','actual_visual_object_id','feature_object_id',
                'mm_kwargs_item_object_id','request_object_id','scheduler_object_id')),'ACTUAL_ENCODER_ROOT_LOGITS_LIFETIME',phase)
        a.check(all(current[k]==binding[k] for k in ('prompt_token_ids','pixel_values','image_grid_thw','placeholder','actual_frame_sources')) and
            driver['prompt_token_ids']==current['prompt_token_ids'] and len(current['prompt_token_ids'])>=2,
            'SAME_PROCESSED_IMAGE_PROMPT_AND_FRAME_SOURCES',phase)
        a.check(reset['encoder_cache_reset_requested'] is True and reset['idle_before'] is reset['idle_after'] is reset['returned_none'] is True and
            reset['encoder_execution_proven'] is reset['qualification_granted'] is False and len(reset['public_resets'])==2 and
            [v['method'] for v in reset['public_resets']]==['LLM.reset_mm_cache','LLMEngine.reset_encoder_cache'] and
            all(C.sha(v['source_sha256']) for v in reset['public_resets']) and
            reset['public_resets']==resets[0]['public_resets'],'PUBLIC_CACHE_RESET_IS_NOT_ENCODER_EXECUTION_PROOF',phase)
        before=trace(a,row['initial_observation'],phase+':upper',natural=natural)
        after=trace(a,row['conv_observation'],phase+':lower',natural=natural)
        if upper is None:upper,lower=before,after
        a.check(signature(before)==signature(upper) and before['module_object_id']==upper['module_object_id'] and
            before['actual_geometry']['padding']==[0,0,0] and after['actual_geometry']==dict(before['actual_geometry'],
                padding=[0,7,7] if index in FAULTS else [0,0,0]),'ONLY_FINAL_PADDING_DIFFERS',phase)
        a.check(all(before[k]==after[k] for k in ('path','sequence','module_object_id','operator','schema','actual_input_metadata',
            'inputs_before','inputs_after','output','output_metadata','lineage_edges','consumer')),'UPPER_LOWER_SAME_ACTUAL_CALL_RETURN',phase)
        a.check(before['inputs_before'][0]['sha256']==current['pixel_values']['sha256'] and
            before['inputs_before'][0]['dtype']==current['pixel_values']['dtype'],'ACTUAL_PIXELS_REACH_PROJECTION',phase)
        for name in ('encoder_output','hidden','logits'):
            a.check(len(row[name])==len(first[name])>0,'COMPLETE_MODEL_OUTPUT_COUNT',phase+':'+name)
            for value,baseline in zip(row[name],first[name]):
                tensor(a,value,phase+':'+name)
                a.check(all(value[k]==baseline[k] for k in ('shape','dtype')),'MODEL_OUTPUT_GEOMETRY_STABLE',phase+':'+name)
        changed=after['output']!=lower['output'];consumed=after['consumer']['consumed_values']!=lower['consumer']['consumed_values']
        consumer_changed=after['consumer']['outputs']!=lower['consumer']['outputs'];logits_changed=row['logits']!=first['logits']
        if index in FAULTS:
            a.check(changed and consumed and consumer_changed and logits_changed,'REAL_PADDING_EFFECT_REACHES_CONSUMER_AND_LOGITS',phase)
            if index!=FAULTS[0]:a.check(outputs(row)==outputs(rows[FAULTS[0]]),'REPEATED_F_ENCODER_HIDDEN_LOGITS_EXACT',phase)
        else:
            a.check(not changed and not consumed and not consumer_changed and outputs(row)==outputs(first),
                    'HEALTHY_LOSSLESS_OFF_CLEANUP_COMPLETE_OUTPUTS_EXACT',phase)
        mechanism=row['mechanism']
        if ARMS[index] is None:a.check(mechanism is None,'OFF_HAS_NO_PADDING_ADAPTER',phase)
        else:
            a.check(mechanism['arm']==ARMS[index] and mechanism['simulation'] is (not natural) and
                not mechanism['errors'] and len(mechanism['events'])==1,'ONE_ORIGINAL_PADDING_EVENT',phase)
            event=mechanism['events'][0]
            a.check(event['padding']==after['actual_geometry']['padding'] and event['baseline_padding']==[0,0,0] and
                all(event[k]==after['actual_geometry'][k] for k in ('stride','dilation','groups')) and event['native_calls']==1 and
                event['complete_input_weight_bias_identity']==after['inputs_before'] and
                event['complete_output_identity']==after['output'] and event['path']==after['path'] and event['ordinal']==0 and
                event['applied'] is (index in FAULTS),'EVENT_CORROBORATED_BY_INDEPENDENT_LOWER_TRACE',phase)
            cleanup=row['adapter_cleanup']
            a.check(cleanup['cleanup_confirmed'] is cleanup['ownership_verified'] is True and
                cleanup['enabled'] is cleanup['hook_owned'] is False and cleanup['inflight']==cleanup['leases']==0,
                'PADDING_ADAPTER_RESOURCES_RELEASED',phase)
        a.check(len(driver['output_token_ids'])==1 and type(driver['output_token_ids'][0]) is int and
            isinstance(driver['text'],str),'ONE_ACTUAL_RETURNED_TOKEN',phase)
        summaries.append(dict(phase=phase,request_id=rid,actual_projection_calls=1,final_padding=after['actual_geometry']['padding'],
            projection_changed=changed,consumer_changed=consumer_changed,logits_changed=logits_changed,
            returned_token_changed=driver['output_token_ids']!=requests[0]['output_token_ids'],
            returned_text_changed=driver['text']!=requests[0]['text'],logits_sha256=C.digest(row['logits'])))
    return summaries


def coverage(a,d,*,natural):
    rows=d['phase_rows'];reports=d['coverage'];discovery=reports['discovery'];candidates=discovery['candidate_set']
    candidate,=candidates['candidates'];target=candidate['identity'];worker=d['last_scope']['worker']
    before=rows[1]['initial_observation']['observed_calls'][0];tid=C.digest(target)
    expected=dict(candidates=[dict(candidate_id=before['path']+':0',eligibility=dict(status='SUPPORTED',reasons=[]),identity=target)],
        discovered=1,discovery_complete=True,eligible=1,execution_authorized=False,
        eligibility_counts={name:int(name=='SUPPORTED') for name in ('SUPPORTED','UNSUPPORTED','UNVERIFIED','AMBIGUOUS','NOT_OBSERVED','RESOURCE_LIMITED')})
    a.check(candidates==expected,'ONE_COMPLETE_ACTUAL_CANDIDATE_NOT_MULTI_TARGET')
    a.check(target['module_object_identity']==before['module_object_id'] and target['operator_api']==before['operator'] and
        target['tensor_signature']==json.dumps(signature(before),sort_keys=True) and
        target['module_roles']==['CONV3D_SPATIAL_PADDING_ARGUMENT'] and target['provider_identity']=='vision17_public_torch_parameter' and
        target['provider_proof']==[['implementation_sha256',discovery['implementation_sha256']]] and
        target['boundary']['kind']=='python_hook' and target['boundary']['call_ordinal']==0 and
        C.sha(target['boundary']['callable_source_sha256']) and bool(target['boundary']['callable_identity']),
        'ACTUAL_PROJECTION_TARGET_AND_RECORDED_CALLABLE')
    a.check(target['scope_kind']=='INITIALIZED_WORKER' and target['process_id']==worker['pid'] and
        target['process_start_identity']==worker['process_start_identity'] and target['worker_id']==worker['worker_start_identity'] and
        target['model_instance_id']==worker['model_instance_id'] and target['rank']==worker['rank'] and
        target['device_type']==('npu' if natural else 'cpu') and target['logical_device']==(0 if natural else None),
        'TARGET_EXACT_ACTUAL_WORKER_LIFETIME')
    if natural:
        marker=re.fullmatch(r'proc:([^:]+):([1-9][0-9]*):([0-9]+)',worker['process_start_identity'])
        a.check(marker is not None and int(marker[2])==worker['pid'] and worker['rank']==0 and
            worker['logical_device']==worker['configured_logical_device']==worker['observed_logical_device']==0,
            'NATURAL_NPU_WORKER_PROCESS_SCOPE')
    target_worker=[target[k] for k in ('process_id','process_start_identity','worker_id','rank','device_type','logical_device')]
    a.check(discovery['implementation_sha256']==C.digest(discovery['source_manifest']),'RECORDED_IMPLEMENTATION_MANIFEST_DIGEST')
    baseline=C.digest(outputs(rows[0]));summary={}
    for name,indices in (('C2',(4,5)),('C3',(6,7,8))):
        cov=reports[name];group_rows=[rows[i] for i in indices];faults=len(indices)-1
        scope=cov['scope'];owner=cov['owner'];gen=cov['generation'];saved_discovery=cov['discovery']
        a.check(cov['phase_rows']==group_rows and cov['selection']=='all_eligible' and
            cov['trigger']==('once' if name=='C2' else 'every_match') and C.positive(gen) and
            cov['actual_target_count']==cov['planned_targets']==cov['planned_targets_hit']==1 and
            cov['budget']==dict(max_calls=faults,max_seconds=600,max_concurrency=1), 'COMPLETE_SINGLE_TARGET_GROUP_SCOPE',name)
        a.check(scope==dict(mechanism='17.spatial_padding',workload_sha256=d['setup']['workload_sha256'],
            environment_sha256=d['setup']['environment_sha256'],implementation_sha256=discovery['implementation_sha256'],
            evidence_kind='natural_worker' if natural else 'cpu_contract',model_weights_sha256=d['input']['weights_sha256'],
            model_source_sha256=rows[0]['request_binding']['actual_frame_sources']['embed_multimodal']['source_sha256']),
            'EXACT_WORKLOAD_ENVIRONMENT_MODEL_SCOPE',name)
        a.check(saved_discovery==dict(artifact_sha256=C.digest(rows[:4]),candidates_sha256=C.digest(candidates),complete=True,
            expected_workers=[target_worker],implementation_sha256=scope['implementation_sha256'],observed_target_ids=[tid],
            observed_workers=[target_worker],observer_id=saved_discovery['observer_id'],source='independent_workload_observer',
            workload_sha256=scope['workload_sha256']) and bool(saved_discovery['observer_id']),
            'ORIGINAL_HEALTHY_DISCOVERY_REFERENCE',name)
        lease=dict(target=target,owner=owner,generation=gen,lease_id='target-0',device_bytes=0,host_bytes=0)
        tickets=[];a.check(len(cov['calls'])==len(group_rows),'ALL_ACTUAL_F_AND_OFF_CALLS',name)
        for index,(call,row) in enumerate(zip(cov['calls'],group_rows)):
            rid=row['request_id'];hit=index<faults;phase='PRE_FAULT_PROPAGATION' if hit else 'OFF_RECOVERY'
            ticket=dict(observer_id=saved_discovery['observer_id'],sequence=index,eligible=True,propagation_added=False,
                call=dict(target=target,call_id=rid+':0',forward_context=rid,request_ids=[rid],phase=phase))
            tickets.append(ticket)
            decision=dict(session_id=cov['session_id'],ticket=ticket,lease=lease,admitted=hit,reason='ADMITTED' if hit else 'OFF_RECOVERY')
            after=row['conv_observation']['observed_calls'][0]
            receipt=dict(decision=decision,outcome='HIT' if hit else 'PASSTHROUGH',native_applied=hit,original_calls=1,
                execution_sha256=C.digest(after),effect_sha256=C.digest(dict(consumer=after['consumer'],output=outputs(row))) if hit else None)
            a.check(call==dict(call_id=rid+':0',forward_id=rid,request_ids=[rid],model_instance=target['model_instance_id'],
                operator=target['operator_api'],target_id=tid,rank=target['rank'],device=target['logical_device'],worker=target_worker,
                owner=owner,generation=gen,phase=phase,eligible=True,propagation_added=False,decision=decision,receipt=receipt),
                'EXACT_ORIGINAL_CALL_EFFECT_RECEIPT',name+':'+str(index))
        a.check(cov['hits']==cov['observed_eligible_calls']==faults and cov['observed_calls']==len(group_rows) and
            not cov['missing_outcomes'] and not cov['propagation_added_calls'] and
            cov['observation']==dict(artifact_sha256=C.digest(group_rows),calls_sha256=C.digest(tickets),complete=True,
                dropped_calls=0,observed_calls=len(group_rows)), 'INDEPENDENT_RECOMPUTED_CALL_TOTALS',name)
        recovery=cov['recovery'];off=group_rows[-1]
        a.check(recovery==dict(complete=True,errors=[],generation=gen,held_leases=[],inflight=0,owner=owner,pending_drains=[],
            pending_leases=[],state='DISABLED_CONFIRMED',targets=[dict(drained=True,lease=lease,off_exact=True,
                off_output_sha256=baseline,pending_reason=None,proof_sha256=C.digest(off),released=True)],
            workers=[dict(active_leases=0,baseline_output_sha256=baseline,device_bytes=0,generation=gen,host_bytes=0,
                inflight_calls=0,off_output_sha256=C.digest(outputs(off)),owned_hooks=0,owner=owner,
                proof_sha256=C.digest(off),worker=target_worker)]),'EXACT_OFF_RELEASE_AND_WORKER_HEALTH_REFERENCES',name)
        a.check(all(cov[key] is False for key in ('aclnn_launch_arguments','device_payload_observed','native_qualification','native_loaded',
            'E5','multi_target_verified','public_qualification_granted')) and cov['execution_boundary']=='python_hook' and
            cov['invocation_scope']=='INITIALIZED_WORKER','COVERAGE_RETAINS_TORCH_SINGLE_PROJECTION_LIMIT',name)
        summary[name]=dict(actual_requests=len(group_rows),actual_f_requests=faults,actual_targets=1,recomputed_hits=faults,
                           generation=gen,original_canonical_references_verified=True)
    one,two=reports['C2'],reports['C3'];issued,=one['live_c2_proof']['issued'];consumed=two['live_c2_proof']['consumed']
    proof=issued['proof'];base={k:one[k] for k in BASE_KEYS}
    a.check(one['live_c2_proof']['consumed'] is None and proof==dict(scope=one['scope'],owner=one['owner'],generation=one['generation'],
        session_id=one['session_id'],target_ids=[tid],candidates_sha256=C.digest(candidates),discovery_sha256=C.digest(rows[:4]),
        observation_sha256=C.digest(rows[4:6]),recovery_sha256=C.digest(one['recovery']),report_sha256=C.digest(base)) and
        issued['proof_sha256']==C.digest(proof) and issued['object_binding']['pid']==worker['pid'] and
        C.positive(issued['object_binding']['python_object_id']) and issued['exact_issued_object_verified'] is True and
        issued['serializable_authority'] is issued['qualification_granted'] is False,'ORIGINAL_C2_PROOF_SCOPE_AND_HASH_NOT_AUTHORITY')
    a.check(two['live_c2_proof']['issued']==[] and {k:v for k,v in consumed.items() if k!='consumer'}==issued and
        consumed['consumer']==dict(scope=two['scope'],owner=two['owner'],generation=two['generation'],session_id=two['session_id'],
            candidates_sha256=C.digest(candidates),target_ids=[tid]) and one['scope']==two['scope'] and one['owner']==two['owner'] and
        two['generation']==one['generation']+1 and two['session_id']!=one['session_id'],'SAME_LIVE_PROOF_OBJECT_NEXT_GENERATION_RECORDED')
    before_issuance=deepcopy(one);before_issuance['live_c2_proof']['issued']=[]
    a.check(reports['C1']['proof_report_sha256']==C.digest(before_issuance) and
        reports['C1']['actual_target_count']==1 and reports['C1']['native_qualification'] is False,'C1_REFERENCE_BINDS_RECORDED_C2')
    health=reports['same_worker_healthy'];last=rows[-1]
    a.check(health['worker']==worker and health['request_id']==last['request_id'] and health['fault_hooks']==0 and
        health['cleanup_before_request'] is True and outputs(last)==outputs(rows[0]) and
        reports['last_request']==dict(phase='cleanup_healthy',request_id=last['request_id'],cleanup=dict(
            fault_injections=0,held_tensors=0,owned_hooks=0,owned_modes=0,qualification_granted=False)),
        'ACTUAL_CLEANUP_HEALTH_SAME_WORKER')
    return summary


def public_protocol(a,d):
    history=d['rpc'];host=d['last_scope'];worker=host['worker'];rows=d['phase_rows']
    expected=['attach','prepare','register','check','enable']+['advance']*20+['finish','unregister','detach']
    a.check([x['action'] for x in history]==expected,'EXACT_EMBEDDED_PUBLIC_RPC_HISTORY')
    attachment=history[0]['response'];ticket=history[1]['response']['ticket'];private_owner=None
    a.check(attachment['status']=='ATTACHED' and attachment['ticket']['worker']==worker and
        all(ticket[k]==v for k,v in attachment['ticket'].items()) and ticket['owner']==host['owner'] and
        ticket['mechanism']=='17.spatial_padding' and C.positive(ticket['generation']) and
        ticket['tenant_id']==host['tenant_id'],'ATTACH_AND_RUNTIME_TICKET_JOIN')
    for index,item in enumerate(history[1:-1],start=1):
        reply=item['response'];state=reply['provider_state'];serving=state['serving'];private_owner=private_owner or state['owner']
        a.check(reply['status']=='ACTION_COMPLETED' and reply['action']==item['action'] and reply['ticket']==ticket and
            reply['native'] is None and state['owner']==private_owner and state['generation']==ticket['generation'],
            'ORIGINAL_HOST_PROVIDER_READBACK',index)
        a.check(state['inflight']==state['leases']==0 and serving['error'] is None and not serving['failures'] and
            state['mechanism']=='17.spatial_padding' and state['execution_boundary']=='python_hook' and
            state['simulation'] is d['setup']['simulation'] and
            all(state[k] is False for k in ('native_loaded','native_registered','aclnn_launch_arguments','device_payload_observed',
                'native_qualification','E5','multi_target_verified','public_qualification_granted')),
            'PUBLIC_READBACK_RETAINS_TORCH_SCOPE_NO_FAILURE',index)
        if 5<=index<25:
            offset=index-5;request_index=offset//2;collected=offset%2==1;count=request_index+int(collected)
            row=rows[request_index]
            a.check(serving['completed_requests']==count and state['phase_rows']==rows[:count] and
                serving['pending_request']==(None if collected else [row['request_id'],row['phase']]) and
                state['hook_owned'] is (not collected),'ARM_COLLECT_JOINS_ACTUAL_REQUEST_PREFIX',index)
            if collected:
                last=state['reports']['last_request']
                a.check(last['request_id']==row['request_id'] and last['phase']==row['phase'] and
                    all(last['cleanup'][k]==0 for k in ('owned_hooks','owned_modes','held_tensors','fault_injections')),
                    'EACH_REQUEST_OWNED_RESOURCES_RELEASED',index)
        elif index<5:
            a.check(state['phase_rows']==[] and serving['completed_requests']==0,'CONTROL_ACTION_DOES_NOT_RUN_MODEL',index)
        else:
            a.check(state['phase_rows']==rows and state['reports']==d['coverage'] and serving['completed_requests']==10 and
                serving['pending_request'] is serving['next_phase'] is None and state['hook_owned'] is False and
                state['coverage_target_leases_held'] is False and state['replay_pending'] is False,
                'FINAL_COMPLETE_REQUESTS_AND_ZERO_GROUP_RESOURCES',index)
    unregister=history[-2]['response']['provider_state']
    a.check(unregister['registered'] is unregister['binding_reserved'] is unregister['enabled'] is False and
        history[-1]['response']==dict(status='DETACHED',recoveries=[]) and host['response']==history[-1]['response'] and
        host['status']=='DETACHED' and host['ownership_retained'] is False and
        host['attachment'] is host['ticket'] is host['census_ticket'] is None and
        d['binding_still_owned'] is False and d['unbind']==dict(status='VISION17_GROUP_BINDING_REMOVED'),
        'EXPLICIT_UNREGISTER_DETACH_AND_DEPLOYMENT_REMOVAL')
    a.check(d['coverage']['C2']['owner']==d['coverage']['C3']['owner']==private_owner and
        d['coverage']['C2']['generation']==ticket['generation'],'COVERAGE_BINDS_PUBLIC_CORE_OWNER_GENERATION')
    return dict(actions=len(history),actual_enqueued_requests=10,scope='HOST_RECORDED_RPC_HISTORY',
                independent_transport_receipts_available=False)


def source_review(a,d,source_root,manifest_sha,commit):
    path=C.under(source_root,'source_manifest.json');manifest=C.read(path)
    a.check(C.sha(manifest_sha) and C.file_sha(path)==manifest_sha and manifest['base_commit']==commit and
        re.fullmatch('[0-9a-f]{40}',commit) is not None,'EXACT_FROZEN_SOURCE_MANIFEST')
    files={v['path']:v for v in manifest['files']};a.check(len(files)==len(manifest['files']),'UNIQUE_SOURCE_MANIFEST_PATHS')
    declared=d['coverage']['discovery']['source_manifest'];expected={name:'fault_injection/'+name for name in SOURCE_NAMES}
    expected.update({name:('src/' if name.startswith('demoncat.') else '')+name.replace('.','/')+'.py' for name in SOURCE_MODULES})
    a.check(set(declared)==set(expected),'ALL_IMPLEMENTATION_SOURCE_IDENTITIES')
    checked={}
    def verify(name,expected_sha=None):
        path=C.under(source_root,name);record=files[name];sha=C.file_sha(path)
        a.check(sha==record['sha256'] and path.stat().st_size==record['bytes'] and
            (expected_sha is None or sha==expected_sha),'ORIGINAL_SOURCE_BYTES_AND_RECORDED_IDENTITY',name)
        checked[name]=sha
    for name,relative in expected.items():verify(relative,declared[name])
    verify(PROBE,d['probe_sha256'])
    # These were source inputs to the actual public driver/runtime. Their disk
    # fingerprint is checked; no claim is made that JSON proves a loaded module.
    for relative in ('experiments/worker_core_v11_20261118/vision17_cache_control.py',
            'experiments/c2c3_closure_20261117/vision17_worker_census_probe.py',
            'experiments/c2c3_closure_20261117/natural_tensor_worker_core_probe.py',
            'experiments/c2c3_closure_20261117/natural_vision_input.py',
            *('src/demoncat/runtime/vllm_ascend/worker_extension/'+name+'.py' for name in
              ('host','runtime','identity','config','registry','serving_context','providers/_scope'))):verify(relative)
    a.check(C.digest(declared)==d['coverage']['discovery']['implementation_sha256'],'SOURCE_BOUND_IMPLEMENTATION_DIGEST')
    return dict(source_commit=commit,source_manifest_sha256=manifest_sha,files_rehashed=len(checked),
        probe_sha256=checked[PROBE],implementation_sha256=C.digest(declared),
        runtime_framework_sources_rehashed=False,source_only_not_loaded_module_attestation=True)


def review(d,*,natural=True):
    a=C.Audit();details={}
    extracted=d.get('extract_only') or d.get('extraction_only') or d.get('source_original',{}).get('extraction_only')
    if extracted:
        a.check(False,'EXTRACT_ONLY_IS_NOT_COMPLETE_ORIGINAL')
        return dict(**a.report(),originals_reconciled=False,classification='EXTRACT_REJECTED_NOT_ORIGINAL')
    failed=any(d.get(k) for k in ('error','cleanup_error','abort_error'))
    if failed:
        return dict(**a.report(),originals_reconciled=False,classification='FAILED_OR_PARTIAL_NOT_QUALIFIED',
            original_error=d.get('error'),original_cleanup_error=d.get('cleanup_error'),
            recorded_phase_count=len(d.get('phase_rows',[])),recorded_request_count=len(d.get('requests',[])))
    details['phases']=a.section('actual_phases',phases,a,d,natural=natural)
    details['coverage']=a.section('actual_coverage',coverage,a,d,natural=natural)
    details['public_protocol']=a.section('embedded_public_protocol',public_protocol,a,d)
    def input_scope():
        binding=d['phase_rows'][0]['request_binding'];setup=d['setup'];tokens=binding['prompt_token_ids']
        a.check(all(type(t) is int and t>=0 for t in tokens) and C.sha(d['input']['image_sha256']) and
            C.sha(d['input']['weights_sha256']) and C.sha(d['input']['config_sha256']) and
            setup['simulation'] is (not natural) and setup['hooks_installed']==0 and setup['qualification_granted'] is False and
            setup['workload_sha256']==C.digest(dict(prompt=tokens,image_sha256=d['input']['image_sha256'])),
            'IMAGE_PROMPT_WEIGHT_AND_INERT_SETUP_IDENTITY')
        actual_env=hashlib.sha256(json.dumps(setup['environment'],sort_keys=True).encode()).hexdigest()
        a.check(actual_env==setup['environment_sha256'],'RECORDED_ENVIRONMENT_DIGEST')
        a.check(d['input']['processor_overrides']=={} and d['input']['raw_PIL_image'] is True,'NATURAL_IMAGE_PROCESSOR_WITHOUT_OVERRIDES')
        frames=binding['actual_frame_sources']
        a.check(set(frames)=={'_execute_mm_encoder','_batch_mm_inputs_from_scheduler','embed_multimodal','_process_image_input'} and
            all(v['method']==key and C.sha(v['source_sha256']) and C.positive(v['first_line']) and v['module'] and v['qualname']
                for key,v in frames.items()),'PINNED_ACTUAL_MULTIMODAL_FRAME_SOURCES')
        if natural:
            geometry=d['input']['geometry'];call=d['phase_rows'][0]['conv_observation']['observed_calls'][0]
            a.check(all(C.positive(geometry[k]) for k in ('patch_size','temporal_patch_size','in_channels','hidden_size','spatial_merge_size')) and
                call['inputs_before'][1]['shape']==[geometry['hidden_size'],geometry['in_channels'],geometry['temporal_patch_size'],
                    geometry['patch_size'],geometry['patch_size']] and call['actual_geometry']==dict(stride=[geometry['temporal_patch_size'],
                    geometry['patch_size'],geometry['patch_size']],padding=[0,0,0],dilation=[1,1,1],groups=1),
                'ACTUAL_PRETRAINED_PATCH_GEOMETRY')
    a.section('input_scope',input_scope)
    return dict(**a.report(),details=details,originals_reconciled=not a.issues,
        classification='SAVED_WORKER_TORCH_PARAMETER_SINGLE_PROJECTION' if natural else 'CPU_CONTRACT_ONLY',
        actual_target_count=1,multi_target_verified=False,aclnn_launch_arguments=False,device_payload_observed=False,native_qualification=False,
        limits=['Saved original consistency does not issue or reconstruct a live C2/C3 capability.',
            'Upper/lower Torch dispatch and exact-object first-consumer lineage are checked; ACLNN launch and device payload remain unobserved.',
            'Full tensor hashes are original runtime readbacks; this auditor does not read private tensors or recompute argmax.',
            'Returned first-token/text comparisons use the original request mapping. There is no continuous-generation or multi-target claim.',
            'RPC evidence is the host history embedded in the original result, not separately captured transport receipts.',
            'Package/frame source hashes are retained; installed framework files and model weights are not rehashed by this auditor.',
            'E5 requires a separate raw statistics comparison and is not inferred from mechanism logs or output hashes.'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True);parser.add_argument('--result-sha256',required=True)
    parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--source-manifest-sha256',required=True)
    parser.add_argument('--source-commit',required=True)
    args=parser.parse_args(argv);sys.dont_write_bytecode=True;sys.addaudithook(C.deny_runtime)
    a=C.Audit();document=None
    try:
        if not C.sha(args.result_sha256) or C.file_sha(args.result)!=args.result_sha256:
            raise ValueError('COMPLETE_ORIGINAL_RESULT_SHA_MISMATCH')
        document=C.read(args.result);result=review(document)
        if result['classification']!='EXTRACT_REJECTED_NOT_ORIGINAL' and document.get('coverage',{}).get('discovery'):
            provenance=a.section('source_provenance',source_review,a,document,args.source_root,args.source_manifest_sha256,args.source_commit)
            result['provenance']=dict(**a.report(),details=provenance)
            result['consistent']=bool(result['consistent'] and not a.issues)
        else:result['consistent']=False
        # Detect file replacement during parsing/reconciliation as well.
        if C.file_sha(args.result)!=args.result_sha256:raise ValueError('ORIGINAL_CHANGED_DURING_REVIEW')
        result['originals_reconciled']=bool(result.get('originals_reconciled') and result['consistent'])
    except (ValueError,KeyError,TypeError,OSError,IndexError,AttributeError) as error:
        a.check(False,'ORIGINAL_REVIEW_REFUSED',repr(error));result=dict(**a.report(),originals_reconciled=False)
    result.update(schema='demoncat.vision17.worker.independent_originals.v1',reviewer_sha256=C.file_sha(__file__),
        common_reviewer_sha256=C.file_sha(Path(__file__).with_name('group_review_common.py')),
        original_result_sha256=args.result_sha256,read_only=True,runtime_executed=False)
    print(json.dumps(result,indent=2,allow_nan=False));return 0 if result['consistent'] and result['originals_reconciled'] else 2


if __name__=='__main__':raise SystemExit(main())
