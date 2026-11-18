"""Independent raw C1 review; no injector/framework/DSO import or device call.

Reuses the normal raw byte, ELF, ABI and complete CPU SplitCore readers. It
never imports a provider or treats C1/status/contract flags as evidence.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path,PurePosixPath
import re
import struct
import sys
from types import SimpleNamespace

import arch22_normal_bridge_evidence as E
import review_arch22_normal_bridge as N

PHASES=('reference','lossless','qualification_off','fault','recovery_off')
LAYERS=tuple(f'model.layers.{n}.self_attn.attn' for n in range(28));FIRST=LAYERS[0]
ACTIONS=['attach','prepare','register','check','enable']+['advance','query','advance']*5+['finish','disable','unregister','detach']
need=E.need


def local(run,path):
    value=PurePosixPath(path)
    need('..' not in value.parts and value.is_relative_to('/evidence/probe'),'S2_RAW_PATH_OUTSIDE_OWNED_RUN')
    return E.child(run,str(value.relative_to('/evidence/probe')))


def artifact(run,reference,*,limit=1024**3):
    path=E.child(run,reference['path'])
    need(E.sha(path)==reference['sha256'] and ('bytes' not in reference or reference['bytes']==path.stat().st_size),
         'S2_ORIGINAL_ARTIFACT_HASH_DIFFERS')
    return E.read(path,limit=limit)


def normalized(signature):
    value=deepcopy(signature);table=value['block_table_values'];geometry=value['cache_geometry'];pages=table[0][:65]
    need(len(table)==1 and len(table[0])==80 and len(pages)==len(set(pages))==65
        and geometry['key']['shape']==geometry['value']['shape']==[128,16,8,128]
        and all(type(p) is int and 0<=p<128 for p in table[0])
        and value['writer_consumer_slot']==[pages[-1]*16],'S2_INVALID_COMPLETE_LOGICAL_PAGE_MAP')
    need(value['tensors']['block_table']['sha256']==hashlib.sha256(struct.pack('<80i',*table[0])).hexdigest()
        and value['tensors']['block_table']['bytes']==320,'S2_COMPLETE_RAW_PAGE_TABLE_HASH_DIFFERS')
    table[0][:65]=list(range(65));value['tensors']['block_table']['sha256']='PROVED_USED_PAGES_ONLY'
    value['writer_consumer_slot']=['PROVED_CURRENT_LOGICAL_SLOT'];return value


def cleanup(state):
    need(all(state[key]==0 for key in ('inflight','leases','owned_hooks','owned_api_wrappers'))
        and not any(state[key] for key in ('enabled','native_registered','replay_pending','replay_failed')),
        'S2_FINAL_PUBLIC_RESOURCES_NOT_ZERO')
    native=state['native_state'];resource=state['serving']['resource_state']
    need(native['scope_owned'] is native['scope_active'] is False and native['environment_owned']==0
        and native['retained_input_objects']==resource['owned_input_references']==0
        and native['effect_attempted'] is native['off_confirmed'] is True
        and native['first_error'] is None and native['failures']==[], 'S2_ACTUAL_NATIVE_OFF_OR_REFERENCE_RELEASE_MISSING')
    need(state['serving']['pending_request'] is resource['pending_request'] is None
        and state['serving']['failures']==resource['failures']==resource['failed_requests']==[]
        and resource['needs_off'] is False,'S2_PENDING_OR_FAILED_REQUEST_NOT_CLEAN')


def protocol(run,data,*,allow_cpu_contract=False,group_phases=None):
    phases=tuple(group_phases) if group_phases is not None else PHASES
    count=len(phases);group=group_phases is not None
    actions=['attach','prepare','register','check','enable']+['advance','query','advance']*count+['finish','disable','unregister','detach']
    last=4+3*count
    host=artifact(run,data['host_receipt']);history=host['history'];refs=data['worker_core']['receipts']
    need(host['schema']=='demoncat.worker_core.host_receipt.v1' and [r['action'] for r in history]==actions
        and len(refs)==len(actions),'S2_COMPLETE_ORIGINAL_PUBLIC_PROTOCOL_REQUIRED')
    for ref,row in zip(refs,history):need(ref['action']==row['action'] and artifact(run,ref)==row,'S2_RAW_RPC_HISTORY_DIFFERS')
    requests=data['requests'];need([r['phase'] for r in requests]==list(phases),'S2_FIVE_ACTUAL_REQUESTS_REQUIRED')
    ticket=history[1]['response']['ticket'];owner=history[1]['response']['provider_state']['owner'];generation=ticket['generation']
    need(host['worker']==data['setup']['worker']==ticket['worker'] and host['owner']==ticket['owner']
        and host['tenant_id']==ticket['tenant_id'] and ticket['mechanism']=='33.paged_decode'
        and type(generation) is int and generation>0 and re.fullmatch('[0-9a-f]{32}',owner)
        and owner!=host['owner'],'S2_HOST_RUNTIME_PRIVATE_OWNER_JOIN_CHANGED')
    attached=history[0]['response'];need(attached['status']=='ATTACHED'
        and all(attached['ticket'][k]==ticket[k] for k in ('tenant_id','attach_nonce','worker')),'S2_ORIGINAL_ATTACH_CHANGED')
    installed=history[4]['response']['provider_state'];counts=[]
    simulation=installed['simulation'];need(type(simulation) is bool and (not simulation or allow_cpu_contract),
        'CPU_CONTRACT_IS_NOT_NATIVE_QUALIFICATION')
    for index,item in enumerate(history[1:-1],1):
        response=item['response'];state=response['provider_state'];active=4<=index<=last;reserved=2<=index<=last+2
        complete=max(0,min(count,(index-4)//3));pending=None
        if 5<=index<=last and (index-5)%3 in (0,1):
            n=(index-5)//3;pending=[requests[n]['mapping']['internal_request_id'],phases[n]]
        phase='PREPARED' if index==1 else 'REGISTERED' if index==2 else 'CHECKED' if index==3 else (
            'ATTEMPTED' if index<=last else 'FINISHED' if index==last+1 else 'STOPPED' if index==last+2 else 'UNREGISTERED')
        need(response['ticket']==ticket and response['phase']==phase and response['recovery_required'] is False
            and response['status']==('QUERY_OK' if item['action']=='query' else 'ACTION_COMPLETED')
            and state['owner']==owner and state['generation']==generation and state['simulation'] is simulation
            and response['native']==state and (item['action']=='query' or response['provider']==state),
            'S2_ORIGINAL_PUBLIC_STATE_OWNER_OR_PHASE_DIFFERS:'+str(index))
        need(state['execution_boundary']=='native' and state['native_loaded'] is True
            and state['registered'] is reserved and state['binding_reserved'] is reserved
            and state['replay_pending'] is active and state['native_registered'] is active and state['replay_failed'] is False
            and state['enabled'] is False and state['inflight']==state['leases']==0
            and state['C3'] is False and (group or state['C2'] is False),'S2_PROVIDER_RESOURCE_LIFECYCLE_DIFFERS:'+str(index))
        resource=state['serving']['resource_state'];native=state['native_state']
        actual_calls=max(0,min(count,(index-3)//3))*(28 if group else 1);held=actual_calls if active else 0
        need(all(state[k]==resource[k] and (state[k]==installed[k]>0 if active else state[k]==0)
                 for k in ('owned_hooks','owned_api_wrappers'))
            and native['scope_active'] is native['scope_owned'] is False and native['environment_owned']==0
            and native['first_error'] is None and native['failures']==[]
            and native['calls']==actual_calls and native['retained_input_objects']==resource.get('owned_input_references',0)==held,
            'S2_ACTUAL_IDLE_SCOPE_OR_HOOKS_DIFFERS')
        need(state['serving']['pending_request']==resource['pending_request']==pending
            and state['serving']['next_phase']==('complete' if complete==count else phases[complete])
            and resource['completed_requests']==complete
            and state['evidence']['requests']==[r['worker'] for r in requests[:complete]],
            'S2_QUERY_PURITY_OR_ACTUAL_ROW_COLLECTION_DIFFERS:'+str(index))
        if item['action']=='query':need(response['quarantined'] is False,'S2_QUERY_QUARANTINED')
        counts.append(dict(action=item['action'],completed=complete,pending=pending))
    final=history[last+1]['response']['provider_state'];need(final==data['final_cleanup'],'S2_ORIGINAL_FINISH_CHANGED');cleanup(final)
    unregistered=history[-2]['response']['provider_state'];cleanup(unregistered)
    need(not unregistered['registered'] and not unregistered['binding_reserved'] and history[-1]['response']['status']=='DETACHED'
        and host['ticket'] is host['attachment'] is host['census_ticket'] is None and host['ownership_retained'] is False
        and data['worker_core']['ownership_retained'] is False,'S2_UNREGISTER_DETACH_NOT_CONFIRMED')
    need(data['binding_removal']['status']=='SERVING_DEPLOYMENT_REMOVED'
        and data['binding_removal']['mechanism']=='33.paged_decode'
        and data['binding_removal']['native_scope_active'] is False,'S2_BINDING_REMOVAL_NOT_CONFIRMED')
    health=data['post_unregister_health'];need(health['scope']=='TOKENS_TEXT_ONLY_NO_POST_UNREGISTER_HIDDEN_OR_LOGITS_OBSERVER'
        and health['response']['outputs']==data['warm_healthy']['outputs'],'S2_POST_UNREGISTER_TOKEN_HEALTH_DIFFERS')
    ids=[r['mapping']['internal_request_id'] for r in requests]+[health['mapping']['internal_request_id']]
    need(len(set(ids))==count+1 and all(type(x) is str and x for x in ids),'S2_FRESH_ACTUAL_REQUEST_IDS_REQUIRED')
    return dict(worker=host['worker'],provider_owner=owner,host_owner=host['owner'],generation=generation,
                actions=counts,simulation=simulation,qualification_flags_used_as_proof=False)


def raw_tensor(run,row,shape,dtype,helper,expected_path):
    meta=row.get('metadata',row)
    need(row['path']==expected_path and meta['shape']==shape and meta['dtype']==dtype
        and row['finite'] is True and row['bytes']==math.prod(shape)*(2 if dtype=='torch.float16' else 4),
        'S2_COMPLETE_PAYLOAD_GEOMETRY_OR_PATH_DIFFERS')
    path=local(run,row['path']);need(path.stat().st_size==row['bytes'] and E.sha(path)==row['sha256'],'S2_RAW_PAYLOAD_BYTES_CHANGED')
    value=dict(path=str(path),shape=shape,dtype=dtype,bytes=row['bytes'],sha256=row['sha256'])
    if row['bytes']:helper.differences.raw_difference(value,value)  # independently verify every finite value
    return value


def actual_integer_input(forward,metadata,name,values,dtype,device):
    """Reconcile CPU row values with the actual root tensor's complete hash."""
    need(type(values) is list and all(type(value) is int for value in values),
         'S2_ACTUAL_INPUT_VALUES_NOT_INTEGER_ROWS')
    packed=struct.pack('<'+('i' if dtype=='torch.int32' else 'q')*len(values),*values)
    actual=forward['input']
    need(type(actual) is list and len(actual)==2 and actual[0]==[]
        and set(actual[1])=={'input_ids','positions','inputs_embeds','intermediate_tensors'}
        and actual[1]['inputs_embeds'] is actual[1]['intermediate_tensors'] is None,
        'S2_ACTUAL_ROOT_INPUT_BINDING_DIFFERS')
    need(metadata['dtype']==dtype and metadata['device']==device
        and metadata['shape']==[len(values)] and metadata['stride']==[1] and metadata['storage_offset']==0
        and all(type(metadata[key]) is int and metadata[key]>0 for key in ('object_id','storage_identity'))
        and actual[1][name]==dict(shape=[len(values)],dtype=dtype,finite=True,sha256=hashlib.sha256(packed).hexdigest()),
        'S2_CPU_ROWS_TO_ACTUAL_ROOT_TENSOR_HASH_DIFFERS:'+name)


def request_lineage(request,prompt,device):
    """Read the two actual scheduler schemas, without inventing common keys."""
    rid=request['mapping']['internal_request_id'];phase=request['phase'];row=request['worker']
    need(request['completed'] is True and type(rid) is str and rid
        and type(request['mapping']['request_state_object_id']) is int and request['mapping']['request_state_object_id']>0
        and request['mapping']['source']=='ACTUAL_PUBLIC_ENQUEUE_AND_OUTPUT_PROCESSOR'
        and request['response']['request_id']==request['mapping']['external_request_id']
        and request['response']['prompt_token_ids']==prompt and row['request_id']==rid and row['phase']==phase,
        'S2_ACTUAL_REQUEST_MAPPING_DIFFERS')
    forwards=row['forwards'];need(len(forwards)==2 and [x['decode'] for x in forwards]==[False,True],
        'S2_PREFILL_FIRST_DECODE_REQUIRED')
    prefill,decode=forwards;ps,ds=prefill['scheduler'],decode['scheduler']
    need('request_ids' not in ps and 'internal_request_id' not in ds,'S2_SCHEDULER_SCHEMA_DOMAINS_CHANGED')
    need(type(prompt) is list and len(prompt)==1024 and all(type(n) is int and n>=0 for n in prompt)
        and ps['internal_request_id']==rid and ps['external_request_id'] is None
        and ps['num_requests']==1 and ps['num_prompt_tokens']==ps['num_scheduled_tokens']==1024
        and ps['num_computed_tokens']==0 and ps['token_ids']==prompt and ps['positions']==list(range(1024))
        and ps['token_ids_sha256']==hashlib.sha256(json.dumps(prompt,separators=(',',':')).encode()).hexdigest()
        and ps['token_count_evidence']['request_num_tokens']==ps['token_count_evidence']['batch_num_tokens_no_spec']==1024
        and ps['token_count_evidence']['legacy_field_role']=='POST_SAMPLING_COUNTER_NOT_PREFILL_AUTHORITY',
        'S2_FRESH_PREFILL_REQUEST_TOKEN_OR_POSITION_DIFFERS')
    # batch_num_tokens_legacy is explicitly non-authoritative in the captured
    # producer. It may contain a prior request's post-sampling count.
    need(ps['attention_metadata_keys']==list(LAYERS) and set(ps['attention_metadata_evidence'])==set(LAYERS)
        and all(value==dict(adapter='DENSE_EXPLICIT_LENGTHS_AND_CAUSAL',actual_seq_lengths_q=[1024],
                seq_lens_list=[1024],causal=True) for value in ps['attention_metadata_evidence'].values()),
        'S2_PREFILL_ALL_ACTUAL_METADATA_ROWS_DIFFERS')
    pages=ps['block_ids']
    need(ps['block_size']==16 and len(pages)==len(set(pages))==64
        and all(type(page) is int and 0<=page<128 for page in pages)
        and ps['slot_mapping']==[pages[position//16]*16+position%16 for position in ps['positions']],
        'S2_PREFILL_ACTUAL_PAGE_SLOT_MAP_DIFFERS')
    outputs=request['response']['outputs']
    need(len(outputs)==1 and len(outputs[0]['tokens'])==2
        and ds['request_ids']==ds['logical_ids']==[rid] and ds['row_request_indices']==[0]
        and ds['request_count']==ds['token_row_count']==ds['scheduler_capacity']==1
        and ds['counts']==[1] and ds['row_token_positions']==[1024] and ds['pure_decode'] is True
        and ds['prefixes']==[dict(prompt=prompt,output=outputs[0]['tokens'][:1])]
        and ds['input_ids']==outputs[0]['tokens'][:1], 'S2_PREFILL_TO_FIRST_DECODE_REQUEST_ROWS_DIFFERS')
    for forward,metas,values in (
            (prefill,ps['input_tensor_metadata'],dict(input_ids=ps['token_ids'],positions=ps['positions'])),
            (decode,dict(input_ids=ds['actual_input_metadata'],positions=ds['actual_position_metadata']),
                dict(input_ids=ds['input_ids'],positions=ds['row_token_positions']))):
        need(forward['request_id']==rid and forward['phase']==phase and forward['root_returned'] is True,
             'S2_ROOT_RETURN_REQUEST_OR_PHASE_DIFFERS')
        for name,dtype in (('input_ids','torch.int32'),('positions','torch.int64')):
            actual_integer_input(forward,metas[name],name,values[name],dtype,device)
        calls=forward['calls']
        need(len(calls)==28 and [call['module_path'] for call in calls]==list(LAYERS)
            and len({call['module_object_id'] for call in calls})==28,'S2_COMPLETE_56_ACTUAL_CALLS_REQUIRED')
        for index,call in enumerate(calls):
            need(call['request_id']==rid and call['scheduler']==forward['scheduler'] and call['ordinal']==index
                and call['occurrence']==0 and call['module_path'] in call['aliases']
                and call['parent_path']==call['module_path'].rsplit('.',1)[0]
                and type(call['module_object_id']) is int and call['module_object_id']>0
                and call['original_calls']==1 and call['original_returned'] is call['inputs_unchanged'] is True
                and call['actual_operator']=='torch_npu.npu_fused_infer_attention_score',
                'S2_ACTUAL_CALL_REQUEST_PARENT_OR_ONCE_JOIN_DIFFERS')
    need([call['module_object_id'] for call in prefill['calls']]==[call['module_object_id'] for call in decode['calls']]
        and all(call['native'] is None for call in prefill['calls']),'S2_PREFILL_DECODE_MODULE_OR_NATIVE_SCOPE_DIFFERS')
    return dict(internal_request_id=rid,prefill_tokens=1024,first_decode_position=1024,
        prefill_page_count=64,observed_forwards=2,actual_calls=56,
        actual_root_tensor_hashes_recomputed=True,prefill_legacy_counter_used_as_authority=False)


def decode_cache_rows(prefill,scheduler,calls,*,device):
    """Derive the actual consumer slot and join the returned writer storage."""
    need(len(calls)==28,'S2_COMPLETE_28_ACTUAL_WRITER_CONSUMERS_REQUIRED')
    for call in calls:
        need(call['metadata_binding']=='VERIFIED' and call['layout']=='TND' and call['sequence_lengths']==[1025]
            and call['query']['shape']==[1,16,128] and call['scheduler']==scheduler and call['writer']['returned'] is True
            and call['writer']['module_path']==call['module_path'] and call['slots']==call['writer']['slots']==call['expected_slots']
            and len(call['used_pages'])==1 and len(call['used_pages'][0])==len(set(call['used_pages'][0]))==65
            and all(type(page) is int and 0<=page<128 for page in call['used_pages'][0])
            and call['used_pages'][0][:64]==prefill['block_ids']
            and call['slots']==[call['used_pages'][0][scheduler['row_token_positions'][0]//16]*16
                                +scheduler['row_token_positions'][0]%16],
            'S2_ACTUAL_WRITER_CONSUMER_OR_LOGICAL_ROWS_DIFFERS')
        slot=call['writer']['slot_mapping']
        need(slot['shape']==[1] and slot['stride']==[1] and slot['storage_offset']==0
            and slot['dtype']=='torch.int32' and slot['device']==device
            and type(slot['storage_identity']) is int and slot['storage_identity']>0,
            'S2_ACTUAL_WRITER_SLOT_TENSOR_DIFFERS')
        for key in ('key','value'):
            actual=call[key];cache=call['live_'+key+'_cache'];writer=call['writer'][key+'_cache']
            need(actual['shape']==[128,16,1024] and cache['shape']==[128,16,8,128]
                and actual['stride']==[16384,1024,1] and cache['stride']==[16384,1024,128,1]
                and actual['dtype']=='torch.float16' and actual['device']==device
                and all(actual[k]==cache[k] for k in ('storage_identity','storage_offset','dtype','device'))
                and all(writer[k]==cache[k] for k in ('storage_identity','storage_offset','shape','stride','dtype','device')),
                'S2_ACTUAL_FLAT_API_TO_LIVE_WRITER_CACHE_STORAGE_DIFFERS')
    return dict(actual_consumers=len(calls),logical_page_count=65,logical_token_position=1024,
                physical_slot_derived_from_actual_pages=True,returned_writer_storage_matched=True)


def phase_values(run,request,context,helper,normal_signature,normal_capture,*,prompt_token_ids):
    phase=request['phase'];row=request['worker'];rid=request['mapping']['internal_request_id']
    device='npu:'+str(context['worker']['logical_device'])
    lineage=request_lineage(request,prompt_token_ids,device)
    forwards=row['forwards'];prefill=forwards[0]['scheduler'];scheduler=forwards[1]['scheduler']
    decode=forwards[1]['calls'];hits=[c for f in forwards for c in f['calls'] if c['native'] is not None]
    need(len(hits)==1 and hits[0] is decode[0] and all(c['input_contract_matches'] is True for c in decode),
         'S2_ONE_SCOPED_TARGET_WITH_27_OTHER_KERNEL_MATCHES_REQUIRED')
    lineage['writer_consumer']=decode_cache_rows(prefill,scheduler,decode,device=device)
    live=row['live_input'];sealed=live['evidence'];identity=live['identity'];call=hits[0];native=call['native']
    need(live['actual_original_calls']==1 and live['complete'] is True and live['closed'] is live['inflight'] is live['simulation'] is False
        and live['first_error'] is None and sealed['before']==sealed['after'] and sealed['request_id']==rid
        and identity['generation']==context['generation'] and identity['pid']==context['worker']['pid']
        and identity['module_object_id']==call['module_object_id'] and identity['impl_object_id']==call['impl_object_id']
        and native['logical_input_binding']==live,
        'S2_LIVE_INPUT_OWNER_RETURN_OR_UNCHANGED_PROOF_DIFFERS')
    need(all(call[k]==sealed['before']['boundary'][k] for k in ('query','key','value','live_key_cache',
        'live_value_cache','page_table','used_pages','slots','writer','scheduler')),'S2_FIRST_CALL_TO_LIVE_BINDING_DIFFERS')
    need(normalized(sealed['before']['signature'])==normalized(normal_signature),'S2_FRESH_NORMAL_INPUT_DIFFERS_BEYOND_PROVEN_PAGE_RELOCATION')
    files={};prefix='/evidence/probe/native/public_payloads/'+phase+'/'
    for name,shape in (('hidden',[1,1024]),('logits_hidden',[1,1024]),('logits',[1,151936])):
        dtype=normal_capture[name]['dtype'];need(dtype in ('torch.float16','torch.float32'),'S2_NORMAL_OUTPUT_DTYPE_UNADAPTED')
        files[name]=raw_tensor(run,row['payloads'][name],shape,dtype,helper,prefix+name+'.bin')
    for index,path in enumerate(LAYERS):
        for kind in ('output','key','value'):
            name=path+'/'+kind;shape=[1,16,128] if kind=='output' else [1025,8,128]
            value=row['payloads'][name];files[name]=raw_tensor(run,value,shape,'torch.float16',helper,prefix+f'layer-{index}-{kind}.bin')
            if kind!='output':need(value['readback']=='VALID_LOGICAL_PAGE_ORDER_AFTER_CONSUMER'
                and value['operator_input_repacked'] is False and value['physical_pages']==decode[index]['used_pages'][0],
                'S2_COMPLETE_LOGICAL_CACHE_READBACK_REQUIRED')
    need(set(row['payloads'])==set(files) and len(files)==87 and files['hidden']['sha256']==files['logits_hidden']['sha256'],
        'S2_ALL_87_PAYLOADS_OR_ACTUAL_LOGITS_INPUT_DIFFERS')
    for key in ('hidden','logits'):need(files[key]['sha256']==forwards[1][key]['sha256'],'S2_ACTUAL_FINAL_OUTPUT_HASH_DIFFERS')
    query=raw_tensor(run,row['first_query'],[1,16,128],'torch.float16',helper,prefix+'query.bin')
    need(query['sha256']==sealed['before']['tensors']['query']['sha256'],'S2_QUERY_READ_BEFORE_AFTER_DIFFERS')
    immediate=[]
    for index,(shape,dtype) in enumerate((([1,16,128],'torch.float16'),([0],'torch.float32'))):
        immediate.append(raw_tensor(run,row['immediate_outputs'][index],shape,dtype,helper,prefix+f'immediate-{index}.bin'))
        need(immediate[-1]['sha256']==sealed['outputs'][index]['sha256'],'S2_IMMEDIATE_OUTPUT_CHANGED_AFTER_CONSUMER')
    need(immediate[0]['sha256']==files[FIRST+'/output']['sha256'],'S2_FIRST_API_RETURN_DIFFERS_FROM_OBSERVED_LAYER_OUTPUT')
    for kind in ('key','value'):need(files[FIRST+'/'+kind]['sha256']==sealed['before']['logical_kv'][kind]['sha256'],
        'S2_COMPLETE_FIRST_KV_READ_BEFORE_AFTER_DIFFERS')
    return dict(row=row,call=call,files=files,query=query,immediate=immediate,signature=sealed['before']['signature'],
                identity=identity,request_lineage=lineage)


def native_values(run,value,context,runtime,helper,bundle,symbols):
    call=value['call'];receipt=call['native'];host=receipt['host'];phase=value['row']['phase'];token=receipt['call_id']
    tile=256 if phase=='fault' else 512;identity=value['identity']
    need(E.read(E.child(run,'native/'+token+'.json'))==receipt and receipt['original_calls']==1
        and receipt['module_path']==FIRST and receipt['module_object_id']==call['module_object_id']
        and receipt['phase']==phase and receipt['owner']==host['owner']==context['provider_owner']
        and receipt['generation']==host['generation']==context['generation'] and receipt['token']==host['token']==token
        and receipt['selection_applied'] is (tile==256) and all(receipt[k] is False for k in
            ('api_arguments_changed','cache_repacked','output_mutated')),'S2_ORIGINAL_NATIVE_PUBLIC_OWNER_JOIN_DIFFERS')
    need(host['target_path']==FIRST and host['actual_observer_scope_matched'] is True
        and host['phase']==('fault' if tile==256 else 'baseline' if phase in PHASES[:2] else 'off')
        and (host['pid'],host['tid'])==(identity['pid'],identity['native_tid']), 'S2_ACTUAL_TLS_THREAD_PHASE_DIFFERS')
    resolved=N.one_launch(run,runtime,token,identity['pid'],helper,native_tid=identity['native_tid'],
        kernel_sha=bundle['completion']['kernel_binary_sha256'],symbols=symbols,block_dim=host['block_dim'])
    launch=resolved['launches'][0]
    decoded=N.final_private_pod(runtime[launch['raw_line']-1],launch,host,bundle,helper.abi,selected_s2=tile)
    helper.arch22_s2_build_evidence.verify_scope_geometry(decoded['typed'],host,bundle['s2_cpu_matrix']['first_decode_rows'][tile][host['aic']])
    original=receipt['native']
    need(all(original[k]==v for k,v in decoded.items()) and original['call_id']==token
        and original['pid']==launch['pid'] and original['tid']==launch['tid']
        and original['kernel_sha256']==launch['registered_binary']['sha256'], 'S2_ORIGINAL_NATIVE_DIFFERS_FROM_RAW_REPLAY')
    return dict(phase=phase,token=token,selected_s2=tile,block_dim=host['block_dim'],
        full_typed_sha256=E.digest(decoded['typed']),raw_tiling_sha256=decoded['raw_sha256'],launch=launch)


def review(args):
    checks=N.Checks();report=dict(schema='demoncat.paged33.public_s2_single_review.v1',C2=False,C3=False,E5='NOT_ASSESSED',
        public_qualification_granted=False,numerical_fault_observed=False,model_end_propagation_observed=False,
        reviewer_source_sha256=E.sha(__file__),original_results_modified=False)
    def perform():
        source=E.frozen_source(args.source,args.source_manifest_sha256,build_variant=E.S2_VARIANT)
        helper=E.helpers(source);bundle=E.build_bundle(args.build_root,source,helper,args.completion_sha256)
        observer=E.observer_bundle(args.observer,source);need(observer['sha256']==args.observer_receipt_sha256,'S2_OBSERVER_CHANGED')
        normal=N.review(SimpleNamespace(**{k:getattr(args,k) for k in ('source','source_manifest_sha256','build_root',
            'completion_sha256','observer','observer_receipt_sha256','stock_run','stock_sha256','baseline_run','baseline_sha256')},arch22_s2_normal=True))
        need(not normal['issues'] and normal['normal_payloads_exact'] and normal['complete_payload_pairs']==435,'S2_FRESH_NORMAL_RAW_REVIEW_FAILED')
        need(E.sha(args.normal_report)==args.normal_report_sha256 and E.sha(args.run/'result.json')==args.result_sha256,'S2_PINNED_ORIGINAL_CHANGED')
        data=E.read(args.run/'result.json');report['original_status']=data.get('status');report['original_failures']=N.original_failures(data)
        need(not report['original_failures'],'S2_ORIGINAL_FAILURES_PRESERVED')
        report['protocol']=protocol(args.run,data);context=report['protocol']
        report['library_coexistence']=N.check_coexistence(data,source,bundle,context['worker']['pid'],
            cleanup_receipts=[data['library_coexistence_after_owned_cleanup']])
        original_normal=E.read(args.baseline_run/'result.json');normal_capture=original_normal['phases'][0]['finish']['capture']
        normal_live=normal_capture['first_decode_boundary']['logical_input_binding']
        need(data['prompt_token_ids']==original_normal['phases'][0]['response']['prompt_token_ids']
            and data['final_cleanup']['evidence']['source']==original_normal['phases'][0]['setup']['source'],
            'S2_ACTUAL_PROMPT_OR_MODEL_IMPLEMENTATION_SOURCE_DIFFERS')
        signature=normal_live['evidence']['before']['signature'];proof=data['setup']['normal_evidence']
        need(E.digest(proof)==data['setup']['normal_evidence_sha256'] and proof['reference_input']==signature
            and proof['source_manifest_sha256']==source['manifest_sha256'] and proof['completion_sha256']==args.completion_sha256
            and proof['observer_receipt_sha256']==args.observer_receipt_sha256
            and proof['published_report_sha256']==args.normal_report_sha256
            and proof['stock_result_sha256']==args.stock_sha256 and proof['private_result_sha256']==args.baseline_sha256,
            'S2_TRUSTED_NORMAL_INPUTS_NOT_ORIGINAL')
        contract=data['execution_contract'];normal_contract=original_normal['execution_contract']
        need({k:v for k,v in contract.items() if k!='driver_sha256'}=={k:v for k,v in normal_contract.items() if k!='driver_sha256'}
            and contract['driver_sha256']==source['pins'][E.EXPERIMENT+'natural33_arch22_s2_worker_core_probe.py'],
            'S2_SAME_SOURCE_MODEL_PACKAGE_RUNNER_REQUIRED')
        runtime,runtime_file=helper.boundary.read_runtime(args.run);report['runtime_file']=runtime_file
        host_raw=E.child(args.run,'native/host.jsonl').read_bytes();need(host_raw.endswith(b'\n'),'S2_INCOMPLETE_HOST_LOG')
        hosts=[json.loads(line,object_pairs_hook=E.unique) for line in host_raw.splitlines()]
        need(len(hosts)==5 and len({x['token'] for x in hosts})==5,'S2_EXACT_FIVE_HOST_SCOPES_REQUIRED')
        values=[];native=[];symbols={}
        for request in data['requests']:
            value=checks.run(request['phase']+'.public',lambda r=request:phase_values(args.run,r,context,helper,signature,normal_capture,
                prompt_token_ids=data['prompt_token_ids']))
            if value is not None:
                values.append(value);need(value['call']['native']['host'] in hosts,'S2_HOST_ORIGINAL_DIFFERS')
                item=checks.run(request['phase']+'.native',lambda v=value:native_values(args.run,v,context,runtime,helper,bundle,symbols))
                if item:native.append(item)
        need(len(values)==len(native)==5,'S2_ALL_FIVE_ORIGINAL_NATIVE_AND_PUBLIC_CHAINS_REQUIRED')
        need(len({x['token'] for x in native})==5 and {x['token'] for x in native}=={x['token'] for x in hosts},'S2_NATIVE_HOST_TOKEN_SET_DIFFERS')
        need([r['call_id'] for r in runtime if r['event']=='begin']==[x['token'] for x in native]
            and [r['call_id'] for r in runtime if r['event']=='end']==[x['token'] for x in native]
            and not any(r['event']=='capture_overflow' for r in runtime),
            'S2_EXTRA_REORDERED_OR_INCOMPLETE_ACTUAL_NATIVE_SCOPE')
        base=values[0];comparisons=[]
        for value in values:
            need(value['identity']==base['identity'],'S2_LIVE_WORKER_MODEL_MODULE_OWNER_CHANGED')
            for index in (0,1):need(value['row']['forwards'][index]['input']==base['row']['forwards'][index]['input'],
                                   'S2_ACTUAL_MODEL_INPUT_CHANGED')
            need(all(value['row']['forwards'][0][k]==base['row']['forwards'][0][k] for k in ('hidden','logits')),
                 'S2_NON_TARGET_PREFILL_OUTPUT_CHANGED')
            pair=helper.differences.compare(base['files'],value['files'],list(LAYERS));comparisons.append(dict(phase=value['row']['phase'],comparison=pair))
            if value['row']['phase']!='fault':helper.bridge.compare_payload(base['files'],value['files'])
        # Re-read values, not just hashes; keep operator and model propagation
        # separate. Unchanged logits do not erase a real operator effect.
        immediate=helper.differences.raw_difference(base['immediate'][0],values[3]['immediate'][0])
        logits=helper.differences.raw_difference(base['files']['logits'],values[3]['files']['logits'])
        need(not immediate['exact'],'S2_NATIVE_SELECTION_WITHOUT_NUMERICAL_OPERATOR_EFFECT')
        gate=values[3]['call']['native']['normal_gate']
        need(gate==values[4]['call']['native']['normal_gate'] and gate['owner']==context['provider_owner']
            and gate['generation']==context['generation'] and gate['serializable_authority'] is False
            and gate['normal_evidence_sha256']==data['setup']['normal_evidence_sha256']
            and gate['actual_native_tokens']==[x['token'] for x in native[:3]]
            and gate['actual_request_ids']==[r['mapping']['internal_request_id'] for r in data['requests'][:3]]
            and gate['model_rows_sha256']==E.digest([x['row'] for x in values[:3]]),'S2_LIVE_CONTROL_ISSUANCE_AND_USE_JOIN_DIFFERS')
        report.update(source_manifest_sha256=source['manifest_sha256'],result_sha256=args.result_sha256,
            completion_sha256=args.completion_sha256,observer_receipt_sha256=observer['sha256'],normal_replay_sha256=E.digest(normal),
            denominator_per_request=56,decode_kernel_input_matching_targets=28,source_bound_normal_targets=1,
            other_matching_targets_without_per_site_proof=27,native=native,registered_symbols=list(symbols.values()),
            comparisons=comparisons,operator_effect=immediate,model_logits_effect=logits,
            request_lineage=[dict(phase=value['row']['phase'],**value['request_lineage']) for value in values],
            operator_output_bytes_changed=True,numerical_fault_observed=immediate['numeric_changed_elements']>0,
            model_end_propagation_observed=logits['numeric_changed_elements']>0)
    checks.run('original_files_and_protocol',perform)
    report.update(checks=checks.checks,issues=checks.issues,
        status='SOURCE_BOUND_SINGLE_SITE_OUTPUT_EFFECT_AND_OFF_RECONCILED_NOT_C2_C3_E5' if not checks.issues else 'REFUSED_WITH_ORIGINAL_EVIDENCE',
        limits=['C1 scope is the actual first-layer experiment only; 27 other input-matching decode sites remain unproved.',
            'Before-read mask/logical KV values are source-bound hashes; later complete matching KV bytes were independently read.',
            'Selected ACLNN caller owner is unobserved; actual scoped launch/ELF/POD is checked.',
            'Post-unregister health proves only tokens/text, not hidden/logits.'])
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','build-root','observer','stock-run','baseline-run','normal-report','run','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('source-manifest-sha256','completion-sha256','observer-receipt-sha256','stock-sha256','baseline-sha256','normal-report-sha256','result-sha256'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args();need(not args.out.exists(),'S2_FRESH_REVIEW_OUTPUT_REQUIRED');sys.dont_write_bytecode=True
    numeric=N.initialize_cpu_byte_reader();sys.addaudithook(E.prohibit_runtime)
    result=review(args);result['cpu_numeric_reader']=numeric
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],checks=result['checks'],issues=len(result['issues']),sha256=E.sha(args.out),C2=False,C3=False,E5='NOT_ASSESSED')))
    return 0 if not result['issues'] else 2


if __name__=='__main__':raise SystemExit(main())
