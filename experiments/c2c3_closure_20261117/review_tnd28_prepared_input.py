"""Independent, bounded normal-only typed Contiguous -> FIA input join.

No torch, injector, CDLL, subprocess or device. Slice POD is deliberately not
decoded. The observed typed call and live executor offsets identify its output.
This post-execution report cannot authorize an earlier host tiling decision.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
import arch22_normal_bridge_evidence as E
import attention28_census_raw as RAW

VARIANT='tnd_prefill_denominator_replay'
PHASES=('prepare_enter','contiguous_prepare_enter','contiguous_prepare_return',
    'workspace_enter','workspace_return','prepare_return','prepare_exit','execute_enter',
    'contiguous_execute_enter','contiguous_execute_return','fia_execute_enter','fia_execute_return',
    'execute_return','execute_exit')
SERIALIZER_SHA='b54c905c4cda0d049892b3ade7edd0aa51dce3452caf2dcf233722e892288fa5'
NNOPBASE_SHA='6dc3b4fcfcf57b10be0018e89f4edbc079f02b4bb0aaadb109df350939fd4c99'
GENERATED_SHA='d35f8b60d1ad50d97b7cf0d2050be5c23fd0d53eba77a58c053c184d96af4fe2'
LSE_SOURCE_COMMIT='c77c0829f1f0f0b8f882db1d9b051a758b605472'
LSE_SOURCES={
    'attention/fused_infer_attention_score/op_api/aclnn_fused_infer_attention_score_v3.cpp':
        '551cc8b6b7439bfc231dba0d17305d07df56226f2e771346fe78a7d2054f4ed2',
    'attention/fused_infer_attention_score/op_api/aclnn_fused_infer_attention_score_inner.cpp':
        '1efdfba88a2463376d11df2dbd7b9aed515f6f42a875314406c15b39e4119808'}
RPC_ACTIONS=['attach','prepare','register','check','enable','advance','query','advance','disable','unregister','detach']
RPC_ITEM_LIMIT=32*1024**2
RPC_TOTAL_LIMIT=128*1024**2
SCALARS=('num_heads','pre_tokens','next_tokens','num_key_value_heads','sparse_mode','inner_precise',
    'block_size','antiquant_mode','key_antiquant_mode','value_antiquant_mode','query_quant_mode','pse_type','out_dtype')
V3_FIXED={'antiquant_mode':0,'key_antiquant_mode':0,'value_antiquant_mode':0,'query_quant_mode':0,'pse_type':0,'out_dtype':0}
STORAGE_FORMATS={'query':47,'key':47,'value':47,'output':47,'atten_mask':2}
FORMAT_ENUM_SOURCES={
    'include/graph/types.h':'0582beb87dfc4aaf9019005150fd8a34ece1120fabbd2159abee2aecce51416a',
    'include/graph/c_types.h':'365ba1e3c0cee98494c322969a7ac80e7304139acc00e69c3dbf2cafd5d6d21d'}
FUNCTION_PATH_ALIASES={
    '/usr/local/Ascend/cann-8.5.1/lib64/libopapi_math.so':
        '/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64/libopapi_math.so',
    '/usr/local/Ascend/cann-8.5.1/lib64/libnnopbase.so':
        '/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64/libnnopbase.so'}


def need(ok,why):
    if not ok:raise ValueError(why)


def raw_file(path,limit):
    path=Path(path);a=path.stat()
    need(not path.is_symlink() and 0<a.st_size<=limit,'ORIGINAL_FILE_BOUND_OR_LINK:'+str(path))
    raw=path.read_bytes();b=path.stat()
    need((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns)
        and len(raw)==a.st_size,'ORIGINAL_FILE_CHANGED:'+str(path))
    return raw


def json_file(path,limit=4*1024**2):
    return json.loads(raw_file(path,limit),object_pairs_hook=E.unique,
        parse_constant=lambda x:(_ for _ in ()).throw(ValueError('NONFINITE_JSON')))


def lines(path,limit):
    raw=raw_file(path,limit);need(raw.endswith(b'\n'),'INCOMPLETE_JSONL:'+str(path))
    need(all(len(line)<=2*1024**2 for line in raw.splitlines()),'JSONL_LINE_BOUND')
    return [json.loads(line,object_pairs_hook=E.unique) for line in raw.splitlines()]


def tensor_join(public,typed,where):
    need(type(public) is dict and type(typed) is dict,'MISSING_TYPED_TENSOR:'+where)
    dtype={'torch.float16':1,'torch.int8':2,'torch.float32':0}.get(public['dtype'])
    # The compiled header calls GetStorageFormat(), not GetViewFormat(). In
    # this pinned SDK, 47 is NCL; the public FIA input_layout remains TND.
    need(where in STORAGE_FORMATS and type(typed['format']) is int
        and typed['format']==STORAGE_FORMATS[where],'TENSOR_STORAGE_FORMAT_FOR_ROLE:'+where)
    need(dtype is not None and public['device'].startswith('npu:') and typed['dtype']==dtype
        and typed['descriptor']>0 and typed['native_storage_offset']==0,
        'TENSOR_DTYPE_DEVICE_OR_DESCRIPTOR:'+where)
    for left,right in (('storage_address','storage_address'),('data_address','data_address'),
            ('storage_offset','offset'),('shape','shape'),('stride','stride')):
        need(public[left]==typed[right],'ACTUAL_TYPED_TENSOR_DIFFERS:'+where+':'+left)
    need(public['object_id']>0 and public['storage_identity']>0 and public['element_bytes'] in (1,2,4)
        and public['data_address']==public['storage_address']+public['storage_offset']*public['element_bytes'],
        'PUBLIC_STORAGE_VIEW_IDENTITY:'+where)
    need(len(public['shape'])==(2 if where=='atten_mask' else 3)
        and type(public['storage_bytes']) is int and public['storage_bytes']>=0
        and public['storage_bytes']%public['element_bytes']==0
        and typed['storage_shape']==[public['storage_bytes']//public['element_bytes']],
        'ACTUAL_NATIVE_FLAT_STORAGE_EXTENT:'+where)
    if all(public['shape']):
        last=public['storage_offset']+sum((n-1)*s for n,s in zip(public['shape'],public['stride']))+1
        need(len(public['shape'])==len(public['stride']) and all(s>0 for s in public['stride'])
            and last*public['element_bytes']<=public['storage_bytes'],'PUBLIC_STORAGE_EXTENT:'+where)


def function_identity(function):
    need(function['mapped_file_identity_matches'] is True and function['mapped_inode']==function['file_inode']>0
        and function['mapped_device']==function['file_device']
        and function['mapped_begin']<=function['address']<function['mapped_end']
        and 0<function['base']<=function['address'] and function['file_bytes']>0,
        'ACTUAL_FUNCTION_MAPPED_FILE_IDENTITY')
    need(function['path'].startswith('/') and function['mapped_path'].startswith('/'),
        'ACTUAL_FUNCTION_OWNER_PATH_MISSING')


def function_library_entry(function,entries):
    function_identity(function)
    observed=function['path'];mapped=function['mapped_path']
    if observed==mapped:
        via='EXACT_OBSERVED_PATH'
    else:
        need(FUNCTION_PATH_ALIASES.get(observed)==mapped,'UNREVIEWED_ACTUAL_FUNCTION_PATH_ALIAS')
        via='OBSERVED_CANN851_ALIAS_WITH_MATCHED_FILE_IDENTITY'
    need(mapped in entries and entries[mapped]['recorded_path']==mapped,
        'ACTUAL_FUNCTION_CANONICAL_MANIFEST_ENTRY_MISSING')
    return entries[mapped],dict(observed_path=observed,mapped_path=mapped,
        manifest_recorded_path=mapped,matched_via=via)


def false_lse(typed,public,output):
    # The pinned V3 wrapper supplies a newly-created, temporary host placeholder
    # to inner GWS and destroys its descriptor before returning to Python.
    # Its recorded address is metadata only: do not read the expired storage.
    need(type(typed) is dict and typed['descriptor']>0 and typed['storage_address']>0
        and typed['data_address']==typed['storage_address']
        and all(typed[k]==v for k,v in dict(shape=[0],storage_shape=[0],stride=[0],offset=0,
            native_storage_offset=0,dtype=0,format=2).items()),'FALSE_LSE_TYPED_PLACEHOLDER_DIFFERS')
    need(type(public) is dict and public['object_id']>0 and public['storage_identity']>0
        and public['device']==output['device'] and public['device'].startswith('npu:')
        and all(public[k]==v for k,v in dict(shape=[0],stride=[1],storage_offset=0,
            storage_bytes=0,element_bytes=4,dtype='torch.float32',data_address=0,storage_address=0).items()),
        'FALSE_LSE_PUBLIC_EMPTY_RETURN_DIFFERS')


def false_lse_sources(completion,build):
    need(build['source_commit']==LSE_SOURCE_COMMIT,'FALSE_LSE_SOURCE_REVISION_CHANGED')
    source=Path(completion).parent.parent/'source'
    for name,expected in LSE_SOURCES.items():
        need(hashlib.sha256(raw_file(E.child(source,name),64*1024)).hexdigest()==expected,
            'FALSE_LSE_ACTUAL_BUILD_SOURCE_CHANGED:'+name)
    return dict(LSE_SOURCES)


def protocol(public,returned,events):
    need(public['schema']=='demoncat.tnd28.public_input.v1' and returned['schema']=='demoncat.tnd28.public_return.v1',
        'PUBLIC_INPUT_RETURN_SCHEMA')
    need([r.get('phase') for r in events]==list(PHASES),'ACTUAL_TYPED_CALL_PHASES_INCOMPLETE_OR_REORDERED')
    identity=public['identity'];prior=0
    for row in events:
        need(row['schema']=='demoncat.tnd28.compiled_input.v1' and not row['first_error']
            and row['fault_authorized'] is False,'NATIVE_INPUT_OBSERVER_FAILED:'+row['phase'])
        for key in ('token','owner','generation','module_object_id','request_sha256','input_sha256'):
            need(row[key]==public[key],'CURRENT_OWNER_REQUEST_OR_INPUT_CHANGED:'+key)
        need(row['pid']==identity['pid'] and row['tid']==identity['native_tid']
            and row['process_start_ticks']==identity['process_start_ticks']
            and row['monotonic_ns']>prior,'NATIVE_THREAD_LIFETIME_OR_EVENT_ORDER')
        prior=row['monotonic_ns']
        if 'function' in row['data']:function_identity(row['data']['function'])
    need(public['input_sha256']==E.digest(public['tensors']) and public['request_sha256']==E.digest(public['request_id'])
        and public['scheduler']['internal_request_id']==public['request_id']
        and public['scheduler']['num_requests']==1 and public['scheduler']['num_scheduled_tokens']==2048,
        'ACTUAL_SCHEDULER_OR_PUBLIC_INPUT_HASH')
    for key in ('token','identity','owner','generation','request_sha256','input_sha256'):
        need(returned[key]==public[key],'ACTUAL_PUBLIC_RETURN_SCOPE:'+key)
    need(returned['original_calls']==1 and returned['return_container_type']=='tuple' and len(returned['outputs'])==2,
        'ACTUAL_ORIGINAL_RETURN_REQUIRED')
    data={r['phase']:r['data'] for r in events};p=data['prepare_enter'];api=public['api']
    for name,native in (('query',p['query']),('key',p['key']['members'][0]),('value',p['value']['members'][0]),('atten_mask',p['mask'])):
        tensor_join(public['tensors'][name],native,name)
        metadata=api['inputs'][name]['snapshot']['metadata'];actual=public['tensors'][name]
        need(all(metadata[k]==actual[k] for k in ('object_id','storage_identity','shape','stride','dtype','device'))
            and metadata['storage_offset']==actual['storage_offset'],'ACTUAL_PYTHON_CAPTURE_OBJECT:'+name)
    need(len(p['key']['members'])==len(p['value']['members'])==1 and p['key']['object']>0 and p['value']['object']>0,
        'ACTUAL_SINGLE_KV_LIST_REQUIRED')
    shapes={'query':[2048,16,128],'key':[2048,8,128],'value':[2048,8,128],'atten_mask':[2048,2048]}
    need(all(public['tensors'][k]['shape']==v for k,v in shapes.items()),'SOURCE87_ACTUAL_PREFILL_SHAPES')
    need(public['tensors']['value']['stride']==[4096,128,1] and public['tensors']['value']['storage_offset']==3072,
        'ACTUAL_NONCONTIGUOUS_V_VIEW_UNADAPTED')
    def attr(k):
        if k in api['inputs']:
            need(api['inputs'][k]['kind']=='scalar','PUBLIC_SCALAR_REQUIRED:'+k);return api['inputs'][k]['value']
        if k in api['omitted_argument_defaults']:return api['omitted_argument_defaults'][k]
        need(k in V3_FIXED,'UNOBSERVED_PUBLIC_ATTRIBUTE:'+k);return V3_FIXED[k]
    need(p['scalars']==[attr(k) for k in SCALARS] and len(p['optionals'])==25 and p['optionals']==[0]*25,
        'ACTUAL_TYPED_SCALARS_OR_OPTIONAL_ARGUMENTS_DIFFER')
    need(p['layout']==attr('input_layout')=='TND' and p['scale']==attr('scale')==128**-.5
        and p['softmax_lse']==attr('softmax_lse_flag') is False
        and p['seq_q']['values']==attr('actual_seq_lengths')==[2048]
        and p['seq_kv']['values']==attr('actual_seq_lengths_kv')==[2048], 'ACTUAL_TYPED_LAYOUT_SCALE_OR_LENGTHS')
    c=data['contiguous_prepare_enter'];cr=data['contiguous_prepare_return'];wr=data['workspace_return']
    need(c['inputs']['members']==p['value']['members'],'ACTUAL_CONTIGUOUS_INPUT_IS_NOT_ORIGINAL_V_DESCRIPTOR')
    for phase in ('contiguous_prepare_return','workspace_return','contiguous_execute_return','fia_execute_return'):
        need(data[phase]['status']==0 and data[phase]['original_calls']==1,'ORIGINAL_CALL_DID_NOT_RETURN_ONCE:'+phase)
    for phase in ('prepare_return','execute_return'):
        need(data[phase]['status']==0 and data[phase]['original_returns']==1,'ORIGINAL_API_RETURN:'+phase)
    need(data['prepare_exit']=={'workspace_called':True,'original_returned':True}
        and data['execute_exit']=={'fia_called':True,'original_returned':True},'INCOMPLETE_ORIGINAL_API_EXIT')
    need(data['workspace_enter']['executor']==wr['executor']==data['prepare_return']['executor']==data['execute_enter']['executor']>0,
        'ACTUAL_PARENT_EXECUTOR_CHANGED')
    ce=data['contiguous_execute_enter'];fe=data['fia_execute_enter']
    need(ce['executor']==cr['executor']>0 and ce['workspace_offsets']==cr['workspace_offsets']
        and len(ce['workspace_offsets'])==1 and ce['workspace_bytes']==cr['workspace_bytes']
        and data['prepare_return']['total_workspace_bytes']==wr['fia_workspace_bytes']+cr['workspace_bytes'],
        'ACTUAL_CHILD_EXECUTOR_OFFSETS_OR_WORKSPACE_CHANGED')
    offset=ce['workspace_offsets'][0]
    need(0<=offset and offset+2048*8*128*2<=ce['workspace_bytes'] and ce['workspace']>0
        and ce['prepared_v_address']==ce['workspace']+offset
        and fe['workspace']+fe['workspace_bytes']==ce['workspace']
        and fe['workspace_bytes']==wr['fia_workspace_bytes'] and fe['executor']==wr['executor']
        and fe['stream']==ce['stream']>0,'ACTUAL_PREPARED_V_PARENT_CHILD_SPAN_OR_STREAM')
    tensor_join(returned['outputs'][0],p['output'],'output')
    false_lse(p['lse'],returned['outputs'][1],returned['outputs'][0])
    return data


def launch_inputs(raw,abi,typed):
    need(raw['argument_layout']=='aclrt_host_placeholders' and abi['pointer_bytes']==8,'FRESH_FIA_ARGUMENT_ABI_REQUIRED')
    args=bytes.fromhex(raw['args_hex']);placeholders=bytes.fromhex(raw['placeholders_hex'])
    need(len(args)==raw['args_size'] and len(placeholders)%8==0,'FRESH_FIA_RAW_ARGUMENT_BOUNDS')
    pairs=list(struct.iter_unpack('<II',placeholders));need(len({a for a,b in pairs})==len(pairs),'DUPLICATE_PLACEHOLDER')
    locations=dict(pairs);prefix=abi['hidden_hardware_sync_prefix_bytes'];names=abi['names']
    slots={name:prefix+i*8 for i,name in enumerate(names)};header=prefix+len(names)*8
    need(len(args)>=header and all(a in slots.values() and header<=b<len(args) for a,b in pairs),'PLACEHOLDER_OUTSIDE_SOURCE_ABI')
    p=typed['prepare_enter'];ce=typed['contiguous_execute_enter']
    def pointer(name):
        slot=slots[name];need(slot not in locations,'UNEXPECTED_HOST_PLACEHOLDER:'+name);return struct.unpack_from('<Q',args,slot)[0]
    need(pointer('query')==p['query']['data_address'] and pointer('attenMask')==p['mask']['data_address']
        and pointer('attentionOut')==p['output']['data_address'],'FRESH_FIA_DIRECT_INPUT_OR_OUTPUT_POINTER')
    if 'softmaxLse' in slots:need(pointer('softmaxLse')==p['lse']['data_address'],'ACTUAL_LSE_OUTPUT_POINTER')
    known={'query','key','value','attenMask','actualSeqLengths','actualSeqLengthsKV','attentionOut','softmaxLse','workspace','tiling'}
    need(all(pointer(name)==0 for name in names if name not in known),'UNEXPECTED_OPTIONAL_KERNEL_OPERAND')
    # This exact public opbase serializer is pinned separately. Never search
    # argument bytes for a matching address and never decode Slice arguments.
    spans=[]
    for name,address,shape in (('key',p['key']['members'][0]['data_address'],p['key']['members'][0]['storage_shape']),
            ('value',ce['prepared_v_address'],p['value']['members'][0]['shape'])):
        need(slots[name] in locations,'DYNAMIC_KV_PLACEHOLDER_MISSING:'+name);start=locations[slots[name]]
        size=8+8+len(shape)*8+8
        need(start+size<=len(args),'DYNAMIC_KV_DESCRIPTOR_TRUNCATED:'+name)
        expected=struct.pack('<QII',16+len(shape)*8,len(shape),1)+struct.pack('<'+'q'*len(shape),*shape)+struct.pack('<Q',address)
        need(args[start:start+size]==expected,'ACTUAL_KV_LIST_DESCRIPTOR_OR_ADDRESS_DIFFERS:'+name);spans.append((start,start+size))
    for name in ('actualSeqLengths','actualSeqLengthsKV'):
        need(slots[name] in locations,'ACTUAL_LENGTH_PLACEHOLDER_MISSING');start=locations[slots[name]]
        need(args[start:start+8]==struct.pack('<q',2048),'ACTUAL_LENGTH_BYTES_CHANGED');spans.append((start,start+8))
    need(all(a1<=a2 and b1<=a2 for (a1,b1),(a2,b2) in zip(sorted(spans),sorted(spans)[1:])),
        'HOST_INPUT_DESCRIPTORS_OVERLAP')
    return dict(query_address=p['query']['data_address'],key_address=p['key']['members'][0]['data_address'],
        prepared_value_address=ce['prepared_v_address'],actual_output_address=p['output']['data_address'],
        serializer_source_sha256=SERIALIZER_SHA,slice_pod_decoded=False,
        lse=dict(rule='FALSE_LSE_TEMPORARY_GWS_PLACEHOLDER',snapshot_phase='prepare_enter',
            placeholder_address=p['lse']['data_address'],kernel_argument_present='softmaxLse' in slots,
            public_empty_return_checked_separately=True,descriptor_live_after_gws_claimed=False,
            temporary_storage_bytes_read=False))


def runtime_join(run,rows,public,events,abi,kernel_sha):
    resolved=RAW.scope(run,rows,public['token'],expected_pid=public['identity']['pid'])
    typed=protocol(public,public['_return'],events);byphase={r['phase']:r for r in events}
    begin=[r for r in rows if r.get('call_id')==public['token'] and r['event']=='begin'][0]
    end=[r for r in rows if r.get('call_id')==public['token'] and r['event']=='end'][0]
    need(begin['monotonic_ns']<events[0]['monotonic_ns']<events[-1]['monotonic_ns']<end['monotonic_ns'],
        'TYPED_OBSERVATION_NOT_WITHIN_ACTUAL_RUNTIME_SCOPE')
    need(len(resolved['launches'])==2,'ONE_ACTUAL_CONTIGUOUS_AND_ONE_FIA_LAUNCH_REQUIRED')
    original=[rows[x['raw_line']-1] for x in resolved['launches']]
    for launch,raw in zip(resolved['launches'],original):
        need(raw['tid']==public['identity']['native_tid'],'LAUNCH_NATIVE_THREAD_DIFFERS')
        launch['entry']=RAW.entry_symbols(launch['registered_binary']['path'],launch['registered_binary']['sha256'],
            selected_keys=[launch['tiling_key']])
    copy,fia=original;copyres,fiares=resolved['launches']
    need(fiares['registered_binary']['sha256']==kernel_sha and copyres['registered_binary']['sha256']!=kernel_sha,
        'ACTUAL_FIA_REGISTERED_BINARY_NOT_NEW_BUILD')
    need(all('Slice_' in s['name'] for s in copyres['entry']['actual_selected_entry_symbols']),
        'ACTUAL_AUXILIARY_ENTRY_NOT_SLICE')
    for raw,start,end in ((copy,'contiguous_execute_enter','contiguous_execute_return'),(fia,'fia_execute_enter','fia_execute_return')):
        need(byphase[start]['monotonic_ns']<raw['launch_begin_ns']<=raw['launch_end_ns']<byphase[end]['monotonic_ns']
            and raw['stream']==typed[start]['stream'],'ACTUAL_TYPED_CALL_LAUNCH_WINDOW_OR_STREAM')
    inputs=launch_inputs(fia,abi,typed)
    return dict(launches=resolved['launches'],input_join=inputs,
        timing='HOST_WORKSPACE_TILING_PRECEDES_CONTIGUOUS_EXECUTION_AND_FIA_LAUNCH',
        pre_tiling_fault_permission=False)


def vendor_link_rule(name,target):
    # This is the packaged public alias, not the real ELF's basename. The
    # caller still verifies its resolved vendor-root boundary and target bytes.
    need(name=='op_impl/ai_core/tbe/op_tiling/liboptiling.so'
        and target=='lib/linux/aarch64/libcust_opmaster_rt2.0.so',
        'UNREVIEWED_VENDOR_LINK')


def bundle(completion,expected,vendor,source,source_sha):
    need(E.sha(completion)==expected,'EXACT_NEW_BUILD_COMPLETION_REQUIRED');build=json_file(completion)
    need(build['status']=='CPU_BUILD_ONLY_REQUIRES_NATIVE_QUALIFICATION' and build['variant']==VARIANT
        and build['npu_executed'] is False,'EXACT_NEW_VARIANT_REQUIRED')
    obs=build['tnd28_compiled_input_observer'];need(obs['original_sha256']==GENERATED_SHA
        and build['tnd28_input_serializer_sha256']==SERIALIZER_SHA and obs['observer_only'] is True
        and obs['fault_authorized'] is False,'COMPILED_INPUT_HOOK_SOURCE_REQUIRED')
    source=Path(source);manifest=json_file(source/'source_manifest.json',16*1024**2)
    need(E.sha(source/'source_manifest.json')==source_sha,'FROZEN_SOURCE_MANIFEST_CHANGED')
    pins={r['path']:r for r in manifest['files']};base='fault_injection/native_adapters/attention28/'
    participating=[base+'launch.py',base+'fia/tnd28_input_observation.h',base+'fia/tnd28_input_compile.py',
        'src/demoncat/runtime/vllm_ascend/worker_extension/attention28_prepared_input.py',
        'src/demoncat/runtime/vllm_ascend/worker_extension/attention28_stock_census.py']
    for name in participating:need(E.sha(E.child(source,name))==pins[name]['sha256'],'FROZEN_CAPTURE_SOURCE_CHANGED:'+name)
    need(obs['header_sha256']==pins[base+'fia/tnd28_input_observation.h']['sha256']
        and obs['launcher_sha256']==pins[base+'fia/tnd28_input_compile.py']['sha256'],'COMPILED_CAPTURE_SOURCE_MISMATCH')
    files={r['path']:r for r in build['artifacts']}
    need(len(files)==len(build['artifacts'])<=2048 and sum(r['bytes'] for r in files.values())<=512*1024**2,'VENDOR_ARTIFACT_BOUNDS')
    for name,r in files.items():
        path=E.child(vendor,name)
        if path.is_symlink():
            import os
            vendor_link_rule(name,os.readlink(path))
        need(path.stat().st_size==r['bytes']<=128*1024**2 and E.sha(path)==r['sha256'],'VENDOR_BYTES_CHANGED:'+name)
    path=source/base/'launch.py';tree=ast.parse(path.read_bytes());names={'derive_fia_kernel_abi','bundle_kernel_abi'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    need({n.name for n in nodes}==names,'PURE_SOURCE_ABI_FUNCTION_SET')
    ns=dict(require=need,sha=E.sha,Path=Path,re=re,struct=struct,json=json)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    metadata,abi=ns['bundle_kernel_abi'](vendor,files,build['kernel_binary_sha256'])
    return build,files,metadata,abi


def public_lifecycle(result,public):
    row=result['request']['worker'];calls=[c for f in row['forwards'] for c in f['calls']]
    need(row['request_id']==public['request_id'] and row['phase']=='observe' and len(calls)==28
        and all(c['original_calls']==1 and c['original_returned'] is True and c['inputs_unchanged'] is True for c in calls),
        'ACTUAL_FULL_SINGLE_REQUEST_DENOMINATOR_OR_CALLS')
    selected=[c for c in calls if 'prepared_input_capture' in c.get('native_census',{})]
    need(len(selected)==1,'ACTUAL_SELECTED_WINDOW_NOT_UNIQUE');call=selected[0];native=call['native_census']
    need(call['module_object_id']==public['module_object_id'] and call['module_path']==public['module_path']
        and native['call_id']==public['token'] and native['api']==public['api']==native['api_inputs_after'],
        'ACTUAL_PUBLIC_CALL_INPUTS_OR_OBJECT_CHANGED')
    for key in ('worker_object_id','model_object_id'):
        need(public[key]==result['setup'][key],'ACTUAL_DEPLOYMENT_OBJECT:'+key)
    state=result['cleanup'];need(state['owner']==public['owner'] and state['generation']==public['generation']
        and not any(state[k] for k in ('owned_hooks','owned_api_wrappers','inflight','leases','C1','C2','C3')),
        'ACTUAL_PUBLIC_RESOURCE_OR_QUALIFICATION_STATE')
    ns=state['native_state'];need(not any(ns[k] for k in ('scope_owned','scope_active','pending_drains','pending_references','environment_owned')),
        'ACTUAL_NATIVE_OWNERSHIP_REMAINS')
    removal=result['binding_removal'];window=removal['input_window']
    need(removal['removal']['status']=='SERVING_DEPLOYMENT_REMOVED' and removal['first_error'] is None
        and window['token']==public['token'] and window['window_complete'] is True
        and window['actual_native_observed'] is False and window['prepared_input_join'] is False,
        'ACTUAL_BINDING_REMOVAL_OR_WINDOW_CLEANUP')
    need(result['ownership_retained'] is False and not result.get('cleanup_errors')
        and not result.get('error') and not result.get('post_protocol_errors')
        and result['health']['scope']=='POST_UNREGISTER_TOKENS_ONLY'
        and result['health']['output_tokens']==result['request']['output_tokens']
        and all(result[k] is False for k in ('C1','C2','C3','E5','fault_authorized')),
        'ACTUAL_FINAL_OWNER_OR_ORDINARY_HEALTH')


def public_rpc_summaries(result,public,records):
    need([r['action'] for r in records]==RPC_ACTIONS,'ACTUAL_SINGLE_WINDOW_PUBLIC_RPC_PROTOCOL')
    # collect_replay is an advance RPC. Use the raw response that produced each
    # summary; matching action names/history alone cannot bind copied summaries.
    collected=records[7]['response']['provider_state'];stopped=records[8]['response']['provider_state']
    need(collected['evidence']['requests']==[result['request']['worker']],
        'ACTUAL_COLLECT_RPC_REQUEST_SUMMARY_DIFFERS')
    need(stopped==result['cleanup'],'ACTUAL_DISABLE_RPC_CLEANUP_SUMMARY_DIFFERS')
    for state in (collected,stopped):
        need(state['owner']==public['owner'] and state['generation']==public['generation'],
            'ACTUAL_COLLECT_DISABLE_RPC_OWNER_CHANGED')


def original_rpc_records(root,refs):
    # This one-request protocol repeats the complete 28-call metadata in later
    # replies. Bound all eleven originals before reading any JSON; do not trim
    # their evidence or relax the shared JSON reader's default limit.
    need(type(refs) is list and len(refs)==len(RPC_ACTIONS)
        and [r['action'] for r in refs]==RPC_ACTIONS,'ACTUAL_SINGLE_WINDOW_RPC_COUNT_OR_ORDER')
    inputs=[];paths=set();total=0
    for ref in refs:
        path=E.child(root,ref['path']);size=path.stat().st_size
        need(not path.is_symlink() and 0<size<=RPC_ITEM_LIMIT,'ORIGINAL_RPC_ITEM_BYTE_BOUND')
        identity=str(path.resolve(strict=True))
        need(identity not in paths,'ORIGINAL_RPC_PATH_REUSED');paths.add(identity)
        total+=size;need(total<=RPC_TOTAL_LIMIT,'ORIGINAL_RPC_TOTAL_BYTE_BOUND')
        inputs.append((path,size,ref))
    records=[]
    for path,size,ref in inputs:
        raw=raw_file(path,RPC_ITEM_LIMIT)
        need(len(raw)==size and hashlib.sha256(raw).hexdigest()==ref['sha256'],'ORIGINAL_RPC_ARTIFACT_CHANGED')
        record=json.loads(raw,object_pairs_hook=E.unique,
            parse_constant=lambda x:(_ for _ in ()).throw(ValueError('NONFINITE_JSON')))
        need(record['action']==ref['action'],'ORIGINAL_RPC_ACTION_CHANGED');records.append(record)
    return records,dict(expected_records=len(RPC_ACTIONS),record_count=len(records),
        per_record_limit_bytes=RPC_ITEM_LIMIT,total_limit_bytes=RPC_TOTAL_LIMIT,
        record_bytes=[size for _,size,_ in inputs],total_bytes=total)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('public','returned','native','run','result','build','vendor','source','libraries','out'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('build-sha256','source-manifest-sha256','libraries-sha256'):p.add_argument('--'+name,required=True)
    a=p.parse_args();report=dict(schema='demoncat.tnd28.prepared_input_review.v1',status='INCOMPLETE_OR_REJECTED',issues=[],
        C1=False,C2=False,C3=False,E5=False,fault_authorized=False,normal_stock_private_exact_proven=False,
        old_journal_result_rewritten=False,slice_pod_decoded=False,reader_sha256=E.sha(__file__),
        storage_format_rule=dict(getter='aclTensor::GetStorageFormat',sdk='CANN 8.5.1',
            roles=STORAGE_FORMATS,enum_source_sha256=FORMAT_ENUM_SOURCES,
            ncl_value=47,nd_value=2,public_input_layout='TND',view_format_observed=False))
    sys.addaudithook(E.prohibit_runtime)
    try:
        build,files,metadata,abi=bundle(a.build,a.build_sha256,a.vendor,a.source,a.source_manifest_sha256)
        report['false_lse_build_sources']=false_lse_sources(a.build,build)
        public=json_file(a.public);returned=json_file(a.returned);events=lines(a.native,1024**2)
        result=json_file(a.result,128*1024**2);public_lifecycle(result,public)
        need(result['build_sha256']==a.build_sha256 and result['source_manifest_sha256']==a.source_manifest_sha256,
            'ACTUAL_DEPLOYMENT_SOURCE_OR_BUILD')
        original_rpc,report['rpc_input_budget']=original_rpc_records(a.result.parent,result['rpc'])
        actions=[record['action'] for record in original_rpc]
        public_rpc_summaries(result,public,original_rpc)
        host=E.child(a.result.parent,result['host_receipt']['path'])
        need(E.sha(host)==result['host_receipt']['sha256'] and host.stat().st_size==result['host_receipt']['bytes'],
            'ORIGINAL_HOST_RECEIPT_CHANGED')
        host=json_file(host,128*1024**2)
        need(host['ownership_retained'] is False and [r['action'] for r in host['history']]==actions
            and host['history']==original_rpc and host['attachment'] is None and host['ticket'] is None,'ACTUAL_HOST_FINAL_RELEASE')
        need(public['build_sha256']==a.build_sha256,'PUBLIC_NEW_BUILD_BINDING')
        typed=protocol(public,returned,events)
        need(E.sha(a.libraries)==a.libraries_sha256,'TRUSTED_LIBRARY_MANIFEST_CHANGED');libraries=json_file(a.libraries)
        entries={r['recorded_path']:r for r in libraries['files']}
        need(len(entries)==len(libraries['files'])<=32,'MAPPED_LIBRARY_MANIFEST_AMBIGUOUS')
        seen={}
        for phase,data in typed.items():
            if 'function' not in data:continue
            f=data['function'];r,binding=function_library_entry(f,entries);path=Path(r['path'])
            need(path.stat().st_size==r['bytes']==f['file_bytes'] and E.sha(path)==r['sha256'],'ACTUAL_FUNCTION_LIBRARY_BYTES:'+phase)
            if phase in ('prepare_enter','execute_enter'):
                need(any(name.endswith('/libcust_opapi.so') and row['sha256']==r['sha256'] for name,row in files.items()),
                    'ACTUAL_INNER_FUNCTION_NOT_NEW_PRIVATE_OPAPI')
            if phase in ('workspace_enter','fia_execute_enter'):
                need(r['sha256']==NNOPBASE_SHA,'ACTUAL_NNOPBASE_DIFFERS_FROM_PINNED_SERIALIZER_OWNER')
            seen[f['path']]=dict(binding,bytes=r['bytes'],sha256=r['sha256'])
        need(typed['workspace_enter']['function']['path']==typed['fia_execute_enter']['function']['path'],
            'ACTUAL_NNOPBASE_OWNER_CHANGED')
        public['_return']=returned
        rows=lines(a.run/'runtime/runtime.jsonl',128*1024**2)
        report['actual']=runtime_join(a.run,rows,public,events,abi,build['kernel_binary_sha256'])
        key=int(report['actual']['launches'][1]['tiling_key'])
        need(key in build['compiled_keys'] and key in {int(r['tilingKey']) for r in metadata['kernelList']}
            and key not in (109000000030400003,119000000030400003),'NORMAL_OBSERVATION_MUST_NOT_SELECT_CONTROL_OR_REPLAY')
        report.update(status='SOURCE_BOUND_NORMAL_PREPARED_INPUT_JOIN_RECORDED_NO_F_AUTHORITY',actual_function_libraries=seen,
            build_sha256=a.build_sha256,source_manifest_sha256=a.source_manifest_sha256,
            native_record_count=len(events),current_public_objects_and_actual_typed_calls=True)
    except Exception as error:report['issues'].append(dict(error=repr(error)))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:report[k] for k in ('status','issues','fault_authorized')}))
    return 2 if report['issues'] else 0


if __name__=='__main__':raise SystemExit(main())
