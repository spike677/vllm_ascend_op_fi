"""CPU receipts and real Host serialization; no model, tensor, DSO or NPU proof."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

from experiments.c2c3_closure_20261117 import review_attention28_stock_census as R

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')


def tensor(shape, value='CPU fixture', dtype='torch.float16'):
    return dict(shape=shape,dtype=dtype,sha256=hashlib.sha256(value.encode()).hexdigest(),finite=True)


def elf(key='777000000030000999',prefix='FusedInferAttentionScore_deadbeef',companions=False):
    names=b'\0.text\0.shstrtab\0.strtab\0.symtab\0';strings=b'\0';symbols=bytearray(24)
    for suffix in ('_mix_aic','_mix_aiv'):
        label=prefix+'_'+key+suffix;offset=len(strings);strings+=label.encode()+b'\0'
        symbols.extend(struct.pack('<IBBHQQ',offset,0x12,0,1,64,128))
        if companions:
            offset=len(strings);strings+=label.encode()+b'__\0'
            symbols.extend(struct.pack('<IBBHQQ',offset,0x21,0,1,0,4))
    blob=bytearray(64);parts=[]
    for value in (bytes(256),names,strings,symbols):parts.append((len(blob),len(value)));blob.extend(value)
    while len(blob)%8:blob.append(0)
    shoff=len(blob)
    for row in ((0,0,0,0,0,0,0,0,0,0),(1,1,6,0,*parts[0],0,0,4,0),
                (7,3,0,0,*parts[1],0,0,1,0),(17,3,0,0,*parts[2],0,0,1,0),(25,2,0,0,*parts[3],3,1,8,24)):
        blob.extend(struct.pack('<IIQQQQIIQQ',*row))
    blob[:64]=struct.pack('<16sHHIQQQIHHHHHH',b'\x7fELF\x02\x01\x01'+bytes(9),1,183,1,0,0,shoff,0,64,0,0,64,5,2)
    return bytes(blob)


def call_fixture(index,request,phase):
    path='model.layers.'+str(index)+'.self_attn.attn';inputs={}
    for ordinal,(name,shape,dtype,width) in enumerate((('query',[2048,16,128],'torch.float16',2),
            ('key',[2048,8,128],'torch.float16',2),('value',[2048,8,128],'torch.float16',2),
            ('atten_mask',[2048,2048],'torch.int8',1))):
        size=1;stride=[]
        for n in reversed(shape):stride.insert(0,size);size*=n
        snap=dict(metadata=dict(shape=shape,stride=stride,storage_offset=0,storage_nbytes=size*width,
            object_id=100+ordinal,storage_identity=200+ordinal,dtype=dtype,device='npu:0'),
            bytes=size*width,finite=True,path=None,version=None,sha256=tensor(shape,name)['sha256'])
        if name == 'atten_mask':snap['sha256']=R.mask_sha()
        inputs[name]=dict(kind='tensor',snapshot=snap)
    scalars=dict(input_layout='TND',block_size=128,block_table=None,num_heads=16,num_key_value_heads=8,
        sparse_mode=3,actual_seq_lengths=[2048],actual_seq_lengths_kv=[2048],scale=128**-.5)
    inputs.update({k:dict(kind='scalar',value=v) for k,v in scalars.items()})
    api=dict(inputs=inputs,positional_count=0,supplied_argument_names=sorted(inputs),
        original_callable=dict(type='torch._ops.OpOverloadPacket',object_id=30),
        omitted_argument_defaults=dict(pre_tokens=2147483647,next_tokens=2147483647,inner_precise=0,softmax_lse_flag=False),
        public_default_schema='npu::npu_fused_infer_attention_score(CPU schema fixture)',
        default_schema_dispatch_observed=False,actual_native_attributes_observed=False,
        live_metadata=dict(num_prefills=1,num_decodes=0,num_actual_tokens=2048,causal=True,
            seq_lens_list=[2048],actual_seq_lengths_q=[2048],mask_is_actual_metadata_object=True))
    scheduler=dict(internal_request_id=request,num_requests=1,num_computed_tokens=0,num_prompt_tokens=2048,num_scheduled_tokens=2048)
    call=dict(module_path=path,module_object_id=1000+index,aliases=[path],parent_path=path.rpartition('.')[0],
        request_id=request,actual_operator='torch_npu.npu_fused_infer_attention_score',occurrence=0,ordinal=index,
        scheduler=scheduler,original_calls=1,original_returned=True,inputs_unchanged=True,native=None,
        actual_callable=api['original_callable'],input=dict(scalars),output=[tensor([2048,16,128],'output'),tensor([0],'empty','torch.float32')])
    for k in ('query','key','value','atten_mask'):
        snap=inputs[k]['snapshot'];call['input'][k]={f:(snap['metadata'][f] if f in ('shape','dtype') else snap[f])
            for f in ('shape','dtype','sha256','finite')}
    call['target_key']=R.E.digest(dict(path=path,operator=call['actual_operator'],occurrence=0,input=call['input'],decode=False))
    scope=dict(owner='cpu-owner',generation=1,rank=0,logical_device=0,device_type='npu',phase=phase,
        worker_object_id=10,runner_object_id=11,model_object_id=12)
    c={k:call[k] for k in ('request_id','module_path','module_object_id','actual_operator','occurrence','ordinal','target_key')}
    c.update(schema='demoncat.attention28.stock_native_census.v1',status='OBSERVED_INTERCEPTED_STOCK_LAUNCHES',
        reasons=[],api=api,api_inputs_after=deepcopy(api),execution_scope=deepcopy(scope),original_calls=1,original_returned=True,
        input_substitutions=0,operator_arguments_modified=False,fault_authorized=False,native_fault_qualified=False,
        thread_identity=dict(pid=12,python_thread_ident=1234567,native_tid=34,source='CURRENT_THREAD_GET_IDENT_AND_GET_NATIVE_ID'))
    call.update(native_census=c,native_census_scope=scope)
    return call


def fixture(root):
    runtime=root/'runtime';runtime.mkdir();binary=runtime/'registered-12-1.bin';binary.write_bytes(elf())
    def row(event,now,token='',**fields):
        return dict(schema='demoncat.attention.runtime_observer.v1',event=event,pid=12,tid=34,monotonic_ns=now,call_id=token,**fields)
    rows=[row('register',1,handle=80,generation=1,result=0,binary_file=binary.name,binary_bytes=binary.stat().st_size)]
    requests=[]
    for ri,phase in enumerate(R.PHASES):
        request='cpu-request-'+str(ri);calls=[]
        for index in range(28):
            call=call_fixture(index,request,phase);token=format(ri*28+index+1,'032x');now=(ri*28+index+1)*10
            call['native_census']['call_id']=token
            rows.extend([row('begin',now,token),row('function_bind',now+1,token,handle=80,generation=1,
                function_handle=90,tiling_key='777000000030000999',result=0),
                row('launch',now+4,token,api='aclrtLaunchKernelWithHostArgs',handle=80,generation=1,function_handle=90,
                    binary_file=binary.name,tiling_key='777000000030000999',block_dim=16,schedule_mode=1,
                    launch_begin_ns=now+2,launch_end_ns=now+3,stream=9,result=0,snapshot_valid=True,
                    observer_mutates_arguments=False,args_hex='01020304',args_size=4,placeholders_hex='',argument_layout='aclrt_host_placeholders'),
                row('stream_sync',now+5,token,stream=9,result=0),row('device_sync',now+6,token,result=0),row('end',now+7,token)])
            save_call(root,call);calls.append(call)
        forward=dict(request_id=request,phase=phase,calls=calls,decode=False,root_returned=True,scheduler=calls[0]['scheduler'],
            input=[[],dict(input_ids=tensor([2048],'ids','torch.int32'),positions=tensor([2048],'positions','torch.int64'))],
            hidden=tensor([2048,1024],'hidden'),logits=tensor([1,151936],'logits'))
        requests.append(dict(phase=phase,mapping=dict(internal_request_id=request,external_request_id=request),output_tokens=[7],
            worker=dict(phase=phase,request_id=request,forwards=[forward])))
    (runtime/'runtime.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    return dict(requests=requests),rows


def save_call(root,call):
    c=call['native_census'];c.pop('artifact',None);path=root/'native_calls'/(c['call_id']+'.json');write(path,c)
    c['artifact']=dict(path='/evidence/probe/native_calls/'+path.name,sha256=R.E.sha(path),bytes=path.stat().st_size)


def cached_auxiliary(root,rows):
    for r in rows:
        for key in ('monotonic_ns','launch_begin_ns','launch_end_ns'):
            if key in r:r[key]*=10
    binary=root/'runtime/registered-12-2.bin';binary.write_bytes(elf('524111000','Slice_cpu_fixture',companions=True))
    base={k:rows[0][k] for k in ('schema','pid','tid')}
    extra=[dict(base,event='register',monotonic_ns=11,call_id='',handle=81,generation=2,result=0,
                binary_file=binary.name,binary_bytes=binary.stat().st_size),
           dict(base,event='function_bind',monotonic_ns=12,call_id='',handle=81,generation=2,
                function_handle=91,tiling_key='524111000',result=0)]
    result=[rows[0],*extra]
    for r in rows[1:]:
        result.append(r)
        if r['event'] == 'begin':
            now=r['monotonic_ns']
            result.append(dict(base,event='launch',monotonic_ns=now+3,call_id=r['call_id'],
                api='aclrtLaunchKernelWithHostArgs',handle=81,generation=2,function_handle=91,
                binary_file=binary.name,tiling_key='524111000',block_dim=40,schedule_mode=0,
                launch_begin_ns=now+1,launch_end_ns=now+2,stream=9,result=0,snapshot_valid=True,
                observer_mutates_arguments=False,args_hex='00'*208,args_size=208,placeholders_hex='',argument_layout='aclrt_host_placeholders'))
    return result


def protocol_fixture(root,data):
    worker=dict(pid=12,rank=0,logical_device=0);mechanism=R.MECHANISMS[1]
    attached=dict(tenant_id='tenant',attach_nonce='nonce',worker=worker)
    ticket=dict(attached,mechanism=mechanism,implementation_version='1',owner='cpu-owner',generation=1)
    state=dict(owner='cpu-owner',generation=1,simulation=False,registered=True,binding_reserved=True,
        C1=False,C2=False,C3=False,public_qualification_granted=False,
        inflight=0,leases=0,owned_hooks=0,owned_api_wrappers=0,enabled=False,native_registered=False,
        replay_pending=False,replay_failed=False,
        native_state=dict(scope_active=False,scope_owned=False,environment_owned=0,pending_drains=0,pending_references=0,
            first_error=None,failures=[],observer_only=True),
        serving=dict(pending_request=None,failures=[],resource_state=dict(failures=[],failed_requests=[],pending_request=None,needs_off=False)),
        evidence=dict(requests=[]))
    history=[dict(action='attach',response=dict(status='ATTACHED',ticket=attached))]
    # Independent of the reader's expectation: these are wire actions, while
    # the third step of each request is the collect_replay() convenience call.
    actions=['prepare','register','check','enable','advance','query','advance',
             'advance','query','advance','finish','disable','unregister']
    for index,action in enumerate(actions,1):
        current=deepcopy(state)
        if 5 <= index <= 10:
            ri=(index-5)//3;request=data['requests'][ri]
            collecting=index in (7,10)
            current['evidence']['requests']=[x['worker'] for x in data['requests'][:ri+collecting]]
            if not collecting:current['serving']['pending_request']=[request['mapping']['internal_request_id'],request['phase']]
        elif index >= 11:current['evidence']['requests']=[x['worker'] for x in data['requests']]
        if action == 'unregister':current.update(registered=False,binding_reserved=False)
        history.append(dict(action=action,response=dict(status='QUERY_OK' if action=='query' else 'ACTION_COMPLETED',
            action=action,ticket=deepcopy(ticket),recovery_required=False,provider_state=current)))
    history.append(dict(action='detach',response=dict(status='DETACHED',recoveries=[])))
    host=dict(schema='demoncat.worker_core.host_receipt.v1',worker=worker,owner='cpu-owner',tenant_id='tenant',
        history=history,ticket=None,attachment=None,census_ticket=None,ownership_retained=False)
    data.update(setup=dict(worker=worker),mechanism=mechanism,final_cleanup=deepcopy(history[-4]['response']['provider_state']),
        binding_removal=dict(status='SERVING_DEPLOYMENT_REMOVED',mechanism=mechanism,native_scope_active=False),
        healthy_before=[7],post_unregister_health=dict(mapping=dict(internal_request_id='cpu-post-unregister'),
            output_tokens=[7],scope='TOKENS_ONLY_NO_POST_UNREGISTER_LOGITS_OBSERVER'))
    save_protocol(root,data,host)
    return host


def save_protocol(root,data,host):
    refs=[]
    for i,row in enumerate(host['history']):
        name=f"worker_core_rpc/{i:03}-{row['action']}.json";path=root/name;write(path,row)
        refs.append(dict(action=row['action'],path=name,sha256=R.E.sha(path)))
    path=root/'host_receipt.json';write(path,host)
    data['host_receipt']=dict(path=path.name,sha256=R.E.sha(path),bytes=path.stat().st_size)
    data['worker_core']=dict(receipts=refs,ownership_retained=False)


class IndependentStockCensusTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(self.enterContext(tempfile.TemporaryDirectory()));self.data,self.rows=fixture(self.root)
        source=dict(root=ROOT,pins={R.RAW_READER:R.E.sha(ROOT/R.RAW_READER)})
        self.parser=R.raw_reader(source);self.worker=dict(pid=12,rank=0,logical_device=0)

    def review(self):
        a=R.Review(self.root)
        result=R.review_files(a,self.data,self.parser,worker=self.worker,owner='cpu-owner',generation=1)
        return a,result

    def rewrite(self):
        (self.root/'runtime/runtime.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows),encoding='utf-8')

    def test_full_56_actual_scopes_and_unknown_actual_key_are_read_without_expected_key(self):
        a,result=self.review();self.assertEqual(a.issues,[])
        self.assertEqual(result['independently_resolved_scopes'],56)
        self.assertEqual(result['actual_registered_entry_keys'],['777000000030000999'])
        self.assertTrue(all(p['native_algorithm_branch'].startswith('UNRESOLVED_') for p in result['actual_launch_profiles']))
        self.assertEqual(result['scopes'][-1]['actual_event_counts']['launch'],1)

    def test_complete_api_hash_input_binding_rejects_changed_value_dtype_stride_and_mask(self):
        original=self.data['requests'][0]['worker']['forwards'][0]['calls'][0]
        mutations=(lambda c:c['input']['query'].__setitem__('sha256','f'*64),
            lambda c:c['native_census']['api']['inputs']['value']['snapshot']['metadata'].__setitem__('dtype','torch.bfloat16'),
            lambda c:c['native_census']['api']['inputs']['value']['snapshot']['metadata'].__setitem__('stride',[999999,128,1]),
            lambda c:c['native_census']['api']['inputs']['atten_mask']['snapshot'].__setitem__('sha256','f'*64))
        for mutate in mutations:
            call=deepcopy(original);mutate(call)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):R.api_input(call,'cpu-request-0')

    def test_after_capture_mutation_and_defaults_claiming_native_observation_refuse(self):
        call=self.data['requests'][0]['worker']['forwards'][0]['calls'][0]
        call['native_census']['api_inputs_after']['live_metadata']['num_actual_tokens']=2047
        with self.assertRaisesRegex(ValueError,'AFTER_DIFFERS'):R.api_input(call,'cpu-request-0')
        call['native_census']['api']['actual_native_attributes_observed']=True
        with self.assertRaisesRegex(ValueError,'DECLARED_DEFAULTS'):R.api_input(call,'cpu-request-0')

    def test_missing_call_is_not_hidden_by_summary_success(self):
        self.data['requests'][1]['worker']['forwards'][0]['calls'].pop()
        self.data['stock_launch_census']=dict(observed_call_scopes=56,intercepted_call_denominator_complete=True)
        a,result=self.review()
        self.assertIn('COMPLETE_28_NATURAL_PREFILL_CALLS',[x['reason'] for x in a.issues])
        self.assertEqual(result['independently_resolved_scopes'],55)

    def test_old_original_unobserved_reasons_are_retained_after_new_offline_resolution(self):
        call=self.data['requests'][0]['worker']['forwards'][0]['calls'][0]
        call['native_census'].update(status='UNOBSERVED_OR_INCOMPLETE',reasons=['original capture gap']);save_call(self.root,call)
        a,result=self.review();self.assertEqual(a.issues,[])
        self.assertEqual(result['original_unobserved_scopes'],[dict(where='observe/0',reasons=['original capture gap'])])

    def test_foreign_thread_old_conflict_and_missing_completion_refuse_actual_scope(self):
        call=self.data['requests'][1]['worker']['forwards'][0]['calls'][0];original=deepcopy(self.rows)
        mutate=[lambda:self.rows[2].__setitem__('tiling_key','33'),
            lambda:call['native_census']['thread_identity'].__setitem__('native_tid',1234567)]
        for change in mutate:
            self.rows=deepcopy(original);change()
            with self.subTest(change=change),self.assertRaises(ValueError):
                R.raw_scope(R.Review(self.root),self.parser,self.rows,call,self.worker,'cpu-owner',1)

    def test_wrong_owner_and_duplicate_tokens_do_not_reconcile(self):
        call=self.data['requests'][0]['worker']['forwards'][0]['calls'][0]
        with self.assertRaisesRegex(ValueError,'OWNER_WORKER'):
            R.raw_scope(R.Review(self.root),self.parser,self.rows,call,self.worker,'foreign-owner',1)
        self.rows.append(deepcopy(self.rows[1]));self.rewrite()
        with self.assertRaisesRegex(ValueError,'BEGIN_END'):
            R.raw_scope(R.Review(self.root),self.parser,self.rows,call,self.worker,'cpu-owner',1)

    def test_actual_elf_entry_must_match_key_and_original_binary_hash(self):
        call=self.data['requests'][0]['worker']['forwards'][0]['calls'][0]
        path=self.root/'runtime/registered-12-1.bin';path.write_bytes(elf('888000000030000999'))
        with self.assertRaisesRegex(ValueError,'SYMBOLS_UNOBSERVED'):
            R.raw_scope(R.Review(self.root),self.parser,self.rows,call,self.worker,'cpu-owner',1)

    def test_cached_auxiliary_is_not_dropped_or_misidentified_as_an_attention_entry(self):
        self.rows=cached_auxiliary(self.root,self.rows);self.rewrite()
        a,result=self.review();self.assertEqual(a.issues,[])
        self.assertEqual(result['binding_kinds'],dict(CACHED_PRE_SCOPE_LAST_EFFECTIVE_BIND=56,FRESH_CURRENT_SCOPE_BIND=56))
        self.assertEqual(sum(len(s['launches']) for s in result['scopes']),112)
        auxiliary=next(p for p in result['actual_launch_profiles'] if p['actual_tiling_key']=='524111000')
        self.assertEqual(len(auxiliary['matching_non_function_symbols']),2)
        self.assertTrue(all(s['name'].startswith('Slice_cpu_fixture') and len(s['code_sha256'])==64
            for s in auxiliary['selected_entry_symbols']))

    def test_cached_binding_foreign_thread_conflicting_key_and_missing_history_refuse(self):
        self.rows=cached_auxiliary(self.root,self.rows);original=deepcopy(self.rows)
        call=self.data['requests'][0]['worker']['forwards'][0]['calls'][0]
        for mutation in (lambda r:r[2].__setitem__('tid',99),lambda r:r[2].__setitem__('tiling_key','wrong'),
                         lambda r:r.pop(2),lambda r:r.insert(3,dict(r[1],generation=3,monotonic_ns=13))):
            rows=deepcopy(original);mutation(rows)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                R.raw_scope(R.Review(self.root),self.parser,rows,call,self.worker,'cpu-owner',1)

    def test_later_foreign_binding_and_ambiguous_latest_timestamp_cannot_be_cached(self):
        self.rows=cached_auxiliary(self.root,self.rows);call=self.data['requests'][0]['worker']['forwards'][0]['calls'][0]
        for time in (12,99):
            rows=deepcopy(self.rows);rows.insert(3,dict(rows[2],monotonic_ns=time,call_id='foreign'))
            with self.subTest(time=time),self.assertRaises(ValueError):
                R.raw_scope(R.Review(self.root),self.parser,rows,call,self.worker,'cpu-owner',1)

    def test_data_companions_are_preserved_but_empty_or_out_of_section_functions_refuse(self):
        path=self.root/'runtime/registered-12-1.bin';path.write_bytes(elf(companions=True))
        value=R.RAW.entry_symbols(path,R.E.sha(path),selected_keys=['777000000030000999'])
        self.assertEqual(len(value['actual_selected_entry_symbols']),2)
        self.assertEqual(len(value['matching_non_function_symbols']),2)
        raw=bytearray(path.read_bytes());shoff=struct.unpack_from('<Q',raw,40)[0]
        symoff=struct.unpack_from('<Q',raw,shoff+4*64+24)[0]
        struct.pack_into('<Q',raw,symoff+24+16,999999);path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'OUTSIDE_EXECUTABLE_SECTION'):
            R.RAW.entry_symbols(path,R.E.sha(path),selected_keys=['777000000030000999'])

    def test_changed_per_call_file_is_rejected_without_loading_a_runtime(self):
        call=self.data['requests'][0]['worker']['forwards'][0]['calls'][0];ref=call['native_census']['artifact']
        (self.root/'native_calls'/Path(ref['path']).name).write_text('{}')
        with self.assertRaisesRegex(ValueError,'HASH_OR_SIZE'):R.Review(self.root).artifact(ref)

    def test_cold_import_and_explicit_audit_refuse_framework_and_dynamic_loading(self):
        script=ROOT/'experiments/c2c3_closure_20261117/review_attention28_stock_census.py'
        code="""import importlib.util,sys
spec=importlib.util.spec_from_file_location('review',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sys.addaudithook(m.E.prohibit_runtime)
assert not any(x in sys.modules for x in ('torch','torch_npu','vllm','numpy'))
for event,args in [('ctypes.dlopen',('forbidden.so',)),('import',('torch',None,None,None,None))]:
 try:sys.audit(event,*args)
 except RuntimeError:pass
 else:raise AssertionError(event)
print('STDLIB_NO_DEVICE_OR_DSO')
"""
        result=subprocess.run([sys.executable,'-B','-c',code,str(script)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('STDLIB_NO_DEVICE_OR_DSO',result.stdout)

    def test_all_15_raw_public_messages_and_cleanup_are_independently_joined(self):
        protocol_fixture(self.root,self.data)
        worker,owner,generation=R.protocol(R.Review(self.root),self.data)
        self.assertEqual((worker,owner,generation),(self.worker,'cpu-owner',1))

    def test_real_host_collect_produces_advance_rpc_and_reader_accepts_original_history(self):
        from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
        from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig

        expected=protocol_fixture(self.root,self.data);test=self

        class SerializedWorkerExecutor:
            """Only the Worker endpoint is a fixture; Host dispatch is real."""
            def __init__(self):self.calls=[];self.index=0

            def collective_rpc(self,method,timeout=None,args=()):
                self.calls.append((method,deepcopy(args)))
                if method == 'demoncat_worker_identity':
                    return [dict(status='WORKER_IDENTIFIED',worker=test.worker,quarantined=False)]
                if method == 'demoncat_capabilities':
                    return [dict(status='CAPABILITIES',worker=test.worker,
                        providers=[dict(mechanism=test.data['mechanism'],implementation_version='1')])]
                row=expected['history'][self.index];self.index+=1
                wire_action={'demoncat_attach':'attach','demoncat_detach':'detach',
                             'demoncat_fault_query':'query'}.get(method)
                if method == 'demoncat_fault_rpc':wire_action=args[0]
                test.assertEqual(wire_action,row['action'])
                return [deepcopy(row['response'])]

        executor=SerializedWorkerExecutor()
        client=WorkerFaultClient(HostAdapter(executor),FaultConfig(28,'fia.denominator_replay'))
        client.owner=expected['owner'];client.tenant_id=expected['tenant_id']
        for method in ('prepare','register','check','enable'):
            self.assertEqual(getattr(client,method)()['status'],'ACTION_COMPLETED')
        for row in self.data['requests']:
            self.assertEqual(client.advance(row['mapping']['internal_request_id'])['status'],'ACTION_COMPLETED')
            self.assertEqual(client.query()['status'],'QUERY_OK')
            self.assertEqual(client.collect_replay()['status'],'ACTION_COMPLETED')
        for method in ('finish','disable','unregister'):
            self.assertEqual(getattr(client,method)()['status'],'ACTION_COMPLETED')
        host=client.detach();self.assertEqual(host['status'],'DETACHED')
        self.assertEqual(executor.index,15)
        settings=[args[1]['settings'] for method,args in executor.calls
                  if method == 'demoncat_fault_rpc' and args[0] == 'advance']
        self.assertEqual(settings,[dict(request_id='cpu-request-0'),dict(collect=True),
                                   dict(request_id='cpu-request-1'),dict(collect=True)])
        self.assertEqual(host['history'],expected['history'])
        self.assertEqual([host['history'][i]['action'] for i in (5,7,8,10)],['advance']*4)
        self.assertTrue(all(set(row) == {'action','response'} for row in host['history']))
        save_protocol(self.root,self.data,host)
        self.assertEqual(R.protocol(R.Review(self.root),self.data),(self.worker,'cpu-owner',1))

    def test_collect_method_alias_or_uncollected_request_is_not_an_original_rpc_receipt(self):
        original=protocol_fixture(self.root,self.data)
        mutations=(lambda h:h['history'][7].__setitem__('action','collect_replay'),
            lambda h:h['history'][7]['response'].__setitem__('action','collect_replay'),
            lambda h:h['history'][7]['response']['provider_state']['serving'].__setitem__(
                'pending_request',['cpu-request-0','observe']),
            lambda h:h['history'][5]['response']['provider_state']['evidence'].__setitem__(
                'requests',[self.data['requests'][0]['worker']]))
        for mutation in mutations:
            host=deepcopy(original);mutation(host);save_protocol(self.root,self.data,host)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                R.protocol(R.Review(self.root),self.data)

    def test_cleanup_or_foreign_ticket_cannot_be_hidden_by_probe_success(self):
        host=protocol_fixture(self.root,self.data);original=deepcopy(host)
        for mutation in (lambda h:h['history'][-2]['response']['provider_state']['native_state'].__setitem__('pending_drains',1),
                lambda h:h['history'][7]['response']['ticket'].__setitem__('owner','foreign'),
                lambda h:h['history'][6]['response']['provider_state']['evidence'].__setitem__('requests',[self.data['requests'][0]['worker']]),
                lambda h:h['history'][-2]['response']['provider_state'].__setitem__('binding_reserved',True)):
            host=deepcopy(original);mutation(host);save_protocol(self.root,self.data,host)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):R.protocol(R.Review(self.root),self.data)

    def test_cold_complete_cli_rehashes_frozen_sources_observer_and_all_originals(self):
        protocol_fixture(self.root,self.data)
        source=self.root/'frozen-source';files=[]
        for name in R.REQUIRED:
            path=source/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,path)
            files.append(dict(path=name,sha256=R.E.sha(path),bytes=path.stat().st_size))
        manifest=source/'source_manifest.json';write(manifest,dict(schema='demoncat.c2c3.source.v1',base_commit='CPU-fixture',files=files))
        observer=self.root/'observer/attention-preflight';observer.mkdir(parents=True)
        library=observer/'observer-fixture.so';library.write_bytes(b'CPU bytes, never a loadable DSO')
        receipt=dict(observer_compiled=True,actual_npu_calls=0,observer_memory_selftest=dict(exit_code=0),
            observer_source=dict(sha256=R.E.sha(ROOT/(R.E.NATIVE+'launch_observer.cpp'))),
            observer_library=dict(path='/output/CPU-fixture/observer-fixture.so',sha256=R.E.sha(library)))
        write(observer/'result.json',receipt)
        self.data.update(mode='ACTUAL_TND_PREFILL_STOCK_NATIVE_CENSUS_ONLY',status='CPU_FIXTURE_NO_NPU_EVIDENCE',
            probe_sha256=R.E.sha(ROOT/(R.E.EXPERIMENT+'natural_attention28_stock_census_probe.py')),
            observer_receipt_sha256=R.E.sha(observer/'result.json'),observer_library_sha256=R.E.sha(library),
            forced_private_vendor=False,fault_authorized=False,public_qualification_granted=False,C1=False,C2=False,C3=False)
        write(self.root/'result.json',self.data)
        output=self.root/'review.json';script=ROOT/(R.E.EXPERIMENT+'review_attention28_stock_census.py')
        argv=[sys.executable,'-B',str(script),'--root',str(self.root),'--source',str(source),'--observer',str(observer.parent),
            '--result-sha256',R.E.sha(self.root/'result.json'),'--source-manifest-sha256',R.E.sha(manifest),
            '--runtime-sha256',R.E.sha(self.root/'runtime/runtime.jsonl'),'--registered-elf-sha256',
            'registered-12-1.bin='+R.E.sha(self.root/'runtime/registered-12-1.bin'),'--out',str(output)]
        result=subprocess.run(argv,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        report=json.loads(output.read_bytes());self.assertEqual(report['issues'],[])
        self.assertEqual(report['census']['independently_resolved_scopes'],56)
        self.assertFalse(report['fault_authorized'] or report['C1'] or report['C2'] or report['C3'])

    def test_external_raw_and_binary_hashes_cannot_be_replaced_by_current_file_hashes(self):
        a=R.Review(self.root,expected_runtime='0'*64)
        with self.assertRaisesRegex(ValueError,'PINNED_RUNTIME'):
            R.review_files(a,self.data,self.parser,worker=self.worker,owner='cpu-owner',generation=1)
        a=R.Review(self.root,expected_elf={'registered-12-1.bin':'0'*64})
        with self.assertRaisesRegex(ValueError,'PINNED_REGISTERED_ELF'):
            R.raw_scope(a,self.parser,self.rows,self.data['requests'][0]['worker']['forwards'][0]['calls'][0],self.worker,'cpu-owner',1)


if __name__ == '__main__':unittest.main()
