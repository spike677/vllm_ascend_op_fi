"""CPU adapter/independent reader contracts. Fixtures are not NPU originals."""
import copy
import ctypes as C
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from fault_injection import fia_input_journal as native
from fault_injection import quant_container_native as qct
from experiments.worker_core_v11_20261118 import fia_input_journal_review as reader
from experiments.worker_core_v11_20261118 import fia_input_journal_build as builder


def blank(cls):
    return native.record(qct.instance(cls))


def tensor(address=80000):
    value=native.record(qct.Tensor())
    value.update(descriptor=address+100,address=address,counts=[3,3,1],called_mask=127,
        offset=3072,shape=[2048,8,128,0,0,0,0,0],strides=[4096,128,1,0,0,0,0,0],
        storage=[8388608,0,0,0,0,0,0,0],dtype=1,format=2,complete=1)
    return value


def fixture():
    """One entirely synthetic typed call, used only for rejection regressions."""
    profile=dict(candidate='fia_input_journal',profile_sha256='b'*64,originals={})
    for i,key in enumerate(('opapi','nn','device_runtime','observer','torch_npu')):
        profile['originals'][key]=dict(file_sha256=str(i)*64,segments=[dict(vaddr=0,size=4096)])
    s=blank(native.State);s.update(installed=1,stopped=1,pid=101,thread=201,owner_hi=3,owner_lo=4,generation=1,
        total_calls=2,total_events=4,retained_events=4,original_prepares=1,original_executes=1,
        prepare_returns=1,execute_returns=1,paired_execute_returns=1)
    owners={}
    for i,(field,key) in enumerate(dict(prepare_function='opapi',execute_function='opapi',device_function='device_runtime',
            list_size_function='nn',int_size_function='nn',observer_scope_function='observer').items(),1):
        s[field]=10000+i;owners[field]=dict(address=10000+i,base=10000,offset=i,
            sha256=profile['originals'][key]['file_sha256'])
    scope=blank(native.ScopeIdentity);scope.update(pid=101,worker=1001,model=1002,module=1003,call_id='a'*32)
    scope['scope'].update(token=1,forward=1,operator_sequence=1,boundary_sequence=1,stage=3,thread=201)
    worker=dict(pid=101,process_start_identity='proc:cpu-fixture:101:1',worker_object_id=1001,model_object_id=1002)
    cs=blank(qct.State);cs.update(installed=1,stopped=1,owner_hi=3,owner_lo=4,generation=1,owner_thread=201,
        total_calls=8,total_events=16,retained_events=16,entries=[2]*4,returns=[2]*4,original_calls=[2]*4)
    cs['arrays'].update(acquired_arrays=24,released_arrays=24)
    cevents=[];creates=[]
    for call in range(1,5):
        op=1 if call<=2 else 3
        row=blank(qct.Event);row.update(call=call,registration=call,operation=op,scope_token=1,forward=1,
            operator_sequence=1,boundary_sequence=1,stage=3,pid=101,thread=201,function=11000+op,function_base=10000,
            function_offset=1000+op,caller=22000,caller_base=20000,caller_offset=2000,caller_kind=1,count=1,
            payload_offset=(call-1)*C.sizeof(qct.Tensor),payload_bytes=C.sizeof(qct.Tensor) if op==1 else 8)
        if op==1:row['tensor_descriptors']=[tensor(80000+call*1000)]
        else:row['integer_values']=[2048]
        before=copy.deepcopy(row);before.update(sequence=len(cevents)+1,phase=1,monotonic_ns=call*10)
        after=copy.deepcopy(row);after.update(sequence=len(cevents)+2,phase=2,monotonic_ns=call*10+1,
            handle=30000+call,original_calls=1)
        cevents.extend((before,after));creates.append((before,after))
    for i,(b,a) in enumerate(creates,5):
        row=copy.deepcopy(a);row.update(call=i,operation=a['operation']+1,payload_offset=0,payload_bytes=0,
            function=11000+a['operation']+1,function_offset=1000+a['operation']+1)
        for phase in (1,2):
            e=copy.deepcopy(row);e.update(sequence=len(cevents)+1,phase=phase,original_calls=phase-1,monotonic_ns=100+i*10+phase)
            cevents.append(e)
    prep=blank(native.Event);prep.update(operation=1,phase=1,sequence=1,call=1,prepare_registration=1,
        owner_hi=3,owner_lo=4,generation=1,scope_token=1,forward=1,operator_sequence=1,boundary_sequence=1,
        pid=101,thread=201,worker=1001,model=1002,module=1003,function=s['prepare_function'],function_base=10000,
        function_offset=1,caller=22000,caller_base=20000,caller_offset=2000,monotonic_ns=100,device=0,
        call_id='a'*32,scale=.125,signed_scalars=[16,-7,11,8,3,0,128,0,0,0],layout='TND',layout_length=3)
    for n in (0,2,22):
        prep['tensors'][n]=tensor(90000+n*1000)
        prep['formals'][reader.TENSOR_INDICES[n]]=prep['tensors'][n]['descriptor']
    for index,value in zip(reader.SCALAR_INDICES,prep['signed_scalars']):prep['formals'][index]=value&((1<<64)-1)
    prep['formals'][28]=struct.unpack('<Q',struct.pack('<d',.125))[0]
    prep['formals'][31]=70000;prep['formals'][42]=70008;prep['formals'][43]=70016
    for n,index in enumerate(reader.CONTAINER_INDICES):
        slot=prep['containers'][n];slot.update(formal=index,kind=1 if n<2 else 3,joined=1)
        if n<4:
            b,a=creates[n];slot.update(handle=a['handle'],registration=a['call'],creator_entry=b['sequence'],
                creator_return=a['sequence'],count=1,actual_count=1,payload_offset=b['payload_offset'],payload_bytes=b['payload_bytes'])
            prep['formals'][index]=slot['handle']
        else:slot['live_at_return']=1
    returned=copy.deepcopy(prep);returned.update(phase=2,sequence=2,monotonic_ns=101,original_calls=1,
        original_status_valid=1,executor=50000,workspace_bytes=64)
    for slot in returned['containers']:slot['live_at_return']=1
    execute=copy.deepcopy(prep);execute.update(operation=2,sequence=3,call=2,monotonic_ns=110,
        function=s['execute_function'],function_offset=2,executor=50000,workspace=60000,workspace_bytes=64)
    execute['formals']=[60000,64,50000,65000]+[0]*40
    executed=copy.deepcopy(execute);executed.update(phase=2,sequence=4,monotonic_ns=111,original_calls=1,original_status_valid=1)
    roles=dict(query=prep['tensors'][0],atten_mask=prep['tensors'][2],key=creates[0][0]['tensor_descriptors'][0],
        value=creates[1][0]['tensor_descriptors'][0])
    actual={name:dict(object_id=i+1,storage_identity=i+10,storage_address=t['address'],data_address=t['address']+6144,
        storage_bytes=16777216,storage_offset=3072,element_bytes=2,shape=[2048,8,128],stride=[4096,128,1],dtype='torch.float16',device='npu:0')
        for i,(name,t) in enumerate(roles.items())}
    data=dict(schema='demoncat.fia_input_journal.observation.v1',source='ACTUAL_NATIVE_RECORDS',state=s,
        build_sha256='c'*64,tensor_formals=list(reader.TENSOR_INDICES),container_formals=list(reader.CONTAINER_INDICES),
        C1=False,C2=False,C3=False,E5=False,qualified=False,slice_pointer_lineage_proven=False,
        prepared_internal_tensor_lineage_proven=False,tensor_list_members_reread_at_gws=False,
        tensor_list_members_captured_at='ACTUAL_CREATOR_ENTRY',worker=worker,
        startup=dict(status='RESOLVED_BEFORE_LLM',explicit_operator_calls=0,explicit_device_queries=0,install_calls=0,
            identity=dict(pid=101,native_tid=201,process_start_identity=worker['process_start_identity']),original_functions=owners),
        containers=dict(state=cs,events=cevents,profile_sha256='b'*64,build_sha256='c'*64),
        events=[prep,returned,execute,executed],sdk_tags=dict(float16_tag=1,int8_tag=2,nd_tag=2),
        public_scopes=[dict(identity=scope,status='RETURNED',original_calls=1,original_error=None,
            evidence=dict(actual_input_addresses=actual))])
    final=copy.deepcopy(s);final.update(installed=0,retired=1,retained_events=0)
    fc=copy.deepcopy(cs);fc.update(installed=0,retired=1,retained_events=0)
    released=dict(fia=final,containers=fc,global_wrappers_unloaded=False,preloaded_library_remains_until_process_exit=True)
    return data,profile,released


class IndependentReviewTests(unittest.TestCase):
    def result(self,mutate=None):
        data,profile,released=fixture()
        if mutate:mutate(data,profile,released)
        return reader.review(data,profile,released)

    def test_complete_synthetic_contract_is_not_qualification(self):
        result=self.result()
        self.assertEqual(result['issues'],[])
        self.assertEqual(result['native_failures'],[])
        self.assertEqual(result['status'],'COMPLETE_ARGUMENT_JOURNAL_NO_QUALIFICATION')
        self.assertTrue(result['observed']['public_storage_geometry_all_joined'])
        for key in ('C1','C2','C3','E5','qualification_granted','slice_pointer_lineage_proven','prepared_internal_tensor_lineage_proven'):
            self.assertIs(result[key],False)

    def test_opaque_value_member_missing_is_not_descriptor_proof(self):
        result=self.result(lambda d,p,r:d['containers']['events'][2].pop('tensor_descriptors'))
        self.assertIn('ACTUAL_LIST_MEMBERS_PRESENT',result['issues'])

    def test_repeated_slot_cannot_hide_missing_container(self):
        result=self.result(lambda d,p,r:d['events'][0]['containers'][1].update(formal=1))
        self.assertIn('CONTAINER_SLOT_ORDER',result['issues'])

    def test_wrong_thread_is_not_same_public_scope(self):
        result=self.result(lambda d,p,r:d['events'][2].update(thread=999))
        self.assertIn('EXACT_CURRENT_PUBLIC_SCOPE',result['issues'])

    def test_wrong_stream_and_executor_are_rejected(self):
        for field,value in (('formals',[60000,64,99999,0]+[0]*40),('prepare_registration',999)):
            result=self.result(lambda d,p,r:d['events'][2].update({field:value}))
            self.assertTrue(result['issues'])

    def test_foreign_caller_offset_is_rejected(self):
        result=self.result(lambda d,p,r:d['events'][0].update(caller=99999,caller_offset=79999))
        self.assertIn('REAL_FUNCTION_AND_CALLER',result['issues'])

    def test_original_not_once_is_rejected(self):
        result=self.result(lambda d,p,r:d['events'][1].update(original_calls=2))
        self.assertIn('ORIGINAL_RETURN_PRESERVED',result['issues'])

    def test_pending_failed_original_is_preserved(self):
        result=self.result(lambda d,p,r:d['state'].update(untracked_executors=1,failed_prepare_outputs_unknown=1,first_error=9))
        self.assertEqual(result['status'],'NO_OBSERVATION_OR_PARTIAL_FAILURE')
        self.assertTrue(result['native_failures'])

    def test_missing_return_or_null_is_partial(self):
        for mutate in (lambda d,p,r:d['events'].pop(),lambda d,p,r:d['events'][0].update(tensors=None)):
            self.assertEqual(self.result(mutate)['status'],'NO_OBSERVATION_OR_PARTIAL_FAILURE')

    def test_prior_registration_reuse_is_rejected(self):
        result=self.result(lambda d,p,r:d['events'][0]['containers'][1].update(registration=1))
        self.assertIn('CREATOR_GWS_DESTROY_JOIN',result['issues'])

    def test_descriptor_must_stay_exact_across_original(self):
        result=self.result(lambda d,p,r:d['events'][1]['tensors'][0].update(offset=0))
        self.assertIn('GWS_ARGUMENT_DESCRIPTORS_STABLE',result['issues'])

    def test_public_contiguous_substitution_not_joined(self):
        result=self.result(lambda d,p,r:d['public_scopes'][0]['evidence']['actual_input_addresses']['value'].update(stride=[1024,128,1],storage_offset=0))
        self.assertFalse(result['observed']['public_storage_geometry_all_joined'])
        self.assertFalse(result['prepared_internal_tensor_lineage_proven'])

    def test_cleanup_missing_or_wrong_owner_is_rejected(self):
        d,p,r=fixture();self.assertIn('RELEASE_ORIGINAL_REQUIRED',reader.review(d,p)['issues'])
        result=self.result(lambda d,p,r:r['fia'].update(owner_lo=99))
        self.assertIn('RELEASE_SAME_FIA_SESSION',result['issues'])

    def test_cleanup_arrays_not_zero_is_rejected(self):
        result=self.result(lambda d,p,r:r['containers']['arrays'].update(live_arrays=1))
        self.assertIn('RELEASE_SAME_CONTAINER_SESSION',result['issues'])

    def test_saved_pass_or_qualified_flag_is_not_oracle(self):
        def mutate(d,p,r):d.update(PASS=True,qualified=True);d['events'][1]['original_calls']=0
        result=self.result(mutate)
        self.assertIn('NO_INFERRED_AUTHORITY_OR_LINEAGE',result['issues'])
        self.assertIn('ORIGINAL_RETURN_PRESERVED',result['issues'])

    def test_independent_reader_imports_only_stdlib(self):
        import ast
        tree=ast.parse(Path(reader.__file__).read_text())
        names=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):names.extend(x.name.split('.')[0] for x in node.names)
            if isinstance(node,ast.ImportFrom):names.append(node.module.split('.')[0])
        self.assertLessEqual(set(names),{'argparse','hashlib','json','math','pathlib','re','struct'})

    def test_original_byte_pin_duplicate_keys_and_nan(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'raw.json'
            for raw in (b'{"a":1,"a":2}',b'{"a":NaN}'):
                path.write_bytes(raw)
                with self.assertRaises(ValueError):reader.read(path,hashlib.sha256(raw).hexdigest())
            path.write_text('{"value":null}')
            with self.assertRaises(ValueError):reader.read(path,'0'*64)
            self.assertIsNone(reader.read(path)[0]['value'])


class AdapterContracts(unittest.TestCase):
    def test_actual_cpu_noncontiguous_view_metadata_not_changed(self):
        import torch
        base=torch.arange(16384,dtype=torch.float16)
        view=base.as_strided((2,8,128),(4096,128,1),3072)
        result=native.public_tensor_metadata(view)
        self.assertEqual(result['shape'],[2,8,128]);self.assertEqual(result['stride'],[4096,128,1])
        self.assertEqual(result['storage_offset'],3072)
        self.assertEqual(result['storage_address'],base.untyped_storage().data_ptr())
        self.assertEqual(result['data_address'],view.data_ptr());self.assertEqual(result['object_id'],id(view))
        self.assertEqual(view.stride(),(4096,128,1));self.assertEqual(view.storage_offset(),3072)

    def adapter(self):
        x=native.NativeFiaInputJournal.__new__(native.NativeFiaInputJournal)
        class Containers:
            config=qct.instance(qct.Config)
            def query(self):return {'scope_token':1}
        x.containers=Containers();x.functions={};x.started=True;x.active=None;x.scopes=[]
        identity=dict(pid=101,native_tid=201,process_start_identity='proc:cpu-fixture:101:1')
        x.startup=dict(identity=identity);x.bound_worker=dict(worker_object_id=1001,model_object_id=1002)
        return x,identity

    def test_window_failure_does_not_clear_owned_state(self):
        x,identity=self.adapter();x.functions['scope_begin']=lambda *_:3
        with patch.object(native,'process_identity',return_value=identity),patch.object(native.os,'getpid',return_value=101),patch.object(native.threading,'get_native_id',return_value=201):
            with self.assertRaisesRegex(RuntimeError,'SCOPE_BEGIN_FAILED'):
                x.enter(call_id='a'*32,module_object_id=1003,forward=1,sequence=1,boundary=1,evidence={})
        self.assertEqual(x.active['status'],'ENTER_FAILED');self.assertEqual(len(x.scopes),1)

    def test_pending_leave_preserves_original_error_and_no_reentry(self):
        x,identity=self.adapter();x.functions['scope_begin']=lambda *_:0;x.functions['scope_end']=lambda *_:1
        with patch.object(native,'process_identity',return_value=identity),patch.object(native.os,'getpid',return_value=101),patch.object(native.threading,'get_native_id',return_value=201):
            x.enter(call_id='a'*32,module_object_id=1003,forward=1,sequence=1,boundary=1,evidence={})
            with self.assertRaisesRegex(RuntimeError,'DRAIN_OR_PENDING'):
                x.leave(original_calls=1,original_error='real original exception')
            with self.assertRaisesRegex(RuntimeError,'NO_OVERLAPPING'):
                x.enter(call_id='b'*32,module_object_id=1003,forward=1,sequence=2,boundary=2,evidence={})
        self.assertEqual(x.active['original_error'],'real original exception')

    def test_foreign_process_window_cannot_enter(self):
        x,identity=self.adapter()
        with patch.object(native,'process_identity',return_value={**identity,'pid':102}):
            with self.assertRaisesRegex(RuntimeError,'NO_OVERLAPPING_OR_FOREIGN'):
                x.enter(call_id='a'*32,module_object_id=1003,forward=1,sequence=1,boundary=1,evidence={})

    def test_exact_sdk_contract_and_bounded_native_cases_declared(self):
        self.assertEqual(len(native.FORMALS),44)
        self.assertEqual(tuple(native.TENSOR_INDICES),reader.TENSOR_INDICES)
        self.assertEqual(tuple(native.CONTAINER_INDICES),reader.CONTAINER_INDICES)
        self.assertEqual(builder.HEADERS['include/aclnnop/aclnn_fused_infer_attention_score_v3.h'],
            'bf73aea4146a9b12f91eda466927c23c5d43e56f78aa5039bd9644732f521550')
        self.assertEqual(len(builder.CASES),24)
        self.assertEqual(len(builder.CASES),len(set(builder.CASES)))
        self.assertTrue({'failed_prepare_untouched','foreign_thread','unknown_member','executor_reused','inactive'}<=set(builder.CASES))


if __name__=='__main__':unittest.main()
