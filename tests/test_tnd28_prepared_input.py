"""CPU source/byte fixtures; no fixture is an observed NPU call."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/c2c3_closure_20261117'))
import review_tnd28_prepared_input as R
from fault_injection.native_adapters.attention28.fia import tnd28_input_compile as C


def fixture():
    def tensor(name,shape,stride,base,dtype='torch.float16',offset=0,storage=None):
        width=1 if dtype=='torch.int8' else 4 if dtype=='torch.float32' else 2
        n=1
        for d in shape:n*=d
        public=dict(object_id=base+1,storage_identity=base+2,storage_address=base,data_address=base+offset*width,
            storage_bytes=(storage or n)*width,storage_offset=offset,element_bytes=width,shape=shape,stride=stride,
            dtype=dtype,device='npu:0')
        typed=dict(descriptor=base+3,storage_address=base,data_address=public['data_address'],offset=offset,
            native_storage_offset=0,dtype={'torch.float16':1,'torch.int8':2,'torch.float32':0}[dtype],format=2,
            shape=shape,stride=stride,storage_shape=shape)
        return public,typed
    tensors={};native={}
    for name,shape,stride,base,dtype,offset,storage in (
        ('query',[2048,16,128],[2048,128,1],0x1000000,'torch.float16',0,None),
        ('key',[2048,8,128],[1024,128,1],0x2000000,'torch.float16',0,None),
        ('value',[2048,8,128],[4096,128,1],0x3000000,'torch.float16',3072,2048*4096),
        ('atten_mask',[2048,2048],[2048,1],0x5000000,'torch.int8',0,None)):
        tensors[name],native[name]=tensor(name,shape,stride,base,dtype,offset,storage)
    out,tout=tensor('out',[2048,16,128],[2048,128,1],0x6000000)
    lse,tlse=tensor('lse',[0],[1],0,'torch.float32')
    identity=dict(pid=101,native_tid=202,python_thread_ident=303,process_start_ticks='444')
    scalar=dict(num_heads=16,num_key_value_heads=8,sparse_mode=3,block_size=128,scale=128**-.5,
        input_layout='TND',actual_seq_lengths=[2048],actual_seq_lengths_kv=[2048],block_table=None)
    api=dict(inputs={k:dict(kind='scalar',value=v) for k,v in scalar.items()},
        omitted_argument_defaults=dict(pre_tokens=2147483647,next_tokens=2147483647,inner_precise=0,softmax_lse_flag=False))
    for name,p in tensors.items():api['inputs'][name]=dict(kind='tensor',snapshot=dict(metadata=deepcopy(p)))
    public=dict(schema='demoncat.tnd28.public_input.v1',token='a'*32,owner='b'*32,generation=1,module_object_id=77,
        request_id='actual-request',identity=identity,tensors=tensors,api=api,worker_object_id=11,model_object_id=12,
        scheduler=dict(internal_request_id='actual-request',num_requests=1,num_scheduled_tokens=2048))
    public.update(request_sha256=R.E.digest(public['request_id']),input_sha256=R.E.digest(tensors))
    returned={k:deepcopy(public[k]) for k in ('token','identity','owner','generation','request_sha256','input_sha256')}
    returned.update(schema='demoncat.tnd28.public_return.v1',outputs=[out,lse],return_container_type='tuple',original_calls=1)
    fn=dict(address=0x8888,base=0x8000,path='/fixture/lib.so',mapped_path='/fixture/lib.so',mapped_begin=0x8000,
        mapped_end=0x9000,mapped_inode=5,file_inode=5,mapped_device=[1,2],file_device=[1,2],file_bytes=100,
        mapped_file_identity_matches=True)
    p=dict(function=fn,query=native['query'],key=dict(object=800,members=[native['key']]),
        value=dict(object=900,members=[native['value']]),mask=native['atten_mask'],seq_q=dict(object=90,values=[2048]),
        seq_kv=dict(object=91,values=[2048]),output=tout,lse=tlse,optionals=[0]*25,
        scalars=[16,2147483647,2147483647,8,3,0,128,0,0,0,0,0,0],scale=128**-.5,layout='TND',softmax_lse=False)
    data=[p,dict(function=fn,inputs=dict(object=999,members=[deepcopy(native['value'])])),
        dict(status=0,original_calls=1,executor=456,workspace_bytes=4198400,workspace_offsets=[4096]),
        dict(function=fn,executor=123),dict(status=0,original_calls=1,executor=123,fia_workspace_bytes=8192),
        dict(status=0,original_returns=1,executor=123,total_workspace_bytes=4206592),
        dict(workspace_called=True,original_returned=True),dict(function=fn,executor=123),
        dict(function=fn,executor=456,workspace=0x8000000,workspace_bytes=4198400,stream=321,
            workspace_offsets=[4096],prepared_v_address=0x8001000),dict(status=0,original_calls=1),
        dict(function=fn,executor=123,workspace=0x8000000-8192,workspace_bytes=8192,stream=321),
        dict(status=0,original_calls=1),dict(status=0,original_returns=1),dict(fia_called=True,original_returned=True)]
    events=[]
    for i,(phase,d) in enumerate(zip(R.PHASES,data)):
        e={k:public[k] for k in ('token','owner','generation','module_object_id','request_sha256','input_sha256')}
        e.update(schema='demoncat.tnd28.compiled_input.v1',phase=phase,pid=101,tid=202,process_start_ticks='444',
            monotonic_ns=(i+1)*1000,first_error='',fault_authorized=False,data=deepcopy(d));events.append(e)
    return public,returned,events


def launch_fixture(data):
    names=['query','key','value','pse_shift','attenMask','actualSeqLengths','actualSeqLengthsKV','attentionOut','workspace','tiling']
    abi=dict(names=names,pointer_bytes=8,hidden_hardware_sync_prefix_bytes=8)
    raw=bytearray(8+len(names)*8);slots={n:8+i*8 for i,n in enumerate(names)};pairs=[]
    p=data['prepare_enter']
    for name,value in dict(query=p['query']['data_address'],attenMask=p['mask']['data_address'],attentionOut=p['output']['data_address']).items():
        struct.pack_into('<Q',raw,slots[name],value)
    for name,address in (('key',p['key']['members'][0]['data_address']),('value',data['contiguous_execute_enter']['prepared_v_address'])):
        pairs.append((slots[name],len(raw)));raw+=struct.pack('<QIIqqqQ',40,3,1,2048,8,128,address)
    for name in ('actualSeqLengths','actualSeqLengthsKV'):
        pairs.append((slots[name],len(raw)));raw+=struct.pack('<q',2048)
    return dict(argument_layout='aclrt_host_placeholders',args_hex=raw.hex(),args_size=len(raw),
        placeholders_hex=b''.join(struct.pack('<II',*p) for p in pairs).hex()),abi


class TypedReadbackContracts(unittest.TestCase):
    def test_complete_actual_shape_fixture_and_prepared_address_join(self):
        p,r,e=fixture();d=R.protocol(p,r,e);raw,abi=launch_fixture(d)
        result=R.launch_inputs(raw,abi,d)
        self.assertEqual(result['prepared_value_address'],0x8001000);self.assertFalse(result['slice_pod_decoded'])

    def test_same_value_cloned_descriptor_cannot_replace_original_v(self):
        p,r,e=fixture();e[1]['data']['inputs']['members'][0]['descriptor']+=1
        with self.assertRaisesRegex(ValueError,'ORIGINAL_V_DESCRIPTOR'):R.protocol(p,r,e)

    def test_current_owner_request_native_thread_and_process_lifetime(self):
        for key,value in (('owner','f'*32),('generation',2),('request_sha256','f'*64),('input_sha256','f'*64),
                ('tid',303),('process_start_ticks','445')):
            p,r,e=fixture();e[9][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):R.protocol(p,r,e)

    def test_no_journal_missing_return_reorder_and_failed_status_stay_unobserved(self):
        for kind in ('empty','missing','reorder','failed','twice','observer_error'):
            p,r,e=fixture()
            if kind=='empty':e=[]
            if kind=='missing':e.pop(5)
            if kind=='reorder':e[8],e[9]=e[9],e[8]
            if kind=='failed':e[9]['data']['status']=8
            if kind=='twice':e[9]['data']['original_calls']=2
            if kind=='observer_error':e[10]['first_error']='offset changed'
            with self.subTest(kind=kind),self.assertRaises(ValueError):R.protocol(p,r,e)

    def test_wrong_live_offset_out_of_span_stream_or_actual_return_reject(self):
        for change in (lambda e,r:e[8]['data'].__setitem__('workspace_offsets',[8192]),
                lambda e,r:e[8]['data'].__setitem__('workspace_bytes',1024),
                lambda e,r:e[10]['data'].__setitem__('stream',322),
                lambda e,r:r['outputs'][0].__setitem__('data_address',1)):
            p,r,e=fixture();change(e,r)
            with self.assertRaises(ValueError):R.protocol(p,r,e)

    def test_native_value_address_from_public_v_is_not_prepared_output(self):
        p,r,e=fixture();d=R.protocol(p,r,e);raw,abi=launch_fixture(d)
        raw['args_hex']=raw['args_hex'].replace(struct.pack('<Q',0x8001000).hex(),struct.pack('<Q',p['tensors']['value']['data_address']).hex())
        with self.assertRaisesRegex(ValueError,'DESCRIPTOR_OR_ADDRESS'):R.launch_inputs(raw,abi,d)

    def test_extra_optional_scalar_and_mapped_library_identity_refuse(self):
        for edit in (lambda e:e[0]['data']['optionals'].__setitem__(3,99),
                lambda e:e[0]['data']['scalars'].__setitem__(2,0),
                lambda e:e[0]['data']['function'].__setitem__('file_inode',6)):
            p,r,e=fixture();edit(e)
            with self.assertRaises(ValueError):R.protocol(p,r,e)

    def test_verified_inode_alias_is_recorded_without_comparing_path_spelling(self):
        p,r,e=fixture();e[0]['data']['function']['mapped_path']='/real/fixture/lib.so'
        R.protocol(p,r,e)

    def test_host_descriptor_shape_offset_or_duplicate_slot_refuse(self):
        p,r,e=fixture();d=R.protocol(p,r,e)
        for kind in ('shape','offset','duplicate','length'):
            raw,abi=launch_fixture(d);b=bytearray.fromhex(raw['args_hex']);pairs=list(struct.iter_unpack('<II',bytes.fromhex(raw['placeholders_hex'])))
            if kind=='shape':struct.pack_into('<q',b,pairs[0][1]+16,2047)
            if kind=='offset':struct.pack_into('<Q',b,pairs[0][1],48)
            if kind=='length':struct.pack_into('<q',b,pairs[-1][1],1)
            if kind=='duplicate':raw['placeholders_hex']+=struct.pack('<II',*pairs[0]).hex()
            raw['args_hex']=b.hex()
            with self.subTest(kind=kind),self.assertRaises(ValueError):R.launch_inputs(raw,abi,d)

    def test_cold_reader_does_not_import_framework_or_ctypes(self):
        script=ROOT/'experiments/c2c3_closure_20261117/review_tnd28_prepared_input.py'
        code='import runpy,sys;runpy.run_path(sys.argv[1]);assert not any(n in sys.modules for n in ("torch","torch_npu","ctypes","vllm"))'
        result=subprocess.run([sys.executable,'-B','-c',code,str(script)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


class CompilerHookContracts(unittest.TestCase):
    def setUp(self):
        self.root=Path(self.enterContext(tempfile.TemporaryDirectory()));self.source=self.root/'source';self.unit=self.source/'build/autogen/inner'/C.GENERATED
        self.unit.parent.mkdir(parents=True);self.ev=self.root/'evidence';self.ev.mkdir()
        source=('// CPU synthetic compiler fixture\n#include "aclnnInner_fused_infer_attention_score.h"\n'
            'aclnnStatus aclnnInnerFusedInferAttentionScoreGetWorkspaceSize()\n{\n    uint64_t timeStamp = 0;\n'
            'aclnnContiguousGetWorkspaceSize(inUnContTensors, &inContWorkspaceSize, &aclInExecutor);\n'
            'NnopbaseRunForWorkspace(*executor, workspaceSize);\n    return ACLNN_SUCCESS;\n}\n'
            'aclnnStatus aclnnInnerFusedInferAttentionScoreTensorGetWorkspaceSize() { CPU_TENSOR_BODY_UNCHANGED; }\n'
            'aclnnStatus aclnnInnerFusedInferAttentionScore(\n)\n{\n    uint64_t timeStamp = 0;\n'
            'aclnnContiguous(inWorkspace, inContWorkspaceSize, aclInExecutor, stream);\n'
            'NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceSize);\n    return ACLNN_SUCCESS;\n}\n').encode()
        self.unit.write_bytes(source);self.unit.with_suffix('.h').write_bytes(b'CPU fake header')
        self.enterContext(patch.object(C,'ORIGINAL_SHA',C.sha(source)))
        self.enterContext(patch.object(C,'GENERATED_HEADER_SHA',C.sha(b'CPU fake header')))
        self.argv=['c++','-c',str(self.unit),'-o','object.o']

    def test_only_exact_generated_unit_patches_and_reentry_rechecks_bytes(self):
        result=C.prepare_unit(self.source,self.ev,self.argv,variant=C.VARIANT)
        self.assertEqual(C.prepare_unit(self.source,self.ev,self.argv,variant=C.VARIANT),result)
        data=self.unit.read_text();self.assertIn('CPU_TENSOR_BODY_UNCHANGED',data);self.assertEqual(data.count('dcat_input.'),6)
        self.assertFalse(result['fault_authorized'] or result['current_input_authority'])
        self.unit.write_text(data+'\n')
        with self.assertRaisesRegex(ValueError,'REENTRY_BYTES_CHANGED'):C.prepare_unit(self.source,self.ev,self.argv,variant=C.VARIANT)

    def test_unknown_source_wrong_variant_and_fixture_compile_cannot_build(self):
        for variant,argv in (('tnd_prefill_normal_bridge',self.argv),(C.VARIANT,self.argv+['-DDCAT_TND28_INPUT_CPU_FIXTURE'])):
            with self.assertRaises(ValueError):C.prepare_unit(self.source,self.ev,argv,variant=variant)
        self.unit.write_bytes(self.unit.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'UNREVIEWED_GENERATED'):C.prepare_unit(self.source,self.ev,self.argv,variant=C.VARIANT)
        self.assertFalse((self.ev/'tnd28_compiled_input_observer.json').exists())

    def test_other_compilation_units_pass_through_without_any_patch(self):
        before=self.unit.read_bytes()
        self.assertIsNone(C.prepare_unit(self.source,self.ev,['c++','-c','other.cpp'],variant=C.VARIANT))
        self.assertEqual(before,self.unit.read_bytes());self.assertEqual(list(self.ev.iterdir()),[])


class OneWindowEntrypoints(unittest.TestCase):
    def test_bounded_case_has_one_observe_request_and_no_policy_or_replay_switch(self):
        from experiments.worker_core_v11_20261118 import prepare_tnd28_prepared_input as P
        args=NS(card=5,timeout=900,source=PurePosixPath('/source'),installed=PurePosixPath('/installed'),model=PurePosixPath('/model'),
            build_root=PurePosixPath('/buildrun'),vendor=PurePosixPath('/vendor'),build_sha256='a'*64,source_manifest_sha256='b'*64,libraries_sha256='c'*64)
        source=dict(manifest=dict(base_commit='d'*40),manifest_sha256='b'*64,pins={})
        observer=dict(folder=PurePosixPath('/observer'),alias='/observer',sha256='e'*64)
        build=dict(variant=R.VARIANT,kernel_binary_sha256='f'*64,tnd28_compiled_input_observer={'observer_only':True})
        cases=P.make_case(args,source,observer,build,[],{})
        self.assertEqual(len(cases),1);case=cases[0]
        self.assertEqual(case['review']['resource_budget']['observed_requests'],1)
        self.assertEqual(case['review']['resource_budget']['ordinary_health_requests'],1)
        self.assertEqual(case['review']['resource_budget']['cpus'],8)
        self.assertNotIn('DCAT28_TND_MODE',case['command']);self.assertFalse(case['review']['fault_authorized'])
        args.timeout=1801
        with self.assertRaises(ValueError):P.make_case(args,source,observer,build,[],{})

    def test_uncertain_prepare_without_ticket_only_uses_public_detach(self):
        from experiments.worker_core_v11_20261118 import natural_tnd28_prepared_input_probe as P
        class Client:
            def __init__(self):self.actions=[];self.owned=True;self.history=[]
            def prepare(self):self.actions.append('prepare');return dict(status='RPC_UNCERTAIN',history=[])
            def detach(self):
                self.actions.append('detach');self.owned=False
                self.history.append(dict(action='detach',response={'status':'DETACHED'}))
                return dict(status='DETACHED',history=self.history)
            def snapshot(self):return dict(ownership_retained=self.owned,ticket=None,attachment=None,worker={},history=self.history)
        client=Client();report={}
        with tempfile.TemporaryDirectory() as folder:
            args=NS(out=Path(folder));P.drive(NS(),client,{},None,args,report,{})
            self.assertEqual(client.actions,['prepare','detach']);self.assertFalse(report['ownership_retained'])
            self.assertEqual(report['status'],'REFUSED_OR_FAILED');self.assertIn('RPC_UNCERTAIN',report['error'])

    def test_cold_probe_import_does_not_initialize_a_model_or_load_an_operator(self):
        script=ROOT/'experiments/worker_core_v11_20261118/natural_tnd28_prepared_input_probe.py'
        code='import runpy,sys;runpy.run_path(sys.argv[1]);assert not any(n in sys.modules for n in ("torch","torch_npu","vllm"))'
        result=subprocess.run([sys.executable,'-B','-c',code,str(script)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
