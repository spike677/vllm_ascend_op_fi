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
            native_storage_offset=0,dtype={'torch.float16':1,'torch.int8':2,'torch.float32':0}[dtype],
            format=47 if name in ('query','key','value','out') else 2,
            shape=shape,stride=stride,storage_shape=[storage or n])
        return public,typed
    tensors={};native={}
    for name,shape,stride,base,dtype,offset,storage in (
        ('query',[2048,16,128],[2048,128,1],0x1000000,'torch.float16',0,None),
        ('key',[2048,8,128],[1024,128,1],0x2000000,'torch.float16',0,None),
        ('value',[2048,8,128],[4096,128,1],0x3000000,'torch.float16',3072,2048*4096),
        ('atten_mask',[2048,2048],[2048,1],0x5000000,'torch.int8',0,None)):
        tensors[name],native[name]=tensor(name,shape,stride,base,dtype,offset,storage)
    out,tout=tensor('out',[2048,16,128],[2048,128,1],0x6000000)
    lse,_=tensor('lse',[0],[1],0,'torch.float32')
    # V3 false-LSE uses a temporary host placeholder for inner GWS. Python
    # returns a separate empty NPU tensor; neither descriptor nor stride joins.
    _,tlse=tensor('temporary_lse',[0],[0],0x7000000,'torch.float32')
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
    names=['query','key','value','pse_shift','attenMask','actualSeqLengths','actualSeqLengthsKV','attentionOut','softmaxLse','workspace','tiling']
    abi=dict(names=names,pointer_bytes=8,hidden_hardware_sync_prefix_bytes=8)
    raw=bytearray(8+len(names)*8);slots={n:8+i*8 for i,n in enumerate(names)};pairs=[]
    p=data['prepare_enter']
    for name,value in dict(query=p['query']['data_address'],attenMask=p['mask']['data_address'],
            attentionOut=p['output']['data_address'],softmaxLse=p['lse']['data_address']).items():
        struct.pack_into('<Q',raw,slots[name],value)
    for name,address in (('key',p['key']['members'][0]['data_address']),('value',data['contiguous_execute_enter']['prepared_v_address'])):
        shape=p['key']['members'][0]['storage_shape'] if name=='key' else p['value']['members'][0]['shape']
        pairs.append((slots[name],len(raw)))
        raw+=struct.pack('<QII',16+len(shape)*8,len(shape),1)+struct.pack('<'+'q'*len(shape),*shape)+struct.pack('<Q',address)
    for name in ('actualSeqLengths','actualSeqLengthsKV'):
        pairs.append((slots[name],len(raw)));raw+=struct.pack('<q',2048)
    return dict(argument_layout='aclrt_host_placeholders',args_hex=raw.hex(),args_size=len(raw),
        placeholders_hex=b''.join(struct.pack('<II',*p) for p in pairs).hex()),abi


class TypedReadbackContracts(unittest.TestCase):
    def test_sdk_ncl_is_limited_to_actual_rank3_roles_and_nd_to_mask(self):
        p,r,e=fixture();native=e[0]['data'];R.protocol(p,r,e)
        for role,typed in (('query',native['query']),('key',native['key']['members'][0]),
                ('value',native['value']['members'][0]),('output',native['output'])):
            self.assertEqual(typed['format'],47,role);self.assertEqual(len(typed['shape']),3)
            self.assertEqual(len(typed['storage_shape']),1)
        self.assertEqual(native['mask']['format'],2);self.assertEqual(native['lse']['format'],2)

    def test_wrong_role_or_unknown_storage_format_cannot_be_widened(self):
        for role,value in (('query',2),('key',2),('value',2),('output',2),('mask',47),('query',48)):
            p,r,e=fixture();typed=e[0]['data'][role]
            if role in ('key','value'):typed=typed['members'][0]
            typed['format']=value
            with self.subTest(role=role,value=value),self.assertRaisesRegex(ValueError,'STORAGE_FORMAT_FOR_ROLE'):
                R.protocol(p,r,e)

    def test_native_and_public_storage_extents_still_must_match_actual_view(self):
        for kind in ('native_extent','public_extent','address','stride','offset'):
            p,r,e=fixture();typed=e[0]['data']['value']['members'][0]
            if kind=='native_extent':typed['storage_shape'][0]-=1
            if kind=='public_extent':p['tensors']['value']['storage_bytes']-=2
            if kind=='address':typed['data_address']+=2
            if kind=='stride':typed['stride'][0]-=1
            if kind=='offset':typed['offset']-=1
            p['input_sha256']=R.E.digest(p['tensors']);r['input_sha256']=p['input_sha256']
            for row in e:row['input_sha256']=p['input_sha256']
            with self.subTest(kind=kind),self.assertRaises(ValueError):R.protocol(p,r,e)

    def test_complete_actual_shape_fixture_and_prepared_address_join(self):
        p,r,e=fixture();d=R.protocol(p,r,e);raw,abi=launch_fixture(d)
        result=R.launch_inputs(raw,abi,d)
        self.assertEqual(result['prepared_value_address'],0x8001000);self.assertFalse(result['slice_pod_decoded'])

    def test_false_lse_temporary_placeholder_is_separate_from_public_empty_output(self):
        p,r,e=fixture();typed=e[0]['data']['lse'];public=r['outputs'][1]
        self.assertNotEqual(typed['data_address'],public['data_address'])
        self.assertNotEqual(typed['stride'],public['stride'])
        d=R.protocol(p,r,e);raw,abi=launch_fixture(d);result=R.launch_inputs(raw,abi,d)
        self.assertEqual(result['lse']['placeholder_address'],typed['data_address'])
        self.assertFalse(result['lse']['descriptor_live_after_gws_claimed'])
        b=bytearray.fromhex(raw['args_hex']);slot=8+abi['names'].index('softmaxLse')*8
        struct.pack_into('<Q',b,slot,public['data_address']);raw['args_hex']=b.hex()
        with self.assertRaisesRegex(ValueError,'ACTUAL_LSE_OUTPUT_POINTER'):R.launch_inputs(raw,abi,d)

    def test_false_lse_invalid_temporary_geometry_or_public_nonempty_return_refuse(self):
        for key,value in (('shape',[1]),('storage_shape',[1]),('stride',[1]),('offset',1),
                ('native_storage_offset',1),('dtype',1),('format',3),('descriptor',0),
                ('data_address',0),('storage_address',0)):
            p,r,e=fixture();e[0]['data']['lse'][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'FALSE_LSE_TYPED_PLACEHOLDER'):
                R.protocol(p,r,e)
        for key,value in (('shape',[1]),('dtype','torch.float16'),('storage_bytes',4),('data_address',0x7000000),
                ('stride',[0]),('device','cpu')):
            p,r,e=fixture();r['outputs'][1][key]=value
            with self.subTest(public_key=key),self.assertRaisesRegex(ValueError,'FALSE_LSE_PUBLIC_EMPTY_RETURN'):
                R.protocol(p,r,e)

    def test_false_lse_rule_requires_exact_retained_build_source(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);completion=run/'evidence/completion.json';completion.parent.mkdir()
            data=b'CPU source identity fixture; not SDK evidence\n';name=next(iter(R.LSE_SOURCES))
            path=run/'source'/name;path.parent.mkdir(parents=True);path.write_bytes(data)
            pins={name:hashlib.sha256(data).hexdigest()};build={'source_commit':R.LSE_SOURCE_COMMIT}
            with patch.object(R,'LSE_SOURCES',pins):
                self.assertEqual(R.false_lse_sources(completion,build),pins)
                path.write_bytes(data+b'changed')
                with self.assertRaisesRegex(ValueError,'ACTUAL_BUILD_SOURCE_CHANGED'):
                    R.false_lse_sources(completion,build)
                with self.assertRaisesRegex(ValueError,'SOURCE_REVISION_CHANGED'):
                    R.false_lse_sources(completion,{'source_commit':'0'*40})

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


class VendorLinkContracts(unittest.TestCase):
    def test_exact_packaged_alias_and_relative_target_are_accepted(self):
        R.vendor_link_rule('op_impl/ai_core/tbe/op_tiling/liboptiling.so',
            'lib/linux/aarch64/libcust_opmaster_rt2.0.so')

    def test_other_aliases_including_old_elf_basename_are_rejected(self):
        target='lib/linux/aarch64/libcust_opmaster_rt2.0.so'
        for name in ('op_impl/ai_core/tbe/op_tiling/libcust_opmaster_rt2.0.so',
                'foreign/op_tiling/liboptiling.so',
                'op_impl/ai_core/tbe/op_tiling/../op_tiling/liboptiling.so'):
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'UNREVIEWED_VENDOR_LINK'):
                R.vendor_link_rule(name,target)

    def test_wrong_absolute_or_traversing_targets_are_rejected(self):
        name='op_impl/ai_core/tbe/op_tiling/liboptiling.so'
        for target in ('lib/linux/aarch64/other.so',
                '/lib/linux/aarch64/libcust_opmaster_rt2.0.so',
                '../op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so',
                'lib/linux/aarch64/./libcust_opmaster_rt2.0.so'):
            with self.subTest(target=target),self.assertRaisesRegex(ValueError,'UNREVIEWED_VENDOR_LINK'):
                R.vendor_link_rule(name,target)


class RawRpcBudgetContracts(unittest.TestCase):
    def setUp(self):
        self.root=Path(self.enterContext(tempfile.TemporaryDirectory()));self.refs=[]
        for i,action in enumerate(R.RPC_ACTIONS):
            path=self.root/('%03d.json'%i);raw=json.dumps(dict(action=action,response={})).encode()
            path.write_bytes(raw);self.refs.append(dict(path=path.name,action=action,sha256=R.E.sha(path)))

    def test_original_metadata_above_old_16mib_limit_is_read_without_rewriting(self):
        path=self.root/self.refs[7]['path']
        raw=json.dumps(dict(action='advance',response={},cpu_only_padding='x'*(16*1024**2))).encode()
        path.write_bytes(raw);self.refs[7]['sha256']=hashlib.sha256(raw).hexdigest()
        records,budget=R.original_rpc_records(self.root,self.refs)
        self.assertEqual([r['action'] for r in records],R.RPC_ACTIONS)
        self.assertGreater(budget['record_bytes'][7],16*1024**2)
        self.assertEqual(budget['per_record_limit_bytes'],32*1024**2)
        self.assertEqual(budget['total_limit_bytes'],128*1024**2)
        self.assertEqual(budget['total_bytes'],sum((self.root/r['path']).stat().st_size for r in self.refs))
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),self.refs[7]['sha256'])

    def test_over_item_budget_refuses_before_any_json_read(self):
        path=self.root/self.refs[-1]['path']
        with path.open('wb') as stream:stream.truncate(R.RPC_ITEM_LIMIT+1)
        with patch.object(R,'raw_file') as read:
            with self.assertRaisesRegex(ValueError,'RPC_ITEM_BYTE_BOUND'):R.original_rpc_records(self.root,self.refs)
        read.assert_not_called()

    def test_over_total_budget_refuses_before_any_json_read(self):
        total=sum((self.root/r['path']).stat().st_size for r in self.refs)
        with patch.object(R,'RPC_TOTAL_LIMIT',total-1),patch.object(R,'raw_file') as read:
            with self.assertRaisesRegex(ValueError,'RPC_TOTAL_BYTE_BOUND'):R.original_rpc_records(self.root,self.refs)
        read.assert_not_called()

    def test_extra_missing_and_reordered_rpc_are_not_the_bounded_protocol(self):
        variants=(self.refs+[self.refs[-1]],self.refs[:-1],[self.refs[1],self.refs[0],*self.refs[2:]])
        for refs in variants:
            with self.subTest(count=len(refs)),patch.object(R,'raw_file') as read:
                with self.assertRaisesRegex(ValueError,'RPC_COUNT_OR_ORDER'):R.original_rpc_records(self.root,refs)
                read.assert_not_called()

    def test_byte_change_and_reused_artifact_are_not_hidden_by_larger_limit(self):
        path=self.root/self.refs[-1]['path'];path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'ARTIFACT_CHANGED'):R.original_rpc_records(self.root,self.refs)
        refs=deepcopy(self.refs);refs[7]['path']=refs[5]['path']
        with self.assertRaisesRegex(ValueError,'RPC_PATH_REUSED'):R.original_rpc_records(self.root,refs)


class RawRpcSummaryContracts(unittest.TestCase):
    def fixture(self):
        public={'owner':'b'*32,'generation':1}
        request={'request_id':'actual-request','phase':'observe','forwards':[{'calls':[{'original_calls':1}]}]}
        stopped=dict(public,owned_hooks=0,owned_api_wrappers=0)
        records=[dict(action=action,response={}) for action in R.RPC_ACTIONS]
        records[7]['response']['provider_state']=dict(public,evidence={'requests':[deepcopy(request)]})
        records[8]['response']['provider_state']=deepcopy(stopped)
        result={'request':{'worker':request},'cleanup':stopped}
        return result,public,records

    def test_exact_advance_collect_and_disable_raw_responses_bind_summaries(self):
        result,public,records=self.fixture();R.public_rpc_summaries(result,public,records)
        self.assertEqual(records[7]['action'],'advance')

    def test_replaced_request_or_cleanup_summary_and_cross_owner_refuse(self):
        for kind in ('request','cleanup','owner','collect_name','extra_request'):
            result,public,records=self.fixture()
            if kind=='request':result['request']['worker']['forwards'][0]['calls'][0]['original_calls']=2
            if kind=='cleanup':result['cleanup']['owned_hooks']=1
            if kind=='owner':records[7]['response']['provider_state']['owner']='c'*32
            if kind=='collect_name':records[7]['action']='collect_replay'
            if kind=='extra_request':records[7]['response']['provider_state']['evidence']['requests'].append({})
            with self.subTest(kind=kind),self.assertRaises(ValueError):R.public_rpc_summaries(result,public,records)


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
        self.sdk=self.root/'cann-8.5.1';pins={}
        for name in C.SDK_HEADERS:
            path=self.sdk/name;path.parent.mkdir(parents=True,exist_ok=True)
            data=('CPU synthetic SDK header: '+name).encode();path.write_bytes(data);pins[name]=C.sha(data)
        self.enterContext(patch.object(C,'SDK_HEADERS',pins))
        self.transitive=[]
        for name in ('include/aclnn/opdev/fast_vector.h','include/external/exe_graph/runtime/shape.h','pkg_inc/base/dlog_pub.h'):
            path=self.sdk/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'CPU transitive header')
            self.transitive.append(path)
        self.obj=self.source/'build/observer.o';self.dep=self.source/'build/observer.o.d'
        self.argv=['c++','-MD','-MF',str(self.dep),'-MT',str(self.obj),'-c',str(self.unit),'-o',str(self.obj)]

    def compile_result(self,argv,check=False,*,change=None,status=0):
        self.assertFalse(check)
        self.assertEqual(argv,[self.argv[0],'-I',str(self.sdk.resolve()/'include/aclnn'),'-I',str(self.sdk.resolve()/'pkg_inc'),*self.argv[1:]])
        self.obj.write_bytes(b'CPU object fixture; never a compiled SDK operator')
        paths=[self.unit,self.unit.with_name(C.HEADER),*(self.sdk/name for name in C.SDK_HEADERS),*self.transitive]
        if change is not None:paths=change(paths)
        escape=lambda p:p.as_posix().replace(' ','\\ ')
        self.dep.write_text(escape(Path(C.option(self.argv,'-MT')))+': '+ ' \\\n '.join(escape(p) for p in paths)+'\n')
        return NS(returncode=status)

    def test_only_exact_generated_unit_patches_and_reentry_rechecks_bytes(self):
        result=C.prepare_unit(self.source,self.ev,self.argv,variant=C.VARIANT,sdk=self.sdk)
        self.assertEqual(C.prepare_unit(self.source,self.ev,self.argv,variant=C.VARIANT,sdk=self.sdk),result)
        data=self.unit.read_text();self.assertIn('CPU_TENSOR_BODY_UNCHANGED',data);self.assertEqual(data.count('dcat_input.'),6)
        self.assertFalse(result['fault_authorized'] or result['current_input_authority'])
        self.unit.write_text(data+'\n')
        with self.assertRaisesRegex(ValueError,'REENTRY_BYTES_CHANGED'):C.prepare_unit(self.source,self.ev,self.argv,variant=C.VARIANT,sdk=self.sdk)

    def test_unknown_source_wrong_variant_and_fixture_compile_cannot_build(self):
        for variant,argv in (('tnd_prefill_normal_bridge',self.argv),(C.VARIANT,self.argv+['-DDCAT_TND28_INPUT_CPU_FIXTURE'])):
            with self.assertRaises(ValueError):C.prepare_unit(self.source,self.ev,argv,variant=variant,sdk=self.sdk)
        self.unit.write_bytes(self.unit.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'UNREVIEWED_GENERATED'):C.prepare_unit(self.source,self.ev,self.argv,variant=C.VARIANT,sdk=self.sdk)
        self.assertFalse((self.ev/'tnd28_compiled_input_observer.json').exists())

    def test_other_compilation_units_pass_through_without_any_patch(self):
        before=self.unit.read_bytes()
        argv=['c++','-c','other.cpp']
        self.assertIsNone(C.prepare_unit(self.source,self.ev,argv,variant=C.VARIANT,sdk=self.sdk))
        with patch.object(C.subprocess,'run',return_value=NS(returncode=7)) as compiler:
            self.assertEqual(C.run_compiler(self.source,self.ev,argv,variant=C.VARIANT,sdk=self.sdk),7)
        compiler.assert_called_once_with(argv,check=False)
        self.assertEqual(before,self.unit.read_bytes());self.assertEqual(list(self.ev.iterdir()),[])

    def test_exact_sdk_compile_once_and_actual_dependency_closure_revalidated(self):
        with patch.object(C.subprocess,'run',side_effect=self.compile_result) as compiler:
            self.assertEqual(C.run_compiler(self.source,self.ev,self.argv,variant=C.VARIANT,sdk=self.sdk),0)
        compiler.assert_called_once()
        result=C.verify_compile_receipt(self.ev,self.source,self.sdk)
        self.assertEqual(result['compiler_calls'],1);self.assertEqual(result['dependency_count'],8)
        self.assertEqual(result['sdk_headers'],C.sdk_headers(self.sdk))
        self.transitive[0].write_bytes(b'changed container ABI after compilation')
        with self.assertRaisesRegex(ValueError,'DEPENDENCY_CLOSURE_CHANGED'):
            C.verify_compile_receipt(self.ev,self.source,self.sdk)

    def test_original_relative_output_and_dep_targets_use_recorded_compiler_cwd(self):
        for flag,value in (('-MF','observer.o.d'),('-o','observer.o'),('-MT','observer.o')):
            self.argv[self.argv.index(flag)+1]=value
        with patch.object(C.Path,'cwd',return_value=(self.source/'build').resolve()),\
                patch.object(C.subprocess,'run',side_effect=self.compile_result):
            self.assertEqual(C.run_compiler(self.source,self.ev,self.argv,variant=C.VARIANT,sdk=self.sdk),0)
        self.assertEqual(C.verify_compile_receipt(self.ev,self.source,self.sdk)['object']['path'],str(self.obj.resolve()))

    def test_compiler_that_changes_the_observed_source_is_not_confirmed(self):
        def changed(paths):self.unit.write_bytes(self.unit.read_bytes()+b'\n');return paths
        with patch.object(C.subprocess,'run',side_effect=lambda argv,check:self.compile_result(argv,check,change=changed)):
            with self.assertRaisesRegex(ValueError,'OBSERVER_BYTES_CHANGED_DURING_COMPILE'):
                C.run_compiler(self.source,self.ev,self.argv,variant=C.VARIANT,sdk=self.sdk)
        self.assertEqual(json.loads((self.ev/C.COMPILE_RECEIPT).read_bytes())['status'],'FAILED_OR_UNCONFIRMED')

    def test_wrong_pinned_sdk_or_missing_original_dependency_flags_refuse_before_compile(self):
        argv=[arg for arg in self.argv if arg!='-MD']
        with patch.object(C.subprocess,'run') as compiler:
            with self.assertRaisesRegex(ValueError,'ORIGINAL_MD'):
                C.run_compiler(self.source,self.ev,argv,variant=C.VARIANT,sdk=self.sdk)
            (self.sdk/next(iter(C.SDK_HEADERS))).write_bytes(b'wrong SDK header')
            with self.assertRaisesRegex(ValueError,'SDK_HEADER_BYTES_CHANGED'):
                C.run_compiler(self.source,self.ev,self.argv,variant=C.VARIANT,sdk=self.sdk)
        compiler.assert_not_called();self.assertFalse(self.unit.with_name(C.HEADER).exists())

    def test_dependency_opbase_header_even_identical_bytes_cannot_replace_sdk(self):
        other=self.root/'dependencies/opbase/include/nnopbase/opdev/common_types.h'
        other.parent.mkdir(parents=True);other.write_bytes((self.sdk/'include/aclnn/opdev/common_types.h').read_bytes())
        def changed(paths):return [other if p.name=='common_types.h' else p for p in paths]
        with patch.object(C.subprocess,'run',side_effect=lambda argv,check:self.compile_result(argv,check,change=changed)) as compiler:
            with self.assertRaisesRegex(ValueError,'OPBASE_INTERNAL_HEADER_FORBIDDEN'):
                C.run_compiler(self.source,self.ev,self.argv,variant=C.VARIANT,sdk=self.sdk)
        compiler.assert_called_once();report=json.loads((self.ev/C.COMPILE_RECEIPT).read_bytes())
        self.assertEqual(report['status'],'FAILED_OR_UNCONFIRMED');self.assertTrue(self.obj.exists() and self.dep.exists())
        with self.assertRaisesRegex(ValueError,'RECEIPT_UNCONFIRMED'):C.verify_compile_receipt(self.ev,self.source,self.sdk)

    def test_missing_real_shape_dependency_is_not_a_successful_compile_receipt(self):
        def changed(paths):return [p for p in paths if p.name!='shape.h']
        with patch.object(C.subprocess,'run',side_effect=lambda argv,check:self.compile_result(argv,check,change=changed)):
            with self.assertRaisesRegex(ValueError,'SDK_TRANSITIVE_HEADER_REQUIRED'):
                C.run_compiler(self.source,self.ev,self.argv,variant=C.VARIANT,sdk=self.sdk)
        report=json.loads((self.ev/C.COMPILE_RECEIPT).read_bytes())
        self.assertEqual(report['compiler_returncode'],0);self.assertIn('TRANSITIVE',report['first_error'])

    def test_original_compiler_failure_is_recorded_and_never_retried(self):
        with patch.object(C.subprocess,'run',side_effect=lambda argv,check:self.compile_result(argv,check,status=7)) as compiler:
            with self.assertRaisesRegex(ValueError,'ACTUAL_ORIGINAL_COMPILER_FAILED'):
                C.run_compiler(self.source,self.ev,self.argv,variant=C.VARIANT,sdk=self.sdk)
        compiler.assert_called_once();path=self.ev/C.COMPILE_RECEIPT;original=path.read_bytes()
        self.assertEqual(json.loads(original)['compiler_returncode'],7)
        with patch.object(C.subprocess,'run') as compiler:
            with self.assertRaises(FileExistsError):C.run_compiler(self.source,self.ev,self.argv,variant=C.VARIANT,sdk=self.sdk)
        compiler.assert_not_called();self.assertEqual(path.read_bytes(),original)


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

    def test_real_drive_final_host_write_failure_does_not_replace_primary_prepare_error(self):
        from experiments.worker_core_v11_20261118 import natural_tnd28_prepared_input_probe as P
        from demoncat.runtime.vllm_ascend.worker_extension import paged33_loading,serving_context
        class Client:
            def __init__(self):self.actions=[];self.owned=True;self.history=[]
            def prepare(self):self.actions.append('prepare');return dict(status='RPC_UNCERTAIN',history=[])
            def detach(self):
                self.actions.append('detach');self.owned=False
                self.history.append(dict(action='detach',response={'status':'DETACHED'}))
                return dict(status='DETACHED',history=self.history)
            def snapshot(self):return dict(ownership_retained=self.owned,ticket=None,attachment=None,worker={},history=self.history)
        client=Client();active=NS(report=lambda:{},prepare_platform=lambda p:{},verify=lambda:{})
        llm=NS(get_tokenizer=lambda:NS(encode=lambda *a,**k:[1]))
        saved=P.common.save
        def fail_host(path,value):
            if Path(path).name=='host_receipt.json':raise OSError('CPU host receipt write failure')
            saved(path,value)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);model=root/'model';model.mkdir();(model/'fixture.safetensors').write_bytes(b'CPU fixture')
            (model/'config.json').write_text(json.dumps(dict(num_hidden_layers=28,num_attention_heads=16,num_key_value_heads=8,hidden_size=1024)))
            out=root/'probe';out.mkdir()
            args=NS(out=out,model=model,build=root/'completion.json',build_sha256='a'*64,
                source_manifest_sha256='b'*64,observer_receipt=root/'observer.json',observer_receipt_sha256='c'*64,memory=.25)
            with patch.object(paged33_loading,'bootstrap',return_value=active),\
                    patch.dict(sys.modules,{'vllm':NS(LLM=lambda **k:llm,SamplingParams=lambda **k:None),
                        'vllm.platforms':NS(current_platform=object())}),\
                    patch.object(serving_context,'require_host_driven_core'),\
                    patch.object(P.common,'HostAdapter',return_value=object()),\
                    patch.object(P.common,'WorkerFaultClient',return_value=client),\
                    patch.object(P.common,'save',side_effect=fail_host):
                self.assertEqual(P.run(args,{}),2)
            report=json.loads((out/'result.json').read_text())
            self.assertEqual(client.actions,['prepare','detach'],report.get('traceback'))
            self.assertIn('RPC_UNCERTAIN',report['error']);self.assertIn('RPC_UNCERTAIN',report['traceback'])
            self.assertIn('CPU host receipt write failure',report['post_protocol_errors'][0]['error'])
            self.assertNotIn('host_receipt',report);self.assertFalse((out/'host_receipt.json').exists())
            self.assertFalse(client.owned)

    def test_cold_probe_import_does_not_initialize_a_model_or_load_an_operator(self):
        script=ROOT/'experiments/worker_core_v11_20261118/natural_tnd28_prepared_input_probe.py'
        code='import runpy,sys;runpy.run_path(sys.argv[1]);assert not any(n in sys.modules for n in ("torch","torch_npu","vllm"))'
        result=subprocess.run([sys.executable,'-B','-c',code,str(script)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
