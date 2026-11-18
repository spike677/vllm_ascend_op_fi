"""CPU byte/JSON contracts only; no Worker, NPU, native library or qualification."""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT/'experiments/c2c3_closure_20261117'
sys.path.insert(0,str(EXPERIMENT))
import arch22_normal_bridge_evidence as evidence
import prepare_arch22_normal_bridge_cases as prepare
import review_arch22_normal_bridge as review
import test_attention33_first_boundary_review as boundary_fixtures


def hash_bytes(raw): return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')


def link_bytes(target, original):
    try: target.hardlink_to(original)
    except PermissionError: shutil.copyfile(original,target)


def elf_fixture():
    names=b'\0.shstrtab\0.strtab\0.symtab\0';strings=b'\0';symbols=bytearray(24)
    for core in ('aic','aiv'):
        label=f'FusedInferAttentionScore_8cd36e66e4ceb2d60ef96db29a147347_{evidence.KEY}_mix_{core}'
        offset=len(strings);strings+=label.encode()+b'\0'
        symbols.extend(struct.pack('<IBBHQQ',offset,0x12,0,1,64,128))
    blob=bytearray(64);parts=[]
    for value in (names,strings,symbols):parts.append((len(blob),len(value)));blob.extend(value)
    while len(blob)%8:blob.append(0)
    shoff=len(blob)
    for row in ((0,0,0,0,0,0,0,0,0,0),(1,3,0,0,*parts[0],0,0,1,0),
                (11,3,0,0,*parts[1],0,0,1,0),(19,2,0,0,*parts[2],2,1,8,24)):
        blob.extend(struct.pack('<IIQQQQIIQQ',*row))
    blob[:64]=struct.pack('<16sHHIQQQIHHHHHH',b'\x7fELF\x02\x01\x01'+bytes(9),1,183,1,0,0,shoff,0,64,0,0,64,4,1)
    return bytes(blob)


def source():
    return dict(root=ROOT,pins={name:evidence.sha(ROOT/name) for name in evidence.REQUIRED},
                manifest=dict(base_commit='test-source-only'),manifest_sha256='1'*64)


def pod_fixture(helper):
    # Synthetic field layout: exercises the source-derived decoder and every
    # array element. It is explicitly not an actual compiled ABI receipt.
    expected = {'baseParams.bSize':1,'baseParams.n2Size':8,'baseParams.gSize':2,
        'baseParams.headDim':128,'baseParams.headDimRope':0,'baseParams.softmaxLseFlag':0,
        'baseParams.actualSeqS1Dims':1,'baseParams.actualSeqS2Dims':1,'baseParams.s1Size':1,
        'baseParams.s2Size':2048,'innerSplitParams.s2BaseSize':512,'pageAttenParams.blockSize':16,
        'pageAttenParams.maxBlockNumPerBatch':80,'maskParams.attenMaskFlag':1,'maskParams.sparseMode':3,
        'maskParams.preToken':2147483647,'maskParams.nextToken':2147483647,
        'leftPaddingParams.qPaddingFlag':0,'leftPaddingParams.kvPaddingFlag':0,
        'prefixParams.prefixFlag':0,'pseParams.pseShiftFlag':0}
    fields = []; raw = bytearray(1576)
    for index,(name,value) in enumerate(expected.items()):
        fields.append(dict(name=name,offset=128+index*8,width=8,count=1,dtype='uint64_t'))
        struct.pack_into('<Q',raw,128+index*8,value)
    fields += [dict(name='baseParams.scaleValue',offset=400,width=4,count=1,dtype='float'),
        dict(name='actual_fixture.full_array',offset=512,width=4,count=28,dtype='uint32_t'),
        dict(name='baseParams.FusedInferAttentionBaseParamsPH',offset=81,width=1,count=7,dtype='uint8_t'),
        dict(name='prefixParams.FusedInferAttentionPrefixParamsPH',offset=1537,width=1,count=7,dtype='uint8_t')]
    struct.pack_into('<f',raw,400,128**-.5)
    struct.pack_into('<28I',raw,512,*range(28))
    schema=dict(schema='fault28.fia_graph.public_serializer_layout.v1',
        type='FusedInferAttentionScoreTilingData',size=len(raw),fields=fields)
    typed=helper.abi.decode_raw(raw,schema)
    host=dict(schema='demoncat.paged33.arch22_normal_host.v1',route='FiaTilingNonQuant',mode='arch22_normal',
        normal_route_preference_changed=True,eligible=True,policy_applied=False,native_s2=512,selected_s2=512,
        token='a'*32,pid=12,tid=34,tiling_key=evidence.KEY,block_dim=16,schedule_mode=1,
        raw_size=len(raw),raw_hex=raw.hex(),typed={name:value for name,value in typed.items() if not name.endswith('ParamsPH')})
    abi=dict(pointer_bytes=8,tiling_pointer_offset=24)
    args=bytes(64)+raw+bytes(16)
    event=dict(argument_layout='aclrt_host_placeholders',args_size=len(args),args_hex=args.hex(),
        placeholders_hex=struct.pack('<IIII',24,64,8,64+1576).hex())
    resolved=dict(call_id=host['token'],pid=12,tid=34,tiling_key=str(evidence.KEY),block_dim=16,schedule_mode=1)
    return event,resolved,host,dict(abi=abi,schema=schema)


def bundle_fixture(root, frozen, helper):
    build=root/'build/run'; vendor=build/'stage/vendors'/evidence.VENDOR
    kernel='op_impl/ai_core/tbe/kernel/ascend910b/FusedInferAttentionScore_8cd36e66e4ceb2d60ef96db29a147347.o'
    binary=elf_fixture()
    path=vendor/kernel;path.parent.mkdir(parents=True);path.write_bytes(binary)
    metadata=dict(binFileName=Path(kernel).stem,sha256=hash_bytes(binary),parameters=[None,None],
        coreType='MIX',intercoreSync=1,kernelList=[dict(tilingKey=key,kernelType='MIX_AIC',crossCoreSync=1,taskRation='1:2')
            for key in sorted(evidence.KEYS)],supportInfo=dict(tilingKey=sorted(evidence.KEYS)))
    write(path.with_suffix('.json'),metadata)
    code=vendor/'source/attention/fused_infer_attention_score/op_kernel/fused_infer_attention_score.cpp'
    code.parent.mkdir(parents=True);code.write_text('void fused_infer_attention_score(__gm__ uint8_t* query, '
        '__gm__ uint8_t* workspace, __gm__ uint8_t* tiling) {}')
    lib=vendor/'op_api/lib/libcust_opapi.so';lib.parent.mkdir(parents=True);lib.write_bytes(b'not loaded')
    schema=pod_fixture(helper)[3]['schema'];schema_path=build/'evidence/arch22_schema/fia_abi_schema.json'
    projection=schema_path.with_name('fia_schema_projection.h');projection.parent.mkdir(parents=True)
    projection.write_bytes(b'CPU test public projection')
    public=build/'source/attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.h'
    public.parent.mkdir(parents=True);public.write_bytes(b'CPU test public header')
    decoder=schema_path.with_name('fia_abi_decoder');decoder.write_bytes(b'CPU fixture never executed')
    schema.update(projection_sha256=evidence.sha(projection),public_source_sha256=evidence.sha(public))
    write(schema_path,schema)
    compile_path=schema_path.with_name('compile_receipt.json')
    write(compile_path,dict(schema='demoncat.fia.fresh_abi_compile.v1',schema_sha256=evidence.sha(schema_path),
        npu_executed=False,loading_allowed=False,compiler_source_sha256=frozen['pins'][evidence.NATIVE+'fia/fia_abi_decoder.cpp'],
        projection_sha256=evidence.sha(projection),public_source_sha256=evidence.sha(public),decoder_sha256=evidence.sha(decoder),
        actual_new_fia_compile_entry={'file':'/build/run/source/attention/common/op_host/arch22/fia_tiling_nonquant.cpp'}))
    write(schema_path.with_name('fia_abi_selftest.json'),dict(status='PASS_CPU_DERIVED_LAYOUT_SENTINEL',serialized_bytes=1576,
        stock_host_abi_verified=False,npu_devices_required=False,invalid_lengths_rejected=2,
        fields={row['name']:{k:v for k,v in row.items() if k!='name'} for row in schema['fields']}))
    patch_path=root/'patch/policy_source_manifest.json'
    write(patch_path,dict(source_commit=evidence.COMMIT,variant=evidence.VARIANT,
        recipe_sha256=frozen['pins'][evidence.NATIVE+'fia/apply_tnd33_arch22_bridge.py'],normal_route_preference_changed=True,
        tensor_payload_mutated=False,kernel_source_modified=False,tiling_arithmetic_modified=False,
        fault_execution_ready=False,stock_private_exact_proven=False,files=[dict(path=public.relative_to(build/'source').as_posix(),
            bytes=public.stat().st_size,after_sha256=evidence.sha(public))]))
    data=dict(schema='demoncat.fia.fresh_build.v1',source_commit=evidence.COMMIT,variant=evidence.VARIANT,
        status='CPU_BUILD_ONLY_REQUIRES_NATIVE_QUALIFICATION',loading_allowed=False,npu_executed=False,
        previous_experiment_objects_used=False,fault_execution_ready=False,stock_private_exact_proven=False,
        compiled_keys=sorted(evidence.KEYS),normal_bridge_recipe_sha256=frozen['pins'][evidence.NATIVE+'fia/apply_tnd33_arch22_bridge.py'],
        normal_bridge_header_sha256=frozen['pins'][evidence.NATIVE+'fia/paged33_arch22_bridge.h'],
        contract_sha256=frozen['pins'][evidence.NATIVE+'fia/build_contract.json'],kernel_binary_sha256=hash_bytes(binary),
        sdk_interface_manifest_sha256='d'*64,
        patch_receipt_sha256=evidence.sha(patch_path),
        arch22_public_schema_sha256=evidence.sha(schema_path),arch22_schema_compile_sha256=evidence.sha(compile_path),
        artifacts=[dict(path=p.relative_to(vendor).as_posix(),bytes=p.stat().st_size,sha256=evidence.sha(p))
                   for p in vendor.rglob('*') if p.is_file()])
    write(build/'evidence/completion.json',data)
    write(root/'controller.json',dict(status='BUILD_COMMAND_COMPLETED',exit_code=0,actual_npu_calls=0,devices=[],
        source_commit=evidence.COMMIT,sdk_interfaces_sha256='d'*64,image_id=evidence.IMAGE,
        strategy='arch22-normal-bridge',owned_container_removed=True,final_state={'Running':False}))
    return build,data


def run_fixture(run, kind, frozen, helper, bundle, observer):
    """Complete protocol file shapes with CPU-generated bytes, never a Worker."""
    run.mkdir(parents=True)
    base,bound=boundary_fixtures.boundary_fixture(run)
    saved=base['finish']['capture']['first_decode_boundary']
    mask=saved['inputs']['atten_mask']['snapshot']
    mask.update(metadata=boundary_fixtures.metadata([2048,2048],'torch.int8',25),bytes=2048**2,
                sha256=hash_bytes(bytes(2048**2)))
    saved['inputs'].pop('optional_tensor')
    for name,value in dict(scale=128**-.5,num_heads=16,num_key_value_heads=8,actual_seq_lengths=[1],
        actual_seq_lengths_kv=[1025],pse_shift=None).items():saved['inputs'][name]=dict(kind='scalar',value=value)
    saved['omitted_argument_defaults'].update(pre_tokens=2147483647,next_tokens=2147483647)
    saved['supplied_argument_names']=sorted(saved['inputs'])
    saved['inputs_after']['atten_mask']=deepcopy(mask);saved['inputs_after']['atten_mask']['path']=None
    saved['subsequent_checks']={name:True for name in set(saved['inputs'])|{'logical_key','logical_value',
        'impl_key_same_object','impl_value_same_object','output_0','output_1'}}
    saved['thread']=34
    tokens=[10]*1024;output=[dict(tokens=[11,12],text='fixture',finish_reason='length')]
    installed={'demoncat':{},'fault_injection':{}}
    for name,digest in frozen['pins'].items():
        if name.startswith('src/demoncat/'):installed['demoncat'][name.removeprefix('src/demoncat/')]=dict(sha256=digest,bytes=1)
        elif name.startswith('fault_injection/'):installed['fault_injection'][name.removeprefix('fault_injection/')]=dict(sha256=digest,bytes=1)
    execution=dict(schema='demoncat.paged33.execution_contract.v1',driver_sha256=frozen['pins'][evidence.EXPERIMENT+'attention33_worker_probe.py'],
        contract_source_sha256=frozen['pins'][evidence.EXPERIMENT+'attention33_contract.py'],installed_files=installed,
        runner=dict(helper.contract.RUNNER,gpu_memory_utilization=.35),sampling=dict(temperature=0,max_tokens=2,ignore_eos=True),
        prompt_tokens=1024,environment={name:prepare.case_environment(2)[name] for name in helper.contract.ENVIRONMENT},
        model_layout=dict(model_type='qwen3',hidden_size=1024,num_hidden_layers=28,num_attention_heads=16,
                          num_key_value_heads=8,head_dim=128,vocab_size=151936),
        model_files={'config.json':dict(sha256='1'*64,bytes=1),'model.safetensors':dict(sha256='2'*64,bytes=1)})
    doc=dict(schema='demoncat.paged33.first_boundary_worker_run.v1',phase=kind,diagnostic_only=True,
        statistics=False,normal_bridge_variant=evidence.VARIANT,normal_route_preference_changed=kind=='baseline',
        bridge_compatible=False,fault_authorized=False,stock_private_exact_proven=False,public_worker_qualified=False,c2=False,c3=False,
        observer_receipt_sha256=observer['sha256'],completion_sha256=bundle['completion_sha256'] if kind=='baseline' else None,
        schema_sha256=bundle['summary']['schema_sha256'] if kind=='baseline' else None,execution_contract=execution,
        warm_healthy=dict(prompt_token_ids=tokens,outputs=output),prompt=dict(token_count=1024,
            token_sha256=hash_bytes(json.dumps(tokens,separators=(',',':')).encode())),
        healthy_after_owned_observers_removed=dict(prompt_token_ids=tokens,outputs=output),
        status='A_FIXTURE_SUCCESS_WORD_IS_NOT_A_PROOF',phases=[])
    runtime=run/'runtime';runtime.mkdir();binary=runtime/'registered-12-1.bin';binary.write_bytes(elf_fixture())
    def event(name,now,token='',**kwargs):return dict(schema=helper.boundary.SCHEMA,event=name,pid=12,tid=34,
        monotonic_ns=now,call_id=token,**kwargs)
    events=[event('register',1,handle=80,generation=1,binary_file=binary.name,binary_bytes=binary.stat().st_size,result=0)]
    kernel=next(row for row in bundle['artifacts'].values() if row['path'].endswith('.o'))
    abi_event,_,base_host,_=pod_fixture(helper);counter=0;now=10
    c=review.literal_constants(frozen,'paged33_loading.py',('STOCK_FILES','LOADER_HEADER_SHA','NORM_SYMBOLS','FIA_SYMBOLS','PRIVATE_LIB'))
    norm_path='/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/'+c['PRIVATE_LIB']
    norm=[dict(symbol=name,address=100+i,path=norm_path,sha256=c['STOCK_FILES']['_cann_ops_custom/vendors/vllm-ascend/'+c['PRIVATE_LIB']]) for i,name in enumerate(c['NORM_SYMBOLS'])]
    fia=[dict(symbol=name,address=200+i,path='/native/stage/vendors/'+evidence.VENDOR+'/'+c['PRIVATE_LIB'],
        sha256=bundle['artifacts'][c['PRIVATE_LIB']]['sha256']) for i,name in enumerate(c['FIA_SYMBOLS'])]
    coexist=dict(schema='demoncat.paged33.api_coexistence.v1',pid=12,stock_files=c['STOCK_FILES'],stock_loader_source_sha256=c['LOADER_HEADER_SHA'],
        private_global_library_path_added=False,operator_function_invocations=0,model_or_norm_replaced=False,actual_fia_launch_proven=False,
        qualification='SYMBOL_BINDING_ONLY',platform_ready=True,stock_binding_before_private=norm,
        bindings=dict(stock_norm=norm,private_fia=fia,distinct_library_handles=True),
        vendor_order=['/native/stage/vendors/'+evidence.VENDOR,norm_path.removesuffix('/'+c['PRIVATE_LIB'])])
    if kind=='baseline':
        doc.update(library_bootstrap=dict(deepcopy(coexist),platform_ready=False,vendor_order=coexist['vendor_order'][:1]),
                   library_coexistence=deepcopy(coexist),library_coexistence_after_standard_worker_initialization=deepcopy(coexist))
    original_files={kind:run/'baseline_before/private'/('layer-0-'+kind+'.bin') for kind in ('output','key','value')}
    for label,size in (('hidden',2048),('logits-hidden',2048),('logits',151936*2)):
        (run/'baseline_before/private'/(label+'.bin')).write_bytes(bytes(size))
    for phase_index,name in enumerate(review.PHASES):
        private=kind=='baseline' and name!='healthy_after';folder=run/name
        (folder/'private').mkdir(parents=True,exist_ok=True);(folder/'first_boundary').mkdir(exist_ok=True)
        value=deepcopy(saved);value['inputs']['query']['snapshot']['path']=f'/evidence/run/{name}/first_boundary/query.bin'
        for index,item in enumerate(value['outputs']):item['path']=f'/evidence/run/{name}/first_boundary/output-{index}.bin'
        for filename in ('query.bin','output-0.bin','output-1.bin'):
            target=folder/'first_boundary'/filename;original=run/'baseline_before/first_boundary'/filename
            if target!=original:link_bytes(target,original)
        attention={};rows=[];request_id='request-'+name
        scheduler=dict(request_ids=[request_id],counts=[1],pure_decode=True,row_token_positions=[1024],input_ids=[11],
                       prefixes=[dict(prompt=tokens,output=[11])])
        for index,path in enumerate(sorted(review.LAYERS)):
            row=deepcopy(bound);row.update(module_path=path,module_object_id=100+int(path.split('.')[2]),
                impl_object_id=200+int(path.split('.')[2]),actual_operator='torch_npu.npu_fused_infer_attention_score',
                consumer_returned=True,layout='TND',virtual_engine=0,scheduler=deepcopy(scheduler),expected_slots=[1024])
            row['writer']=dict(slots=[1024],returned=True,module_path=path,
                key_cache=deepcopy(row['live_key_cache']),value_cache=deepcopy(row['live_value_cache']))
            rows.append(row);attention[path]={}
            for tensor in ('output','key','value'):
                target=folder/'private'/f'layer-{index}-{tensor}.bin'
                if target!=original_files[tensor]:link_bytes(target,original_files[tensor])
                record=deepcopy(base['finish']['capture']['attention'][review.FIRST][tensor])
                record['path']=f'/evidence/run/{name}/private/'+target.name;attention[path][tensor]=record
        value['boundary']=deepcopy(next(row for row in rows if row['module_path']==review.FIRST))
        capture=dict(schema='demoncat.paged33.natural_probe.v1',all_represented_tensors_finite=True,
            complete_tensor_readback_during_statistics=False,attention=attention,
            audit=dict(candidate_count=28,decode_forwards=1,prefill_forwards=1,errors=[],rows=rows,
                       core_methods_replaced=False,source={'fixture':'explicit CPU only'}),native_calls=[],first_decode_boundary=value)
        for label,filename,shape in (('hidden','hidden.bin',[1,1024]),('logits_hidden','logits-hidden.bin',[1,1024]),('logits','logits.bin',[1,151936])):
            target=folder/'private'/filename;original=run/'baseline_before/private'/filename
            if target!=original:link_bytes(target,original)
            capture[label]=dict(path=f'/evidence/run/{name}/private/'+filename,shape=shape,dtype='torch.float16',
                bytes=target.stat().st_size,sha256=evidence.sha(target),finite=True)
        native_rows=[];hosts=[]
        if private:(folder/'native').mkdir()
        for path in (review.LAYERS if private else [review.FIRST]):
            counter+=1;token=f'{counter:032x}';raw_event=deepcopy(abi_event)
            events += [event('begin',now,token),event('function_bind',now+1,token,handle=80,generation=1,function_handle=90,tiling_key=str(evidence.KEY),result=0),
                event('launch',now+5,token,**raw_event,api='aclrtLaunchKernelWithHostArgs',handle=80,generation=1,
                    function_handle=90,binary_file=binary.name,tiling_key=str(evidence.KEY),block_dim=16,schedule_mode=1,
                    launch_begin_ns=now+2,launch_end_ns=now+4,stream=9,result=0,snapshot_valid=True,observer_mutates_arguments=False),
                event('stream_sync',now+6,token,stream=9,result=0),event('device_sync',now+7,token,result=0),event('end',now+8,token)]
            now+=10
            if path==review.FIRST:value['native_call_id']=token
            if private:
                host=deepcopy(base_host);host['token']=token;hosts.append(host)
                resolved=dict(call_id=token,pid=12,tid=34,tiling_key=str(evidence.KEY),block_dim=16,schedule_mode=1)
                decoded=review.final_private_pod(raw_event,resolved,host,bundle,helper.abi)
                native=dict(decoded,**resolved,kernel_sha256=kernel['sha256'])
                row=next(row for row in rows if row['module_path']==path)
                call=dict(call_id=token,token=token,module_path=path,module_object_id=row['module_object_id'],
                    scheduler=scheduler,virtual_engine=0,original_calls=1,phase='baseline' if name=='measured' else 'off',
                    selection_applied=False,cache_repacked=False,api_arguments_changed=False,output_mutated=False,host=host,native=native)
                native_rows.append(call);write(folder/'native'/(token+'.json'),call)
        capture['native_calls']=native_rows
        if private:(folder/'native/host.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in hosts))
        cleanup=dict(scope_cleared=True,local_streams_drained=True,owned_hooks=0,owned_api_wrappers=0,boundary_refs_released=True,
            core_methods_replaced=False,failures=[],scope_active=False,scope_owned=False,environment_owned=0,first_error=None,
            calls=28,simulation=False,library_coexistence=deepcopy(coexist),
            stock_launch_scope=dict(scope_cleared=True,local_streams_drained=True,errors=[]))
        phase=dict(phase=name,finish=dict(closed=True,cleanup=cleanup,capture=capture),
            response=dict(prompt_token_ids=tokens,outputs=output,request_id=request_id),
            mapping=dict(external_request_id=request_id,internal_request_id=request_id),expected=dict(actual_internal_request_id=request_id),
            setup=dict(candidate_count=28,source=capture['audit']['source'],stock_read_only=not private,
                module_sha256=frozen['pins'][evidence.WORKER+'paged33_tnd_probe.py'],core_methods_replaced=False,first_boundary_diagnostic=True))
        doc['phases'].append(phase)
        write(folder/'runtime_result.json',capture);write(folder/'first_boundary/boundary.json',value)
    (runtime/'runtime.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in events))
    write(run/'result.json',doc)
    return doc


class NormalBridgeReaderContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.frozen=source();cls.helper=evidence.helpers(cls.frozen)

    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        if os.name == 'nt': temporary.name='\\\\?\\'+str(Path(temporary.name).resolve())
        self.folder=Path(self.enterContext(temporary))

    def test_bounded_normal_build_uses_actual_entry_source_and_full_artifact_bytes(self):
        root,data=bundle_fixture(self.folder,self.frozen,self.helper)
        found=evidence.build_bundle(root,self.frozen,self.helper,evidence.sha(root/'evidence/completion.json'))
        self.assertEqual(found['abi']['names'],['query','workspace','tiling'])
        self.assertEqual(found['abi']['tiling_pointer_offset'],24)
        self.assertEqual(found['summary']['compiled_keys'],sorted(evidence.KEYS))
        path=found['vendor']/data['artifacts'][0]['path'];path.write_bytes(b'changed after build')
        with self.assertRaisesRegex(ValueError,'ARTIFACT_BYTES_DIFFER'):evidence.build_bundle(root,self.frozen,self.helper)

    def test_old_fd_build_or_false_fault_authority_cannot_be_reused(self):
        root,data=bundle_fixture(self.folder,self.frozen,self.helper)
        for key,value in (('variant','tnd_page16_fai_fd_s2_256'),('fault_execution_ready',True),('stock_private_exact_proven',True)):
            with self.subTest(key=key):
                changed=deepcopy(data);changed[key]=value;write(root/'evidence/completion.json',changed)
                with self.assertRaisesRegex(ValueError,'SEPARATE_NORMAL_ONLY'):evidence.build_bundle(root,self.frozen,self.helper)

    def test_missing_controller_cleanup_or_extra_artifact_is_not_a_valid_build(self):
        root,data=bundle_fixture(self.folder,self.frozen,self.helper)
        controller=evidence.read(self.folder/'controller.json');controller['owned_container_removed']=False
        write(self.folder/'controller.json',controller)
        with self.assertRaisesRegex(ValueError,'DEVICE_FREE_BUILD_CONTROLLER'):evidence.build_bundle(root,self.frozen,self.helper)
        controller['owned_container_removed']=True;write(self.folder/'controller.json',controller)
        (root/'stage/vendors'/evidence.VENDOR/'extra.bin').write_bytes(b'not in manifest')
        with self.assertRaisesRegex(ValueError,'MISSING_OR_EXTRA_FILES'):evidence.build_bundle(root,self.frozen,self.helper)

    def test_observer_original_path_alias_is_same_readonly_bytes_in_both_arms(self):
        folder=self.folder/'observer/attention-preflight';folder.mkdir(parents=True)
        library=folder/'observer.so';library.write_bytes(b'not loaded')
        row=dict(observer_compiled=True,actual_npu_calls=0,observer_memory_selftest={'exit_code':0},
            observer_source={'sha256':self.frozen['pins'][evidence.NATIVE+'launch_observer.cpp']},
            observer_library={'path':'/output/cann851/attention-preflight/observer.so','sha256':evidence.sha(library)})
        write(folder/'result.json',row)
        observer=evidence.observer_bundle(self.folder/'observer',self.frozen)
        args=NS(source=self.folder/'source',installed=self.folder/'installed',model=self.folder/'model',
            build_root=self.folder/'build',card=2,timeout=1800)
        bundle=dict(completion_sha256='2'*64,completion={'arch22_public_schema_sha256':'3'*64},summary={'variant':evidence.VARIANT})
        cases=prepare.make_cases(args,self.frozen,bundle,observer,{'exact_contract':'same'}, {})
        self.assertEqual(len(cases),2)
        for case in cases:
            self.assertIn([str(observer['folder']),'/output/cann851/attention-preflight','ro'],case['mounts'])
            self.assertIn('--arch22-normal-bridge',case['command'])
            self.assertNotIn('--statistics',case['command']);self.assertNotIn('--bridge ',case['command'])
            self.assertFalse(case['review']['fault_authorized'])
            self.assertTrue(all(row[2]=='ro' for row in case['mounts']))
        self.assertEqual(cases[0]['review']['execution_contract'],cases[1]['review']['execution_contract'])
        self.assertNotIn('--vendor',cases[0]['command'])
        self.assertIn(evidence.VENDOR,cases[1]['command'])
        row['observer_library']['path']='/usr/lib/observer.so';write(folder/'result.json',row)
        with self.assertRaisesRegex(ValueError,'ALIAS_UNREVIEWED'):evidence.observer_bundle(self.folder/'observer',self.frozen)

    def test_prepared_normal_cases_satisfy_actual_root_queue_contract(self):
        from queue_contract import validate_cases
        from pathlib import PurePosixPath
        for card in (2, 5):
            args=NS(source=PurePosixPath('/source'), installed=PurePosixPath('/installed'),
                model=PurePosixPath('/model'), build_root=PurePosixPath('/build'),card=card,timeout=1800)
            bundle=dict(completion_sha256='2'*64,completion={'arch22_public_schema_sha256':'3'*64},summary={'variant':evidence.VARIANT})
            observer=dict(folder=PurePosixPath('/observer/attention-preflight'),alias='/output/cann851/attention-preflight',
                sha256='4'*64,receipt={'observer_library':{'sha256':'5'*64}})
            cases=prepare.make_cases(args,self.frozen,bundle,observer,{'exact_contract':'same'}, {})
            self.assertEqual(validate_cases(cases,[card]),cases)
            self.assertEqual({case['review']['soc'] for case in cases},{'Ascend910B4/A2'})
            del cases[0]['review']['soc']
            with self.assertRaisesRegex(ValueError,'REVIEW_MANIFEST_INCOMPLETE'):
                validate_cases(cases,[card])

    def test_final_native_full_array_and_alignment_are_recomputed_without_adapter_status(self):
        event,resolved,host,bundle=pod_fixture(self.helper)
        decoded=review.final_private_pod(event,resolved,host,bundle,self.helper.abi)
        self.assertEqual(decoded['typed']['actual_fixture.full_array'],list(range(28)))
        self.assertEqual(len(decoded['verified_public_serializer_padding']),2)
        self.assertNotIn('native_launch_verified',decoded)
        changed=deepcopy(host);changed['typed']['actual_fixture.full_array'][-1]=999
        with self.assertRaisesRegex(ValueError,'HOST_ARGUMENT_VALUES_DIFFER'):
            review.final_private_pod(event,resolved,changed,bundle,self.helper.abi)
        changed=deepcopy(event);raw=bytearray.fromhex(changed['args_hex']);raw[64+1537]=1
        changed['args_hex']=raw.hex();changed_host=deepcopy(host);changed_host['raw_hex']=raw[64:64+1576].hex()
        with self.assertRaisesRegex(ValueError,'PADDING_DIFFERS'):
            review.final_private_pod(changed,resolved,changed_host,bundle,self.helper.abi)

    def test_final_runtime_bytes_owner_and_actual_slot_cannot_follow_host_claim(self):
        event,resolved,host,bundle=pod_fixture(self.helper)
        for change in ('slot','owner','raw','extra_field','fault_mode'):
            with self.subTest(change=change):
                e,r,h=deepcopy(event),deepcopy(resolved),deepcopy(host)
                if change=='slot': e['placeholders_hex']=struct.pack('<II',16,64).hex()
                elif change=='owner': h['tid']+=1
                elif change=='raw': e['args_hex']='ff'+e['args_hex'][2:64*2]+'ff'+e['args_hex'][64*2+2:]
                elif change=='extra_field': h['typed']['invented']=[0]
                else: h['mode']='s2_256'
                with self.assertRaises(ValueError):review.final_private_pod(e,r,h,bundle,self.helper.abi)

    def test_raw_runtime_normal_key_joins_current_scope_and_rejects_old_fd_or_owner_conflict(self):
        fixture=boundary_fixtures.RuntimeResolutionTests('test_current_scope_resolves_29_identical_bindings_without_rewriting_gap')
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        result=review.one_launch(fixture.run,fixture.rows,fixture.token,12,self.helper)
        self.assertEqual(result['launches'][0]['function_binding']['repeated_identical_history'],28)
        self.assertFalse(result['original_runtime_observation_rewritten'])
        fixture.rows[2]['handle']=88
        with self.assertRaisesRegex(ValueError,'CONFLICTING_FUNCTION'):review.one_launch(fixture.run,fixture.rows,fixture.token,12,self.helper)
        fixture.rows[2]['handle']=80
        for row in fixture.rows:
            if 'tiling_key' in row:row['tiling_key']='5100000000010200103'
        with self.assertRaisesRegex(ValueError,'NORMAL_ENTRY_PROFILE'):review.one_launch(fixture.run,fixture.rows,fixture.token,12,self.helper)

    def test_failed_sync_is_not_masked_by_a_later_success(self):
        fixture=boundary_fixtures.RuntimeResolutionTests('test_missing_completion_sync_refused')
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        fixture.rows[-3]['result']=9
        with self.assertRaisesRegex(ValueError,'SCOPED_CAPTURE_OR_COMPLETION_FAILED'):
            review.one_launch(fixture.run,fixture.rows,fixture.token,12,self.helper)

    def test_only_actual_aic_and_aiv_elf_symbols_can_reconcile_normal_entry(self):
        fixture=boundary_fixtures.RuntimeResolutionTests('test_missing_completion_sync_refused')
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        with self.assertRaisesRegex(ValueError,'ELF_ENTRY_SYMBOL_MISSING'):
            review.one_launch(fixture.run,fixture.rows,fixture.token,12,self.helper,symbols={})

    def test_capture_cleanup_cannot_discard_native_scope_or_first_error(self):
        cleanup=dict(scope_cleared=True,local_streams_drained=True,owned_hooks=0,owned_api_wrappers=0,
            boundary_refs_released=True,core_methods_replaced=False,failures=[],scope_active=False,scope_owned=False,
            environment_owned=0,first_error=None,calls=28,simulation=False)
        finish=dict(closed=True,cleanup=cleanup)
        self.assertTrue(review.check_cleanup(finish,True)['scope_cleared'])
        for name,value in (('scope_active',True),('scope_owned',True),('environment_owned',1),('first_error','original drain failure')):
            changed=deepcopy(finish);changed['cleanup'][name]=value
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'SCOPE_OR_ERROR_REMAINS'):
                review.check_cleanup(changed,True)

    def test_actual_scale_is_bound_before_fp32_native_rounding_and_other_scalars_remain_exact(self):
        inputs={key:dict(kind='scalar',value=value) for key,value in dict(input_layout='TND',block_size=16,
            sparse_mode=3,num_heads=16,num_key_value_heads=8,scale=1/math.sqrt(128),
            actual_seq_lengths=[1],actual_seq_lengths_kv=[1025]).items()}
        inputs.update(atten_mask={'snapshot':{'metadata':dict(dtype='torch.int8',shape=[2048,2048],stride=[2048,1])}},
            block_table={'snapshot':{'metadata':dict(dtype='torch.int32',shape=[1,80],stride=[80,1])}})
        value=dict(inputs=inputs,omitted_argument_defaults=dict(pre_tokens=2147483647,next_tokens=2147483647,softmax_lse_flag=False),
            return_container_type='tuple',outputs=[{}, {'metadata':{'shape':[0],'dtype':'torch.float32'}}])
        review.check_actual_inputs(value)
        inputs['scale']['value']=.125
        with self.assertRaisesRegex(ValueError,'ACTUAL_SCALAR_DIFFERS:scale'):review.check_actual_inputs(value)
        inputs['scale']['value']=128**-.5;inputs['block_size']['value']=128
        with self.assertRaisesRegex(ValueError,'ACTUAL_SCALAR_DIFFERS:block_size'):review.check_actual_inputs(value)

    def test_all_87_payloads_are_compared_when_first_boundary_is_incomplete(self):
        left=self.folder/'left.bin';right=self.folder/'right.bin'
        left.write_bytes(struct.pack('<e',0.0));right.write_bytes(struct.pack('<e',-0.0))
        def tensor(path):return dict(path=str(path),dtype='torch.float16',shape=[1],bytes=2,sha256=evidence.sha(path))
        names=['hidden','logits_hidden','logits']+[path+'/'+kind for path in review.LAYERS for kind in ('output','key','value')]
        a=dict(boundary=None,payloads={name:tensor(left) for name in names})
        b=deepcopy(a);b['payloads'][review.LAYERS[27]+'/output']=tensor(right)
        result=review.compare_pair(a,b,self.helper)
        self.assertEqual(result['all_complete_payloads']['payload_pairs'],87)
        self.assertEqual(result['all_complete_payloads']['differing_payloads'],1)
        self.assertEqual(result['all_complete_payloads']['first_recorded_attention_output_difference'],review.LAYERS[27])
        difference=result['all_complete_payloads']['selected_payload_summaries'][review.LAYERS[27]+'/output']
        self.assertEqual(difference['max_abs_difference'],0)
        self.assertEqual(difference['changed_elements'],1)
        self.assertFalse(result['exact'])

    def test_read_before_cache_and_immediate_output_cannot_be_replaced_by_later_snapshot(self):
        phase,bound=boundary_fixtures.boundary_fixture(self.folder)
        saved=phase['finish']['capture']['first_decode_boundary']
        saved['logical_kv_before']['key']['sha256']='0'*64
        write(self.folder/'baseline_before/first_boundary/boundary.json',saved)
        with self.assertRaisesRegex(ValueError,'PRE_POST_LOGICAL_KV'):
            self.helper.boundary.first_boundary(self.folder,phase,bound,difference_reader=self.helper.differences)
        saved['logical_kv_before']['key']['sha256']=saved['logical_kv_after']['key']['sha256']
        saved['outputs_after'][0]['sha256']='0'*64;write(self.folder/'baseline_before/first_boundary/boundary.json',saved)
        with self.assertRaisesRegex(ValueError,'IMMEDIATE_OUTPUT_CHECKPOINT'):
            self.helper.boundary.first_boundary(self.folder,phase,bound,difference_reader=self.helper.differences)

    def test_failed_partial_run_keeps_original_errors_and_cannot_inherit_old_certificate(self):
        for name in ('stock','baseline'):
            write(self.folder/name/'result.json',dict(status='FAILED_WITH_EVIDENCE',error='original preparation failed',
                traceback='complete original traceback',phases=[dict(phase='measured',finish={'cleanup_error':'native scope retained'})]))
        args=NS(source=self.folder,source_manifest_sha256='1'*64,build_root=self.folder,observer=self.folder,
            stock_run=self.folder/'stock',baseline_run=self.folder/'baseline',
            stock_sha256=evidence.sha(self.folder/'stock/result.json'),baseline_sha256=evidence.sha(self.folder/'baseline/result.json'),
            completion_sha256='2'*64,observer_receipt_sha256='3'*64)
        report=review.review(args)
        self.assertEqual(report['status'],'INCOMPLETE_OR_REJECTED_NORMAL_BRIDGE_NO_F')
        self.assertEqual(len(report['arms']['stock']['original_failures']),3)
        self.assertFalse(report['fault_authorized']);self.assertFalse(report['bridge_certificate'])
        self.assertNotEqual(report['schema'],'demoncat.paged33.stock_private_bridge.v1')
        self.assertEqual(report['E5'],'NOT_ASSESSED')

    def test_three_tools_import_under_runtime_and_framework_audit_without_dso(self):
        code="import sys; sys.path.insert(0,sys.argv[1]); import arch22_normal_bridge_evidence as e; " \
             "sys.addaudithook(e.prohibit_runtime); import prepare_arch22_normal_bridge_cases, review_arch22_normal_bridge; " \
             "h=e.helpers({'root':__import__('pathlib').Path(sys.argv[2])}); assert h.abi.decode_raw; " \
             "assert not any(x in sys.modules for x in ('torch','torch_npu','vllm')); print('readonly')"
        result=subprocess.run([sys.executable,'-B','-c',code,str(EXPERIMENT),str(ROOT)],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(result.stdout.strip(),'readonly')

    def test_preparer_cli_has_no_fault_stage_or_certificate_switch(self):
        result=subprocess.run([sys.executable,'-B',str(EXPERIMENT/'prepare_arch22_normal_bridge_cases.py'),'--help'],
                              capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertNotIn('--stage',result.stdout);self.assertNotIn('--bridge ',result.stdout)
        self.assertNotIn('--statistics',result.stdout)

    def test_complete_two_arm_files_replay_112_native_calls_and_all_435_payload_pairs(self):
        build,_=bundle_fixture(self.folder,self.frozen,self.helper)
        bundle=evidence.build_bundle(build,self.frozen,self.helper);observer={'sha256':'b'*64}
        checks=review.Checks();symbols={};arms={}
        for kind in ('stock','baseline'):
            run=self.folder/kind;run_fixture(run,kind,self.frozen,self.helper,bundle,observer)
            arms[kind]=review.review_arm(run,evidence.sha(run/'result.json'),kind,self.frozen,self.helper,bundle,observer,checks,symbols)
        self.assertEqual(checks.issues,[])
        self.assertEqual(arms['baseline']['summary']['private_native_replayed'],112)
        self.assertEqual(len(arms['stock']['summary']['within_run']),4)
        total=0
        for name in review.PHASES:
            result=review.compare_pair(arms['stock']['values'][name],arms['baseline']['values'][name],self.helper)
            self.assertTrue(result['exact']);total+=result['all_complete_payloads']['payload_pairs']
        self.assertEqual(total,435)
        self.assertEqual(len(next(iter(symbols.values()))['actual_selected_entry_symbols']),2)


if __name__ == '__main__': unittest.main()
