"""Bounded experimental FIA qualification for a newly built vendor.

This tool does not modify the production FiaBackend certificate/SHA gates.
Disk validation and typed host raw observations are not kernel-argument or
profiler proof. Synthetic input acceptance is not model/Worker acceptance.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import traceback
import uuid

COMMIT='c77c0829f1f0f0b8f882db1d9b051a758b605472'
SHA=re.compile(r'[0-9a-f]{64}')
KEYS={'legal_tiling':{103000000000000000,109000000000000000,109000000000400000},
      'denominator_replay':{103000000000000000,109000000000000000,109000000000400000,119000000000400000}}
TILING_LINK='op_impl/ai_core/tbe/op_tiling/liboptiling.so'
TILING_TARGET='lib/linux/aarch64/libcust_opmaster_rt2.0.so'

def verify_links(root,files):
    """Only the actually audited official in-vendor tiling link is admissible."""
    for path in root.rglob('*'):
        if not path.is_symlink():continue
        relative=path.relative_to(root).as_posix()
        require(relative==TILING_LINK and os.readlink(path)==TILING_TARGET,'UNREVIEWED_VENDOR_LINK')
        target=path.parent/TILING_TARGET
        target_relative=target.relative_to(root).as_posix()
        require(target.is_file() and not target.is_symlink() and target.resolve().is_relative_to(root)
            and relative in files and target_relative in files,'INVALID_TILING_LINK_TARGET')
        require(all(files[relative][key]==files[target_relative][key] for key in ('bytes','sha256')),
            'TILING_LINK_MANIFEST_BYTES_DIFFER')

def require(ok,message):
    if not ok:raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()

def save(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def verify_build(completion,expected,vendor):
    require(SHA.fullmatch(expected or '') and sha(completion)==expected,'NEW_COMPLETION_HASH_MISMATCH')
    result=json.loads(Path(completion).read_bytes())
    require(result.get('schema')=='demoncat.fia.fresh_build.v1'
        and result.get('status')=='CPU_BUILD_ONLY_REQUIRES_NATIVE_QUALIFICATION'
        and result.get('loading_allowed') is False,'EXPERIMENTAL_BUILD_RECEIPT_REQUIRED')
    require(result.get('source_commit')==COMMIT and result.get('npu_executed') is False
        and result.get('previous_experiment_objects_used') is False,'NEW_SOURCE_BUILD_CONTRACT_DIFFERS')
    variant=result.get('variant')
    require(variant in KEYS and set(result.get('compiled_keys',[]))==KEYS[variant],'NEW_COMPILED_KEYS_DIFFER')
    root=Path(vendor).resolve(strict=True)
    require(root.is_dir() and not Path(vendor).is_symlink(),'DIRECT_VENDOR_DIRECTORY_REQUIRED')
    artifacts=result.get('artifacts')
    require(isinstance(artifacts,list) and 0<len(artifacts)<=20000,'BOUNDED_NEW_INVENTORY_REQUIRED')
    expected_files={}
    declared={row['path']:row for row in artifacts}
    require(len(declared)==len(artifacts),'DUPLICATE_ARTIFACT_PATH')
    verify_links(root,declared)
    total=0
    for row in artifacts:
        rel=PurePosixPath(row['path'])
        require(not rel.is_absolute() and rel.as_posix()==row['path'] and '..' not in rel.parts
            and '\\' not in row['path'] and '.' not in rel.parts,'INVALID_ARTIFACT_PATH')
        path=root.joinpath(*rel.parts)
        require(row['path'] not in expected_files and path.is_file()
            and path.resolve().is_relative_to(root),'INVALID_OR_DUPLICATE_ARTIFACT')
        require(type(row.get('bytes')) is int and 0<=row['bytes']<=2*1024**3
            and path.stat().st_size==row['bytes'] and SHA.fullmatch(row.get('sha256',''))
            and sha(path)==row['sha256'],'ARTIFACT_BYTES_DIFFER:'+row['path'])
        total+=row['bytes'];require(total<=8*1024**3,'VENDOR_SIZE_EXCEEDS_BOUND')
        expected_files[row['path']]=row
    require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==set(expected_files),'VENDOR_INVENTORY_DIFFERS')
    require('op_api/lib/libcust_opapi.so' in expected_files,'NEW_OPAPI_MISSING')
    return dict(variant=variant,completion_sha256=expected,vendor_root=str(root),files=expected_files,
                kernel_binary_sha256=result['kernel_binary_sha256'],production_loading_allowed=False)

def plan(variant,phase):
    if phase in ('stock','off','restore'):return ('off',512,512,103000000000000000)
    require(phase in ('baseline','fault'),'UNKNOWN_PHASE')
    if variant=='legal_tiling':
        return ('coupled_base512',512,512,109000000000000000) if phase=='baseline' else ('coupled_m128',128,128,109000000000400000)
    require(variant=='denominator_replay','UNKNOWN_VARIANT')
    return ('coupled_m128',128,128,109000000000400000) if phase=='baseline' else ('coupled_replay_m128',128,128,119000000000400000)

def decode_host(row,schema):
    require(schema.get('schema')=='fault28.fia_graph.public_serializer_layout.v1'
        and schema.get('kernel_argument_raw_observed') is False,'PUBLIC_HOST_SCHEMA_REQUIRED')
    raw=bytes.fromhex(row['raw_hex'])
    require(len(raw)==row['raw_size']==schema['size'],'RAW_SERIALIZER_SIZE_DIFFERS')
    fields={x['name']:x for x in schema['fields']}
    codes={'bool':'?','uint8_t':'B','int8_t':'b','uint16_t':'H','int16_t':'h','uint32_t':'I','int32_t':'i','uint64_t':'Q','int64_t':'q','float':'f','double':'d'}
    typed=row['typed'];require(isinstance(typed,dict) and len(typed)>10,'MISSING_TYPED_HOST_FIELDS')
    decoded={}
    for name,value in typed.items():
        require(name in fields,'PUBLIC_SCHEMA_FIELD_MISSING:'+name)
        field=fields[name];code=codes[field['dtype']];count=field['count']
        require(count>0 and field['width']==struct.calcsize('<'+code)
            and field['offset']>=0 and field['offset']+count*field['width']<=len(raw),'PUBLIC_FIELD_BOUNDS_INVALID')
        values=struct.unpack_from('<'+code*count,raw,field['offset'])
        actual=values[0] if count==1 else list(values)
        require(actual==value,'HOST_TYPED_RAW_DIFFERS:'+name);decoded[name]=actual
    return decoded

def mapped_vendor(verified,stock=False):
    root=Path(verified['vendor_root'])
    rows=[];seen=set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts=line.split(maxsplit=5)
        if len(parts)!=6 or not parts[5].startswith(str(root)+'/'):continue
        require(not parts[5].endswith(' (deleted)'),'MAPPED_VENDOR_FILE_DELETED')
        path=Path(parts[5]);relative=path.relative_to(root).as_posix()
        require(relative in verified['files'] and path.is_file(),'MAPPED_FILE_NOT_IN_NEW_BUILD')
        stat=path.stat();major,minor=(int(x,16) for x in parts[3].split(':'))
        require(stat.st_ino==int(parts[4]) and os.major(stat.st_dev)==major and os.minor(stat.st_dev)==minor,'MAPPED_INODE_DIFFERS')
        if relative not in seen:
            require(sha(path)==verified['files'][relative]['sha256'],'MAPPED_LIBRARY_HASH_DIFFERS')
            rows.append(dict(path=str(path),sha256=sha(path),inode=stat.st_ino));seen.add(relative)
    if stock:require(not rows,'STOCK_PROCESS_MAPPED_CUSTOM_VENDOR')
    else:
        require('op_api/lib/libcust_opapi.so' in seen,'NEW_OPAPI_NOT_MAPPED')
        require(TILING_LINK in seen or str(PurePosixPath(TILING_LINK).parent/TILING_TARGET) in seen,
            'NEW_HOST_TILING_NOT_MAPPED')
    return rows

def collector(out):
    import msprobe
    from msprobe.pytorch import PrecisionDebugger
    config=dict(task='statistics',level='mix',precision='high',rank=[],step=[0],
        dump_path=str(out/'observation'),async_dump=False,
        statistics=dict(scope=[],list=[],data_mode=['all'],summary_mode='statistics',tensor_list=[]))
    save(out/'config.json',config)
    package=Path(msprobe.__file__).resolve().parent
    save(out/'collector_identity.json',dict(package=str(package),edited_by_experiment=False,
        files={p.relative_to(package).as_posix():sha(p) for p in package.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.yaml')},
        step_meaning={'0':'second identical standalone FIA call; not decode or complete model'}))
    return PrecisionDebugger(config_path=str(out/'config.json'))

def run_child(args,verified):
    require(args.experimental_qualification,'EXPLICIT_EXPERIMENTAL_QUALIFICATION_REQUIRED')
    out=args.out.resolve()
    require(out.is_relative_to(Path('/evidence')) and not out.exists(),'FRESH_PRIVATE_EVIDENCE_REQUIRED')
    out.mkdir(parents=True);private=out/'private';private.mkdir()
    result=dict(schema='demoncat.fia.fresh_qualification.v1',status='RUNNING',phase=args.phase,
        variant=verified['variant'],batch=args.batch,production_loading_allowed=False,
        public_worker_qualified=False,model_acceptance=False,native_launch_verified=False,
        kernel_argument_raw_observed=False,completion_sha256=args.completion_sha256,
        public_schema_sha256=args.schema_sha256,calls=[],limitations=['Standalone synthetic QKV only',
        'Host serializer raw/typed equality is not independent kernel-argument or launch proof',
        'No production NativeRuntime or FiaBackend qualification is granted'])
    try:
        require(sha(args.public_schema)==args.schema_sha256,'PUBLIC_SCHEMA_HASH_DIFFERS')
        schema=json.loads(args.public_schema.read_bytes())
        require(os.environ.get('ACLNN_CACHE_LIMIT')=='0','FRESH_UNCACHED_ACLNN_REQUIRED')
        if args.phase=='stock':require(not os.environ.get('ASCEND_CUSTOM_OPP_PATH'),'STOCK_VENDOR_ENV_NOT_EMPTY')
        else:require(os.environ.get('ASCEND_CUSTOM_OPP_PATH')==verified['vendor_root'],'EXACT_NEW_VENDOR_ENV_REQUIRED')
        import torch
        import torch_npu
        from model_capture import statistics_model_call
        torch.use_deterministic_algorithms(True)
        torch.npu.set_device(0);torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.set_device_limit(0,cube_num=16,vector_num=32)
        require(torch.npu.get_device_limit(0)==dict(cube_core_num=16,vector_core_num=32),'ACTUAL_CORE_LIMIT_DIFFERS')
        result['versions']=dict(torch=torch.__version__,torch_npu=torch_npu.__version__)
        result['execution']=dict(eager=True,jit_compile=False,core_limit=torch.npu.get_device_limit(0),logical_device=0)
        cpu=[]
        for heads,factor,offset in ((16,17,3),(8,29,5),(8,43,7)):
            x=((torch.arange(heads*2048*128,dtype=torch.int32)*factor+offset)%257-128).to(torch.float16)/128
            cpu.append(x.reshape(1,heads,2048,128).repeat(args.batch,1,1,1).contiguous())
        cpu.append(torch.ones((2048,2048),dtype=torch.bool).triu(1))
        digest=lambda tensor:hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        inputs=tuple(t.to('npu:0') for t in cpu);torch.npu.synchronize()
        input_hashes=[digest(t) for t in cpu]
        save(out/'inputs.json',dict(rule='exact integer modular pattern: ((i*factor+offset)%257-128)/128',
            qkv_parameters=[[16,17,3],[8,29,5],[8,43,7]],batch=args.batch,all_request_rows_identical=True,
            tensors=[dict(shape=list(t.shape),dtype=str(t.dtype),sha256=h) for t,h in zip(cpu,input_hashes)]))
        class FiaProbe(torch.nn.Module):
            def forward(self,q,k,v,mask):
                return torch_npu.npu_fused_infer_attention_score(q,k,v,atten_mask=mask,
                    actual_seq_lengths=[2048]*args.batch,actual_seq_lengths_kv=[2048]*args.batch,
                    num_heads=16,num_key_value_heads=8,scale=128**-0.5,input_layout='BNSD',
                    pre_tokens=2147483647,next_tokens=0,sparse_mode=2,inner_precise=0)[0]
        model=FiaProbe().eval()
        phases=['off','fault','restore','restore'] if args.phase=='off_recovery' else [args.phase]*2
        require(not args.statistics or args.phase in ('baseline','fault'),'STATISTICS_ONLY_FOR_BF_PAIRS')
        debugger=None;owner_start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
        outputs=[]
        for index,phase in enumerate(phases):
            mode,m,s2,key=plan(verified['variant'],phase)
            token=uuid.uuid4().hex;log=out/f'host-{index}.jsonl'
            env=dict(FAULT28_FIA_GRAPH_MODE=mode,FAULT28_FIA_GRAPH_TOKEN=token,FAULT28_FIA_GRAPH_PID=str(os.getpid()),
                FAULT28_FIA_GRAPH_OWNER_START=owner_start,FAULT28_FIA_GRAPH_LOG=str(log))
            if phase!='stock':os.environ.update(env)
            if args.statistics and index==1:debugger=collector(out)
            try:
                scope=statistics_model_call(debugger,model,torch.npu.synchronize) if debugger else nullcontext()
                with scope,torch.inference_mode():output=model(*inputs)
                torch.npu.synchronize()
            finally:
                for name in env:os.environ.pop(name,None)
            if debugger:debugger.step()
            require(all(digest(t)==h for t,h in zip(inputs,input_hashes)),'ACTUAL_INPUT_BYTES_CHANGED')
            value=output.detach().cpu().contiguous()
            require(list(value.shape)==[args.batch,16,2048,128] and value.dtype==torch.float16 and bool(torch.isfinite(value).all()),'OUTPUT_CONTRACT_FAILED')
            raw=value.numpy().tobytes();digest_out=hashlib.sha256(raw).hexdigest();outputs.append(digest_out)
            if index==len(phases)-1:(private/'output.fp16.bin').write_bytes(raw)
            row=dict(index=index,phase=phase,output_sha256=digest_out,full_output_shape=list(value.shape),
                finite=True,input_hashes_unchanged=True,statistics_step=0 if debugger else None)
            if phase!='stock':
                host=[json.loads(line) for line in log.read_text().splitlines()]
                matches=[h for h in host if h.get('token')==token]
                require(len(matches)==1,'EXACT_ONE_HOST_DECISION_REQUIRED')
                h=matches[0]
                require(h.get('eligible') is True and h.get('mode')==mode and h.get('selected_m')==m
                    and int(h.get('tiling_key',0))==key and h.get('owner_pid')==os.getpid()
                    and h.get('owner_starttime_ticks')==owner_start and h.get('checks') and all(h['checks'].values())
                    and h.get('serializer_roundtrip_matches') is True and h.get('raw_unchanged_by_observer') is True,'ACTUAL_HOST_CONTRACT_FAILED')
                decoded=decode_host(h,schema)
                require(decoded['innerSplitParams.mBaseSize']==m and decoded['innerSplitParams.s2BaseSize']==s2,'ACTUAL_HOST_M_S2_DIFFERS')
                require(h.get('kernel_argument_raw_observed') is False,'HOST_RECEIPT_OVERCLAIMS_KERNEL_RAW')
                row.update(host_log=log.name,host_log_sha256=sha(log),host_typed_raw_equal=True,actual_m=m,actual_s2=s2,
                    actual_host_tiling_key=key,kernel_argument_raw_observed=False,raw_sha256=hashlib.sha256(bytes.fromhex(h['raw_hex'])).hexdigest())
            result['calls'].append(row)
        if args.phase=='off_recovery':
            require(outputs[0]==outputs[2]==outputs[3],'OFF_NOT_EXACTLY_RESTORED')
            result['off_exact_restore']=True
            result['fault_changed_complete_output']=outputs[1]!=outputs[0]
        else:require(outputs[0]==outputs[1],'REPEATED_COMPLETE_OUTPUT_DIFFERS')
        result['loaded_vendor_libraries']=mapped_vendor(verified,stock=args.phase=='stock')
        if debugger:
            folder=out/'observation/step0/rank0'
            files={name:sha(folder/name) for name in ('dump.json','stack.json','construct.json')}
            dump=json.loads((folder/'dump.json').read_bytes())
            require(dump['task']=='statistics' and dump['level']=='mix' and dump.get('dump_data_dir') is None,'RAW_STATISTICS_CONTRACT_FAILED')
            fia=[key for key in dump['data'] if key.startswith('NPU.npu_fused_infer_attention_score.')]
            require(len(fia)==1,'ONE_NATIVE_FIA_STATISTICS_RECORD_REQUIRED')
            result['statistics']=dict(step=0,files=files,fia_records=fia,original_nulls_preserved=True,
                complete_model_accepted=False,independent_first_difference_pending=True)
        result.update(status='EXPERIMENTAL_EXECUTION_COMPLETED_NOT_LAUNCH_QUALIFIED',
            actual_npu_fia_calls=len(phases),
            repeated_complete_output_stable=True if args.phase!='off_recovery' else None)
    except BaseException as error:
        result.update(status='EXPERIMENTAL_QUALIFICATION_FAILED',error=repr(error),traceback=traceback.format_exc())
    save(out/'result.json',result)
    print(json.dumps({k:result.get(k) for k in ('status','phase','variant','actual_npu_fia_calls','error')}),flush=True)
    return 0 if result['status']=='EXPERIMENTAL_EXECUTION_COMPLETED_NOT_LAUNCH_QUALIFIED' else 2

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--completion',type=Path,required=True);p.add_argument('--completion-sha256',required=True)
    p.add_argument('--vendor',type=Path,required=True)
    p.add_argument('--public-schema',type=Path,required=True);p.add_argument('--schema-sha256',required=True)
    p.add_argument('--phase',choices=('stock','off','baseline','fault','off_recovery'),required=True)
    p.add_argument('--batch',type=int,choices=(1,8),default=8);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--statistics',action='store_true')
    p.add_argument('--experimental-qualification',action='store_true')
    p.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    verified=verify_build(args.completion,args.completion_sha256,args.vendor)
    require(SHA.fullmatch(args.schema_sha256) and sha(args.public_schema)==args.schema_sha256,'NEW_PUBLIC_SCHEMA_HASH_MISMATCH')
    require(args.experimental_qualification,'EXPLICIT_EXPERIMENTAL_QUALIFICATION_REQUIRED')
    if args.child:return run_child(args,verified)
    require(not os.environ.get('LD_PRELOAD') and not os.environ.get('ASCEND_CUSTOM_OPP_PATH')
        and not any(k.startswith('FAULT28_') for k in os.environ),'CLEAN_NEW_PROCESS_ENVIRONMENT_REQUIRED')
    require(not any(name in sys.modules for name in ('torch','torch_npu')),'PARENT_MUST_NOT_LOAD_FRAMEWORK')
    env=dict(os.environ,ACLNN_CACHE_LIMIT='0',PYTHONDONTWRITEBYTECODE='1',TASK_QUEUE_ENABLE='0')
    if args.phase!='stock':
        env['ASCEND_CUSTOM_OPP_PATH']=verified['vendor_root']
        env['LD_LIBRARY_PATH']=':'.join([str(Path(verified['vendor_root'])/'op_api/lib')]+[x for x in env.get('LD_LIBRARY_PATH','').split(':') if x])
    os.execve(sys.executable,[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--child'],env)
if __name__=='__main__':raise SystemExit(main())

