"""Generate the missing FIA binary summary index from this fresh build.

Uses the pinned official source function in memory, without its CLI (which
rewrites metadata). Neither old vendor files nor old binaries are inputs.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

COMMIT='c77c0829f1f0f0b8f882db1d9b051a758b605472'
BINARY='FusedInferAttentionScore_8cd36e66e4ceb2d60ef96db29a147347'
SCRIPT_SHA='f8d66eea73466a6e9b82e95af677130d6f26221fbfbb4b637042f710ead1022a'
CONSTANTS_SHA='447a9cc6d794f4473bc59b6b464a7b03aa951e436b8a09c880cd5b44c0873fef'
KEYS={'legal_tiling':{103000000000000000,109000000000000000,109000000000400000},
      'denominator_replay':{103000000000000000,109000000000000000,109000000000400000,119000000000400000},
      'tnd_page16_s2_256':{103000000010200003,103000000010300003,104000000010200003,104000000010300003},
      'tnd_page16_fai_s2_256':{5000000000000200100,5000000000000200103,
                             5000000000010200100,5000000000010200103,
                             5900000000010200100,5900000000010200103},
      'tnd_page16_fai_fd_s2_256':{5000000000000200100,5000000000000200103,
                                5000000000010200100,5000000000010200103,
                                5100000000010200100,5100000000010200103,
                                6000000000010200100,6000000000010200103}}
KEYS['tnd_page16_arch22_normal_bridge'] = KEYS['tnd_page16_s2_256'] | {
    5000000000000200100, 5000000000000200103, 5000000000010200100,
    5000000000010200103, 5100000000010200100, 5100000000010200103}

def require(ok,message):
    if not ok:raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4*1024**2),b''):h.update(part)
    return h.hexdigest()

def load_official(source):
    folder=Path(source)/'cmake/scripts/util'
    script=folder/'ascendc_ops_config.py';constants=folder/'const_var.py'
    require(sha(script)==SCRIPT_SHA and sha(constants)==CONSTANTS_SHA,'OFFICIAL_GENERATOR_SOURCE_DIFFERS')
    old_path=list(sys.path);old_const=sys.modules.pop('const_var',None);old_write=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True
        sys.path.insert(0,str(folder))
        spec=importlib.util.spec_from_file_location('dcat_fia_official_index_generator',script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        require(Path(module.const_var.__file__).resolve()==constants.resolve(),'OFFICIAL_CONSTANTS_RESOLVED_ELSEWHERE')
        return module
    finally:
        sys.path[:]=old_path;sys.dont_write_bytecode=old_write;sys.modules.pop('const_var',None)
        if old_const is not None:sys.modules['const_var']=old_const

def validate_metadata(metadata,binary,variant):
    require(variant in KEYS,'UNREVIEWED_VARIANT')
    require(metadata.get('binFileName')==BINARY and metadata.get('binFileSuffix')=='.o'
        and metadata.get('sha256')==sha(binary),'NEW_BINARY_METADATA_IDENTITY_MISMATCH')
    actual=metadata.get('kernelList',[])
    require(len(actual)==len(KEYS[variant]) and {int(r['tilingKey']) for r in actual}==KEYS[variant],
            'NEW_MATCHING_KEY_SET_DIFFERS')
    require(metadata.get('coreType')=='MIX' and metadata.get('taskRation')=='tilingKey','NEW_KERNEL_KIND_DIFFERS')
    support=metadata.get('supportInfo',{})
    require(support.get('simplifiedKeyMode')==0 and support.get('simplifiedKey')
        and support.get('optionalInputMode')=='gen_placeholder'
        and support.get('optionalOutputMode')=='gen_placeholder','NEW_KERNEL_SUPPORT_CONTRACT_DIFFERS')
    require(len(support.get('tilingKey',[]))==len(KEYS[variant])
        and {int(x) for x in support['tilingKey']}==KEYS[variant],'SUPPORT_KEY_SET_DIFFERS')

def validate_generated(generated,existing,metadata):
    require(set(generated)=={'binary_info_config.json','fused_infer_attention_score.json'},'UNEXPECTED_GENERATED_REGISTRY')
    require(generated['fused_infer_attention_score.json']==existing,'EXISTING_REGISTRY_AND_NEW_METADATA_DISAGREE')
    index=generated['binary_info_config.json']
    require(set(index)=={'FusedInferAttentionScore'},'GENERATED_INDEX_OPERATOR_DIFFERS')
    op=index['FusedInferAttentionScore'];rows=op.get('binaryList',[])
    base='ascend910b/fused_infer_attention_score/'+BINARY
    require(len(rows)==1 and rows[0].get('binPath')==base+'.o' and rows[0].get('jsonPath')==base+'.json'
        and rows[0].get('coreType')==0 and rows[0].get('multiKernelType')==1
        and rows[0].get('simplifiedKey')==metadata['supportInfo']['simplifiedKey'],'GENERATED_INDEX_BINARY_DISAGREES')
    require(op.get('simplifiedKeyMode')==0 and op.get('optionalInputMode')=='gen_placeholder'
        and op.get('optionalOutputMode')=='gen_placeholder','GENERATED_INDEX_MODE_DIFFERS')
    return index

def register_index(source,vendor,variant):
    require(os.name=='posix','LINUX_OFFICIAL_REGISTRATION_REQUIRED')
    vendor=Path(vendor).resolve(strict=True)
    root=vendor/'op_impl/ai_core/tbe/kernel'
    metadata_path=root/('ascend910b/fused_infer_attention_score/'+BINARY+'.json')
    binary=metadata_path.with_suffix('.o')
    registry=root/'config/ascend910b/fused_infer_attention_score.json'
    output=registry.with_name('binary_info_config.json')
    require(not output.exists() and not output.is_symlink(),'SUMMARY_INDEX_ALREADY_PRESENT')
    protected={p:sha(p) for p in (metadata_path,binary,registry)}
    metadata=json.loads(metadata_path.read_bytes());validate_metadata(metadata,binary,variant)
    generated={'binary_info_config.json':{}}
    official=load_official(source)
    official.gen_ops_config(str(metadata_path),'ascend910b','binary_info_config.json',generated)
    index=validate_generated(generated,json.loads(registry.read_bytes()),metadata)
    require(all(sha(path)==digest for path,digest in protected.items()),'SOURCE_BYTES_CHANGED_DURING_GENERATION')
    with output.open('x',encoding='utf-8') as stream:
        json.dump(index,stream,indent=2,allow_nan=False);stream.write('\n')
    return dict(schema='demoncat.fia.binary_summary_registration.v1',status='REGISTERED_REQUIRES_NPU_VALIDATION',
        variant=variant,generated_index=dict(path=output.relative_to(vendor).as_posix(),sha256=sha(output),bytes=output.stat().st_size),
        generator=dict(source='cmake/scripts/util/ascendc_ops_config.py',sha256=SCRIPT_SHA,
                       function='gen_ops_config',constants_sha256=CONSTANTS_SHA),
        helper_sha256=sha(Path(__file__)),metadata_sha256=protected[metadata_path],
        binary_sha256=protected[binary],existing_registry_sha256=protected[registry],
        generated_registry_matches_existing=True,host_kernel_bytes_changed=False,
        old_index_read=False,old_binaries_used=False,npu_executed=False,
        actual_launch_verified=False,production_loading_allowed=False)

