"""Reviewed deployment acceptance, separate from structural/live-call admission.

Only a trusted blue-zone harness may install a scoped experimental permit.
FaultConfig has no switch for this. A permit never creates an accepted record
or bypasses a mechanism's actual source, ABI, scheduler or call checks.
"""
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
import hashlib
import json
import os
import sys
import threading

from .prerequisites import expectation, assessment

_BLUE = '_demoncat_blue_acceptance_permit'
# 8/17 a5 reviewed: two independent scopes, actual four-request recovery.
_PENDING_REVALIDATION={}
_COMMON = dict(torch='2.9.0+cpu', torch_npu='2.9.0.post1+gitee7ba04',
    vllm='0.18.0', vllm_ascend='0.18.0', cann=['8.5.1'], dtype='torch.bfloat16',
    tp=1, pp=1, dp=1, eager=True)
# These are accepted *combinations*, not structural support rules. Exact model
# config identity prevents a model-name alias granting a new model acceptance.
_CONFIGS = {
    '8.restore_offset': ('229913ea2a346ccdb571bb3cb23414ca3a0ee1a9455fe88a15a5788bc837cb75',
        'vllm.model_executor.models.deepseek_v2.DeepseekV2ForCausalLM', '16f40f1f/ordinary8-a5'),
    '17.spatial_padding': ('bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b',
        'vllm.model_executor.models.qwen3_vl.Qwen3VLForConditionalGeneration', '16f40f1f/ordinary17-a5'),
    '22.wrong_row': ('660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd',
        'vllm.model_executor.models.qwen3.Qwen3ForCausalLM', '4c56701f/ordinary22-a1'),
    '18.api_half': ('660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd',
        'vllm.model_executor.models.qwen3.Qwen3ForCausalLM', 'Source126'),
    '39.output_proxy': ('660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd',
        'vllm.model_executor.models.qwen3.Qwen3ForCausalLM', 'Source116'),
    'quant.batch_shared_scale': ('bf644c8607476632c4ece502205dbd5b4793723025471c375fe9626688a19816',
        'vllm.model_executor.models.qwen3.Qwen3ForCausalLM', 'Source118'),
}


def _get(root,path,default=None):
    for part in path.split('.'):
        root=getattr(root,part,default)
        if root is default:return default
    return root


def observe_deployment(worker,model):
    """Read existing objects/files only; no package import, model or NPU call."""
    runner=getattr(worker,'model_runner',None)
    config=_get(runner,'vllm_config.model_config')
    actual={}
    for name in ('torch','torch_npu','vllm'):
        version=getattr(sys.modules.get(name),'__version__',None)
        actual[name]=str(version) if isinstance(version,str) else None
    try:actual['vllm_ascend']=metadata.version('vllm-ascend')
    except metadata.PackageNotFoundError:actual['vllm_ascend']=None
    from fault_injection.environment import _toolkit
    toolkit=_toolkit()
    actual['cann']=sorted({item['fields'].get('version') for item in toolkit['installations']
        if item['fields'].get('version')})
    actual['dtype']=str(getattr(config,'dtype',None))
    for key,name in (('tp','tensor'),('pp','pipeline'),('dp','data')):
        actual[key]=_get(runner,'vllm_config.parallel_config.'+name+'_parallel_size')
    actual['eager']=getattr(config,'enforce_eager',None)
    actual['model_class']=type(model).__module__+'.'+type(model).__qualname__
    actual['model_config_sha256']=None
    actual['live_config_matches_file']=False
    path=getattr(config,'model',None)
    if isinstance(path,(str,os.PathLike)):
        try:
            raw=(Path(path)/'config.json').read_bytes()
            disk=json.loads(raw)
            actual['model_config_sha256']=hashlib.sha256(raw).hexdigest()
            live=getattr(config,'hf_config',None)
            def matches(d,obj):
                keys=('model_type','architectures','hidden_size','num_hidden_layers','num_attention_heads',
                      'num_key_value_heads','head_dim','vocab_size','n_routed_experts','num_experts_per_tok',
                      'depth','patch_size','temporal_patch_size','in_channels','spatial_merge_size')
                return all(getattr(obj,k,None)==d[k] for k in keys if k in d) and all(
                    matches(d[k],getattr(obj,k,None)) for k in ('text_config','vision_config') if k in d)
            actual['live_config_matches_file']=matches(disk,live)
        except (OSError,ValueError,TypeError):pass
    return actual


@dataclass(frozen=True)
class BlueAcceptancePermit:
    worker: object
    model: object
    mechanisms: tuple
    attempt: str
    source_manifest_sha256: str
    pid: int
    thread: int


def install_blue_acceptance_permit(worker,*,mechanisms,attempt,source_manifest_sha256):
    """Trusted deployment harness only. Never an option in public FaultConfig."""
    if hasattr(worker,_BLUE):raise RuntimeError('BLUE_ACCEPTANCE_PERMIT_ALREADY_BOUND')
    if (not isinstance(mechanisms,(tuple,list)) or not mechanisms or len(mechanisms)>17
        or any(type(x) is not str for x in mechanisms) or len(set(mechanisms))!=len(mechanisms)
        or not isinstance(attempt,str) or not 1<=len(attempt)<=128
        or not isinstance(source_manifest_sha256,str) or len(source_manifest_sha256)!=64
        or any(c not in '0123456789abcdef' for c in source_manifest_sha256)):
        raise ValueError('EXPLICIT_BOUNDED_BLUE_ACCEPTANCE_SCOPE_REQUIRED')
    model=worker.model_runner.model
    permit=BlueAcceptancePermit(worker,model,tuple(mechanisms),attempt,source_manifest_sha256,
        os.getpid(),threading.get_ident())
    setattr(worker,_BLUE,permit)
    return dict(mode='BLUE_ACCEPTANCE_ONLY',attempt=attempt,acceptance_granted=False)


def remove_blue_acceptance_permit(worker):
    permit=getattr(worker,_BLUE,None)
    if type(permit) is not BlueAcceptancePermit or permit.worker is not worker:
        raise RuntimeError('BLUE_ACCEPTANCE_PERMIT_NOT_BOUND')
    runtime=getattr(worker,'_demoncat_worker_fault_runtime',None)
    if any(k in getattr(runtime,'_leases',{}) for k in permit.mechanisms):
        raise RuntimeError('BLUE_ACCEPTANCE_OWNER_CLEAN_REQUIRED')
    delattr(worker,_BLUE)


def _cpu_fixture(worker,model,mechanism):
    bindings=getattr(worker,'_demoncat_natural_tensor_serving_bindings',{})
    b=bindings.get(mechanism) if type(bindings) is dict else None
    if mechanism=='17.spatial_padding':b=getattr(worker,'_demoncat_vision17_task_binding',None)
    if mechanism=='39.output_proxy':b=getattr(worker,'_demoncat_output39_serving_binding',None)
    if b is None or getattr(b,'simulation',None) is not True:return False
    tensors=list(model.parameters())+list(model.buffers())
    return bool(tensors) and all(t.device.type=='cpu' for t in tensors)


def deployment_acceptance_report(mechanism,worker=None,model=None,*,implemented=True):
    record=_CONFIGS.get(mechanism)
    actual=observe_deployment(worker,model) if worker is not None and model is not None else None
    expected=dict(_COMMON,model_config_sha256=record[0],model_class=record[1],live_config_matches_file=True) if record else None
    checks=([expectation('deployment_identity','accepted_combination.'+k,v,
        (actual or {}).get(k), 'ORDINARY_ROUTE_NOT_ACCEPTED_FOR_DEPLOYMENT') for k,v in expected.items()]
        if expected else [])
    accepted=bool(expected and actual and mechanism not in _PENDING_REVALIDATION and all(c['status']=='matched' for c in checks))
    permit=getattr(worker,_BLUE,None)
    blue=(type(permit) is BlueAcceptancePermit and permit.worker is worker and permit.model is model
        and worker.model_runner.model is model and mechanism in permit.mechanisms
        and permit.pid==os.getpid() and permit.thread==threading.get_ident())
    cpu=worker is not None and model is not None and _cpu_fixture(worker,model,mechanism)
    status=('NOT_IMPLEMENTED' if not implemented else 'ACCEPTED_DEPLOYMENT' if accepted
        else 'IMPLEMENTED_NOT_ACCEPTED_FOR_DEPLOYMENT')
    return dict(schema='demoncat.deployment_acceptance.v1',status=status,accepted=accepted,
        can_prepare=bool(implemented and (accepted or blue or cpu)),
        mode='CPU_FIXTURE_ONLY' if cpu else 'BLUE_ACCEPTANCE_ONLY' if blue else 'DEFAULT',
        explicit_blue_attempt=permit.attempt if blue else None,
        reviewed_combination=record[2] if record else None,
        pending_revalidation=_PENDING_REVALIDATION.get(mechanism),
        expected=expected,actual=actual,checks=checks,acceptance_granted=False,
        structural_compatibility='CHECKED_SEPARATELY_NOT_AN_ACCEPTANCE_CERTIFICATE',
        live_call_qualification='PENDING')
