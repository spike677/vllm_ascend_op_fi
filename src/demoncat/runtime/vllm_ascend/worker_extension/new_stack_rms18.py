"""Trusted 0.26 text-only RMS18 binding; acceptance permit remains mandatory.
No user layer/rank selection, automatic deployment grant or model execution.
"""
import json,os,threading
from copy import deepcopy
from .serving_context import require
from .serving_tensor_binding import TensorServingBinding,_ATTRIBUTE
from .v026_text_observer import SourceBoundTextObserver
from .rms18_tp2 import TP2GroupContext,TP2Rms18Provider,digest
from .providers.rms18_task import Rms18TaskProvider
from .rms18_request import _OneRMSReturn
from .new_stack_request import NewStackTaskRequest

PROFILE='v026_installed_c9c9432b'

def _install_newstack_tensor_binding(worker,mechanism,*,prompt_token_ids,weight_manifest,
        workload_sha256,environment_sha256,group_contract=None):
    require(mechanism in ('18.api_half','22.wrong_row'),'NEWSTACK_MECHANISM_UNREVIEWED')
    require(isinstance(prompt_token_ids,(list,tuple)) and 2<=len(prompt_token_ids)<=4096
        and all(type(t) is int and t>=0 for t in prompt_token_ids),'BOUNDED_FRESH_PREFILL_PROMPT_REQUIRED')
    require(isinstance(weight_manifest,dict) and bool(weight_manifest),'WEIGHT_MANIFEST_REQUIRED')
    require(all(isinstance(x,str) and len(x)==64 and all(c in '0123456789abcdef' for c in x)
        for x in (workload_sha256,environment_sha256)),'REVIEWED_SCOPE_HASHES_REQUIRED')
    runner=worker.model_runner;model=runner.model
    require(not model.training and runner.supports_mm_inputs is False,'NEWSTACK_REVIEWED_TOKEN_ONLY_MODEL_REQUIRED')
    validation=SourceBoundTextObserver(runner,{'binding-pending':list(prompt_token_ids)},source_profile=PROFILE).validate()
    require(validation.snapshot()['owned_hooks']==validation.snapshot()['owned_wrappers']==0,'NEWSTACK_BINDING_MUST_BE_INERT')
    from vllm.distributed.parallel_state import get_tp_group
    group=get_tp_group();tp=group.world_size
    require(tp in (1,2),'NEWSTACK_TP1_TP2_ONLY')
    context=None
    if tp==2:
        identified=worker.demoncat_worker_identity();require(identified['status']=='WORKER_IDENTIFIED','TP2_CURRENT_WORKER_REQUIRED')
        identity=identified['worker'];rank=identity['rank'];c=deepcopy(group_contract)
        require(isinstance(c,dict) and c.get('mechanism')==mechanism and c.get('tp')==2
            and c.get('injection_policy')=='ONE_AUTOMATIC_RANK' and c.get('group_generation')==1
            and isinstance(c.get('owner'),str) and c['owner'].startswith('worker-group-')
            and isinstance(c.get('members'),list) and [m['rank'] for m in c['members']]==[0,1]
            and c['members'][rank]==identity and c.get('members_sha256')==digest(c['members']),
            'TP2_GROUP_CURRENT_MEMBERS_REQUIRED')
        context=TP2GroupContext(c,identity,group)
    else:require(group_contract is None,'TP1_NO_GROUP_CONTRACT')
    bindings=getattr(worker,_ATTRIBUTE,{})
    require(type(bindings) is dict and mechanism not in bindings,'SERVING_DEPLOYMENT_ALREADY_BOUND')
    binding=TensorServingBinding(mechanism,worker,runner,model,tuple(prompt_token_ids),
        json.dumps(weight_manifest,sort_keys=True,allow_nan=False),workload_sha256,environment_sha256,
        os.getpid(),threading.get_ident(),runtime_only=True,group_context=context,execution_profile=PROFILE)
    bindings[mechanism]=binding;setattr(worker,_ATTRIBUTE,bindings)
    return dict(status='NEWSTACK_TENSOR_BOUND_NOT_QUALIFIED',mechanism=mechanism,hooks_installed=0,qualification_granted=False,
        execution_profile=PROFILE,tp=tp,source_readback=validation.snapshot(),request_plan='EXISTING_FOUR_PHASE_PLAN')

def install_rms18_newstack_binding(worker,**kwargs):
    return _install_newstack_tensor_binding(worker,'18.api_half',**kwargs)

def install_rope22_newstack_binding(worker,**kwargs):
    return _install_newstack_tensor_binding(worker,'22.wrong_row',**kwargs)

class NewStackRms18Provider(Rms18TaskProvider):
    def make_request(self):
        require(self.binding.execution_profile==PROFILE,'NEWSTACK_SOURCE_PROFILE_REQUIRED')
        return NewStackTaskRequest(self,_OneRMSReturn)

class NewStackTP2Rms18Provider(TP2Rms18Provider):
    def make_request(self):
        require(self.binding.execution_profile==PROFILE,'NEWSTACK_SOURCE_PROFILE_REQUIRED')
        return NewStackTaskRequest(self,_OneRMSReturn)

def commit_newstack_rms18_role(worker,decision):
    lease=getattr(getattr(worker,'_demoncat_worker_fault_runtime',None),'_leases',{}).get('18.api_half')
    require(lease is not None and type(lease.provider) is NewStackTP2Rms18Provider,'NEWSTACK_TP2_RUNTIME_LEASE_REQUIRED')
    return lease.provider.commit_role(decision)
