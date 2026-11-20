"""Source-bound fused three-axis mRoPE, ordinary task's existing four phases."""
from contextlib import nullcontext
from .serving_context import require
from .fused_mrope22_scope import FusedMrope22CallScope,reviewed_attention_structure
from .new_stack_request import NewStackTaskRequest
from .providers.rms18_task import Rms18TaskProvider
from .rms18_tp2 import TP2Rms18Provider
from .providers.tensor_modules import _capture,_identity
from .new_stack_rms18 import PROFILE

class _OneFusedMrope:
    def __init__(self,task):
        self.task=task;self.hooks=[];self.entry_hooks=[];self.scope=None;self.path=None;self.actual=None
    @property
    def owned_hook_count(self):
        return 0 if self.scope is None else len(self.scope.hooks)+len(self.scope.patches)
    def __enter__(self):
        task=self.task
        candidates=[(path,module) for path,module in task.model.named_modules()
            if reviewed_attention_structure(module)
            and 'qwen3_5' in module.config.model_type]
        require(bool(candidates),'FUSED22_NO_REVIEWED_ACTUAL_STRUCTURE')
        self.path,self.module=(candidates[0] if task.target is None else next(
            ((p,m) for p,m in candidates if p==task.target['path'] and m is task.target_module),(None,None)))
        require(self.module is not None,'FUSED22_HELD_TARGET_NOT_IN_MODEL')
        pos=task.session.root_arguments['positions'];expected=[list(range(len(task.binding.prompt)))]*3
        self.scope=FusedMrope22CallScope(self.module,pos,expected,arm=task.injection_arm,
            admit=self.admit,source_profile=task.binding.execution_profile,simulation=task.binding.simulation).install()
        return self
    def diagnostics(self):return nullcontext()
    def actual_inputs(self):
        s=self.scope;m=self.module
        require(s.projection is not None,'FUSED22_ACTUAL_PROJECTION_NOT_OBSERVED')
        return dict(module_object_id=id(m),cache_object_id=id(m.rotary_emb.cos_sin_cache),
            inputs=_identity(_capture((s.projection,s.positions,m.rotary_emb.cos_sin_cache,m.q_norm.weight,m.k_norm.weight))))
    def event(self):
        return dict(path=self.path,ordinal=0,operator=str(self.scope.op),schema=str(self.scope.op._schema))
    def admit(self):
        actual=self.actual_inputs();self.task.admit(self.event(),actual);return True
    def report(self):
        s=self.scope;r=s.result
        require(r is not None and r['source_bound_observation_verified'] and r['registered_implementation_observed']
            and r['fused_original_calls']==r['attention_consumer_calls']==1 and r['qkv_unchanged']
            and r['value_and_gate_preserved'] and r['non_target_rows_unchanged'],'FUSED22_COMPLETE_ACTUAL_NATIVE_CHAIN_REQUIRED')
        actual=self.actual_inputs();changed=False
        if self.task.normal is not None:
            normal=self.task.normal['mechanism']['events'][0]
            require(actual==normal['actual'],'FUSED22_MATCHED_PROJECTION_CACHE_POSITIONS_REQUIRED')
            b=normal['fused_reader']
            require(r['qkv_sha256']==b['qkv_sha256']
                and all(r[k]==b[k] for k in ('fused_q_non_target_sha256','fused_k_non_target_sha256')),
                'FUSED22_NON_TARGET_QK_CHANGED')
            changed=any(r[k]!=b[k] for k in ('fused_q_last_sha256','fused_k_last_sha256'))
            require(changed==bool(r['applied']),'FUSED22_LOCAL_QK_EFFECT_OR_GUARD_UNPROVEN')
        event=dict(self.event(),actual=actual,applied=r['applied'],original_execution_verified=True,
            return_readback_exact=True,output_changed=changed,fused_reader=r,
            consumers=[dict(operator='actual Attention.forward',returned=True,actual_return_lineage=True)],
            effect=dict(qk_last_row_changed=changed,non_target_qk_unchanged=True,
                changed_cache_elements=r['changed_cache_elements'],original_positions_unchanged=r['original_positions_unchanged']))
        return dict(events=[event],errors=[],forwards=[dict(index=0,status='returned')])
    def close(self):
        if self.scope:self.scope.close()
    def runtime_state(self):
        s=self.scope
        verified=s is None or all(vars(obj).get(name) is f for obj,name,f in s.patches)
        return dict(ownership_verified=verified,leases=int(s is not None and bool(s.patches)))

class _Fused22Mixin:
    mechanism_name='22.wrong_row';code_prefix='ROPE22'
    target_selector='FIRST_STRUCTURAL_FUSED_MROPE_CONFIRMED_BY_ACTUAL_CALL'
    task_scope='ONE_FUSED_THREE_AXIS_LEGAL_LAST_POSITION_FROM_FIRST_NOT_UINT32_OVERFLOW'
    def make_request(self):
        require(self.binding.execution_profile==PROFILE,'NEWSTACK_SOURCE_PROFILE_REQUIRED')
        return NewStackTaskRequest(self,_OneFusedMrope)
    def admit(self,event,actual):
        self._context();target=self.target;window=self.session.window
        require(self.injection_arm=='fault' and self.phase=='fault' and self.fault_started
            and not self.fault_applied and target is not None and window is not None,'FUSED22_LIVE_CURRENT_ROLE_REQUIRED')
        require(window.module is self.target_module and window.runtime_state()['ownership_verified']
            and (event['path'],event['ordinal'])==(target['path'],target['ordinal'])
            and actual==target['actual'] and event['operator']==target['operator']
            and event['schema']==target['native_schema'],'FUSED22_CURRENT_CALL_OR_SOURCE_CHANGED')
        # begin_actual checked the original callable before this request's
        # owned scope wrapper was installed. The scope independently pins the
        # wrapped original and registered implementation, and owns all patches.
        self.fault_applied=True
    def query(self,**kwargs):
        r=super().query(**kwargs)
        f=next((x for x in self.rows if x['phase']=='fault'),None)
        if f:
            effect=dict(f['mechanism']['events'][0]['effect'],request_id=f['request_id'],
                path=f['mechanism']['events'][0]['path'],fused_reader=f['mechanism']['events'][0]['fused_reader'])
            r['last_fault_effect']=effect;r['evidence']['last_fault_effect']=effect
        return r

class NewStackRope22Provider(_Fused22Mixin,Rms18TaskProvider):pass
class NewStackTP2Rope22Provider(_Fused22Mixin,TP2Rms18Provider):pass

def commit_newstack_rope22_role(worker,decision):
    lease=getattr(getattr(worker,'_demoncat_worker_fault_runtime',None),'_leases',{}).get('22.wrong_row')
    require(lease is not None and type(lease.provider) is NewStackTP2Rope22Provider,'NEWSTACK_TP2_RUNTIME_LEASE_REQUIRED')
    return lease.provider.commit_role(decision)
