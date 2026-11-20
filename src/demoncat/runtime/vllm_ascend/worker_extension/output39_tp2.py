"""Explicit TP2 output-proxy task; a guard never applies a fault write."""
from copy import deepcopy
from .rms18_tp2 import TP2GroupContext
from .multi_rank_prerequisites import tp2_eager_assessment,snapshot_tp2_fresh_prefill
from .serving_context import require,ContextRefused
from .output39_task import Output39TaskSession
from .providers.output39_task import Output39TaskProvider

class TP2Output39Session(Output39TaskSession):
    def __init__(self,provider):
        self.provider=provider
        super().__init__(provider)
        require(self.processor.use_all_gather is True,'TP2_GATHER_NULL_LOGITS_PATH_NOT_YET_ADAPTED')
    def _require_runner(self):
        c=self.provider.binding.group_context
        r=tp2_eager_assessment(self.runner,c.tp_group,self.provider.identity['rank'])
        if r['unmet']:
            e=ContextRefused(r['primary']['code']);e.prerequisites=r;raise e
        return r
    def _snapshot_prefill(self,actual,context):
        return snapshot_tp2_fresh_prefill(self.runner,self.model,actual,context,
            actual_tp_group=self.provider.binding.group_context.tp_group,rank=self.provider.identity['rank'],
            expected_prompt_token_ids=self.prompt,expected_internal_request_id=self.expected,
            seen_internal_request_ids=self.provider.requests)

class TP2Output39Provider(Output39TaskProvider):
    def __init__(self,worker,model,identity):
        super().__init__(worker,model,identity)
        require(type(self.binding.group_context) is TP2GroupContext and self.binding.group_context.identity==identity,'TP2_TYPED_GROUP_REQUIRED')
    def dispatch(self,action,settings,*,owner,generation):
        if action=='recover' and self.binding.group_context.decision is not None:
            # Another rank may have written although this rank is a guard.
            # All participants must perform fresh OFF/health, including a rank
            # that completed its local sequence before its peer failed.
            from .providers._scope import reject_unknown
            self.verify(owner,generation);reject_unknown(settings,());self._context()
            require(self.session is not None and self.p is not None,'TP2_REGISTERED_RECOVERY_REQUIRED')
            self.session.close_window();self.p.recover()
            self.recovering='off';self.stopped=False;self.recovered=False;self.complete=False;self.failure=None
            return self.query(owner=owner,generation=generation)
        return super().dispatch(action,settings,owner=owner,generation=generation)

    def make_session(self):return TP2Output39Session(self)
    @property
    def execution_phase(self):
        if self.phase!='fault':return self.phase
        c=self.binding.group_context
        require(c.decision is not None,'TP2_ROLE_DECISION_REQUIRED')
        return 'fault' if self.identity['rank']==c.decision['selected_rank'] else 'guard'
    def group_proof(self):
        c=self.binding.group_context;n=next((r for r in self.rows if r['phase']=='normal'),None)
        return dict(owner=c.contract['owner'],group_generation=c.contract['group_generation'],worker=deepcopy(self.identity),
            members_sha256=c.contract['members_sha256'],actual_normal_request_observed=n is not None,
            eligible_target=self.p is not None and self.p.normal is not None,
            normal_request_id=n['actual_internal_request_id'] if n else None,request_fingerprint=n['input_sha256'] if n else None,
            selected_rank=c.decision['selected_rank'] if c.decision else None,
            role=('injector' if self.identity['rank']==c.decision['selected_rank'] else 'guard') if c.decision else None)
    def commit_role(self,d):
        self._context();c=self.binding.group_context;p=self.group_proof()
        require(c.decision is None and self.phase=='fault' and self.p is not None and self.p.normal is not None
            and not self.p._fault_attempted and not self.p.guard_observed,'TP2_ROLE_COMMIT_ONCE_AFTER_NORMAL_REQUIRED')
        require(d.get('group')==c.contract and d.get('selected_rank') in (0,1)
            and {str(k):v for k,v in d.get('roles',{}).items()}=={str(r):'injector' if r==d['selected_rank'] else 'guard' for r in (0,1)}
            and d.get('normal_request_id')==p['normal_request_id'] and d.get('request_fingerprint')==p['request_fingerprint'],
            'TP2_ROLE_DECISION_NOT_BOUND_TO_ACTUAL_NORMAL')
        c.decision=deepcopy(d);self.p.group_guard=self.identity['rank']!=d['selected_rank']
        return self.group_proof()
    def end_actual(self,output,result,request):
        if self.phase=='fault' and self.execution_phase=='guard':
            require(result and result['phase']=='guard' and result['applied']==0 and result['guard_boundary_unchanged']
                and result['effect'] is None and result['native_lease_receipt'] is None
                and result['selected_return_sha256']==self.p.normal['before_sha256'] and result['consumers']
                and not self.p.live_lease_count,'OUTPUT39_GUARD_BOUNDARY_CHANGED')
        return super().end_actual(output,result,request)
    def query(self,*,owner,generation):
        r=super().query(owner=owner,generation=generation);r['multi_rank']=self.group_proof();r['complete']=self.complete;return r

def commit_output39_tp2_role(worker,decision):
    runtime=getattr(worker,'_demoncat_worker_fault_runtime',None)
    lease=getattr(runtime,'_leases',{}).get('39.output_proxy')
    require(lease is not None and type(lease.provider) is TP2Output39Provider,'TP2_OUTPUT39_RUNTIME_LEASE_REQUIRED')
    return lease.provider.commit_role(decision)
