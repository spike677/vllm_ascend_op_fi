"""TP2 KV41 ordinary role binding; Native cache consumer remains rank local."""
from copy import deepcopy
from .providers.kv41_task import KV41TaskProvider
from .serving_context import require
from .rms18_tp2 import TP2GroupContext,digest

class TP2KV41TaskProvider(KV41TaskProvider):
    def __init__(self,worker,model,identity):
        super().__init__(worker,model,identity)
        require(type(self.binding.group_context) is TP2GroupContext
            and self.binding.group_context.identity==identity,'TP2_KV41_CURRENT_GROUP_REQUIRED')
    def group_proof(self):
        c=self.binding.group_context;n=self.p.history[0] if self.p and self.p.history else None
        rows=None if n is None else n['actual_scheduler']
        return dict(owner=c.contract['owner'],group_generation=1,worker=deepcopy(self.identity),
            members_sha256=c.contract['members_sha256'],actual_normal_request_observed=n is not None,
            eligible_target=self.p is not None and self.p.target is not None,
            normal_request_id=None if n is None else n['request_ids'][0],
            request_fingerprint=None if rows is None else digest({k:rows[k] for k in ('prefixes','logical_ids','input_ids','row_token_positions')}),
            selected_rank=None if c.decision is None else c.decision['selected_rank'],
            role=None if c.decision is None else ('injector' if c.decision['selected_rank']==self.identity['rank'] else 'guard'))
    def commit_role(self,decision):
        self._context();c=self.binding.group_context;proof=self.group_proof()
        require(c.decision is None and self.phase=='fault' and self.pending is None and self.p.lossless
            and self.p.qualified and len(self.p.history)==3,'TP2_KV41_ROLE_AFTER_REAL_CALIBRATION')
        require(decision.get('group')==c.contract and decision.get('selected_rank') in (0,1)
            and decision.get('normal_request_id')==proof['normal_request_id']
            and decision.get('request_fingerprint')==proof['request_fingerprint']
            and {str(k):v for k,v in decision.get('roles',{}).items()}=={str(i):'injector' if i==decision['selected_rank'] else 'guard' for i in (0,1)},'TP2_KV41_CURRENT_NORMAL_ROLE_REQUIRED')
        c.decision=deepcopy(decision);return self.group_proof()
    def query(self,*,owner,generation):
        result=super().query(owner=owner,generation=generation);result['multi_rank']=self.group_proof();return result

def commit_kv41_tp2_role(worker,decision):
    runtime=getattr(worker,'_demoncat_worker_fault_runtime',None)
    lease=getattr(runtime,'_leases',{}).get('41.tail_column')
    require(lease is not None and type(lease.provider) is TP2KV41TaskProvider,'TP2_KV41_CURRENT_LEASE_REQUIRED')
    return lease.provider.commit_role(decision)
