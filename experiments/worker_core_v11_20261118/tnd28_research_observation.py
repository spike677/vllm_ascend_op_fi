"""Explicit research-only zero-F driver. Never registered in public Runtime/config.

Calls the retained provider directly in an already-created standard Worker.
It has its own research owner; its receipts are not public WorkerFaultClient RPC receipts.
"""
from copy import deepcopy
import uuid
from experiments.c2c3_closure_20261117.natural_attention_worker_core_probe import actual
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
MECHANISM='28.fia.denominator_replay'
ATTRIBUTE='_dcat28_zero_fault_research_owner'

def require(v,why):
    if not v:raise RuntimeError(why)

def command(wrapper,expected,owner,action,request_id=None):
    require(isinstance(owner,str) and len(owner)==32 and all(c in '0123456789abcdef' for c in owner),'NATIVE_RESEARCH_OWNER_HEX32_REQUIRED')
    worker=actual(wrapper)
    require(worker.demoncat_worker_identity()['worker']==expected,'RESEARCH_WORKER_CHANGED')
    runtime=getattr(worker,'_demoncat_worker_fault_runtime',None)
    require(not getattr(runtime,'_leases',{}) and not getattr(runtime,'_tenants',{}),'ORDINARY_RUNTIME_MUST_BE_UNOWNED')
    held=getattr(worker,ATTRIBUTE,None)
    if action=='prepare':
        require(held is None,'RESEARCH_OWNER_ALREADY_HELD')
        from demoncat.runtime.vllm_ascend.worker_extension.registry import create_research_provider
        provider=create_research_provider(MECHANISM,'1',worker,worker.model_runner.model,expected)
        held=dict(owner=owner,provider=provider);setattr(worker,ATTRIBUTE,held)
    require(held is not None and held['owner']==owner,'RESEARCH_OWNER_MISMATCH')
    p=held['provider']
    if action=='detach':
        state=p.query(owner=owner,generation=1)
        require(not state['registered'] and not any(state[k] for k in ('owned_hooks','owned_api_wrappers','inflight','leases')),'RESEARCH_RESOURCES_REMAIN')
        delattr(worker,ATTRIBUTE);return dict(status='DETACHED',provider_state=state)
    if action=='query':state=p.query(owner=owner,generation=1)
    elif action=='collect_replay':state=p.dispatch('advance',{'collect':True},owner=owner,generation=1)
    else:
        require(action in ('prepare','register','check','enable','advance','disable','unregister'),'ZERO_F_ACTION_ONLY')
        if action=='advance':
            require(p.index==0 and p.phases[0]=='observe' and request_id,'EXACTLY_ONE_OBSERVE_REQUEST')
        state=p.dispatch(action,dict(request_id=request_id) if action=='advance' else {},owner=owner,generation=1)
    return dict(status='QUERY_OK' if action=='query' else 'ACTION_COMPLETED',provider_state=state)

class ResearchObservationClient:
    def __init__(self,llm,prepare_binding):
        self.llm=llm;self.prepare_binding=prepare_binding;self.owner=uuid.uuid4().hex
        self.worker=None;self.history=[];self.owned=False;self.prepared=False
    def _rpc(self,action,request_id=None):
        if action=='prepare':
            rows=self.llm.collective_rpc('demoncat_worker_identity')
            require(len(rows)==1 and rows[0]['status']=='WORKER_IDENTIFIED' and not rows[0]['quarantined'],'ONE_HEALTHY_WORKER')
            self.worker=rows[0]['worker'];self.prepare_binding(self.worker,FaultConfig(28,'fia.denominator_replay'))
            self.owned=True
        rows=self.llm.collective_rpc(command,args=(self.worker,self.owner,action,request_id),timeout=300)
        require(len(rows)==1,'ONE_RESEARCH_WORKER');row=rows[0]
        if action=='prepare':self.prepared=True
        if action=='detach':self.owned=False
        receipt=dict(action=action,status=row['status'],response=dict(provider_state=row['provider_state']),research_direct_provider=True)
        self.history.append(receipt)
        return dict(receipt,history=deepcopy(self.history))
    def snapshot(self):return dict(worker=self.worker,owner=self.owner,ticket=dict(owner=self.owner) if self.prepared else None,ownership_retained=self.owned,history=deepcopy(self.history),protocol='EXPLICIT_RESEARCH_PROVIDER_NOT_PUBLIC_RUNTIME')
    def prepare(self):return self._rpc('prepare')
    def register(self):return self._rpc('register')
    def check(self):return self._rpc('check')
    def enable(self):return self._rpc('enable')
    def advance(self,request_id):return self._rpc('advance',request_id)
    def query(self):return self._rpc('query')
    def collect_replay(self):return self._rpc('collect_replay')
    def disable(self):return self._rpc('disable')
    def unregister(self):return self._rpc('unregister')
    def detach(self):return self._rpc('detach')
