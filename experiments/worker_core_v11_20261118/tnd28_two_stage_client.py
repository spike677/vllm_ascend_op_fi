"""Direct research client: original standard Worker, real B/F/OFF phases."""
from copy import deepcopy
import uuid
from experiments.c2c3_closure_20261117.natural_attention_worker_core_probe import actual
from experiments.worker_core_v11_20261118.tnd28_two_stage_research import TwoStageResearchProvider
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require
ATTRIBUTE='_dcat28_two_stage_research_owner'


def command(wrapper,expected,owner,action,request_id=None):
    require(type(owner) is str and len(owner)==32 and all(x in '0123456789abcdef' for x in owner),'RESEARCH_OWNER_ID')
    worker=actual(wrapper);require(worker.demoncat_worker_identity()['worker']==expected,'RESEARCH_WORKER_CHANGED')
    runtime=getattr(worker,'_demoncat_worker_fault_runtime',None)
    require(not getattr(runtime,'_leases',{}) and not getattr(runtime,'_tenants',{}),'ORDINARY_RUNTIME_OWNED')
    held=getattr(worker,ATTRIBUTE,None)
    if action=='prepare':
        require(held is None,'RESEARCH_OWNER_HELD')
        held=dict(owner=owner,provider=TwoStageResearchProvider(worker,worker.model_runner.model,expected));setattr(worker,ATTRIBUTE,held)
    require(held is not None and held['owner']==owner,'RESEARCH_OWNER_CHANGED');p=held['provider']
    if action=='detach':
        state=p.query(owner=owner,generation=1)
        require(not state['registered'] and not state['binding_reserved']
            and not any(state[k] for k in ('owned_hooks','owned_api_wrappers','inflight','leases')),'RESEARCH_RESOURCES_REMAIN')
        delattr(worker,ATTRIBUTE);return dict(status='DETACHED',provider_state=state)
    if action=='query':state=p.query(owner=owner,generation=1)
    elif action=='collect':state=p.dispatch('advance',{'collect':True},owner=owner,generation=1)
    else:
        require(action in ('prepare','register','check','enable','advance','finish','disable','unregister'),'RESEARCH_UNKNOWN_ACTION')
        state=p.dispatch(action,dict(request_id=request_id) if action=='advance' else {},owner=owner,generation=1)
    return dict(status='QUERY_OK' if action=='query' else 'ACTION_COMPLETED',provider_state=state)


class TwoStageResearchClient:
    def __init__(self,llm,worker):
        self.llm=llm;self.worker=worker;self.owner=uuid.uuid4().hex;self.history=[];self.owned=False
    def call(self,action,request_id=None):
        if action=='prepare':self.owned=True
        rows=self.llm.collective_rpc(command,args=(self.worker,self.owner,action,request_id),timeout=300)
        require(type(rows) is list and len(rows)==1,'ONE_RESEARCH_WORKER')
        row=rows[0];self.history.append(dict(action=action,request_id=request_id,response=deepcopy(row)))
        if action=='detach':self.owned=False
        return row['provider_state']
    def snapshot(self):
        return dict(protocol='EXPLICIT_TWO_STAGE_RESEARCH_NOT_PUBLIC_RUNTIME',worker=self.worker,owner=self.owner,
                    ownership_retained=self.owned,history=deepcopy(self.history))
