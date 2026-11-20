"""Development TP2 RMS18 binding: actual normal candidate, one injector, one guard.

Only explicit deployment acceptance permits may exercise this unregistered
combination. Current session scope requires the observed all-gather logits path.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib,json,os,threading
from .serving_tensor_binding import TensorServingBinding, _ATTRIBUTE
from .serving_context import require,ContextRefused
from .multi_rank_prerequisites import tp2_eager_assessment,snapshot_tp2_fresh_prefill
from .providers.rms18_task import Rms18TaskProvider
from .rms18_request import Rms18TaskRequest
from fault_injection.request_plan import FOUR


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@dataclass
class TP2GroupContext:
    contract: dict
    identity: dict
    tp_group: object
    decision: object = None


def validate_tp2_group(worker, mechanism, group_contract, *, expected_max_seqs=1):
    from vllm.distributed import get_tp_group
    actual=worker.demoncat_worker_identity()
    require(actual['status']=='WORKER_IDENTIFIED','TP2_CURRENT_WORKER_REQUIRED')
    identity=actual['worker'];rank=identity['rank'];group=get_tp_group();runner=worker.model_runner
    report=tp2_eager_assessment(runner,group,rank,expected_max_seqs=expected_max_seqs)
    require(not report['unmet'],'TP2_PREREQUISITES:'+json.dumps(report,sort_keys=True))
    c=deepcopy(group_contract)
    require(c.get('mechanism')==mechanism and c.get('tp')==2
        and c.get('injection_policy')=='ONE_AUTOMATIC_RANK' and c.get('group_generation')==1
        and isinstance(c.get('owner'),str) and c['owner'].startswith('worker-group-')
        and isinstance(c.get('members'),list) and [m['rank'] for m in c['members']]==[0,1]
        and c['members'][rank]==identity and c.get('members_sha256')==digest(c['members']),
        'TP2_GROUP_CURRENT_MEMBERS_REQUIRED')
    return TP2GroupContext(c,identity,group)


def _install_tp2_tensor_binding(worker, mechanism, *, group_contract, prompt_token_ids, weight_manifest,
                              workload_sha256, environment_sha256):
    require(mechanism in ('18.api_half','22.wrong_row'),'TP2_TENSOR_ADAPTER_REQUIRED')
    context=validate_tp2_group(worker,mechanism,group_contract)
    runner=worker.model_runner;rank=context.identity['rank']
    require(isinstance(prompt_token_ids,(list,tuple)) and 2<=len(prompt_token_ids)<=4096
        and all(type(x) is int and x>=0 for x in prompt_token_ids),'BOUNDED_FRESH_PREFILL_PROMPT_REQUIRED')
    require(isinstance(weight_manifest,dict) and bool(weight_manifest),'WEIGHT_MANIFEST_REQUIRED')
    require(all(isinstance(x,str) and len(x)==64 and all(c in '0123456789abcdef' for c in x)
        for x in (workload_sha256,environment_sha256)),'REVIEWED_SCOPE_HASHES_REQUIRED')
    require(not runner.model.training,'INFERENCE_MODEL_REQUIRED')
    bindings=getattr(worker,_ATTRIBUTE,{})
    require(type(bindings) is dict and mechanism not in bindings,'SERVING_DEPLOYMENT_ALREADY_BOUND')
    binding=TensorServingBinding(mechanism,worker,runner,runner.model,tuple(prompt_token_ids),
        json.dumps(weight_manifest,sort_keys=True,allow_nan=False),workload_sha256,environment_sha256,
        os.getpid(),threading.get_ident(),runtime_only=True,group_context=context)
    bindings[mechanism]=binding;setattr(worker,_ATTRIBUTE,bindings)
    return dict(status='TP2_BINDING_NOT_QUALIFIED',rank=rank,hooks_installed=0,qualification_granted=False)


def install_rms18_tp2_binding(worker, **kwargs):
    return _install_tp2_tensor_binding(worker,'18.api_half',**kwargs)


class TP2Rms18Request(Rms18TaskRequest):
    def __init__(self,provider):
        super().__init__(provider)
        require(self.processor.use_all_gather is True,'TP2_GATHER_NULL_LOGITS_PATH_NOT_YET_ADAPTED')

    def _require_runner(self):
        ctx=self.provider.binding.group_context
        report=tp2_eager_assessment(self.runner,ctx.tp_group,self.provider.identity['rank'])
        if report['unmet']:
            error=ContextRefused(report['primary']['code']);error.prerequisites=report;raise error
        return report

    def _snapshot_prefill(self,actual,context):
        return snapshot_tp2_fresh_prefill(self.runner,self.model,actual,context,
            actual_tp_group=self.provider.binding.group_context.tp_group,rank=self.provider.identity['rank'],
            expected_prompt_token_ids=self.prompt,expected_internal_request_id=self.expected,
            seen_internal_request_ids=self.provider.requests)


class TP2Rms18Provider(Rms18TaskProvider):
    def __init__(self,worker,model,identity):
        super().__init__(worker,model,identity)
        self.require(type(self.binding.group_context) is TP2GroupContext,'TP2_TYPED_GROUP_REQUIRED')
        self.require(self.binding.group_context.identity==identity,'TP2_HELD_IDENTITY_CHANGED')

    def dispatch(self,action,settings,*,owner,generation):
        if action=='recover' and self.binding.group_context.decision is not None:
            from .providers._scope import reject_unknown
            self.verify(owner,generation);reject_unknown(settings,());self._context()
            self.require(self.session is not None,'TP2_REGISTERED_RECOVERY_REQUIRED')
            self.session.close_window()
            self.recovering='off';self.stopped=False;self.recovered=False;self.complete=False
            self.error=self.session.request_error=None
            return self.query(owner=owner,generation=generation)
        return super().dispatch(action,settings,owner=owner,generation=generation)

    def make_request(self):return TP2Rms18Request(self)

    @property
    def injection_arm(self):
        if self.phase!='fault':return 'baseline'
        c=self.binding.group_context
        self.require(c.decision is not None,'TP2_ROLE_DECISION_REQUIRED')
        return 'fault' if self.identity['rank']==c.decision['selected_rank'] else 'baseline'

    def commit_role(self,decision):
        c=self.binding.group_context
        self._context()
        self.require(c.decision is None and self.normal is not None and not self.fault_started
            and self.phase=='fault','TP2_ROLE_COMMIT_ONCE_AFTER_NORMAL_REQUIRED')
        proof=self.group_proof()
        self.require(decision.get('group')==c.contract and decision.get('selected_rank') in (0,1)
            and (self.mechanism_name!='18.api_half' or decision['selected_rank']==0)
            and {str(k):v for k,v in decision.get('roles',{}).items()}=={str(r):'injector' if r==decision['selected_rank'] else 'guard' for r in (0,1)}
            and decision.get('normal_request_id')==proof['normal_request_id']
            and decision.get('request_fingerprint')==proof['request_fingerprint'],
            'TP2_ROLE_DECISION_NOT_BOUND_TO_ACTUAL_NORMAL')
        c.decision=deepcopy(decision)
        return self.group_proof()

    def group_proof(self):
        c=self.binding.group_context;normal=self.normal;rank=self.identity['rank']
        return dict(owner=c.contract['owner'],group_generation=c.contract['group_generation'],
            worker=deepcopy(self.identity),members_sha256=c.contract['members_sha256'],
            actual_normal_request_observed=normal is not None,eligible_target=self.target is not None,
            normal_request_id=normal['request_id'] if normal else None,
            request_fingerprint=normal['input_sha256'] if normal else None,
            selected_rank=None if c.decision is None else c.decision['selected_rank'],
            role=None if c.decision is None else 'injector' if rank==c.decision['selected_rank'] else 'guard')

    def admit(self,event,actual):
        self.require(self.injection_arm=='fault' and self.identity['rank']==self.binding.group_context.decision['selected_rank'],'TP2_GUARD_MUST_NOT_WRITE')
        return super().admit(event,actual)

    def end_actual(self,row):
        if self.phase!='fault' or self.injection_arm=='fault':return super().end_actual(row)
        self._context()
        self.require(self.injection_arm=='baseline' and self.fault_started and not self.fault_applied,
                     'TP2_GUARD_ROLE_REQUIRED')
        self.require(row['phase']=='fault' and row['request_id'] in self.requests,'TP2_GUARD_REQUEST_CHANGED')
        report=row['mechanism'];events=report.get('events',[])
        self.require(report.get('forwards')==[dict(index=0,status='returned')] and not report.get('errors')
            and len(events)==1 and not events[0].get('applied') and not events[0].get('output_changed')
            and events[0].get('original_execution_verified') and events[0].get('return_readback_exact')
            and events[0].get('consumers') and events[0]['actual']==self.target['actual'],
            'TP2_GUARD_UNCHANGED_SELECTED_BOUNDARY_REQUIRED')
        row.update(guard_boundary_unchanged=True,hidden_changed=row['output']['hidden']!=self.normal['output']['hidden'],
                   logits_changed=row['output']['logits']!=self.normal['output']['logits'])
        self.rows.append(deepcopy(row));self.index+=1;self.complete=self.index==len(FOUR)

    def query(self,*,owner,generation):
        result=super().query(owner=owner,generation=generation)
        result['multi_rank']=self.group_proof()
        return result


def commit_rms18_tp2_role(worker,decision):
    runtime=getattr(worker,'_demoncat_worker_fault_runtime',None)
    lease=getattr(runtime,'_leases',{}).get('18.api_half')
    require(lease is not None and type(lease.provider) is TP2Rms18Provider,'TP2_RUNTIME_LEASE_REQUIRED')
    return lease.provider.commit_role(decision)
