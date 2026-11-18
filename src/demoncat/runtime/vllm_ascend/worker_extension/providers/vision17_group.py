"""Public Worker Core owner for fresh natural image Conv3d C1/C2/C3 requests.

Uses the existing CoverageSession and original padding mechanism. Only Torch
dispatch arguments/consumers are proved; no Native ABI or device claim.
"""
from copy import deepcopy
from dataclasses import dataclass,field
import os
import threading

from ._scope import Scope,ProviderError,reject_unknown
from .tensor_natural import _sha
from .. import identity as identity_reader

ATTRIBUTE='_demoncat_vision17_group_binding'
REQUESTS=(('native_baseline',None),('baseline','baseline'),('baseline_repeat','baseline'),
    ('lossless','baseline'),('c2_fault','fault'),('c2_off',None),
    ('c3_fault_0','fault'),('c3_fault_1','fault'),('c3_off',None),('cleanup_healthy',None))


def require(condition,reason):
    if not condition:raise ProviderError(reason)


@dataclass(frozen=True)
class Vision17GroupBinding:
    worker:object
    runner:object
    model:object
    prompt:tuple
    image_sha256:str
    weights_sha256:str
    workload_sha256:str
    environment_sha256:str
    model_source_sha256:str
    pid:int
    thread:int
    simulation:bool=False
    processor:object=None
    synchronize:object=None
    leases:set=field(default_factory=set,compare=False,repr=False)


def install_vision17_group_binding(worker,*,prompt_token_ids,image_sha256,weights_sha256,environment_sha256,
        first_live_validation=False,_simulation=False,_processor=None,_synchronize=None):
    from fault_injection.vision17_context import Vision17RequestContext
    require(not hasattr(worker,ATTRIBUTE) and not hasattr(worker,'_demoncat_vision17_observation_binding'),
            'VISION17_DEPLOYMENT_ALREADY_BOUND')
    require(first_live_validation is True and type(_simulation) is bool
        and (_simulation or (_processor is None and _synchronize is None)),
        'VISION17_EXPLICIT_DEPLOYMENT_CPU_OVERRIDES_ONLY')
    require(not _simulation or callable(_synchronize),'VISION17_CPU_SYNCHRONIZER_REQUIRED')
    require(isinstance(prompt_token_ids,(tuple,list)) and 2<=len(prompt_token_ids)<=4096
        and all(type(t) is int and t>=0 for t in prompt_token_ids),'VISION17_BOUNDED_ACTUAL_PROMPT_REQUIRED')
    require(all(isinstance(s,str) and len(s)==64 and all(c in '0123456789abcdef' for c in s)
        for s in (image_sha256,weights_sha256,environment_sha256)),'VISION17_SCOPE_IDENTITIES_REQUIRED')
    context=Vision17RequestContext(worker,simulation=_simulation)
    source=context.pins['embed_multimodal'].digest;context.close()
    workload=_sha(dict(prompt=list(prompt_token_ids),image_sha256=image_sha256))
    binding=Vision17GroupBinding(worker,worker.model_runner,worker.model_runner.model,tuple(prompt_token_ids),
        image_sha256,weights_sha256,workload,environment_sha256,source,os.getpid(),threading.get_ident(),
        _simulation,_processor,_synchronize)
    setattr(worker,ATTRIBUTE,binding)
    return dict(status='VISION17_GROUP_BOUND_NOT_QUALIFIED',hooks_installed=0,qualification_granted=False,
                workload_sha256=workload,simulation=_simulation)


def remove_vision17_group_binding(worker):
    binding=getattr(worker,ATTRIBUTE,None)
    require(type(binding) is Vision17GroupBinding and binding.worker is worker,'VISION17_GROUP_NOT_BOUND')
    require(not binding.leases,'VISION17_GROUP_OWNER_CLEAN_REQUIRED')
    delattr(worker,ATTRIBUTE)
    return dict(status='VISION17_GROUP_BINDING_REMOVED')


class Vision17GroupProvider(Scope):
    execution_boundary='python_hook'
    def __init__(self,worker,model,identity):
        super().__init__(worker,model,identity)
        self.binding=getattr(worker,ATTRIBUTE,None)
        require(type(self.binding) is Vision17GroupBinding,'VISION17_TYPED_GROUP_BINDING_REQUIRED')
        self.reserved=self.checked=self.started=self.complete=self.recovered=False
        self.stopped=True;self.session=self.engine=None;self.index=0;self.seen=[]
        self.error=self.pending=self.recovering=None;self.reports={};self.failures=[]
        self._context()

    def _context(self):
        b=self.binding;value=self.identity;device=getattr(self.worker,'device',None)
        visible,source=identity_reader._visible_devices()
        require(getattr(self.worker,ATTRIBUTE,None) is b and b.worker is self.worker
            and b.runner is self.worker.model_runner and b.model is self.model is b.runner.model
            and b.pid==os.getpid() and b.thread==threading.get_ident(),'VISION17_GROUP_LIFETIME_CHANGED')
        require(value['pid']==os.getpid() and self.worker.rank==value['rank'] and getattr(device,'type',None)=='npu'
            and getattr(device,'index',None)==value['configured_logical_device']
            and identity_reader._current_device()==value['logical_device'] and visible==value['visible_devices']
            and source==value['visible_devices_source'],'VISION17_GROUP_WORKER_IDENTITY_CHANGED')
        if self.reserved or b.leases:require(self.reserved and b.leases=={self._ticket},'VISION17_GROUP_OWNER_CHANGED')

    def _failure(self,error,action):
        if self.error is None:self.error=type(error).__name__+': '+str(error)
        self.failures.append(dict(action=action,error=repr(error),first_error=self.error,
            request=self.session.report() if self.session else None))
        if self.engine:self.engine.fail(error)

    def _hooks(self):
        s=self.session
        return int(bool(s and (s.handles or s.trace or s.upper or s.adapter)))

    def dispatch(self,action,settings,*,owner,generation):
        if action=='prepare':self.bind(owner,generation)
        self.verify(owner,generation)
        try:
            self._context();reject_unknown(settings,('request_id','collect') if action=='advance' else ())
            if action=='prepare':require(not self.binding.leases,'VISION17_GROUP_BUSY')
            elif action=='register':
                require(not self.binding.leases,'VISION17_GROUP_BUSY')
                self.binding.leases.add(self._ticket);self.reserved=self._registered=True
            elif action=='check':
                require(self._registered and not self.model.training,'VISION17_REGISTERED_INFERENCE_REQUIRED');self.checked=True
            elif action in ('enable','execute'):
                require(self.checked and not self.started,'VISION17_GROUP_SINGLE_ATTEMPT')
                from fault_injection.vision17_group import Vision17Group
                self.started=True;self.stopped=False
                self.engine=Vision17Group(self.worker,self.identity,self.binding,owner,generation)
                self.engine.owned_host_hooks=self._hooks
            elif action=='advance':
                require(self.started and not self.stopped and not self.error,'VISION17_GROUP_ACTIVE_REQUIRED')
                if settings=={'collect':True} and type(settings['collect']) is bool:self._collect()
                else:
                    require(set(settings)=={'request_id'} and self.pending is None,'VISION17_GROUP_EXPLICIT_COLLECTION_REQUIRED')
                    rid=settings['request_id']
                    require(isinstance(rid,str) and rid and rid not in self.seen,'VISION17_GROUP_FRESH_REQUEST_REQUIRED')
                    require(self.recovering or self.index<len(REQUESTS),'VISION17_GROUP_PROTOCOL_COMPLETE')
                    phase,arm=('recovery_'+self.recovering,None) if self.recovering else REQUESTS[self.index]
                    require(not self._hooks(),'VISION17_GROUP_PREVIOUS_HOOKS_REMAIN')
                    from fault_injection.vision17_request import ActualVision17Request
                    b=self.binding;self.pending=(rid,phase)
                    self.session=ActualVision17Request(self.worker,engine=self.engine,phase=phase,arm=arm,
                        prompt=b.prompt,simulation=b.simulation,_processor=b.processor)
                    self.session.start();self.session.expect(rid)
            elif action=='finish':
                require(self.pending is None and (self.complete or self.recovered),'VISION17_GROUP_REAL_PROTOCOL_UNFINISHED')
                self._cleanup()
            elif action=='recover':self._recover()
            elif action in ('disable','abort','clean','unregister'):
                self._cleanup()
                if action in ('clean','unregister'):self._registered=False
                if action=='unregister' and self.reserved:
                    self.binding.leases.remove(self._ticket);self.reserved=False
            else:raise ProviderError('VISION17_GROUP_UNSUPPORTED_ACTION')
        except BaseException as error:self._failure(error,action);raise
        return self.query(owner=owner,generation=generation)

    def _collect(self):
        require(self.pending is not None and self.session is not None,'VISION17_GROUP_NO_PENDING_REQUEST')
        require(not self.session.error,'VISION17_GROUP_ACTUAL_REQUEST_FAILED')
        row=self.session.collect();cleanup=self.session.close()
        self.engine.collect(row);rid,phase=self.pending
        self.seen.append(rid);self.pending=None
        self.reports['last_request']=dict(phase=phase,request_id=rid,cleanup=cleanup)
        if phase=='lossless':
            self.reports['discovery']=self.engine.qualify();self.engine.open_group('once',forwards=1)
        elif phase=='c2_off':
            report=self.engine.close_group();require(report['contract_satisfied'],'VISION17_C2_NOT_COMPLETE')
            self.reports['C1']=dict(complete=True,actual_target_count=1,
                qualification_scope='SINGLE_LIVE_TORCH_PARAMETER_BOUNDARY',aclnn_launch_arguments=False,
                native_qualification=False,proof_report_sha256=_sha(report),simulation=self.binding.simulation)
            proof=self.engine.group.c2_proof();report['live_c2_proof']=self.engine.group.report()['live_c2_proof']
            self.reports['C2']=report;self.engine.open_group('every_match',forwards=2,c2_proof=proof)
        elif phase=='c3_off':
            report=self.engine.close_group();require(report['contract_satisfied'],'VISION17_C3_NOT_COMPLETE')
            self.reports['C3']=report
        elif phase=='recovery_off':
            report=self.engine.close_group();self.reports['failed_attempt_recovery']=report
            require(report['recovery']['complete'],'VISION17_GROUP_RECOVERY_NOT_COMPLETE')
            self.recovering='health'
        elif phase in ('cleanup_healthy','recovery_health'):
            require(not self._hooks() and self.engine.group.recovery_report()['complete'],
                    'VISION17_GROUP_CLEANUP_HEALTH_NOT_CONFIRMED')
            self.reports['same_worker_healthy']=dict(request_id=rid,worker=deepcopy(self.identity),
                actual_encoder_observed=True,complete_encoder_hidden_logits_exact=True,
                cleanup_before_request=True,fault_hooks=0,aclnn_launch_arguments=False)
            self.recovered=True
            if phase=='recovery_health':self.recovering=None;self.stopped=True
        if not phase.startswith('recovery_'):
            self.index+=1;self.complete=self.index==len(REQUESTS)

    def _abort_owned(self):
        if self.session and not self.session.closed:
            self.failures.append(dict(action='owned_request_abort',request=self.session.report()))
            self.session.close()
        if self.engine:
            if self.engine._running:self.engine.fail(RuntimeError('VISION17_REQUEST_ABORTED'))
            self.engine.current_request=None
        self.pending=None

    def _recover(self):
        self._abort_owned()
        group=self.engine.group if self.engine else None
        if group is None:
            self.error=None;self.recovered=True;self.stopped=True;return
        self.engine.enabled=False
        self.recovering='health' if group.recovery_report()['complete'] else 'off'
        self.error=None;self.stopped=False

    def _cleanup(self):
        self._abort_owned()
        group=self.engine.group if self.engine else None
        require(group is None or group.recovery_report()['complete'],'VISION17_GROUP_ACTUAL_OFF_REQUIRED')
        require(group is None or self.recovered,'VISION17_GROUP_ACTUAL_CLEANUP_HEALTH_REQUIRED')
        require(not self._hooks(),'VISION17_GROUP_OWNED_HOOKS_REMAIN')
        self.error=None;self.stopped=True

    def query(self,*,owner,generation):
        self.verify(owner,generation);self._context()
        session=self.session;engine=self.engine
        if session and not session.closed:session.verify()
        active=bool(session and (session.callbacks or session.visual_active or session.root_active))
        group=engine.group if engine else None
        held=bool(group and not group.recovery_report()['complete'])
        adapter=getattr(session,'adapter',None)
        error=self.error or (session.error if session and not session.closed else None)
        return deepcopy(dict(execution_boundary='python_hook',state_source='OWNED_PYTHON_HOOKS',
            hook_state_available=True,hook_owned=bool(self._hooks()),owner=owner,generation=generation,
            registered=self._registered,enabled=bool(adapter and adapter.active),inflight=int(active),
            leases=int(bool(adapter and adapter.locked)),coverage_target_leases_held=held,
            native_loaded=False,native_registered=False,binding_reserved=self.reserved,
            replay_pending=self.started and not self.stopped,replay_failed=bool(error),
            mechanism='17.spatial_padding',execution_mode='REAL_MULTIMODAL_WORKER_TORCH_PARAMETER_GROUP',
            actual_target_count=len(engine.selected) if engine else 0,multi_target_verified=False,
            aclnn_launch_arguments=False,device_payload_observed=False,native_qualification=False,E5=False,
            public_qualification_granted=False,simulation=self.binding.simulation,
            serving=dict(pending_request=self.pending,error=error,failures=self.failures,completed_requests=self.index,
                next_phase='recovery_'+self.recovering if self.recovering else REQUESTS[self.index][0] if self.index<len(REQUESTS) else None),
            reports=self.reports,phase_rows=engine.rows if engine else []))
