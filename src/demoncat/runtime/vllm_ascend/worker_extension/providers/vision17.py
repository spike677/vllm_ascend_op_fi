"""Existing WorkerFaultRuntime lease for an optional two-image-request census.

Observation only: no Fault17 admission, replay, F, CoverageSession proof or
Native qualification is produced. The Runtime remains the sole owner authority.
"""
from dataclasses import dataclass, field
from copy import deepcopy
import os
import threading

from ._scope import Scope, ProviderError, reject_unknown
from .. import identity as identity_reader


ATTRIBUTE='_demoncat_vision17_observation_binding'


def require(value,reason):
    if not value:raise ProviderError(reason)


@dataclass(frozen=True)
class Vision17Binding:
    worker: object
    runner: object
    model: object
    prompt: tuple
    image_sha256: str
    weights_sha256: str
    pid: int
    thread: int
    simulation: bool=False
    processor: object=None
    leases: set=field(default_factory=set,compare=False,repr=False)


def install_vision17_binding(worker,*,prompt_token_ids,image_sha256,weights_sha256,
                             first_live_validation=False,_simulation=False,_processor=None):
    from fault_injection.vision17_context import Vision17RequestContext
    require(not hasattr(worker,ATTRIBUTE),'VISION17_DEPLOYMENT_ALREADY_BOUND')
    require(first_live_validation is True and type(_simulation) is bool
            and (_processor is None or _simulation),'VISION17_EXPLICIT_REVIEWED_DEPLOYMENT_REQUIRED')
    require(isinstance(prompt_token_ids,(list,tuple)) and 2<=len(prompt_token_ids)<=4096
            and all(type(t) is int and t>=0 for t in prompt_token_ids),'VISION17_ACTUAL_BOUNDED_PROMPT_REQUIRED')
    require(all(isinstance(s,str) and len(s)==64 and all(c in '0123456789abcdef' for c in s)
                for s in (image_sha256,weights_sha256)),'VISION17_INPUT_WEIGHT_IDENTITIES_REQUIRED')
    # Construction inspects actual methods/source/objects only; no hooks/inference.
    context=Vision17RequestContext(worker,simulation=_simulation)
    binding=Vision17Binding(worker,worker.model_runner,worker.model_runner.model,tuple(prompt_token_ids),
        image_sha256,weights_sha256,os.getpid(),threading.get_ident(),_simulation,_processor)
    context.close();setattr(worker,ATTRIBUTE,binding)
    return dict(status='VISION17_OBSERVATION_BOUND',hooks_installed=0,fault_enabled=False,qualification_granted=False)


def remove_vision17_binding(worker):
    binding=getattr(worker,ATTRIBUTE,None)
    require(type(binding) is Vision17Binding and binding.worker is worker,'VISION17_DEPLOYMENT_NOT_BOUND')
    require(not binding.leases,'VISION17_OWNER_CLEAN_REQUIRED')
    delattr(worker,ATTRIBUTE)
    return dict(status='VISION17_OBSERVATION_BINDING_REMOVED')


class Vision17ObservationProvider(Scope):
    execution_boundary='python_hook'

    def __init__(self,worker,model,identity):
        super().__init__(worker,model,identity)
        self.binding=getattr(worker,ATTRIBUTE,None)
        require(type(self.binding) is Vision17Binding,'VISION17_OBSERVATION_BINDING_REQUIRED')
        self.session=None;self.reserved=False;self.checked=False;self.started=False;self.stopped=True
        self.error=None;self.failures=[];self.final_report=None
        self._context()

    def _context(self):
        b=self.binding;value=self.identity;device=getattr(self.worker,'device',None)
        visible,source=identity_reader._visible_devices()
        require(getattr(self.worker,ATTRIBUTE,None) is b and b.worker is self.worker and b.runner is self.worker.model_runner
                and b.model is self.model is b.runner.model and b.pid==os.getpid() and b.thread==threading.get_ident(),
                'VISION17_BINDING_LIFETIME_CHANGED')
        require(value['pid']==os.getpid() and self.worker.rank==value['rank']
                and getattr(device,'type',None)=='npu' and getattr(device,'index',None)==value['configured_logical_device']
                and identity_reader._current_device()==value['logical_device'] and visible==value['visible_devices']
                and source==value['visible_devices_source'],'VISION17_HELD_WORKER_IDENTITY_CHANGED')
        if b.leases or self.reserved:
            require(self.reserved and b.leases=={self._ticket},'VISION17_BINDING_OWNER_CHANGED')

    def dispatch(self,action,settings,*,owner,generation):
        if action=='prepare':self.bind(owner,generation)
        self.verify(owner,generation)
        try:
            self._context();reject_unknown(settings,('request_id','collect') if action=='advance' else ())
            if action=='prepare':require(not self.binding.leases,'VISION17_DEPLOYMENT_BUSY')
            elif action=='register':
                require(not self.binding.leases,'VISION17_DEPLOYMENT_BUSY')
                self.binding.leases.add(self._ticket);self.reserved=self._registered=True
            elif action=='check':
                require(self._registered and not self.model.training,'VISION17_REGISTERED_INFERENCE_REQUIRED');self.checked=True
            elif action in ('enable','execute'):
                require(self.checked and not self.started,'VISION17_SINGLE_OBSERVATION_ATTEMPT')
                from fault_injection.vision17_worker import Vision17WorkerCensus
                self.started=True;self.stopped=False
                b=self.binding
                self.session=Vision17WorkerCensus(self.worker,prompt=b.prompt,simulation=b.simulation,_processor=b.processor)
                self.session.start()
            elif action=='advance':
                require(self.started and not self.stopped and self.error is None,'VISION17_ACTIVE_OBSERVATION_REQUIRED')
                if settings=={'collect':True} and type(settings['collect']) is bool:self.session.collect()
                else:
                    require(set(settings)=={'request_id'},'VISION17_EXPLICIT_REQUEST_OR_COLLECT_REQUIRED')
                    self.session.expect(settings['request_id'])
            elif action=='finish':
                require(self.session is not None and len(self.session.rows)==2 and self.session.pending is None
                        and self.session.error is None,
                        'VISION17_TWO_ACTUAL_REQUESTS_REQUIRED')
                self._cleanup()
            elif action in ('disable','abort','recover','clean','unregister'):
                self._cleanup()
                if action in ('clean','unregister'):self._registered=False
                if action=='unregister' and self.reserved:
                    self.binding.leases.remove(self._ticket);self.reserved=False
            else:raise ProviderError('VISION17_UNSUPPORTED_ACTION')
        except BaseException as error:
            self.error=type(error).__name__+': '+str(error)
            self.failures.append(dict(action=action,error=self.error));raise
        return self.query(owner=owner,generation=generation)

    def _cleanup(self):
        if self.session is not None and not self.session.closed:
            self.final_report=self.session.report()
            cleanup=self.session.close()
            self.final_report['cleanup']=cleanup
            self.session.error=None
        self.error=None;self.stopped=True

    def query(self,*,owner,generation):
        self.verify(owner,generation);self._context()
        session=self.session
        if session is not None and not session.closed:session.verify()
        active=bool(session and (session.callbacks or session.visual_active or session.root_active))
        hooks=bool(session and (session.handles or session.trace is not None))
        error=self.error or getattr(session,'error',None)
        return deepcopy(dict(execution_boundary='python_hook',state_source='OWNED_PYTHON_HOOKS',
            hook_state_available=True,hook_owned=hooks,owner=owner,generation=generation,registered=self._registered,
            enabled=False,inflight=int(active),leases=int(bool(session and session.trace is not None)),
            native_loaded=False,native_registered=False,binding_reserved=self.reserved,
            replay_pending=self.started and not self.stopped,replay_failed=bool(error),
            mechanism='17.spatial_padding',execution_mode='TWO_HEALTHY_MM_REQUEST_CENSUS_ONLY',
            fault_injections=0,public_qualification_granted=False,C1=False,C2=False,C3=False,E5=False,
            serving=dict(pending_request=getattr(session,'pending',None),error=error,failures=self.failures,
                         completed_requests=len(session.rows) if session else 0),
            observation=self.final_report if self.stopped else session.report() if session else None))
