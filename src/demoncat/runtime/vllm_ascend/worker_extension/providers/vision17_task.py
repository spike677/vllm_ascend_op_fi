"""Four ordinary image requests at the reviewed Torch Conv3d boundary.

No CoverageSession, C1/group driver or Native ABI qualification is inherited.
Only a single actual projection is supported; unchanged arguments and its first
real consumer are independently observed above and below the parameter change.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import os
import threading
from ._scope import Scope, reject_unknown, ProviderError
from .tensor_natural import _sha
from .. import identity as identity_reader
from fault_injection.vision17_trace import require
from fault_injection.vision17_observation import call_signature, output_identity
from fault_injection.tensor_callable import TensorCallableGuard

ATTRIBUTE = '_demoncat_vision17_task_binding'
PHASES = ('normal', 'fault', 'off', 'health')

@dataclass(frozen=True)
class Vision17TaskBinding:
    worker: object
    runner: object
    model: object
    prompt: tuple
    image_sha256: str
    weights_sha256: str
    environment_sha256: str
    pid: int
    thread: int
    simulation: bool = False
    processor: object = None
    leases: set = field(default_factory=set, compare=False, repr=False)


def install_vision17_task_binding(worker, *, prompt_token_ids, image_sha256,
        weights_sha256, environment_sha256, _simulation=False, _processor=None):
    from fault_injection.vision17_context import Vision17RequestContext
    require(not any(hasattr(worker, attr) for attr in (ATTRIBUTE,
        '_demoncat_vision17_group_binding','_demoncat_vision17_observation_binding')),
        'VISION17_DEPLOYMENT_ALREADY_BOUND')
    require(type(_simulation) is bool and (_simulation or _processor is None),
        'VISION17_EXPLICIT_CPU_PROCESSOR_ONLY')
    require(isinstance(prompt_token_ids,(tuple,list)) and 2 <= len(prompt_token_ids) <= 4096
        and all(type(t) is int and t >= 0 for t in prompt_token_ids), 'VISION17_ACTUAL_PROMPT_REQUIRED')
    require(all(isinstance(s,str) and len(s)==64 and all(c in '0123456789abcdef' for c in s)
        for s in (image_sha256,weights_sha256,environment_sha256)), 'VISION17_SCOPE_IDENTITIES_REQUIRED')
    context=Vision17RequestContext(worker,simulation=_simulation)
    context.close()  # Actual source/layout/scheduler check; no hooks or forward.
    binding=Vision17TaskBinding(worker,worker.model_runner,worker.model_runner.model,tuple(prompt_token_ids),
        image_sha256,weights_sha256,environment_sha256,os.getpid(),threading.get_ident(),_simulation,_processor)
    setattr(worker,ATTRIBUTE,binding)
    return dict(status='VISION17_ORDINARY_BOUND_NOT_OBSERVED',hooks_installed=0,qualification_granted=False)


def remove_vision17_task_binding(worker):
    binding=getattr(worker,ATTRIBUTE,None)
    require(type(binding) is Vision17TaskBinding and binding.worker is worker,'VISION17_TASK_NOT_BOUND')
    require(not binding.leases,'VISION17_OWNER_CLEAN_REQUIRED')
    delattr(worker,ATTRIBUTE)
    return dict(status='VISION17_TASK_BINDING_REMOVED')


class _ActualProjection:
    def __init__(self, provider):
        self.provider=provider
        self.model=provider.model
        self.visual=self.model.visual
        self.references=None
        self.normal=None
        self.rows=[]
        self.calls=[]
        self.current_request=None
        self.guard=None
        self.target=None
        self._running=False

    def begin_request(self, request, phase, arm):
        require(not self._running and request.model is self.model and request.visual is self.visual,
            'VISION17_CURRENT_MODEL_OR_REQUEST_CHANGED')
        if self.guard:self.guard.verify()
        self.current_request=request
        self.phase,self.arm=phase,arm
        self.calls=[]
        self._running=True

    def observe(self, row):
        request=self.current_request
        require(self._running and request.current['request_binding']['single_image_request_join_proven'],
            'VISION17_ACTUAL_IMAGE_REQUEST_JOIN_REQUIRED')
        require(not self.calls and row['sequence']==0,'VISION17_MULTIPLE_PROJECTIONS_NOT_ADAPTED')
        if self.normal:
            before=self.normal['initial_observation']['observed_calls'][0]
            require(call_signature(row)==call_signature(before) and row['module_object_id']==self.target,
                'VISION17_ACTUAL_INPUT_WEIGHT_SOURCE_CHANGED')
        self.calls.append(row)

    def admit(self, event):
        require(self.phase=='fault' and self.normal is not None and len(self.calls)==1
            and (event['path'],event['ordinal'])==(self.calls[0]['path'],0),
            'VISION17_NORMAL_MATCHED_ACTUAL_CALL_REQUIRED')
        require(not self.provider.fault_admitted,'VISION17_SINGLE_FAULT_ADMISSION_REQUIRED')
        self.guard.verify()
        self.provider.fault_admitted=True
        return True

    def collect(self, row):
        require(self.current_request.closed,'VISION17_OWNED_REQUEST_MUST_CLOSE')
        upper,lower=row['initial_observation'],row['conv_observation']
        require(upper['torch_dispatch_chain_complete'] and lower['torch_dispatch_chain_complete']
            and not any(x['hooks'] or x['owned_mode'] for x in (upper,lower))
            and len(upper['observed_calls'])==len(lower['observed_calls'])==len(self.calls)==1,
            'VISION17_ONE_COMPLETE_ACTUAL_PROJECTION_REQUIRED')
        before,after=upper['observed_calls'][0],lower['observed_calls'][0]
        require(before['path']==after['path'] and before['sequence']==after['sequence']==0
            and before['inputs_before']==after['inputs_before'] and before['output']==after['output']
            and before['inputs_unchanged'] and after['inputs_unchanged'],
            'VISION17_INDEPENDENT_UPPER_LOWER_JOIN_REQUIRED')
        expected=[0,7,7] if self.phase=='fault' else [0,0,0]
        require(before['actual_geometry']['padding']==[0,0,0]
            and after['actual_geometry']==dict(before['actual_geometry'],padding=expected),
            'VISION17_ACTUAL_FINAL_PADDING_OR_OTHER_ARGUMENT_CHANGED')
        events=(row['mechanism'] or {}).get('events',[])
        if self.phase=='normal':
            require(len(events)==1 and not events[0]['applied'] and self.references,
                'VISION17_NORMAL_ACTUAL_PARAMETER_REFERENCE_REQUIRED')
            self.normal=deepcopy(row)
            module=dict(self.visual.named_modules()).get('' if before['path']=='<root>' else before['path'])
            require(module is not None,'VISION17_ACTUAL_PROJECTION_NOT_HELD')
            self.guard=TensorCallableGuard(module)
            self.target=id(module)
        elif self.phase=='fault':
            baseline=self.normal['conv_observation']['observed_calls'][0]
            require(len(events)==1 and events[0]['applied'] and self.provider.fault_admitted
                and events[0]['native_calls']==1 and events[0]['original_execution_verified']
                and events[0]['output_changed'] and after['output']!=baseline['output']
                and after['consumer']['consumed_values']!=baseline['consumer']['consumed_values']
                and after['consumer']['outputs']!=baseline['consumer']['outputs'],
                'VISION17_ACTUAL_PADDING_EFFECT_AND_CONSUMER_REQUIRED')
            row['effect']=dict(comparison='MATCHED_COMPLETE_INPUT_WEIGHT_BIAS_AND_FIRST_CONSUMER',
                position=dict(path=before['path'],ordinal=0),baseline=baseline['output'],fault=after['output'],
                complete_output_changed=True,first_consumer_changed=True,
                logits_changed=row['logits']!=self.normal['logits'],native_calls=1,
                aclnn_launch_arguments=False)
        else:
            require(not events and output_identity(row)==output_identity(self.normal)
                and after['output']==self.normal['conv_observation']['observed_calls'][0]['output'],
                'VISION17_COMPLETE_ENCODER_HIDDEN_LOGITS_OFF_NOT_EXACT')
        self.rows.append(deepcopy(row))
        self._running=False
        self.current_request=None

    def fail(self,error):
        self._running=False
        self.current_request=None


class Vision17TaskProvider(Scope):
    execution_boundary='python_hook'
    def __init__(self,worker,model,identity):
        super().__init__(worker,model,identity)
        self.binding=getattr(worker,ATTRIBUTE,None)
        require(type(self.binding) is Vision17TaskBinding,'VISION17_ORDINARY_BINDING_REQUIRED')
        self.reserved=self.checked=self.started=self.complete=self.recovered=False
        self.stopped=True
        self.session=self.engine=None
        self.index=0
        self.seen=set()
        self.pending=self.recovering=self.error=self.first_error=None
        self.failures=[]
        self.fault_started=self.fault_admitted=False
        self._context()

    @property
    def phase(self):
        return self.recovering or (PHASES[self.index] if self.index<4 else 'complete')

    def _context(self):
        b=self.binding;value=self.identity;device=getattr(self.worker,'device',None)
        visible,source=identity_reader._visible_devices()
        require(getattr(self.worker,ATTRIBUTE,None) is b and b.worker is self.worker
            and b.runner is self.worker.model_runner and b.model is self.model is b.runner.model
            and b.pid==os.getpid() and b.thread==threading.get_ident(),'VISION17_BINDING_LIFETIME_CHANGED')
        require(value['pid']==os.getpid() and self.worker.rank==value['rank'] and getattr(device,'type',None)=='npu'
            and getattr(device,'index',None)==value['configured_logical_device']
            and identity_reader._current_device()==value['logical_device'] and visible==value['visible_devices']
            and source==value['visible_devices_source'],'VISION17_HELD_WORKER_IDENTITY_CHANGED')
        if self.reserved or b.leases:
            require(self.reserved and b.leases=={self._ticket},'VISION17_BINDING_OWNER_CHANGED')

    def _hooks(self):
        s=self.session
        return int(bool(s and (s.handles or s.trace or s.upper or s.adapter)))

    def dispatch(self,action,settings,*,owner,generation):
        if action=='prepare':self.bind(owner,generation)
        self.verify(owner,generation)
        try:
            self._context()
            reject_unknown(settings,('request_id','collect') if action=='advance' else ())
            if action=='prepare':require(not self.binding.leases,'VISION17_DEPLOYMENT_BUSY')
            elif action=='register':
                require(not self.binding.leases and self.engine is None,'VISION17_NEW_TASK_SCOPE_REQUIRED')
                self.binding.leases.add(self._ticket);self.reserved=self._registered=True
                self.engine=_ActualProjection(self)
            elif action=='check':
                require(self._registered and not self.model.training,'VISION17_REGISTERED_INFERENCE_REQUIRED')
                self.checked=True
            elif action in ('enable','execute'):
                require(self.checked and not self.started,'VISION17_SINGLE_ATTEMPT_REQUIRED')
                self.started=True;self.stopped=False
            elif action=='advance':
                require(self.started and not self.stopped and not self.error,'VISION17_ACTIVE_TASK_REQUIRED')
                if settings=={'collect':True} and type(settings['collect']) is bool:self._collect()
                else:
                    require(self.phase!='complete','VISION17_TASK_PHASES_EXHAUSTED')
                    require(set(settings)=={'request_id'} and self.pending is None,'VISION17_EXPLICIT_COLLECTION_REQUIRED')
                    rid=settings['request_id']
                    require(isinstance(rid,str) and 0<len(rid)<=256 and rid not in self.seen and len(self.seen)<8,
                        'VISION17_BOUNDED_FRESH_REQUEST_REQUIRED')
                    self.seen.add(rid)
                    require(not self._hooks(),'VISION17_PREVIOUS_HOOKS_REMAIN')
                    from fault_injection.vision17_request import ActualVision17Request
                    phase=self.phase
                    if phase=='fault':
                        require(self.engine.normal is not None and not self.fault_started,'VISION17_ONE_FAULT_ATTEMPT_REQUIRED')
                        self.fault_started=True
                    self.pending=(rid,phase)
                    b=self.binding
                    self.session=ActualVision17Request(self.worker,engine=self.engine,phase=phase,
                        arm='baseline' if phase=='normal' else 'fault' if phase=='fault' else None,
                        prompt=b.prompt,simulation=b.simulation,_processor=b.processor)
                    self.session.start();self.session.expect(rid)
            elif action=='finish':
                require(self.complete and self.recovered and not self.error,'VISION17_OFF_AND_HEALTH_REQUIRED')
                self._cleanup()
            elif action=='recover':
                self._abort_owned()
                if self.fault_started and not self.recovered:
                    self.recovering='off';self.error=None;self.stopped=False
                else:self._cleanup()
            elif action in ('disable','abort','clean','unregister'):
                self._cleanup()
                if action in ('clean','unregister'):self._registered=False
                if action=='unregister' and self.reserved:
                    self.binding.leases.remove(self._ticket);self.reserved=False
            else:raise ProviderError('VISION17_UNSUPPORTED_ACTION')
        except BaseException as error:
            self.first_error=self.first_error or repr(error)
            self.error=self.error or repr(error)
            self.failures.append(dict(action=action,error=repr(error)))
            raise
        return self.query(owner=owner,generation=generation)

    def _collect(self):
        require(self.pending and self.session and not self.session.error,'VISION17_ACTUAL_REQUEST_FAILED_OR_MISSING')
        row=self.session.collect()
        cleanup=self.session.close()
        self.engine.collect(row)
        row['cleanup']=cleanup
        phase=self.pending[1];self.pending=None
        if phase=='health':self.recovered=True
        if self.recovering:
            self.recovering='health' if phase=='off' else None
            if phase=='health':self.complete=True
        else:
            self.index+=1;self.complete=self.index==4

    def _abort_owned(self):
        if self.session and not self.session.closed:
            if self.session.error:self.first_error=self.first_error or self.session.error
            self.session.close()
        if self.engine:self.engine.fail(RuntimeError('OWNED_REQUEST_CLOSED'))
        self.pending=None

    def _cleanup(self):
        self._abort_owned()
        require(not self.fault_started or self.recovered,'VISION17_REAL_OFF_AND_HEALTH_REQUIRED')
        require(not self._hooks(),'VISION17_OWNED_HOOKS_REMAIN')
        self.stopped=True

    def query(self,*,owner,generation):
        self.verify(owner,generation);self._context()
        s=self.session
        if s and not s.closed:s.verify()
        rows=deepcopy(self.engine.rows) if self.engine else []
        fault=next((r for r in reversed(rows) if r['phase']=='fault'),None)
        active=bool(s and (s.callbacks or s.visual_active or s.root_active))
        error=self.error or (s.error if s and not s.closed else None)
        evidence=dict(actual_requests=rows,complete=self.complete,
            matched_off_and_health_confirmed=self.complete and self.recovered,
            last_fault_effect=deepcopy(fault.get('effect')) if fault else None,
            first_error=self.first_error,failures=deepcopy(self.failures),selection='FIRST_ACTUAL_SINGLE_PROJECTION')
        return dict(execution_boundary='python_hook',state_source='OWNED_PYTHON_HOOKS',
            hook_state_available=True,hook_owned=bool(self._hooks()),owner=owner,generation=generation,
            registered=self._registered,enabled=bool(s and s.adapter),inflight=int(active),
            leases=int(bool(s and s.adapter)),owned_hooks=self._hooks(),native_loaded=False,native_registered=False,
            replay_pending=self.started and not self.stopped,replay_failed=bool(error and not self.stopped),
            recovery_validation_pending=self.fault_started and not self.recovered,
            ordinary_task=True,mechanism='17.spatial_padding',simulation=self.binding.simulation,
            aclnn_launch_arguments=False,native_qualification=False,C1=False,C2=False,C3=False,E5=False,
            qualification_granted=False,main_project_integration='NOT_VERIFIED',
            evidence=evidence,serving=dict(next_phase=self.phase,pending_request=self.pending,
                complete=self.complete,recovered=self.recovered,failure=error,actual_requests=rows))
