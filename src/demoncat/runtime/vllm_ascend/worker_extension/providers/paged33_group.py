"""Public isolated all-target C2; C3 current-input authority remains absent."""
from copy import deepcopy

from .attention_serving import AttentionServingProvider,require
from ..paged33_group_execution import GroupExecutionAdapter
from ..paged33_group_native import GroupEffectsNativeCall
from ..paged33_group_evidence import PHASES
from ..paged33_s2_evidence import digest


class Paged33GroupServingProvider(AttentionServingProvider):
    mechanism='33.paged_decode'
    phases=PHASES
    execution_adapter_type=GroupExecutionAdapter

    def __init__(self,worker,model,identity):
        super().__init__(worker,model,identity)
        require(type(self.binding.native) is GroupEffectsNativeCall or self.binding.simulation
            and getattr(self.binding.native,'public_group_protocol',False),'GROUP_TRUSTED_NATIVE_DRIVER_REQUIRED')
        require(self.binding.native.worker_identity==identity,'GROUP_DEPLOYMENT_ACTUAL_WORKER_IDENTITY_DIFFERS')
        self.coverage_artifact=None;self.recovery_phase=None

    def _failure(self,action,error):
        super()._failure(action,error)
        if self.session is not None and hasattr(self.session,'control'):self.session.control.failure(error)

    def _prepare_phase(self,phase):
        require(self.binding.native.state()['native_loaded'],'GROUP_REAL_NATIVE_REQUIRED')
        require(not self.recovery_phase or phase==self.recovery_phase,'GROUP_ONLY_FRESH_RECOVERY_REQUEST_ALLOWED')

    def dispatch(self,action,settings,*,owner,generation):
        if action=='recover' and settings=={} and self.session is not None and self.session.needs_off:
            self.verify(owner,generation);self._context()
            try:self.session.prepare_off_recovery()
            except BaseException as error:self._failure('prepare_recovery',error);raise
            self.pending=None;self.error=None;self.recovery_phase='recovery_off'
            return self.query(owner=owner,generation=generation)
        if action=='advance' and self.recovery_phase and set(settings)=={'request_id'}:
            self.verify(owner,generation);self._context()
            require(self.pending is None and not self.error,'GROUP_RECOVERY_REQUEST_PENDING')
            try:
                self.session.expect(settings['request_id'],self.recovery_phase)
                self.pending=(settings['request_id'],self.recovery_phase)
            except BaseException as error:self._failure('advance_recovery',error);raise
            return self.query(owner=owner,generation=generation)
        return super().dispatch(action,settings,owner=owner,generation=generation)

    def _collect(self):
        require(self.pending is not None,'GROUP_ACTUAL_PENDING_REQUEST_REQUIRED')
        artifact=self.session.collect()
        require((artifact['request_id'],artifact['phase'])==self.pending,'GROUP_COLLECT_REQUEST_MAPPING_CHANGED')
        self.records.append(artifact);self.pending=None
        if self.recovery_phase:self.recovery_phase='recovery_health' if self.recovery_phase=='recovery_off' else None
        else:self.index+=1

    def _cleanup(self):
        session=self.session
        if session is not None and hasattr(session,'handles'):
            session.close()
            if hasattr(session,'control'):
                session.control.finalize()
                self.coverage_artifact=session.write_coverage()
        else:self.binding.native.stop()
        self.pending=None;self.stopped=True;self.error=None

    def query(self,*,owner,generation):
        self.verify(owner,generation);self._context();native=self.binding.native.state();session=self.session
        state=session.state() if session is not None and hasattr(session,'handles') else dict(owned_hooks=0,owned_api_wrappers=0,
            inflight=0,owned_input_references=0,needs_off=False,pending_request=None,completed_requests=0,failures=[])
        controller=session.control if session is not None and hasattr(session,'control') else None
        pending=native.get('pending_drains',0)+native.get('pending_references',0)
        owned=int(native['scope_owned'])+native['environment_owned']+pending
        resources=owned+state['owned_hooks']+state['owned_api_wrappers']+state['owned_input_references']
        summary=controller.summary() if controller else None
        ready=bool(controller and controller.c2_proof is not None and self.stopped and not resources
                   and not self.failures and not self.binding.simulation)
        return dict(execution_boundary=self.execution_boundary,owner=owner,generation=generation,
            native_loaded=native['native_loaded'],native_registered=bool(resources),registered=self._registered,
            enabled=bool(native['scope_active']),inflight=state['inflight'],leases=owned,binding_reserved=self.binding_reserved,
            replay_pending=bool(self.attempted and not self.stopped),replay_failed=bool(self.error or session and session.last_error),
            owned_hooks=state['owned_hooks'],owned_api_wrappers=state['owned_api_wrappers'],mechanism=self.mechanism,
            simulation=self.binding.simulation,native_state=native,
            serving=dict(next_phase=self.recovery_phase or (self.phases[self.index] if self.index<len(self.phases) else 'complete'),
                pending_request=self.pending,failures=deepcopy(self.failures),resource_state=state),
            evidence=dict(schema='demoncat.paged33.public_group_c2.v1',requests=deepcopy(self.records),
                actual_requests_sha256=digest(self.records),coverage=deepcopy(self.coverage_artifact),group=summary,
                actual_call_denominator=sum(r['actual_calls'] for r in self.records),
                historical_qualification_inherited=False,normal_evidence_sha256=self.binding.native.normal_evidence.fingerprint),
            C1=False,C2=ready,C3=False,E5='NOT_ASSESSED',public_qualification_granted=ready)
