"""Continuous site provider using the existing WorkerFaultRuntime lease protocol."""
from copy import deepcopy
from ._scope import Scope,reject_unknown
from ..continuous_binding import current_binding,ContinuousRequest,digest
from ..serving_context import require
from fault_injection.continuous_contract import continuous_request_plan


class ContinuousProvider(Scope):
    execution_boundary='python_hook'
    ordinary_only=True
    continuous_protocol=True

    def __init__(self,worker,model,identity,mechanism):
        super().__init__(worker,model,identity)
        self.mechanism=mechanism;self.binding=current_binding(worker,model,mechanism)
        require(self.binding is not None,'CONTINUOUS_TYPED_BINDING_REQUIRED')
        self.plan=continuous_request_plan(mechanism,batch_size=len(self.binding.prompts))
        self.rows=[];self.session=None;self.checked=False;self.started=False
        self.error=None;self.closed=False;self.reserved=False
        self.recovery_phase=None;self.task_failure=None

    @property
    def phase(self):
        if self.recovery_phase is not None:return self.recovery_phase
        return self.plan['phases'][len(self.rows)]['phase'] if len(self.rows)<4 else 'complete'

    def collect(self):
        if self.session is not None and self.session.samples==4:
            r=self.session.collect();self.session=None
            require(r['all_finite'] or (self.mechanism=='39.output_proxy' and r['phase']=='fault'),
                    'CONTINUOUS_NONFINITE_MODEL_OUTPUT')
            if self.rows and r['phase'] in ('off','health'):
                require(r['output']==self.rows[0]['output'],'CONTINUOUS_OFF_HEALTH_NOT_EXACT')
            if r['phase']=='fault':
                require(r['output']!=self.rows[0]['output'],'CONTINUOUS_NO_MODEL_PROPAGATION')
            self.rows.append(r)
            if self.recovery_phase is not None:
                self.recovery_phase='health' if r['phase']=='off' else 'complete'

    def dispatch(self,action,settings,*,owner,generation):
        if action=='prepare':self.bind(owner,generation)
        self.verify(owner,generation);self.binding.check()
        reject_unknown(settings,('request_id','request_ids') if action=='advance' else ())
        try:
            if action not in ('disable','abort','recover','clean','unregister'):self.collect()
            if action=='prepare':
                require(not self.binding.leases and self.binding.warmup_valid,'CONTINUOUS_STABLE_WARMUP_REQUIRED')
            elif action=='register':
                require(not self.binding.leases,'CONTINUOUS_BINDING_ALREADY_OWNED')
                self.binding.leases.add(self._ticket);self.reserved=self._registered=True
            elif action=='check':
                require(self._registered,'CONTINUOUS_REGISTER_REQUIRED');self.checked=True
            elif action in ('enable','execute'):
                require(self.checked and not self.started,'CONTINUOUS_SINGLE_SCOPE_START')
                self.started=True
            elif action=='advance':
                require(self.started and not self.closed and self.error is None and self.session is None
                    and self.phase!='complete','CONTINUOUS_PHASE_NOT_ADVANCEABLE')
                field='request_ids' if self.plan['unit']=='batch' else 'request_id'
                require(set(settings)=={field},'CONTINUOUS_REAL_REQUEST_IDS_REQUIRED')
                ids=settings[field] if field=='request_ids' else [settings[field]]
                require(not any(set(ids)&set(r['request_ids']) for r in self.rows),'CONTINUOUS_FRESH_REQUEST_REQUIRED')
                self.session=ContinuousRequest(self.binding,ids,self.phase);self.session.install()
            elif action in ('finish','disable','abort','recover','clean','unregister'):
                if action=='finish':require(self.phase=='complete' and self.error is None and self.task_failure is None,
                                           'CONTINUOUS_FOUR_PHASE_RECOVERY_REQUIRED')
                if self.session is not None:self.session.close();self.session=None
                self.closed=True
                if action=='recover' and self.rows and self.rows[0]['phase']=='normal':
                    self.task_failure=self.task_failure or self.error or 'EXPLICIT_RECOVERY_REQUESTED'
                    self.error=None;self.recovery_phase='off';self.closed=False
                if action in ('clean','unregister'):
                    self._registered=False
                    if self.reserved:self.binding.leases.remove(self._ticket);self.reserved=False
            else:raise RuntimeError('UNSUPPORTED_PROVIDER_ACTION')
        except BaseException as e:self.error=repr(e);raise
        return self.query(owner=owner,generation=generation)

    def query(self,*,owner,generation):
        self.verify(owner,generation);self.binding.check()
        try:self.collect()
        except BaseException as e:self.error=repr(e);raise
        b=self.binding;normal=self.rows[0] if self.rows else None
        recovered=self.phase=='complete' and self.error is None
        evidence=dict(actual_requests=deepcopy(self.rows),matched_off_and_health_confirmed=recovered,
            recovered=recovered,complete=recovered and self.task_failure is None,failure=self.task_failure or self.error,
            coverage=deepcopy(self.rows[1]['coverage']) if len(self.rows)>1 else None)
        live_leases=0
        if self.session is not None and self.session.site is not None:
            site=self.session.site;pending=getattr(site,'pending',0)
            live_leases=len(pending) if isinstance(pending,dict) else pending
            regions=getattr(site,'region_leases',None)
            if regions is not None:live_leases=max(live_leases,regions.count())
        return dict(execution_boundary='python_hook',state_source='OWNED_PYTHON_HOOKS',
            hook_state_available=True,hook_owned=self.session is not None,owner=owner,generation=generation,
            native_loaded=False,native_registered=False,registered=self._registered,enabled=False,
            inflight=0 if self.session is None else self.session.inside,leases=live_leases,
            owned_hooks=0 if self.session is None else len(self.session.hooks)+len(self.session.patches),
            replay_pending=self.session is not None,replay_failed=bool(self.error and not self.closed),
            ordinary_task=True,request_plan=deepcopy(self.plan),next_phase=self.phase,complete=recovered and self.task_failure is None,
            evidence=evidence,multi_rank=dict(owner=b.group['owner'],group_generation=b.group['group_generation'],
                members_sha256=b.group['members_sha256'],worker=self.identity,
                actual_normal_request_observed=normal is not None,
                normal_request_id=(normal['request_ids'] if self.plan['unit']=='batch' else normal['request_ids'][0]) if normal else None,
                request_fingerprint=digest(b.prompts),eligible_target=normal is not None,
                injection_policy='ALL_RANK_EVERY_ELIGIBLE_CALL',role='injector',selected_rank=None))
