"""One-use live C2 admission for the separate group Native package."""
import time
from copy import deepcopy
from .serving_context import require
from .paged33_group_normal import GroupNormalNativeCall,GroupNormalWorkerSession,validate_group_host,verify_bundle,VARIANT
from .paged33_s2_evidence import digest


class GroupEffectsNativeCall(GroupNormalNativeCall):
    """A permit is retained by the adapter before any uncertain Native call."""
    public_group_protocol=True

    def __init__(self,*,normal_evidence,case_budget_seconds,case_started_monotonic,worker_identity,**kwargs):
        from .paged33_group_evidence import GroupNormalEvidence
        require(type(normal_evidence) is GroupNormalEvidence and not normal_evidence.simulation,
                'GROUP_FRESH_FINAL_PACKAGE_NORMAL_OBJECT_REQUIRED')
        self.normal_evidence=normal_evidence
        self.worker_identity=deepcopy(worker_identity)
        self.budget=normal_evidence.cost_gate(case_budget_seconds)
        self._budget_fingerprint=digest(self.budget)
        require(type(case_started_monotonic) in (int,float) and 0<case_started_monotonic<=time.monotonic(),
                'GROUP_TRUSTED_CASE_START_CLOCK_REQUIRED')
        self.case_started=case_started_monotonic
        self.controller=None
        super().__init__(**kwargs)
        data=normal_evidence.data()
        require(data['completion_sha256']==self.fingerprint['completion_sha256']
            and data['observer_receipt_sha256']==kwargs['observer_receipt_sha256'],
            'GROUP_FINAL_PACKAGE_OR_OBSERVER_DIFFERS_FROM_NORMAL')
        from fault_injection.native_adapters.attention28.group_resource_budget import GroupRuntimeLogScope
        from fault_injection.native_adapters.attention28.runtime_log_index import RuntimeLogIndex
        self.attempt=self.runtime_dir.parent
        require(self.folder.parent==self.attempt,'GROUP_RUNTIME_AND_ARTIFACT_ATTEMPT_DIFFER')
        self.resource_scope=GroupRuntimeLogScope.from_verified_normal(normal_evidence,
            build_sha256=self.fingerprint['completion_sha256'],attempt=self.attempt)
        self.runtime_index=RuntimeLogIndex(self.runtime_dir,group_scope=self.resource_scope)
        self.resource_checks=[];self.first_resource_error=None;self.phase_resources=[];self.resource_phase=None

    def start_resource_phase(self,phase,index):
        self._check();require(self.resource_phase is None,'GROUP_RESOURCE_PHASE_STILL_PENDING')
        self.runtime_index.scope(self.runtime_dir,'0'*32)
        self.resource_phase=dict(phase=phase,phase_index=index,started_monotonic=time.monotonic(),
            start=self.runtime_index.report())

    def finish_resource_phase(self,phase):
        self._check();require(self.resource_phase is not None and self.resource_phase['phase']==phase,
                              'GROUP_RESOURCE_PHASE_NOT_CURRENT')
        self.runtime_index.scope(self.runtime_dir,'0'*32)
        row=deepcopy(self.resource_phase);row.update(finished_monotonic=time.monotonic(),end=self.runtime_index.report())
        self.phase_resources.append(row);self.resource_phase=None

    def abandon_resource_phase(self):
        # Preserve the partial measurement; it never becomes a complete normal
        # sample or permission to resume faults after recovery.
        if self.resource_phase is not None:
            self.phase_resources.append(dict(self.resource_phase,partial=True));self.resource_phase=None

    def check_resources(self,phase,index,stage):
        """Refresh the immutable log prefix and actual CPU budgets before F."""
        from fault_injection.native_adapters.attention28 import group_resource_budget as R
        self._check();self.controller.check_owner()
        require(digest(self.budget)==self._budget_fingerprint,'GROUP_ORIGINAL_RESOURCE_PLAN_CHANGED')
        require(stage in ('before_request','before_native') and phase==f'fault_{(index-3)//2:02d}',
                'GROUP_RESOURCE_CHECK_PHASE_REQUIRED')
        row=dict(phase=phase,phase_index=index,stage=stage,owner=self.controller.owner,
            public_generation=self.controller.public_generation,scope=self.resource_scope.check(self.runtime_dir),
            plan_sha256=digest(self.budget),passed=False)
        try:
            for actual in self.phase_resources:
                require(not actual.get('partial'),'NO_F_CAPACITY_REFUSED:PARTIAL_RESOURCE_PHASE')
                increment=actual['end']['prefix_bytes']-actual['start']['prefix_bytes']
                require(0<=increment<=2*self.budget['maximum_phase_increment_bytes'],
                        'NO_F_CAPACITY_REFUSED:ACTUAL_PHASE_LOG_GROWTH_EXCEEDS_BOUND')
                elapsed=actual['finished_monotonic']-actual['started_monotonic']
                bound=2*(self.budget['constant_phase_seconds']+self.budget['indexed_history_seconds']*
                    max(1,actual['end']['prefix_bytes']/self.budget['maximum_normal_prefix_bytes']))
                require(0<elapsed<=bound,'NO_F_TIME_REFUSED:ACTUAL_PHASE_TIME_EXCEEDS_BOUND')
            row['completed_phase_resources_sha256']=digest(self.phase_resources)
            self.runtime_index.scope(self.runtime_dir,'0'*32)
            row['runtime_index']=self.runtime_index.report()
            resources=R.host_resources(self.attempt);now=time.monotonic()
            row.update(started_monotonic=self.case_started,checked_monotonic=now)
            row['check']=R.before_fault(self.budget,phase_index=index,started_monotonic=self.case_started,
                runtime_bytes=row['runtime_index']['prefix_bytes'],resources=resources,now=now)
            row['passed']=True
        except BaseException as error:
            if self.first_resource_error is None:self.first_resource_error=repr(error)
            row.update(error=repr(error),first_error=self.first_resource_error)
            raise
        finally:
            self.resource_checks.append(dict(receipt=deepcopy(row),sha256=digest(row)))
        return deepcopy(self.resource_checks[-1])

    def bind_controller(self,controller):
        from .paged33_group_control import GroupC2Controller
        require(type(controller) is GroupC2Controller and self.controller is None and not self.calls,
                'GROUP_EXACT_UNUSED_CONTROLLER_REQUIRED')
        self.controller=controller;self.owner_token=controller.owner

    def _admission(self,phase,binding,scope,owner,generation):
        require(self.controller is not None and self.controller.adapter is owner
            and self.controller.targets is scope and self.controller.public_generation==generation,
            'GROUP_CURRENT_CONTROLLER_OWNER_REQUIRED')
        self.controller.check_owner()
        if phase!='fault':return super()._admission(phase,binding,scope,owner,generation)
        permit=self.controller.consume(binding)
        return permit['sha256'],self.controller.coverage_generation

    def state(self):
        state=super().state()
        state.update(group_protocol=True,case_budget=deepcopy(getattr(self,'budget',None)),
            normal_evidence_sha256=getattr(getattr(self,'normal_evidence',None),'fingerprint',None),
            off_pending=bool(getattr(self,'controller',None) and self.controller.off_pending))
        state.update(resource_check_count=len(getattr(self,'resource_checks',[])),
            resource_checks_sha256=digest(getattr(self,'resource_checks',[])),
            first_resource_error=getattr(self,'first_resource_error',None))
        state['retained_input_objects']=sum(not r['binding'].closed for r in
            getattr(getattr(getattr(self,'controller',None),'adapter',None),'owned_records',[]) if 'binding' in r)
        state['effect_attempted']=bool(getattr(self,'controller',None) and self.controller.native_attempted)
        state['off_confirmed']=bool(getattr(self,'controller',None) and self.controller.last_off and not self.controller.off_pending)
        return state
