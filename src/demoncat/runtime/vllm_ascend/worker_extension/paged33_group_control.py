"""Live isolated all-eligible C2. C3 propagated-input authority is absent."""
from copy import deepcopy
import os
import threading
import time
import uuid
import hashlib
import math
from pathlib import Path
import struct

from .paged33_group_evidence import GroupNormalEvidence,CONTROL_PHASES,PHASES,payload_signatures
from .paged33_s2_evidence import digest,normalized_input
from .serving_context import require
from fault_injection.coverage import (BoundaryTargetIdentity,Candidate,CandidateSet,CoverageCapability,
    Eligibility,EligibilityResult,ObservedCall,OperatorApiBoundary,Outcome,Phase,QualifiedTargetSet,
    RecoveryReport,TargetLease,TargetLosslessProof)
from fault_injection.coverage_session import (CoverageScope,CoverageSession,GroupRecoveryHooks,
    IndependentCallObserver,SessionBudget,WorkerHealthProof,target_id)


def complete_numeric_effect(before,after,root):
    """Read the two owned complete FP16 output bins, not a tensor sample."""
    values=[];root=Path(root).resolve(strict=True)
    for row in (before,after):
        path=Path(row['path'])
        require(row['shape']==[1,16,128] and row['dtype']=='torch.float16' and row['bytes']==4096
            and not path.is_symlink() and path.resolve(strict=True).is_relative_to(root)
            and path.stat().st_size==4096,'GROUP_COMPLETE_OWNED_OPERATOR_OUTPUT_REQUIRED')
        raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==row['sha256'],'GROUP_OPERATOR_OUTPUT_BYTES_CHANGED')
        value=[x[0] for x in struct.iter_unpack('<e',raw)]
        require(all(math.isfinite(x) for x in value),'GROUP_OPERATOR_OUTPUT_NONFINITE');values.append(value)
    return dict(numeric_changed_elements=sum(a!=b for a,b in zip(*values)),
        max_abs_difference=max(abs(a-b) for a,b in zip(*values)),
        before_sha256=before['sha256'],after_sha256=after['sha256'],complete_elements=2048)


class GroupC2Controller:
    def __init__(self,adapter,identity):
        self.adapter=adapter;self.targets=adapter.targets;self.normal=adapter.driver.normal_evidence
        require(type(self.normal) is GroupNormalEvidence,'GROUP_EXACT_NORMAL_OBSERVATIONS_REQUIRED')
        self.owner,self.public_generation=next(iter(adapter.deployment.leases))
        self.coverage_generation=1  # Separate namespace from the public Runtime ticket.
        self.identity=deepcopy(identity);self.thread=(os.getpid(),threading.get_ident(),threading.get_native_id())
        self.started=adapter.driver.case_started;self.budget=deepcopy(adapter.driver.budget)
        self.phase=None;self.request_id=None;self.index=0;self.records=[];self.controls=[];self.control_inputs=[]
        self.session=None;self.c2_proof=None;self.permit=None;self.held=None;self.decisions=[]
        self.off_pending=False;self.recovering=False;self.last_off=None;self.health=None
        self.first_error=None;self.errors=[];self.recovery_history=[];self.native_attempted=False
        self.phase_calls=[];self.expected_paths=[p for p,_,_ in self.targets.rows]
        self.admissions=[]
        self.finalized=False
        self.request_resource=None
        require(set(self.expected_paths)==set(self.normal.data()['targets']), 'GROUP_ALL_NORMAL_OBJECT_PATH_SET_DIFFERS')

    def check_owner(self):
        a=self.adapter;a.guard();self.targets.report()
        require(self.thread==(os.getpid(),threading.get_ident(),threading.get_native_id())
            and a.deployment.leases=={(self.owner,self.public_generation)}
            and a.worker.model_runner is a.runner and a.runner.model is a.model,'GROUP_CURRENT_OWNER_CHANGED')

    def failure(self,error):
        if self.first_error is None:self.first_error=repr(error)
        self.errors.append(dict(phase=self.phase,request_id=self.request_id,error=repr(error),first_error=self.first_error))
        if self.session is not None and self.health is None:
            self.off_pending=True;self.adapter.needs_off=True

    def begin_request(self,request_id,phase):
        self.check_owner()
        require(self.request_id is None and (phase==PHASES[self.index] if self.index<len(PHASES) else False)
            or self.recovering and self.request_id is None and phase in ('recovery_off','recovery_health'),'GROUP_REQUEST_PHASE_ORDER_CHANGED')
        require(not self.errors or self.recovering,'GROUP_FAILED_EXPERIMENT_CANNOT_RESUME_FAULTS')
        require(not self.off_pending or phase.startswith('off_') or phase=='recovery_off', 'GROUP_ACTUAL_OFF_REQUIRED')
        if phase.startswith('fault_'):
            require(self.session is not None and not self.held and self.permit is None,'GROUP_LIVE_CONTROLS_REQUIRED')
            self.request_resource=self.adapter.driver.check_resources(phase,self.index,'before_request')
        self.phase,self.request_id=phase,request_id;self.phase_calls=[];self.decisions=[]

    def _target(self,row,decode):
        key=(row['module_path'],decode)
        require(key in self.identities and row['module_object_id']==self.identities[key].module_object_identity
            and row['actual_operator']==self.identities[key].operator_api and row['occurrence']==0,
            'GROUP_ACTUAL_CALL_IDENTITY_CHANGED')
        return self.identities[key]

    def observe(self,row,decode,binding=None):
        self.check_owner();require(row['request_id']==self.request_id,'GROUP_ACTUAL_REQUEST_ID_DIFFERS')
        expected=(False,self.expected_paths[len(self.phase_calls)]) if len(self.phase_calls)<28 else \
            (True,self.expected_paths[len(self.phase_calls)-28]) if len(self.phase_calls)<56 else None
        require(expected==(decode,row['module_path']),'GROUP_FULL_NATURAL_CALL_ORDER_CHANGED')
        self.phase_calls.append((decode,row['module_path']))
        if self.session is None or not (self.phase.startswith(('fault_','off_')) or self.phase=='recovery_off'):return None
        off=not self.phase.startswith('fault_')
        stage=Phase.OFF_RECOVERY if off else Phase.POST_FIRST_ANOMALY if self.held else Phase.PRE_FAULT_PROPAGATION
        observed=ObservedCall(self._target(row,decode),uuid.uuid4().hex,self.request_id+(':decode' if decode else ':prefill'),
                             (self.request_id,),stage)
        ticket=self.session.observer.observe(observed,eligible=decode)
        row.update(coverage_call_id=observed.call_id,coverage_target_id=target_id(observed.target),
                   coverage_phase=stage.value,coverage_eligible=decode)
        decision=self.session.begin(ticket);self.decisions.append(decision)
        if decision.admitted:
            require(decode and self.held is None and binding is not None
                and row['module_path']==self.expected_paths[int(self.phase.split('_')[1])],
                'GROUP_ISOLATED_NEXT_ACTUAL_TARGET_REQUIRED')
            match=self.normal.match(binding)
            original=self.control_inputs[0][binding.module_path]
            require(not original.closed and all(getattr(original,k) is getattr(binding,k)
                    for k in ('worker','runner','model','module','impl'))
                and all(a is b for a,b in zip(original.caches,binding.caches))
                and normalized_input(original.before['signature'])==normalized_input(binding.before['signature']),
                'GROUP_CURRENT_INPUT_OR_LIVE_NORMAL_OWNER_CHANGED')
            proof=dict(owner=self.owner,public_generation=self.public_generation,coverage_generation=self.coverage_generation,
                request_id=self.request_id,phase=self.phase,target_path=binding.module_path,normal_match=match,
                binding_object_id=id(binding),target_set_sha256=self.targets.fingerprint,
                coverage_call_id=observed.call_id,coverage_target_id=target_id(observed.target),
                actual_observation=digest(dict(call_id=observed.call_id,target=repr(observed.target))),
                before_request_resource_sha256=self.request_resource['sha256'],
                control_request_ids=[r['request_id'] for r in self.controls])
            self.permit=dict(binding=binding,decision=decision,proof=proof,sha256=digest(proof),consumed=False)
            self.admissions.append(dict(proof=deepcopy(proof),sha256=digest(proof),consumed=False,returned=False))
            self.held=decision
        return decision

    def consume(self,binding):
        self.check_owner();p=self.permit
        require(p is not None and p['binding'] is binding and not p['consumed'] and self.held is p['decision']
            and self.phase.startswith('fault_') and digest(p['proof'])==p['sha256'], 'GROUP_FRESH_CURRENT_CALL_PERMIT_REQUIRED')
        self.normal.match(binding)
        require(binding._snapshot(self.adapter,self.public_generation)==binding.before,'GROUP_INPUT_CHANGED_AFTER_ADMISSION')
        resource=self.adapter.driver.check_resources(self.phase,self.index,'before_native')
        p['proof']['before_native_resource_sha256']=resource['sha256'];p['sha256']=digest(p['proof'])
        self.admissions[-1].update(proof=deepcopy(p['proof']),sha256=p['sha256'])
        p['consumed']=True;self.off_pending=True;self.native_attempted=True;self.adapter.needs_off=True
        self.admissions[-1]['consumed']=True
        return dict(sha256=p['sha256'],proof=deepcopy(p['proof']))

    def returned(self,decision,row):
        if decision is not None and not decision.admitted:
            require(row['original_calls']==1 and row['original_returned'] and row['inputs_unchanged'],
                    'GROUP_ORIGINAL_PASSTHROUGH_RETURN_REQUIRED')
            # Keep the observation pending until the actual consumer and model
            # return. Its receipt then hashes the complete saved call row.

    def _open(self):
        a=self.adapter;reference=self.controls[0]['capture'];last=self.controls[-1]['capture']
        build=self.normal.data();implementation=digest(build['source_pins']);actual=self.identity
        self.identities={};candidates=[]
        for forward in last['natural_request']['forwards']:
            decode=forward['decode']
            for row in forward['calls']:
                path=row['module_path'];boundary=OperatorApiBoundary(digest(dict(api=row['actual_operator'],decode=decode)),
                    implementation,'actual-target:'+digest(dict(path=path,module=row['module_object_id'],decode=decode)), 'torch_dispatch')
                identity=BoundaryTargetIdentity(actual['pid'],actual['process_start_identity'],actual['worker_start_identity'],
                    'INITIALIZED_WORKER',actual['rank'],'cpu' if a.deployment.simulation else 'npu',
                    None if a.deployment.simulation else actual['logical_device'],actual['model_instance_id'],id(a.model),
                    row['module_object_id'],('attention',),row['actual_operator'],'TND_Q1_PAGE16' if decode else 'ACTUAL_PREFILL',
                    boundary,'33:all-target-isolated-s2',(('implementation_sha256',implementation),))
                self.identities[(path,decode)]=identity
                candidates.append(Candidate(path+(':decode' if decode else ':prefill'),
                    EligibilityResult(Eligibility.SUPPORTED) if decode else
                    EligibilityResult(Eligibility.UNSUPPORTED,('ACTUAL_PREFILL_OUTSIDE_Q1_S2_SCOPE',)),identity))
        candidate_set=CandidateSet(tuple(candidates),True)
        require(len(candidate_set.candidates)==56 and len(candidate_set.eligible)==28,'GROUP_COMPLETE_DENOMINATOR_REQUIRED')
        leases=tuple(TargetLease(self.identities[(p,True)],self.owner,self.coverage_generation,'target:'+str(i),0,0)
                     for i,p in enumerate(self.expected_paths));proofs=[]
        for lease,path in zip(leases,self.expected_paths):
            for control in self.controls:
                logical=control['capture']['all_decode_boundaries'][path]['logical_input_binding'];ev=logical['evidence']
                before=digest(normalized_input(ev['before']['signature']));after=digest(normalized_input(ev['after']['signature']))
                ref=reference['all_decode_boundaries'][path]['logical_input_binding']['evidence']['outputs'][0]['sha256']
                proofs.append(TargetLosslessProof(lease,control['phase'],digest(logical),(before,),(after,),ref,
                    ev['outputs'][0]['sha256'],logical['actual_original_calls'],ev['before']==ev['after'],True))
        cap=CoverageCapability(single_target=True,multi_target=True,every_match=False,max_concurrency=1,
            max_resources=0,max_host_bytes=0,group_recovery=True,recovery_scope='target_set',
            stop_policy='bounded_session',lossless_modes=CONTROL_PHASES)
        group=QualifiedTargetSet(candidate_set,leases,tuple(proofs),cap,selection='all_eligible',trigger='once')
        scope=CoverageScope('33.paged_decode',digest(a.deployment.prompt),digest(build['execution_contract']),implementation,
                            evidence_kind='cpu_contract' if a.deployment.simulation else 'natural_worker')
        observer=IndependentCallObserver(source='independent_workload_observer',observer_id=uuid.uuid4().hex,
            implementation_sha256=implementation,workload_sha256=scope.workload_sha256,max_calls=56*58)
        observer.seal_discovery(candidate_set,observed_targets=tuple(self.identities.values()),artifact_sha256=digest(last),
            expected_workers=(leases[0].target.worker,),observed_workers=(leases[0].target.worker,))
        hooks=GroupRecoveryHooks(self._stop,self._drain,self._recover,self._health)
        remaining=max(.001,self.budget['case_seconds']-(time.monotonic()-self.started))
        self.session=CoverageSession(group=group,scope=scope,observer=observer,
            budget=SessionBudget(56*58,remaining),recovery=hooks)

    def complete_request(self,capture,artifact):
        self.check_owner();row=capture['natural_request'];phase=self.phase
        require((row['phase'],row['request_id'])==(phase,self.request_id) and len(self.phase_calls)==56,
                'GROUP_COMPLETE_CURRENT_56_CALL_REQUEST_REQUIRED')
        if not self.adapter.deployment.simulation:
            require(all(not b['errors'] and b['all_checked_values_unchanged'] is True
                and b['runtime_observation'].get('observation_gaps')==[]
                and b['mapped_api_libraries'].get('complete') is True
                for b in capture['all_decode_boundaries'].values()),'GROUP_COMPLETE_ACTUAL_BOUNDARY_OBSERVATION_REQUIRED')
        baseline=self.controls[0]['capture'] if self.controls else capture
        for left,right in zip(baseline['natural_request']['forwards'],row['forwards']):
            require(left['input']==right['input'],'GROUP_CURRENT_ROOT_INPUT_DIFFERS')
            if not right['decode']:require(left['hidden']==right['hidden'] and left['logits']==right['logits'],
                                           'GROUP_NON_TARGET_PREFILL_CHANGED')
        values=payload_signatures(capture)
        if not phase.startswith('fault_'):
            require(values==self.normal.data()['payloads']==payload_signatures(baseline),
                    'GROUP_COMPLETE_NORMAL_OR_OFF_PAYLOADS_DIFFER')
            for record in self.adapter.records.values():self.normal.match(record['binding'])
        calls=capture['native_calls'];applied=[r for r in calls if r['selection_applied']]
        require(len(calls)==28 and len(applied)==int(phase.startswith('fault_')),
                'GROUP_ACTUAL_NATIVE_SELECTION_COUNT_DIFFERS')
        if phase in CONTROL_PHASES:
            self.controls.append(dict(phase=phase,request_id=self.request_id,capture=capture,artifact=artifact))
            self.control_inputs.append({p:r['binding'] for p,r in self.adapter.records.items()})
            if phase=='qualification_off':self._open()
        elif phase.startswith('fault_'):
            p=self.permit;require(p is not None and p['consumed'] and len(applied)==1
                and applied[0]['admission_sha256']==p['sha256'] and applied[0]['module_path']==p['binding'].module_path,
                'GROUP_ACTUAL_NATIVE_PERMIT_RECEIPT_DIFFERS')
            name=p['binding'].module_path+'/output';changed=values[name]!=payload_signatures(baseline)[name]
            numeric=None
            if not self.adapter.deployment.simulation:
                path=p['binding'].module_path
                numeric=complete_numeric_effect(baseline['attention'][path]['output'],capture['attention'][path]['output'],self.adapter.root_folder)
                changed=changed and numeric['numeric_changed_elements']>0
            # A successful strategy switch without numerical effect remains
            # in the raw record, but cannot be promoted to a fault HIT.
            require(changed,'GROUP_POLICY_SELECTED_WITHOUT_REQUIRED_ATTENTION_EFFECT')
            self.session.finish(self.held,Outcome.HIT,execution_sha256=digest(applied[0]),
                effect_sha256=digest(dict(before=payload_signatures(baseline)[name],after=values[name],artifact=artifact)),
                native_applied=True,original_calls=1)
            self.admissions[-1].update(returned=True,native_call_id=applied[0]['call_id'],artifact=artifact,
                attention_output_changed=True,numeric_effect=numeric,
                logits_changed=values['logits']!=payload_signatures(baseline)['logits'])
            self.decisions.remove(self.held);self.held=None;self.permit=None
        elif phase.startswith('off_') or phase=='recovery_off':
            require(self.off_pending,'GROUP_OFF_WITHOUT_PENDING_EFFECT')
            self.last_off=artifact;self.off_pending=False;self.adapter.needs_off=False
        elif phase in ('cleanup_health','recovery_health'):self.health=artifact
        completed={call.get('coverage_call_id'):call for forward in row['forwards'] for call in forward['calls']}
        for decision in list(self.decisions):
            require(not decision.admitted,'GROUP_UNFINISHED_ADMITTED_DECISION')
            call=completed[decision.ticket.call.call_id]
            require(call['original_calls']==1 and call['original_returned'] and call['inputs_unchanged']
                and (not decision.ticket.eligible or call['consumer_returned']),'GROUP_ACTUAL_PASSTHROUGH_CONSUMER_REQUIRED')
            self.session.finish(decision,Outcome.PASSTHROUGH,execution_sha256=digest(call),original_calls=1)
            self.decisions.remove(decision)
        require(not self.decisions,'GROUP_UNFINISHED_ACTUAL_CALL_DECISION')
        self.records.append(artifact);self.request_id=None
        if not self.recovering:self.index+=1

    def _stop(self,lease):self.check_owner();self.adapter.driver.stop()
    def _drain(self,lease,remaining):
        self.check_owner();state=self.adapter.driver.stop()
        return not self.off_pending and not self.adapter.root_active and not self.adapter.callbacks \
            and not state['scope_owned'] and not state['scope_active'] and not state['environment_owned']
    def _recover(self,lease):
        require(self.health and self.last_off and not self.off_pending,'GROUP_HEALTH_AND_FULL_OFF_BEFORE_RELEASE_REQUIRED')
        state=self.adapter.state();require(not any(state[k] for k in ('owned_hooks','owned_api_wrappers','inflight','owned_input_references')),
                                          'GROUP_ZERO_OWNED_RESOURCES_BEFORE_RELEASE_REQUIRED')
        path=next(p for p in self.expected_paths if self.identities[(p,True)]==lease.target)
        return RecoveryReport(lease,True,True,True,proof_sha256=digest(self.health),
                              off_output_sha256=self.normal.data()['payloads'][path+'/output']['sha256'])
    def _health(self,worker,owner,generation):
        self.check_owner();self._recover(self.session.group.leases[0]);value=self.normal.data()['payloads']['logits']['sha256']
        return WorkerHealthProof(worker,owner,generation,digest(self.health),value,value,0,0,0,0,0)

    def check_cleanup_ready(self):
        """Keep the actual observers installed until required OFF/health can run."""
        self.check_owner();require(not self.off_pending,'GROUP_OFF_PENDING_OWNER_RETAINED')
        if self.session is None:return
        if self.finalized:return
        if self.errors:
            # Failed experiments never issue a C2 proof. Cleanup still needs
            # a real OFF and a complete healthy request before lease release.
            require(self.health and self.last_off,'GROUP_FAILED_EXPERIMENT_RECOVERY_HEALTH_REQUIRED')
        require(self.index==len(PHASES) or self.recovering and self.health,'GROUP_PROTOCOL_INCOMPLETE')

    def finalize(self):
        self.check_cleanup_ready()
        if self.session is None or self.finalized:return
        if self.session.observer.evidence is None:self.session.observer.seal(artifact_sha256=digest(self.records),complete=not self.errors)
        recovery=self.session.disable(timeout_seconds=30);self.recovery_history.append(deepcopy(recovery))
        require(recovery['complete'],'GROUP_COVERAGE_RECOVERY_PENDING')
        if not self.errors and self.c2_proof is None:self.c2_proof=self.session.c2_proof()
        self.finalized=True

    def prepare_recovery(self):
        self.check_owner();a=self.adapter
        require(self.off_pending and not a.root_active and not a.callbacks,'GROUP_RECOVERY_REQUIRES_UNWOUND_ACTUAL_FAULT')
        state=a.driver.stop();require(not state['scope_owned'] and not state['scope_active'] and not state['environment_owned'],
                                     'GROUP_NATIVE_DRAIN_BEFORE_FRESH_OFF_REQUIRED')
        # Settle only failed observations, never a synthetic HIT or repeated F.
        for decision in list(self.decisions):
            self.session.finish(decision,Outcome.FAILED,original_calls=0)
        self.decisions.clear();self.held=None;self.permit=None;self.request_id=None;self.recovering=True

    def summary(self):
        return dict(owner=self.owner,public_generation=self.public_generation,coverage_generation=self.coverage_generation,
            phase=self.phase,request_id=self.request_id,completed=len(self.records),off_pending=self.off_pending,
            first_error=self.first_error,failures=deepcopy(self.errors),case_budget=deepcopy(self.budget),
            C2=self.c2_proof is not None and not self.adapter.deployment.simulation,C3=False,
            C3_blocker='CURRENT_PROPAGATED_INPUT_STOCK_OWNER_AND_EXACT_NORMAL_NOT_PROVEN',
            live_c2_proof=self.c2_proof.report() if self.c2_proof is not None else None)
