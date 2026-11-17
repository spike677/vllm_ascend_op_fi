"""Common all-target coverage for actual source-bound live KV consumers.

Each Native slot lease drains/restores immediately after its real FIA call.
Logical target reservations remain held through group OFF, removal of all owned
hooks/APIs, and a separate normal-request full-output health observation.
"""
import hashlib
import inspect
import os
from pathlib import Path
import struct
import threading
import time
import uuid

import torch

from fault_injection.coverage import (BoundaryTargetIdentity, Candidate, CoverageCapability, Eligibility,
    EligibilityResult, ObservedCall, OperatorApiBoundary, Outcome, Phase, QualifiedTargetSet,
    RecoveryReport, TargetLease, TargetLosslessProof)
from fault_injection.coverage_session import (CoverageScope, CoverageSession, GroupRecoveryHooks,
    IndependentCallObserver, SessionBudget, WorkerHealthProof, target_id)
from fault_injection.runtime_discovery import RuntimeCandidateInventory
from fault_injection.deployments.postwrite import _cpu
from fault_injection.deployments.postwrite_evidence import digest, snapshot, output_digest
from fault_injection.deployments.postwrite_kv import LiveKVSlotLease
from .postwrite41 import _output_hash
from .serving_context import require


def _span_hash(transport, plan):
    return hashlib.sha256(snapshot(transport, plan.cache, plan.element_offset, plan.elements)).hexdigest()


class KVGroupCoverage:
    def __init__(self, provider, scope, *, max_seconds=180, requests=2):
        require(isinstance(scope, CoverageScope) and scope.mechanism == '41.tail_column', 'KV_GROUP_SCOPE_REQUIRED')
        require(scope.evidence_kind in ('natural_worker', 'cpu_contract'), 'KV_GROUP_ACTUAL_WORKER_SCOPE_REQUIRED')
        require(type(requests) is int and requests == 2 and 0 < max_seconds <= 1500, 'KV_GROUP_BOUNDED_WORKLOAD_REQUIRED')
        history = provider.history
        require(history and history[-1]['phase'] == 'off' and history[-1]['exact_healthy_output']
                and any(row['phase'] == 'fault' and row['new_fault_hits'] == 1
                        and not row['exact_healthy_output'] and row['recovery_confirmed'] for row in history)
                and provider.lease is None, 'KV_GROUP_SAME_PROCESS_C1_EFFECT_OFF_REQUIRED')
        require({id(module) for _, module in provider.candidates}
                == {id(module) for _, module in provider.structural_candidates},
                'KV_GROUP_UNREVIEWED_ATTENTION_IMPL_IN_STRUCTURAL_UNIVERSE')
        if scope.evidence_kind == 'natural_worker':
            require(getattr(provider.transport, '_snapshot', None) is not None, 'KV_GROUP_FRESH_READ_SPAN_BUILD_REQUIRED')
            schemas = torch._C._jit_get_schemas_for_operator('npu::npu_fused_infer_attention_score')
            schemas = [str(schema) for schema in schemas if not schema.overload_name]
            require(len(schemas) == 1, 'KV_GROUP_ACTUAL_FIA_SCHEMA_REQUIRED')
            self.schema = schemas[0]
            require(type(provider.worker.rank) is int and provider.worker.rank == 0, 'KV_GROUP_ACTUAL_TP1_RANK_REQUIRED')
            self.rank = provider.worker.rank
        else:
            self.schema, self.rank = 'CPU_CONTRACT:actual_cpu_consumer', 0
        self.p, self.scope, self.requests, self.max_seconds = provider, scope, requests, max_seconds
        self.owner, self.generation = str(provider.owner), 0
        self.module_universe = {id(module) for _, module in provider.candidates}
        self.baseline_hash = history[0]['model_output_sha256']
        self.prefixes = history[0]['actual_scheduler']['prefixes']
        self.stage, self.trigger = 'discovery', None
        self.discovery, self.identities, self.reference = [], {}, {}
        self.current_records, self.lossless, self.off_records = {}, {}, {}
        self.observer = self.session = self.group = None
        self.c2_live_proof = None
        self.fault_requests = 0
        self.anomaly_seen = False
        self.failures, self.executions, self.completed_requests = [], [], []
        self.health_handles, self.health_artifact, self.health_current = [], None, None
        self.source_sha256 = hashlib.sha256(Path(inspect.getsourcefile(type(self))).read_bytes()).hexdigest()
        from fault_injection.deployments import postwrite_kv_inventory
        self.observer_source_sha256 = hashlib.sha256(Path(inspect.getsourcefile(postwrite_kv_inventory)).read_bytes()).hexdigest()
        provider.coverage = self

    def _identity(self, trace, plan):
        device = plan.cache.device
        kind = 'INITIALIZED_WORKER' if self.scope.evidence_kind == 'natural_worker' else 'MODEL_LOCAL'
        return BoundaryTargetIdentity(os.getpid(), self.p.process_start,
            'actual-worker:' + str(id(self.p.worker)), kind, self.rank, device.type,
            None if device.type == 'cpu' else device.index, str(id(self.p.model)), id(self.p.model),
            trace['module_object_id'], ('attention',), 'torch_npu.' + trace['operator'],
            digest(dict(target_key=trace['target_key'], shape=list(plan.cache.shape), dtype=str(plan.cache.dtype))),
            OperatorApiBoundary(hashlib.sha256(self.schema.encode()).hexdigest(), self.scope.implementation_sha256,
                'live-kv-target:' + digest(trace['target_key']), 'torch_dispatch'),
            '41.tail_column:live-kv-group-v1',
            (('native_library_sha256', getattr(self.p.transport, 'sha256', self.scope.implementation_sha256)),
             ('writer_consumer_source_sha256', digest(self.p.source))))

    def authorize_phase(self, phase):
        require(not self.health_handles and self.p.current is None and self.p.lease is None,
                'KV_GROUP_REQUEST_OR_HEALTH_ACTIVE')
        if phase == 'group_observe':require(self.stage == 'discovery' and len(self.discovery) < 2, 'KV_GROUP_DISCOVERY_GATE')
        elif phase == 'group_lossless':require(self.stage == 'lossless_pending', 'KV_GROUP_LOSSLESS_GATE')
        elif phase == 'group_fault':require(self.stage == 'enabled' and self.fault_requests < self.requests, 'KV_GROUP_REQUEST_BUDGET')
        elif phase == 'group_off':require(self.stage in ('enabled', 'failed'), 'KV_GROUP_OFF_GATE')
        else:raise RuntimeError('KV_GROUP_UNKNOWN_PHASE')
        self.current_records = {}
        self.anomaly_seen = False

    def _before(self, plan, actual):
        query=actual['query']
        require(query.dtype==torch.bfloat16,'KV_GROUP_BF16_QUERY_REQUIRED')
        slot=snapshot(self.p.transport,plan.cache,plan.element_offset,plan.elements)
        query_bytes=snapshot(self.p.transport,query)
        return dict(slot_sha256=hashlib.sha256(slot).hexdigest(),
            query_sha256=digest(dict(shape=list(query.shape),stride=list(query.stride()),dtype=str(query.dtype),
                                    bytes_sha256=hashlib.sha256(query_bytes).hexdigest())),
            slot_nonfinite=sum((bits & 0x7f80)==0x7f80 for bits, in struct.iter_unpack('<H',slot)),
            query_nonfinite=sum((bits & 0x7f80)==0x7f80 for bits, in struct.iter_unpack('<H',query_bytes)))

    def consume(self, trace, row, plan, actual, original, args, kwargs):
        require(self.p.current['prefixes'] == self.prefixes, 'KV_GROUP_FIXED_LOGICAL_WORKLOAD_CHANGED')
        key = trace['target_key']
        target = self.identities.get(key) or self._identity(trace, plan)
        phase = self.p.phase
        require(key not in self.current_records, 'KV_GROUP_DUPLICATE_TARGET_CALL')
        evidence = dict(target_id=target_id(target), target_key=key, call_id=trace['call_id'],
            forward_id=trace['forward_id'], request_ids=trace['request_ids'], boundary=plan.proof,
            native_generation=self.generation,request_generation=self.p.generation,input_before=self._before(plan,actual))
        self.current_records[key] = evidence
        decision = None
        if phase in ('group_fault', 'group_off'):
            stage = Phase.OFF_RECOVERY if phase == 'group_off' else (
                Phase.POST_FIRST_ANOMALY if self.anomaly_seen else Phase.PRE_FAULT_PROPAGATION)
            ticket = self.observer.observe(ObservedCall(target, trace['call_id'], trace['forward_id'],
                tuple(trace['request_ids']), stage), eligible=True,
                propagation_added=key not in self.identities and stage == Phase.POST_FIRST_ANOMALY)
            decision = self.session.begin(ticket)
        do_native = phase == 'group_lossless' or (decision is not None and decision.admitted)
        fault = phase == 'group_fault' and not self.baseline_control and do_native
        applied, finished = False, False
        try:
            if evidence['input_before']['slot_nonfinite'] or evidence['input_before']['query_nonfinite']:
                result=original(*args,**kwargs)
                evidence['refused_consumer_output_sha256']=output_digest(self.p.transport,result)
                raise RuntimeError('KV_GROUP_NONFINITE_INPUT_REFUSED')
            if do_native:
                require(self.p.lease is None, 'KV_GROUP_PRIOR_NATIVE_LEASE_ACTIVE')
                self.p.lease = LiveKVSlotLease(self.p.transport,plan,self.p.owner,self.generation)
                self.p.lease.arm(fault=fault)
                applied = fault
                evidence['slot_during_sha256'] = _span_hash(self.p.transport, plan)
                result = self.p.lease.consume(original, args, kwargs)
                evidence['native'] = dict(self.p.lease.report)
                row['intervention'] = self.p.lease.report
                require(not self.p.lease.live, 'KV_GROUP_NATIVE_LEASE_NOT_RELEASED')
                self.p.lease = None
            else:
                result = original(*args, **kwargs)
            evidence.update(input_after=self._before(plan, actual),
                consumer_output_sha256=output_digest(self.p.transport, result),
                original_calls=trace['original_calls'], original_returned=trace['original_returned'],
                native_applied=applied, baseline_control=getattr(self, 'baseline_control', False))
            require(evidence['input_before'] == evidence['input_after'], 'KV_GROUP_INPUT_OR_SLOT_NOT_RESTORED')
            if phase == 'group_observe':
                self.identities.setdefault(key, target)
            elif phase == 'group_lossless':
                require(key in self.reference and evidence['consumer_output_sha256'] == self.reference[key]['consumer_output_sha256']
                        and evidence['input_before'] == self.reference[key]['input_before'],
                        'KV_GROUP_TARGET_LOSSLESS_BYTES_OR_CONSUMER_MISMATCH')
                self.lossless[key] = evidence
            elif phase == 'group_off':
                require(key in self.reference and evidence['consumer_output_sha256'] == self.reference[key]['consumer_output_sha256']
                        and evidence['input_before'] == self.reference[key]['input_before'], 'KV_GROUP_TARGET_OFF_MISMATCH')
                self.off_records[key] = evidence
            if decision is not None:
                self.session.finish(decision, Outcome.HIT if applied else Outcome.PASSTHROUGH,
                    execution_sha256=digest(evidence), effect_sha256=evidence.get('slot_during_sha256'),
                    native_applied=applied, original_calls=trace['original_calls'])
                finished = True
                if decision.admitted:self.executions.append(evidence)
                if applied:
                    self.p.pending_run['new_fault_hits'] += 1
                    self.anomaly_seen = True
                if phase != 'group_off' and decision.reason in ('SESSION_BUDGET_EXHAUSTED', 'SESSION_STOPPED'):
                    raise RuntimeError('KV_GROUP_BOUNDED_STOP:' + decision.reason)
            return result
        except BaseException as error:
            evidence.update(error=repr(error), original_calls=trace['original_calls'], native_applied=applied)
            # finish is unique even when a post-finish budget check raises.
            if decision is not None and not finished:
                try:
                    self.session.finish(decision,Outcome.FAILED,execution_sha256=digest(evidence),
                                        native_applied=applied,original_calls=trace['original_calls'])
                except BaseException as receipt_error:
                    evidence['coverage_receipt_error']=repr(receipt_error)
            self.failures.append(evidence)
            self.stage = 'failed'
            raise

    def request_finished(self, run, output):
        trace = run['independent_inventory']
        require(trace['complete'] and set(map(int, trace['module_calls'])) == self.module_universe
                and all(count == 1 for count in trace['module_calls'].values()), 'KV_GROUP_MODULE_DENOMINATOR_INCOMPLETE')
        require(len(trace['calls']) == len(self.module_universe)
                and all(row['original_calls'] == 1 and row['original_returned'] for row in trace['calls']),
                'KV_GROUP_INDEPENDENT_CALL_DENOMINATOR_INCOMPLETE')
        require(len(trace['writers']) == len(self.module_universe)
                and {row['module_object_id'] for row in trace['writers']} == self.module_universe,
                'KV_GROUP_INDEPENDENT_WRITER_DENOMINATOR_INCOMPLETE')
        phase = run['phase']
        if phase == 'group_observe':
            require(run['model_output_sha256'] == self.baseline_hash, 'KV_GROUP_HEALTHY_ROOT_OUTPUT_CHANGED')
            stable = {key:dict(input_before=value['input_before'], consumer_output_sha256=value['consumer_output_sha256'])
                      for key,value in self.current_records.items()}
            if self.discovery:require(stable == self.reference, 'KV_GROUP_HEALTHY_TARGET_BYTES_CHANGED')
            else:self.reference = stable
            self.discovery.append(dict(inventory=trace, byte_evidence=dict(self.current_records)))
            if len(self.discovery) == 2:self.stage = 'ready'
        else:
            require(set(self.current_records) == set(self.identities), 'KV_GROUP_ACTUAL_TARGET_SET_CHANGED')
            if phase == 'group_lossless':
                require(run['model_output_sha256'] == self.baseline_hash and set(self.lossless) == set(self.identities),
                        'KV_GROUP_ALL_TARGET_LOSSLESS_INCOMPLETE')
                self._open_session(run)
                self.stage = 'enabled'
            elif phase == 'group_fault':self.fault_requests += 1
            elif phase == 'group_off':
                require(run['model_output_sha256'] == self.baseline_hash and set(self.off_records) == set(self.identities),
                        'KV_GROUP_ALL_TARGET_OFF_INCOMPLETE')
                self.off_hash = run['model_output_sha256']
                self.stage = 'off_complete'
        self.completed_requests.append(dict(phase=phase, inventory=trace, model_output_sha256=run['model_output_sha256'],
                                            byte_evidence=list(self.current_records.values())))

    def prepare(self, trigger, *, baseline_control=False):
        require(trigger in ('once', 'every_match') and type(baseline_control) is bool, 'KV_GROUP_TRIGGER_REQUIRED')
        require(self.stage in ('ready', 'finished') and self.p.lease is None, 'KV_GROUP_DISCOVERY_OR_PRIOR_RECOVERY_REQUIRED')
        if trigger == 'every_match':require(self.c2_live_proof is not None and not baseline_control, 'KV_GROUP_LIVE_C2_PROOF_REQUIRED')
        self.trigger, self.baseline_control = trigger, baseline_control
        self.generation=max(self.generation+1,self.p.generation+1)
        self.lossless, self.off_records, self.executions, self.failures, self.completed_requests = {}, {}, [], [], []
        self.fault_requests, self.off_hash, self.health_artifact = 0, None, None
        self.observer = IndependentCallObserver(source='independent_workload_observer', observer_id='kv-observer:' + uuid.uuid4().hex,
            implementation_sha256=self.observer_source_sha256, workload_sha256=self.scope.workload_sha256,
            max_calls=(self.requests + 2) * len(self.identities))
        workers = tuple(dict.fromkeys(target.worker for target in self.identities.values()))
        inventory = RuntimeCandidateInventory(expected_modules=self.module_universe, expected_workers=workers)
        for discovery in self.discovery:
            frame = discovery['inventory']; forward = frame['forward_id']
            for row in frame['calls']:
                target = self.identities[row['target_key']]
                inventory.observe(Candidate('live-kv:' + digest(row['target_key']), EligibilityResult(Eligibility.SUPPORTED), target),
                                  forward_id=forward)
            inventory.finish_forward(forward, healthy=True)
        serialized = [dict(inventory=item['inventory'], byte_evidence=list(item['byte_evidence'].values())) for item in self.discovery]
        self.candidates = inventory.seal(self.observer, artifact_sha256=digest(serialized))
        require(self.candidates.discovery_complete, 'KV_GROUP_DISCOVERY_INCOMPLETE')
        self.stage = 'lossless_pending'
        if not self.p.installed:self.p.install()
        return dict(trigger=trigger, targets=len(self.identities), requests=self.requests,
                    baseline_control=baseline_control, generation=self.generation)

    def _open_session(self, run):
        leases = tuple(TargetLease(target, self.owner, self.generation, 'live-kv:' + target_id(target), 0,
            2 * self.lossless[key]['boundary']['span_elements'] * 2) for key,target in self.identities.items())
        keys = {target:key for key,target in self.identities.items()}
        proofs = []
        for lease in leases:
            evidence = self.lossless[keys[lease.target]]
            proofs.append(TargetLosslessProof(lease, 'live_slot_and_consumer', digest(evidence),
                tuple(evidence['input_before'][name] for name in ('slot_sha256','query_sha256')),
                tuple(evidence['input_after'][name] for name in ('slot_sha256','query_sha256')),
                self.baseline_hash, run['model_output_sha256'], evidence['original_calls'], True,
                evidence['native']['released']))
        capability = CoverageCapability(single_target=True,multi_target=True,every_match=True,max_concurrency=1,
            max_resources=0,max_host_bytes=sum(lease.host_bytes for lease in leases),group_recovery=True,
            recovery_scope='target_set',stop_policy='bounded_session',lossless_modes=('live_slot_and_consumer',))
        self.group = QualifiedTargetSet(self.candidates, leases, tuple(proofs), capability,
                                        selection='all_eligible', trigger=self.trigger)
        self.session = CoverageSession(group=self.group,scope=self.scope,observer=self.observer,
            budget=SessionBudget(self.requests * len(leases),self.max_seconds,1),recovery=self._hooks(),
            c2_proof=self.c2_live_proof if self.trigger=='every_match' else None)

    def _hooks(self):
        def stop(lease):self.stage='stopping'
        def drain(lease, remaining):
            begin=time.monotonic()
            if self.scope.evidence_kind=='natural_worker':torch.npu.synchronize()
            return self.p.lease is None and self.p.current is None and time.monotonic()-begin<=remaining
        def recover(lease):
            require(self.off_hash==self.baseline_hash and self.p.lease is None,'KV_GROUP_OFF_AND_NATIVE_RELEASE_REQUIRED')
            key=next(key for key,target in self.identities.items() if target==lease.target)
            return RecoveryReport(lease,True,True,True,proof_sha256=digest(self.off_records[key]),off_output_sha256=self.off_hash)
        def health(worker,owner,generation):
            # Uninstall only after the common whole-target drain/recovery barrier.
            if self.p.installed:self.p.close()
            require(self.health_artifact is not None,'KV_GROUP_NORMAL_REQUEST_HEALTH_PENDING')
            require(not self.health_handles and not self.p.handles and not self.p.patches and self.p.lease is None,
                    'KV_GROUP_OWNED_HOOKS_OR_LEASES_REMAIN')
            return WorkerHealthProof(worker,owner,generation,digest(self.health_artifact),self.baseline_hash,
                self.health_artifact['model_output_sha256'],0,0,0,0,owned_hooks=0)
        return GroupRecoveryHooks(stop,drain,recover,health)

    def begin_recovery(self):
        require(self.stage=='off_complete','KV_GROUP_ACTUAL_OFF_REQUIRED_BEFORE_DRAIN')
        self.observer.seal(artifact_sha256=digest(self.completed_requests),complete=not self.failures)
        result=self.session.disable(timeout_seconds=30)
        require(not self.p.installed and not self.p.handles and not self.p.patches and self.p.lease is None,
                'KV_GROUP_UNINSTALL_NOT_CONFIRMED')
        self.stage='health_pending'
        return result

    def begin_health(self,request_id,prompt):
        require(self.stage=='health_pending' and not self.health_handles and not self.p.installed,
                'KV_GROUP_HEALTH_OBSERVER_GATE')
        require(self.p._identity()==self.p.model_identity,'KV_GROUP_HEALTH_MODEL_BINDING_CHANGED')
        require(list(prompt)==self.prefixes[0]['prompt'],'KV_GROUP_HEALTH_PROMPT_CHANGED')
        expected={request_id:dict(logical_id=0,prompt=list(prompt))}
        self.health_artifact=None
        def pre(module,args,kwargs):
            require(threading.get_ident()==self.p.thread,'KV_GROUP_HEALTH_THREAD_MISMATCH')
            if self.p.runner.with_prefill:return
            from vllm.forward_context import get_forward_context
            from .batch_context import snapshot_first_decode_rows
            actual=self.p.binding.bind(module,args,kwargs)
            require(self.health_current is None and self.health_artifact is None,'KV_GROUP_HEALTH_FORWARD_COUNT')
            self.health_current=snapshot_first_decode_rows(self.p.runner,self.p.model,actual,get_forward_context(),
                                                           expected,expected_max_seqs=1)
            require(self.health_current['prefixes']==self.prefixes,'KV_GROUP_HEALTH_LOGICAL_INPUT_CHANGED')
        def post(module,args,kwargs,output):
            if self.health_current is None:return
            try:
                require(output is not None,'KV_GROUP_HEALTH_MODEL_DID_NOT_RETURN')
                self.health_artifact=dict(request=self.health_current,model_output_sha256=_output_hash(_cpu(output)),
                                         actual_model_object_id=id(module),process_start=self.p.process_start)
            finally:self.health_current=None
        self.health_handles.append(self.p.model.register_forward_pre_hook(pre,with_kwargs=True))
        self.health_handles.append(self.p.model.register_forward_hook(post,with_kwargs=True,always_call=True))
        return dict(actual_request_id=request_id,read_only=True)

    def finish_health(self):
        require(self.health_current is None,'KV_GROUP_HEALTH_FORWARD_ACTIVE')
        self.close_health_observer()
        require(self.health_artifact is not None and self.health_artifact['model_output_sha256']==self.baseline_hash,
                'KV_GROUP_SAME_WORKER_HEALTH_OUTPUT_MISMATCH')
        recovery=self.session.disable(timeout_seconds=30)
        report=self.report()
        require(recovery['complete'],'KV_GROUP_RECOVERY_INCOMPLETE')
        if self.trigger=='once' and report['contract_satisfied']:self.c2_live_proof=self.session.c2_proof()
        self.stage='finished'
        return self.report()

    def close_health_observer(self):
        require(self.health_current is None,'KV_GROUP_HEALTH_FORWARD_ACTIVE')
        for handle in reversed(self.health_handles):handle.remove()
        self.health_handles.clear()

    def abort_request(self,error):
        require(self.p.current is None,'KV_GROUP_ACTUAL_REQUEST_STILL_ACTIVE')
        self.p.recover()
        self.failures.append(dict(stage=self.stage,error=str(error),pending=self.p.pending_run))
        self.p.expected=self.p.phase=self.p.pending_run=self.p.collector=None
        self.stage='failed'

    def report(self):
        result=self.session.report() if self.session is not None else {}
        result.update(stage=self.stage,baseline_control=getattr(self,'baseline_control',False),
            independent_discovery=[dict(inventory=item['inventory'],byte_evidence=list(item['byte_evidence'].values()))
                                   for item in self.discovery],
            actual_request_count=self.fault_requests,request_budget=self.requests,executions=self.executions,
            per_target_lossless=list(self.lossless.values()),completed_requests=self.completed_requests,failures=self.failures,
            all_targets_controlled_once=(len(self.executions)==len(self.identities)
                and len({row['target_id'] for row in self.executions})==len(self.identities)),
            health=self.health_artifact,live_native_leases=int(self.p.lease is not None),
            owned_hooks=len(self.p.handles)+len(self.health_handles),owned_api_patches=len(self.p.patches),
            public_qualification_granted=False)
        return result
