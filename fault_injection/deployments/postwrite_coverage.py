"""Bounded common CoverageSession integration for actual postwrite projections.

C2 retains one Native lease through one complete forward. All other actual
projection calls remain in the independent denominator and are transparently
reported by the common concurrency gate. A new fixed-input forward reaches the
next target. No layer selector, fabricated executor, or hidden NaN exclusion.
"""
from dataclasses import asdict
import hashlib
import inspect
from pathlib import Path
import time
import uuid
import os

import torch

from fault_injection.coverage import (BoundaryTargetIdentity, Candidate, CoverageCapability, Eligibility,
    EligibilityResult, ObservedCall, OperatorApiBoundary, Outcome, Phase, QualifiedTargetSet,
    RecoveryReport, TargetLease, TargetLosslessProof)
from fault_injection.coverage_session import (CoverageScope, CoverageSession, GroupRecoveryHooks,
    IndependentCallObserver, SessionBudget, WorkerHealthProof, target_id)
from fault_injection.runtime_discovery import RuntimeCandidateInventory
from .postwrite import PostwriteRefused, _hash_inputs, _equal
from .postwrite_evidence import digest, process_start, tensor_digest, output_digest


class _CpuEvidence:
    transport_kind = 'cpu_test'


def _cpu_digest(value):
    return output_digest(_CpuEvidence(), value)


def _geometry(row):
    actual = row.output_geometry or {}
    return {key: actual.get(key) for key in ('shape', 'stride', 'storage_offset', 'dtype', 'device', 'storage_bytes', 'logical_bytes')}


class ProjectionCoverage:
    def __init__(self, provider, scope, *, max_seconds=1200):
        if (provider.mechanism != '39.output_proxy' or not isinstance(scope, CoverageScope)
                or scope.mechanism != provider.mechanism):
            raise PostwriteRefused('POSTWRITE_COVERAGE_SCOPE_MISMATCH')
        if not 0 < max_seconds <= 1800:raise PostwriteRefused('POSTWRITE_SESSION_TIME_BOUND_REQUIRED')
        history = provider.query()['history']
        faults = [row for row in history if row['phase'] == 'fault' and row['new_fault_hits'] == 1
                  and row['recovery_confirmed'] and not row['exact_baseline_output']]
        if (not faults or not history or history[-1]['phase'] != 'off'
                or not history[-1]['exact_baseline_output'] or provider.query()['leases']):
            raise PostwriteRefused('POSTWRITE_SAME_PROCESS_C1_FAULT_EFFECT_AND_OFF_REQUIRED')
        self.provider, self.scope, self.max_seconds = provider, scope, max_seconds
        self.owner, self.generation = str(provider.owner), 0
        self.process_start = process_start()
        self.model_id = str(id(provider.model))
        self.baseline_hash = _cpu_digest(provider._baseline_output)
        self.source_hash = hashlib.sha256(Path(inspect.getsourcefile(type(self))).read_bytes()).hexdigest()
        self.inventory_source_hash = hashlib.sha256((Path(__file__).parent/'postwrite_inventory.py').read_bytes()).hexdigest()
        self.discovery_reports = [observer.inventory.report() for observer in provider._observations]
        if len(self.discovery_reports) != 2:raise PostwriteRefused('POSTWRITE_TWO_INDEPENDENT_HEALTHY_INVENTORIES_REQUIRED')
        first, second = provider._observations
        self.identities = {row.key:self._identity(row) for row in first.inventory.calls}
        self.eligible = {row.key for row in first.inventory.calls if row.candidate and row.candidate.consumers}
        self.signatures = {row.key:_geometry(row) for row in first.inventory.calls}
        self.call_order = tuple(row.key for row in first.inventory.calls)
        self.module_universe = {id(module) for module in provider.model.modules()}
        self.discovery_sha = digest(self.discovery_reports)
        self.discovery_rows = []
        for forward, observation in enumerate((first, second)):
            records = []
            seen = set()
            for row in observation.inventory.calls:
                status = Eligibility.SUPPORTED if row.candidate and row.candidate.consumers else Eligibility.UNSUPPORTED
                reason = row.rejection or 'NO_SELECTED_ELEMENT_VALUE_CONSUMER'
                records.append((Candidate('projection:' + digest(row.key), EligibilityResult(status,
                    () if status == Eligibility.SUPPORTED else (reason,)), self._identity(row)), row.module_object_identity))
                seen.add(row.module_object_identity)
            for (_, module), count in observation.inventory.module_calls.items():
                if module not in seen:
                    records.append((Candidate('module:' + str(module), EligibilityResult(Eligibility.UNSUPPORTED,
                        ('COMPLETE_ACTUAL_MODULE_CALL_HAS_NO_ADAPTED_PROJECTION',))), module))
            self.discovery_rows.append(records)
        if (first.inventory.operator_counts != second.inventory.operator_counts
                or first.inventory.module_calls != second.inventory.module_calls
                or self.call_order != tuple(row.key for row in second.inventory.calls)
                or any(self.signatures[row.key] != _geometry(row) for row in second.inventory.calls)):
            raise PostwriteRefused('POSTWRITE_COMPLETE_HEALTHY_INVENTORY_DRIFT')
        if not self.eligible:raise PostwriteRefused('POSTWRITE_EMPTY_ELIGIBLE_SET')
        self.max_forwards = len(self.eligible)
        self.session = self.observer = self.group = None
        self.mode, self.active, self.pending = None, None, None
        self.stopped = False
        self.forward_serial = 0
        self.lossless, self.lossless_outputs, self.off_outputs = {}, {}, {}
        self.execution_artifacts, self.forward_artifacts = [], []
        self.last_output = self.last_observation = None
        self.off_hash = None
        self.c2 = self.c2_live_proof = None
        # Metadata survives; qualification activations no longer reserve device storage.
        provider._observations.clear()

    def _identity(self, row):
        meta = row.output_geometry or {}
        device = meta.get('device', str(next(self.provider.model.parameters()).device))
        kind, _, index = device.partition(':')
        return BoundaryTargetIdentity(os.getpid(), self.process_start, 'model-local:' + self.model_id,
            'MODEL_LOCAL', 0, kind, None if kind == 'cpu' else int(index or 0), self.model_id,
            id(self.provider.model), row.module_object_identity, (), row.operator, digest(_geometry(row)),
            OperatorApiBoundary(hashlib.sha256(row.schema.encode()).hexdigest(), self.scope.implementation_sha256,
                'stable-projection-site:' + digest(row.key), 'torch_dispatch'), '39.output_proxy:bounded-coverage-v1',
            (('native_library_sha256', getattr(self.provider.transport, 'sha256', self.scope.implementation_sha256)),))

    def _new_observer(self):
        observer = IndependentCallObserver(source='independent_workload_observer',
            observer_id='projection-observer:' + uuid.uuid4().hex,
            implementation_sha256=self.inventory_source_hash, workload_sha256=self.scope.workload_sha256,
            max_calls=(self.max_forwards + 5) * max(1, len(self.call_order)))
        workers = tuple(dict.fromkeys(target.worker for target in self.identities.values()))
        inventory = RuntimeCandidateInventory(expected_modules=self.module_universe, expected_workers=workers)
        for index, rows in enumerate(self.discovery_rows):
            forward = 'healthy:' + str(index)
            for candidate, module in rows:inventory.observe(candidate, forward_id=forward, module_object_identity=module)
            inventory.finish_forward(forward, healthy=True)
        candidates = inventory.seal(observer, artifact_sha256=self.discovery_sha)
        if not candidates.discovery_complete:raise PostwriteRefused('POSTWRITE_INDEPENDENT_DISCOVERY_INCOMPLETE')
        return observer, candidates

    def _check_request(self, args, kwargs):
        p = self.provider
        if p._identity() != p._model_identity or _hash_inputs(args, kwargs) != p._input_hash:
            raise PostwriteRefused('POSTWRITE_FIXED_WORKLOAD_OR_MODEL_IDENTITY_CHANGED')

    def _run_forward(self, mode, args, kwargs, model_observation=None):
        self._check_request(args, kwargs)
        if self.provider._coverage is not None or self.provider._lease is not None:
            raise PostwriteRefused('POSTWRITE_GROUP_OWNER_OR_RECOVERY_ACTIVE')
        self.mode, self.pending = mode, None
        self.forward_serial += 1
        # A genuine invocation owned by this MODEL_LOCAL driver, never a vLLM scheduler ID.
        self.request_id = 'model-local-invocation:' + str(self.forward_serial)
        self.forward_id = 'forward:' + str(self.forward_serial)
        self.provider.generation += 1
        self.provider._coverage = self
        try:
            output, _, _ = self.provider._replay('group_' + mode, args, kwargs, model_observation=model_observation)
            if self.active is not None:
                decision, artifact, candidate = self.active
                if not candidate.consumers:raise PostwriteRefused('POSTWRITE_GROUP_VALUE_CONSUMER_NOT_OBSERVED')
                artifact.update(consumers=list(candidate.consumers), consumer_observations=candidate.consumer_observations,
                    original_calls=1, native_recovered=self.provider._lease is None,
                    full_model_output_sha256=_cpu_digest(output))
                self.session.finish(decision, Outcome.PASSTHROUGH if self.baseline_control else Outcome.HIT,
                    execution_sha256=digest(artifact), effect_sha256=artifact['mutated_output_sha256'],
                    native_applied=not self.baseline_control, original_calls=1)
                self.execution_artifacts.append(artifact)
                self.active = None
            if mode == 'lossless':
                if not _equal(output, self.provider._baseline_output):
                    raise PostwriteRefused('POSTWRITE_ALL_TARGET_LOSSLESS_ROOT_OUTPUT_MISMATCH')
                if set(self.lossless) != self.eligible:
                    raise PostwriteRefused('POSTWRITE_ALL_TARGET_LOSSLESS_INCOMPLETE')
            if mode == 'off':
                if not _equal(output, self.provider._baseline_output):
                    raise PostwriteRefused('POSTWRITE_GROUP_OFF_ROOT_OUTPUT_MISMATCH')
                if self.off_outputs != self.lossless_outputs:
                    raise PostwriteRefused('POSTWRITE_PER_TARGET_OFF_BYTES_MISMATCH')
                self.off_hash = _cpu_digest(output)
            return output
        except BaseException as error:
            observed = self.provider._last_observer
            self.forward_artifacts.append(dict(forward_id=self.forward_id, request_id=self.request_id,
                phase=mode, complete=False, error=repr(error),
                inventory=observed.inventory.report() if observed is not None else None))
            if self.active is not None:
                decision, artifact, _ = self.active
                artifact['failure'] = repr(error)
                self.session.finish(decision, Outcome.FAILED, execution_sha256=digest(artifact),
                    native_applied=artifact.get('native_applied', False), original_calls=1)
                self.execution_artifacts.append(artifact)
                self.active = None
            raise
        finally:
            self.provider._coverage = None
            self.mode = None

    def projection(self, observation, row):
        p = self.provider
        known = row.key in self.identities
        target = self.identities[row.key] if known else self._identity(row)
        eligible = row.key in self.eligible
        candidate = row.candidate
        if self.mode == 'lossless':
            if not eligible:return
            if candidate is None or self.signatures[row.key] != _geometry(row):
                raise PostwriteRefused('POSTWRITE_LOSSLESS_TARGET_GEOMETRY_CHANGED')
            before = tensor_digest(p.transport, candidate.tensor)
            p._lease = p.transport.open(candidate.tensor, p.owner, p.generation)
            changed = p._lease.apply(0, candidate.indices)
            after = tensor_digest(p.transport, candidate.tensor)
            p.recover()
            if changed != 0 or before != after:raise PostwriteRefused('POSTWRITE_LOSSLESS_RETURN_BYTES_CHANGED')
            self.lossless[row.key] = dict(target_id=target_id(target), before=before, after=after,
                original_calls=1, native_generation=p.generation, temporary_readback_verified=True, released=True)
            self.lossless_outputs[row.key] = before
            return
        phase = Phase.OFF_RECOVERY if self.mode == 'off' else (
            Phase.POST_FIRST_ANOMALY if self.active is not None else Phase.PRE_FAULT_PROPAGATION)
        ticket = self.observer.observe(ObservedCall(target, self.forward_id + ':' + str(len(observation.inventory.calls)),
            self.forward_id, (self.request_id,), phase), eligible=eligible,
            propagation_added=not known and phase == Phase.POST_FIRST_ANOMALY)
        decision = self.session.begin(ticket)
        if not decision.admitted:
            self.session.finish(decision, Outcome.RESOURCE_LIMITED if decision.reason == 'CONCURRENCY_BUDGET_EXHAUSTED'
                                else Outcome.PASSTHROUGH, original_calls=1)
            if self.mode == 'off' and eligible and candidate is not None:
                self.off_outputs[row.key] = tensor_digest(p.transport, candidate.tensor)
            if eligible and self.mode != 'off' and decision.reason in ('SESSION_BUDGET_EXHAUSTED', 'SESSION_STOPPED'):
                raise PostwriteRefused('POSTWRITE_BOUNDED_SESSION_STOP:' + decision.reason)
            if self.group.trigger == 'every_match' and eligible and self.mode != 'off':
                raise PostwriteRefused('POSTWRITE_EVERY_MATCH_REFUSED:' + decision.reason)
            return
        artifact = dict(target_id=target_id(target), call_id=ticket.call.call_id,
            forward_id=self.forward_id, request_id=self.request_id, native_generation=p.generation,
            native_lifetime='one actual complete forward', original_calls=1, native_applied=False)
        self.active = decision, artifact, candidate
        if candidate is None or self.signatures[row.key] != _geometry(row):
            raise PostwriteRefused('POSTWRITE_DISCOVERED_TARGET_GEOMETRY_CHANGED')
        artifact['original_output_sha256'] = tensor_digest(p.transport, candidate.tensor)
        p._lease = p.transport.open(candidate.tensor, p.owner, p.generation)
        changed = p._lease.apply(0 if self.baseline_control else p.mutation_mode, candidate.indices)
        artifact.update(native_applied=not self.baseline_control, baseline_control=self.baseline_control,
                        changed_elements=changed, selected_elements=len(candidate.indices),
                        mutated_output_sha256=tensor_digest(p.transport, candidate.tensor), readback_exact=True)
        if changed != (0 if self.baseline_control else len(candidate.indices)):
            raise PostwriteRefused('POSTWRITE_GROUP_EFFECT_COUNT_MISMATCH')

    def forward_returned(self, observation, output):
        order = tuple(row.key for row in observation.inventory.calls)
        if order != self.call_order:
            raise PostwriteRefused('POSTWRITE_ACTUAL_PROJECTION_DENOMINATOR_CHANGED')
        if {module for (_, module) in observation.inventory.module_calls} != self.module_universe:
            raise PostwriteRefused('POSTWRITE_ACTUAL_MODULE_DENOMINATOR_CHANGED')
        self.last_output, self.last_observation = output, observation.inventory.report()
        self.forward_artifacts.append(dict(forward_id=self.forward_id, request_id=self.request_id,
            phase=self.mode, inventory=observation.inventory.report(), output_sha256=_cpu_digest(output)))

    def _hooks(self):
        def stop(lease):self.stopped = True
        def drain(lease, remaining):
            started = time.monotonic()
            if self.provider.transport.transport_kind != 'cpu_test':torch.npu.synchronize()
            return self.provider._lease is None and time.monotonic() - started <= remaining
        def recover(lease):
            if self.provider._lease is not None:raise PostwriteRefused('POSTWRITE_NATIVE_RECOVERY_PENDING')
            if self.off_hash != self.baseline_hash:raise PostwriteRefused('POSTWRITE_GROUP_OFF_REQUIRED')
            proof = dict(target_id=target_id(lease.target), off=self.off_hash, native_live_leases=0,
                         off_targets={target_id(self.identities[key]):value for key,value in self.off_outputs.items()})
            return RecoveryReport(lease, True, True, True, proof_sha256=digest(proof), off_output_sha256=self.off_hash)
        def health(worker, owner, generation):
            zero = self.provider._lease is None and not self.provider._stack and not self.provider._lock.locked()
            proof = dict(worker=worker, owner=owner, generation=generation, healthy_before=self.baseline_hash,
                         healthy_after=self.off_hash, no_owned_hooks=zero)
            return WorkerHealthProof(worker, owner, generation, digest(proof), self.baseline_hash,
                                     self.off_hash, 0 if zero else 1, 0, 0, 0, owned_hooks=0 if zero else 1)
        return GroupRecoveryHooks(stop, drain, recover, health)

    def run(self, trigger, *args, fault_repeats=2, model_observation=None, baseline_control=False, **kwargs):
        if trigger not in ('once', 'every_match'):raise ValueError('once or every_match required')
        if type(fault_repeats) is not int or not 1 <= fault_repeats <= 3:
            raise PostwriteRefused('POSTWRITE_BOUNDED_C3_REPEATS_REQUIRED')
        if type(baseline_control) is not bool or (baseline_control and trigger != 'once'):
            raise PostwriteRefused('POSTWRITE_BASELINE_CONTROL_REQUIRES_ONCE')
        self.baseline_control = baseline_control
        if trigger == 'every_match' and self.c2_live_proof is None:
            raise PostwriteRefused('POSTWRITE_LIVE_C2_PROOF_REQUIRED')
        self._check_request(args, kwargs)
        self.generation += 1
        self.lossless, self.lossless_outputs, self.off_outputs = {}, {}, {}
        self.off_hash, self.stopped = None, False
        self.execution_artifacts, self.forward_artifacts = [], []
        self.observer, candidates = self._new_observer()
        self._run_forward('lossless', args, kwargs)
        leases = tuple(TargetLease(target, self.owner, self.generation, 'projection:' + target_id(target), 0,
            2 * self.signatures[key]['logical_bytes']) for key, target in self.identities.items() if key in self.eligible)
        by_target = {target:key for key,target in self.identities.items()}
        proofs = tuple(TargetLosslessProof(lease, 'post_operator_return', digest(self.lossless[by_target[lease.target]]),
            (self.lossless[by_target[lease.target]]['before'],), (self.lossless[by_target[lease.target]]['after'],),
            self.baseline_hash, self.baseline_hash, 1, True, True) for lease in leases)
        capability = CoverageCapability(single_target=True, multi_target=True, every_match=True,
            max_concurrency=1, max_resources=0, max_host_bytes=sum(lease.host_bytes for lease in leases),
            group_recovery=True, recovery_scope='target_set', stop_policy='bounded_session',
            lossless_modes=('post_operator_return',))
        self.group = QualifiedTargetSet(candidates, leases, proofs, capability, selection='all_eligible', trigger=trigger)
        self.session = CoverageSession(group=self.group, scope=self.scope, observer=self.observer,
            budget=SessionBudget((self.max_forwards + 1) * len(self.call_order), self.max_seconds, 1),
            recovery=self._hooks(), c2_proof=self.c2_live_proof if trigger == 'every_match' else None)
        failure = None
        completed = 0
        try:
            for index in range(self.max_forwards if trigger == 'once' else fault_repeats):
                self._run_forward('fault', args, kwargs, model_observation=model_observation)
                completed += 1
        except BaseException as error:
            failure = repr(error)
        try:
            self._run_forward('off', args, kwargs)
        except BaseException as error:
            failure = failure or repr(error)
        independent = dict(discovery=self.discovery_reports, forwards=self.forward_artifacts,
                           tickets=[asdict(ticket) for ticket in self.observer.tickets])
        self.observer.seal(artifact_sha256=digest(independent), complete=failure is None)
        recovery = self.session.disable(timeout_seconds=30)
        report = self.session.report()
        report.update(provider_failure=failure, completed_fault_forwards=completed,
                      max_fault_forwards=self.max_forwards if trigger == 'once' else fault_repeats,
                      per_target_lossless=[dict(key=key, **value) for key,value in self.lossless.items()],
                      execution_artifacts=self.execution_artifacts, forward_artifacts=self.forward_artifacts,
                      independent_discovery=self.discovery_reports, recovery=recovery,
                      isolated_target_replays=trigger == 'once', baseline_control=baseline_control,
                      all_targets_controlled_once=(len(self.execution_artifacts) == len(self.eligible)
                        and len({row['target_id'] for row in self.execution_artifacts}) == len(self.eligible)),
                      baseline_controls_grant_qualification=False)
        if trigger == 'once':
            self.c2 = report
            if report['contract_satisfied']:self.c2_live_proof = self.session.c2_proof()
        return report
