"""Conditional complete-set Native26 coverage; unresolved census never seals."""
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import time

from ..coverage import (BoundaryTargetIdentity, Candidate, CandidateSet, CoverageCapability, Eligibility,
    EligibilityResult, ObservedCall, OperatorApiBoundary, Outcome, Phase, QualifiedTargetSet, RecoveryReport,
    TargetIdentity, TargetLease, TargetLosslessProof)
from ..coverage_session import (CoverageScope, CoverageSession, GroupRecoveryHooks, IndependentCallObserver,
                                SessionBudget, WorkerHealthProof, target_id)
from .writer26_experiment import Writer26Experiment
from .writer26_inventory import digest, require


class Writer26Coverage(Writer26Experiment):
    def __init__(self, *args, trigger='once', c2_proof=None, evidence_kind='natural_worker', **kwargs):
        require(trigger in ('once', 'every_match') and evidence_kind in ('natural_worker', 'cpu_contract'),
                'EXPLICIT_WRITER_COVERAGE_SCOPE_REQUIRED')
        require(trigger == 'once' or (c2_proof is not None and c2_proof.scope.evidence_kind == evidence_kind),
                'MATCHING_LIVE_C2_BEFORE_WRITER_EVERY_MATCH_REQUIRED')
        super().__init__(*args, stage='lossless', **kwargs)
        self.trigger, self.c2_proof, self.evidence_kind = trigger, c2_proof, evidence_kind
        self.coverage = self.observer = self.group = self.proof = self.candidates = None
        self.identities, self.decisions, self.finished, self.off_outcomes = {}, {}, set(), {}
        self.first_anomaly = False
        self.health_sha = None
        self.root_hooks = 0
        self.actual_environment_sha = None
        self.environment_refusal = None
        self.implementation_sha = digest([hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            hashlib.sha256(Path(__file__).with_name('writer26_experiment.py').read_bytes()).hexdigest(), self.runtime.build_identity,
            hashlib.sha256(Path(__file__).with_name('writer26_public.py').read_bytes()).hexdigest(),
            hashlib.sha256(Path(__file__).with_name('writer26_fia_typed.py').read_bytes()).hexdigest(),
            hashlib.sha256(Path(__file__).parents[1].joinpath('environment.py').read_bytes()).hexdigest(),
            self.metadata.implementation_sha256])

    def artifact(self, name, value):
        import json
        path = self.folder / (name + '.json')
        raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        with path.open('xb') as stream:
            stream.write(raw)
        return hashlib.sha256(raw).hexdigest()

    def _identity(self, callback, native, *, unowned=False):
        worker = self.worker_identity
        required = ('worker_start_identity', 'rank', 'logical_device', 'model_instance_id')
        require(all(key in worker for key in required), 'COMPLETE_REAL_WORKER_LIFETIME_REQUIRED')
        from ..runtime_discovery import _process_start_identity
        module = id(self.model) if unowned else callback['module_object_identity']
        signature = digest({key: native[key] for key in ('dtype', 'format', 'dimensions', 'output_bytes',
                                                        'residual_bytes', 'shape', 'residual_shape')})
        method = 'UNOWNED_NATIVE_BOUNDARY_IN_ACTUAL_ROOT_FORWARD' if unowned else digest(callback['method'])
        proof = tuple(sorted((('native_build', self.runtime.build_identity), ('method_identity', method))))
        operator = 'aclnnFusedInferAttentionScoreV3' if self.runtime.fia else 'aclnnQuantMatmulWeightNz'
        if not self.runtime.fia:
            return TargetIdentity(os.getpid(), worker['worker_start_identity'], worker['rank'], worker['logical_device'],
                worker['model_instance_id'], id(self.model), module, (), operator, 'op_exec_cache_run',
                'opcache:' + str(native['cache']), 'actual_registration_epoch:' + str(native['registration']), signature,
                self.runtime.build_identity, 'native26.group.v2', proof, _process_start_identity())
        boundary = OperatorApiBoundary(self.runtime.build_identity, self.runtime.sha256,
            'root-unowned:' + digest([operator, native['cache'], signature]) if unowned else 'module-writer:' + str(module) + ':' + method,
            'public_aclnn')
        return BoundaryTargetIdentity(os.getpid(), _process_start_identity(), worker['worker_start_identity'],
            'INITIALIZED_WORKER', worker['rank'], 'npu', worker['logical_device'], worker['model_instance_id'],
            id(self.model), module, ('native_unowned_in_root',) if unowned else (), operator, signature, boundary,
            'native26.group.v2', proof)

    def _freeze(self):
        super()._freeze()
        if self.blockers:
            return
        stable_fields = ('cache', 'registration', 'dtype', 'format', 'dimensions', 'residual_dimensions',
                         'output_bytes', 'residual_bytes', 'shape', 'strides', 'residual_shape', 'residual_strides')
        for row in self.candidate_inventory:
            if any(row['first_native'][name] != row['second_native'][name] for name in stable_fields):
                row['status'] = 'UNVERIFIED'
                row['reasons'].append('COMPLETE_HEALTHY_NATIVE_BOUNDARY_IDENTITY_CHANGED')
        unresolved = [row for row in self.candidate_inventory if row['status'] == 'UNVERIFIED']
        if unresolved:
            self.blockers.append('UNRESOLVED_COMPLETE_NATIVE_WRITER_DENOMINATOR')
            self.plan = self.plan[:2] + [dict(phase='health', mode=None, target=None)]
            return
        intervals = []
        for candidate in self.rows[1]['candidates']:
            callback = candidate.get('callback')
            if callback and callback.get('target_key') in self.targets:
                intervals.append((candidate['native']['sequence'], candidate['next_consumer'][0]['sequence']))
        intervals.sort()
        if any(left[1] >= right[0] for left, right in zip(intervals, intervals[1:])):
            self.blockers.append('NATURAL_ELIGIBLE_WRITER_LIFETIMES_OVERLAP')
            self.plan = self.plan[:2] + [dict(phase='health', mode=None, target=None)]
            return
        rows = []
        for index, row in enumerate(self.candidate_inventory):
            callback, native = row['callback'], row['second_native']
            identity = self._identity(callback, native, unowned=callback is None)
            state = Eligibility.SUPPORTED if row['status'] == 'OBSERVED_NOT_QUALIFIED' else Eligibility.UNSUPPORTED
            key = row['target_key'] or 'unsupported-native-' + str(index)
            rows.append(Candidate(key, EligibilityResult(state, tuple(row['reasons'])), identity))
            row['identity'] = asdict(identity)
            self.identities[key] = identity
        self.candidates = CandidateSet(tuple(rows), True)
        self.artifact('complete-candidate-set', self.candidates.report())
        # All target qualifications precede a shared fault request. No selected
        # candidate window is reused as an all-eligible result.
        self.plan = self.plan[:2]
        self.plan.extend(dict(phase='qualification_' + mode + '_all', mode=mode, target='*')
                         for mode in ('reference', 'lossless', 'off'))
        self.plan.append(dict(phase='group_fault', mode='fault', target='*'))
        if self.trigger == 'every_match':
            self.plan.append(dict(phase='group_fault_repeat', mode='fault', target='*'))
        self.plan.extend((dict(phase='group_off', mode='off', target='*'), dict(phase='health', mode=None, target=None)))
        events = self.rows[1]['native_after']['total_events'] - self.rows[1]['native_before']['total_events']
        if events * (len(self.plan) - 2) > self.rows[1]['native_after']['capacity'] - self.rows[1]['native_after']['stored_events']:
            self.blockers.append('COMPLETE_GROUP_PLAN_EXCEEDS_NATIVE_EVENT_CAPACITY')
            self.plan = self.plan[:2] + [dict(phase='health', mode=None, target=None)]

    def _planned_qualification_requests(self):
        return 7 if self.trigger == 'every_match' else 6

    def _environment(self):
        if self.evidence_kind != 'natural_worker':
            self.actual_environment_sha = digest(['EXPLICIT_CPU_MOCK_NOT_NPU', self.worker_identity, self.runtime.build_identity])
            return
        from ..coverage import _is_sha256
        from ..environment import inspect_environment, environment_fingerprint, inspect_bound_environment
        libraries = ('libdcat26_group_v2.so', 'libfault26_readonly_trace_abi0.so', 'libnnopbase.so', 'libascendcl.so')
        if self.runtime.build.get('bound_symbol_profile') == 'demoncat.writer26.cached_addresses.v1':
            environment, value = inspect_bound_environment(self.runtime.environment_bindings(), required_libraries=libraries)
        else:
            # Old source-bound attempts retain the original strict gate. They
            # cannot acquire the new collision profile from serialized evidence.
            environment = inspect_environment(additional_libraries=libraries)
            value = environment_fingerprint(environment, libraries)
        artifact_sha = self.artifact('actual-environment', environment)
        if environment.get('pid') != os.getpid() or not _is_sha256(value):
            names = {row.get('name') for row in environment.get('loaded_libraries', ())}
            self.environment_refusal = dict(actual_environment_sha256=artifact_sha,
                maps_status=environment.get('maps_status'), actual_pid=environment.get('pid'), expected_pid=os.getpid(),
                required_libraries=list(libraries), missing_required_libraries=sorted(set(libraries)-names),
                bound_symbol_refusal=environment.get('bound_symbol_evidence', {}).get('refusal'),
                unverified_libraries=[{key: row.get(key) for key in ('name', 'path', 'status', 'identity_failure')}
                    for row in environment.get('loaded_libraries', ()) if row.get('status') != 'MAPPED_FILE_VERIFIED'],
                scope_constructed=False, qualification_granted=False)
            self.artifact('environment-refusal', self.environment_refusal)
        require(environment.get('pid') == os.getpid() and _is_sha256(value), 'LOADED_WORKER_ENVIRONMENT_UNPROVEN')
        self.actual_environment_sha = value

    def _build_group(self):
        require(self.candidates is not None and not self.blockers and self.coverage is None, 'COMPLETE_FROZEN_GROUP_REQUIRED')
        self._environment()
        by_target = {}
        for row in self.outcomes:
            by_target.setdefault(row['target'], {})[row['mode']] = row
        leases, proofs = [], []
        for key in self.targets:
            require(set(by_target.get(key, {})) == {'reference', 'lossless', 'off'}, 'EVERY_TARGET_REQUIRES_B_L_OFF')
            reference, lossless, off = (by_target[key][mode] for mode in ('reference', 'lossless', 'off'))
            require(reference['writer_output_sha256'] == lossless['writer_output_sha256'] == off['writer_output_sha256']
                and reference['residual_before_sha256'] == lossless['residual_before_sha256'] == off['residual_before_sha256'],
                'FULL_TARGET_REFERENCE_LOSSLESS_OFF_CHANGED')
            call = reference['native_call']
            lease = TargetLease(self.identities[key], self.runtime.owner, self.runtime.config.generation,
                digest([self.runtime.owner, self.runtime.config.generation, key]), call['output_bytes'],
                2 * call['residual_bytes'] + 3 * call['output_bytes'] + 8192)
            leases.append(lease)
            # Watched input is the actual live consumer residual. Native v2
            # separately proves its destination-only writer and full output.
            proofs.append(TargetLosslessProof(lease, 'destination_lossless', self.artifact('lossless-' + key, lossless),
                (lossless['residual_before_sha256'],), (lossless['residual_after_sha256'],),
                reference['writer_output_sha256'], lossless['writer_output_sha256'], 1,
                lossless['consumer_receipt']['lossless_qualified'] == 1, lossless['released_receipt']['released'] == 1))
        capability = CoverageCapability(single_target=True, multi_target=True, every_match=True, max_concurrency=1,
            max_resources=64 * 1024 * 1024, max_host_bytes=512 * 1024 * 1024, group_recovery=True,
            recovery_scope='target_set', stop_policy='bounded_session', lossless_modes=('destination_lossless',))
        self.group = QualifiedTargetSet(self.candidates, tuple(leases), tuple(proofs), capability,
                                        selection='all_eligible', trigger=self.trigger)
        scope = CoverageScope(self.runtime.mechanism, self.baseline_input, self.actual_environment_sha,
                              self.implementation_sha, self.evidence_kind)
        self.observer = IndependentCallObserver(source='independent_native_trace', observer_id=self.runtime.owner + ':writer26',
            implementation_sha256=self.implementation_sha, workload_sha256=self.baseline_input, max_calls=8192)
        observed = tuple(self.identities.values())
        workers = tuple(dict.fromkeys(target.worker for target in self.candidates.eligible))
        require({target.worker for target in observed}.issubset(workers), 'MIXED_BOUNDARY_WORKER_IDENTITY_UNSUPPORTED')
        self.observer.seal_discovery(self.candidates, observed_targets=observed,
            artifact_sha256=self.artifact('full-healthy-discovery', self.rows[:2]), expected_workers=workers,
            observed_workers=workers, complete=True)
        self.coverage = CoverageSession(group=self.group, scope=scope, observer=self.observer,
            budget=SessionBudget(8192, 1200, 1), recovery=GroupRecoveryHooks(self._stop, self._drain, self._recover, self._health),
            c2_proof=self.c2_proof)
        self.runtime.seal(tuple(self.targets))

    def begin(self, request_id, input_sha256):
        if self.phase == 'group_fault':
            self._build_group()
        super().begin(request_id, input_sha256)

    def _ticket(self, key, identity, call_id, *, phase, eligible, added=False):
        from ..coverage_session import target_id
        token = str(call_id)
        require(token not in self.decisions, 'REPEATED_INDEPENDENT_NATIVE_CALL')
        ticket = self.observer.observe(ObservedCall(identity, token, self.current['request_id'],
            (self.current['request_id'],), phase), eligible=eligible, propagation_added=added)
        decision = self.coverage.begin(ticket)
        self.decisions[token] = decision
        return decision

    def _observe_callback(self, row, call):
        if self.coverage is None or not self.phase.startswith('group_'):
            return
        key = row['target_key']
        identity = self._identity(row, row['independent_event'])
        known = key in self.identities and self.identities[key] == identity
        off = self.phase == 'group_off'
        phase = Phase.OFF_RECOVERY if off else Phase.POST_FIRST_ANOMALY if self.first_anomaly else Phase.PRE_FAULT_PROPAGATION
        frozen_eligible = known and key in self.targets
        current_eligible = bool(row.get('current_candidate_geometry_eligible'))
        added = not frozen_eligible and (current_eligible or not known) and self.first_anomaly and not off
        # A failed fresh geometry check cannot erase a previously qualified
        # target's call. It remains an eligible failed opportunity.
        self._ticket(key, identity, call.independent_call, phase=phase,
                     eligible=frozen_eligible or current_eligible or not known, added=added)
        if (not known or current_eligible and not frozen_eligible) and (off or not self.first_anomaly):
            self.failure = 'UNKNOWN_NATIVE_LIFETIME_BEFORE_ANOMALY_OR_DURING_OFF'

    def _requested_mode(self, row):
        if self.failure:
            return None, None
        if self.phase.startswith('qualification_'):
            return (self.entry['mode'], row['target_key']) if row['target_key'] in self.targets else (None, None)
        if not self.phase.startswith('group_'):
            return super()._requested_mode(row)
        key = row['target_key']
        decision = self.decisions[str(row['independent_call'])]
        if key in self.targets and (decision.admitted or self.phase == 'group_off'):
            return self.entry['mode'], key
        return None, None

    def _expected_admissions(self):
        if self.phase.startswith('group_'):
            return 8192
        return len(self.targets) if self.phase.startswith('qualification_') else 1

    def _expected_admissions_for(self, row):
        if row['phase'].startswith('qualification_'):
            return len(self.targets)
        if not row['phase'].startswith('group_'):
            return 1
        if row['phase'] == 'group_off':
            return sum(c.get('callback', {}).get('target_key') in self.targets for c in row['candidates'] if c.get('callback'))
        return sum(decision.admitted for decision in self.decisions.values()
                   if decision.ticket.call.forward_context == row['request_id'])

    def _check_denominator(self, row):
        if not row['phase'].startswith('group_'):
            return super()._check_denominator(row)
        require(all(str(candidate['native']['call']) in self.decisions for candidate in row['candidates']),
                'ALL_NATIVE_CALLS_REQUIRE_INDEPENDENT_OUTCOMES')

    def _outcome_complete(self, outcome, row):
        if self.coverage is None or not self.phase.startswith('group_'):
            return
        call_id = str(outcome['native_call']['independent_call'])
        decision = self.decisions[call_id]
        fault = outcome['mode'] == 'fault'
        evidence = self.artifact('execution-' + call_id, outcome)
        self.coverage.finish(decision, Outcome.HIT if fault else Outcome.PASSTHROUGH,
            execution_sha256=evidence, effect_sha256=evidence if fault else None, native_applied=fault, original_calls=1)
        self.finished.add(call_id)
        if fault:
            self.first_anomaly = True
        else:
            self.off_outcomes[outcome['target']] = outcome

    def _request_complete(self, row):
        super()._request_complete(row)
        if self.coverage is None or not row['phase'].startswith('group_'):
            if row['phase'] == 'health':
                self.health_sha = row['output_sha256']
            return
        # Native entry records precede all admission, even when the callback is
        # absent. Join every such record now; absence cannot shrink the scope.
        previous_current = self.current
        self.current = row
        try:
            for index, candidate in enumerate(row['candidates']):
                native = candidate['native']
                call_id = str(native['call'])
                if call_id not in self.decisions:
                    frozen = self.candidate_inventory[index] if index < len(self.candidate_inventory) else None
                    require(frozen is not None and frozen['status'] == 'UNSUPPORTED',
                            'ELIGIBLE_NATIVE_WRITER_WITHOUT_PUBLIC_CALLBACK_REMAINS_INCOMPLETE')
                    same = all(native[name] == frozen['second_native'][name] for name in
                        ('cache', 'registration', 'dtype', 'format', 'dimensions', 'output_bytes', 'residual_bytes', 'shape', 'residual_shape'))
                    require(same, 'UNOWNED_NATIVE_UNSUPPORTED_IDENTITY_CHANGED')
                    key = frozen['target_key'] or 'unsupported-native-' + str(index)
                    phase = Phase.OFF_RECOVERY if row['phase'] == 'group_off' else (
                        Phase.POST_FIRST_ANOMALY if self.first_anomaly else Phase.PRE_FAULT_PROPAGATION)
                    self._ticket(key, self.identities[key], native['call'], phase=phase, eligible=False)
                if call_id in self.finished:
                    continue
                execution = candidate['execution']
                valid = len(execution) == 1 and execution[0]['status'] == 0
                self.coverage.finish(self.decisions[call_id], Outcome.PASSTHROUGH if valid else Outcome.FAILED,
                    execution_sha256=self.artifact('passthrough-' + call_id, candidate), native_applied=False,
                    original_calls=len(execution))
                self.finished.add(call_id)
        finally:
            self.current = previous_current

    def _stop(self, lease):
        if not self.native_removed:
            self.runtime.stop()

    def _drain(self, lease, timeout_seconds):
        if not self.native_removed:
            self.runtime.drain()
            if self.evidence_kind == 'natural_worker':
                import torch_npu
                torch_npu.npu.synchronize()
        self.metadata.read()
        return not self.runtime.state()['core']['pending'] and not self.runtime.state()['core']['executing']

    def _recover(self, lease):
        key = next(key for key, identity in self.identities.items() if identity == lease.target)
        off = self.off_outcomes.get(key)
        reference = self.target_refs[key]
        require(off is not None and off['writer_output_sha256'] == reference['output_sha256']
                and off['residual_after_sha256'] == reference['residual_sha256'], 'ACTUAL_ALL_TARGET_OFF_REQUIRED')
        if not self.native_removed:
            self.runtime.close()
            self.native_removed = True
        state = self.runtime.state()
        require(not any(state['core'][key] for key in ('installed', 'pending', 'executing', 'device_bytes', 'host_bytes')),
                'NATIVE_GROUP_NOT_RELEASED')
        proof = self.artifact('recovery-' + key, dict(lease=asdict(lease), off=off, native_state=state,
            metadata_arrays=self.metadata.read()))
        return RecoveryReport(lease, True, True, True, proof_sha256=proof, off_output_sha256=off['writer_output_sha256'])

    def _health(self, worker, owner, generation):
        require(self.health_sha is not None and self.health_sha == self.baseline, 'SAME_PROCESS_HEALTHY_CONTROL_PENDING')
        hooks = len(self.handles) + self.root_hooks + int(self.installed)
        state = self.runtime.state()
        metadata = self.metadata.read(require_activity=True)
        bridge_host_bytes = self.fia_bridge.host_bytes if self.fia_bridge is not None else 0
        proof = self.artifact('worker-health-' + str(time.monotonic_ns()), dict(native=state, baseline=self.baseline,
            off=self.health_sha, owned_hooks=hooks, residual_refs=len(self.retained), output_refs=len(self.output_refs),
            fia_stock_host_bytes=bridge_host_bytes,
            metadata_arrays=metadata))
        return WorkerHealthProof(worker, owner, generation, proof, self.baseline, self.health_sha,
            state['core']['pending'], state['core']['device_bytes'], state['core']['host_bytes'] + bridge_host_bytes,
            state['core']['executing'], hooks)

    def _before_health(self):
        if self.coverage is None:
            return super()._before_health()
        recovery = self.coverage.disable(timeout_seconds=30)
        require(not recovery['pending_leases'] and not recovery['pending_drains'] and self.native_removed,
                'ALL_STOP_ALL_DRAIN_ALL_TARGET_OFF_RELEASE_FAILED')
        # Health is intentionally still pending; all group reservations remain
        # owned until the actual post-release request and hook removal finish.

    def finalize(self):
        require(self.phase == 'complete' and not self.installed and not self.handles and self.root_hooks == 0,
                'ALL_ACTUAL_REQUESTS_AND_OWNED_HOOKS_MUST_FINISH')
        if self.coverage is None:
            return
        recovery = self.coverage.disable(timeout_seconds=30)
        if self.observer.evidence is None:
            self.observer.seal(artifact_sha256=self.artifact('full-native-public-observation', self.rows), complete=True)
        require(recovery['complete'] and self.coverage.report()['contract_satisfied'], 'WRITER_COVERAGE_OR_GROUP_RECOVERY_INCOMPLETE')
        self.proof = self.coverage.c2_proof() if self.trigger == 'once' else None

    def close(self):
        super().close()
        if self.coverage is not None and self.failure:
            # Stop logical admission too. A failed experiment cannot invent
            # missing all-target OFF/health receipts to release reservations.
            self.coverage.disable(timeout_seconds=0)
            self.save()

    def report(self, *, compact_rpc=False, observation_start=None):
        report = super().report(compact_rpc=compact_rpc, observation_start=observation_start)
        coverage = self.coverage.report() if self.coverage is not None else None
        passed = bool(coverage and coverage['contract_satisfied'] and self.evidence_kind == 'natural_worker')
        report.update(schema='demoncat.writer26.complete_set.v2', stage='coverage', trigger=self.trigger, evidence_kind=self.evidence_kind,
            environment_refusal=self.environment_refusal,
            scope='ALL_ELIGIBLE_NATIVE_WRITERS_IN_FIXED_SINGLE_REQUEST_FRESH_PREFILL_NOT_DECODE',
            coverage=coverage, c2=passed and self.trigger == 'once', c3=passed and self.trigger == 'every_match',
            remaining_gates=[] if passed else ['VALID_NATURAL_COMPLETE_SET_AND_REAL_GROUP_EXECUTION_REQUIRED'])
        if coverage is not None and compact_rpc:
            full_sha = digest(coverage)
            for name in ('calls', 'discovery', 'observation'):
                coverage[name+'_original_sha256'] = digest(coverage.pop(name))
            coverage.update(compact_rpc_projection=True, original_report_sha256=full_sha)
        elif coverage is not None and observation_start is not None:
            request_ids = {row['request_id'] for row in self.rows[observation_start:]}
            coverage['calls'] = [row for row in coverage['calls'] if row['forward_id'] in request_ids]
            coverage['calls_are_complete_request_delta'] = True
        return report
