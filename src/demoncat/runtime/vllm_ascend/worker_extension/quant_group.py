"""Actual all-projection QMM groups over the separate bounded native v3 ABI."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

from .quant_inventory import PublicQuantInventory
from .serving20 import live_tensor
from .serving20_qmm import bind_qmm_arguments
from .serving_context import require

PHASES = ('discovery_1', 'discovery_2', 'reference', 'lossless', 'qualification_off',
          'baseline', 'fault', 'off_observed', 'off')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def inventory_counts(snapshot, operator):
    require(snapshot.get('complete') is True and snapshot.get('overflow_calls') == 0,
            'INDEPENDENT_NATIVE_DENOMINATOR_INCOMPLETE')
    rows = snapshot['targets']
    require(sum(row['observations'] for row in rows) == snapshot['eligible_run_calls'],
            'INDEPENDENT_NATIVE_TOTAL_NOT_RECONCILED')
    return {row['identity_sha256']: row['observations'] for row in rows if row['native']['operator'] == operator}


def inventory_delta(before, after, operator):
    left, right = inventory_counts(before, operator), inventory_counts(after, operator)
    require(all(right.get(key, -1) >= value for key, value in left.items()), 'NATIVE_OBSERVER_COUNTER_RESET')
    return {key: count - left.get(key, 0) for key, count in right.items() if count != left.get(key, 0)}


class QuantGroupBackend(PublicQuantInventory):
    def __init__(self, model, candidates, *, runtime, worker_identity, owner, generation,
                 folder, required_libraries, trigger='once', c2_proof=None, evidence_kind='natural_worker'):
        super().__init__(model, candidates, worker_identity=worker_identity, max_calls=4096)
        require(trigger in ('once', 'every_match') and (trigger == 'once' or c2_proof is not None),
                'LIVE_REAL_C2_REQUIRED_FOR_EVERY_MATCH')
        self.runtime, self.owner, self.generation = runtime, owner, generation
        self.required_libraries = tuple(required_libraries)
        self.folder = Path(folder)
        self.trigger, self.c2_proof, self.evidence_kind = trigger, c2_proof, evidence_kind
        self.group_owner = c2_proof.owner if c2_proof is not None else owner
        self.logical_generation = c2_proof.generation + 1 if c2_proof is not None else 1
        self.phases = list(PHASES)
        if trigger == 'every_match':
            self.phases.insert(self.phases.index('off_observed'), 'fault_repeat')
        self.targets, self.references, self.lossless, self.off = {}, {}, {}, {}
        self.events, self.call_ids = [], set()
        self.phase_index = self.artifact_index = self.owned_hooks = 0
        self.started = self.locked = self.first_anomaly = False
        self.remove_native = self.environment_sha = self.workload_sha = self.baseline_output = self.last_output = None
        self.coverage = self.observer = self.group = self.proof = None
        self.adapter_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.observer_sha = hashlib.sha256(Path(__file__).with_name('quant_inventory.py').read_bytes()).hexdigest()
        from fault_injection.metadata_arrays import MetadataArraySession
        self.metadata = MetadataArraySession(getattr(runtime, 'metadata_arrays', None), runtime.build_profile)

    @property
    def phase(self):
        return self.phases[self.phase_index] if self.phase_index < len(self.phases) else 'complete'

    def artifact(self, label, value):
        self.folder.mkdir(parents=True, exist_ok=True)
        raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        self.artifact_index += 1
        with (self.folder / ('%05d-%s.json' % (self.artifact_index, label))).open('xb') as stream:
            stream.write(raw)
        return hashlib.sha256(raw).hexdigest()

    def environment(self):
        from fault_injection.environment import inspect_environment, environment_fingerprint
        value = inspect_environment(additional_libraries=self.required_libraries)
        identity = environment_fingerprint(value, self.required_libraries)
        require(value.get('pid') == os.getpid() and isinstance(identity, str) and len(identity) == 64,
                'ACTUAL_LOADED_ENVIRONMENT_UNPROVEN')
        require(self.environment_sha in (None, identity), 'LOADED_ENVIRONMENT_CHANGED')
        self.environment_sha = identity
        return value

    def paused(self, registered=True):
        self.metadata.read()
        states = self.runtime.snapshot()
        for operator, row in states.items():
            owned = registered and operator == self.runtime.operator
            require(row['runtime_matched'] and not row['enabled'] and row['inflight'] == row['leases'] == 0
                    and not row['recovery_required'] and row['native_registered'] is owned,
                    'NATIVE_GROUP_NOT_IDLE_OR_OTHER_OWNER_PRESENT')
            if owned:
                require(row['native_owned'] and row['paused'] and row['request_controlled'], 'OWNED_PAUSED_GROUP_REQUIRED')
        return states

    def start(self):
        from fault_injection.native import _LEASE
        require(not self.started and not self.remove_native, 'ONE_GROUP_ATTEMPT_REQUIRED')
        self.started = True
        self.started_at = time.monotonic()
        self.paused(False)
        self.artifact('metadata-arrays-initial', self.metadata.latest)
        self.artifact('environment', self.environment())
        require(_LEASE.acquire(blocking=False), 'ANOTHER_NATIVE_SESSION_OWNS_PROCESS')
        self.locked = True
        self.runtime.configure_group_lifecycle(owner=self.owner, generation=self.generation,
            max_targets=512, max_faults=1 if self.trigger == 'once' else 2, max_calls=16384)
        self.remove_native = self.runtime.install_group_observer(self._native_call,
            lambda: self.frame['module'] if self.frame and self.frame.get('transaction') else None)
        self.paused()

    def begin(self, forward_id, input_sha256, *, request_ids, row_mapping=None):
        import torch
        require(self.started and time.monotonic() - self.started_at < 1200, 'BOUNDED_GROUP_TIME_REQUIRED')
        require(not torch.is_grad_enabled() and self.model.training is False, 'ACTUAL_INFERENCE_GROUP_REQUIRED')
        self.paused()
        self.environment()
        if self.workload_sha is None:
            self.workload_sha = input_sha256
        require(input_sha256 == self.workload_sha, 'ACTUAL_FIXED_WORKLOAD_CHANGED')
        super().begin(forward_id, input_sha256, request_ids=request_ids, row_mapping=row_mapping)
        self.current.update(native_before=self.runtime.independent_inventory(), native_events=[], ordinals=Counter())
        if self.phase in ('fault', 'fault_repeat'):
            if self.coverage is None:
                self._build_group()
            self.fault_started = True
        if self.phase != 'off':
            self.runtime.resume()

    def _qmm_lineage(self, frame, row, args, kwargs):
        super()._qmm_lineage(frame, row, args, kwargs)
        actual = bind_qmm_arguments(args, kwargs)
        values = dict(x1=actual['x1'], x2=actual['x2'], x1_scale=actual['pertoken_scale'], x2_scale=actual['scale'])
        if actual.get('bias') is not None:
            values['bias'] = actual['bias']
        ordinal = self.current['ordinals'][id(frame['module'])]
        self.current['ordinals'][id(frame['module'])] += 1
        frame['site'] = (id(frame['module']), ordinal)
        frame['transaction'] = dict(formals={name: live_tensor(value) for name, value in values.items()},
            output_dtype=str(actual['output_dtype']), quantization_output_storage_verified=True,
            native_x1_equals_float_input_storage=False, float_input=live_tensor(frame['input']))
        row['site'] = list(frame['site'])

    def _api(self, name, original, args, kwargs):
        before = self.runtime.independent_inventory()
        index = len(self.public_calls)
        result = super()._api(name, original, args, kwargs)
        after = self.runtime.independent_inventory()
        if len(self.public_calls) > index:
            self.public_calls[index]['native_delta'] = inventory_delta(before, after, self.runtime.operator)
            self.public_calls[index]['independent_native_after_sha256'] = digest(after)
        return result

    def _identity(self, site, frame, facts):
        from fault_injection.coverage import TargetIdentity
        from fault_injection.runtime_discovery import _process_start_identity
        return TargetIdentity(os.getpid(), self.worker_identity['worker_start_identity'],
            self.worker_identity['rank'], self.worker_identity['logical_device'], self.worker_identity['model_instance_id'],
            id(self.model), site[0], (), self.runtime.operator, 'op_exec_cache_run', facts['cache_identity'],
            facts['cache_lifecycle'], digest([facts['layout'], facts['native_signature'], site]),
            self.runtime.build_profile, 'fault20.target_group.v3',
            (('public_method_sha256', frame['row']['method_source_sha256']), ('public_site', digest(site))),
            _process_start_identity())

    def _native_call(self, call, original, packet):
        from fault_injection.coverage import ObservedCall, Outcome, Phase
        from fault_injection.coverage_session import target_id
        from fault_injection.native import _read_proofs
        from fault_injection.runtime_discovery import native_facts
        from fault_injection.qualification import _validate_receipt
        from fault_injection.serving_qualification import ProjectionQualifiedNativeSession
        require(self.current is not None and call.call_id not in self.call_ids, 'ACTUAL_UNREUSED_NATIVE_CALL_REQUIRED')
        self.call_ids.add(call.call_id)
        independent = self.runtime.independent_inventory()
        stable = {key: value for key, value in packet.items() if key != 'current_typed_call'}
        observed = [row for row in independent['targets'] if row['native'] == stable]
        require(len(observed) == 1, 'ACTUAL_NATIVE_CALL_ABSENT_FROM_INDEPENDENT_OBSERVER')
        event = dict(call_id=call.call_id, native_identity=observed[0]['identity_sha256'],
            independent_native_sha256=digest(independent), original_calls=0, phase=self.phase,
            public_lineage_verified=False, owner_module_object_id=id(call.owner) if call.owner else None)
        self.events.append(event)
        self.current['native_events'].append(event)
        decision = None
        def once():
            require(event['original_calls'] == 0, 'NATIVE_ORIGINAL_ALREADY_EXECUTED')
            event['original_calls'] = 1
            return original()
        try:
            frame = self.frame
            if call.owner is None or not frame or frame.get('transaction') is None or frame['row']['errors']:
                event['reason'] = 'NATIVE_ELIGIBLE_CALL_HAS_NO_COMPLETE_PUBLIC_LINEAGE'
                require(once() == 0, 'ORIGINAL_NATIVE_EXECUTION_FAILED')
                return
            require(call.owner is frame['module'], 'ACTUAL_NATIVE_PUBLIC_MODULE_CHANGED')
            proofs, details = _read_proofs(call, '20.buffer_substitute')
            facts = native_facts(details['runtime_call'], details.keys(), '20.buffer_substitute')
            typed = packet.get('current_typed_call', {})
            require(typed.get('source') == 'typed_gws_execute_scope' and typed.get('executor_bound') is True
                    and typed.get('formal_params_current') is True and typed.get('cached_layout_used') is False
                    and type(typed.get('gws_token')) is int and typed['gws_token'] > 0
                    and facts['worker'] == self.runtime.worker and facts['build_profile_sha256'] == self.runtime.build_profile,
                    'CURRENT_TYPED_EXECUTOR_WORKER_BUILD_UNPROVEN')
            event.update(proofs=proofs, facts=facts, current_typed_call=typed, transaction=frame['transaction'])
            ProjectionQualifiedNativeSession._verify_formals(self, typed, facts, frame['transaction'])
            site = frame['site']
            target = self._identity(site, frame, facts)
            key = target_id(target)
            event.update(site=list(site), target=asdict(target), target_key=key, public_lineage_verified=True)
            expected = self.targets.get(site)
            if self.coverage is not None:
                added = expected is None or expected['identity'] != target
                phase = Phase.OFF_RECOVERY if self.phase.startswith('off') else (
                    Phase.POST_FIRST_ANOMALY if self.first_anomaly else Phase.PRE_FAULT_PROPAGATION)
                ticket = self.observer.observe(ObservedCall(target, call.call_id, self.current['forward_id'],
                    tuple(self.current['request_ids']), phase), eligible=not added,
                    propagation_added=bool(added and self.first_anomaly))
                decision = self.coverage.begin(ticket)
            if self.phase.startswith('discovery_') or expected is None:
                require(once() == 0, 'ORIGINAL_NATIVE_EXECUTION_FAILED')
            else:
                require(expected['identity'] == target and expected['native_identity'] == event['native_identity'],
                        'ACTUAL_TARGET_MODULE_CACHE_OR_LIFETIME_CHANGED')
                stage = 'fault' if self.phase in ('fault', 'fault_repeat') else 'lossless' if self.phase == 'lossless' else 'reference'
                if stage == 'fault' and (decision is None or not decision.admitted):
                    require(once() == 0, 'ORIGINAL_NATIVE_EXECUTION_FAILED')
                    stage = 'passthrough'
                else:
                    event['requested_native_stage'] = stage
                    event['native_effect_state'] = 'UNKNOWN_UNTIL_NATIVE_RECEIPT'
                    _, receipt = self.runtime.execute_group(call, once, {'reference': 1, 'lossless': 2, 'fault': 3}[stage], key)
                    _validate_receipt(receipt, stage, call.operator, len(facts['slots']) - 1)
                    event['receipt'] = receipt
                    event['native_effect_state'] = 'FAULT_TRANSFORM_READ_BACK' if stage == 'fault' else 'NO_FAULT_TRANSFORM'
                    if self.phase == 'reference':
                        self.references[site] = receipt
                    elif stage != 'fault':
                        require(receipt['input_sha256_before'] == self.references[site]['input_sha256_before']
                                and receipt['output_sha256'] == self.references[site]['output_sha256'],
                                'LOSSLESS_OR_OFF_NATIVE_BYTES_CHANGED')
                    if stage == 'lossless':
                        self.lossless[site] = receipt
                    if self.phase == 'off_observed':
                        self.off[site] = receipt
                    if stage == 'fault':
                        self.first_anomaly = True
                event['executed_stage'] = stage
            execution = self.artifact('native-call', event)
            applied = event.get('executed_stage') == 'fault'
            effect = self.artifact('effect', event['receipt']) if applied else None
            if decision is not None:
                self.coverage.finish(decision, Outcome.HIT if applied else Outcome.PASSTHROUGH,
                    execution_sha256=execution, effect_sha256=effect, native_applied=applied, original_calls=event['original_calls'])
        except BaseException as error:
            event['error'] = repr(error)
            self.failure = self.failure or repr(error)
            failed = self.artifact('failed-native-call', event)
            if decision is not None:
                self.coverage.finish(decision, Outcome.FAILED, execution_sha256=failed,
                    native_applied=event.get('executed_stage') == 'fault', original_calls=event['original_calls'])
            raise

    def end(self, model_output_sha256, *, model_output_finite):
        require(self.current is not None and self.frame is None, 'COMPLETE_ACTUAL_GROUP_FORWARD_REQUIRED')
        self._restore_apis()
        if self.phase != 'off':
            self.runtime.pause()
        self.paused()
        after = self.runtime.independent_inventory()
        total = inventory_delta(self.current['native_before'], after, self.runtime.operator)
        public = Counter()
        for row in self.public_calls[self.current['public_start']:]:
            public.update(row.get('native_delta', {}))
        require(dict(public) == total, 'NATIVE_ELIGIBLE_QMM_OUTSIDE_PUBLIC_OBSERVER')
        events = self.current['native_events']
        require(all(row['public_chain_verified'] and not row['errors'] for row in self.current['projections']),
                'ACTUAL_COMPLETE_PUBLIC_PROJECTION_LINEAGE_REJECTED')
        if self.phase != 'off':
            require(dict(Counter(row['native_identity'] for row in events)) == total,
                    'INDEPENDENT_NATIVE_QMM_OUTSIDE_CALLBACK_OBSERVER')
        healthy = self.phase not in ('fault', 'fault_repeat')
        if self.baseline_output is None:
            self.baseline_output = model_output_sha256
        require(not healthy or (model_output_finite and model_output_sha256 == self.baseline_output),
                'COMPLETE_MODEL_HEALTHY_LOSSLESS_OR_OFF_CHANGED')
        require(not self.overflow_calls and not self.failure, 'PUBLIC_OR_NATIVE_OBSERVER_FAILED')
        row = self.current
        row.pop('ordinals')
        row.update(model_output_sha256=model_output_sha256, model_output_finite=model_output_finite,
            public_end=len(self.public_calls), native_after=after, independent_native_qmm_calls=sum(total.values()),
            metadata_arrays=self.metadata.latest)
        self.rows.append(row)
        self.current = None
        self.last_output = model_output_sha256
        if self.phase == 'discovery_2':
            self._freeze()
        elif self.targets and self.phase != 'off':
            require(Counter(tuple(event.get('site', ())) for event in events) == Counter({site: 1 for site in self.targets}),
                    'FROZEN_ALL_TARGET_NATIVE_DENOMINATOR_CHANGED')
        if self.phase == 'qualification_off':
            self.artifact('native-group-seal', self.runtime.seal_group(row['key'] for row in self.targets.values()))
        self.artifact('request', row)
        self.phase_index += 1

    def _freeze(self):
        from fault_injection.coverage import TargetIdentity
        first, second = self.rows[0]['native_events'], self.rows[1]['native_events']
        require(first and all(row['public_lineage_verified'] for row in first + second),
                'ALL_ACTUAL_NATIVE_QMM_CALLS_REQUIRE_PUBLIC_LINEAGE')
        def index(rows):
            result = {tuple(row['site']): row for row in rows}
            require(len(result) == len(rows), 'ONE_NATIVE_QMM_PER_PUBLIC_SITE_REQUIRED')
            return result
        left, right = index(first), index(second)
        require(left.keys() == right.keys(), 'HEALTHY_NATIVE_TARGET_SET_CHANGED')
        for site, row in right.items():
            require(row['target'] == left[site]['target'] and row['native_identity'] == left[site]['native_identity'],
                    'HEALTHY_MODULE_EXECUTOR_OR_CACHE_LIFETIME_CHANGED')
            fields = dict(row['target'], module_roles=tuple(row['target']['module_roles']),
                          provider_proof=tuple(tuple(item) for item in row['target']['provider_proof']))
            target = TargetIdentity(**fields)
            self.targets[site] = dict(identity=target, key=row['target_key'], native_identity=row['native_identity'],
                device_bytes=next(slot['storage_bytes'] for slot in row['facts']['slots'] if slot['formal'] == 'x1'),
                host_bytes=max(slot['storage_bytes'] for slot in row['facts']['slots']))

    def _build_group(self):
        from fault_injection.coverage import (Candidate, CandidateSet, CoverageCapability, Eligibility,
            EligibilityResult, QualifiedTargetSet, TargetLease, TargetLosslessProof)
        from fault_injection.coverage_session import (CoverageScope, CoverageSession, GroupRecoveryHooks,
            IndependentCallObserver, SessionBudget)
        candidate_rows = [Candidate(row['key'], EligibilityResult(Eligibility.SUPPORTED), row['identity'])
                          for row in self.targets.values()]
        bound_modules = {site[0] for site in self.targets}
        observed_modules = {row['module_object_identity'] for forward in self.rows[:2] for row in forward['projections']}
        for path, module, _, _ in self.candidates:
            if id(module) not in bound_modules:
                # Structural candidates do not disappear merely because an
                # eligible callback was absent. A later concrete unsupported
                # proof is required to exclude them from all_eligible.
                state = Eligibility.UNVERIFIED if id(module) in observed_modules else Eligibility.NOT_OBSERVED
                reason = 'PUBLIC_PROJECTION_NATIVE_ELIGIBILITY_UNRESOLVED' if id(module) in observed_modules else 'STRUCTURAL_PROJECTION_NOT_EXECUTED'
                candidate_rows.append(Candidate(path, EligibilityResult(state, (reason,))))
        candidates = CandidateSet(tuple(candidate_rows), True)
        self.artifact('candidate-set', candidates.report())
        leases, proofs = [], []
        for site, row in self.targets.items():
            lease = TargetLease(row['identity'], self.group_owner, self.logical_generation,
                digest([self.group_owner, row['key']]), row['device_bytes'], row['host_bytes'])
            leases.append(lease)
            actual, reference = self.lossless[site], self.references[site]
            proofs.append(TargetLosslessProof(lease, 'lossless', self.artifact('lossless', actual),
                tuple(actual['input_sha256_before']), tuple(actual['input_sha256_after']), reference['output_sha256'],
                actual['output_sha256'], actual['original_calls'], actual['temporary_readback_verified'], actual['resources_released']))
        capability = CoverageCapability(single_target=True, multi_target=True, every_match=True,
            max_concurrency=1, max_resources=8 * 1024**3, max_host_bytes=8 * 1024**3,
            group_recovery=True, recovery_scope='target_set', stop_policy='bounded_session', lossless_modes=('lossless',))
        self.group = QualifiedTargetSet(candidates, tuple(leases), tuple(proofs), capability,
                                        selection='all_eligible', trigger=self.trigger)
        implementation = digest([self.adapter_sha, self.observer_sha, self.runtime.build_profile,
                                 self.metadata.implementation_sha256])
        scope = CoverageScope('20.buffer_substitute', self.workload_sha,
            digest([self.environment_sha, self.worker_identity, self.runtime.operator]), implementation, self.evidence_kind)
        self.observer = IndependentCallObserver(source='independent_native_trace', observer_id=self.group_owner + ':native-v3',
            implementation_sha256=implementation, workload_sha256=self.workload_sha, max_calls=4096)
        targets = tuple(row['identity'] for row in self.targets.values())
        workers = tuple(dict.fromkeys(target.worker for target in targets))
        self.observer.seal_discovery(candidates, observed_targets=targets, artifact_sha256=self.artifact('discovery', self.rows[:2]),
            expected_workers=workers, observed_workers=workers, complete=True)
        self.coverage = CoverageSession(group=self.group, scope=scope, observer=self.observer,
            budget=SessionBudget(4096, 1200, 1),
            recovery=GroupRecoveryHooks(self._stop, self._drain, self._recover_target, self._health), c2_proof=self.c2_proof)

    def _stop(self, lease):
        if self.remove_native:
            self.runtime.pause()

    def _drain(self, lease, timeout_seconds):
        self.paused(registered=self.remove_native is not None)
        return True

    def _release(self):
        from fault_injection.native import _LEASE
        if self.remove_native:
            self.paused()
            self.remove_native()
            self.remove_native = None
        self.paused(False)
        if self.locked:
            _LEASE.release()
            self.locked = False

    def _recover_target(self, lease):
        from fault_injection.coverage import RecoveryReport
        self._release()
        site = next(site for site, row in self.targets.items() if row['identity'] == lease.target)
        off, reference = self.off.get(site), self.references[site]
        equal = bool(off and off['input_sha256_before'] == reference['input_sha256_before']
                     and off['output_sha256'] == reference['output_sha256'])
        proof = self.artifact('target-off', dict(lease=asdict(lease), reference=reference, off=off,
            native=self.runtime.snapshot(), group=self.runtime.group_state(), metadata_arrays=self.metadata.read()))
        return RecoveryReport(lease, True, self.remove_native is None, equal,
            None if equal else 'TARGET_OFF_NOT_PROVEN', proof, off['output_sha256'] if off else None)

    def _health(self, worker, owner, generation):
        from fault_injection.coverage_session import WorkerHealthProof
        states = self.paused(False)
        metadata = self.metadata.read(require_activity=True)
        hooks = self.owned_hooks + len(self.patches) + len(self.api_patches)
        proof = self.artifact('worker-health', dict(native=states, group=self.runtime.group_state(),
            worker=worker, baseline=self.baseline_output, off=self.last_output, owned_hooks=hooks,
            metadata_arrays=metadata))
        return WorkerHealthProof(worker, owner, generation, proof, self.baseline_output, self.last_output,
                                  0, 0, 0, 0, hooks)

    def finalize(self):
        require(self.phase == 'complete' and not self.failure and not self.active and self.owned_hooks == 0
                and not self.patches and not self.api_patches and self.coverage is not None,
                'COMPLETE_GROUP_OFF_AND_OBSERVER_RELEASE_REQUIRED')
        recovery = self.coverage.disable(timeout_seconds=30)
        self.observer.seal(artifact_sha256=self.artifact('full-independent-trace', self.rows), complete=True)
        require(recovery['complete'] and self.coverage.report()['contract_satisfied'], 'GROUP_COVERAGE_OR_RECOVERY_INCOMPLETE')
        self.proof = self.coverage.c2_proof() if self.trigger == 'once' else None
        self.recovered = True

    def close(self):
        require(not self.fault_started or self.recovered, 'SAME_PROCESS_GROUP_OFF_REQUIRED_BEFORE_RELEASE')
        super().close()
        if self.remove_native:
            self.runtime.pause()
        self._release()

    def fail(self, reason):
        self.failure = self.failure or self.runtime.last_error or str(reason)
        if self.current is not None:
            self.artifact('failed-request', self.current)
        self._restore_apis()
        self.current = None
        if self.remove_native:
            self.runtime.pause()

    def request_recovery(self):
        raise RuntimeError('FAILED_GROUP_REQUIRES_REVIEWED_SAME_PROCESS_OFF_RECOVERY')

    def query(self):
        return {'qualification': dict(status='C2_C3_EVIDENCE_READY_FOR_REVIEW' if self.recovered else
            'ATTEMPT_FAILED' if self.failure else 'WAITING_FOR_REAL_REQUEST', failure=self.failure,
            next_phase=self.phase, recovery_verified=self.recovered, trigger=self.trigger, targets=len(self.targets),
            coverage_scope='ALL_ACTUAL_NATIVE_ELIGIBLE_QMM_IN_FIXED_FRESH_PREFILL',
            structural_candidates=len(self.candidates), unobserved_structural_candidates_are_not_qualified=True,
            completed_phases=[row['phase'] for row in self.rows],
            coverage=self.coverage.report() if self.coverage is not None else None)}
