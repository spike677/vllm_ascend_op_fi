"""Bounded exact-cache MatMul groups; C1's unique qualification is unchanged.

The independent native observer counts eligible Runs even while disabled. The
serving adapter supplies actual module/call/request ownership and tensor pairs.
Discovery and every native execution are reconciled against that counter map.
This module schedules no workload and creates no Worker.
"""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

from .inspector import Fault39OutputInspector, validate_statistics


PHASES = ('discovery_1', 'discovery_2', 'reference', 'lossless_self',
          'lossless_mat2', 'lossless_dual', 'fault', 'off_observed', 'off')
MODES = {'reference': ('observer', 3), 'lossless_self': ('lossless', 1),
         'lossless_mat2': ('lossless', 2), 'lossless_dual': ('lossless', 3),
         'fault': ('inject', 3), 'fault_repeat': ('inject', 3), 'off_observed': ('observer', 3)}


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def native_counts(snapshot):
    """Read the independent map, not candidates, hits or injected receipts."""
    require(snapshot.get('independent_observer_complete') is True,
            'INDEPENDENT_NATIVE_OBSERVER_OVERFLOW_OR_UNAVAILABLE')
    require(snapshot.get('independent_observer_max_bytes') == 512 * 1024**2,
            'INDEPENDENT_OBSERVER_ELIGIBILITY_SCOPE_CHANGED')
    result = {}
    for row in snapshot.get('independent_candidates', ()):
        for cache in row['cache_identities']:
            key = (row['signature'], cache['address'], cache['epoch'])
            require(key not in result and type(cache['observations']) is int
                    and cache['observations'] > 0, 'DUPLICATE_OR_INVALID_NATIVE_OBSERVATION')
            result[key] = cache['observations']
    require(sum(result.values()) == snapshot.get('independent_eligible_runs'),
            'INDEPENDENT_NATIVE_DENOMINATOR_INCOMPLETE')
    return result


def native_delta(before, after):
    left, right = native_counts(before), native_counts(after)
    require(all(right.get(key, -1) >= count for key, count in left.items()),
            'INDEPENDENT_NATIVE_COUNTER_RESET')
    return {key: count - left.get(key, 0) for key, count in right.items()
            if count != left.get(key, 0)}


def reconcile_discovery(first, second):
    """Every native eligible call needs one unchanged, owned public boundary."""
    def indexed(rows):
        result = {}
        for row in rows:
            if not row['native_delta']:
                continue
            require(len(row['native_delta']) == 1 and row['native_delta'][0]['count'] == 1,
                    'MULTIPLE_NATIVE_RUNS_AT_ONE_PUBLIC_BOUNDARY')
            require(row.get('public_inputs_verified') is True and row.get('module_object_identity'),
                    'NATIVE_ELIGIBLE_CALL_HAS_NO_REVIEWED_PUBLIC_INPUTS_OR_MODULE')
            key = tuple(row['site'])
            require(key not in result, 'DUPLICATE_ACTUAL_DISPATCH_SITE')
            result[key] = row
        return result
    left, right = indexed(first), indexed(second)
    require(left and left.keys() == right.keys(), 'COMPLETE_HEALTHY_TARGET_SET_CHANGED')
    for site, row in left.items():
        other = right[site]
        fields = ('module_object_identity', 'module_path', 'operator', 'tensor_signature')
        require(all(row.get(field) == other.get(field) for field in fields)
                and row['native_delta'][0]['key'] == other['native_delta'][0]['key'],
                'TARGET_MODULE_EXECUTOR_OR_REGISTRATION_LIFETIME_CHANGED')
    return right


class MatmulCoverageBackend:
    """Trusted per-call v2 dispatcher, followed by full group OFF and health."""
    def __init__(self, native, *, owner, worker_identity, model, folder,
                 identity, trigger='once', c2_proof=None, max_dispatch_calls=20000,
                 max_seconds=1200, evidence_kind='natural_worker'):
        require(trigger in ('once', 'every_match'), 'BOUNDED_TRIGGER_REQUIRED')
        require(trigger != 'every_match' or c2_proof is not None, 'REAL_C2_PROOF_REQUIRED')
        self.native, self.owner, self.worker_identity, self.model = native, owner, worker_identity, model
        self.folder, self.identity = Path(folder), identity
        self.initial_identity = identity()
        self.trigger, self.c2_proof = trigger, c2_proof
        # The public Worker issues a fresh opaque native owner on each prepare.
        # Logical coverage lineage remains the live C2 owner; actual native
        # ownership and every configure generation are separately attested.
        self.group_owner = c2_proof.owner if c2_proof is not None else owner
        self.phases = list(PHASES)
        if trigger == 'every_match':
            self.phases.insert(self.phases.index('off_observed'), 'fault_repeat')
        self.max_dispatch_calls, self.max_seconds = max_dispatch_calls, max_seconds
        self.evidence_kind = evidence_kind
        self.config = self.generation = None
        self.phase_index = 0
        self.started = self.active = self.fault_started = self.recovered = self.recovering = False
        self.failure = None
        self.rows, self.requests = [], set()
        self.current = self.before = self.reference = None
        self.targets, self.references, self.lossless, self.off_outputs = {}, {}, {}, {}
        self.coverage = self.observer = self.group = self.proof = None
        self.last_output = self.baseline_output = self.workload_sha256 = None
        self.first_anomaly = False
        self.owned_hooks = 0
        self.artifact_index = 0
        self.logical_generation = c2_proof.generation + 1 if c2_proof is not None else 1
        self.adapter_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.observer_sha256 = self.adapter_sha256  # CPU adapters explicitly override their scope kind.
        self.build_profile = self.native.snapshot().get('build_profile_sha256')

    @property
    def phase(self):
        return self.phases[self.phase_index] if self.phase_index < len(self.phases) else 'complete'

    def artifact(self, label, value):
        self.folder.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + '\n').encode()
        self.artifact_index += 1
        path = self.folder / ('%05d-%s.json' % (self.artifact_index, label))
        with path.open('xb') as stream:
            stream.write(data)
        return hashlib.sha256(data).hexdigest()

    def snapshot(self):
        state = self.native.snapshot()
        require(self.identity() == self.initial_identity and state.get('runtime_identity') == 'MATCHED_BUILD'
                and state.get('build_profile_sha256') == self.build_profile
                and state.get('exact_cache_api_version') == 2
                and state.get('owner') in ('', self.owner), 'NATIVE_BUILD_OWNER_OR_MODEL_CHANGED')
        native_counts(state)
        return state

    def prepare(self):
        state = self.snapshot()
        require(state['enabled'] is False and state['inflight'] == state['leases'] == 0,
                'NATIVE_GROUP_NOT_IDLE')
        return state

    def register(self, config):
        require(self.config is None and config.max_hits == 1 and config.max_bytes == 512 * 1024**2,
                'ONE_CALL_NATIVE_LEASE_AND_REVIEWED_BYTE_BUDGET_REQUIRED')
        self.config = config

    def start(self):
        require(self.config is not None and not self.started, 'ONE_BOUNDED_GROUP_ATTEMPT_REQUIRED')
        self.started = True
        self.started_at = time.monotonic()

    def begin(self, request_id, input_sha256):
        require(self.started and not self.active and not self.failure and self.phase != 'complete',
                'GROUP_PHASE_NOT_AVAILABLE')
        require(request_id not in self.requests and isinstance(request_id, str) and request_id,
                'FRESH_ACTUAL_REQUEST_REQUIRED')
        require(time.monotonic() - self.started_at < self.max_seconds, 'GROUP_TIME_BUDGET_EXHAUSTED')
        if self.workload_sha256 is None:
            self.workload_sha256 = input_sha256
        require(self.workload_sha256 == input_sha256, 'ACTUAL_WORKLOAD_INPUT_CHANGED')
        self.before = self.prepare()
        self.current = dict(phase=self.phase, request_id=request_id, input_sha256=input_sha256, calls=[])
        self.requests.add(request_id)
        self.active = True
        if self.phase == 'discovery_1':
            self.generation = self.native.configure(self.owner, self.config, self.worker_identity['rank'],
                self.worker_identity['logical_device'], mode='discover')
        elif self.phase == 'discovery_2':
            with self.native.context(self.owner, [request_id]):
                self.native.resume_discovery(self.owner, self.generation)
        elif self.phase in ('fault', 'fault_repeat'):
            if self.coverage is None:
                self._build_group()
            self.fault_started = True

    def _stop_native(self, clean=False):
        if self.generation is None:
            return {'status': 'DISABLED_CONFIRMED'}
        fn = self.native.clean if clean else self.native.disable
        result = fn(self.owner, self.generation, 10000)
        state = result.get('snapshot', {})
        require(result.get('status') == 'DISABLED_CONFIRMED' and result.get('native_code') == 0
                and result.get('owner') == self.owner and result.get('generation') == self.generation
                and state.get('enabled') is False and state.get('inflight') == state.get('leases') == 0
                and (state.get('owner') == '' if clean else state.get('owner') == self.owner),
                'NATIVE_GROUP_DRAIN_OR_RELEASE_UNPROVEN')
        if clean:
            self.generation = None
        return result

    def dispatch(self, site, row, pair_values, original):
        """Observer records all actual dispatches; admission cannot drop a call."""
        from fault_injection.coverage import ObservedCall, Outcome, Phase
        require(self.active and len(self.current['calls']) < self.max_dispatch_calls,
                'GROUP_DISPATCH_BUDGET_OR_ACTIVE_REQUEST_REQUIRED')
        before = self.snapshot()
        expected = self.targets.get(tuple(site))
        decision = None
        phase = self.phase
        mode_mask = MODES.get(phase)
        inject = mode_mask is not None and mode_mask[0] == 'inject'
        if self.coverage is not None and expected is not None:
            call = ObservedCall(expected['identity'], '%s:%d' % (self.current['request_id'], len(self.current['calls'])),
                self.current['request_id'], (self.current['request_id'],),
                Phase.OFF_RECOVERY if phase.startswith('off') else
                (Phase.POST_FIRST_ANOMALY if self.first_anomaly else Phase.PRE_FAULT_PROPAGATION))
            # Native observation from the first two independent healthy passes
            # binds this public boundary. After execution the actual native delta
            # must confirm exactly that identity; it can never be inferred from hits.
            ticket = self.observer.observe(call, eligible=True)
            decision = self.coverage.begin(ticket)
        use_native = expected is not None and mode_mask is not None and (
            not inject or (decision is not None and decision.admitted))
        result = None
        original_calls = 0
        try:
            with self.native.context(self.owner, [self.current['request_id']]):
                if use_native:
                    require(pair_values is not None, 'REVIEWED_LIVE_NATIVE_INPUT_PAIR_REQUIRED')
                    self.native.bind_serving_inputs(self.owner, pair_values)
                    signature, address, epoch = expected['native_key']
                    self.generation = self.native.configure_cache(self.owner, self.config,
                        self.worker_identity['rank'], self.worker_identity['logical_device'],
                        mode=mode_mask[0], input_mask=mode_mask[1], target_signature=signature,
                        cache_address=address, cache_epoch=epoch)
                original_calls += 1
                result = original()
                after = self.snapshot()
            delta = native_delta(before, after)
            row.update(site=list(site), native_delta=[dict(key=list(key), count=count) for key, count in delta.items()],
                       public_inputs_verified=pair_values is not None, original_calls=original_calls,
                       actual_native_owner=self.owner, actual_native_generation=self.generation)
            if expected is not None:
                require(delta == {expected['native_key']: 1}, 'ACTUAL_NATIVE_IDENTITY_OR_CALL_COUNT_CHANGED')
            elif self.targets and delta:
                raise RuntimeError('PROPAGATION_ADDED_NATIVE_TARGET_REQUIRES_NEW_DISCOVERY')
            if use_native:
                self._check_receipt(after, expected, mode_mask)
                row['native'] = after
            else:
                require(after.get('hits') == before.get('hits'), 'NON_TARGET_OR_DISABLED_CALL_WAS_MODIFIED')
            execution_sha = self.artifact('call', row)
            effect_sha = None
            if inject and use_native:
                changed = after['last_observation']['output_storage_sha256'] != self.references[tuple(site)]['output']
                effect_sha = self.artifact('effect', dict(site=site, execution_sha256=execution_sha,
                    reference_output_sha256=self.references[tuple(site)]['output'],
                    actual_output_sha256=after['last_observation']['output_storage_sha256'], changed=changed))
                require(changed, 'TARGET_NATIVE_OUTPUT_EFFECT_MISSING')
                self.first_anomaly = True
            if decision is not None:
                self.coverage.finish(decision, Outcome.HIT if inject and use_native else Outcome.PASSTHROUGH,
                    execution_sha256=execution_sha, effect_sha256=effect_sha,
                    native_applied=bool(inject and use_native), original_calls=original_calls)
            self.current['calls'].append(row)
            return result
        except BaseException as exc:
            row.update(error=repr(exc), original_calls=original_calls)
            if decision is not None:
                try:
                    self.coverage.finish(decision, Outcome.FAILED, execution_sha256=self.artifact('failed-call', row),
                                         native_applied=bool(use_native and inject), original_calls=original_calls)
                except BaseException as finish_error:
                    row['coverage_finish_error'] = repr(finish_error)
            self.current['calls'].append(row)
            self.failure = repr(exc)
            raise
        finally:
            if use_native:
                self._stop_native()

    def _check_receipt(self, state, target, mode_mask):
        mode, mask = mode_mask
        obs = state.get('last_observation', {})
        proof = obs.get('proof', {})
        signature, address, epoch = target['native_key']
        require(state.get('owner') == self.owner and state.get('generation') == self.generation
                and state.get('hits') == 1 and state.get('last_error') == 0
                and state.get('inflight') == state.get('leases') == 0
                and state.get('target_signature') == signature
                and obs.get('cache_address') == address and obs.get('cache_epoch') == epoch
                and obs.get('owner') == self.owner and obs.get('generation') == self.generation
                and obs.get('hit') == proof.get('hit') == 1 and proof.get('generation') == self.generation
                and obs.get('serving_live_inputs_verified') is True
                and json.loads(obs.get('actual_request_ids_json', '[]')) == [self.current['request_id']]
                and all(obs.get(key) == 0 for key in ('pre_sync_status', 'copy_status', 'run_status', 'post_sync_status', 'free_status')),
                'EXACT_CACHE_ORIGINAL_RUN_OR_RESOURCE_PROOF_MISSING')
        before = obs.get('original_storage_sha256_before')
        require(isinstance(before, list) and len(before) == 2 and all(isinstance(v, str) and len(v) == 64 for v in before)
                and before == obs.get('original_storage_sha256_after'), 'ORIGINAL_INPUT_STORAGE_NOT_PRESERVED')
        tensors = proof.get('tensors', ())
        require(proof.get('api') == 'aclnnMatmul' and proof.get('formal_order_proven') is True
                and proof.get('slots') == [0, 1, 2] and proof.get('input_mask') == mask and len(tensors) == 3,
                'TYPED_NATIVE_FORMALS_UNPROVEN')
        replaced_mask = 0 if mode == 'observer' else mask
        require(all(bool(t.get('original_address')) and bool(t.get('run_address')) and
                    (t['original_address'] != t['run_address']) == bool(replaced_mask & (1 << i))
                    for i, t in enumerate(tensors)), 'EXACT_NATIVE_REPLACEMENT_SLOTS_UNPROVEN')
        for name in ('original_self', 'original_mat2', 'temporary_self', 'temporary_mat2', 'output'):
            validate_statistics(obs[name])
        output = obs.get('output_storage_sha256')
        require(isinstance(output, str) and len(output) == 64, 'COMPLETE_NATIVE_OUTPUT_HASH_REQUIRED')
        site = target['site']
        if self.phase == 'reference':
            require(all(obs[n]['finite'] == obs[n]['total'] > 0 for n in ('original_self', 'original_mat2', 'output')),
                    'FINITE_HEALTHY_NATIVE_REFERENCE_REQUIRED')
            self.references[site] = dict(inputs=tuple(before), output=output)
        elif mode != 'inject':
            ref = self.references[site]
            require(tuple(before) == ref['inputs'] and output == ref['output'], 'LOSSLESS_OR_OFF_NATIVE_OUTPUT_CHANGED')
            if mode == 'lossless':
                require(Fault39OutputInspector().inspect(state)['verdict'] == 'REAL_OPERATOR_HIT_VERIFIED'
                        and obs['changed'] == [0, 0] and obs.get('input_device_readback_verified') is True,
                        'LOSSLESS_REPLACEMENT_NOT_VERIFIED')
                self.lossless[(site, self.phase)] = state
            if self.phase == 'off_observed':
                self.off_outputs[site] = output
        else:
            # Propagated nonfinite originals are recorded, never reclassified as
            # healthy. Both substituted inputs themselves must be finite and the
            # same exact-cancellation construction remains mandatory.
            require(all(obs[n]['finite'] == obs[n]['total'] > 0 for n in ('temporary_self', 'temporary_mat2'))
                    and obs.get('input_device_readback_verified') is True
                    and obs.get('self_constant_positive_verified') is True
                    and obs.get('per_column_sign_counts_verified') is True
                    and obs.get('theoretical_output_exact_zero') is True
                    and obs.get('amplitude_equal') is True and sum(obs.get('changed', ())) > 0,
                    'FINITE_EXACT_CANCELLATION_INPUT_CONSTRUCTION_UNPROVEN')

    def end(self, output_sha256, *, model_output_finite):
        require(self.active and self.current is not None, 'NO_ACTIVE_GROUP_REQUEST')
        after = self.snapshot()
        total = native_delta(self.before, after)
        observed = Counter()
        for row in self.current['calls']:
            for item in row.get('native_delta', ()):
                observed[tuple(item['key'])] += item['count']
        require(dict(observed) == total, 'ELIGIBLE_NATIVE_CALL_OUTSIDE_INDEPENDENT_PUBLIC_OBSERVER')
        healthy = self.phase not in ('fault', 'fault_repeat')
        require(not healthy or model_output_finite, 'HEALTHY_COMPLETE_MODEL_OUTPUT_NOT_FINITE')
        if self.baseline_output is None:
            self.baseline_output = output_sha256
        require(not healthy or output_sha256 == self.baseline_output, 'COMPLETE_MODEL_LOSSLESS_OR_OFF_CHANGED')
        self.current.update(model_output_sha256=output_sha256, model_output_finite=model_output_finite,
                            native_denominator=sum(total.values()), independent_denominator_complete=True)
        self._stop_native()
        self.rows.append(self.current)
        self.last_output = output_sha256
        self.active = False
        self.current = None
        if self.phase == 'discovery_2':
            self._freeze_targets(reconcile_discovery(self.rows[0]['calls'], self.rows[1]['calls']))
        self.phase_index += 1
        self.artifact('request', self.rows[-1])

    def _freeze_targets(self, sites):
        from fault_injection.coverage import TargetIdentity
        from fault_injection.runtime_discovery import _process_start_identity
        build = self.snapshot()['build_profile_sha256']
        for site, row in sites.items():
            key = tuple(row['native_delta'][0]['key'])
            identity = TargetIdentity(process_id=os.getpid(), worker_id=self.worker_identity['worker_start_identity'],
                rank=self.worker_identity['rank'], logical_device=self.worker_identity['logical_device'],
                model_instance_id=self.worker_identity['model_instance_id'], model_object_identity=id(self.model),
                module_object_identity=row['module_object_identity'], module_roles=(), operator_api='aclnnMatmul',
                execution_boundary='op_exec_cache_run', executor_identity=str(key[1]),
                cache_lifecycle='AddOpExecCache-registration:' + str(key[2]), tensor_signature=row['tensor_signature'],
                native_build_fingerprint=build, provider_identity='fault39.exact_cache.v2',
                provider_proof=(('public_site', digest(site)), ('runtime_signature', key[0])),
                process_start_identity=_process_start_identity())
            self.targets[site] = dict(identity=identity, native_key=key, site=site,
                device_bytes=sum(row['input_storage_bytes']),
                host_bytes=sum(row['input_storage_bytes']) + row['output_bytes'])

    def _build_group(self):
        from fault_injection.coverage import (Candidate, CandidateSet, CoverageCapability, Eligibility,
            EligibilityResult, QualifiedTargetSet, TargetLease, TargetLosslessProof)
        from fault_injection.coverage_session import (CoverageScope, CoverageSession, GroupRecoveryHooks,
            IndependentCallObserver, SessionBudget)
        candidates = CandidateSet(tuple(Candidate(digest(site), EligibilityResult(Eligibility.SUPPORTED), row['identity'])
                                        for site, row in self.targets.items()), True)
        leases, proofs = [], []
        for site, row in self.targets.items():
            lease = TargetLease(row['identity'], self.group_owner, self.logical_generation, digest([self.group_owner, site]),
                                row['device_bytes'], row['host_bytes'])
            leases.append(lease)
            for phase in ('lossless_self', 'lossless_mat2', 'lossless_dual'):
                state = self.lossless[(site, phase)]
                obs, ref = state['last_observation'], self.references[site]
                proofs.append(TargetLosslessProof(lease, phase, self.artifact('lossless', state), tuple(obs['original_storage_sha256_before']),
                    tuple(obs['original_storage_sha256_after']), ref['output'], obs['output_storage_sha256'], 1,
                    obs['input_device_readback_verified'], state['inflight'] == state['leases'] == 0))
        cap = CoverageCapability(single_target=True, multi_target=True, every_match=True,
            max_concurrency=1, max_resources=8 * 1024**3, max_host_bytes=8 * 1024**3, group_recovery=True,
            recovery_scope='target_set', stop_policy='bounded_session', lossless_modes=('lossless_self', 'lossless_mat2', 'lossless_dual'))
        self.group = QualifiedTargetSet(candidates, tuple(leases), tuple(proofs), cap,
                                       selection='all_eligible', trigger=self.trigger)
        native = self.snapshot()
        implementation_sha256 = digest([self.adapter_sha256, self.observer_sha256, native['build_profile_sha256']])
        scope = CoverageScope('39.matmul_range', self.workload_sha256,
                              digest([native['build_profile_sha256'], self.worker_identity]), implementation_sha256,
                              evidence_kind=self.evidence_kind)
        self.observer = IndependentCallObserver(source='independent_native_trace', observer_id=self.group_owner + ':native-observer',
            implementation_sha256=implementation_sha256, workload_sha256=self.workload_sha256, max_calls=self.max_dispatch_calls)
        targets = tuple(row['identity'] for row in self.targets.values())
        workers = tuple(dict.fromkeys(target.worker for target in targets))
        self.observer.seal_discovery(candidates, observed_targets=targets,
            artifact_sha256=self.artifact('discovery', self.rows[:2]), expected_workers=workers,
            observed_workers=workers, complete=True)
        hooks = GroupRecoveryHooks(self._lease_stop, self._lease_drain, self._lease_recover, self._worker_health)
        self.coverage = CoverageSession(group=self.group, scope=scope, observer=self.observer,
            budget=SessionBudget(self.max_dispatch_calls, self.max_seconds, 1), recovery=hooks, c2_proof=self.c2_proof)

    def _lease_stop(self, lease):
        self._stop_native()

    def _lease_drain(self, lease, timeout_seconds):
        self._stop_native()
        state = self.snapshot()
        return state['enabled'] is False and state['inflight'] == state['leases'] == 0

    def _lease_recover(self, lease):
        from fault_injection.coverage import RecoveryReport
        self._stop_native(clean=True)
        site = next(site for site, row in self.targets.items() if row['identity'] == lease.target)
        ref = self.references[site]['output']
        off = self.off_outputs.get(site)
        released = self.generation is None
        proof = self.artifact('target-off', dict(lease=asdict(lease), reference=ref, off=off, native=self.snapshot()))
        return RecoveryReport(lease, True, released, off == ref,
            None if released and off == ref else 'OFF_OR_RELEASE_INCOMPLETE', proof, off)

    def _worker_health(self, worker, owner, generation):
        from fault_injection.coverage_session import WorkerHealthProof
        state = self.snapshot()
        require(not state['owner'], 'NATIVE_OWNER_NOT_RELEASED')
        proof = self.artifact('worker-health', dict(native=state, baseline=self.baseline_output, off=self.last_output,
                                                   owned_hooks=self.owned_hooks, worker=list(worker)))
        return WorkerHealthProof(worker, owner, generation, proof, self.baseline_output, self.last_output,
                                 state['leases'], 0, 0, state['inflight'], self.owned_hooks)

    def finalize(self):
        require(self.phase == 'complete' and not self.active and not self.failure and self.owned_hooks == 0,
                'COMPLETE_OFF_AND_HOOK_RELEASE_REQUIRED')
        require(self.coverage is not None, 'COVERAGE_SESSION_REQUIRED')
        recovery = self.coverage.disable(timeout_seconds=30)
        self.observer.seal(artifact_sha256=self.artifact('independent-full-trace', self.rows), complete=True)
        require(recovery['complete'] and self.coverage.report()['contract_satisfied'], 'GROUP_COVERAGE_OR_RECOVERY_INCOMPLETE')
        self.proof = self.coverage.c2_proof() if self.trigger == 'once' else None
        self.recovered = True
        return self.proof

    def fail(self, reason):
        self.failure = str(reason)
        self.active = False
        self._stop_native()

    def disable(self, timeout_ms=10000):
        return self._stop_native()

    def unregister(self, timeout_ms=10000):
        require(not self.fault_started or self.recovered, 'GROUP_OFF_RECOVERY_REQUIRED_BEFORE_RELEASE')
        return self._stop_native(clean=True)

    def request_recovery(self):
        raise RuntimeError('FAILED_GROUP_REQUIRES_REVIEWED_OFF_RECOVERY_NOT_AUTOMATIC_RETRY')

    def query(self):
        return {'qualification': dict(status='C2_C3_EVIDENCE_READY_FOR_REVIEW' if self.recovered else
                    'ATTEMPT_FAILED' if self.failure else 'WAITING_FOR_REAL_REQUEST',
                next_phase=self.phase, failure=self.failure, recovery_verified=self.recovered,
                completed_phases=[row['phase'] for row in self.rows],
                actual_request_ids=[row['request_id'] for row in self.rows], targets=len(self.targets),
                trigger=self.trigger, native_matching='EXACT_CACHE_V2',
                coverage=self.coverage.report() if self.coverage is not None else None)}
