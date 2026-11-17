"""Host-driven, one-attempt qualification across genuine serving requests.

This state machine neither schedules requests nor creates a Worker. A reviewed
host adapter calls begin/end on the model thread, around the actual output
projection, and supplies complete model observations. Native state is disabled
between requests. Two extra read-only Runs make the original-storage proof
explicit instead of inventing storage hashes during discovery.
"""
import json

from .backend import NativeBackend
from .host_qualification import is_sha256
from .inspector import Fault39OutputInspector, validate_statistics
from .target import select_stable_target


PHASES = ('discovery_1', 'discovery_2', 'reference', 'lossless_self',
          'lossless_mat2', 'lossless_dual', 'fault', 'off_observed', 'off')
_MODES = {'reference': ('observer', 4, 3), 'lossless_self': ('lossless', 3, 1),
          'lossless_mat2': ('lossless', 3, 2), 'lossless_dual': ('lossless', 3, 3),
          'fault': ('inject', 2, 3), 'off_observed': ('observer', 4, 3)}


class ServingQualification(NativeBackend):
    """Single unique target within a declared, structurally verified boundary."""

    def __init__(self, workers, *, owner, build_profile_sha256, identity, baseline_control=False):
        super().__init__(workers, owner)
        if len(workers) != 1 or not is_sha256(build_profile_sha256) or not callable(identity):
            raise ValueError('MATCHED_SINGLE_WORKER_REQUIRED')
        self.endpoint = next(iter(workers.values()))
        self.profile = build_profile_sha256
        self.identity = identity
        self.initial_identity = identity()
        if not isinstance(self.initial_identity, str) or not self.initial_identity:
            raise ValueError('LIVE_WORKER_IDENTITY_REQUIRED')
        self.phase_index = 0
        self.started = self.active = self.fault_started = self.recovered = False
        self.failure = None
        self.recovering = self.recovery_attempted = self.fault_arm_consumed = False
        self.drain_status = 'NOT_REQUESTED'
        self.rows = []
        self.requests = set()
        self.baseline = self.storage_reference = self.native_reference = None
        self.before = self.request = None
        if type(baseline_control) is not bool:
            raise ValueError('EXPLICIT_BASELINE_CONTROL_BOOLEAN_REQUIRED')
        self.baseline_control = baseline_control
        self.modes = dict(_MODES)
        if baseline_control:
            self.modes['fault'] = ('observer', 4, 3)

    def _snapshot(self):
        state = self.endpoint.provider.snapshot()
        if (self.identity() != self.initial_identity or state.get('abi_version') != 1
                or state.get('runtime_identity') != 'MATCHED_BUILD'
                or state.get('runtime_loaded') is not True
                or state.get('build_profile_sha256') != self.profile
                or state.get('worker_rank') != self.endpoint.rank
                or state.get('owner') not in ('', self.owner)):
            raise RuntimeError('LIVE_WORKER_BUILD_OR_OWNER_CHANGED')
        return state

    def start(self):
        if self.started or self.config is None or self.config.max_hits != 1 or self.config.target_policy != 'unique':
            raise RuntimeError('SINGLE_UNIQUE_REGISTERED_ATTEMPT_REQUIRED')
        self._snapshot()
        self.started = True

    @property
    def phase(self):
        return PHASES[self.phase_index] if self.phase_index < len(PHASES) else 'complete'

    def begin(self, request_id, input_sha256):
        if not self.started or self.active or (self.failure and not self.recovering) or self.phase == 'complete':
            raise RuntimeError('SERVING_PHASE_NOT_AVAILABLE')
        if (not isinstance(request_id, str) or not request_id or request_id in self.requests
                or not is_sha256(input_sha256)):
            raise RuntimeError('FRESH_ACTUAL_SCHEDULER_REQUEST_REQUIRED')
        if self.baseline is not None and input_sha256 != self.baseline['input_sha256']:
            raise RuntimeError('REPLAY_INPUT_CHANGED')
        self.before = self._snapshot()
        if (self.before.get('enabled') is not False or self.before.get('inflight') != 0
                or self.before.get('leases') != 0):
            raise RuntimeError('NATIVE_NOT_IDLE_BETWEEN_REQUESTS')
        self.request = dict(request_id=request_id, input_sha256=input_sha256, phase=self.phase)
        self.requests.add(request_id)
        self.active = True
        try:
            if self.phase == 'discovery_1':
                NativeBackend.discover(self)
            elif self.phase == 'discovery_2':
                # Resume only under provider.context for the actual request.
                self.endpoint.provider.resume_discovery(self.owner, self.generations[self.endpoint.rank])
                self._state = 'DISCOVERING'
            elif self.phase in self.modes:
                mode, _, mask = self.modes[self.phase]
                if self.phase == 'fault' and not self.baseline_control:
                    self.fault_started = True  # consume before a disputed arm
                self._arm(mode, input_mask=mask)
        except BaseException:
            self.fail('ARM_FAILED')
            raise

    def enable(self):
        raise RuntimeError('HOST_DRIVEN_QUALIFICATION_REQUIRED')

    def discover(self):
        raise RuntimeError('HOST_DRIVEN_QUALIFICATION_REQUIRED')

    def _arm(self, mode, *, input_mask=None):
        expected = ('discover', 3) if self.phase == 'discovery_1' else (
            (self.modes[self.phase][0], self.modes[self.phase][2]) if self.phase in self.modes else None)
        if (not self.active or expected != (mode, input_mask or 3)
                or (self.failure and not self.recovering)):
            raise RuntimeError('ARM_OUTSIDE_QUALIFIED_PHASE')
        if mode == 'inject':
            if self.fault_arm_consumed or [r['phase'] for r in self.rows] != list(PHASES[:6]):
                raise RuntimeError('LOSSLESS_SEQUENCE_OR_SINGLE_ATTEMPT_REQUIRED')
            self.fault_arm_consumed = True
        return NativeBackend._arm(self, mode, input_mask=input_mask)

    def _check_native(self, native, request):
        mode, number, mask = self.modes[request['phase']]
        observation = native.get('last_observation', {})
        proof = observation.get('proof', {})
        before = observation.get('original_storage_sha256_before')
        after = observation.get('original_storage_sha256_after')
        output_hash = observation.get('output_storage_sha256')
        if (native.get('owner') != self.owner
                or native.get('generation') != self.generations[self.endpoint.rank]
                or native.get('target_signature') != self.targets[self.endpoint.rank]
                or native.get('rank') != self.endpoint.rank or native.get('device') != self.endpoint.device
                or native.get('mode') != number or native.get('hits') != 1
                or native.get('interposed_run_calls', 0) <= self.before.get('interposed_run_calls', 0)
                or native.get('inflight') != 0 or native.get('leases') != 0 or native.get('last_error') != 0
                or observation.get('owner') != self.owner or observation.get('generation') != native['generation']
                or observation.get('hit') != 1 or proof.get('hit') != 1
                or observation.get('serving_live_inputs_verified') is not True
                or proof.get('generation') != native['generation'] or proof.get('api') != 'aclnnMatmul'
                or proof.get('formal_order_proven') is not True or proof.get('slots') != [0, 1, 2]
                or proof.get('input_mask') != mask
                or json.loads(observation.get('actual_request_ids_json', '[]')) != [request['request_id']]
                or not isinstance(before, list) or len(before) != 2 or before != after
                or not all(is_sha256(value) for value in before) or not is_sha256(output_hash)
                or any(observation.get(key) != 0 for key in
                       ('pre_sync_status', 'run_status', 'post_sync_status', 'copy_status', 'free_status'))):
            raise RuntimeError('COMPLETE_NATIVE_STORAGE_OR_RUN_PROOF_MISSING')
        if self.storage_reference is None:
            self.storage_reference = before
            self.native_reference = output_hash
        if before != self.storage_reference:
            raise RuntimeError('ORIGINAL_NATIVE_INPUT_CHANGED_BETWEEN_REQUESTS')
        if mode != 'inject' and output_hash != self.native_reference:
            raise RuntimeError('LOSSLESS_OR_OFF_NATIVE_OUTPUT_CHANGED')
        for field in ('original_self', 'original_mat2', 'temporary_self', 'temporary_mat2', 'output'):
            validate_statistics(observation[field])
        if mode == 'observer':
            if any(observation[field]['total'] <= 0
                   or observation[field]['finite'] != observation[field]['total']
                   for field in ('original_self', 'original_mat2', 'output')):
                raise RuntimeError('HEALTHY_REFERENCE_NOT_FINITE')
            tensors = proof.get('tensors', [])
            if (len(tensors) != 3 or any(not t.get('original_address')
                    or t['original_address'] != t.get('run_address') for t in tensors)
                    or observation.get('changed') != [0, 0]):
                raise RuntimeError('READONLY_REFERENCE_REPLACED_INPUT')
            verdict = 'READONLY_COMPLETE_STORAGE_VERIFIED'
        else:
            verdict = Fault39OutputInspector().inspect(native)['verdict']
            if verdict not in {'REAL_OPERATOR_HIT_VERIFIED', 'MATMUL_FINITE_INPUT_NONFINITE_REPRODUCED',
                               'MATMUL_INTERMEDIATE_RANGE_FAILURE_REPRODUCED'}:
                raise RuntimeError('NATIVE_HIT_NOT_VERIFIED')
            if mode == 'lossless' and verdict != 'REAL_OPERATOR_HIT_VERIFIED':
                raise RuntimeError('LOSSLESS_OUTPUT_NOT_FINITE')
            if mode == 'lossless' and observation.get('changed') != [0, 0]:
                raise RuntimeError('LOSSLESS_INPUT_CHANGED')
            if mode == 'inject' and not any(observation.get('changed', ())):
                raise RuntimeError('FAULT_CONTENT_UNCHANGED')
        request.update(native_verdict=verdict, native=native,
                       original_storage_preserved=True, complete_native_output_sha256=output_hash)

    def end(self, model_output_sha256, *, model_output_finite=None):
        if not self.active:
            raise RuntimeError('NO_ACTIVE_SERVING_PHASE')
        row = dict(self.request, model_output_sha256=model_output_sha256)
        try:
            if not is_sha256(model_output_sha256):
                raise RuntimeError('COMPLETE_MODEL_OUTPUT_REQUIRED')
            healthy_phase = self.phase != 'fault' or self.baseline_control
            if type(model_output_finite) is not bool or (healthy_phase and not model_output_finite):
                raise RuntimeError('HEALTHY_COMPLETE_MODEL_OUTPUT_NOT_FINITE')
            row['model_output_finite'] = model_output_finite
            native = self._snapshot()
            if self.baseline is None:
                self.baseline = dict(row)
            if healthy_phase and model_output_sha256 != self.baseline['model_output_sha256']:
                raise RuntimeError('HEALTHY_COMPLETE_MODEL_OUTPUT_CHANGED')
            if row['phase'].startswith('discovery'):
                row['inventory'] = self.observe_inventory()
                # The target must be unique in BOTH complete workloads.
                candidates = native.get('candidates', [])
                if len(candidates) != 1 or candidates[0].get('candidate_count') != 1:
                    raise RuntimeError('AMBIGUOUS_OR_MISSING_DECLARED_BOUNDARY_TARGET')
            elif row['phase'] in self.modes:
                self._check_native(native, row)
            elif (native.get('enabled') is not False or native.get('hits') != self.before.get('hits')
                  or native.get('generation') != self.before.get('generation')):
                raise RuntimeError('DISABLED_OFF_EXECUTED_A_FAULT')
            row['model_output_changed'] = model_output_sha256 != self.baseline['model_output_sha256']
            drained = NativeBackend.disable(self)
            self.drain_status = drained['status']
            if drained['status'] != 'DISABLED_CONFIRMED':
                raise RuntimeError('NATIVE_DRAIN_NOT_CONFIRMED')
            self.rows.append(row)
            self.phase_index += 1
            self.active = False
            if self.phase == 'complete':
                self.recovered = True
                self.recovering = False
        except BaseException as exc:
            self.fail(type(exc).__name__ + ': ' + str(exc))
            raise

    def fail(self, reason):
        self.failure = (self.failure + '; ' if self.failure else '') + str(reason)
        self.active = False
        self.recovering = False
        self.drain_status = NativeBackend.disable(self)['status']

    def request_recovery(self):
        """One fresh OFF-only sequence; failed fault qualification stays failed."""
        if not self.failure or not self.fault_started or self.recovery_attempted or self.active:
            raise RuntimeError('RECOVERY_SEQUENCE_NOT_AVAILABLE')
        self.drain_status = NativeBackend.disable(self)['status']
        if self.drain_status != 'DISABLED_CONFIRMED':
            raise RuntimeError('RECOVERY_DRAIN_PENDING')
        self.recovery_attempted = True
        self.phase_index = PHASES.index('off_observed')
        self.recovering = True

    def clean(self, timeout_ms=10000):
        if self.fault_started and not self.recovered:
            raise RuntimeError('OFF_RECOVERY_REQUIRED_BEFORE_CLEAN')
        return super().clean(timeout_ms)

    def query(self):
        result = super().query()
        result['qualification'] = {
            'status': ('RECOVERED_ATTEMPT_FAILED' if self.recovered and self.failure else
                       ('BASELINE_CONTROL_COMPLETED' if self.baseline_control else 'SCOPED_ATTEMPT_COMPLETED')
                       if self.recovered else
                       'ATTEMPT_FAILED' if self.failure else 'WAITING_FOR_REAL_REQUEST'),
            'next_phase': self.phase, 'recovery_verified': self.recovered, 'failure': self.failure,
            'drain_status': self.drain_status, 'recovering': self.recovering,
            'completed_phases': [row['phase'] for row in self.rows],
            'model_output_changed': any(row.get('model_output_changed') for row in self.rows),
            'actual_request_ids': [row['request_id'] for row in self.rows],
            'baseline_control': self.baseline_control,
            'fault_started': self.fault_started,
        }
        return result
