"""Bounded C2/C3 dispatch and recovery; never an alternative Native ABI.

An independent observer supplies the denominator before admission. Providers keep
their operator, native uniqueness and lossless gates, and execute the original
exactly once. CPU receipts exercise this contract, not real Worker qualification.
Public qualification remains a separate review of the emitted evidence artifacts.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import threading
import time
import uuid
import weakref

from .core import UnsupportedCall
from .coverage import (BoundaryTargetIdentity, CandidateSet, Eligibility, ObservedCall,
                       Outcome, Phase, QualifiedTargetSet, RecoveryReport, TargetIdentity,
                       TargetLease, _is_sha256)


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def target_id(target):
    return _sha(asdict(target))


def _require(condition, reason):
    if not condition:
        raise UnsupportedCall(reason)


def _positive_int(value):
    return type(value) is int and value > 0


@dataclass(frozen=True)
class CoverageScope:
    mechanism: str
    workload_sha256: str
    environment_sha256: str
    implementation_sha256: str
    evidence_kind: str = 'natural_worker'
    model_weights_sha256: str | None = None
    model_source_sha256: str | None = None

    def __post_init__(self):
        if (not isinstance(self.mechanism, str) or not self.mechanism.strip()
                or self.evidence_kind not in ('natural_worker', 'natural_pretrained_model',
                                             'constructed_npu', 'cpu_contract')
                or not all(_is_sha256(value) for value in (self.workload_sha256,
                    self.environment_sha256, self.implementation_sha256))):
            raise ValueError('Exact workload, environment, implementation and evidence scope required')
        model_hashes = (self.model_weights_sha256, self.model_source_sha256)
        if any(value is not None and not _is_sha256(value) for value in model_hashes):
            raise ValueError('Model weights/source identities must be SHA256 references')
        if self.evidence_kind == 'natural_pretrained_model' and not all(map(_is_sha256, model_hashes)):
            raise ValueError('Natural pretrained model requires actual weights and model source SHA256')


@dataclass(frozen=True)
class SessionBudget:
    max_calls: int
    max_seconds: float
    max_concurrency: int = 1

    def __post_init__(self):
        if (not _positive_int(self.max_calls) or not _positive_int(self.max_concurrency)
                or type(self.max_seconds) not in (int, float)
                or not math.isfinite(self.max_seconds) or self.max_seconds <= 0):
            raise ValueError('Finite positive call, time and concurrency budgets required')


@dataclass(frozen=True)
class ObservationTicket:
    observer_id: str
    sequence: int
    call: ObservedCall
    eligible: bool
    propagation_added: bool = False


class IndependentCallObserver:
    """Append-only, admission-independent, bounded call inventory.

    Install this on the *actual* execution boundary, including calls that the
    injector declines. The adapter owns its evidence authenticity; the session
    verifies provenance fields, exact joins, completeness and missing outcomes.
    """

    def __init__(self, *, source, observer_id, implementation_sha256, workload_sha256, max_calls):
        _require(source in ('independent_native_trace', 'independent_workload_observer'),
                 'INDEPENDENT_OBSERVER_REQUIRED')
        if (not isinstance(observer_id, str) or not observer_id.strip()
                or not _is_sha256(implementation_sha256) or not _is_sha256(workload_sha256)
                or not _positive_int(max_calls)):
            raise ValueError('Observer identity, implementation, workload and bounded storage required')
        self.source, self.observer_id = source, observer_id
        self.implementation_sha256, self.workload_sha256 = implementation_sha256, workload_sha256
        self.max_calls = max_calls
        self._lock = threading.RLock()
        self._tickets, self._keys = [], set()
        self._discovery = self._seal = None
        self._dropped = 0

    def seal_discovery(self, candidates, *, observed_targets, artifact_sha256,
                       expected_workers, observed_workers, complete=True):
        """Bind a separate healthy discovery artifact before fault admission."""
        with self._lock:
            _require(self._discovery is None and not self._tickets, 'DISCOVERY_ALREADY_USED')
            if not isinstance(candidates, CandidateSet) or not _is_sha256(artifact_sha256):
                raise ValueError('Typed discovery inventory and raw evidence SHA required')
            observed_targets = tuple(observed_targets)
            expected_workers, observed_workers = tuple(expected_workers), tuple(observed_workers)
            _require(type(complete) is bool and complete and candidates.discovery_complete,
                     'DISCOVERY_INCOMPLETE')
            _require(bool(candidates.eligible) and len(set(observed_targets)) == len(observed_targets)
                     and set(candidates.eligible).issubset(observed_targets),
                     'DISCOVERY_MISSING_ELIGIBLE_TARGETS')
            known = {row.identity for row in candidates.candidates if row.identity is not None}
            _require(set(observed_targets).issubset(known), 'DISCOVERY_UNCLASSIFIED_TARGETS')
            _require(bool(expected_workers) and len(set(expected_workers)) == len(expected_workers)
                     and len(set(observed_workers)) == len(observed_workers)
                     and set(expected_workers) == set(observed_workers)
                     and {target.worker for target in observed_targets}.issubset(expected_workers),
                     'DISCOVERY_WORKER_DENOMINATOR_INCOMPLETE')
            self._discovery = dict(candidates_sha256=_sha(candidates.report()),
                artifact_sha256=artifact_sha256, observed_target_ids=sorted(map(target_id, observed_targets)),
                expected_workers=expected_workers, observed_workers=observed_workers,
                workload_sha256=self.workload_sha256, observer_id=self.observer_id,
                implementation_sha256=self.implementation_sha256, source=self.source, complete=True)
            return dict(self._discovery)

    def observe(self, call, *, eligible, propagation_added=False):
        with self._lock:
            _require(self._seal is None, 'OBSERVER_ALREADY_SEALED')
            if (not isinstance(call, ObservedCall) or type(eligible) is not bool
                    or type(propagation_added) is not bool):
                raise ValueError('Actual typed call and explicit eligibility required')
            _require(not propagation_added or call.phase == Phase.POST_FIRST_ANOMALY,
                     'PROPAGATION_ADDITION_REQUIRES_POST_ANOMALY_PHASE')
            _require(call.key not in self._keys, 'DUPLICATE_OBSERVED_CALL')
            if len(self._tickets) >= self.max_calls:
                self._dropped += 1
                raise UnsupportedCall('OBSERVER_RESOURCE_BUDGET_EXHAUSTED')
            ticket = ObservationTicket(self.observer_id, len(self._tickets), call, eligible, propagation_added)
            self._keys.add(call.key)
            self._tickets.append(ticket)
            return ticket

    def owns(self, ticket):
        with self._lock:
            return (isinstance(ticket, ObservationTicket) and 0 <= ticket.sequence < len(self._tickets)
                    and self._tickets[ticket.sequence] is ticket)

    @property
    def tickets(self):
        with self._lock:
            return tuple(self._tickets)

    @property
    def discovery(self):
        with self._lock:
            return None if self._discovery is None else dict(self._discovery)

    def seal(self, *, artifact_sha256, complete=True, dropped_calls=0):
        with self._lock:
            _require(self._seal is None, 'OBSERVER_ALREADY_SEALED')
            if (not _is_sha256(artifact_sha256) or type(complete) is not bool
                    or type(dropped_calls) is not int or dropped_calls < 0):
                raise ValueError('Observation artifact, completeness and dropped count required')
            self._seal = dict(artifact_sha256=artifact_sha256,
                complete=complete and dropped_calls == 0 and self._dropped == 0,
                dropped_calls=dropped_calls + self._dropped,
                observed_calls=len(self._tickets), calls_sha256=_sha([asdict(row) for row in self._tickets]))
            return dict(self._seal)

    @property
    def evidence(self):
        with self._lock:
            return None if self._seal is None else dict(self._seal)


@dataclass(frozen=True)
class WorkerHealthProof:
    worker: tuple
    owner: str
    generation: int
    proof_sha256: str
    baseline_output_sha256: str
    off_output_sha256: str
    active_leases: int
    device_bytes: int
    host_bytes: int
    inflight_calls: int
    owned_hooks: int = 0

    @property
    def verified(self):
        return (isinstance(self.worker, tuple) and bool(self.worker)
            and isinstance(self.owner, str) and bool(self.owner) and _positive_int(self.generation)
            and _is_sha256(self.proof_sha256) and _is_sha256(self.baseline_output_sha256)
            and self.baseline_output_sha256 == self.off_output_sha256
            and all(type(n) is int and n == 0 for n in (self.active_leases, self.device_bytes,
                self.host_bytes, self.inflight_calls, self.owned_hooks)))


@dataclass(frozen=True)
class GroupRecoveryHooks:
    stop: object       # stop(lease): disable admission in the provider
    drain: object      # drain(lease, remaining_seconds) -> bool, synchronize real streams/ranks
    recover: object    # recover(lease) -> RecoveryReport, restore/release and target OFF
    health: object     # health(worker, owner, generation) -> WorkerHealthProof

    def __post_init__(self):
        if not all(callable(callback) for callback in (self.stop, self.drain, self.recover, self.health)):
            raise ValueError('Stop, drain, recover and same-process health callbacks required')


@dataclass(frozen=True)
class DispatchDecision:
    session_id: str
    ticket: ObservationTicket
    lease: TargetLease | None
    admitted: bool
    reason: str


@dataclass(frozen=True)
class DispatchReceipt:
    decision: DispatchDecision
    outcome: Outcome
    execution_sha256: str | None
    effect_sha256: str | None
    native_applied: bool
    original_calls: int


_PROOF_ISSUER = object()
_PROOF_LOCK = threading.Lock()
_ISSUED_C2_PROOFS = {}
_RESERVATION_LOCK = threading.Lock()
_RESERVED_TARGETS = {}
_GENERATIONS = {}


@dataclass(frozen=True)
class C2Proof:
    """Live-process gate receipt. Only the exact unmodified issued object gates C3."""
    scope: CoverageScope
    candidates_sha256: str
    target_ids: tuple[str, ...]
    owner: str
    generation: int
    session_id: str
    report_sha256: str
    discovery_sha256: str
    observation_sha256: str
    recovery_sha256: str
    _issuer: object = field(default=None, repr=False, compare=False)

    def report(self):
        return {key: value for key, value in asdict(self).items() if key != '_issuer'}


def _register_c2_proof(proof):
    """Seal issued identity and fields; the dataclass issuer marker is not authority.

    Keep a weak reference so completed sessions do not leak proof objects. The
    cleanup identity check also prevents an old callback from removing a newer
    object if Python reuses an object ID.
    """
    key, digest = id(proof), _sha(_c2_proof_fields(proof))

    def forget(reference):
        with _PROOF_LOCK:
            entry = _ISSUED_C2_PROOFS.get(key)
            if entry is not None and entry[0] is reference:
                del _ISSUED_C2_PROOFS[key]

    reference = weakref.ref(proof, forget)
    with _PROOF_LOCK:
        _ISSUED_C2_PROOFS[key] = (reference, digest)
    return proof


def _issued_c2_proof(proof):
    if type(proof) is not C2Proof or proof._issuer is not _PROOF_ISSUER:
        return False
    with _PROOF_LOCK:
        entry = _ISSUED_C2_PROOFS.get(id(proof))
        if entry is None or entry[0]() is not proof:
            return False
        expected = entry[1]
    try:
        return _sha(_c2_proof_fields(proof)) == expected
    except (TypeError, ValueError):
        return False


def _c2_proof_fields(proof):
    # Do not trust a replaced/subclassed reporting method as the field seal.
    return {key: value for key, value in asdict(proof).items() if key != '_issuer'}


class CoverageSession:
    """Serialize evidence and ownership around a provider's unchanged semantics."""

    def __init__(self, *, group, scope, observer, budget, recovery, c2_proof=None):
        if (not isinstance(group, QualifiedTargetSet) or not isinstance(scope, CoverageScope)
                or not isinstance(observer, IndependentCallObserver) or not isinstance(budget, SessionBudget)
                or not isinstance(recovery, GroupRecoveryHooks)):
            raise ValueError('Typed coverage group, scope, observer, budget and recovery required')
        _require(group.selection == 'all_eligible', 'ALL_ELIGIBLE_SELECTION_REQUIRED')
        _require(all(row.eligibility.status in (Eligibility.SUPPORTED, Eligibility.UNSUPPORTED)
                     for row in group.candidates.candidates), 'UNRESOLVED_CANDIDATE_DENOMINATOR')
        _require(group.concurrency == budget.max_concurrency
                 and budget.max_concurrency <= group.capability.max_concurrency,
                 'CONCURRENCY_BUDGET_MISMATCH')
        _require(observer.workload_sha256 == scope.workload_sha256, 'OBSERVER_WORKLOAD_MISMATCH')
        discovery = observer.discovery
        _require(discovery is not None and discovery['candidates_sha256'] == _sha(group.candidates.report()),
                 'INDEPENDENT_DISCOVERY_REQUIRED')
        _require(not observer.tickets and observer.evidence is None, 'FRESH_RUN_OBSERVER_REQUIRED')
        targets = {lease.target for lease in group.leases}
        workers = {target.worker for target in targets}
        _require(workers == set(discovery['expected_workers']), 'GROUP_WORKER_SCOPE_MISMATCH')
        for target in targets:
            _require(bool(target.process_start_identity), 'PROCESS_LIFETIME_IDENTITY_REQUIRED')
            if scope.evidence_kind == 'natural_pretrained_model':
                _require(isinstance(target, BoundaryTargetIdentity)
                         and target.scope_kind == 'MODEL_LOCAL' and target.device_type == 'npu',
                         'NATURAL_PRETRAINED_NPU_MODEL_REQUIRED')
            if scope.evidence_kind == 'natural_worker' and isinstance(target, BoundaryTargetIdentity):
                _require(target.scope_kind == 'INITIALIZED_WORKER' and target.device_type == 'npu',
                         'NATURAL_NPU_WORKER_REQUIRED')
        owner, generation = group.leases[0].owner, group.leases[0].generation
        if group.trigger == 'every_match':
            _require(_issued_c2_proof(c2_proof),
                     'MATCHING_C2_PROOF_REQUIRED')
            _require(c2_proof.scope == scope and c2_proof.owner == owner and c2_proof.generation < generation
                     and c2_proof.candidates_sha256 == discovery['candidates_sha256']
                     and c2_proof.target_ids == tuple(sorted(map(target_id, targets))),
                     'C2_SCOPE_OR_LIFETIME_MISMATCH')
        self.group, self.scope, self.observer, self.budget, self.recovery = group, scope, observer, budget, recovery
        self.session_id = uuid.uuid4().hex
        self.owner, self.generation = owner, generation
        self._condition = threading.Condition(threading.RLock())
        self._started = time.monotonic()
        self._state = 'ENABLED'
        self._leases = {lease.target: lease for lease in group.leases}
        self._decisions, self._receipts, self._active = {}, {}, {}
        self._attempted = set()
        self._errors, self._recovery_errors = [], []
        self._stopped, self._drained = set(), set()
        self._recovered, self._health = {}, {}
        self._fault_calls = 0
        self._disable_lock = threading.Lock()
        self._reservation_key = (owner, tuple(sorted(map(repr, workers))))
        with _RESERVATION_LOCK:
            _require(not any(target in _RESERVED_TARGETS for target in targets), 'TARGET_LEASE_ALREADY_OWNED')
            _require(generation > _GENERATIONS.get(self._reservation_key, 0), 'STALE_OWNER_GENERATION')
            for target in targets:
                _RESERVED_TARGETS[target] = self.session_id
            _GENERATIONS[self._reservation_key] = generation

    def begin(self, ticket):
        with self._condition:
            _require(self._state != 'DISABLED_CONFIRMED', 'SESSION_ALREADY_RECOVERED')
            _require(self.observer.owns(ticket), 'FOREIGN_OR_FABRICATED_OBSERVATION')
            _require(ticket.sequence not in self._decisions, 'OBSERVED_CALL_ALREADY_DISPATCHED')
            call, lease = ticket.call, self._leases.get(ticket.call.target)
            if lease is not None and (not ticket.eligible or ticket.propagation_added):
                self._errors.append('DISCOVERED_TARGET_CLASSIFICATION_CHANGED')
            if call.phase != Phase.OFF_RECOVERY:
                self._fault_calls += 1
            reason = 'ADMITTED'
            if self._state != 'ENABLED':
                reason = 'SESSION_STOPPED'
            elif call.phase == Phase.OFF_RECOVERY:
                reason = 'OFF_RECOVERY'
            elif self._fault_calls > self.budget.max_calls or time.monotonic() - self._started >= self.budget.max_seconds:
                reason = 'SESSION_BUDGET_EXHAUSTED'
                self._state = 'STOPPING'
            elif not ticket.eligible:
                reason = 'NON_TARGET'
            elif ticket.propagation_added and lease is None:
                reason = 'PROPAGATION_ADDITION'
            elif lease is None:
                reason = 'TARGET_NOT_IN_DISCOVERY'
                self._errors.append(reason)
            elif call.request_ids is None:
                reason = 'UNPROVEN_REQUEST_MAPPING'
                self._errors.append(reason)
            elif self.group.trigger == 'once' and call.target in self._attempted:
                reason = 'TARGET_ALREADY_ATTEMPTED'
            elif sum(decision.admitted for decision in self._active.values()) >= self.budget.max_concurrency:
                reason = 'CONCURRENCY_BUDGET_EXHAUSTED'
            admitted = reason == 'ADMITTED'
            if admitted:
                self._attempted.add(call.target)
            decision = DispatchDecision(self.session_id, ticket, lease, admitted, reason)
            self._decisions[ticket.sequence] = decision
            self._active[ticket.sequence] = decision
            return decision

    def finish(self, decision, outcome, *, execution_sha256=None, effect_sha256=None,
               native_applied=False, original_calls=1):
        with self._condition:
            _require(isinstance(decision, DispatchDecision)
                     and self._active.get(decision.ticket.sequence) is decision,
                     'STALE_OR_ALREADY_FINISHED_DISPATCH')
            if (not isinstance(outcome, Outcome) or type(native_applied) is not bool
                    or type(original_calls) is not int or original_calls < 0):
                raise ValueError('Explicit outcome, actual native application and original call count required')
            if original_calls != 1:
                self._errors.append('ORIGINAL_EXECUTION_COUNT_MISMATCH')
            if native_applied and not decision.admitted:
                self._errors.append('NON_TARGET_OR_STOPPED_REPLACEMENT')
            if outcome == Outcome.HIT:
                if not (decision.admitted and native_applied and original_calls == 1
                        and _is_sha256(execution_sha256) and _is_sha256(effect_sha256)):
                    self._errors.append('HIT_EXECUTION_OR_EFFECT_UNPROVEN')
                    outcome = Outcome.FAILED
            elif native_applied:
                self._errors.append('UNREPORTED_NATIVE_REPLACEMENT')
            receipt = DispatchReceipt(decision, outcome, execution_sha256, effect_sha256,
                                      native_applied, original_calls)
            self._receipts[decision.ticket.sequence] = receipt
            del self._active[decision.ticket.sequence]
            self._condition.notify_all()
            return receipt

    def disable(self, timeout_seconds=30):
        """Stop all owners first; retain all unsettled reservations on any failure.

        Drain hooks must honour their remaining timeout. In-flight callbacks are
        awaited by a condition variable; this never treats a counter reset as a
        device/stream drain. No restore/release begins until every target's drain
        succeeds within the deadline. Failed disable retains every reservation
        and can be retried on the same session.
        """
        if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
                or timeout_seconds < 0):
            raise ValueError('Finite nonnegative drain timeout required')
        with self._disable_lock:
            with self._condition:
                if self._state == 'DISABLED_CONFIRMED':
                    return self.recovery_report()
                self._state = 'STOPPING'
                self._recovery_errors = []
            deadline = time.monotonic() + timeout_seconds
            for lease in self.group.leases:
                if lease in self._stopped:
                    continue
                try:
                    self.recovery.stop(lease)
                    self._stopped.add(lease)
                except Exception as error:
                    self._recovery_errors.append('STOP_FAILED:' + lease.lease_id + ':' + type(error).__name__)
            with self._condition:
                while self._active and time.monotonic() < deadline:
                    self._condition.wait(max(0, deadline - time.monotonic()))
                if self._active or len(self._stopped) != len(self.group.leases):
                    self._state = 'RECOVERY_PENDING'
                    self._recovery_errors.append('INFLIGHT_OR_STOP_PENDING')
                    return self.recovery_report()
            # A partial barrier is not reusable: a later retry revalidates every
            # target while all reservations remain held. After a successful
            # barrier, retries of a failed restore need not drain already freed
            # resources again.
            if not self._recovered:
                self._drained.clear()
            for lease in self.group.leases:
                try:
                    if lease not in self._drained:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise RuntimeError('GROUP_DRAIN_TIMEOUT')
                        if self.recovery.drain(lease, remaining) is not True:
                            raise RuntimeError('DRAIN_UNCONFIRMED')
                        if time.monotonic() > deadline:
                            raise RuntimeError('GROUP_DRAIN_TIMEOUT')
                        self._drained.add(lease)
                except Exception as error:
                    self._recovery_errors.append('DRAIN_FAILED:' + lease.lease_id + ':' + str(error))
            with self._condition:
                if (self._active or len(self._drained) != len(self.group.leases)
                        or self._recovery_errors):
                    self._state = 'RECOVERY_PENDING'
                    self._recovery_errors.append('GROUP_DRAIN_PENDING')
                    return self.recovery_report()
            # The whole group has reached the drain barrier. Only now may any
            # target restore or release storage that another target could share.
            for lease in self.group.leases:
                try:
                    if lease not in self._recovered:
                        receipt = self.recovery.recover(lease)
                        _require(isinstance(receipt, RecoveryReport) and receipt.lease == lease,
                                 'RECOVERY_LEASE_MISMATCH')
                        reference = next(p.reference_output_sha256 for p in self.group.lossless if p.lease == lease)
                        _require(receipt.complete and _is_sha256(receipt.proof_sha256)
                                 and receipt.off_output_sha256 == reference, 'TARGET_OFF_OR_RELEASE_UNPROVEN')
                        self._recovered[lease] = receipt
                except Exception as error:
                    self._recovery_errors.append('RECOVERY_FAILED:' + lease.lease_id + ':' + str(error))
            if len(self._recovered) == len(self.group.leases):
                for worker in dict.fromkeys(lease.target.worker for lease in self.group.leases):
                    try:
                        proof = self.recovery.health(worker, self.owner, self.generation)
                        _require(isinstance(proof, WorkerHealthProof) and proof.verified
                                 and proof.worker == worker and proof.owner == self.owner
                                 and proof.generation == self.generation, 'WORKER_HEALTH_OR_ZERO_RESOURCES_UNPROVEN')
                        self._health[worker] = proof
                    except Exception as error:
                        self._health.pop(worker, None)
                        self._recovery_errors.append('WORKER_RECOVERY_FAILED:' + str(error))
            with self._condition:
                workers = {lease.target.worker for lease in self.group.leases}
                complete = (not self._active and len(self._recovered) == len(self.group.leases)
                            and set(self._health) == workers and not self._recovery_errors)
                self._state = 'DISABLED_CONFIRMED' if complete else 'RECOVERY_PENDING'
                if complete:
                    with _RESERVATION_LOCK:
                        for target in self._leases:
                            if _RESERVED_TARGETS.get(target) == self.session_id:
                                del _RESERVED_TARGETS[target]
            return self.recovery_report()

    def recovery_report(self):
        with self._condition:
            return dict(state=self._state, complete=self._state == 'DISABLED_CONFIRMED',
                owner=self.owner, generation=self.generation, inflight=len(self._active),
                held_leases=[] if self._state == 'DISABLED_CONFIRMED' else [asdict(lease) for lease in self.group.leases],
                pending_drains=[asdict(lease) for lease in self.group.leases if lease not in self._drained],
                pending_leases=[asdict(lease) for lease in self.group.leases if lease not in self._recovered],
                targets=[asdict(row) for row in self._recovered.values()],
                workers=[asdict(row) for row in self._health.values()], errors=list(self._recovery_errors))

    def report(self):
        with self._condition:
            calls = self.observer.tickets
            observation = self.observer.evidence
            missing = [ticket.sequence for ticket in calls if ticket.sequence not in self._receipts]
            eligible = [ticket for ticket in calls if ticket.eligible
                        and (not ticket.propagation_added or ticket.call.target in self._leases)
                        and ticket.call.phase != Phase.OFF_RECOVERY]
            hits = [row for row in self._receipts.values() if row.outcome == Outcome.HIT]
            target_hits = {target: sum(row.decision.ticket.call.target == target for row in hits)
                           for target in self._leases}
            reasons = list(self._errors)
            if observation is None or not observation['complete']:
                reasons.append('INDEPENDENT_OBSERVATION_INCOMPLETE')
            if missing:
                reasons.append('OBSERVED_CALL_OUTCOMES_MISSING')
            if not eligible or any(ticket.call.request_ids is None for ticket in eligible):
                reasons.append('ELIGIBLE_CALL_OR_REQUEST_MAPPING_MISSING')
            if any(ticket.call.target not in self._leases for ticket in eligible):
                reasons.append('UNPLANNED_ELIGIBLE_CALL')
            if any(count != 1 for count in target_hits.values()) and self.group.trigger == 'once':
                reasons.append('ALL_TARGETS_ONCE_NOT_PROVEN')
            if self.group.trigger == 'every_match' and (not all(target_hits.values()) or any(
                    self._receipts.get(ticket.sequence) is None
                    or self._receipts[ticket.sequence].outcome != Outcome.HIT for ticket in eligible)):
                reasons.append('EVERY_MATCH_NOT_PROVEN')
            if any(row.outcome == Outcome.FAILED or row.original_calls != 1 for row in self._receipts.values()):
                reasons.append('CALL_FAILURE')
            if self._state != 'DISABLED_CONFIRMED':
                reasons.append('GROUP_RECOVERY_UNPROVEN')
            propagated = [ticket for ticket in calls if ticket.propagation_added]
            return dict(schema='demoncat.coverage_session.v1', session_id=self.session_id,
                scope=asdict(self.scope), selection=self.group.selection, trigger=self.group.trigger,
                owner=self.owner, generation=self.generation, budget=asdict(self.budget),
                discovery=self.observer.discovery, observation=observation,
                planned_targets=len(self._leases), planned_targets_hit=sum(bool(n) for n in target_hits.values()),
                observed_calls=len(calls), observed_eligible_calls=len(eligible),
                hits=len(hits), missing_outcomes=missing,
                propagation_added_calls=[asdict(ticket) for ticket in propagated],
                calls=[dict(call_id=t.call.call_id, forward_id=t.call.forward_context,
                    request_ids=t.call.request_ids, model_instance=t.call.target.model_instance_id,
                    operator=t.call.target.operator_api, target_id=target_id(t.call.target),
                    rank=t.call.target.rank, device=getattr(t.call.target, 'logical_device', None),
                    worker=t.call.target.worker, owner=self.owner, generation=self.generation,
                    phase=t.call.phase.value, eligible=t.eligible, propagation_added=t.propagation_added,
                    decision=asdict(self._decisions[t.sequence]) if t.sequence in self._decisions else None,
                    receipt=asdict(self._receipts[t.sequence]) if t.sequence in self._receipts else None)
                    for t in calls], recovery=self.recovery_report(),
                contract_satisfied=not reasons, rejection_reasons=sorted(set(reasons)),
                real_worker_evidence=self.scope.evidence_kind == 'natural_worker',
                real_pretrained_model_evidence=self.scope.evidence_kind == 'natural_pretrained_model',
                public_qualification_granted=False)

    def c2_proof(self):
        report = self.report()
        _require(self.group.trigger == 'once' and report['contract_satisfied'], 'C2_CONTRACT_NOT_COMPLETE')
        return _register_c2_proof(C2Proof(self.scope, self.observer.discovery['candidates_sha256'],
            tuple(sorted(map(target_id, self._leases))), self.owner, self.generation, self.session_id,
            _sha(report), self.observer.discovery['artifact_sha256'],
            self.observer.evidence['artifact_sha256'], _sha(report['recovery']), _PROOF_ISSUER))
