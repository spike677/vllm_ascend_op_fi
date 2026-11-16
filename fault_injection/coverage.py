"""Evidence types for the existing adaptation pipeline. No execution authority."""
from dataclasses import asdict, dataclass
from enum import Enum
import re

from .core import UnsupportedCall


def _is_sha256(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


class Eligibility(str, Enum):
    SUPPORTED = 'SUPPORTED'
    UNSUPPORTED = 'UNSUPPORTED'
    UNVERIFIED = 'UNVERIFIED'
    AMBIGUOUS = 'AMBIGUOUS'
    NOT_OBSERVED = 'NOT_OBSERVED'
    RESOURCE_LIMITED = 'RESOURCE_LIMITED'


@dataclass(frozen=True)
class EligibilityResult:
    status: Eligibility
    reasons: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.status, Eligibility) or (
            self.status != Eligibility.SUPPORTED and not self.reasons
        ):
            raise ValueError('Ineligible or unverified candidates require a reason')


@dataclass(frozen=True)
class TargetIdentity:
    """Identity within a live Worker and model load. Cache addresses are insufficient."""
    process_id: int
    worker_id: str
    rank: int
    logical_device: int
    model_instance_id: str
    model_object_identity: int
    module_object_identity: int
    module_roles: tuple[str, ...]
    operator_api: str
    execution_boundary: str
    executor_identity: str
    cache_lifecycle: str
    tensor_signature: str
    native_build_fingerprint: str
    provider_identity: str | None = None
    provider_proof: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if any(type(n) is not int or n <= 0 for n in (
            self.process_id, self.model_object_identity, self.module_object_identity
        )) or any(type(n) is not int or n < 0 for n in (self.rank, self.logical_device)):
            raise ValueError('Incomplete target process, module, rank or device')
        for identity in (self.worker_id, self.model_instance_id, self.operator_api, self.execution_boundary,
                         self.executor_identity, self.cache_lifecycle, self.tensor_signature):
            if not isinstance(identity, str) or not identity.strip():
                raise ValueError('Incomplete Worker, model, operator or cache lifecycle identity')
        if not isinstance(self.module_roles, tuple) or any(
            not isinstance(role, str) or not role for role in self.module_roles
        ) or tuple(sorted(set(self.module_roles))) != self.module_roles:
            raise ValueError('Expected distinct, sorted structural roles')
        if not _is_sha256(self.native_build_fingerprint):
            raise ValueError('Expected native build fingerprint')
        if self.provider_identity is not None and (
            not isinstance(self.provider_identity, str) or not self.provider_identity.strip()
        ):
            raise ValueError('Invalid provider identity')
        fields = self.provider_proof
        if not isinstance(fields, tuple) or any(
            not isinstance(item, tuple) or len(item) != 2
            or any(not isinstance(value, str) or not value.strip() for value in item)
            for item in fields
        ):
            raise ValueError('Expected hashable provider proof fields')
        keys = tuple(key for key, _ in fields)
        if keys != tuple(sorted(set(keys))) or (fields and self.provider_identity is None):
            raise ValueError('Expected distinct ordered fields bound to a provider')

    @property
    def worker(self):
        return self.process_id, self.worker_id, self.rank, self.logical_device


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    eligibility: EligibilityResult
    identity: TargetIdentity | None = None

    def __post_init__(self):
        if self.eligibility.status == Eligibility.SUPPORTED and self.identity is None:
            raise ValueError('Eligible candidates require a complete target identity')


@dataclass(frozen=True)
class CandidateSet:
    candidates: tuple[Candidate, ...]
    discovery_complete: bool

    def __post_init__(self):
        if (type(self.discovery_complete) is not bool or not isinstance(self.candidates, tuple)
                or not all(isinstance(row, Candidate) for row in self.candidates)):
            raise ValueError('Expected an immutable candidate set with known observation completeness')
        names = [row.candidate_id for row in self.candidates]
        identities = [row.identity for row in self.candidates if row.identity is not None]
        if any(not name for name in names) or len(set(names)) != len(names) or len(set(identities)) != len(identities):
            raise ValueError('Candidate set contains absent or duplicate identities')

    @property
    def eligible(self):
        return tuple(row.identity for row in self.candidates if row.eligibility.status == Eligibility.SUPPORTED)

    def select(self, policy='unique', *, target=None, priority=None):
        """Make an offline selection. Lossless and native admission stay with the session."""
        eligible = self.eligible
        if policy == 'explicit':
            if target not in eligible:
                raise UnsupportedCall('Explicit target is absent or ineligible')
            return (target,)
        if not self.discovery_complete:
            raise UnsupportedCall('Candidate discovery is incomplete')
        if policy == 'all_eligible':
            return eligible
        if policy == 'max_k_min_m_unique':
            # K and M are interpreted by the operator adapter, including fused layouts.
            if priority is None or set(priority) != set(eligible):
                raise UnsupportedCall('Expected provider-supplied K and M for each eligible target')
            if eligible:
                best = min((-k, m) for k, m in priority.values())
                eligible = tuple(row for row in eligible if (-priority[row][0], priority[row][1]) == best)
        elif policy != 'unique':
            raise ValueError('Unknown selection policy')
        if len(eligible) != 1:
            raise UnsupportedCall('Expected one eligible target')
        return eligible

    def report(self):
        counts = {status.value: 0 for status in Eligibility}
        for candidate in self.candidates:
            counts[candidate.eligibility.status.value] += 1
        return {**asdict(self), 'eligible': len(self.eligible),
                'discovered': len(self.candidates) - counts[Eligibility.NOT_OBSERVED.value],
                'eligibility_counts': counts, 'execution_authorized': False}


@dataclass(frozen=True)
class CoverageCapability:
    multi_target: bool = False
    every_match: bool = False
    max_concurrency: int = 0
    max_resources: int | None = None  # Maximum device bytes, None when unqualified.
    cross_worker: bool = False
    recovery_scope: str = 'unqualified'
    stop_policy: str = 'not_qualified'
    single_target: bool = False
    rank_isolation: bool = False
    group_recovery: bool = False
    max_host_bytes: int | None = None
    lossless_modes: tuple[str, ...] = ('input',)

    def __post_init__(self):
        switches = (self.single_target, self.multi_target, self.every_match,
                    self.cross_worker, self.rank_isolation, self.group_recovery)
        if any(type(switch) is not bool for switch in switches):
            raise ValueError('Coverage capabilities must be explicitly boolean')
        if type(self.max_concurrency) is not int or self.max_concurrency < 0:
            raise ValueError('Expected a nonnegative concurrency budget')
        limits = (self.max_resources, self.max_host_bytes)
        if any(n is not None and (type(n) is not int or n < 0) for n in limits):
            raise ValueError('Expected nonnegative byte budgets or None')
        modes = self.lossless_modes
        if not isinstance(modes, tuple) or not modes or any(
            not isinstance(mode, str) or not mode.strip() for mode in modes
        ) or len(set(modes)) != len(modes):
            raise ValueError('Expected distinct lossless modes')


class Phase(str, Enum):
    PRE_FAULT_PROPAGATION = 'PRE_FAULT_PROPAGATION'
    POST_FIRST_ANOMALY = 'POST_FIRST_ANOMALY'
    OFF_RECOVERY = 'OFF_RECOVERY'


def coverage_phase(order, *, first_anomaly=None, off_from=None):
    """Use independent workload order. The first anomalous call ends the pre-propagation window."""
    if off_from is not None and order >= off_from:
        return Phase.OFF_RECOVERY
    if first_anomaly is not None and order > first_anomaly:
        return Phase.POST_FIRST_ANOMALY
    return Phase.PRE_FAULT_PROPAGATION


@dataclass(frozen=True)
class ObservedCall:
    target: TargetIdentity
    call_id: str
    forward_context: str
    request_ids: tuple[str, ...] | None
    phase: Phase

    def __post_init__(self):
        if not self.call_id or not self.forward_context or not isinstance(self.phase, Phase):
            raise ValueError('Observed call requires a call ID, forward context and phase')
        if self.request_ids is not None and (not self.request_ids or any(not key for key in self.request_ids)
                                             or len(set(self.request_ids)) != len(self.request_ids)):
            raise ValueError('Request mapping must be proven or absent')

    @property
    def request_mapping(self):
        return 'UNPROVEN_REQUEST_MAPPING' if self.request_ids is None else 'PROVEN'

    @property
    def key(self):
        return self.target.worker, self.call_id


@dataclass(frozen=True)
class IndependentObservation:
    """Eligible calls matched by an independent observer, before resource or admission outcomes."""
    source: str
    calls: tuple[ObservedCall, ...]
    complete: bool

    def __post_init__(self):
        if self.source not in ('independent_native_trace', 'independent_workload_observer'):
            raise ValueError('Coverage denominator requires independent observation')
        if (type(self.complete) is not bool or not isinstance(self.calls, tuple)
                or not all(isinstance(call, ObservedCall) for call in self.calls)):
            raise ValueError('Expected immutable calls and known observation completeness')
        if len({call.key for call in self.calls}) != len(self.calls):
            raise ValueError('Duplicate independently observed call')


@dataclass(frozen=True)
class TargetLease:
    target: TargetIdentity
    owner: str
    generation: int
    lease_id: str
    device_bytes: int
    host_bytes: int = 0

    def __post_init__(self):
        if (any(not isinstance(value, str) or not value.strip() for value in (self.owner, self.lease_id))
                or type(self.generation) is not int or self.generation <= 0
                or any(type(n) is not int or n < 0 for n in (self.device_bytes, self.host_bytes))):
            raise ValueError('Invalid target owner, generation or resource reservation')


@dataclass(frozen=True)
class RecoveryReport:
    lease: TargetLease
    drained: bool
    released: bool
    off_exact: bool | None
    pending_reason: str | None = None
    proof_sha256: str | None = None
    off_output_sha256: str | None = None

    @property
    def complete(self):
        return (self.drained is True and self.released is True and self.off_exact is True
                and self.pending_reason is None)


@dataclass(frozen=True)
class TargetLosslessProof:
    """Reviewed lossless receipt for a particular target, lease and mode."""
    lease: TargetLease
    mode: str
    proof_sha256: str
    input_sha256_before: tuple[str, ...]
    input_sha256_after: tuple[str, ...]
    reference_output_sha256: str
    output_sha256: str
    original_calls: int
    temporary_readback_verified: bool
    resources_released: bool

    @property
    def verified(self):
        inputs = self.input_sha256_before
        return (
            type(self.original_calls) is int and self.original_calls == 1
            and self.temporary_readback_verified is True and self.resources_released is True
            and _is_sha256(self.proof_sha256)
            and isinstance(inputs, tuple) and bool(inputs) and inputs == self.input_sha256_after
            and all(_is_sha256(value) for value in inputs)
            and _is_sha256(self.reference_output_sha256)
            and self.reference_output_sha256 == self.output_sha256
        )


@dataclass(frozen=True)
class GroupRecoveryReport:
    owner: str
    generation: int
    reports: tuple[RecoveryReport, ...]
    pending_leases: tuple[TargetLease, ...]

    @property
    def complete(self):
        return not self.pending_leases


@dataclass(frozen=True)
class QualifiedTargetSet:
    """Offline qualification of a bounded set; it cannot authorize a native call."""
    candidates: CandidateSet
    leases: tuple[TargetLease, ...]
    lossless: tuple[TargetLosslessProof, ...]
    capability: CoverageCapability
    concurrency: int = 1
    selection: str = 'explicit'
    trigger: str = 'once'
    held_leases: tuple[TargetLease, ...] = ()

    def __post_init__(self):
        for rows, row_type in (
            (self.leases, TargetLease), (self.held_leases, TargetLease),
            (self.lossless, TargetLosslessProof),
        ):
            if not isinstance(rows, tuple) or not all(isinstance(row, row_type) for row in rows):
                raise ValueError('Multi-target qualification requires immutable typed evidence')
        cap = self.capability
        targets = {lease.target for lease in self.leases}
        eligible = set(self.candidates.eligible)
        if (len(self.leases) < 2 or len(targets) != len(self.leases)
                or not self.candidates.discovery_complete or targets - eligible):
            raise UnsupportedCall('A target set needs distinct eligible targets and complete discovery')
        if self.selection not in ('explicit', 'all_eligible') or (
            self.selection == 'all_eligible' and targets != eligible
        ):
            raise UnsupportedCall('Selection does not cover the requested eligible target set')
        if (not cap.single_target or not cap.multi_target or not cap.group_recovery
                or cap.recovery_scope != 'target_set'):
            raise UnsupportedCall('Provider requires single-target, multi-target and group recovery qualification')
        if self.trigger not in ('once', 'every_match') or (self.trigger == 'every_match' and not cap.every_match):
            raise UnsupportedCall('Trigger requires provider qualification')
        if type(self.concurrency) is not int or not 1 <= self.concurrency <= min(
            cap.max_concurrency, len(self.leases)
        ):
            raise UnsupportedCall('Requested concurrency exceeds provider budget')
        if len({target.worker for target in targets}) > 1 and not (cap.cross_worker and cap.rank_isolation):
            raise UnsupportedCall('Cross-worker target sets require rank isolation')
        providers = {target.provider_identity for target in targets}
        if None in providers or len(providers) != 1:
            raise UnsupportedCall('Target set requires one proven provider identity')
        owners = {(lease.owner, lease.generation) for lease in self.leases}
        lease_ids = {lease.lease_id for lease in self.leases}
        if (len(owners) != 1 or len(lease_ids) != len(self.leases) or any(
            lease.target in targets or lease.lease_id in lease_ids for lease in self.held_leases
        )):
            raise UnsupportedCall('Conflicting owner, generation or target lease')
        reservations = self.leases + self.held_leases
        if sum(lease.device_bytes for lease in reservations) > (cap.max_resources or 0) or sum(
            lease.host_bytes for lease in reservations
        ) > (cap.max_host_bytes or 0):
            raise UnsupportedCall('Target set exceeds provider resource budget')
        expected = {(lease, mode) for lease in self.leases for mode in cap.lossless_modes}
        actual = {(proof.lease, proof.mode) for proof in self.lossless}
        if actual != expected or len(actual) != len(self.lossless) or not all(
            proof.verified for proof in self.lossless
        ):
            raise UnsupportedCall('Lossless evidence is missing, stale or incomplete for the target set')
        references = {}
        for proof in self.lossless:
            reference = (proof.input_sha256_before, proof.reference_output_sha256)
            if references.setdefault(proof.lease, reference) != reference:
                raise UnsupportedCall('Lossless modes do not share an exact reference for the target')

    @property
    def execution_authorized(self):
        return False

    def report(self):
        return dict(selection=self.selection, trigger=self.trigger, concurrency=self.concurrency,
                    leases=[asdict(lease) for lease in self.leases], execution_authorized=False)

    def reconcile_recovery(self, reports):
        """Retain every unsettled lease, even for targets the run never reached."""
        reports = tuple(reports)
        by_lease = {row.lease: row for row in reports}
        if len(by_lease) != len(reports) or set(by_lease) - set(self.leases):
            raise ValueError('Recovery receipt has a duplicate or unrelated lease')
        references = {proof.lease: proof.reference_output_sha256 for proof in self.lossless}
        settled = {
            row.lease for row in reports if row.complete and _is_sha256(row.proof_sha256)
            and row.off_output_sha256 == references[row.lease]
        }
        owner = self.leases[0]
        return GroupRecoveryReport(
            owner.owner, owner.generation, reports,
            tuple(lease for lease in self.leases if lease not in settled),
        )


class Outcome(str, Enum):
    HIT = 'HIT'
    PASSTHROUGH = 'PASSTHROUGH'
    FAILED = 'FAILED'
    REJECTED = 'REJECTED'
    RESOURCE_LIMITED = 'RESOURCE_LIMITED'
    STOPPED = 'STOPPED'


@dataclass(frozen=True)
class CallOutcome:
    call: ObservedCall
    lease: TargetLease
    outcome: Outcome
    proof_sha256: str | None = None

    def __post_init__(self):
        if not isinstance(self.outcome, Outcome) or self.call.target != self.lease.target:
            raise ValueError('Outcome must be bound to its target lease')
        if self.outcome == Outcome.HIT and not _is_sha256(self.proof_sha256):
            raise ValueError('A hit requires reviewed execution evidence')


@dataclass(frozen=True)
class CoverageReport:
    planned_targets: int
    discovered_targets: int
    eligible_targets: int
    selected_targets: int
    planned_targets_hit: int
    planned_targets_unobserved: int | None
    observed_eligible_calls: int | None
    observed_calls_hit: int | None
    observation_complete: bool
    coverage_by_phase: dict
    outcome_counts: dict
    target_outcomes: tuple[dict, ...]
    off_hits: int
    uncorrelated_outcomes: int
    request_mapping_unproven: int | None

    @classmethod
    def reconcile(cls, planned, candidates, leases, observation, outcomes):
        """Join attempt receipts to current leases and actual call identity, never shape."""
        planned, leases, outcomes = tuple(planned), tuple(leases), tuple(outcomes)
        plan = set(planned)
        by_target = {lease.target: lease for lease in leases}
        if len(plan) != len(planned) or len(by_target) != len(leases) or set(by_target) - plan:
            raise ValueError('Expected distinct selected targets and leases within the plan')
        if set(by_target) - set(candidates.eligible):
            raise ValueError('Selected targets must be eligible')
        by_call = {row.call.key: row for row in outcomes}
        if len(by_call) != len(outcomes):
            raise ValueError('Duplicate execution outcome')
        correlated = set()
        hits = set()
        phase_counts = {}
        observed = None if observation is None else observation.calls
        observed_by_call = {call.key: call for call in observed or ()}
        bound = {
            key: row for key, row in by_call.items()
            if row.lease == by_target.get(row.call.target)
            and (observation is None or not observation.complete or key in observed_by_call)
            and observed_by_call.get(key, row.call) == row.call
        }
        target_counts = {lease: {kind.value: 0 for kind in Outcome} for lease in leases}
        for row in bound.values():
            if row.call.phase != Phase.OFF_RECOVERY:
                target_counts[row.lease][row.outcome.value] += 1
                if row.outcome == Outcome.HIT:
                    hits.add(row.lease.target)
        for phase in Phase:
            calls = None if observed is None else tuple(call for call in observed if call.phase == phase)
            counts = None if calls is None else {'observed': len(calls), 'hit': 0, 'outcome_missing': 0}
            for call in calls or ():
                row = bound.get(call.key)
                if row is None:
                    counts['outcome_missing'] += 1
                    continue
                correlated.add(call.key)
                if row.outcome == Outcome.HIT:
                    counts['hit'] += 1
            phase_counts[phase.value] = counts
        discovered = [row for row in candidates.candidates if row.eligibility.status != Eligibility.NOT_OBSERVED]
        return cls(
            len(planned), len(discovered), len(candidates.eligible), len(leases), len(hits & plan),
            None if observed is None else len(plan - {
                call.target for call in observed if call.phase != Phase.OFF_RECOVERY
            }),
            None if observed is None else sum(call.phase != Phase.OFF_RECOVERY for call in observed),
            None if observed is None else sum(row['hit'] for phase, row in phase_counts.items()
                                             if phase != Phase.OFF_RECOVERY.value),
            observation is not None and observation.complete, phase_counts,
            {kind.value: sum(row.outcome == kind and row.call.phase != Phase.OFF_RECOVERY
                             for row in bound.values())
             for kind in Outcome},
            tuple({'lease': asdict(lease), 'counts_in_fault_window': counts}
                  for lease, counts in target_counts.items()),
            sum(row.outcome == Outcome.HIT and row.call.phase == Phase.OFF_RECOVERY for row in bound.values()),
            len(by_call) - len(correlated),
            None if observed is None else sum(call.request_ids is None and call.phase != Phase.OFF_RECOVERY
                                             for call in observed),
        )
