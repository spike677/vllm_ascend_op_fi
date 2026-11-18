"""Actual Worker callbacks for the existing projection coverage algorithm."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
from pathlib import Path

from fault_injection.coverage_session import CoverageScope, GroupRecoveryHooks, WorkerHealthProof, target_id
from fault_injection.deployments.postwrite_coverage import ProjectionCoverage, _cpu_digest
from fault_injection.deployments.postwrite_evidence import digest
from .serving39_output import Serving39OutputSession
from .serving_context import require


class WorkerProjectionCoverage(ProjectionCoverage):
    def __init__(self, core):
        self.core = core
        self.worker_identity = deepcopy(core.identity)
        self.healthy_after = None
        self.native_receipts, self.retained_receipts = [], {}
        self.reviewed_consumers = {}
        self.parameter_consumption = [deepcopy(observation.parameter_consumption) for observation in core.p._observations]
        require(len(self.parameter_consumption) == 2 and self.parameter_consumption[0] == self.parameter_consumption[1],
                'OUTPUT39_ACTUAL_BORROWED_PARAMETER_CONSUMPTION_CHANGED')
        for row in core.p._observations[0].inventory.calls:
            if row.candidate is not None:
                self.reviewed_consumers[row.key] = [deepcopy(item['reviewed_consumer'])
                    for item in row.candidate.consumer_observations if item.get('reviewed_consumer')]
        sources = [Path(__file__), Path(__file__).with_name('providers')/'fault39_output_group.py',
            Path(__file__).with_name('serving39_output.py'), Path(__file__).with_name('providers')/'fault39_output.py',
            Path(__file__).with_name('gemm_lineage.py')]
        from fault_injection.deployments import postwrite_coverage, postwrite, postwrite_boundaries, postwrite_npu_consumers
        from fault_injection.deployments import postwrite_observed, postwrite_inventory, postwrite_structure
        sources.extend(Path(module.__file__) for module in
            (postwrite_coverage, postwrite, postwrite_boundaries, postwrite_npu_consumers,
             postwrite_observed, postwrite_inventory, postwrite_structure))
        self.source_pins = {str(path.name): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
        if core.binding.simulation:
            environment = digest(dict(explicit_cpu_contract=True, worker=core.identity, native=core.binding.native_sha256))
        else:
            from fault_injection.environment import inspect_environment, environment_fingerprint
            environment = environment_fingerprint(inspect_environment(additional_libraries=('libdcat_postwrite.so',)),
                                                  ('libdcat_postwrite.so',))
        scope = CoverageScope('39.output_proxy', core.input_hash, environment, digest(self.source_pins),
                              'cpu_contract' if core.binding.simulation else 'natural_worker')
        super().__init__(core.p, scope, max_seconds=1200, max_live_leases=core.binding.max_live_leases)
        require(len(self.eligible) <= 256, 'OUTPUT39_COMPLETE_TARGET_PLAN_BUDGET_EXCEEDED')
        # Group scope generation and actual Native per-request generation are
        # separate; both remain under the same Runtime provider owner namespace.
        self.owner, self.generation = core._ticket[0], core._ticket[1]-1
        self.aliases = tuple(core.module_bindings)
        self.raw_eligibility = {tuple(row['key']): row['eligible'] for row in self.discovery_reports[0]['calls']}

    def _identity(self, row):
        original = super()._identity(row)
        worker = self.worker_identity
        return replace(original, worker_id=worker['worker_start_identity'], scope_kind='INITIALIZED_WORKER',
            rank=worker['rank'], logical_device=None if self.core.binding.simulation else worker['logical_device'],
            device_type='cpu' if self.core.binding.simulation else 'npu',
            model_instance_id=worker['model_instance_id'], provider_identity='39.output_proxy:worker-core-group-v1')

    def _check_request(self, args, kwargs):
        self.core._context()
        require(tuple((name, id(module), getattr(module.forward, '__func__', module.forward))
            for name, module in self.provider.model.named_modules(remove_duplicate=False)) == self.aliases,
            'OUTPUT39_GROUP_ACTUAL_ALIAS_OR_METHOD_CHANGED')
        super()._check_request(args, kwargs)

    def _structural_inventory(self, module_calls):
        rows = super()._structural_inventory(module_calls)
        head, processor = self.core.session.head, self.core.session.processor
        for row in rows:
            if row['object_identity'] != id(head):
                continue
            proofs = [item for item in self.parameter_consumption[0]
                if item['module_object_identity'] == id(head) and item['calling_module_identity'] == id(processor)
                and item['parameter_name'] == 'weight' and item['parameter_object_identity'] == id(head.weight)
                and item['original_returned'] and not item['module_forward_observed']]
            if proofs and not tuple(head.children()):
                row.update(structural_noncall=True, reason='ACTUAL_LOGITS_ARGUMENT_PARAMETER_OWNER_NOT_MODULE_FORWARD',
                    actual_parameter_consumption=proofs, actual_logits_binding=self.core.session.boundary_report)
        return rows

    def prepare_coverage(self, trigger, **kwargs):
        if trigger == 'every_match':
            require(self.max_live_leases >= len(self.eligible), 'OUTPUT39_EVERY_MATCH_COMPLETE_LEASE_CAPACITY_REQUIRED')
        super().prepare_coverage(trigger, **kwargs)
        self.healthy_after = None
        self.native_receipts, self.retained_receipts = [], {}

    def projection(self, observation, row):
        if row.key in self.eligible and self.mode in ('lossless', 'fault'):
            import torch
            from fault_injection.deployments.postwrite_npu_consumers import npu_consumer_registration
            for prior in self.reviewed_consumers[row.key]:
                function = getattr(torch.ops.npu, prior['operator'].split('.')[1]).default
                current, reason = npu_consumer_registration(function)
                require(not reason and all(prior.get(name) == value for name, value in (current or {}).items()),
                        'OUTPUT39_GROUP_REVIEWED_CONSUMER_REGISTRATION_CHANGED')
        try:
            super().projection(observation, row)
        finally:
            candidate = row.candidate
            if candidate is not None:
                lease = self.provider._group_leases.get(candidate.key)
                if lease is not None:
                    self.retained_receipts[row.key] = lease
                elif self.mode == 'lossless' and row.key in self.lossless:
                    self.lossless[row.key]['native_lease_receipt'] = deepcopy(self.provider._last_native_lease_receipt)

    def forward_returned(self, observation, output):
        require(observation.parameter_consumption == self.parameter_consumption[0],
                'OUTPUT39_ACTUAL_LOGITS_PARAMETER_CONSUMPTION_CHANGED')
        super().forward_returned(observation, output)
        for row in observation.inventory.calls:
            if self.mode in ('lossless', 'off', 'recovery_retry'):
                require(bool(row.candidate and row.candidate.consumers) == self.raw_eligibility[row.key],
                        'OUTPUT39_GROUP_ELIGIBLE_CONSUMER_DENOMINATOR_CHANGED')

    def _capture_native_receipts(self):
        for key, lease in tuple(self.retained_receipts.items()):
            receipt = lease.receipt() if callable(getattr(lease, 'receipt', None)) else None
            self.native_receipts.append(dict(forward_id=self.forward_id, actual_request_id=self.request_id,
                target_key=key, target_id=target_id(self.identities[key]),
                receipt=deepcopy(receipt), explicit_cpu_mock=receipt is None and self.core.binding.simulation))
            if lease.closed:
                del self.retained_receipts[key]

    def complete_observed_forward(self, output):
        self._capture_native_receipts()
        super().complete_observed_forward(output)

    def fail_observed_forward(self, error):
        self._capture_native_receipts()
        super().fail_observed_forward(error)

    def _hooks(self):
        original = super()._hooks()
        def health(worker, owner, generation):
            self.core._context()
            zero = (not self.provider.live_lease_count and not self.provider._stack
                and not self.provider._lock.locked() and not self.core.session.handles
                and self.core.session.window is None)
            require(self.healthy_after is not None and self.healthy_after['output_sha256'] == self.baseline_hash,
                    'OUTPUT39_GROUP_ACTUAL_HEALTH_PENDING')
            require(zero, 'OUTPUT39_GROUP_HOOK_OR_NATIVE_RECOVERY_PENDING')
            return WorkerHealthProof(worker, owner, generation, digest(self.healthy_after), self.baseline_hash,
                self.healthy_after['output_sha256'], 0, 0, 0, 0, owned_hooks=0)
        return GroupRecoveryHooks(original.stop, original.drain, original.recover, health)

    def finish_stage(self, *, failed=False):
        require(self.session is not None, 'OUTPUT39_GROUP_SESSION_REQUIRED')
        if self.observer.evidence is None:
            independent = dict(discovery=self.discovery_reports, forwards=self.forward_artifacts,
                tickets=[asdict(ticket) for ticket in self.observer.tickets], native_receipts=self.native_receipts)
            self.observer.seal(artifact_sha256=digest(independent), complete=not failed)
        recovery = self.session.disable(timeout_seconds=30)
        require(recovery['complete'], 'OUTPUT39_GROUP_RECOVERY_PENDING')
        if not failed:
            require(self.session.report()['contract_satisfied'], 'OUTPUT39_GROUP_CONTRACT_INCOMPLETE')
            if self.group.trigger == 'once' and self.c2_live_proof is None:
                self.c2_live_proof = self.session.c2_proof()
        return self.report()

    def report(self):
        return dict(schema='demoncat.output39.worker_group.v1', scope=asdict(self.scope),
            worker=deepcopy(self.worker_identity), source_pins=self.source_pins,
            coverage=self.session.report() if self.session is not None else None,
            independent_discovery=self.discovery_reports, discovery=self.discovery_diagnostics,
            all_projection_count=len(self.call_order), eligible_target_count=len(self.eligible),
            trigger=getattr(self, 'prepared_trigger', None), group_generation=self.generation,
            native_generation=self.provider.generation, native_owner=self.provider.owner,
            execution_scope='INDEPENDENT_TARGET_REPLAY' if getattr(self, 'prepared_trigger', None) == 'once'
                else 'SAME_REQUEST_EVERY_MATCH_MULTIPLE_LIVE_LEASES',
            per_target_lossless=[dict(key=key, **value) for key, value in self.lossless.items()],
            execution_artifacts=self.execution_artifacts, forward_artifacts=self.forward_artifacts,
            native_lease_receipts=self.native_receipts, actual_health=self.healthy_after,
            actual_parameter_consumption=self.parameter_consumption,
            serializable_authority=False, qualification_granted=False)


class Serving39OutputGroupSession(Serving39OutputSession):
    def observed_parameter_owners(self):
        return (self.head,)

    def collect(self, *, force=False):
        self._same_worker()

    def _save(self):
        # All full evidence is captured by explicit collection after the true
        # logits return; query and ordinary hook teardown do not duplicate it.
        pass
