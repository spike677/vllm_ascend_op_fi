"""Public Worker lease for the existing single-target postwrite C1 mechanism."""
from copy import deepcopy
import hashlib
import os
import threading

from ._scope import Scope, ProviderError, reject_unknown
from .. import identity as identity_reader
from ..serving39_output import Output39ServingBinding, Serving39OutputSession, _ATTRIBUTE, read_native_binding


def require(value, reason):
    if not value:
        raise ProviderError(reason)


_PHASES = ('observe', 'observe_repeat', 'baseline', 'fault', 'off', 'health')


class Fault39OutputServingProvider(Scope):
    execution_boundary = 'native'

    def __init__(self, worker, model, identity):
        super().__init__(worker, model, identity)
        self.runner, self.thread = worker.model_runner, threading.get_ident()
        self.binding = getattr(worker, _ATTRIBUTE, None)
        # Resolver admission is read-only and precedes Runtime ownership. A
        # missing Native deployment cannot supply the ABI readback needed to
        # release even an otherwise empty Runtime lease.
        require(type(self.binding) is Output39ServingBinding, 'REVIEWED_OUTPUT39_SERVING_BINDING_REQUIRED')
        self.session = self.p = None
        self.binding_reserved = False
        self.checked = self.attempted = self.complete = self.recovered = False
        self.recovering = False
        self.stopped = True
        self.index = 0
        self.failure = None
        self.failures, self.rows, self.observations = [], [], []
        self.requests = set()
        self.actual_identity = self.input_hash = None
        self.selection = None
        self._context()
        read_native_binding(self.binding)

    @property
    def phase(self):
        return 'off' if self.recovering == 'off' else 'health' if self.recovering == 'health' else (
            _PHASES[self.index] if self.index < len(_PHASES) else 'complete')

    @property
    def native_phase(self):
        return 'observe' if self.phase in ('observe', 'observe_repeat') else self.phase

    def _context(self):
        value, device = self.identity, getattr(self.worker, 'device', None)
        visible, source = identity_reader._visible_devices()
        require(os.getpid() == value['pid'] and threading.get_ident() == self.thread
            and self.worker.model_runner is self.runner and self.runner.model is self.model
            and self.worker.rank == value['rank'] and getattr(device, 'type', None) == 'npu'
            and getattr(device, 'index', None) == value['configured_logical_device']
            and identity_reader._current_device() == value['logical_device']
            and value['observed_logical_device'] == value['logical_device']
            and visible == value['visible_devices'] and source == value['visible_devices_source'],
            'OUTPUT39_HELD_WORKER_CONTEXT_CHANGED')
        if self.binding is not None:
            require(getattr(self.worker, _ATTRIBUTE, None) is self.binding
                and self.binding.worker is self.worker and self.binding.runner is self.runner
                and self.binding.model is self.model and self.binding.pid == os.getpid()
                and self.binding.thread == self.thread, 'OUTPUT39_DEPLOYMENT_BINDING_CHANGED')
            if self.binding_reserved or self.binding.leases:
                require(self.binding_reserved and self.binding.leases == {self._ticket},
                        'OUTPUT39_DEPLOYMENT_OWNER_CHANGED')
        if self.p is not None:
            require(self.p.owner == self.native_owner and self.p.model is self.model
                and self.p.generation == self.native_generation
                and self.p.transport is self.binding.transport, 'OUTPUT39_NATIVE_OWNER_CHANGED')

    def dispatch(self, action, settings, *, owner, generation):
        if action == 'prepare':
            self.bind(owner, generation)
        self.verify(owner, generation)
        reject_unknown(settings, ('request_id',) if action == 'advance' else ())
        self._context()
        if action == 'prepare':
            require(not self.binding.leases, 'OUTPUT39_DEPLOYMENT_ALREADY_OWNED')
            read_native_binding(self.binding)
        elif action == 'register':
            require(self.binding is not None and not self.binding.leases, 'OUTPUT39_DEPLOYMENT_ALREADY_OWNED')
            self.binding.leases.add(self._ticket)
            self.binding_reserved = True
            self._registered = True
            from fault_injection.deployments.postwrite import Fault39OutputProxyProvider
            self.p = Fault39OutputProxyProvider(self.model, self.binding.transport)
            self.native_owner = int.from_bytes(hashlib.sha256(owner.encode()).digest()[:8], 'little') % (2**63-1) + 1
            self.p.owner, self.p.generation = self.native_owner, generation
            self.native_generation = generation
            self.module_bindings = tuple((name, id(module), getattr(module.forward, '__func__', module.forward))
                for name, module in self.model.named_modules(remove_duplicate=False))
            self.session = Serving39OutputSession(self)
        elif action == 'check':
            require(self._registered and self.session is not None, 'OUTPUT39_REGISTER_REQUIRED')
            require(not self.p.live_lease_count, 'OUTPUT39_NATIVE_RECOVERY_REQUIRED')
            self.checked = True
        elif action in ('enable', 'execute'):
            require(self.checked and not self.attempted, 'OUTPUT39_ONE_ATTEMPT_REQUIRED')
            self.attempted, self.stopped = True, False
            self.session.install()
        elif action == 'advance':
            require(self.attempted and not self.stopped and (not self.failure or self.recovering),
                    'OUTPUT39_FAILED_ATTEMPT_REQUIRES_RECOVERY')
            self.session.expect(settings.get('request_id'))
        elif action == 'recover':
            require(self.session is not None, 'OUTPUT39_REGISTER_REQUIRED')
            self.session.close_window()
            self.p.recover()
            if self.p._fault_attempted and not self.recovered:
                self.recovering, self.stopped = 'off', False
            else:
                self._cleanup()
            self.failure = None
        elif action in ('finish', 'disable', 'abort', 'clean', 'unregister'):
            if action == 'finish':
                require(self.complete and self.recovered and not self.failure, 'OUTPUT39_REAL_OFF_AND_HEALTH_REQUIRED')
            self._cleanup()
            if action in ('clean', 'unregister'):
                self._registered = False
            if action == 'unregister' and self.binding_reserved:
                # clean is inert but still owned. Only unregister releases the
                # exact reservation checked above; absence/foreign is not idle.
                self.binding.leases.remove(self._ticket)
                self.binding_reserved = False
        else:
            raise ProviderError('UNSUPPORTED_PROVIDER_ACTION')
        return self.query(owner=owner, generation=generation)

    def begin_actual(self, args, kwargs):
        import torch
        from fault_injection.deployments.postwrite import _hash_inputs
        self._context()
        require(not self.model.training and torch.is_inference_mode_enabled(), 'OUTPUT39_ACTUAL_INFERENCE_REQUEST_REQUIRED')
        require(tuple((name, id(module), getattr(module.forward, '__func__', module.forward))
            for name, module in self.model.named_modules(remove_duplicate=False)) == self.module_bindings,
            'OUTPUT39_ACTUAL_MODULE_OR_FORWARD_CHANGED')
        require(not self.failure and not self.stopped, 'OUTPUT39_FAILED_ATTEMPT_CANNOT_CONTINUE')
        request = self.session.current['actual_internal_request_id']
        require(request not in self.requests, 'OUTPUT39_REUSED_ACTUAL_REQUEST')
        self.requests.add(request)
        if self.native_phase == 'observe':
            identity, input_hash = self.p._identity(), _hash_inputs(args, kwargs)
            if self.actual_identity is None:
                self.actual_identity, self.input_hash = identity, input_hash
                self.p._qualification_diagnostics = dict(status='IN_PROGRESS', observations=[], guards_relaxed=False)
            require(identity == self.actual_identity and input_hash == self.input_hash,
                    'OUTPUT39_HEALTHY_MODEL_OR_INPUT_CHANGED')
        elif self.phase != 'health':
            self.p._begin_run(self.native_phase, args, kwargs)
            self.native_generation = self.p.generation
        else:
            require(self.p._identity() == self.actual_identity and _hash_inputs(args, kwargs) == self.input_hash
                    and not self.p.live_lease_count, 'OUTPUT39_HEALTH_MODEL_OR_INPUT_CHANGED')

    def end_actual(self, output, result, request):
        from fault_injection.deployments.postwrite import _equal
        self._context()
        phase = self.phase
        report = None
        inventory = None
        if phase == 'health':
            require(_equal(output, self.p._baseline_output) and not self.p.live_lease_count,
                    'OUTPUT39_SAME_WORKER_HEALTH_CHANGED')
            self.recovered, self.stopped = True, True
        elif self.native_phase == 'observe':
            copied, candidates, report = result
            observer = self.p._last_observer
            self.observations.append((copied, candidates))
            self.p._observations.append(observer)
            self.p._qualification_diagnostics['observations'].append(report)
            if len(self.observations) == 2:
                left, right = self.p._observations
                require(left.inventory.operator_counts == right.inventory.operator_counts
                    and left.inventory.module_calls == right.inventory.module_calls
                    and tuple(row.key for row in left.inventory.calls) == tuple(row.key for row in right.inventory.calls),
                    'OUTPUT39_COMPLETE_HEALTHY_DENOMINATOR_CHANGED')
                self.selection = self.p._qualify_observed_pair(self.actual_identity, self.input_hash,
                    *self.observations[0], *self.observations[1])
        else:
            _, _, report = result
            _, report = self.p._complete_run(self.native_phase, output, report)
            require(report['recovery_confirmed'] and not self.p.live_lease_count and report['consumers'],
                    'OUTPUT39_ACTUAL_CONSUMER_OR_NATIVE_RECOVERY_UNPROVEN')
            intervention = report['intervention']
            if phase == 'fault':
                require(report['new_fault_hits'] == 1 and intervention
                    and intervention['changed_elements'] == intervention['selected_elements'] > 0
                    and intervention['readback_exact'], 'OUTPUT39_ACTUAL_NATIVE_EFFECT_UNPROVEN')
            elif phase == 'baseline':
                require(intervention and intervention['changed_elements'] == 0
                    and intervention['selected_elements'] > 0 and report['exact_baseline_output'],
                    'OUTPUT39_ACTUAL_LOSSLESS_UNPROVEN')
            elif phase == 'off':
                require(intervention is None and report['new_fault_hits'] == 0 and report['exact_baseline_output'],
                    'OUTPUT39_ACTUAL_OFF_UNPROVEN')
        if self.p._last_observer is not None and phase != 'health':
            inventory = self.p._last_observer.inventory.report()
        self.rows.append(dict(phase=phase, actual_internal_request_id=request['actual_internal_request_id'],
            snapshot=deepcopy(request['snapshot']), output_sha256=request['model_output_sha256'],
            output_finite=request['model_output_finite'], report=report, complete_inventory=inventory,
            native_owner=self.native_owner, native_generation=self.p.generation,
            native_leases=self.p.live_lease_count))
        if self.recovering:
            self.recovering = 'health' if phase == 'off' else False
        else:
            self.index += 1
            self.complete = self.index == len(_PHASES)

    def fail(self, error):
        if self.failure is None:
            self.failure = str(error)
        window = self.session.window if self.session is not None else None
        self.failures.append(dict(phase=self.phase, error=str(error),
            partial_inventory=deepcopy(window.observer.inventory.report()) if window is not None else None,
            intervention=deepcopy(self.p._last_intervention) if self.p is not None else None,
            native_lease_receipt=deepcopy(self.p._last_native_lease_receipt) if self.p is not None else None))

    def _cleanup(self):
        if self.session is not None:
            self.session.close_window()
        if self.p is not None:
            self.p.recover()
            require(not self.p._fault_attempted or self.recovered, 'OUTPUT39_REAL_OFF_AND_HEALTH_REQUIRED')
        if self.session is not None:
            self.session.close()
        if self.p is not None:
            self.p._observations.clear()
            self.p._last_observer = None
            self.observations.clear()
        self.stopped = True

    def evidence(self):
        return dict(mechanism='39.output_proxy', scope='INITIALIZED_WORKER_SINGLE_SITE_C1_FIXED_FRESH_PREFILL',
            evidence_kind='cpu_contract' if self.binding is not None and self.binding.simulation else 'natural_worker',
            boundary='actual_public_projection_return_then_value_consumer', native_out_of_bounds_claim=False,
            selection=deepcopy(self.selection), complete=self.complete, recovered=self.recovered,
            module_aliases=[dict(path=name, object_identity=identity, forward_identity=id(function))
                for name, identity, function in getattr(self, 'module_bindings', ())],
            failures=deepcopy(self.failures), actual_requests=deepcopy(self.rows),
            last_native_lease_receipt=deepcopy(self.p._last_native_lease_receipt) if self.p is not None else None,
            worker_identity=deepcopy(self.identity), rpc_owner=self._ticket[0], rpc_generation=self._ticket[1],
            c1_evidence_complete=bool(self.complete and self.recovered and not self.failures),
            model_local_qualification_inherited=False, qualification_granted=False, c2=False, c3=False,
            E5='NOT_ASSESSED_SEPARATE_RAW_STATISTICS_REQUIRED')

    def query(self, *, owner, generation):
        self.verify(owner, generation)
        self._context()
        native = read_native_binding(self.binding) if self.binding is not None else None
        serving = self.session.readback() if self.session is not None else None
        leases = []
        if self.p is not None:
            for lease in self.p._live_leases():
                require(lease.owner == self.native_owner and lease.generation == self.p.generation,
                        'OUTPUT39_NATIVE_LEASE_OWNER_CHANGED')
                live = lease.live()
                if live:
                    lease._check()
                leases.append(dict(owner=lease.owner, generation=lease.generation, native_live=live,
                    retained=True, release_readback_error=getattr(lease, 'release_readback_error', None)))
        return dict(execution_boundary='native', native_loaded=native is not None, native_registered=False,
            native_registration='NOT_APPLICABLE_DIRECT_ABI_LEASES', registered=self._registered,
            binding_reserved=self.binding_reserved,
            native_global_owner_table_observed=False,
            owner=owner, generation=generation, enabled=any(row['native_live'] for row in leases),
            inflight=0 if serving is None else serving['inflight'], leases=len(leases),
            owned_hooks=0 if serving is None else serving['owned_hooks'] + (
                serving['observer']['owned_hooks'] if serving['observer'] else 0), native=native, native_leases=leases,
            replay_pending=bool(self.attempted and not self.stopped), replay_failed=bool(self.failure and not self.stopped),
            serving=serving, next_phase=self.phase, failure=self.failure, evidence=self.evidence())
