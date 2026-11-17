"""Bounded natural model C2/C3 integration for RMSNorm-return and RoPE lookup.

The independent observer enters each actual module/API before admission. The
mechanism adapter owns output semantics; the common coverage session owns all
leases, once/every-match decisions and the group drain/recovery barrier. This
model-local deployment is separate from the public standard Worker controller.
"""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import uuid

from ._scope import ProviderError
from .tensor_modules import _capture, _identity, _same, _signature


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


class NaturalTensorGroupProvider:
    def __init__(self, model, mechanism, *, replay, synchronize, workload_sha256,
                 environment_sha256, model_weights_sha256, model_source_sha256,
                 simulation=False, model_observation=None, worker_identity=None):
        if mechanism not in ('8.restore_offset', '18.api_half', '22.wrong_row'):
            raise ValueError('Only reviewed natural RMSNorm, RoPE-lookup and MoE-restore groups are enabled')
        if not callable(replay) or not callable(synchronize) or model.training or type(simulation) is not bool:
            raise ValueError('An actual inference model replay and synchronizer are required')
        from fault_injection.coverage_session import CoverageScope
        from fault_injection.runtime_discovery import _python_execution_context
        from fault_injection.rmsnorm_return import eager_rms_contract
        self.model, self.mechanism = model, mechanism
        self.replay, self.synchronize, self.simulation = replay, synchronize, simulation
        self.model_observation = model_observation
        self.owner = 'natural-tensor-' + uuid.uuid4().hex
        self.context = _python_execution_context(model, worker_identity)
        import fault_injection.rmsnorm_return as rms
        import fault_injection.index22 as rope
        import fault_injection.tensor_observer as observer
        import fault_injection.tensor_callable as callable_proof
        import fault_injection._scoped_dispatch as scoped
        import fault_injection.module_faults as module_faults
        import fault_injection.route8 as moe
        import fault_injection.moe_census as moe_observer
        import fault_injection.moe_restore as moe_restore
        host = Path(__file__).resolve().parents[1]
        self.source_sha = _sha({str(Path(p).name): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (__file__, rms.__file__, rope.__file__, observer.__file__,
                      callable_proof.__file__, scoped.__file__, module_faults.__file__,
                      moe.__file__, moe_observer.__file__, moe_restore.__file__,
                      host/'serving_tensor.py', host/'serving_moe.py')})
        self.scope = CoverageScope(mechanism, workload_sha256, environment_sha256, self.source_sha,
            evidence_kind='cpu_contract' if simulation else
                'natural_worker' if worker_identity else 'natural_pretrained_model',
            model_weights_sha256=model_weights_sha256, model_source_sha256=model_source_sha256)
        self.eager_paths = {path or '<root>': eager_rms_contract(module)
            for path, module in model.named_modules()}
        self.rows, self.calls = [], []
        self.group = self.active_adapter = self.candidates = None
        self.selected = {}
        self.target_sources = {}
        self.target_guards = {}
        self.enabled = False
        self.off_row = None
        self.generation = 0
        self.error = None
        self._running = False
        self._statistics_arm = None
        self.current_request_ids = None
        self.owned_host_hooks = lambda: 0

    def _event(self, path, operator, metadata):
        key = (path, len(self.calls))
        row = dict(key=list(key), path=path, ordinal=key[1], operator=operator, metadata=metadata)
        self.calls.append(row)
        if self.group is not None:
            from fault_injection.coverage import ObservedCall, Phase
            target = self.selected.get(key)
            if target is None:
                # Every actual unknown mechanism boundary stays visible. An
                # added call aborts before injection rather than being hidden.
                raise ProviderError('NATURAL_TARGET_DENOMINATOR_CHANGED', str(key))
            phase = Phase.OFF_RECOVERY if not self.enabled else (
                Phase.PRE_FAULT_PROPAGATION if len(self.calls) == 1 else Phase.POST_FIRST_ANOMALY)
            invocation = self.current_invocation
            requests = self.current_request_ids if self.context['scope_kind'] == 'INITIALIZED_WORKER' else (invocation,)
            call = ObservedCall(target, invocation + ':' + str(key[1]), invocation, requests, phase)
            ticket = self.group.observer.observe(call, eligible=True)
            row['decision'] = self.group.begin(ticket)

    def _module_enter(self, path, module, args, kwargs):
        if self.mechanism == '18.api_half' and self.eager_paths.get(path) is not None:
            from fault_injection.tensor_observer import metadata, tensors
            self._event(path, 'RMSNorm.forward', [metadata(t) for t in tensors((args, kwargs))])

    def _operator_enter(self, func, args, kwargs, stack):
        from fault_injection.tensor_observer import metadata, tensors
        schema = getattr(func, '_schema', None)
        name = getattr(schema, 'name', '')
        inputs = tuple(tensors((args, kwargs)))
        if self.mechanism == '18.api_half':
            if (name.split('::', 1)[0] not in ('npu', '_C_ascend') or name.rsplit('::', 1)[-1]
                not in ('npu_rms_norm', 'npu_add_rms_norm', 'npu_add_rms_norm_bias')):
                return
            owners = [(path, module) for path, module, _ in stack
                if any(getattr(module, 'weight', None) is tensor for tensor in inputs)]
            if len(owners) != 1:
                raise ProviderError('RMSNORM_NATIVE_WEIGHT_OWNER_AMBIGUOUS')
            path, module = owners[0]
            if self.eager_paths.get(path) is not None:
                raise ProviderError('EAGER_NATIVE_RMSNORM_DOUBLE_BOUNDARY')
            self._event(path, str(func), [metadata(t) for t in inputs])
        elif self.mechanism == '22.wrong_row':
            if not ((name == 'aten::index_select' and not getattr(schema, 'overload_name', ''))
                    or (name == 'aten::index' and getattr(schema, 'overload_name', '') == 'Tensor')):
                return
            source = args[0] if args else kwargs.get('self', kwargs.get('input'))
            owners = [(path, module) for path, module, _ in stack
                      if getattr(module, 'cos_sin_cache', None) is source]
            if len(owners) == 1:
                self._event(stack[-1][0], str(func), [metadata(t) for t in inputs])

    def _admission(self, event):
        key = (event['path'], event['ordinal'])
        if len(self.calls) <= key[1] or tuple(self.calls[key[1]]['key']) != key:
            raise ProviderError('INDEPENDENT_OBSERVER_MUST_PRECEDE_INJECTION')
        if self.group is None:
            return self._statistics_arm == 'fault' and key in self.selected
        decision = self.calls[key[1]].get('decision')
        if decision is None:
            raise ProviderError('INDEPENDENT_ADMISSION_MISSING')
        return self.enabled and decision.admitted

    def _adapter(self, arm):
        if self.mechanism == '18.api_half':
            from fault_injection.rmsnorm_return import RMSNormReturnSession
            factory = RMSNormReturnSession
        else:
            from fault_injection.index22 import WrongRowSession
            factory = WrongRowSession
        return factory(self.model, arm=arm, simulation=self.simulation, admission=self._admission,
            diagnostics_context=getattr(self.model_observation, 'diagnostics', None))

    def _phase(self, name, *, arm=None):
        from fault_injection.tensor_observer import NaturalTensorObserver
        import torch
        if self._running:
            raise ProviderError('NATURAL_REPLAY_ALREADY_INFLIGHT')
        actual_modules = {path or '<root>': module for path, module in self.model.named_modules()}
        if self.model.training:
            raise ProviderError('NATURAL_INFERENCE_MODEL_CHANGED')
        for path, (module, source_sha) in self.target_sources.items():
            if actual_modules.get(path) is not module:
                raise ProviderError('NATURAL_TARGET_SOURCE_OR_MODEL_LIFETIME_CHANGED', path)
            self.target_guards[path].verify()
        self._running = True
        self.calls = []
        self.current_invocation = 'model-local-forward:' + uuid.uuid4().hex
        inputs = []
        def capture(module, args, kwargs):
            diagnostics = getattr(self.model_observation, 'diagnostics', None)
            with diagnostics() if diagnostics else nullcontext():
                inputs.append(_capture((args, kwargs)))
        handle = self.model.register_forward_pre_hook(capture, with_kwargs=True, prepend=True)
        adapter = self._adapter(arm) if arm else None
        observer = NaturalTensorObserver(self.model, on_module_enter=self._module_enter,
            on_operator_enter=self._operator_enter,
            diagnostics_context=getattr(self.model_observation, 'diagnostics', None))
        completed = False
        try:
            self.active_adapter = adapter
            with (adapter or nullcontext()), observer, torch.inference_mode():
                with (self.model_observation(self.model, name) if self.model_observation else nullcontext()):
                    output = self.replay(self.model, name)
                    self.synchronize()
                    completed = True
            if len(inputs) != 1:
                raise ProviderError('ONE_ACTUAL_ROOT_FORWARD_REQUIRED')
            output = _capture(output)
            report = adapter.report() if adapter else None
            census = observer.report()
            if not census['complete']:
                raise ProviderError('INDEPENDENT_CENSUS_INCOMPLETE')
            if not self.simulation and any(t['device'].split(':')[0] != 'npu'
                for row in census['forwards'] for t in row['output']):
                raise ProviderError('REAL_NPU_MODEL_OUTPUT_REQUIRED')
            row = dict(phase=name, invocation=self.current_invocation, input=_identity(inputs[0]),
                output=_identity(output), independent=census, mechanism=report,
                calls=[{key: value for key, value in call.items() if key != 'decision'} for call in self.calls],
                cleanup=adapter.runtime_state() if adapter else dict(cleanup_confirmed=True,
                    hook_owned=False, leases=0, inflight=0))
            self.rows.append(row)
            if self.group is not None:
                self._finish_calls(row)
            return inputs[0], output, row
        except BaseException as exc:
            self.error = repr(exc)
            if self.group is not None:
                from fault_injection.coverage import Outcome
                for call in self.calls:
                    decision = call.get('decision')
                    if decision is not None:
                        try:
                            self.group.finish(decision, Outcome.FAILED, original_calls=1 if completed else 0)
                        except BaseException:
                            pass
            raise
        finally:
            handle.remove()
            self.active_adapter = None
            self._running = False

    def _finish_calls(self, row):
        from fault_injection.coverage import Outcome
        observed = {(event['path'], event['ordinal']): event
                    for event in (row['mechanism'] or {}).get('events', ())}
        for call in self.calls:
            decision = call['decision']; key = tuple(call['key']); event = observed.get(key)
            if decision.admitted:
                if (event is None or event.get('applied') is not True or
                    event.get('original_execution_verified') is not True or not event.get('consumers') or
                    event.get('output_changed') is not True or
                    not event.get('return_readback_exact', event.get('replacement_readback_exact', False))):
                    raise ProviderError('NATURAL_EFFECT_AND_CONSUMER_UNPROVEN', str(key))
                self.group.finish(decision, Outcome.HIT, native_applied=True,
                    execution_sha256=_sha(event), effect_sha256=_sha(dict(event=event, model_output=row['output'])))
            else:
                if event is not None and event.get('applied'):
                    raise ProviderError('NON_TARGET_TENSOR_REPLACEMENT')
                self.group.finish(decision, Outcome.PASSTHROUGH,
                                  execution_sha256=_sha(dict(call=call['key'], output=row['output'])))

    def qualify(self):
        from fault_injection.coverage import (BoundaryTargetIdentity, Candidate, CandidateSet,
            Eligibility, EligibilityResult, PythonHookBoundary)
        if self.candidates is not None:
            raise ProviderError('NATURAL_GROUP_ALREADY_QUALIFIED')
        phases = []
        for name, arm in (('native_baseline', None), ('baseline', 'baseline'),
                          ('baseline_repeat', 'baseline'), ('lossless', 'baseline')):
            phases.append(self._phase(name, arm=arm))
        return self.qualify_observations(phases)

    def qualify_observations(self, phases):
        """Trusted synchronous/serving host joins real completed healthy phases."""
        from fault_injection.coverage import (BoundaryTargetIdentity, Candidate, CandidateSet,
            Eligibility, EligibilityResult, PythonHookBoundary)
        if self.candidates is not None or len(phases) != 4:
            raise ProviderError('FOUR_FRESH_ACTUAL_HEALTHY_PHASES_REQUIRED')
        self.reference_input, self.reference_output, bare = phases[0]
        if not all(_same(self.reference_input, input_) and _same(self.reference_output, output)
                   for input_, output, row in phases[1:]):
            raise ProviderError('NATIVE_BASELINE_REPEAT_OR_LOSSLESS_NOT_EXACT')
        first, repeat, lossless = [row for _, _, row in phases[1:]]
        events = first['mechanism']['events']
        if not events or len(events) != len(first['calls']) or len(events) != len(repeat['calls']):
            raise ProviderError('ACTUAL_NATURAL_TARGET_ABSENT_OR_DENOMINATOR_INCOMPLETE')
        if any([_signature(e, self.mechanism) for e in events] !=
                [_signature(e, self.mechanism) for e in phase['mechanism']['events']]
                for phase in (repeat, lossless)):
            raise ProviderError('NATURAL_TARGET_IDENTITY_UNSTABLE')
        modules = {path or '<root>': module for path, module in self.model.named_modules()}
        rows = []
        for event, independent in zip(events, first['calls']):
            key = (event['path'], event['ordinal'])
            if (list(key) != independent['key'] or event.get('status') != 'baseline'
                    or not event.get('original_execution_verified') or not event.get('consumers')):
                raise ProviderError('NATURAL_ELIGIBLE_CALL_OR_CONSUMER_UNPROVEN', str(key))
            module = modules[key[0]]
            from fault_injection.tensor_callable import TensorCallableGuard
            guard = TensorCallableGuard(module,
                extra_methods=('forward_impl',) if self.mechanism == '8.restore_offset' else (),
                # AscendFusedMoE overrides forward and uses runner/forward_impl;
                # its inherited, unused CustomOp dispatch field is not this
                # boundary. A direct _forward_method access always stays bound.
                bind_optional_dispatch=self.mechanism != '8.restore_offset')
            source = guard.source_sha256
            self.target_sources[key[0]] = (module, source)
            self.target_guards[key[0]] = guard
            device = next(iter(first['independent']['forwards'][0]['output']))['device'].split(':')
            target = BoundaryTargetIdentity(process_id=self.context['process_id'],
                process_start_identity=self.context['process_start_identity'], worker_id=self.context['worker_id'],
                scope_kind=self.context['scope_kind'], rank=self.context['rank'], device_type=device[0],
                logical_device=None if device[0] == 'cpu' else int(device[1]) if len(device) > 1 else 0,
                model_instance_id=self.context['model_instance_id'], model_object_identity=id(self.model),
                module_object_identity=id(module), module_roles={
                    '18.api_half': ('RMSNORM_RETURN',), '22.wrong_row': ('ROPE_ROW_READER',),
                    '8.restore_offset': ('MOE_RESTORE_INPUTS',)}[self.mechanism],
                operator_api=event.get('operator', event['boundary']),
                tensor_signature=json.dumps(_signature(event, self.mechanism), sort_keys=True),
                boundary=PythonHookBoundary(guard.callable_identity, source, key[1]),
                provider_identity='natural_tensor:' + self.mechanism,
                provider_proof=(('implementation_sha256', self.source_sha),))
            self.selected[key] = target
            rows.append(Candidate(key[0] + ':' + str(key[1]), EligibilityResult(Eligibility.SUPPORTED), target))
        observed_paths = {event['path'] for event in events}
        if self.mechanism == '18.api_half':
            required = {path for path, contract in self.eager_paths.items() if contract is not None}
            if not required.issubset(observed_paths):
                raise ProviderError('STRUCTURAL_RMSNORM_NOT_OBSERVED', str(sorted(required-observed_paths)))
            structural = [item for item in first['independent']['module_inventory']
                          if item['role'] == 'rmsnorm_source_candidate']
            for item in structural:
                if item['path'] not in observed_paths:
                    rows.append(Candidate(item['path'] + ':unreviewed_rms_source',
                        EligibilityResult(Eligibility.UNSUPPORTED,
                            ('No reviewed eager implementation or actual native RMS weight/return lineage',))))
        elif self.mechanism == '22.wrong_row':
            for path, module in modules.items():
                if hasattr(module, 'cos_sin_cache') and path not in observed_paths:
                    rows.append(Candidate(path + ':opaque_rope_reader', EligibilityResult(Eligibility.UNSUPPORTED,
                        ('No exposed index_select/advanced-index row reader in this actual workload',))))
        else:
            # Independent actual Python-frame inventory must match every MoE
            # module, including custom ops opaque to the outer dispatch mode.
            trace = first['independent']['moe_trace']
            expected = set(trace['structural_moe_paths'])
            if (expected != observed_paths or len(events) != len(expected)
                    or len(events) != trace['observed_unpermute_calls']):
                raise ProviderError('NATURAL_MOE_STRUCTURAL_DENOMINATOR_INCOMPLETE')
            if not all(event.get('complete_permutation_value_verified')
                       and event.get('complete_native_output_oracle_exact') for event in events):
                raise ProviderError('NATURAL_MOE_COMPLETE_GROUP_VALUES_UNPROVEN')
        self.candidates = CandidateSet(tuple(rows), True)
        self.discovery_sha = _sha(dict(baseline=first, repeat=repeat, lossless=lossless))
        self.baseline_sha = _sha(_identity(self.reference_output))
        self.input_sha = _sha(_identity(self.reference_input))
        self.lossless_sha = _sha(lossless)
        return dict(candidate_set=self.candidates.report(), independent_discovery_sha256=self.discovery_sha,
                    native_baseline_exact=True, lossless_exact=True,
                    actual_dispatch={path: guard.report() for path, guard in self.target_guards.items()})

    def _recover(self, lease):
        from fault_injection.coverage import RecoveryReport
        if self.off_row is None:
            input_, output, row = self._phase('off')
            if not _same(input_, self.reference_input) or not _same(output, self.reference_output):
                raise ProviderError('GROUP_COMPLETE_OFF_NOT_EXACT')
            self.off_row = row
        return RecoveryReport(lease, True, True, True, proof_sha256=_sha(self.off_row),
                              off_output_sha256=self.baseline_sha)

    def _health(self, worker, owner, generation):
        from fault_injection.coverage_session import WorkerHealthProof
        if self._running or self.active_adapter is not None or self.off_row is None or self.owned_host_hooks():
            raise ProviderError('GROUP_HEALTH_STILL_INFLIGHT')
        return WorkerHealthProof(worker, owner, generation, _sha(self.off_row), self.baseline_sha,
            _sha(self.off_row['output']), 0, 0, 0, 0, 0)

    def open_group(self, trigger='once', *, forwards=1, c2_proof=None):
        from fault_injection.coverage import (CoverageCapability, QualifiedTargetSet,
            TargetLease, TargetLosslessProof)
        from fault_injection.coverage_session import (CoverageSession, GroupRecoveryHooks,
            IndependentCallObserver, SessionBudget)
        if self.candidates is None or type(forwards) is not int or not 1 <= forwards <= 8:
            raise ProviderError('QUALIFIED_BOUNDED_NATURAL_REPLAY_REQUIRED')
        if trigger == 'once' and forwards != 1:
            raise ProviderError('C2_USES_ONE_COMPLETE_WORKLOAD_FORWARD')
        count = len(self.selected)
        self.generation += 1
        leases = tuple(TargetLease(target, self.owner, self.generation, 'target-'+str(index), 0, 0)
                       for index, target in enumerate(self.candidates.eligible))
        # These hooks allocate return tensors through the model's ordinary
        # allocator; no persistent native buffer is reserved across forwards.
        group = QualifiedTargetSet(self.candidates, leases,
            tuple(TargetLosslessProof(lease, 'return_or_lookup', self.lossless_sha,
                (self.input_sha,), (self.input_sha,), self.baseline_sha, self.baseline_sha, 1, True, True)
                for lease in leases), CoverageCapability(single_target=True, multi_target=True,
                every_match=True, max_concurrency=count, max_resources=0, max_host_bytes=0,
                group_recovery=True, recovery_scope='target_set', stop_policy='bounded_session',
                lossless_modes=('return_or_lookup',)), concurrency=count, selection='all_eligible', trigger=trigger)
        observer = IndependentCallObserver(source='independent_workload_observer',
            observer_id=uuid.uuid4().hex, implementation_sha256=self.source_sha,
            workload_sha256=self.scope.workload_sha256, max_calls=count*(forwards+2))
        workers = tuple(set(target.worker for target in self.selected.values()))
        observer.seal_discovery(self.candidates, observed_targets=self.candidates.eligible,
            artifact_sha256=self.discovery_sha, expected_workers=workers, observed_workers=workers)
        self.enabled = True; self.off_row = None
        def stop(lease):
            self.enabled = False
        def drain(lease, seconds):
            self.synchronize()
            return not self._running and self.active_adapter is None
        self.group = CoverageSession(group=group, scope=self.scope, observer=observer,
            budget=SessionBudget(count*forwards, 600, max_concurrency=count),
            recovery=GroupRecoveryHooks(stop, drain, self._recover, self._health), c2_proof=c2_proof)
        self.group_first_row = len(self.rows)
        return self.group

    def close_group(self):
        if self.group is None:
            raise ProviderError('NATURAL_GROUP_NOT_OPEN')
        recovery = self.group.disable(timeout_seconds=300)
        self.group.observer.seal(artifact_sha256=_sha(self.rows[self.group_first_row:]), complete=True)
        report = self.group.report()
        report.update(execution_boundary='python_hook', native_loaded=False,
            mechanism_semantics={
                '18.api_half': 'RMSNorm output half proxy',
                '22.wrong_row': 'Legal-address RoPE wrong-row lookup output proxy; not uint32 overflow',
                '8.restore_offset': 'Actual MoE unpermute paired token-group index/probability remap'}[self.mechanism],
            recovery=recovery, independent_statistics='SEPARATE_RAW_COMPARISON_REQUIRED',
            phase_rows=self.rows[self.group_first_row:],
            scheduler_requests_claimed=self.context['scope_kind'] == 'INITIALIZED_WORKER',
            invocation_scope=self.context['scope_kind'])
        return report

    def run(self, trigger='once', *, forwards=1, c2_proof=None):
        self.open_group(trigger, forwards=forwards, c2_proof=c2_proof)
        try:
            for index in range(forwards):
                self._phase(trigger + ':' + str(index), arm='fault')
        finally:
            report = self.close_group()
        return report

    def collect_statistics_pair(self, capture):
        """Actual B/F at one common callsite after group recovery, then full OFF.

        This adds independent statistics evidence; it grants no extra coverage.
        The collector is instantiated by the caller after source qualification.
        """
        if self.group is None or not self.group.report()['contract_satisfied']:
            raise ProviderError('RECOVERED_COVERAGE_GROUP_REQUIRED_BEFORE_STATISTICS')
        saved_group, saved_observation = self.group, self.model_observation
        self.group = None; self.model_observation = capture
        first_row = len(self.rows)
        try:
            for name, arm in (('statistics_baseline', 'baseline'),
                              ('statistics_baseline_repeat', 'baseline'),
                              ('statistics_fault', 'fault'), ('statistics_off', None)):
                self._statistics_arm = arm
                input_, output, row = self._phase(name, arm=arm)
                if not _same(input_, self.reference_input):
                    raise ProviderError('STATISTICS_MODEL_INPUT_CHANGED')
                if arm != 'fault' and not _same(output, self.reference_output):
                    raise ProviderError('STATISTICS_BASELINE_OR_OFF_NOT_EXACT')
                if arm == 'fault':
                    hits = [event for event in row['mechanism']['events'] if event.get('applied')]
                    if len(hits) != len(self.selected) or not all(event.get('consumers') and
                        event.get('output_changed') for event in hits):
                        raise ProviderError('STATISTICS_ACTUAL_EFFECT_OR_CONSUMER_UNPROVEN')
            return dict(phases=self.rows[first_row:], baseline_exact=True, off_exact=True,
                        first_difference='INDEPENDENT_RAW_COMPARATOR_REQUIRED')
        finally:
            self.group, self.model_observation = saved_group, saved_observation
            self._statistics_arm = None


def inspect_source(module):
    from fault_injection.tensor_callable import TensorCallableGuard
    return TensorCallableGuard(module).source_sha256
