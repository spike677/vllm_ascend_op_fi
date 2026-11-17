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
                 simulation=False, model_observation=None):
        if mechanism not in ('18.api_half', '22.wrong_row'):
            raise ValueError('Only reviewed natural RMSNorm-return and RoPE-lookup groups are enabled')
        if not callable(replay) or not callable(synchronize) or model.training or type(simulation) is not bool:
            raise ValueError('An actual inference model replay and synchronizer are required')
        from fault_injection.coverage_session import CoverageScope
        from fault_injection.runtime_discovery import _python_execution_context
        from fault_injection.rmsnorm_return import eager_rms_contract
        self.model, self.mechanism = model, mechanism
        self.replay, self.synchronize, self.simulation = replay, synchronize, simulation
        self.model_observation = model_observation
        self.owner = 'natural-tensor-' + uuid.uuid4().hex
        self.context = _python_execution_context(model, None)
        import fault_injection.rmsnorm_return as rms
        import fault_injection.index22 as rope
        import fault_injection.tensor_observer as observer
        self.source_sha = _sha({str(Path(p).name): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (__file__, rms.__file__, rope.__file__, observer.__file__)})
        self.scope = CoverageScope(mechanism, workload_sha256, environment_sha256, self.source_sha,
            evidence_kind='cpu_contract' if simulation else 'natural_pretrained_model',
            model_weights_sha256=model_weights_sha256, model_source_sha256=model_source_sha256)
        self.eager_paths = {path or '<root>': eager_rms_contract(module)
            for path, module in model.named_modules()}
        self.rows, self.calls = [], []
        self.group = self.active_adapter = self.candidates = None
        self.selected = {}
        self.target_sources = {}
        self.enabled = False
        self.off_row = None
        self.generation = 0
        self.error = None
        self._running = False

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
            call = ObservedCall(target, invocation + ':' + str(key[1]), invocation,
                                (invocation,), phase)
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
            return False
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
        return factory(self.model, arm=arm, simulation=self.simulation, admission=self._admission)

    def _phase(self, name, *, arm=None):
        from fault_injection.tensor_observer import NaturalTensorObserver
        import torch
        if self._running:
            raise ProviderError('NATURAL_REPLAY_ALREADY_INFLIGHT')
        actual_modules = {path or '<root>': module for path, module in self.model.named_modules()}
        if self.model.training:
            raise ProviderError('NATURAL_INFERENCE_MODEL_CHANGED')
        for path, (module, source_sha) in self.target_sources.items():
            if actual_modules.get(path) is not module or inspect_source(module) != source_sha:
                raise ProviderError('NATURAL_TARGET_SOURCE_OR_MODEL_LIFETIME_CHANGED', path)
        self._running = True
        self.calls = []
        self.current_invocation = 'model-local-forward:' + uuid.uuid4().hex
        inputs = []
        def capture(module, args, kwargs):
            inputs.append(_capture((args, kwargs)))
        handle = self.model.register_forward_pre_hook(capture, with_kwargs=True, prepend=True)
        adapter = self._adapter(arm) if arm else None
        observer = NaturalTensorObserver(self.model, on_module_enter=self._module_enter,
                                         on_operator_enter=self._operator_enter)
        completed = False
        try:
            self.active_adapter = adapter
            with (adapter or nullcontext()), observer, torch.inference_mode():
                with (self.model_observation(self.model) if self.model_observation else nullcontext()):
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
            source = inspect_source(module)
            self.target_sources[key[0]] = (module, source)
            device = next(iter(first['independent']['forwards'][0]['output']))['device'].split(':')
            target = BoundaryTargetIdentity(process_id=self.context['process_id'],
                process_start_identity=self.context['process_start_identity'], worker_id=self.context['worker_id'],
                scope_kind='MODEL_LOCAL', rank=self.context['rank'], device_type=device[0],
                logical_device=None if device[0] == 'cpu' else int(device[1]) if len(device) > 1 else 0,
                model_instance_id=self.context['model_instance_id'], model_object_identity=id(self.model),
                module_object_identity=id(module), module_roles=('RMSNORM_RETURN',) if self.mechanism == '18.api_half'
                    else ('ROPE_ROW_READER',), operator_api=event.get('operator', event['boundary']),
                tensor_signature=json.dumps(_signature(event, self.mechanism), sort_keys=True),
                boundary=PythonHookBoundary(str(id(getattr(module.forward, '__func__', module.forward))), source, key[1]),
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
        else:
            for path, module in modules.items():
                if hasattr(module, 'cos_sin_cache') and path not in observed_paths:
                    rows.append(Candidate(path + ':opaque_rope_reader', EligibilityResult(Eligibility.UNSUPPORTED,
                        ('No exposed index_select/advanced-index row reader in this actual workload',))))
        self.candidates = CandidateSet(tuple(rows), True)
        self.discovery_sha = _sha(dict(baseline=first, repeat=repeat, lossless=lossless))
        self.baseline_sha = _sha(_identity(self.reference_output))
        self.input_sha = _sha(_identity(self.reference_input))
        self.lossless_sha = _sha(lossless)
        return dict(candidate_set=self.candidates.report(), independent_discovery_sha256=self.discovery_sha,
                    native_baseline_exact=True, lossless_exact=True)

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
        if self._running or self.active_adapter is not None or self.off_row is None:
            raise ProviderError('GROUP_HEALTH_STILL_INFLIGHT')
        return WorkerHealthProof(worker, owner, generation, _sha(self.off_row), self.baseline_sha,
            _sha(self.off_row['output']), 0, 0, 0, 0, 0)

    def run(self, trigger='once', *, forwards=1, c2_proof=None):
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
        first_row = len(self.rows)
        try:
            for index in range(forwards):
                self._phase(trigger + ':' + str(index), arm='fault')
        finally:
            recovery = self.group.disable(timeout_seconds=300)
        observer.seal(artifact_sha256=_sha(self.rows[first_row:]), complete=True)
        report = self.group.report()
        report.update(execution_boundary='python_hook', native_loaded=False,
            mechanism_semantics='RMSNorm output half proxy' if self.mechanism == '18.api_half'
                else 'Legal-address RoPE wrong-row lookup output proxy; not uint32 overflow',
            recovery=recovery, independent_statistics='SEPARATE_RAW_COMPARISON_REQUIRED',
            phase_rows=self.rows[first_row:], scheduler_requests_claimed=False,
            invocation_scope='Actual model-local root forwards; not synthetic serving request IDs')
        return report


def inspect_source(module):
    from fault_injection.tensor_observer import callable_identity
    identity = callable_identity(module)
    if 'source_sha256' not in identity:
        raise ProviderError('ACTUAL_MODULE_SOURCE_IDENTITY_REQUIRED')
    return identity['source_sha256']
