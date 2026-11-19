"""Public Runtime lease for existing natural Fault8/18 serving groups.

This adapter sequences actual enqueued requests; it never invokes replay or
model.forward. CoverageSession remains the target/proof/recovery authority.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

from ._scope import Scope, ProviderError, reject_unknown
from .tensor_modules import _capture, _identity, _same
from .tensor_natural import _sha
from .. import identity as identity_reader
from ..serving_tensor_binding import TensorServingBinding, MECHANISMS, _ATTRIBUTE
from ..serving_context import snapshot_fresh_prefill
from fault_injection.effect_summary import actual_request_effect, current_completed_row


def require(condition, reason):
    if not condition:
        raise ProviderError(reason)


REQUESTS = (('native_baseline', None), ('baseline', 'baseline'),
    ('baseline_repeat', 'baseline'), ('lossless', 'baseline'), ('c2_fault', 'fault'),
    ('c2_off', None), ('c3_fault_0', 'fault'), ('c3_fault_1', 'fault'), ('c3_off', None),
    ('cleanup_healthy', None))


class TensorServingProvider(Scope):
    execution_boundary = 'python_hook'

    def __init__(self, worker, model, identity, mechanism):
        super().__init__(worker, model, identity)
        require(mechanism in MECHANISMS, 'REVIEWED_NATURAL_TENSOR_MECHANISM_REQUIRED')
        self.mechanism = mechanism
        self.runner, self.thread = worker.model_runner, threading.get_ident()
        bindings = getattr(worker, _ATTRIBUTE, None)
        self.binding = bindings.get(mechanism) if type(bindings) is dict else None
        # Resolver readback precedes Runtime's owner creation, and installs
        # nothing. A missing binding must not leave an empty unreleasable lease.
        require(type(self.binding) is TensorServingBinding, 'REVIEWED_TENSOR_SERVING_BINDING_REQUIRED')
        self.binding_reserved = False
        self.session = self.proof = None
        self.checked = self.attempted = self.complete = self.recovered = False
        self.stopped = True
        self.recovering = self.pending = self.error = None
        self.index = 0
        self.reports, self.failures = {}, []
        self.health_hooks, self.health_current = [], None
        self.health_active = False
        self.health_callbacks = 0
        self._context()

    def _identity(self):
        value = self.identity
        device = getattr(self.worker, 'device', None)
        visible, source = identity_reader._visible_devices()
        require(os.getpid() == value['pid'] and threading.get_ident() == self.thread
            and self.worker.model_runner is self.runner and self.runner.model is self.model
            and self.worker.rank == value['rank'] and getattr(device, 'type', None) == 'npu'
            and getattr(device, 'index', None) == value['configured_logical_device']
            and identity_reader._current_device() == value['logical_device']
            and value['observed_logical_device'] == value['logical_device']
            and visible == value['visible_devices'] and source == value['visible_devices_source'],
            'TENSOR_HELD_WORKER_CONTEXT_CHANGED')
        return dict(status='WORKER_IDENTIFIED', worker=deepcopy(value), quarantined=False)

    def _context(self):
        self._identity()
        bindings = getattr(self.worker, _ATTRIBUTE, None)
        binding = self.binding
        require(type(bindings) is dict and bindings.get(self.mechanism) is binding
            and binding.mechanism == self.mechanism and binding.worker is self.worker
            and binding.runner is self.runner and binding.model is self.model
            and binding.pid == os.getpid() and binding.thread == self.thread,
            'TENSOR_DEPLOYMENT_BINDING_CHANGED')
        if self.binding_reserved or binding.leases:
            require(self.binding_reserved and binding.leases == {self._ticket}, 'TENSOR_DEPLOYMENT_OWNER_CHANGED')

    def _failure(self, error, action):
        message = getattr(self.session, 'request_error', None) or type(error).__name__ + ': ' + str(error)
        if self.error is None:
            self.error = message
        self.failures.append(dict(action=action, error=message, first_error=self.error))

    def dispatch(self, action, settings, *, owner, generation):
        if action == 'prepare':
            self.bind(owner, generation)
        self.verify(owner, generation)
        try:
            self._context()
            reject_unknown(settings, ('request_id', 'collect') if action == 'advance' else ())
            if action == 'prepare':
                require(not self.binding.leases, 'TENSOR_DEPLOYMENT_ALREADY_OWNED')
            elif action == 'register':
                require(not self.binding.leases, 'TENSOR_DEPLOYMENT_ALREADY_OWNED')
                self.binding.leases.add(self._ticket)
                self.binding_reserved = self._registered = True
            elif action == 'check':
                require(self._registered and not self.model.training, 'TENSOR_REGISTERED_INFERENCE_REQUIRED')
                self.checked = True
            elif action in ('enable', 'execute'):
                require(self.checked and not self.attempted, 'TENSOR_SINGLE_ATTEMPT_REQUIRED')
                self.attempted, self.stopped = True, False
                self._open_session(owner, generation)
            elif action == 'advance':
                require(self.attempted and not self.stopped and not self.error,
                        'TENSOR_ACTIVE_SERVING_ATTEMPT_REQUIRED')
                if 'collect' in settings:
                    require(settings == {'collect': True} and type(settings['collect']) is bool,
                            'EXPLICIT_EXCLUSIVE_COLLECTION_REQUIRED')
                    require(self.pending is not None, 'TENSOR_NO_REQUEST_TO_COLLECT')
                    self._complete_request()
                else:
                    require(set(settings) == {'request_id'} and self.pending is None,
                            'TENSOR_EXPLICIT_COLLECTION_REQUIRED')
                    request_id = settings['request_id']
                    require(isinstance(request_id, str) and request_id and request_id not in self.session.seen,
                            'FRESH_REAL_ENQUEUED_REQUEST_REQUIRED')
                    require(self.recovering or self.index < len(REQUESTS), 'TENSOR_PROTOCOL_ALREADY_COMPLETE')
                    phase, arm = (('recovery_' + self.recovering, None) if self.recovering else REQUESTS[self.index])
                    self.pending = (request_id, phase)
                    if phase in ('cleanup_healthy', 'recovery_health'):
                        self._schedule_health(request_id)
                    else:
                        self.session.expect(request_id, phase, arm)
            elif action == 'finish':
                if self.pending is not None:
                    self._complete_request()
                require(self.complete or self.recovered, 'TENSOR_REAL_REQUEST_PROTOCOL_NOT_FINISHED')
                self._cleanup()
            elif action == 'recover':
                self._start_recovery()
            elif action in ('disable', 'abort', 'clean', 'unregister'):
                self._cleanup()
                if action in ('clean', 'unregister'):
                    self._registered = False
                if action == 'unregister' and self.binding_reserved:
                    self.binding.leases.remove(self._ticket)
                    self.binding_reserved = False
            else:
                raise ProviderError('UNSUPPORTED_PROVIDER_ACTION')
        except BaseException as error:
            self._failure(error, action)
            raise
        return self.query(owner=owner, generation=generation)

    def _open_session(self, owner, generation):
        from ..serving_tensor_request import ActualTensorRequest, ActualMoERequest
        factory = ActualMoERequest if self.mechanism == '8.restore_offset' else ActualTensorRequest
        binding = self.binding
        self.session = factory.__new__(factory)
        options = dict(_trace_factory=binding.trace_factory) if self.mechanism == '8.restore_offset' else {}
        self.session.__init__(self.worker, prompt=binding.prompt,
            weight_manifest=json.loads(binding.weight_manifest_json), workload_sha256=binding.workload_sha256,
            environment_sha256=binding.environment_sha256, _simulation=binding.simulation,
            _synchronize=binding.synchronize, _identity_guard=self._identity, **options)
        group = self.session.provider
        group.owner, group.generation = owner, generation - 1
        base = Path(__file__).parents[1]
        sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            Path(__file__), base/'serving_tensor_binding.py', base/'serving_tensor_request.py', base/'registry.py')}
        group.source_sha = _sha(dict(group=group.source_sha, worker_rpc_binding=sources))
        group.scope = replace(group.scope, implementation_sha256=group.source_sha)
        self.reports['deployment'] = dict(binding_sources=sources, worker_rpc_owner=owner,
            worker_rpc_generation=generation, qualification_granted=False, simulation=binding.simulation,
            mechanism=self.mechanism, historical_scope_inherited=False, constructed_replay_fallback=False)

    def _complete_request(self):
        request_id, phase = self.pending
        session = self.session
        require(not self.error and not session.request_error, 'TENSOR_REQUEST_FAILED_RECOVERY_REQUIRED')
        require(request_id in session.seen and session.expected is None and session.current is None
            and session.deferred_request is None and self.health_current is None
            and not session.callbacks_active and not self.health_callbacks,
            'TENSOR_REAL_ENQUEUED_REQUEST_NOT_RETURNED')
        if phase == 'lossless':
            self.reports['discovery'] = session.qualify()
            session.provider.open_group('once', forwards=1)
        elif phase == 'c2_off':
            report = session.finish_group()
            require(report['contract_satisfied'], 'TENSOR_C2_CONTRACT_NOT_COMPLETE')
            self.proof = session.provider.group.c2_proof()
            report['live_c2_proof'] = session.provider.group.report()['live_c2_proof']
            self.reports['C2'] = report
            # Only this original in-process object can authorize C3. JSON,
            # copied dataclasses or certificates from old scopes are not inputs.
            session.provider.open_group('every_match', forwards=2, c2_proof=self.proof)
        elif phase == 'c3_off':
            report = session.finish_group()
            require(report['contract_satisfied'], 'TENSOR_C3_CONTRACT_NOT_COMPLETE')
            self.reports['C3'] = report
            self.reports['cleanup'] = session.close()
        elif phase == 'recovery_off':
            report = session.finish_group()
            self.reports['failed_attempt_recovery'] = report
            require(report['recovery']['complete'], 'TENSOR_GROUP_RECOVERY_UNCONFIRMED')
            self.reports['cleanup'] = session.close()
            self.recovering = 'health'
        elif phase in ('cleanup_healthy', 'recovery_health'):
            require(self.reports.get('same_worker_healthy', {}).get('request_id') == request_id,
                    'TENSOR_REAL_CLEANUP_HEALTH_REQUIRED')
            self.recovered = True
            if phase == 'recovery_health':
                self.recovering, self.stopped = None, True
        self.pending = None
        if not phase.startswith('recovery_'):
            self.index += 1
            self.complete = self.index == len(REQUESTS)

    def _remove_health_hooks(self):
        for handle, callback in reversed(self.health_hooks[:]):
            hooks = handle.hooks_dict_ref()
            require(hooks is not None and (handle.id not in hooks or hooks[handle.id] is callback),
                    'TENSOR_HEALTH_HOOK_OWNER_CHANGED')
            try:
                handle.remove()
            finally:
                if handle.id not in hooks:
                    self.health_hooks.remove((handle, callback))
            require(handle.id not in hooks, 'TENSOR_HEALTH_HOOK_REMOVAL_UNCONFIRMED')

    def _schedule_health(self, request_id):
        session = self.session
        require(not session.handles and session.adapter is None and session.observer is None
            and not self.health_hooks and session.provider.group.recovery_report()['complete'],
            'TENSOR_OWNED_CLEANUP_REQUIRED_BEFORE_HEALTH')
        def guarded(callback):
            def call(*args):
                self.health_callbacks += 1
                try:
                    self._context()
                    return callback(*args)
                except BaseException as error:
                    self._failure(error, 'actual_health')
                    raise
                finally:
                    self.health_callbacks -= 1
            return call
        @guarded
        def pre(module, args, kwargs):
            from vllm.forward_context import get_forward_context
            require(self.health_current is None, 'TENSOR_HEALTH_FORWARD_ALREADY_STARTED')
            actual = session.binding.bind(module, args, kwargs)
            scheduler = snapshot_fresh_prefill(self.runner, self.model, actual, get_forward_context(),
                expected_prompt_token_ids=session.prompt, expected_internal_request_id=request_id,
                seen_internal_request_ids=session.seen)
            self.health_current = dict(input=_capture((args, kwargs)), scheduler=scheduler, entry_error=sys.exc_info()[1])
            self.health_active = True
        @guarded
        def post(module, args, kwargs, output):
            try:
                require(self.health_current is not None and output is not None
                    and sys.exc_info()[1] is self.health_current['entry_error']
                    and next(reversed(module._forward_hooks.values())) is post, 'TENSOR_HEALTH_ROOT_NOT_RETURNED')
                self.health_current['hidden'] = _capture(output)
            finally:
                self.health_active = False
        @guarded
        def logits(module, args, kwargs, output):
            current = self.health_current
            require(current is not None and 'hidden' in current and output is not None
                and sys.exc_info()[1] is current['entry_error']
                and next(reversed(module._forward_hooks.values())) is logits, 'TENSOR_HEALTH_LOGITS_NOT_RETURNED')
            result = dict(hidden=current['hidden'], logits=_capture(output))
            session.provider.synchronize()
            require(_same(current['input'], session.provider.reference_input)
                and _same(result, session.provider.reference_output), 'TENSOR_CLEANUP_HEALTH_NOT_EXACT')
            self._remove_health_hooks()
            self.reports['same_worker_healthy'] = dict(request_id=request_id, worker=deepcopy(self.identity),
                input=_identity(current['input']), output=_identity(result), scheduler=current['scheduler'],
                cleanup_before_request=True, full_hidden_logits_exact=True, fault_hooks=0, qualification_granted=False)
            session.seen.append(request_id)
            self.health_current = None
        for module, callback, before in ((self.model, pre, True), (self.model, post, False),
                                        (session.processor, logits, False)):
            register = module.register_forward_pre_hook if before else module.register_forward_hook
            handle = register(callback, with_kwargs=True, **({} if before else {'always_call': True}))
            self.health_hooks.append((handle, callback))

    def _owned_abort(self):
        require(not self.health_active and not self.health_callbacks, 'TENSOR_ACTUAL_HEALTH_STILL_ACTIVE')
        self._remove_health_hooks()
        self.health_current = None
        session = self.session
        if session is not None and hasattr(session, 'handles'):
            require(not session.root_active and not session.callbacks_active, 'TENSOR_ACTUAL_FORWARD_STILL_ACTIVE')
            if (session.handles or session.expected is not None or session.deferred_request is not None
                    or session.observer is not None or session.adapter is not None or session.completion_handle is not None):
                self.failures.append(session.abort())
        self.pending = None

    def _start_recovery(self):
        self._owned_abort()
        group = getattr(getattr(self.session, 'provider', None), 'group', None)
        if group is None:
            self._cleanup()
            self.recovered = True
            return
        if not group.recovery_report()['complete'] and self.session.provider.off_row is not None:
            self.reports['cleanup_recovery'] = self.session.finish_group()
        self.recovering = 'health' if group.recovery_report()['complete'] else 'off'
        self.session.request_error = self.error = None
        self.stopped = False

    def _cleanup(self):
        self._owned_abort()
        session = self.session
        if session is not None and hasattr(session, 'handles'):
            group = session.provider.group
            if group is not None and not group.recovery_report()['complete'] and session.provider.off_row is not None:
                self.reports['cleanup_recovery'] = session.finish_group()
            require(group is None or group.recovery_report()['complete'], 'TENSOR_REAL_OFF_REQUEST_REQUIRED')
            require(group is None or self.recovered, 'TENSOR_REAL_CLEANUP_HEALTH_REQUIRED')
            self.reports['cleanup'] = session.close()
            session.request_error = None
        self.error = None
        self.stopped = True

    def query(self, *, owner, generation):
        self.verify(owner, generation)
        self._context()
        session = self.session
        adapter = getattr(session, 'adapter', None)
        state = adapter.runtime_state() if adapter is not None else dict(enabled=False, inflight=0, leases=0)
        if adapter is not None:
            require(state['ownership_verified'], 'TENSOR_DISPATCH_OWNERSHIP_CHANGED')
        observer = getattr(session, 'observer', None)
        if observer is not None:
            actual_observer = getattr(observer, 'observer', observer)
            for handle, callback in actual_observer.handles:
                hooks = handle.hooks_dict_ref()
                require(hooks is not None and hooks.get(handle.id) is callback, 'TENSOR_OBSERVER_HOOK_OWNER_CHANGED')
            trace = getattr(observer, 'trace', None)
            if trace is not None and trace.active:
                require(sys.getprofile() is trace.callback and trace.thread == self.thread, 'TENSOR_PROFILE_OWNER_CHANGED')
        if session is not None and hasattr(session, 'handles'):
            session._context(verify_targets=False)
        for handle, callback in self.health_hooks:
            require(handle.hooks_dict_ref() is not None and handle.hooks_dict_ref().get(handle.id) is callback,
                    'TENSOR_HEALTH_HOOK_OWNER_CHANGED')
        group_provider = getattr(session, 'provider', None)
        group = getattr(group_provider, 'group', None)
        rows = getattr(group_provider, 'rows', [])
        current_row = rows[-1] if rows else None
        pending_matches = self.pending is not None and current_row is not None and (
            current_row.get('request_id'), current_row.get('phase')) == self.pending
        # An enqueued OFF/health request cannot borrow the preceding F values.
        # The final health request has its own receipt rather than a mechanism
        # row; after completion it also has no current mechanism effect.
        health = self.reports.get('same_worker_healthy')
        health_without_row = self.pending is None and health is not None and (
            current_row is None or current_row.get('request_id') != health.get('request_id'))
        current_row = current_completed_row(current_row,
            pending=self.pending is not None or health_without_row,
            pending_matches_latest=pending_matches)
        last_fault_row = next((row for row in reversed(rows) if any(event.get('applied') is True
            for event in (row.get('mechanism') or {}).get('events', ()))), None)
        last_fault_effect = actual_request_effect(last_fault_row)
        last_fault_effect['historical'] = True
        error = self.error or getattr(session, 'request_error', None)
        return deepcopy(dict(execution_boundary='python_hook', state_source='OWNED_PYTHON_HOOKS',
            hook_state_available=True, owner=owner, generation=generation,
            hook_owned=bool(getattr(session, 'handles', []) or observer is not None or adapter is not None
                            or getattr(session, 'completion_handle', None) or self.health_hooks),
            native_loaded=False, native_registered=False, registered=self._registered,
            binding_reserved=self.binding_reserved, enabled=state['enabled'],
            inflight=max(state['inflight'], int(getattr(session, 'current', None) is not None
                or getattr(session, 'root_active', False) or getattr(session, 'callbacks_active', 0)
                or self.health_current is not None or self.health_callbacks)), leases=state['leases'],
            replay_pending=bool(self.attempted and not self.stopped), replay_failed=bool(error),
            mechanism=self.mechanism, constructed_replay_fallback=False,
            current_request_effect=actual_request_effect(current_row), last_fault_effect=last_fault_effect,
            serving=dict(next_phase=('recovery_' + self.recovering if self.recovering else
                REQUESTS[self.index][0] if self.index < len(REQUESTS) else 'complete'),
                pending_request=self.pending, complete=self.complete, recovered=self.recovered,
                error=error, failures=self.failures, provenance=getattr(session, 'provenance', None),
                last_request=rows[-1] if rows else None,
                group=dict(session_id=group.session_id, owner=group.owner, generation=group.generation,
                    targets=len(group.group.leases), trigger=group.group.trigger) if group else None),
            coverage=self.reports, public_qualification_granted=False,
            C1='NOT_SEPARATELY_ASSESSED', E5='NOT_ASSESSED_SEPARATE_RAW_STATISTICS_REQUIRED'))
