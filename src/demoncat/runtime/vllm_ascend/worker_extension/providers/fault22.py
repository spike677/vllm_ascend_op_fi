"""Worker RPC bridge to the existing natural RoPE group and CoverageSession.

Only request sequencing lives here. Runtime owns the public lease; the existing
group owns discovery, admission, proof authority and recovery. No model replay
or serving lifecycle method is invoked by an RPC.
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
from ..serving22 import RoPE22ServingBinding, _ATTRIBUTE
from ..serving_context import snapshot_fresh_prefill


def require(condition, reason):
    if not condition:
        raise ProviderError(reason)


_REQUESTS = (('native_baseline', None), ('baseline', 'baseline'),
    ('baseline_repeat', 'baseline'), ('lossless', 'baseline'), ('c2_fault', 'fault'),
    ('c2_off', None), ('c3_fault_0', 'fault'), ('c3_fault_1', 'fault'), ('c3_off', None),
    ('cleanup_healthy', None))


class Fault22ServingProvider(Scope):
    execution_boundary = 'python_hook'

    def __init__(self, worker, model, identity):
        super().__init__(worker, model, identity)
        self.runner, self.thread = worker.model_runner, threading.get_ident()
        self.binding = self.session = self.proof = None
        self.checked = self.attempted = self.stopped = self.complete = False
        self.recovering = self.recovered = False
        self.index, self.pending = 0, None
        self.error = None
        self.failures, self.reports = [], {}
        self.health_hooks, self.health_current = [], None

    def _identity(self):
        # Runtime passes this exact identity after reading the current Worker
        # on every RPC. Between RPCs check the held objects and actual local
        # execution facts, without recursively calling demoncat_worker_identity.
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
                'ROPE_HELD_WORKER_CONTEXT_CHANGED')
        return dict(status='WORKER_IDENTIFIED', worker=deepcopy(value), quarantined=False)

    def _context(self):
        self._identity()
        if self.binding is not None:
            require(getattr(self.worker, _ATTRIBUTE, None) is self.binding
                    and self.binding.worker is self.worker and self.binding.runner is self.runner
                    and self.binding.model is self.model and self.binding.pid == os.getpid()
                    and self.binding.thread == self.thread, 'ROPE_DEPLOYMENT_BINDING_CHANGED')
            if self._registered:
                require(self.binding.leases == {self._ticket}, 'ROPE_DEPLOYMENT_OWNER_CHANGED')

    def dispatch(self, action, settings, *, owner, generation):
        if action == 'prepare':
            self.bind(owner, generation)
        self.verify(owner, generation)
        reject_unknown(settings, ('request_id',) if action == 'advance' else ())
        self._context()
        if action == 'prepare':
            binding = getattr(self.worker, _ATTRIBUTE, None)
            require(type(binding) is RoPE22ServingBinding, 'REVIEWED_ROPE_SERVING_BINDING_REQUIRED')
            self.binding = binding
            self._context()
            require(not binding.leases, 'ROPE_DEPLOYMENT_ALREADY_OWNED')
        elif action == 'register':
            require(self.binding is not None and not self.binding.leases, 'ROPE_DEPLOYMENT_ALREADY_OWNED')
            self.binding.leases.add(self._ticket)
            self._registered = True
        elif action == 'check':
            require(self._registered and self.model.training is False, 'ROPE_REGISTERED_INFERENCE_REQUIRED')
            self.checked = True
        elif action in ('execute', 'enable'):
            require(self.checked and not self.attempted, 'ROPE_SINGLE_ATTEMPT_REQUIRED')
            self.attempted = True
            from fault_injection.rope_serving import ServingRoPEGroup
            binding = self.binding
            # Keep a partially initialized object reachable for owned cleanup.
            self.session = ServingRoPEGroup.__new__(ServingRoPEGroup)
            self.session.__init__(self.worker, prompt=binding.prompt,
                weight_manifest=json.loads(binding.weight_manifest_json),
                workload_sha256=binding.workload_sha256, environment_sha256=binding.environment_sha256,
                _simulation=binding.simulation, _synchronize=binding.synchronize,
                _trace_factory=binding.trace_factory, _identity_guard=self._identity, _defer_to_forward=True)
            self.session.provider.owner = owner
            self.session.provider.generation = generation - 1
            source_files = (Path(__file__), Path(__file__).parents[1] / 'serving22.py')
            binding_sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
            group = self.session.provider
            group.source_sha = _sha(dict(group=group.source_sha, worker_rpc_binding=binding_sources))
            group.scope = replace(group.scope, implementation_sha256=group.source_sha)
            self.reports['deployment'] = dict(binding_sources=binding_sources,
                worker_rpc_owner=owner, worker_rpc_generation=generation,
                qualification_granted=False, simulation=binding.simulation)
        elif action == 'advance':
            require(self.attempted and not self.stopped, 'ROPE_ACTIVE_SERVING_ATTEMPT_REQUIRED')
            self._complete_request()
            request_id = settings.get('request_id')
            require(isinstance(request_id, str) and request_id and request_id not in self.session.seen,
                    'FRESH_REAL_ENQUEUED_REQUEST_REQUIRED')
            require(self.recovering or self.index < len(_REQUESTS), 'ROPE_PROTOCOL_ALREADY_COMPLETE')
            phase, arm = ('recovery_off', None) if self.recovering else _REQUESTS[self.index]
            self.pending = (request_id, phase)
            if phase == 'cleanup_healthy':
                self._schedule_health(request_id)
            else:
                self.session.expect(request_id, phase, arm)
        elif action == 'finish':
            self._complete_request()
            require(self.complete or self.recovered, 'ROPE_REAL_REQUEST_PROTOCOL_NOT_FINISHED')
            self._cleanup()
        elif action == 'recover':
            self._start_recovery()
        elif action in ('disable', 'abort', 'clean', 'unregister'):
            self._cleanup()
            if action in ('clean', 'unregister'):
                if self.binding is not None and action == 'unregister':
                    self.binding.leases.discard(self._ticket)
                self._registered = False
        else:
            raise ProviderError('UNSUPPORTED_PROVIDER_ACTION')
        return self.query(owner=owner, generation=generation)

    def _complete_request(self):
        if self.pending is None:
            return
        request_id, phase = self.pending
        require(not self.error and not self.session.request_error, 'ROPE_REQUEST_FAILED_RECOVERY_REQUIRED')
        require(request_id in self.session.seen and self.session.expected is None
                and self.session.deferred_request is None and self.health_current is None,
                'ROPE_REAL_ENQUEUED_REQUEST_NOT_RETURNED')
        if phase == 'lossless':
            self.reports['discovery'] = self.session.qualify()
            require(self.session.provider.mechanism == '22.wrong_row.fused_positions',
                    'ACTUAL_FUSED_ROPE_VARIANT_REQUIRED')
            self.session.provider.open_group('once', forwards=1)
        elif phase == 'c2_off':
            report = self.session.finish_group()
            require(report['contract_satisfied'], 'ROPE_C2_CONTRACT_NOT_COMPLETE')
            self.proof = self.session.provider.group.c2_proof()
            report['live_c2_proof'] = self.session.provider.group.report()['live_c2_proof']
            self.reports['C2'] = report
            self.session.provider.open_group('every_match', forwards=2, c2_proof=self.proof)
        elif phase == 'c3_off':
            report = self.session.finish_group()
            require(report['contract_satisfied'], 'ROPE_C3_CONTRACT_NOT_COMPLETE')
            self.reports['C3'] = report
            self.reports['cleanup'] = self.session.close()
        elif phase == 'recovery_off':
            report = self.session.finish_group()
            self.reports['failed_attempt_recovery'] = report
            require(report['recovery']['complete'], 'ROPE_GROUP_RECOVERY_UNCONFIRMED')
            self.reports['cleanup'] = self.session.close()
            self.recovering, self.recovered, self.stopped = False, True, True
        self.pending = None
        if phase != 'recovery_off':
            self.index += 1
            self.complete = self.index == len(_REQUESTS)

    def _remove_health_hooks(self):
        for handle, callback in reversed(self.health_hooks[:]):
            hooks = handle.hooks_dict_ref()
            require(hooks is not None and (handle.id not in hooks or hooks[handle.id] is callback),
                    'ROPE_HEALTH_HOOK_OWNER_CHANGED')
            try:
                handle.remove()
            finally:
                if handle.id not in hooks:
                    self.health_hooks.remove((handle, callback))
            require(handle.id not in hooks, 'ROPE_HEALTH_HOOK_REMOVAL_UNCONFIRMED')

    def _schedule_health(self, request_id):
        session = self.session
        require(not session.handles and session.adapter is None and session.trace is None
                and not self.health_hooks and session.provider.group.recovery_report()['complete'],
                'ROPE_OWNED_CLEANUP_REQUIRED_BEFORE_HEALTH')

        def guarded(callback):
            def call(*args):
                try:
                    self._context()
                    return callback(*args)
                except BaseException as error:
                    self.error = type(error).__name__ + ': ' + str(error)
                    raise
            return call

        @guarded
        def pre(module, args, kwargs):
            from vllm.forward_context import get_forward_context
            require(self.health_current is None, 'ROPE_HEALTH_FORWARD_ALREADY_STARTED')
            actual = session.binding.bind(module, args, kwargs)
            scheduler = snapshot_fresh_prefill(self.runner, self.model, actual, get_forward_context(),
                expected_prompt_token_ids=session.prompt, expected_internal_request_id=request_id,
                seen_internal_request_ids=session.seen)
            self.health_current = dict(input=_capture((args, kwargs)), scheduler=scheduler,
                                       entry_error=sys.exc_info()[1])

        @guarded
        def post(module, args, kwargs, output):
            require(self.health_current is not None and output is not None, 'ROPE_HEALTH_ROOT_NOT_RETURNED')
            require(sys.exc_info()[1] is self.health_current['entry_error']
                    and next(reversed(module._forward_hooks.values())) is post,
                    'ROPE_HEALTH_ROOT_RETURN_NOT_CONFIRMED')
            self.health_current['hidden'] = _capture(output)

        @guarded
        def logits(module, args, kwargs, output):
            current = self.health_current
            require(current is not None and 'hidden' in current and output is not None,
                    'ROPE_HEALTH_LOGITS_NOT_RETURNED')
            require(sys.exc_info()[1] is current['entry_error']
                    and next(reversed(module._forward_hooks.values())) is logits,
                    'ROPE_HEALTH_LOGITS_RETURN_NOT_CONFIRMED')
            result = dict(hidden=current['hidden'], logits=_capture(output))
            session.provider.synchronize()
            require(_same(current['input'], session.provider.reference_input)
                    and _same(result, session.provider.reference_output), 'ROPE_CLEANUP_HEALTH_NOT_EXACT')
            self._remove_health_hooks()
            self.reports['same_worker_healthy'] = dict(request_id=request_id, worker=self.identity,
                input=_identity(current['input']), output=_identity(result), scheduler=current['scheduler'],
                cleanup_before_request=True, full_hidden_logits_exact=True, fault_hooks=0,
                qualification_granted=False)
            session.seen.append(request_id)
            self.health_current = None

        for module, callback, before in ((self.model, pre, True), (self.model, post, False),
                                        (session.processor, logits, False)):
            register = module.register_forward_pre_hook if before else module.register_forward_hook
            handle = register(callback, with_kwargs=True, **({} if before else {'always_call': True}))
            self.health_hooks.append((handle, callback))

    def _owned_abort(self):
        self._remove_health_hooks()
        self.health_current = None
        session = self.session
        if (session is not None and hasattr(session, 'handles')
                and (session.handles or session.expected is not None or session.deferred_request is not None
                     or session.trace is not None or session.adapter is not None or session.completion_handle is not None)):
            failure = session.abort(retain_recovery=True)
            self.failures.append(failure)
        self.pending = None

    def _start_recovery(self):
        self._owned_abort()
        group = getattr(getattr(self.session, 'provider', None), 'group', None)
        if group is None or group.recovery_report()['complete']:
            self._cleanup()
            return
        if self.session.provider.off_row is not None:
            self._cleanup()
            self.recovered = True
            return
        self.session.request_error = self.error = None
        self.recovering, self.stopped = True, False

    def _cleanup(self):
        self._owned_abort()
        session = self.session
        if session is not None and hasattr(session, 'handles'):
            group = session.provider.group
            if group is not None and not group.recovery_report()['complete'] and session.provider.off_row is not None:
                self.reports['cleanup_recovery'] = session.finish_group()
            require(group is None or group.recovery_report()['complete'], 'ROPE_REAL_OFF_REQUEST_REQUIRED')
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
            require(state['ownership_verified'], 'ROPE_DISPATCH_OWNERSHIP_CHANGED')
        if session is not None and hasattr(session, 'handles'):
            session._context(verify_targets=False)
        for handle, callback in self.health_hooks:
            require(handle.hooks_dict_ref() is not None
                    and handle.hooks_dict_ref().get(handle.id) is callback, 'ROPE_HEALTH_HOOK_OWNER_CHANGED')
        error = self.error or getattr(session, 'request_error', None)
        hooks = bool(getattr(session, 'handles', []) or getattr(session, 'trace', None)
                     or adapter is not None or getattr(session, 'completion_handle', None) or self.health_hooks)
        return dict(execution_boundary=self.execution_boundary, state_source='OWNED_PYTHON_HOOKS',
            hook_state_available=True, owner=owner, generation=generation, hook_owned=hooks,
            native_loaded=False, native_registered=False, registered=self._registered,
            enabled=state['enabled'], inflight=max(state['inflight'],
                int(getattr(session, 'current', None) is not None or self.health_current is not None)),
            leases=state['leases'],
            replay_pending=bool(self.attempted and not self.stopped), replay_failed=bool(error),
            mechanism='22.wrong_row', selected_variant=('22.wrong_row.fused_positions'
                if 'discovery' in self.reports else None), legacy_lookup_fallback=False,
            serving=dict(next_phase=('recovery_off' if self.recovering else
                _REQUESTS[self.index][0] if self.index < len(_REQUESTS) else 'complete'),
                pending_request=self.pending, complete=self.complete, recovered=self.recovered,
                error=error, failures=self.failures),
            coverage=self.reports, public_qualification_granted=False,
            E5='NOT_ASSESSED_SEPARATE_RAW_STATISTICS_REQUIRED')
