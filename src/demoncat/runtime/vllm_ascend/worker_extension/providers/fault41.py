"""Public Worker tickets over the existing natural KV41 request/group protocol.

RPCs never execute a model or manufacture writer/cache/consumer evidence. Native
span ownership and the all-target recovery barrier remain in the original code.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import threading

from ._scope import Scope, ProviderError, reject_unknown
from .. import identity as identity_reader
from ..serving41 import KV41ServingBinding, _ATTRIBUTE, read_native_binding


def require(condition, reason):
    if not condition:
        raise ProviderError(reason)


_REQUESTS = (
    ('c1_observe_0', 'observe'), ('c1_observe_1', 'observe'), ('c1_lossless', 'lossless'),
    ('c1_fault', 'fault'), ('c1_off', 'off'),
    ('group_observe_0', 'group_observe'), ('group_observe_1', 'group_observe'),
    ('c2_lossless', 'group_lossless'), ('c2_fault_0', 'group_fault'), ('c2_fault_1', 'group_fault'),
    ('c2_off', 'group_off'), ('c2_health', 'health'),
    ('c3_lossless', 'group_lossless'), ('c3_fault_0', 'group_fault'), ('c3_fault_1', 'group_fault'),
    ('c3_off', 'group_off'), ('c3_health', 'health'))


class Fault41ServingProvider(Scope):
    execution_boundary = 'native'

    def __init__(self, worker, model, identity):
        super().__init__(worker, model, identity)
        self.binding = getattr(worker, _ATTRIBUTE, None)
        # The resolver remains inert. Missing deployment cannot create an empty
        # Native owner for which the Runtime could never obtain ABI readback.
        require(type(self.binding) is KV41ServingBinding, 'REVIEWED_KV41_SERVING_BINDING_REQUIRED')
        self.runner, self.thread = worker.model_runner, threading.get_ident()
        self.p = self.group = self.proof = None
        self.native_owner = None
        self.checked = self.attempted = self.stopped = self.complete = self.recovered = False
        self.recovering = None
        self.index, self.pending = 0, None
        self.seen, self.requests, self.failures, self.reports = set(), [], [], {}
        self.error = None
        self.binding_reserved = False
        self.root_active = self.health_active = False
        self.root_unwound = self.health_unwound = False
        self.cleaning = False
        self._context()
        read_native_binding(self.binding)

    def _context(self):
        value, binding = self.identity, self.binding
        visible, source = identity_reader._visible_devices()
        device = getattr(self.worker, 'device', None)
        require(os.getpid() == value['pid'] and threading.get_ident() == self.thread
            and self.worker.model_runner is self.runner and self.runner.model is self.model
            and self.worker.rank == value['rank'] and getattr(device, 'type', None) == 'npu'
            and getattr(device, 'index', None) == value['configured_logical_device']
            and identity_reader._current_device() == value['logical_device']
            and value['observed_logical_device'] == value['logical_device']
            and visible == value['visible_devices'] and source == value['visible_devices_source'],
            'KV41_HELD_WORKER_CONTEXT_CHANGED')
        require(getattr(self.worker, _ATTRIBUTE, None) is binding and binding.worker is self.worker
            and binding.runner is self.runner and binding.model is self.model
            and binding.pid == os.getpid() and binding.thread == self.thread, 'KV41_DEPLOYMENT_BINDING_CHANGED')
        if self.binding_reserved or binding.leases:
            require(binding.leases == {self._ticket}, 'KV41_DEPLOYMENT_OWNER_CHANGED')
        if self.p is not None:
            require(self.p.worker is self.worker and self.p.model is self.model and self.p.runner is self.runner
                and self.p.transport is binding.transport and self.p.owner == self.native_owner,
                'KV41_NATIVE_OR_MODEL_OWNER_CHANGED')
        if self.group is not None:
            require(self.group.p is self.p and self.group.owner == self._ticket[0], 'KV41_GROUP_OWNER_CHANGED')

    def _failure(self, error, source):
        message = type(error).__name__ + ': ' + str(error)
        if self.error is None:
            self.error = message
        self.failures.append(dict(source=source, reason=message, pending_request=self.pending))

    def _wrap_callback(self, kind, callback):
        provider, epoch = self.p, self.p.install_epoch
        health = kind.startswith('health_')
        health_request = self.pending if health else None
        def wrapped(*args, **kwargs):
            before = kind in ('root_pre', 'health_pre')
            after = kind in ('root_post', 'health_post')
            active = 'health_active' if health else 'root_active'
            unwound = 'health_unwound' if health else 'root_unwound'
            try:
                self._context()
                require(self._registered and self.p is provider and provider.install_epoch == epoch,
                        'KV41_LATE_OR_FOREIGN_CALLBACK')
                if health:
                    require(self.pending == health_request and self.group.stage == 'health_pending'
                        and not provider.installed, 'KV41_STALE_HEALTH_CALLBACK')
                else:
                    from .. import postwrite41
                    require(provider.installed and postwrite41._ACTIVE is provider, 'KV41_STALE_INSTALLED_CALLBACK')
                if kind not in ('root_post', 'attention_post', 'health_post'):
                    require(not self.error, 'KV41_FAILED_CALLBACK_REQUIRES_RECOVERY')
                if before:
                    require(not getattr(self, active), 'KV41_CONCURRENT_ROOT_CALLBACK')
                    setattr(self, active, True)
                    setattr(self, unwound, False)
                    require(next(reversed(args[0]._forward_pre_hooks.values())) is wrapped,
                            'KV41_FINAL_ROOT_INPUT_OBSERVER_REQUIRED')
                elif after:
                    require(getattr(self, active), 'KV41_ROOT_POST_WITHOUT_ACTUAL_PRE')
                    require(next(reversed(args[0]._forward_hooks.values())) is wrapped,
                            'KV41_FINAL_ROOT_RETURN_OBSERVER_REQUIRED')
                elif provider.current is not None:
                    require(self.root_active, 'KV41_CALLBACK_AFTER_ROOT_UNWIND')
                return callback(*args, **kwargs)
            except BaseException as error:
                self._failure(error, kind)
                raise
            finally:
                if (after and self.p is provider and provider.install_epoch == epoch
                        and threading.get_ident() == self.thread
                        and (not health or self.pending == health_request)):
                    setattr(self, active, False)
                    setattr(self, unwound, True)
        return wrapped

    def dispatch(self, action, settings, *, owner, generation):
        if action == 'prepare':
            self.bind(owner, generation)
        self.verify(owner, generation)
        reject_unknown(settings, ('request_id',) if action == 'advance' else ())
        self._context()
        try:
            if action == 'prepare':
                require(not self.binding.leases, 'KV41_DEPLOYMENT_ALREADY_OWNED')
            elif action == 'register':
                require(not self.binding.leases, 'KV41_DEPLOYMENT_ALREADY_OWNED')
                self.binding.leases.add(self._ticket)
                self.binding_reserved = True
                self._registered = True
            elif action == 'check':
                require(self._registered and self.model.training is False, 'KV41_REGISTERED_INFERENCE_REQUIRED')
                read_native_binding(self.binding)
                self.checked = True
            elif action in ('execute', 'enable'):
                require(self.checked and not self.attempted, 'KV41_SINGLE_ATTEMPT_REQUIRED')
                self.attempted = True
                from ..postwrite41 import Postwrite41WorkerProvider
                from fault_injection.deployments.postwrite_evidence import digest
                factory = self.binding.provider_factory if self.binding.simulation else Postwrite41WorkerProvider
                extra={'group_context':self.binding.group_context} if self.binding.group_context is not None else {}
                self.p = factory(self.worker, self.binding.transport, tail_elements=2, **extra)
                require(type(self.p) is Postwrite41WorkerProvider and not self.p.installed,
                        'KV41_EXISTING_UNINSTALLED_PROVIDER_REQUIRED')
                self.native_owner = self.p.owner
                self.p._core_callback_wrapper = self._wrap_callback
                sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
                           (Path(__file__), Path(__file__).parents[1] / 'serving41.py')}
                self.scope = replace(self.binding.scope, implementation_sha256=digest(dict(
                    reviewed_implementation=self.binding.scope.implementation_sha256, worker_rpc_binding=sources)))
                self.reports['deployment'] = dict(binding_sources=sources, worker_rpc_owner=owner,
                    worker_rpc_generation=generation, native_owner=self.native_owner,
                    native_sha256=self.binding.native_sha256, simulation=self.binding.simulation,
                    qualification_granted=False)
                self.p.install()
            elif action == 'advance':
                require(self.attempted and not self.stopped, 'KV41_ACTIVE_SERVING_ATTEMPT_REQUIRED')
                self._complete_request()
                require(not self.error and self.pending is None, 'KV41_FAILED_REQUEST_REQUIRES_RECOVERY')
                request_id = settings.get('request_id')
                require(isinstance(request_id, str) and 0 < len(request_id) <= 256 and request_id not in self.seen,
                        'FRESH_REAL_ENQUEUED_REQUEST_REQUIRED')
                if self.recovering:
                    label, phase = 'recovery_' + self.recovering, {
                        'c1_off': 'off', 'group_off': 'group_off', 'health': 'health'}[self.recovering]
                else:
                    require(self.index < len(_REQUESTS), 'KV41_PROTOCOL_ALREADY_COMPLETE')
                    label, phase = _REQUESTS[self.index]
                self.pending = (request_id, label, phase)
                self.seen.add(request_id)
                self.cleaning = False
                if phase == 'health':
                    self.group.begin_health(request_id, self.binding.prompt)
                else:
                    self.p.expect(phase, request_id, self.binding.prompt)
            elif action == 'finish':
                self._complete_request()
                require(self.complete or self.recovered, 'KV41_REAL_REQUEST_PROTOCOL_NOT_FINISHED')
                self._cleanup()
            elif action == 'recover':
                self._start_recovery()
            elif action in ('disable', 'abort', 'clean', 'unregister'):
                self._cleanup()
                if action in ('clean', 'unregister'):
                    if action == 'unregister':
                        self.binding.leases.discard(self._ticket)
                        self.binding_reserved = False
                    self._registered = False
            else:
                raise ProviderError('UNSUPPORTED_PROVIDER_ACTION')
        except BaseException as error:
            self._failure(error, 'rpc_' + action)
            raise
        return self.query(owner=owner, generation=generation)

    def _complete_request(self):
        if self.pending is None:
            return
        require(not self.error, 'KV41_REQUEST_FAILED_RECOVERY_REQUIRED')
        request_id, label, phase = self.pending
        require(not self.root_active and not self.health_active, 'KV41_REAL_REQUEST_STILL_ACTIVE')
        if phase == 'health':
            require(self.group.health_artifact is not None and self.group.health_current is None,
                    'KV41_REAL_HEALTH_REQUEST_NOT_RETURNED')
            record = self.group.finish_health()
        else:
            require(self.p.current is None and self.p.pending_run is not None
                and request_id in self.p.pending_run['request_ids'], 'KV41_REAL_ENQUEUED_REQUEST_NOT_RETURNED')
            record = self.p.finish_request()
        self.requests.append(dict(request_id=request_id, phase=label, actual=deepcopy(record)))
        if label == 'c1_off':
            from ..postwrite41_coverage import KVGroupCoverage
            self.reports['C1'] = deepcopy(self.p.history)
            self.group = KVGroupCoverage(self.p, self.scope, max_seconds=self.binding.max_seconds, requests=2)
            self.group.owner = self._ticket[0]
        elif label == 'group_observe_1':
            self.group.prepare('once')
        elif label in ('c2_off', 'c3_off', 'recovery_group_off'):
            self.reports[label + '_drain'] = self.group.begin_recovery()
            if label == 'recovery_group_off':
                self.recovering = 'health'
        elif label == 'c2_health':
            require(record['contract_satisfied'] and self.group.c2_live_proof is not None,
                    'KV41_C2_CONTRACT_AND_ORIGINAL_PROOF_REQUIRED')
            self.proof = self.group.c2_live_proof
            self.reports['C2'] = deepcopy(record)
            self.group.prepare('every_match')
        elif label == 'c3_health':
            require(record['contract_satisfied'], 'KV41_C3_CONTRACT_NOT_COMPLETE')
            self.reports['C3'] = deepcopy(record)
        elif label == 'recovery_health':
            require(record['recovery']['complete'], 'KV41_GROUP_RECOVERY_UNCONFIRMED')
            self.reports['failed_attempt_recovery'] = deepcopy(record)
            self.recovering, self.recovered, self.stopped = None, True, True
        elif label == 'recovery_c1_off':
            self.p.close()
            self.reports['failed_c1_off'] = deepcopy(record)
            self.recovering, self.recovered, self.stopped = None, True, True
        self.pending = None
        if not label.startswith('recovery_'):
            self.index += 1
            self.complete = self.index == len(_REQUESTS)

    def _cancel_returned_request(self):
        require(not self.root_active and not self.health_active, 'KV41_ACTUAL_FORWARD_STILL_ACTIVE')
        if self.p is None:
            return
        if self.p.current is not None:
            require(self.root_unwound, 'KV41_ROOT_UNWIND_UNCONFIRMED')
            require(self.p.lease is None or not self.p.lease.consuming, 'KV41_NATIVE_CONSUMER_STILL_ACTIVE')
            self.p.current = None
        if self.group is not None and self.group.health_current is not None:
            require(self.health_unwound, 'KV41_HEALTH_UNWIND_UNCONFIRMED')
            self.group.health_current = None
        self.p.recover()
        self.p.workload.abort('PUBLIC_CORE_REQUEST_ABORTED')
        if self.p.pending_run is not None:
            self.reports.setdefault('failed_requests', []).append(deepcopy(self.p.pending_run))
        self.p.stack.clear()
        self.p.expected = self.p.phase = self.p.pending_run = self.p.collector = None
        self.pending = None

    def _c1_off_pending(self):
        phases = [row['phase'] for row in self.p.history if row['phase'] in ('fault', 'off')]
        return self.p.fault_attempted and (not phases or phases[-1] != 'off')

    def _start_recovery(self):
        self._cancel_returned_request()
        group = self.group
        if group is not None and group.session is not None and not group.session.recovery_report()['complete']:
            if group.stage in ('off_complete', 'recovery_pending'):
                group.begin_recovery()
            if group.stage == 'health_pending':
                group.close_health_observer()
                self.recovering = 'health'
            else:
                group.abort_request(self.error or 'PUBLIC_CORE_RECOVERY_REQUESTED')
                self.recovering = 'group_off'
        elif self.p is not None and self._c1_off_pending():
            self.recovering = 'c1_off'
        else:
            self._cleanup()
            self.recovered = True
        self.error = None
        if self.recovering:
            self.stopped = False

    def _cleanup(self):
        self.cleaning = True
        # A completed request is consumed only by an action, never by query.
        if self.pending is not None and not self.error:
            request_id, _, phase = self.pending
            returned = (self.group is not None and self.group.health_artifact is not None) if phase == 'health' else (
                self.p.pending_run is not None and 'model_output_sha256' in self.p.pending_run)
            if returned:
                self._complete_request()
        self._cancel_returned_request()
        if self.group is not None and self.group.session is not None:
            recovery = self.group.session.recovery_report()
            if not recovery['complete'] and self.group.stage in ('off_complete', 'recovery_pending'):
                self.group.begin_recovery()
            require(self.group.session.recovery_report()['complete'], 'KV41_REAL_GROUP_OFF_AND_HEALTH_REQUIRED')
            self.group.close_health_observer()
        if self.p is not None:
            require(not self._c1_off_pending(), 'KV41_REAL_C1_OFF_REQUEST_REQUIRED')
            self.p.close()
            require(not self.p.handles and not self.p.patches and self.p.lease is None and not self.p.installed,
                    'KV41_OWNED_CLEANUP_UNCONFIRMED')
        self.error = None
        self.recovering, self.stopped = None, True

    def query(self, *, owner, generation):
        self.verify(owner, generation)
        self._context()
        native = read_native_binding(self.binding)
        p, group = self.p, self.group
        hook_count, api_state, lease_state = 0, dict(read_only=True, owned_links_present=0), None
        if p is not None:
            for handle in p.handles:
                hooks = handle.hooks_dict_ref()
                require(hooks is not None and (handle.id not in hooks and self.cleaning
                    or hooks.get(handle.id) is p.hook_owners[handle.id]), 'KV41_OWNED_HOOK_READBACK_FAILED')
                hook_count += int(handle.id in hooks)
            from ..postwrite_collector import restore_owned_api_patches
            api_state = restore_owned_api_patches(p.patches, collectors=p.collectors, read_only=True)
            require(self.cleaning or self.error or not p.installed
                or bool(p.patches) and all(row['owned_links_present'] for row in api_state['patches']),
                'KV41_OWNED_API_LINK_MISSING')
            if p.lease is not None:
                lease = p.lease
                require(lease.transport is self.binding.transport and lease.owner == self.native_owner,
                        'KV41_NATIVE_LEASE_OWNER_CHANGED')
                expected_generation = group.generation if group is not None and p.phase and p.phase.startswith('group_') else p.generation
                require(lease.generation == expected_generation, 'KV41_NATIVE_LEASE_GENERATION_CHANGED')
                handle = lease.native
                if handle is not None:
                    require(handle.owner == lease.owner and handle.generation == lease.generation,
                            'KV41_NATIVE_HANDLE_OWNER_CHANGED')
                live = False if handle is None else handle.live()
                if live:
                    handle._check()  # Metadata/owner check only; never restore during query.
                lease_state = dict(owner=lease.owner, generation=lease.generation, native_live=live,
                    retained=True, consuming=lease.consuming, receipt=deepcopy(lease.report),
                    native_release_readback_error=getattr(handle, 'release_readback_error', None))
        if group is not None:
            for handle in group.health_handles:
                hooks = handle.hooks_dict_ref()
                require(hooks is not None and hooks.get(handle.id) is group.health_hook_owners[handle.id],
                        'KV41_HEALTH_HOOK_READBACK_FAILED')
                hook_count += 1
        inflight = int(self.root_active or self.health_active or p is not None and p.current is not None
                       or group is not None and group.health_current is not None)
        held = len(group.session.recovery_report()['held_leases']) if group is not None and group.session is not None else 0
        return dict(execution_boundary='native', state_source='NATIVE_ABI_OWNER_LEASE_AND_OWNED_HOST_READBACK',
            native_loaded=True, native_registered=False, native_registration='NOT_APPLICABLE_DIRECT_ABI_LEASES',
            registered=self._registered, owner=owner, generation=generation, enabled=bool(lease_state and lease_state['native_live']),
            inflight=inflight, leases=int(p is not None and p.lease is not None),
            native=native, native_lease=lease_state, owned_hooks=hook_count, owned_api_state=api_state,
            coverage_held_leases=held, replay_pending=bool(self.attempted and not self.stopped), replay_failed=bool(self.error),
            mechanism='41.tail_column', boundary='actual_writer_live_value_slot_tnd_fia',
            controlled_legal_address_proxy=True, native_out_of_bounds_claim=False,
            serving=dict(next_phase=('recovery_' + self.recovering if self.recovering else
                _REQUESTS[self.index][0] if self.index < len(_REQUESTS) else 'complete'),
                pending_request=self.pending, complete=self.complete, recovered=self.recovered,
                error=self.error, failures=deepcopy(self.failures), actual_requests=deepcopy(self.requests)),
            provider=p.report() if p is not None else None, group=group.report() if group is not None else None,
            coverage=deepcopy(self.reports), public_qualification_granted=False,
            E5='NOT_ASSESSED_SEPARATE_RAW_STATISTICS_REQUIRED')
