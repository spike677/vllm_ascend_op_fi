"""Read-only, single-request census attached to an already loaded Worker.

This is a deployment helper, not an RPC deserializer or a Worker launcher.
The runtime stores this object in its owner table BEFORE calling start().
Native observer factories are trusted host code, never a user RPC setting.
"""

from copy import deepcopy
import json
import os
import threading


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


class WorkerCallCensus:
    """Reuse the natural dispatch/path observers with retained cleanup ownership.

    Identity must come from the runtime's current IdentityTracker read. This
    helper checks the live Python objects and supplied identity without querying
    or initializing a device. Scheduler evidence is read only in a natural
    model pre-hook via the existing worker_request_probe.
    """

    def __init__(self, worker, identity, *, phase, expected, expected_max_seqs=1,
                 max_events=10000, native_observer_factory=None):
        if (not isinstance(identity, dict) or identity.get('pid') != os.getpid()
                or type(identity.get('rank')) is not int or identity['rank'] < 0
                or type(identity.get('logical_device')) is not int or identity['logical_device'] < 0
                or any(not isinstance(identity.get(key), str) or not identity[key]
                       for key in ('process_start_identity', 'worker_start_identity', 'model_instance_id'))):
            raise ValueError('CENSUS_ACTUAL_WORKER_IDENTITY_REQUIRED')
        if type(expected_max_seqs) is not int or expected_max_seqs not in (1, 8):
            raise ValueError('CENSUS_REVIEWED_SEQUENCE_CAPACITY_REQUIRED')
        if type(max_events) is not int or max_events < 1:
            raise ValueError('CENSUS_FINITE_EVENT_BUDGET_REQUIRED')
        if native_observer_factory is not None and not callable(native_observer_factory):
            raise ValueError('CENSUS_TRUSTED_NATIVE_FACTORY_REQUIRED')
        if not isinstance(expected, dict):
            raise ValueError('CENSUS_SINGLE_REQUEST_REQUIRED')
        if phase == 'prefill':
            if set(expected) != {'internal_request_id', 'prompt_token_ids'} or expected_max_seqs != 1:
                raise ValueError('CENSUS_SINGLE_PREFILL_REQUEST_REQUIRED')
            request_id, prompt = expected['internal_request_id'], expected['prompt_token_ids']
        elif phase == 'decode':
            if len(expected) != 1:
                raise ValueError('CENSUS_SINGLE_REQUEST_REQUIRED')
            request_id, binding = next(iter(expected.items()))
            if (not isinstance(binding, dict) or set(binding) != {'prompt', 'logical_id'}
                    or type(binding['logical_id']) not in (str, int)
                    or binding['logical_id'] == ''):
                raise ValueError('CENSUS_DECODE_REQUEST_BINDING_REQUIRED')
            prompt = binding['prompt']
        else:
            raise ValueError('CENSUS_REVIEWED_PREFILL_OR_FIRST_DECODE_REQUIRED')
        if (not isinstance(request_id, str) or not request_id or not isinstance(prompt, list)
                or not prompt or any(type(token) is not int or token < 0 for token in prompt)):
            raise ValueError('CENSUS_ENQUEUED_REQUEST_AND_PROMPT_REQUIRED')

        self.worker = worker
        self.identity = deepcopy(identity)
        self._identity_key = _canonical(identity)
        self.runner = getattr(worker, 'model_runner', None)
        self.model = getattr(self.runner, 'model', None)
        self._owner_thread = threading.get_ident()
        self.phase, self.expected = phase, deepcopy(expected)
        self.expected_max_seqs, self.max_events = expected_max_seqs, max_events
        self.request_id = request_id
        self._native_observer_factory = native_observer_factory
        self._capture = {}
        self._cm = None
        self._exit_attempted = False
        self._started = self._stopped = self._cleaned = False
        self._errors = []
        self._request_failures = []
        self._state = 'BOUND'
        self._require_binding(worker, identity)

    def _binding_error(self, worker, identity):
        if threading.get_ident() != self._owner_thread:
            return 'CENSUS_OWNER_THREAD_MISMATCH'
        if worker is not self.worker:
            return 'CENSUS_WORKER_OBJECT_MISMATCH'
        if (self.model is None or getattr(worker, 'model_runner', None) is not self.runner
                or getattr(self.runner, 'model', None) is not self.model):
            return 'CENSUS_RUNNER_OR_MODEL_OBJECT_MISMATCH'
        device = getattr(worker, 'device', None)
        if (type(getattr(worker, 'rank', None)) is not int or worker.rank != self.identity['rank']
                or getattr(device, 'type', None) != 'npu'
                or type(getattr(device, 'index', None)) is not int
                or device.index != self.identity['logical_device'] or os.getpid() != self.identity['pid']):
            return 'CENSUS_LIVE_WORKER_IDENTITY_MISMATCH'
        try:
            if _canonical(identity) != self._identity_key:
                return 'CENSUS_WORKER_GENERATION_MISMATCH'
        except (TypeError, ValueError, RecursionError):
            return 'CENSUS_WORKER_GENERATION_MISMATCH'
        return None

    def _require_binding(self, worker, identity):
        reason = self._binding_error(worker, identity)
        if reason:
            raise RuntimeError(reason)

    def _record_error(self, stage, error):
        self._errors.append(dict(stage=stage, error_type=type(error).__name__, reason=str(error)))

    def start(self):
        """Install observation only; the serving host submits the real request."""
        self._require_binding(self.worker, self.identity)
        if self._started or self._stopped:
            raise RuntimeError('CENSUS_CANNOT_BE_RESTARTED')
        self._started = True
        self._state = 'STARTING'
        try:
            from engineering_closure.native_observer import NativeObserver
            from engineering_closure.path_discovery import observe_execution_path, worker_request_probe
            from .serving_context import ContextRefused

            probe = worker_request_probe(self.runner, phase=self.phase, expected=self.expected,
                                         expected_max_seqs=self.expected_max_seqs)

            def request_at_forward(model, args, kwargs):
                reason = self._binding_error(self.worker, self.identity)
                try:
                    if reason:
                        raise ContextRefused(reason)
                    request = probe(model, args, kwargs)
                    if (request.get('request_ids') != [self.request_id]
                            or request.get('phase') != self.phase or not request.get('evidence')):
                        raise ContextRefused('CENSUS_SINGLE_ACTUAL_REQUEST_BINDING_REQUIRED')
                    return request
                except ContextRefused as error:
                    self._request_failures.append(error.code)
                    raise

            native = self._native_observer_factory(self.worker) if self._native_observer_factory else {}
            if (not isinstance(native, dict) or any(not isinstance(key, str) or not key
                    or type(value) is not NativeObserver for key, value in native.items())):
                raise ValueError('CENSUS_TYPED_READ_ONLY_NATIVE_OBSERVERS_REQUIRED')
            self._cm = observe_execution_path(self.model, request_at_forward=request_at_forward,
                worker_identity=self.identity, runner=self.runner, known_native_observers=native,
                max_events=self.max_events, capture=self._capture)
            self._cm.__enter__()
            self._state = 'OBSERVING'
        except BaseException as error:
            if self._cm is not None:
                self._exit_attempted = True
            self._state = 'START_FAILED'
            self._record_error('start', error)
            raise
        return self.query(self.worker, self.identity)

    def _resources(self):
        reader = self._capture.get('state')
        if reader is not None:
            return reader()
        # The observation capture installs its cleanup handle before any hook.
        return dict(inflight=False, observer_hooks=0, foreign_hooks=False,
                    metadata_scope_active=False, cleanup_complete=self._cleaned)

    def query(self, worker, identity):
        """Return readback; errors and partial evidence never confer eligibility."""
        reason = self._binding_error(worker, identity)
        resources = dict(self._resources(), stop_requested=self._stopped)
        report = self._capture.get('report')
        if (reason is None and not resources['inflight'] and self._capture.get('snapshot') is not None):
            try:
                self._capture['snapshot']()
                report = self._capture.get('report')
            except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as error:
                self._record_error('report', error)
        forwards = report.get('forwards', []) if report else []
        request_valid = bool(forwards) and not self._request_failures and all(
            row.get('request_status') == 'OBSERVED'
            and row.get('request', {}).get('request_ids') == [self.request_id]
            and row.get('request', {}).get('phase') == self.phase for row in forwards)
        observation_complete = bool(report and report.get('observation_complete'))
        native_failed = bool(report and report.get('native_windows_complete') is False)
        resource_clean = (resources['cleanup_complete'] is True and not resources['inflight']
            and resources['observer_hooks'] == 0 and not resources['foreign_hooks']
            and not resources['metadata_scope_active'])
        failed = bool(self._request_failures or self._errors or native_failed or
                      (self._stopped and not (request_valid and observation_complete)))
        return dict(schema='demoncat.worker.call_census.v1', state=self._state,
            worker=deepcopy(self.identity), worker_object_id=id(self.worker),
            runner_object_id=id(self.runner), model_object_id=id(self.model),
            identity_valid=reason is None, identity_error=reason,
            **resources,
            resource_cleanup_confirmed=resource_clean,
            releasable=reason is None and self._stopped and resource_clean,
            request_binding_valid=request_valid,
            native_observation_complete=report.get('native_windows_complete') if report else None,
            observation_status='FAILED' if failed else ('OBSERVED' if request_valid and observation_complete else 'PENDING'),
            qualification_granted=False, tensor_mutations=0,
            request_failures=list(self._request_failures), errors=deepcopy(self._errors),
            capture_errors={key: self._capture[key] for key in
                ('workload_error', 'workload_error_reason', 'report_error') if key in self._capture},
            report=deepcopy(report))

    def stop(self, worker, identity):
        """Stop accepting new forwards; never block or force an active call out."""
        self._require_binding(worker, identity)
        self._stopped = True
        try:
            if self._capture.get('stop') is not None:
                self._capture['stop']()
            self._state = 'DRAINING' if self._resources()['inflight'] else 'STOPPED'
        except BaseException as error:
            self._state = 'STOP_FAILED'
            self._record_error('stop', error)
            raise
        return self.query(worker, identity)

    def cleanup(self, worker, identity):
        """Retry on the owner thread; release only after query confirms removal."""
        self._require_binding(worker, identity)
        self.stop(worker, identity)
        if self._resources()['inflight']:
            self._state = 'DRAINING'
            raise RuntimeError('CENSUS_FORWARD_STILL_INFLIGHT')
        try:
            if self._cm is not None and not self._exit_attempted:
                self._exit_attempted = True
                self._cm.__exit__(None, None, None)
            elif self._capture.get('cleanup') is not None:
                self._capture['cleanup']()
            self._cleaned = True
            if self._resources()['cleanup_complete'] is not True:
                raise RuntimeError('CENSUS_CLEANUP_UNCONFIRMED')
            self._state = 'CLEANED'
        except BaseException as error:
            self._state = 'CLEANUP_FAILED'
            self._record_error('cleanup', error)
            raise
        return self.query(worker, identity)
