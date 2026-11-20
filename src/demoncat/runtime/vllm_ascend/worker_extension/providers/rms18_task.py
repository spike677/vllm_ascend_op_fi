"""Ordinary one-target RMS18 task, separate from the ten-request group test.

Control RPCs only queue real request IDs. The existing Runtime owns this
provider; actual model execution stays entirely in the normal serving path.
"""
from copy import deepcopy
from fault_injection.request_plan import FOUR

from ._scope import ProviderError, reject_unknown
from .tensor_serving import TensorServingProvider, require
from fault_injection.effect_summary import actual_request_effect, current_completed_row
from fault_injection.tensor_callable import TensorCallableGuard


class Rms18TaskProvider(TensorServingProvider):
    mechanism_name = '18.api_half'
    code_prefix = 'RMS18'
    target_selector = 'FIRST_ACTUAL_RMS_RETURN_REQUIRES_CONSUMER'
    task_scope = 'ONE_ACTUAL_CONSUMED_RMS_RETURN_MATCHED_FRESH_PREFILL'

    def require(self, condition, reason):
        require(condition, reason.replace('RMS18', self.code_prefix))

    def make_request(self):
        from ..rms18_request import Rms18TaskRequest
        return Rms18TaskRequest(self)

    def __init__(self, worker, model, identity):
        super().__init__(worker, model, identity, self.mechanism_name)
        self.require(self.binding.runtime_only, 'RMS18_ORDINARY_BINDING_REQUIRED')
        self.rows, self.requests = [], set()
        self.target = self.target_module = self.target_guard = None
        self.normal = None
        self.first_error = None
        self.fault_started = self.fault_applied = False

    @property
    def phase(self):
        return self.recovering or (FOUR[self.index]
                                  if self.index < len(FOUR) else 'complete')

    def dispatch(self, action, settings, *, owner, generation):
        if action == 'prepare': self.bind(owner, generation)
        self.verify(owner, generation)
        try:
            self._context()
            reject_unknown(settings, ('request_id',) if action == 'advance' else ())
            if action == 'prepare':
                self.require(not self.binding.leases, 'TENSOR_DEPLOYMENT_ALREADY_OWNED')
            elif action == 'register':
                self.require(not self.binding.leases and self.session is None, 'RMS18_NEW_TASK_SCOPE_REQUIRED')
                self.binding.leases.add(self._ticket)
                self.binding_reserved = self._registered = True
                self.session = self.make_request()
            elif action == 'check':
                self.require(self._registered and self.session is not None and not self.model.training,
                        'TENSOR_REGISTERED_INFERENCE_REQUIRED')
                self.checked = True
            elif action in ('enable', 'execute'):
                self.require(self.checked and not self.attempted, 'TENSOR_SINGLE_ATTEMPT_REQUIRED')
                self.attempted, self.stopped = True, False
                self.session.install()
            elif action == 'advance':
                self.require(self.attempted and not self.stopped and not self.error,
                        'RMS18_ACTIVE_OR_RECOVERY_REQUEST_REQUIRED')
                self.require(set(settings) == {'request_id'}, 'FRESH_REAL_ENQUEUED_REQUEST_REQUIRED')
                self.session.expect(settings['request_id'])
            elif action == 'recover':
                self.require(self.session is not None, 'RMS18_REGISTER_REQUIRED')
                self.session.close_window()
                if self.fault_started and not self.recovered:
                    self.recovering, self.stopped = 'off', False
                    self.error = self.session.request_error = None
                else:
                    self._cleanup()
            elif action in ('finish', 'disable', 'abort', 'clean', 'unregister'):
                if action == 'finish':
                    self.require(self.complete and self.recovered and not self.error,
                            'RMS18_ACTUAL_OFF_AND_HEALTH_REQUIRED')
                self._cleanup()
                if action in ('clean', 'unregister'): self._registered = False
                if action == 'unregister' and self.binding_reserved:
                    self.binding.leases.remove(self._ticket)
                    self.binding_reserved = False
            else:
                raise ProviderError('UNSUPPORTED_PROVIDER_ACTION')
        except BaseException as error:
            self.fail(error, action)
            raise
        return self.query(owner=owner, generation=generation)

    def begin_actual(self, row):
        self._context()
        self.require(not self.model.training, 'RMS18_CURRENT_INFERENCE_MODEL_REQUIRED')
        request_id = row['request_id']
        self.require(request_id not in self.requests and len(self.requests) < 8,
                'RMS18_BOUNDED_FRESH_ACTUAL_REQUEST_REQUIRED')
        self.requests.add(request_id)
        self.require(not self.error and row['phase'] == self.phase, 'RMS18_PHASE_OR_ERROR_CHANGED')
        if self.normal is not None:
            self.require(row['input'] == self.normal['input'], 'RMS18_CURRENT_COMPLETE_INPUT_CHANGED')
            modules = dict(self.model.named_modules())
            self.require(modules.get(self.target['path']) is self.target_module,
                    'RMS18_HELD_TARGET_MODULE_CHANGED')
            self.target_guard.verify()
        if self.phase == 'fault':
            self.require(self.normal is not None and not self.fault_started, 'RMS18_ONE_FAULT_ATTEMPT_REQUIRED')
            # The attempted latch is set before compute and is never cleared.
            self.fault_started = True

    def admit(self, event, actual):
        self._context()
        target = self.target
        if target is None or (event['path'], event['ordinal']) != (target['path'], target['ordinal']):
            return False
        self.require(self.phase == 'fault' and self.fault_started and not self.fault_applied,
                'RMS18_FAULT_SINGLE_CURRENT_RETURN_REQUIRED')
        self.target_guard.verify()
        self.require(actual == target['actual'] and event['operator'] == target['operator']
                and event.get('schema') == target['native_schema'], 'RMS18_CURRENT_RETURN_INPUT_WEIGHT_OR_SOURCE_CHANGED')
        self.fault_applied = True
        return True

    def end_actual(self, row):
        self._context()
        phase = self.phase
        self.require(row['phase'] == phase and row['request_id'] in self.requests,
                'RMS18_ACTUAL_COMPLETED_REQUEST_MAPPING_CHANGED')
        report = row['mechanism'] or {}
        events = report.get('events', [])
        if phase in ('normal', 'fault'):
            self.require(report.get('forwards') == [dict(index=0, status='returned')]
                    and not report.get('errors'), 'RMS18_ORIGINAL_ROOT_RETURN_UNCONFIRMED')
        if phase == 'normal':
            eligible = [event for event in events if event.get('original_execution_verified')
                and event.get('return_readback_exact') and event.get('consumers')
                and event.get('actual') is not None and not event.get('applied')]
            self.require(bool(eligible), 'RMS18_NO_ACTUAL_CONSUMED_RETURN')
            event = eligible[0]
            self.target = dict(path=event['path'], ordinal=event['ordinal'], actual=event['actual'],
                               operator=event['operator'], native_schema=event.get('schema'),
                               selector=self.target_selector)
            self.target_module = dict(self.model.named_modules()).get(event['path'])
            self.require(self.target_module is not None, 'RMS18_TARGET_NOT_IN_HELD_MODEL')
            self.target_guard = TensorCallableGuard(self.target_module)
            self.target['source'] = self.target_guard.report()
            self.normal = deepcopy(row)
        elif phase == 'fault':
            hits = [event for event in events if event.get('applied')]
            self.require(len(hits) == 1 and self.fault_applied and hits[0].get('output_changed')
                    and hits[0].get('return_readback_exact') and hits[0].get('consumers'),
                    'RMS18_SELECTED_RETURN_EFFECT_AND_CONSUMER_REQUIRED')
            row['logits_changed'] = row['output']['logits'] != self.normal['output']['logits']
            row['hidden_changed'] = row['output']['hidden'] != self.normal['output']['hidden']
        else:
            self.require(row['output'] == self.normal['output'] and not any(e.get('applied') for e in events),
                    'RMS18_COMPLETE_HIDDEN_LOGITS_OFF_NOT_EXACT')
            if phase == 'health': self.recovered = True
        self.rows.append(deepcopy(row))
        if self.recovering:
            self.recovering = 'health' if phase == 'off' else None
            if phase == 'health':
                self.index = len(FOUR)
                self.complete = True
        else:
            self.index += 1
            self.complete = self.index == len(FOUR)

    def fail(self, error, action='actual_request'):
        message = type(error).__name__ + ': ' + str(error)
        if self.first_error is None: self.first_error = message
        if self.error is None: self.error = message
        self.failures.append(dict(action=action, phase=self.phase, error=message, first_error=self.first_error))

    def _cleanup(self):
        if self.session is not None:
            self.session.close_window()
            self.require(not self.fault_started or self.recovered, 'RMS18_REAL_OFF_AND_HEALTH_REQUIRED')
            self.session.close()
        self.stopped = True

    def query(self, *, owner, generation):
        self.verify(owner, generation)
        self._context()
        state = self.session.readback() if self.session else dict(
            installed=False, owned_hooks=0, inflight=0, leases=0, pending=False)
        current = current_completed_row(self.rows[-1] if self.rows else None, pending=state['pending'])
        history = next((row for row in reversed(self.rows) if row['phase'] == 'fault'), None)
        last = actual_request_effect(history)
        last['historical'] = True
        result = dict(execution_boundary='python_hook', state_source='OWNED_PYTHON_HOOKS',
            hook_state_available=True, hook_owned=bool(state['owned_hooks']), owner=owner,
            generation=generation, native_loaded=False, native_registered=False,
            registered=self._registered, enabled=self.session is not None and self.session.window is not None,
            inflight=state['inflight'], leases=state['leases'], owned_hooks=state['owned_hooks'],
            replay_pending=self.attempted and not self.stopped, replay_failed=bool(self.error and not self.stopped),
            recovery_validation_pending=self.fault_started and not self.recovered,
            current_request_phase=self.phase if state['pending'] else current['phase'] if current else None,
            current_request_effect=actual_request_effect(current), last_fault_effect=last,
            serving=dict(next_phase=self.phase, complete=self.complete, recovered=self.recovered,
                fault_attempted=self.fault_started, fault_return_admitted=self.fault_applied,
                pending_request=None if self.session is None else self.session.expected,
                actual_requests=deepcopy(self.rows), selection=deepcopy(self.target),
                failure=self.error, first_error=self.first_error, failures=deepcopy(self.failures)),
            ordinary_task=True, scope=self.task_scope,
            constructed_replay_fallback=False, default_dump=False, default_msprobe=False,
            historical_qualification_inherited=False, C1=False, C2=False, C3=False, E5=False,
            qualification_granted=False, main_project_integration='NOT_VERIFIED',
            npu_changed_route_validation='PENDING', simulation=self.binding.simulation,
            next_task='NEW_RUNTIME_OWNER_AND_PROVIDER_REQUIRED')
        result['evidence'] = deepcopy(result['serving'])
        result['evidence'].update(last_fault_effect=deepcopy(last),
            matched_off_and_health_confirmed=bool(self.complete and self.recovered))
        return result
