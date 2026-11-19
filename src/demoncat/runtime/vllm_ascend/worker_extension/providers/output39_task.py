"""Ordinary Runtime-owned output39 task; historical C1/C2/C3 are separate."""
from copy import deepcopy
from fault_injection.request_plan import FOUR
import hashlib
from fault_injection.effect_summary import current_completed_row

from ._scope import ProviderError, reject_unknown
from .fault39_output import Fault39OutputServingProvider, require
from ..serving39_output import read_native_binding


class Output39TaskProvider(Fault39OutputServingProvider):
    def __init__(self, worker, model, identity):
        super().__init__(worker, model, identity)
        require(self.binding.runtime_only and self.binding.stage == 'c1'
            and self.binding.max_live_leases == 1 and self.binding.folder is None,
            'OUTPUT39_EXPLICIT_ORDINARY_TASK_BINDING_REQUIRED')

    @property
    def phase(self):
        return self.recovering or (FOUR[self.index]
                                  if self.index < len(FOUR) else 'complete')

    @property
    def native_phase(self):
        return self.phase

    def dispatch(self, action, settings, *, owner, generation):
        if action == 'prepare': self.bind(owner, generation)
        self.verify(owner, generation)
        reject_unknown(settings, ('request_id',) if action == 'advance' else ())
        self._context()
        if action == 'prepare':
            require(not self.binding.leases, 'OUTPUT39_DEPLOYMENT_ALREADY_OWNED')
            read_native_binding(self.binding)
        elif action == 'register':
            require(not self.binding.leases and self.p is None, 'OUTPUT39_NEW_TASK_SCOPE_REQUIRED')
            self.binding.leases.add(self._ticket)
            self.binding_reserved = self._registered = True
            from fault_injection.deployments.output39_execution import Output39SingleExecution
            from ..output39_task import Output39TaskSession
            self.p = Output39SingleExecution(self.model, self.binding.transport, guard=self._context)
            self.native_owner = int.from_bytes(hashlib.sha256(owner.encode()).digest()[:8], 'little') % (2**63-1)+1
            self.p.owner, self.p.generation = self.native_owner, generation
            self.native_generation = generation
            self.session = Output39TaskSession(self)
        elif action == 'check':
            require(self._registered and self.session is not None and not self.p.live_lease_count,
                    'OUTPUT39_REGISTER_OR_RECOVERY_REQUIRED')
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
            if action in ('clean', 'unregister'): self._registered = False
            if action == 'unregister' and self.binding_reserved:
                self.binding.leases.remove(self._ticket)
                self.binding_reserved = False
        else:
            raise ProviderError('UNSUPPORTED_PROVIDER_ACTION')
        return self.query(owner=owner, generation=generation)

    def begin_actual(self, args, kwargs):
        self._context()
        request_id = self.session.current['actual_internal_request_id']
        require(request_id not in self.requests and (not self.failure or self.recovering),
                'OUTPUT39_FRESH_ACTUAL_REQUEST_REQUIRED')
        require(len(self.requests) < 8, 'OUTPUT39_BOUNDED_TASK_REQUESTS_EXHAUSTED')
        self.requests.add(request_id)
        if self.phase == 'health':
            from fault_injection.deployments.postwrite import _hash_inputs
            self.p.context()
            require(_hash_inputs(args, kwargs) == self.p._input_hash and not self.p.live_lease_count,
                    'OUTPUT39_MATCHED_HEALTH_INPUT_REQUIRED')
            return None
        window = self.p.begin(request_id, self.phase, args, kwargs)
        self.native_generation = self.p.generation
        return window

    def end_actual(self, output, result, request):
        self._context()
        phase = self.phase
        if phase == 'health':
            require(output == self.p.normal_output and not self.p.live_lease_count,
                    'OUTPUT39_MATCHED_HEALTH_OUTPUT_CHANGED')
            self.recovered = True
        self.rows.append(dict(phase=phase, actual_internal_request_id=request['actual_internal_request_id'],
            input_sha256=request['input_sha256'], hidden_logits_sha256=output,
            model_output_sha256=request['model_output_sha256'], report=deepcopy(result)))
        if self.recovering:
            self.recovering = 'health' if phase == 'off' else False
            if phase == 'health': self.complete, self.stopped = True, True
        else:
            self.index += 1
            self.complete = self.index == len(FOUR)

    def fail(self, error):
        if self.failure is None: self.failure = type(error).__name__ + ': ' + str(error)
        if self.p is not None: self.p.failed(error)
        self.failures.append(dict(phase=self.phase, error=str(error), first_error=self.failure))

    def _cleanup(self):
        if self.session is not None: self.session.close_window()
        if self.p is not None:
            self.p.recover()
            require(not self.p._fault_attempted or self.recovered, 'OUTPUT39_REAL_OFF_AND_HEALTH_REQUIRED')
        if self.session is not None: self.session.close()
        self.stopped = True

    def evidence(self):
        effect = next((row['report']['effect'] for row in reversed(self.rows)
                       if row['phase'] == 'fault' and row['report']), None)
        pending = self.session is not None and (
            self.session.expected is not None or self.session.current is not None)
        current = current_completed_row(self.rows[-1] if self.rows else None, pending=pending)
        return dict(mechanism='39.output_proxy', scope='SINGLE_TASK_SINGLE_TARGET_MATCHED_PREFILL',
            boundary='actual_public_projection_return_then_value_consumer', native_out_of_bounds_claim=False,
            runtime_only=True, full_census=False, dump_enabled=False, msprobe_enabled=False,
            actual_requests=deepcopy(self.rows), failures=deepcopy(self.failures),
            last_fault_effect=deepcopy(effect), effect_scope='THIS_TASK_MATCHED_NORMAL_AND_FAULT',
            current_request_phase=self.phase if pending else current['phase'] if current is not None else None,
            current_request_effect=deepcopy((current.get('report') or {}).get('effect')) if current is not None else None,
            native_restore_confirmed=self.p is not None and not self.p.live_lease_count,
            matched_off_and_health_confirmed=self.recovered, complete=self.complete,
            next_task='NEW_RUNTIME_OWNER_AND_PROVIDER_REQUIRED',
            c1=False, c2=False, c3=False, E5=False, qualification_granted=False,
            main_project_integration='NOT_VERIFIED')
