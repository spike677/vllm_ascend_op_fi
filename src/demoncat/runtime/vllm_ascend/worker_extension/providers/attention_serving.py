"""Four public RPC entries over one actual Attention execution census.

The initial protocol deliberately stops at a stock/private bridge gate. The
zero-F census and matched Native baseline are executable; a factory is not
qualification, and no historical or serialized success flag authorizes F.
"""
from copy import deepcopy
import json
import os
import threading

from ._scope import Scope, ProviderError, reject_unknown
from .. import identity as identity_reader
from ..attention_execution import AttentionExecutionAdapter, digest
from ..serving_attention import ATTRIBUTE, AttentionDeployment

PHASES = ('observe', 'observe_repeat', 'baseline', 'baseline_repeat', 'fault', 'off', 'health')


def require(ok, reason):
    if not ok:raise ProviderError(reason)


def request_values(row):
    return [dict(input=x['input'], hidden=x['hidden'], logits=x['logits']) for x in row['forwards']]


class AttentionServingProvider(Scope):
    mechanism = None
    phases=PHASES
    execution_adapter_type=AttentionExecutionAdapter

    def __init__(self, worker, model, identity):
        super().__init__(worker, model, identity)
        bindings = getattr(worker, ATTRIBUTE, None)
        self.binding = bindings.get(self.mechanism) if type(bindings) is dict else None
        require(type(self.binding) is AttentionDeployment, 'ATTENTION_TRUSTED_DEPLOYMENT_REQUIRED')
        self.runner, self.thread = worker.model_runner, threading.get_ident()
        self.binding_reserved = self.checked = self.attempted = False
        self.stopped, self.index = True, 0
        self.session = None
        self.pending = self.error = None
        self.failures, self.records, self.selection = [], [], None
        self.execution_boundary = 'native' if self.binding.native.state()['native_loaded'] else 'python_hook'
        self._context()

    def _context(self):
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
            'ATTENTION_HELD_WORKER_CONTEXT_CHANGED')
        bindings, binding = getattr(self.worker, ATTRIBUTE, None), self.binding
        require(type(bindings) is dict and bindings.get(self.mechanism) is binding
            and binding.worker is self.worker and binding.runner is self.runner and binding.model is self.model
            and binding.pid == os.getpid() and binding.thread == self.thread,
            'ATTENTION_DEPLOYMENT_BINDING_CHANGED')
        if self.binding_reserved or binding.leases:
            require(self.binding_reserved and binding.leases == {self._ticket}, 'ATTENTION_DEPLOYMENT_OWNER_CHANGED')

    def _failure(self, action, error):
        if self.error is None:self.error = repr(error)
        self.failures.append(dict(action=action, error=repr(error), first_error=self.error))

    def dispatch(self, action, settings, *, owner, generation):
        if action == 'prepare':self.bind(owner, generation)
        self.verify(owner, generation)
        try:
            self._context()
            reject_unknown(settings, ('request_id', 'collect') if action == 'advance' else ())
            if action == 'prepare':
                require(not self.binding.leases, 'ATTENTION_DEPLOYMENT_ALREADY_OWNED')
            elif action == 'register':
                require(not self.binding.leases, 'ATTENTION_DEPLOYMENT_ALREADY_OWNED')
                self.binding.leases.add(self._ticket)
                self.binding_reserved = self._registered = True
            elif action == 'check':
                require(self._registered and not self.model.training, 'ATTENTION_REGISTERED_INFERENCE_REQUIRED')
                self.checked = True
            elif action in ('enable', 'execute'):
                require(self.checked and not self.attempted, 'ATTENTION_SINGLE_ATTEMPT_REQUIRED')
                self.attempted, self.stopped = True, False
                # Save object before construction/install so partial starts
                # retain all resource handles for public cleanup retry.
                self.session = self.execution_adapter_type.__new__(self.execution_adapter_type)
                self.session.__init__(self.binding, self._context)
                self.session.install()
            elif action == 'advance':
                require(self.session is not None and not self.stopped and not self.error,
                        'ATTENTION_ACTIVE_PROTOCOL_REQUIRED')
                if 'collect' in settings:
                    require(settings == {'collect': True} and type(settings['collect']) is bool,
                            'ATTENTION_EXPLICIT_COLLECTION_REQUIRED')
                    self._collect()
                else:
                    require(set(settings) == {'request_id'} and self.pending is None and self.index < len(self.phases),
                            'ATTENTION_EXPLICIT_FRESH_REQUEST_REQUIRED')
                    phase = self.phases[self.index]
                    self._prepare_phase(phase)
                    request_id = settings['request_id']
                    self.session.expect(request_id, phase)
                    self.pending = (request_id, phase)
            elif action == 'finish':
                if self.pending is not None:self._collect()
                require(len(self.records) >= 2, 'ATTENTION_TWO_ACTUAL_REQUESTS_REQUIRED')
                self._cleanup()
            elif action in ('disable', 'abort', 'recover', 'clean', 'unregister'):
                self._cleanup()
                if action in ('clean', 'unregister'):self._registered = False
                if action == 'unregister' and self.binding_reserved:
                    self.binding.leases.remove(self._ticket);self.binding_reserved = False
            else:
                raise ProviderError('UNSUPPORTED_PROVIDER_ACTION')
        except BaseException as error:
            self._failure(action, error)
            raise
        return self.query(owner=owner, generation=generation)

    def _prepare_phase(self,phase):
        if phase.startswith('baseline'):
            require(not getattr(self.binding.native,'observer_only',False), 'ATTENTION_STOCK_CENSUS_OBSERVE_ONLY')
            require(self.selection is not None, 'ATTENTION_NO_STABLE_MATCHING_ACTUAL_CALL')
            require(self.binding.native.state()['native_loaded'], 'ATTENTION_MATCHING_NATIVE_DEPLOYMENT_REQUIRED')
        if phase=='fault':
            raise ProviderError('ATTENTION_STOCK_PRIVATE_BRIDGE_REQUIRED')

    def _collect(self):
        require(self.pending is not None, 'ATTENTION_NO_REQUEST_TO_COLLECT')
        row = self.session.collect()
        require((row['request_id'], row['phase']) == self.pending, 'ATTENTION_REQUEST_BINDING_CHANGED')
        self.records.append(row)
        self.pending = None
        self.index += 1
        if row['phase'] == 'observe_repeat':
            left, right = self.records[:2]
            require(request_values(left) == request_values(right), 'ATTENTION_HEALTHY_INPUT_OR_OUTPUT_CHANGED')
            before = [call for forward in left['forwards'] for call in forward['calls']]
            after = [call for forward in right['forwards'] for call in forward['calls']]
            require(len(before) == len(after) and before, 'ATTENTION_ACTUAL_DENOMINATOR_CHANGED_OR_EMPTY')
            stable = []
            for first, second in zip(before, after):
                require((first['actual_operator'], first.get('module_path'), first.get('occurrence')) ==
                    (second['actual_operator'], second.get('module_path'), second.get('occurrence')),
                    'ATTENTION_ACTUAL_CALL_ORDER_CHANGED')
                if all(call['input_contract_matches'] and call['original_returned'] and call['original_calls'] == 1
                       for call in (first, second)) and first['target_key'] == second['target_key']:
                    require(first['module_object_id'] == second['module_object_id'], 'ATTENTION_ACTUAL_MODULE_CHANGED')
                    stable.append(first)
            if stable:
                chosen = min(stable, key=lambda item: item['target_key'])
                self.selection = dict(target_key=chosen['target_key'], module_path=chosen['module_path'],
                    actual_operator=chosen['actual_operator'], actual_denominator=len(after),
                    matching_repeated_calls=len(stable), selector='MIN_ACTUAL_STABLE_CALL_KEY')
                self.session.target = chosen['target_key']
        elif row['phase'].startswith('baseline'):
            calls = [call for forward in row['forwards'] for call in forward['calls']]
            hits = [call for call in calls if call['native'] is not None]
            require(len(hits) == 1 and hits[0]['target_key'] == self.selection['target_key']
                and hits[0]['native']['original_calls'] == 1
                and hits[0]['native']['selection_applied'] is False,
                'ATTENTION_EXACT_ONE_NATIVE_BASELINE_REQUIRED')
            if row['phase'] == 'baseline_repeat':
                require(request_values(row) == request_values(self.records[-2]), 'ATTENTION_NATIVE_BASELINE_NOT_STABLE')

    def _cleanup(self):
        session = self.session
        if session is not None and hasattr(session, 'handles'):
            session.close()
        else:
            self.binding.native.stop()
        self.pending = None
        self.stopped = True
        self.error = None  # Historical first errors remain in failures and rows.

    def query(self, *, owner, generation):
        self.verify(owner, generation);self._context()
        native = self.binding.native.state()
        session = self.session
        state = session.state() if session is not None and hasattr(session, 'handles') else dict(
            owned_hooks=0, owned_api_wrappers=0, inflight=0, needs_off=False, pending_request=None,
            completed_requests=0, failures=[])
        pending_native = native.get('pending_drains',0) + native.get('pending_references',0)
        resources = state['owned_hooks'] + state['owned_api_wrappers'] + int(native['scope_owned']) + native['environment_owned'] + pending_native
        actual_calls=[call for row in self.records for forward in row['forwards'] for call in forward['calls']]
        observed_census=sum(call.get('native_census',{}).get('status') == 'OBSERVED_INTERCEPTED_STOCK_LAUNCHES'
                            for call in actual_calls if call.get('native_census') is not None)
        result = dict(execution_boundary=self.execution_boundary, owner=owner, generation=generation,
            native_loaded=native['native_loaded'], native_registered=bool(resources) if self.execution_boundary == 'native' else False,
            registered=self._registered, enabled=bool(native['scope_active']), inflight=state['inflight'],
            leases=int(native['scope_owned']) + native['environment_owned'] + pending_native, binding_reserved=self.binding_reserved,
            replay_pending=bool(self.attempted and not self.stopped),
            replay_failed=bool(self.error or not self.stopped and session and getattr(session, 'last_error', None)),
            owned_hooks=state['owned_hooks'], owned_api_wrappers=state['owned_api_wrappers'],
            mechanism=self.mechanism, simulation=self.binding.simulation, native_state=native,
            serving=dict(next_phase=self.phases[self.index] if self.index < len(self.phases) else 'complete',
                         pending_request=self.pending, failures=deepcopy(self.failures), resource_state=state),
            evidence=dict(schema='demoncat.attention.actual_calls.v1', provenance=json.loads(self.binding.provenance_json),
                selection=self.selection, requests=deepcopy(self.records), actual_requests_sha256=digest(self.records),
                observed_call_count=sum(len(forward['calls']) for row in self.records for forward in row['forwards']),
                qualified_targets=[], independent_native_denominator=False,
                stock_launch_census=dict(observer_only=bool(native.get('observer_only')),
                    actual_calls=len(actual_calls),observed_call_scopes=observed_census,
                    unobserved_call_scopes=len(actual_calls)-observed_census,
                    intercepted_call_denominator_complete=bool(actual_calls) and observed_census == len(actual_calls),
                    all_runtime_entrypoints_covered=False,stock_pod_decoded=False,native_fault_qualified=False),
                source=getattr(session, 'source', None), historical_qualification_inherited=False,
                stock_private_bridge='NOT_ESTABLISHED', fault_admission='BLOCKED_STOCK_PRIVATE_BRIDGE_REQUIRED'),
            C1=False, C2=False, C3=False, E5='NOT_ASSESSED', public_qualification_granted=False)
        if self.execution_boundary == 'python_hook':
            result.update(state_source='OWNED_PYTHON_HOOKS', hook_state_available=True, hook_owned=bool(resources))
        return result


class FaLegalTilingServingProvider(AttentionServingProvider):
    mechanism = '28.fa.legal_tiling'


class FiaLegalTilingServingProvider(AttentionServingProvider):
    mechanism = '28.fia.legal_tiling'


class FiaDenominatorReplayServingProvider(AttentionServingProvider):
    mechanism = '28.fia.denominator_replay'


class PagedDecodeServingProvider(AttentionServingProvider):
    mechanism = '33.paged_decode'
