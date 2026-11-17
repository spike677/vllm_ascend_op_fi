"""Real synchronous tensor/module replays with typed Python-hook ownership.

The deployment supplies an actual model replay, not a layer/address from RPC.
This is a bounded unique-call adapter. It does not certify a vLLM scheduler or
silently turn a constructed module experiment into a pretrained-model result.
"""
from dataclasses import dataclass, field
import hashlib
import threading

from ._scope import Scope, ProviderError, reject_unknown


MECHANISMS = ('15.alias_detach', '17.spatial_padding', '18.api_half', '8.restore_offset', '22.wrong_row',
              '25.reduce_scatter')


@dataclass(frozen=True)
class TensorReplayBinding:
    worker: object
    model: object
    replay: object
    synchronize: object
    workload_kind: str
    simulation: bool = False
    format_provider: object = None
    measured_pair: object = None
    _leases: set = field(default_factory=set, compare=False, repr=False)


def install_tensor_replay(worker, mechanism, *, replay, synchronize,
                          workload_kind, simulation=False, _format_provider=None, _measured_pair=None):
    """Trusted deployment entry. No arbitrary callback is accepted over RPC."""
    if mechanism not in MECHANISMS or not callable(replay) or not callable(synchronize):
        raise ValueError('A known mechanism and executable replay/synchronizer are required')
    if workload_kind not in ('constructed_module', 'pretrained_model') or type(simulation) is not bool:
        raise ValueError('Explicit workload scope and simulation status required')
    model = worker.model_runner.model
    if any(name in vars(worker.model_runner) for name in ('execute_model', '_prepare_inputs')):
        raise ValueError('Core runner replacement is not supported')
    if _format_provider is not None and not simulation:
        raise ValueError('Custom format providers are CPU-test seams, not NPU validation')
    if mechanism == '25.reduce_scatter':
        from fault_injection.collective25 import ReduceScatterSession
        # Constructor verifies the calibration was produced by actual native
        # observations. It installs no hooks and does not execute a collective.
        ReduceScatterSession(model,mode='replay_delta',pair=_measured_pair,simulation=simulation)
    elif _measured_pair is not None:
        raise ValueError('A measured collective pair belongs only to Fault25')
    bindings = getattr(worker, '_dcat_tensor_module_bindings', None)
    if bindings is None:
        bindings = {}
        worker._dcat_tensor_module_bindings = bindings
    if mechanism in bindings:
        raise ValueError('This Worker already owns a deployment binding for the mechanism')
    binding = TensorReplayBinding(worker, model, replay, synchronize, workload_kind,
                                  simulation, _format_provider, _measured_pair)
    bindings[mechanism] = binding
    return binding


def uninstall_tensor_replay(worker, mechanism, binding):
    bindings = getattr(worker, '_dcat_tensor_module_bindings', {})
    if bindings.get(mechanism) is not binding:
        raise ValueError('Deployment binding ownership changed')
    if binding._leases:
        raise ValueError('Close the provider before removing its deployment binding')
    del bindings[mechanism]


def _capture(value):
    """Private complete output comparison, separate from independent statistics."""
    import torch
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous().clone()
        return tensor
    if isinstance(value, dict):
        return {str(key): _capture(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_capture(item) for item in value)
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise ProviderError('UNSUPPORTED_REPLAY_VALUE_TYPE')


def _devices(value):
    import torch
    if isinstance(value, torch.Tensor):
        return {value.device.type}
    if isinstance(value, dict):
        return set().union(*(_devices(x) for x in value.values()))
    if isinstance(value, (list, tuple)):
        return set().union(*(_devices(x) for x in value))
    return set()


def _identity(value):
    import torch
    if isinstance(value, torch.Tensor):
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        return dict(shape=list(value.shape), dtype=str(value.dtype),
                    sha256=hashlib.sha256(raw).hexdigest(),
                    finite=bool(torch.isfinite(value).all()) if value.is_floating_point() else True)
    if isinstance(value, dict):
        return {key: _identity(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_identity(item) for item in value]
    return value


def _same(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.shape == right.shape
                and left.dtype == right.dtype and torch.equal(left, right))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, tuple):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _candidate(event, mechanism):
    if mechanism == '15.alias_detach':
        return (event.get('status') == 'routed' and event.get('original_execution_verified') is True
                and bool(event.get('writes')) and bool(event.get('consumers'))
                and all(w['view_write_correct'] and w['base_matches_expected_update']
                        and w['non_target_unchanged'] for w in event['writes']))
    if mechanism == '22.wrong_row':
        return (event.get('status') == 'baseline' and event.get('original_execution_verified') is True
                and bool(event.get('consumers')))
    if mechanism == '25.reduce_scatter':
        return (event.get('status') == 'observed' and event.get('original_execution_verified') is True
                and event.get('source') == 'ACTUAL_COLLECTIVE_RETURN')
    return event.get('status') == 'baseline' and event.get('original_execution_verified') is True


def _signature(event, mechanism):
    fields = ('path', 'ordinal', 'output_shape', 'dtype', 'input_shape', 'weight_shape', 'boundary')
    result = {key: event[key] for key in fields if key in event}
    if mechanism == '15.alias_detach':
        result.update({key: event[key] for key in ('size', 'stride', 'offset', 'actual_format')})
        result['base'] = {key: event['base'][key] for key in
                          ('shape', 'stride', 'offset', 'dtype', 'device', 'actual_format')}
    if mechanism == '22.wrong_row':
        result.update({key: event[key] for key in
                       ('source_shape','target_row','correct_source_row','replacement_source_row')})
    if mechanism == '8.restore_offset':
        result.update({key: event[key] for key in ('token_groups','topk')})
    if mechanism == '25.reduce_scatter':
        result['actual_collective_signature']=event['signature']
    return result


class TensorModuleProvider(Scope):
    execution_boundary = 'python_hook'

    def __init__(self, worker, model, identity, mechanism):
        if mechanism not in MECHANISMS:
            raise ValueError(mechanism)
        super().__init__(worker, model, identity)
        self.mechanism = mechanism
        self.binding = self.session = None
        self.checked = self.attempted = self.recovered = False
        self.thread = threading.get_ident()
        self.running = False
        self.rows = []
        self.error = None
        self.selected = None
        self.extra_hooks = []
        self._extra_callbacks = {}
        self._private_observations = {}

    def _context(self):
        if threading.get_ident() != self.thread:
            raise ProviderError('PYTHON_HOOK_OWNER_THREAD_CHANGED')
        if getattr(getattr(self.worker, 'model_runner', None), 'model', None) is not self.model:
            raise ProviderError('LOADED_MODEL_IDENTITY_CHANGED')
        if self.model.training:
            raise ProviderError('INFERENCE_MODEL_REQUIRED')
        if self.binding is not None:
            bindings = getattr(self.worker, '_dcat_tensor_module_bindings', {})
            if bindings.get(self.mechanism) is not self.binding:
                raise ProviderError('DEPLOYMENT_BINDING_CHANGED')

    def dispatch(self, action, settings, *, owner, generation):
        reject_unknown(settings)
        if action == 'prepare':
            self.bind(owner, generation)
            self._context()
            binding = getattr(self.worker, '_dcat_tensor_module_bindings', {}).get(self.mechanism)
            if (type(binding) is not TensorReplayBinding or binding.worker is not self.worker
                    or binding.model is not self.model):
                raise ProviderError('TRUSTED_TENSOR_REPLAY_REQUIRED')
            self.binding = binding
        else:
            self.verify(owner, generation)
            self._context()
            if action == 'register':
                if self.binding is None:
                    raise ProviderError('TENSOR_PREPARE_REQUIRED')
                self._registered = True
                self.binding._leases.add((owner, generation))
            elif action == 'check':
                if not self._registered:
                    raise ProviderError('TENSOR_REGISTER_REQUIRED')
                self.checked = True
            elif action in ('execute', 'enable'):
                if not self.checked or self.attempted:
                    raise ProviderError('SINGLE_QUALIFIED_ATTEMPT_REQUIRED')
                self.attempted = True
                self._execute()
            elif action == 'finish':
                if not self.recovered:
                    raise ProviderError('OFF_REPLAY_NOT_VERIFIED')
                self._stop()
            elif action in ('disable', 'abort', 'recover', 'clean', 'unregister'):
                self._stop()
                if action in ('clean', 'unregister'):
                    self._registered = False
                    if self.binding:
                        self.binding._leases.discard((owner, generation))
            else:
                raise ProviderError('UNSUPPORTED_PROVIDER_ACTION')
        return self.query(owner=owner, generation=generation)

    def _make_session(self, arm):
        if self.mechanism == '25.reduce_scatter':
            from fault_injection.collective25 import ReduceScatterSession
            return ReduceScatterSession(self.model,mode='replay_delta' if arm=='fault' else 'observe',
                pair=self.binding.measured_pair if arm=='fault' else None,simulation=self.binding.simulation)
        if self.mechanism in ('8.restore_offset','22.wrong_row'):
            from fault_injection.route8 import RestoreOffsetSession
            from fault_injection.index22 import WrongRowSession
            session = RestoreOffsetSession if self.mechanism == '8.restore_offset' else WrongRowSession
            return session(self.model, arm=arm, simulation=self.binding.simulation,
                           selected_call=self.selected)
        if self.mechanism == '15.alias_detach':
            from fault_injection.alias import AliasSession
            return AliasSession(self.model, arm=arm, unsupported='passthrough',
                _selected_call=self.selected, _format_provider=self.binding.format_provider)
        from fault_injection.module_faults import ModuleFaultSession
        return ModuleFaultSession(self.model, self.mechanism, arm=arm,
                                  unsupported='passthrough', _selected_call=self.selected)

    def _phase(self, phase, arm=None):
        import torch
        self._context()
        seen = []
        root_active = []
        rejected = []

        def before(module, args, kwargs):
            if root_active or seen:
                rejected.append(True)
                raise ProviderError('EXACTLY_ONE_ROOT_FORWARD_PER_REPLAY_REQUIRED')
            root_active.append(True)
            seen.append(_capture((args, kwargs)))

        def after(module, args, kwargs, result):
            if root_active:
                root_active.pop()

        handles = [self.model.register_forward_pre_hook(before, with_kwargs=True, prepend=True),
                   self.model.register_forward_hook(after, with_kwargs=True, always_call=True)]
        self.extra_hooks.extend(handles)
        self._extra_callbacks.update({handles[0].id:before,handles[1].id:after})
        try:
            self.session = self._make_session(arm) if arm else None
            if self.session:
                self.session.__enter__()
                if not self.session.runtime_state()['ownership_verified']:
                    raise ProviderError('PYTHON_HOOK_OWNERSHIP_CHANGED')
            self.running = True
            with torch.inference_mode():
                output = self.binding.replay(self.model, phase)
            self.binding.synchronize()
            devices = _devices(output)
            if not self.binding.simulation and devices != {'npu'}:
                raise ProviderError('REAL_NPU_OUTPUT_REQUIRED')
            if len(seen) != 1 or root_active or rejected:
                raise ProviderError('REPLAY_DID_NOT_COMPLETE_ONE_MODEL_FORWARD')
            output = _capture(output)
            report = self.session.report() if self.session else None
            if self.session is not None:
                self._private_observations[phase] = getattr(self.session,'private_observations',{})
            if report is not None and (len(report['forwards']) != 1
                    or report['forwards'][0]['status'] != 'returned'):
                raise ProviderError('ORIGINAL_MODEL_EXECUTION_UNPROVEN')
            if self.session and not self.session.runtime_state()['ownership_verified']:
                raise ProviderError('PYTHON_HOOK_OWNERSHIP_CHANGED')
            row = dict(phase=phase, input=_identity(seen[0]), output=_identity(output),
                       mechanism_observation=report, original_model_returned=True,
                       execution_devices=sorted(devices), simulation=self.binding.simulation)
            self.rows.append(row)
            return seen[0], output, report
        finally:
            self.running = False
            # Keep a failed session reachable so clean can retry owned release.
            try:
                if self.session is not None:
                    self.session.close()
                    if not self.session.runtime_state()['cleanup_confirmed']:
                        raise ProviderError('PYTHON_HOOK_RECOVERY_PENDING')
                    self.session = None
            finally:
                for handle in reversed(handles):
                    self._remove_extra_hook(handle)

    def _execute(self):
        try:
            first_input, reference, first = self._phase('baseline', 'baseline')
            second_input, repeat, second = self._phase('baseline_repeat', 'baseline')
            if not _same(first_input, second_input) or not _same(reference, repeat):
                raise ProviderError('BASELINE_REPLAY_NOT_EXACTLY_STABLE')
            if self.mechanism in ('15.alias_detach','17.spatial_padding','18.api_half'):
                from fault_injection.runtime_discovery import observed_python_candidate_set
                candidates_set = observed_python_candidate_set(self.model,self.mechanism,
                    reports=(first,second),worker_identity=self.identity,provider_version='1')
                self.candidate_set = candidates_set.report()
                from fault_injection.core import UnsupportedCall
                try:
                    candidates_set.select('unique')
                except UnsupportedCall as exc:
                    raise ProviderError('ACTUAL_TARGET_ABSENT_OR_AMBIGUOUS',str(exc)) from exc
            candidates = [[e for e in row['events'] if _candidate(e, self.mechanism)]
                          for row in (first, second)]
            if any(len(rows) != 1 for rows in candidates):
                raise ProviderError('ACTUAL_TARGET_ABSENT_OR_AMBIGUOUS')
            target, repeat_target = candidates[0][0], candidates[1][0]
            if _signature(target, self.mechanism) != _signature(repeat_target, self.mechanism):
                raise ProviderError('ACTUAL_TARGET_IDENTITY_UNSTABLE')
            self.selected = (target['path'], target['ordinal'])
            lossless_input, lossless, _ = self._phase('lossless', 'baseline')
            if not _same(first_input, lossless_input) or not _same(reference, lossless):
                raise ProviderError('LOSSLESS_GATE_FAILED')
            fault_input, fault, observation = self._phase('fault', 'fault')
            if not _same(first_input, fault_input):
                raise ProviderError('FAULT_MODEL_INPUT_CHANGED')
            hits = [e for e in observation['events'] if e.get('applied') is True
                    or e.get('status') == 'applied']
            if len(hits) != 1 or not hits[0].get('original_execution_verified'):
                raise ProviderError('EXACTLY_ONE_REAL_INTERVENTION_REQUIRED')
            if self.mechanism == '15.alias_detach':
                hit = hits[0]
                if (hit['aliases_original_base'] or not hit['initial_values_correct'] or not hit['consumers']
                    or not hit['writes'] or not all(w['view_write_correct'] and w['non_target_unchanged']
                        and w['base_target_retains_prewrite_values'] and w['write_changes_target_values']
                        and not w['base_matches_expected_update'] for w in hit['writes'])):
                    raise ProviderError('ALIAS_WRITE_LOSS_EFFECT_NOT_VERIFIED')
            elif self.mechanism == '22.wrong_row':
                if not hits[0]['consumers'] or not hits[0]['replacement_readback_exact']:
                    raise ProviderError('WRONG_ROW_CONSUMER_NOT_VERIFIED')
            elif self.mechanism == '8.restore_offset':
                if not hits[0]['paired_group_readback_exact'] or not hits[0]['consumer_returned']:
                    raise ProviderError('PAIRED_MOE_CONSUMER_NOT_VERIFIED')
                # Compare the complete actual native outputs, not a Python
                # arithmetic model of an unknown reduction/rounding order.
                before = self._private_observations['baseline'][self.selected]
                after = self._private_observations['fault'][self.selected]
                if (not _same(before['inputs'],after['inputs'])
                    or not _same(before['output'][after['source_groups']],after['output'])):
                    raise ProviderError('NATIVE_MOE_COMPLETE_GROUP_MAPPING_NOT_EXACT')
                hits[0]['complete_native_group_mapping_exact'] = True
            elif self.mechanism == '25.reduce_scatter':
                if not hits[0].get('exact_measured_alternate'):
                    raise ProviderError('MEASURED_COLLECTIVE_REPLAY_NOT_VERIFIED')
            off_input, off, _ = self._phase('off')
            if not _same(first_input, off_input) or not _same(reference, off):
                raise ProviderError('COMPLETE_OFF_OUTPUT_NOT_RESTORED')
            self.recovered = True
            self.effect = dict(original_execution_verified=True, interventions=1,
                output_changed=not _same(reference, fault), off_exact=True,
                native_execution_claimed=not self.binding.simulation,
                workload_kind=self.binding.workload_kind)
        except BaseException as exc:
            self.error = type(exc).__name__ + ': ' + str(exc)
            raise

    def _stop(self):
        if self.running:
            raise ProviderError('PYTHON_REPLAY_INFLIGHT')
        if self.session is not None:
            self.session.close()
            self.binding.synchronize()
            if not self.session.runtime_state()['cleanup_confirmed']:
                raise ProviderError('PYTHON_HOOK_RECOVERY_PENDING')
            self.session = None
        for handle in reversed(self.extra_hooks[:]):
            self._remove_extra_hook(handle)
        self._private_observations.clear()

    def _remove_extra_hook(self, handle):
        hooks=handle.hooks_dict_ref()
        if hooks is not None and handle.id in hooks and hooks[handle.id] is not self._extra_callbacks[handle.id]:
            raise ProviderError('PYTHON_REPLAY_HOOK_OWNERSHIP_CHANGED')
        handle.remove()
        self.extra_hooks.remove(handle)
        self._extra_callbacks.pop(handle.id,None)

    def query(self, *, owner, generation):
        self.verify(owner, generation)
        self._context()
        state = self.session.runtime_state() if self.session else dict(hook_owned=False,
                    ownership_verified=True, enabled=False, inflight=0, leases=0,
                    cleanup_confirmed=True)
        if not state['ownership_verified']:
            raise ProviderError('PYTHON_HOOK_OWNERSHIP_CHANGED')
        if any(handle.hooks_dict_ref() is None or handle.hooks_dict_ref().get(handle.id)
               is not self._extra_callbacks.get(handle.id) for handle in self.extra_hooks):
            raise ProviderError('PYTHON_REPLAY_HOOK_OWNERSHIP_CHANGED')
        return dict(execution_boundary=self.execution_boundary, state_source='OWNED_PYTHON_HOOKS',
            hook_state_available=True, owner=owner, generation=generation,
            hook_owned=state['hook_owned'] or bool(self.extra_hooks), native_loaded=False, native_registered=False,
            registered=self._registered, enabled=state['enabled'],
            inflight=max(int(self.running), state['inflight']), leases=state['leases']+int(bool(self.extra_hooks)),
            replay_pending=False, replay_failed=False,
            mechanism=self.mechanism, selected_call=self.selected,
            qualification=dict(recovered=self.recovered, phases=self.rows,
                               candidate_set=getattr(self,'candidate_set',None),
                               effect=getattr(self, 'effect', None), error=self.error,
                               coverage='UNIQUE_ONCE_ONLY', all_eligible=False, every_match=False,
                               independent_statistics='SEPARATE_ANALYSIS_REQUIRED'))
