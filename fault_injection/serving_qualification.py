"""Fault20's one-target qualification over real host-driven serving requests.

The eight phases reuse the native facts, selection and byte-receipt contracts
from QualifiedNativeSession. No model forward is replayed inside an RPC: the
reviewed serving adapter supplies a completed real root and its real processor
scope. Discovery never promotes a profile to native/model acceptance.
"""
from contextlib import contextmanager
from dataclasses import asdict
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import threading

from .core import UnsupportedCall
from .deployments.matmul import _REQUIRED_BUNDLE
from .environment import environment_fingerprint, inspect_environment
from .model_discovery import inspect_structure
from .native import NativeCall, UninstalledObserver, _LEASE, _STACK, _read_proofs
from .providers import ProviderProfile
from .qualification import _validate_receipt
from .runtime_discovery import native_facts, plan_adaptation


MECHANISM = '20.buffer_substitute'
PHASES = ('discovery_1', 'discovery_2', 'reference', 'lossless', 'qualification_off',
          'fault', 'off_observed', 'disabled_off')
REVIEW_CONTRACT = 'fault20.current_typed_formals.single_target_lossless_off.v1'
_SHA = re.compile(r'[0-9a-f]{64}\Z')


def _require(condition, reason):
    if not condition:
        raise UnsupportedCall(reason)


def _sha(value):
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def load_reviewed_manifest(path, *, build_profile, first_live_validation):
    """Read a deployment review, not an RPC-provided promise of compatibility."""
    _require(first_live_validation is True, 'EXPLICIT_FIRST_LIVE_VALIDATION_REQUIRED')
    path = Path(path)
    _require(path.is_absolute() and path.is_file() and not path.is_symlink()
             and path.stat().st_size <= 1024 * 1024, 'REVIEWED_MANIFEST_FILE_REQUIRED')
    raw = path.read_bytes()
    value = json.loads(raw)
    _require(value.get('schema') == 'demoncat.fault20.serving_review.v1'
             and value.get('reviewed_contract') == REVIEW_CONTRACT
             and value.get('validation_level') == 'CPU_VERIFIED'
             and value.get('build_profile_sha256') == build_profile and _sha(build_profile)
             and isinstance(value.get('implementation_version'), str)
             and bool(value['implementation_version'].strip()), 'REVIEWED_BUILD_CONTRACT_MISMATCH')
    names = value.get('required_libraries')
    required = _REQUIRED_BUNDLE | ({'libopapi_math.so', 'libopapi_math_real.so'}
                                  if os.environ.get('DEMONCAT_F20_REQUIRE_TYPED_GWS') == '1' else set())
    _require(isinstance(names, list) and len(set(names)) == len(names)
             and all(isinstance(name, str) and Path(name).name == name for name in names)
             and required <= set(names), 'REVIEWED_LOADED_LIBRARIES_REQUIRED')
    cpu = value.get('cpu_evidence')
    _require(isinstance(cpu, list) and bool(cpu) and len(cpu) <= 64
             and isinstance(value.get('build_evidence'), dict), 'BUILD_AND_CPU_REVIEW_EVIDENCE_REQUIRED')
    for item in [value['build_evidence'], *cpu]:
        _require(isinstance(item, dict) and isinstance(item.get('path'), str)
                 and _sha(item.get('sha256')), 'REVIEW_EVIDENCE_DESCRIPTOR_INVALID')
        evidence = Path(item['path'])
        _require(evidence.is_absolute() and evidence.is_file() and not evidence.is_symlink()
                 and evidence.stat().st_size <= 16 * 1024 * 1024,
                 'BOUNDED_REVIEW_EVIDENCE_FILE_REQUIRED')
        _require(hashlib.sha256(evidence.read_bytes()).hexdigest() == item['sha256'],
                 'REVIEW_EVIDENCE_HASH_MISMATCH')
    return dict(value, manifest_sha256=hashlib.sha256(raw).hexdigest(), manifest_path=str(path))


class ServingQualifiedNativeSession:
    """The QualifiedNativeSession gates, driven by actual requests rather than replay()."""

    def __init__(self, model, processor, head, boundary, runtime, *, owner, generation,
                 reviewed_manifest_path, first_live_validation=False, evidence_dir,
                 baseline_control=False, _projection_candidates=None):
        _require(getattr(model, 'training', None) is False, 'LOADED_EVAL_MODEL_REQUIRED')
        self.model, self.processor, self.head, self.runtime = model, processor, head, runtime
        self.owner, self.generation = owner, generation
        _require(isinstance(owner, str) and bool(owner) and type(generation) is int and generation > 0,
                 'LIVE_PROVIDER_OWNER_REQUIRED')
        self.owner_thread = threading.get_ident()
        self.manifest = load_reviewed_manifest(reviewed_manifest_path, build_profile=runtime.build_profile,
                                               first_live_validation=first_live_validation)
        self.first_live_validation = first_live_validation
        _require(type(baseline_control) is bool, 'EXPLICIT_BASELINE_CONTROL_REQUIRED')
        self.baseline_control = baseline_control
        self.required_libraries = tuple(self.manifest['required_libraries'])
        self.boundary = copy.deepcopy(boundary)
        self.structure = inspect_structure(model)
        if _projection_candidates is None:
            rows = [row for row in self.structure['modules'] if row['object_identity'] == id(processor)]
            _require(len(rows) == 1 and rows[0]['path'] == boundary['processor_path']
                     and any(module is head for _, module in model.named_modules()),
                     'ACTUAL_UNIQUE_PROCESSOR_MODULE_REQUIRED')
            self.module_row = rows[0]
        else:
            _require(processor is None and head is None and bool(_projection_candidates)
                     and boundary.get('scope') == 'ACTUAL_ROOT_ACTIVE_QUANT_PROJECTION',
                     'EXPLICIT_PROJECTION_SCOPE_REQUIRED')
            self.module_row = None
        self.directory = Path(evidence_dir)
        self.phase_index = 0
        self.started = self.active = self.fault_started = self.recovered = False
        self.recovering = self.recovery_attempted = False
        self.failure = None
        self.last_native_failure = None
        self.drain_status = 'NOT_REQUESTED'
        self.rows, self.events, self.requests, self.seen_calls = [], [], set(), set()
        self.before = self.reference = self.baseline = self.selection = self.selected_facts = self.profile = None
        self.environment_before = self.environment_after = self.environment_hash = None
        self.environment_refusal = None
        self.current = self.scope = self.remove = None
        self.locked = False
        self.attempted = False
        self.installation_unknown = False

    @property
    def phase(self):
        return PHASES[self.phase_index] if self.phase_index < len(PHASES) else 'complete'

    def _thread(self):
        _require(threading.get_ident() == self.owner_thread, 'FOREIGN_MODEL_THREAD')

    def _observer_candidates(self):
        return ((self.module_row['path'], self.processor, ('serving_logits_processor',)),)

    def _dispatch_owner(self):
        return self.processor

    def _environment(self):
        actual = inspect_environment(additional_libraries=self.required_libraries)
        fingerprint = environment_fingerprint(actual, self.required_libraries)
        if actual.get('pid') != os.getpid() or not _sha(fingerprint):
            names = {row.get('name') for row in actual.get('loaded_libraries', ())}
            self.environment_refusal = dict(
                maps_status=actual.get('maps_status'), actual_pid=actual.get('pid'), expected_pid=os.getpid(),
                missing_required_libraries=sorted(set(self.required_libraries) - names),
                unverified_libraries=[{key: row.get(key) for key in ('name', 'path', 'status')}
                    for row in actual.get('loaded_libraries', ()) if row.get('status') != 'MAPPED_FILE_VERIFIED'])
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / 'environment_refusal.json').write_text(json.dumps(
                dict(diagnostic=self.environment_refusal, actual_environment=actual),
                indent=2, allow_nan=False) + '\n')
        _require(actual.get('pid') == os.getpid() and _sha(fingerprint),
                 'LOADED_WORKER_ENVIRONMENT_UNPROVEN')
        if self.environment_hash is not None:
            _require(fingerprint == self.environment_hash, 'LOADED_ENVIRONMENT_CHANGED')
        return actual, fingerprint

    def _paused(self, *, registered=True):
        states = self.runtime.snapshot()
        _require(self.runtime.operator in states, 'NATIVE_OPERATOR_STATE_MISSING')
        for operator, row in states.items():
            owned = operator == self.runtime.operator and registered
            _require(row.get('runtime_matched') is True and row.get('enabled') is False
                     and row.get('inflight') == row.get('leases') == 0
                     and row.get('recovery_required') is False
                     and row.get('native_registered') is owned,
                     'NATIVE_NOT_PAUSED_OR_OTHER_OWNER_PRESENT')
            if owned:
                _require(row.get('native_owned') is True and row.get('paused') is True
                         and row.get('request_controlled') is True,
                         'OWNED_PAUSED_CALLBACK_REQUIRED')
        return states

    def start(self):
        self._thread()
        _require(not self.started and self.remove is None, 'SINGLE_SERVING_ATTEMPT_REQUIRED')
        self.started = True  # Failed installation also consumes this instance.
        try:
            self._paused(registered=False)
            self.environment_before, self.environment_hash = self._environment()
            _require(_LEASE.acquire(blocking=False), 'ANOTHER_NATIVE_SESSION_OWNS_PROCESS')
            self.locked = True
            self.runtime.configure_request_lifecycle(owner=self.owner, generation=self.generation)
            self.installation_unknown = True
            try:
                self.remove = self.runtime.install_observer(self._dispatch, self._observer_candidates(), MECHANISM)
            except UninstalledObserver:
                self.installation_unknown = False
                raise
            self.installation_unknown = False
            _require(callable(self.remove), 'OWNED_CALLBACK_REMOVAL_REQUIRED')
            self._paused()
        except BaseException as error:
            self.failure = type(error).__name__ + ': ' + str(error)
            raise

    @contextmanager
    def processor_scope(self, processor):
        self._thread()
        _require(self.started and processor is self.processor and self.scope is None
                 and not _STACK.get(), 'REAL_PROCESSOR_SCOPE_REQUIRED')
        self.scope = _STACK.set(((id(self), processor, len(self.rows)),))
        try:
            yield
        finally:
            token = self.scope
            self.scope = None
            if token is not None:
                _STACK.reset(token)

    def begin(self, request_id, input_sha256, *, transaction):
        self._thread()
        _require(self.started and not self.active and self.scope is not None
                 and self.phase != 'complete' and (not self.failure or self.recovering),
                 'SERVING_PHASE_NOT_AVAILABLE')
        _require(self.model.training is False, 'LOADED_EVAL_MODEL_REQUIRED')
        _require(isinstance(request_id, str) and 0 < len(request_id) <= 256
                 and request_id not in self.requests and _sha(input_sha256),
                 'FRESH_ACTUAL_SCHEDULER_REQUEST_REQUIRED')
        _require(transaction.get('root_forward_returned') is True
                 and transaction.get('processor_object_id') == id(self.processor)
                 and transaction.get('head_argument_object_id') == id(self.head)
                 and _sha(transaction.get('hidden_sha256')) and _sha(transaction.get('weight_sha256')),
                 'ACTUAL_ROOT_AND_PROCESSOR_ARGUMENTS_UNPROVEN')
        protected = {key: transaction[key] for key in ('hidden_sha256', 'weight_sha256')}
        protected['input_sha256'] = input_sha256
        if self.baseline is not None:
            _require(protected == self.baseline['protected'], 'COMPLETE_REPLAY_INPUT_CHANGED')
        self.before = self._paused()
        self._environment()
        self.current = dict(phase=self.phase, request_id=request_id, protected=protected,
                            transaction=copy.deepcopy(transaction), events_start=len(self.events))
        self.requests.add(request_id)
        self.active = True
        if self.phase != 'disabled_off':
            self.runtime.resume()

    def _dispatch(self, call, original):
        self._thread()
        _require(self.active and self.current is not None and self.phase != 'disabled_off'
                 and self.scope is not None and isinstance(call, NativeCall)
                 and call.owner is self._dispatch_owner() and callable(original), 'ACTUAL_SCOPED_CALLBACK_REQUIRED')
        _require(call.call_id not in self.seen_calls, 'NATIVE_CALL_TOKEN_REUSED')
        self.seen_calls.add(call.call_id)
        event = dict(path=self.module_row['path'], object_identity=id(self._dispatch_owner()),
                     aliases=self.module_row['aliases'], canonical_roles=self.module_row['canonical_roles'],
                     role_status=self.module_row['role_status'], call_id=call.call_id, operator=call.operator,
                     forward=len(self.rows), scope_ordinal=(0, 0), actual_original_calls=0,
                     phase=self.phase, status='observed', source='deployment_native_observer')
        self.events.append(event)
        try:
            _require(len(self.events) - self.current['events_start'] == 1,
                     'NATURAL_PROCESSOR_NATIVE_TARGET_NOT_UNIQUE')
            proofs, details = _read_proofs(call, MECHANISM)
            facts = native_facts(details.get('runtime_call'), details.keys(), MECHANISM)
            current = details['runtime_call'].get('current_typed_call', {})
            _require(current.get('source') == 'typed_gws_execute_scope'
                     and current.get('executor_bound') is True
                     and current.get('formal_params_current') is True
                     and current.get('cached_layout_used') is False
                     and type(current.get('gws_token')) is int and current['gws_token'] > 0,
                     'CURRENT_TYPED_FORMALS_REQUIRED')
            _require(call.operator == self.runtime.operator and facts['worker'] == self.runtime.worker
                     and facts['build_profile_sha256'] == self.runtime.build_profile,
                     'ACTUAL_CALLBACK_WORKER_OR_BUILD_CHANGED')
            # Preserve the actual native and public observations even if their
            # correspondence is refused. A failed check is never proof of a
            # qualified target, but must remain diagnosable without a retry.
            event.update(proofs=proofs, native_facts=facts, current_typed_call=current,
                         observed_transaction=copy.deepcopy(self.current['transaction']),
                         formal_verification='PENDING')
            self._verify_formals(current, facts, self.current['transaction'])
            event['formal_verification'] = 'VERIFIED'

            def once():
                _require(event['actual_original_calls'] == 0, 'NATIVE_CONTINUATION_ALREADY_CONSUMED')
                event['actual_original_calls'] = 1
                return original()

            if self.phase.startswith('discovery_'):
                result = once()
                _require(result == 0, 'DISCOVERY_NATIVE_EXECUTION_FAILED')
                event['status'] = 'observed_readonly'
                return result
            _require(self.selection is not None and facts == self.selected_facts
                     and event['scope_ordinal'] == self.selection['scope_ordinal'],
                     'SELECTED_NATIVE_TARGET_CHANGED')
            stage = ('reference' if self.phase in ('qualification_off', 'off_observed') else self.phase)
            if stage == 'fault':
                import torch
                _require(not self.attempted and self.reference is not None
                         and not torch.is_grad_enabled() and self.model.training is False,
                         'SINGLE_QUALIFIED_INFERENCE_FAULT_REQUIRED')
                self.attempted = True  # Consume before the uncertain native call.
                if self.baseline_control:
                    stage = 'reference'
                else:
                    self.fault_started = True
            result, receipt = self.runtime.execute_scoped(call, once,
                                                          {'reference': 1, 'lossless': 2, 'fault': 3}[stage])
            _require(event['actual_original_calls'] == 1, 'EXACTLY_ONE_NATIVE_CONTINUATION_REQUIRED')
            _validate_receipt(receipt, stage, call.operator,
                              sum(slot['formal'] != 'out' for slot in facts['slots']))
            if self.reference is not None:
                for name in ('input_sha256_before', 'input_sha256_after'):
                    _require(receipt[name] == self.reference[name], 'QUALIFIED_NATIVE_INPUT_BYTES_CHANGED')
                if stage != 'fault':
                    _require(receipt['output_sha256'] == self.reference['output_sha256'],
                             'HEALTHY_NATIVE_OUTPUT_BYTES_CHANGED')
            event.update(status='routed', executed_stage=self.phase, provider_observation=receipt)
            return result
        except BaseException as error:
            event.update(status='failed', reason=type(error).__name__ + ': ' + str(error))
            # ctypes must return an ACL error; the subsequent always_call hook
            # sees output=None. Preserve this original refusal before that
            # secondary output error, without trying to pause an active Run.
            self.last_native_failure = event['reason']
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / 'failed_native_event.json').write_text(json.dumps(
                event, indent=2, allow_nan=False) + '\n')
            raise

    def _verify_formals(self, current, facts, transaction):
        """Match actual typed native geometry/storage to the observed head argument.

        QMM x1 is a quantized intermediate: its shape is checked, never asserted
        byte-identical to the floating hidden input. Native slot/address checks
        remain mandatory for all present formals and all optional scales.
        """
        rows = current.get('formals')
        _require(isinstance(rows, list) and bool(rows)
                 and all(isinstance(row, dict) and isinstance(row.get('formal'), str) for row in rows)
                 and len({row['formal'] for row in rows}) == len(rows), 'CURRENT_FORMAL_METADATA_REQUIRED')
        formals = {row['formal']: row for row in rows}
        slots = {row['formal']: row for row in facts['slots']}
        _require(set(slots) == {row['formal'] for row in rows if row.get('present') is True},
                 'CURRENT_FORMAL_SLOT_SET_CHANGED')
        for name, slot in slots.items():
            row = formals[name]
            _require(row.get('run_slot') == slot['slot'] and row.get('storage_bytes') == slot['storage_bytes']
                     and type(row.get('storage_addr')) is int and row['storage_addr'] > 0
                     and row.get('run_address') == row['storage_addr'], 'CURRENT_FORMAL_RUN_SLOT_MISMATCH')
        hidden, weight = transaction.get('hidden'), transaction.get('weight')
        _require(isinstance(hidden, dict) and isinstance(weight, dict), 'ACTUAL_TENSOR_GEOMETRY_REQUIRED')
        input_name, weight_name = ('self', 'mat2') if self.runtime.operator == 'aclnnMatmulWeightNz' else ('x1', 'x2')
        _require({input_name, weight_name, 'out'} <= set(slots), 'REQUIRED_OUTPUT_PROJECTION_FORMALS_MISSING')
        left, right, out = (formals[name] for name in (input_name, weight_name, 'out'))
        _require(left.get('view_shape') == hidden['shape'] == facts['layout']['shape']
                 and left.get('strides') == facts['layout']['stride']
                 and left.get('view_offset') == facts['layout']['offset']
                 and out.get('view_shape') == [hidden['shape'][0], weight['shape'][0]],
                 'ACTUAL_OUTPUT_PROJECTION_NATIVE_SHAPE_MISMATCH')
        _require(right.get('storage_addr') == weight['storage_addr']
                 and right.get('storage_bytes') == weight['storage_bytes']
                 and right.get('view_offset') == weight['storage_offset']
                 and right.get('element_size') == weight['element_size']
                 and right.get('format') == weight['actual_format'], 'ACTUAL_HEAD_STORAGE_MISMATCH')
        direct = (right.get('view_shape') == weight['shape'] and right.get('strides') == weight['stride'])
        transposed = (right.get('view_shape') == list(reversed(weight['shape']))
                      and right.get('strides') == list(reversed(weight['stride'])))
        _require(transposed or (input_name == 'x1' and direct), 'ACTUAL_HEAD_LOGICAL_LAYOUT_MISMATCH')
        if input_name == 'self':
            _require(left.get('storage_addr') == hidden['storage_addr']
                     and left.get('storage_bytes') == hidden['storage_bytes']
                     and left.get('view_offset') == hidden['storage_offset']
                     and left.get('strides') == hidden['stride']
                     and left.get('element_size') == hidden['element_size']
                     and left.get('format') == hidden['actual_format']
                     and facts['layout']['dtype'] == {'torch.bfloat16': 'BF16', 'torch.float16': 'FP16'}.get(hidden['dtype'])
                     and hidden['dtype'] == weight['dtype'], 'ACTUAL_HIDDEN_STORAGE_OR_DTYPE_MISMATCH')
        else:
            _require(facts['layout']['dtype'] == 'INT8' and left.get('element_size') == 1
                     and weight['dtype'] == 'torch.int8', 'QUANTIZED_OUTPUT_PROJECTION_UNPROVEN')

    def _profile_from_discovery(self):
        facts, event = self.events[0]['native_facts'], self.events[0]
        layout, worker = facts['layout'], facts['worker']
        material = dict(review=self.manifest, boundary=self.boundary,
                        events=self.events, requests=self.rows,
                        environment_fingerprint=self.environment_hash)
        self.profile = ProviderProfile(
            implementation_version=self.manifest['implementation_version'], validation_level='CPU_VERIFIED',
            evidence_sha256=_digest(material), environment_fingerprint=self.environment_hash,
            required_libraries=self.required_libraries, build_profile_sha256=facts['build_profile_sha256'],
            operator=event['operator'], boundary=facts['boundary'], native_signature=facts['native_signature'],
            layout_sha256=facts['layout_sha256'], soc=worker['soc'], dtype=layout['dtype'],
            format=layout['format'], ndim=len(layout['shape']), execution_mode=worker['execution_mode'],
            tp=worker['tp'], rank=worker['rank'], logical_device=worker['logical_device'],
            model_class=self.structure['model_class'], model_type=self.structure['model_type'])
        report = dict(mechanism=MECHANISM, events=copy.deepcopy(self.events), closed=False,
                      active=True, cleanup_errors=[], cleanup_pending=False, rejected_root_calls=[],
                      forwards=[dict(index=i, status='returned', errors=[]) for i in range(len(self.rows))],
                      module_invocations=[dict(observed_calls=1) for _ in self.rows], unobserved_candidates=[])
        # Count only completed real requests and the actually scoped owner.
        # A projection transaction is closed by subsequent complete logits;
        # the head-scope implementation owns its actual processor object.
        decision = plan_adaptation(report, self.structure, (self.profile,), self.environment_before,
                                   self.environment_after, discovery_finished=True,
                                   first_live_validation=self.first_live_validation)
        _require(decision['status'] == 'LOSSLESS_GATE_REQUIRED',
                 'AUTOMATIC_SERVING_TARGET_REFUSED:' + decision['status'])
        self.selection, self.selected_facts = decision['target'], copy.deepcopy(facts)
        slots = facts['slots']
        required = {'self', 'mat2', 'out'} if self.runtime.operator == 'aclnnMatmulWeightNz' else {'x1', 'x2', 'out'}
        _require(required <= {slot['formal'] for slot in slots}
                 and len({slot['slot'] for slot in slots}) == len(slots)
                 and facts.get('cache_lifecycle') and worker.get('worker_id') and worker.get('model_instance_id'),
                 'DISTINCT_LIVE_NATIVE_SLOTS_AND_MODEL_LOAD_REQUIRED')
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / 'reviewed_profile_draft.json').write_text(json.dumps(
            dict(profile=asdict(self.profile), material=material, decision=decision,
                 acceptance='FIRST_LIVE_VALIDATION_ONLY_NOT_AUTOMATIC_NATIVE_PROMOTION'),
            indent=2, allow_nan=False) + '\n')

    def end(self, model_output_sha256, *, model_output_finite):
        self._thread()
        _require(self.active and self.current is not None, 'NO_ACTIVE_SERVING_TRANSACTION')
        try:
            self.runtime.pause()
            self._paused()
            self.environment_after, _ = self._environment()
            _require(_sha(model_output_sha256) and type(model_output_finite) is bool,
                     'COMPLETE_MODEL_OUTPUT_EVIDENCE_REQUIRED')
            phase = self.phase
            events = self.events[self.current['events_start']:]
            _require(len(events) == (0 if phase == 'disabled_off' else 1)
                     and all(event['status'] in ('observed_readonly', 'routed') for event in events),
                     'EXACT_PHASE_NATIVE_CALL_COUNT_REQUIRED')
            if phase != 'fault' or self.baseline_control:
                _require(model_output_finite, 'HEALTHY_MODEL_OUTPUT_NOT_FINITE')
                if self.baseline is not None:
                    _require(model_output_sha256 == self.baseline['model_output_sha256'],
                             'HEALTHY_COMPLETE_MODEL_OUTPUT_CHANGED')
            row = {key: copy.deepcopy(value) for key, value in self.current.items() if key != 'events_start'}
            row.update(model_output_sha256=model_output_sha256, model_output_finite=model_output_finite,
                       native_calls=len(events), native=[copy.deepcopy(event) for event in events],
                       paused_after=True)
            if phase == 'fault':
                row['model_output_changed'] = model_output_sha256 != self.baseline['model_output_sha256']
                row['baseline_control'] = self.baseline_control
            self.rows.append(row)
            if phase == 'discovery_1':
                self.baseline = row
            elif phase == 'discovery_2':
                self._profile_from_discovery()
            elif phase == 'reference':
                self.reference = copy.deepcopy(events[0]['provider_observation'])
            self.phase_index += 1
            self.current = None
            self.active = False
            self.drain_status = 'DISABLED_CONFIRMED'
            if self.phase == 'complete':
                self.recovered = True
                self.recovering = False
        except BaseException as error:
            self.fail(type(error).__name__ + ': ' + str(error))
            raise

    def fail(self, reason):
        self._thread()
        if self.last_native_failure and not self.failure:
            self.failure = self.last_native_failure
        self.failure = (self.failure + '; ' if self.failure else '') + str(reason)
        self.active = False
        self.current = None
        self.recovering = False
        try:
            if self.remove is not None:
                self.runtime.pause()
                self._paused()
            else:
                self._paused(registered=False)
            self.drain_status = 'DISABLED_CONFIRMED'
        except BaseException:
            self.drain_status = 'RECOVERY_PENDING'
            raise

    def request_recovery(self):
        self._thread()
        _require(self.failure and self.fault_started and not self.recovery_attempted and not self.active,
                 'RECOVERY_SEQUENCE_NOT_AVAILABLE')
        self.runtime.pause()
        self._paused()
        self.recovery_attempted = True
        self.phase_index = PHASES.index('off_observed')
        self.recovering = True

    def close(self):
        self._thread()
        _require(not self.active and self.scope is None, 'ACTIVE_PROCESSOR_CANNOT_CLOSE')
        _require(not self.fault_started or self.recovered, 'OFF_RECOVERY_REQUIRED_BEFORE_CLEAN')
        _require(not self.installation_unknown, 'NATIVE_INSTALLATION_RECOVERY_UNKNOWN')
        if self.remove is not None:
            self.runtime.pause()
            self._paused()
            self.remove()
            self.remove = None
        self._paused(registered=False)
        if self.locked:
            _LEASE.release()
            self.locked = False

    def query(self):
        return {'qualification': dict(
            status=('RECOVERED_ATTEMPT_FAILED' if self.recovered and self.failure else
                    'BASELINE_CONTROL_COMPLETED' if self.recovered and self.baseline_control else
                    'SCOPED_ATTEMPT_COMPLETED' if self.recovered else
                    'ATTEMPT_FAILED' if self.failure else 'WAITING_FOR_REAL_REQUEST'),
            next_phase=self.phase, recovery_verified=self.recovered, failure=self.failure,
            last_native_failure=self.last_native_failure,
            environment_refusal=copy.deepcopy(self.environment_refusal),
            drain_status=self.drain_status, recovering=self.recovering,
            completed_phases=[row['phase'] for row in self.rows],
            model_output_changed=any(row.get('model_output_changed') for row in self.rows),
            actual_request_ids=[row['request_id'] for row in self.rows],
            selected=copy.deepcopy(self.selection), profile=asdict(self.profile) if self.profile else None,
            first_live_validation=self.first_live_validation, manifest_sha256=self.manifest['manifest_sha256'],
            baseline_control=self.baseline_control, fault_native_attempted=self.fault_started,
            coverage_scope='ONE_NATURAL_OUTPUT_PROJECTION_NOT_ALL_ELIGIBLE',
            transaction_scope='REAL_ROOT_FORWARD_RETURNED_THEN_REAL_PROCESSOR_SCOPE')}


class ProjectionQualifiedNativeSession(ServingQualifiedNativeSession):
    """The same native gates, inside an observed quantized projection.

    The real root is still executing when QMM runs. Completion is certified
    later by the real processor's complete logits; no head call is invented.
    Selection is the first actual dynamic projection consuming the first
    embedding-normalizer lineage, then frozen across the two healthy requests.
    """

    def __init__(self, model, candidates, runtime, **kwargs):
        _require(runtime.operator == 'aclnnQuantMatmulWeightNz', 'QMM_PROJECTION_OPERATOR_REQUIRED')
        self.projection_candidates = tuple(candidates)
        self.bound_module = None
        boundary = dict(scope='ACTUAL_ROOT_ACTIVE_QUANT_PROJECTION',
                        selection='FIRST_ACTUAL_EMBED_NORM_DYNAMIC_PROJECTION',
                        completion='ACTUAL_COMPLETE_GLOBAL_LOGITS',
                        candidate_paths=[row[0] for row in candidates])
        super().__init__(model, None, None, boundary, runtime,
                         _projection_candidates=self.projection_candidates, **kwargs)
        objects = {id(module): path for path, module, *_ in candidates}
        _require(len(objects) == len(candidates)
                 and all(any(row['object_identity'] == identity and row['path'] == path
                             for row in self.structure['modules']) for identity, path in objects.items()),
                 'ACTUAL_UNALIASED_PROJECTION_CANDIDATES_REQUIRED')

    def _observer_candidates(self):
        return tuple((path, module, ('actual_dynamic_quant_projection',))
                     for path, module, *_ in self.projection_candidates)

    def _dispatch_owner(self):
        return self.bound_module

    @contextmanager
    def projection_scope(self, module):
        self._thread()
        _require(self.started and self.scope is None and not _STACK.get()
                 and any(module is candidate[1] for candidate in self.projection_candidates),
                 'REAL_QUANT_PROJECTION_SCOPE_REQUIRED')
        if self.bound_module is None:
            _require(self.phase == 'discovery_1', 'HEALTHY_FIRST_PROJECTION_REQUIRED')
            self.bound_module = module
            rows = [row for row in self.structure['modules'] if row['object_identity'] == id(module)]
            _require(len(rows) == 1, 'PROJECTION_OBJECT_NOT_UNIQUE')
            self.module_row = rows[0]
        _require(module is self.bound_module, 'SELECTED_PROJECTION_CHANGED')
        self.scope = _STACK.set(((id(self), module, len(self.rows)),))
        try:
            yield
        finally:
            token, self.scope = self.scope, None
            if token is not None:
                _STACK.reset(token)

    def begin(self, request_id, input_sha256, *, transaction):
        self._thread()
        _require(self.started and not self.active and self.scope is not None
                 and self.phase != 'complete' and (not self.failure or self.recovering),
                 'SERVING_PHASE_NOT_AVAILABLE')
        _require(self.model.training is False and isinstance(request_id, str)
                 and 0 < len(request_id) <= 256 and request_id not in self.requests and _sha(input_sha256),
                 'FRESH_ACTUAL_SCHEDULER_REQUEST_REQUIRED')
        _require(transaction.get('root_forward_active') is True
                 and transaction.get('projection_object_id') == id(self.bound_module)
                 and transaction.get('quant_method_object_id') == id(self.bound_module.quant_method)
                 and transaction.get('quant_apply_live_identity') == id(self.bound_module.quant_method.apply)
                 and transaction.get('source') == 'ACTUAL_QUANT_APPLY_AND_PUBLIC_QMM'
                 and transaction.get('quantization_output_storage_verified') is True
                 and transaction.get('native_x1_equals_float_input_storage') is False
                 and 'root_forward_returned' not in transaction and 'head_argument_object_id' not in transaction,
                 'ACTUAL_PROJECTION_AND_QUANTIZATION_LINEAGE_REQUIRED')
        hashes = transaction.get('input_hashes')
        _require(isinstance(hashes, dict) and {'float_input', 'x1', 'x2', 'x1_scale', 'x2_scale'} <= set(hashes)
                 and all(_sha(value) for value in hashes.values()), 'COMPLETE_ACTUAL_QUANT_INPUT_HASHES_REQUIRED')
        protected = dict(hashes, input_sha256=input_sha256)
        if self.baseline is not None:
            _require(protected == self.baseline['protected'], 'COMPLETE_REPLAY_INPUT_CHANGED')
        self.before = self._paused()
        self._environment()
        self.current = dict(phase=self.phase, request_id=request_id, protected=protected,
                            transaction=copy.deepcopy(transaction), events_start=len(self.events))
        self.requests.add(request_id)
        self.active = True
        if self.phase != 'disabled_off':
            self.runtime.resume()

    def _verify_formals(self, current, facts, transaction):
        rows, actual = current.get('formals'), transaction.get('formals')
        _require(isinstance(rows, list) and isinstance(actual, dict)
                 and all(isinstance(row, dict) and isinstance(row.get('formal'), str) for row in rows)
                 and len({row['formal'] for row in rows}) == len(rows), 'CURRENT_FORMAL_METADATA_REQUIRED')
        formals = {row['formal']: row for row in rows}
        slots = {row['formal']: row for row in facts['slots']}
        formal_order = ('x1', 'x2', 'x1_scale', 'x2_scale', 'y_scale', 'x1_offset',
                        'x2_offset', 'y_offset', 'bias', 'out')
        _require(tuple(formals) == formal_order and {'x1', 'x2', 'x1_scale', 'x2_scale'} <= set(actual)
                 and set(actual) <= {'x1', 'x2', 'x1_scale', 'x2_scale', 'bias'}
                 and set(slots) == {row['formal'] for row in rows
                                    if row.get('present') is True and row.get('storage_bytes', 0) > 0}
                 and set(slots) == set(actual) | {'out'}, 'CURRENT_FORMAL_SLOT_SET_CHANGED')
        output_dtype = {'torch.bfloat16': 27, 'torch.float16': 1}.get(transaction.get('output_dtype'))
        _require(output_dtype is not None and formals['out'].get('dtype') == output_dtype
                 and formals['out'].get('element_size') == 2 and formals['out'].get('format') == 2,
                 'ACTUAL_QMM_OUTPUT_DTYPE_UNPROVEN')
        # The real 8.5.1 lowering supplies three present but zero-length
        # optional tensors. They have formal ordinals, but no Run storage slot.
        # Do not infer a slot from `present`, drop nonempty formals, or permit
        # arbitrary empty metadata: only the observed placeholders are admitted.
        placeholders = []
        for name in set(formals) - set(slots):
            row = formals[name]
            _require(type(row.get('present')) is bool and row.get('run_slot') == -1
                     and all(row.get(key) == 0 for key in ('run_address', 'storage_addr', 'raw_addr',
                         'data_addr', 'storage_bytes', 'view_bytes', 'storage_elements', 'numel', 'view_offset')),
                     'OPTIONAL_QMM_FORMAL_HAS_UNBOUND_STORAGE:' + name)
            if row['present'] is True:
                _require(name in ('y_scale', 'x1_offset', 'y_offset')
                         and row.get('view_shape') == row.get('storage_shape') == [0]
                         and row.get('strides') == [1] and row.get('contiguous') is True
                         and row.get('dtype') == output_dtype and row.get('element_size') == 2
                         and row.get('format') == 2, 'UNREVIEWED_EMPTY_QMM_FORMAL:' + name)
                placeholders.append(name)
            else:
                _require(row.get('view_shape') == row.get('storage_shape') == row.get('strides') == []
                         and row.get('element_size') == row.get('dtype') == row.get('format') == 0
                         and row.get('contiguous') is False, 'INVALID_ABSENT_QMM_FORMAL:' + name)
        self.events[-1]['verified_empty_optional_formals'] = sorted(placeholders)
        for name, slot in slots.items():
            row = formals[name]
            _require(row.get('run_slot') == slot['slot'] and row.get('storage_bytes') == slot['storage_bytes']
                     and type(row.get('storage_addr')) is int and row['storage_addr'] > 0
                     and row.get('run_address') == row['storage_addr'], 'CURRENT_FORMAL_RUN_SLOT_MISMATCH')
            if name == 'out':
                continue
            tensor = actual[name]
            _require(all(row.get(native) == tensor.get(public) for native, public in (
                ('storage_addr', 'storage_addr'), ('storage_bytes', 'storage_bytes'),
                ('view_offset', 'storage_offset'), ('view_shape', 'shape'), ('strides', 'stride'),
                ('element_size', 'element_size'), ('format', 'actual_format'))),
                'ACTUAL_QUANT_FORMAL_STORAGE_OR_GEOMETRY_MISMATCH:' + name)
            _require(row.get('dtype') == {'torch.float32': 0, 'torch.float16': 1, 'torch.int8': 2,
                         'torch.int32': 3, 'torch.bfloat16': 27}.get(tensor.get('dtype')),
                     'ACTUAL_QUANT_FORMAL_DTYPE_MISMATCH:' + name)
        left, right, out = formals['x1'], formals['x2'], formals['out']
        _require(facts['layout']['dtype'] == 'INT8'
                 and actual['x1']['dtype'] == actual['x2']['dtype'] == 'torch.int8'
                 and left['view_shape'] == facts['layout']['shape']
                 and left['strides'] == facts['layout']['stride']
                 and left['view_offset'] == facts['layout']['offset']
                 and len(left['view_shape']) == len(right['view_shape']) == 2
                 and left['view_shape'][1] == right['view_shape'][0]
                 and out.get('view_shape') == [left['view_shape'][0], right['view_shape'][1]],
                 'ACTUAL_QUANT_PROJECTION_NATIVE_SHAPE_MISMATCH')

    def pause_projection(self):
        self._thread()
        _require(self.active and self.scope is not None, 'ACTIVE_PROJECTION_SCOPE_REQUIRED')
        self.runtime.pause()
        self._paused()

    def query(self):
        value = super().query()
        value['qualification'].update(
            coverage_scope='ONE_FIRST_EMBED_NORM_QUANT_PROJECTION_NOT_ALL_ELIGIBLE',
            transaction_scope='ACTUAL_ROOT_ACTIVE_QUANT_PROJECTION_THEN_COMPLETE_GLOBAL_LOGITS',
            selection_policy='FIRST_ACTUAL_EMBED_NORM_DYNAMIC_PROJECTION',
            floating_input_is_native_x1=False)
        return value
