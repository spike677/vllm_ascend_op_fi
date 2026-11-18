"""Bounded one-site natural Quant C1 through the existing Worker provider.

Two complete censuses precede automatic selection. Unadapted Attention/MoE
rows remain in the original inventory; this adapter grants no C2/C3 proof.
"""
from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading

from fault_injection.quant_group_native import Config, Lease, QuantGroupNative, key, snapshot as native_snapshot
from fault_injection.deployments.quant_group_c1 import public_views_match, verify_completion
from .quant_row_provenance import digest, metadata, snapshot
from .serving_context import require, require_serial_eager
from .serving_quant import MECHANISM, QuantServingSession, same_tensor
from .serving_shared_scale_group_inventory import GroupInventoryRuntime, SharedScaleGroupInventorySession
from .shared_scale_group_inventory import SharedScaleGroupInventory


PHASES = ('discovery_1', 'discovery_2', 'reference', 'lossless', 'qualification_off', 'fault', 'off')
MODES = dict(reference=1, lossless=2, fault=3, qualification_off=4)
SCOPE = 'ONE_AUTOMATICALLY_PROVEN_NATURAL_B8_SITE_ONCE_NOT_ALL_ELIGIBLE'


@dataclass(frozen=True)
class QuantGroupC1ServingBinding:
    make_session: object


class GroupC1Runtime(GroupInventoryRuntime):
    def native_state(self):
        state = self.binding.query()
        core, census = state['core'], state['census']
        return dict(owned=int(self.binding.config is not None or core['installed'] or state['quarantined']
                or state['abi_exceptions'] or state['orphan_descriptors']),
            armed=int(bool(core['pending_parent'])),
            pending=core['pending_parent'] + census['preparing'] + census['pending'],
            executing=census['executing'] + int(bool(state['native_stack_active']))
                + state['callback_active'] + state['completion_active'], raw=state)


class C1Inventory(SharedScaleGroupInventory):
    def __init__(self, *args, session, **kwargs):
        self.session = session
        super().__init__(*args, **kwargs)

    @property
    def phase(self):
        return self.session.phase_name() if len(self.session.rows) < len(PHASES) else 'complete'

    def invoke_dynamic(self, original, args, kwargs):
        return self.session.invoke_dynamic(self, original, args, kwargs)

    def verify_dynamic(self, events, x, q, scale):
        done = self.native_call.get('completion')
        if done is None:
            return super().verify_dynamic(events, x, q, scale)
        lease = self.session.current_lease
        require(lease is not None and self.native_call['phase'] == self.session.phase_name(),
                'CURRENT_C1_NATIVE_LEASE_REQUIRED')
        public_views_match(done['call'], x, q, scale, self.native.device)
        proof = verify_completion(done, events, lease, released=True)
        packet = dict(phase=self.phase, lease=native_snapshot(lease), completion=done, raw_events=events,
            row_proof=self.native_call['row_proof'], binding=proof)
        packet['artifact_sha256'] = digest(packet)
        self.session.receipts.append(snapshot(packet))
        return proof


def stable_geometry(view):
    return {name: view[name] for name in ('bytes', 'rows', 'columns', 'stride0', 'stride1',
        'offset', 'dtype', 'format', 'dimensions', 'reserved')}


def row_signature(proof):
    require(proof.get('verified') is True and isinstance(proof.get('actual_rows'), dict),
            'CURRENT_ACTUAL_REQUEST_ROW_PROOF_REQUIRED')
    return QuantServingSession.row_signature(proof['actual_rows'])


def closed_projection_prefix(public, projection, forward):
    """A completed target/consumer prefix may precede an unrelated unknown path.

    Whole-forward refusals remain untouched. No later event can erase an error
    that already existed at either actual query snapshot.
    """
    inputs, outputs = public.get('request_row_lineage', {}), projection.get('output_row_lineage', {})
    trace = forward.get('row_provenance') or {}
    require(inputs.get('verified') is True and outputs.get('verified') is True
            and projection['public_chain_verified'] and not projection['errors'] and not public['errors'],
            'C1_COMPLETED_TARGET_AND_QMM_ROW_PREFIX_REQUIRED')
    left, right = inputs.get('observation_prefix', {}), outputs.get('observation_prefix', {})
    require(left and right and left['observer_id'] == right['observer_id']
            and left['forward_id'] == right['forward_id'] == forward['forward_id']
            and inputs['forward_id'] == outputs['forward_id'] == forward['forward_id']
            and inputs['actual_rows']['request_ids'] == outputs['actual_rows']['request_ids'] == public['request_ids']
            and row_signature(inputs) == row_signature(outputs)
            and 0 < left['observed_at_monotonic_ns'] <= right['observed_at_monotonic_ns']
            and 0 <= left['actual_calls'] <= right['actual_calls'] <= trace['actual_calls']
            and all(value['actual_calls'] == value['retained_calls'] and not value['missing_calls']
                    and not value['errors'] and not value['source_errors'] for value in (left, right)),
            'C1_QUERY_PREFIX_TIME_REQUEST_OR_ERROR_BINDING_CHANGED')
    sources = trace.get('source_boundaries')
    require(not trace['active'] and not trace.get('cleanup_pending') and not trace.get('owned_dispatch_modes')
            and (sources is None or not sources['active'] and not sources.get('owned_nested_dispatch_modes')
                 and not sources.get('owned_profile_callbacks')), 'C1_PREFIX_OBSERVER_RESOURCES_NOT_RELEASED')
    for sequence in set(left['parent_source_sequences'] + right['parent_source_sequences']):
        matches = [row for row in (sources or {}).get('boundaries', []) if row['sequence'] == sequence]
        require(len(matches) == 1 and matches[0]['status'] == 'OBSERVED', 'C1_PREFIX_SOURCE_PARENT_NOT_CLOSED')
    return snapshot(dict(input=inputs, consumer=outputs, complete_at_consumer=True,
                         full_forward_row_complete=trace['complete']))


class QuantGroupC1ServingSession(SharedScaleGroupInventorySession):
    def __init__(self, *args, owner, generation, worker_identity, **kwargs):
        require(isinstance(owner, str) and bool(owner) and type(generation) is int and generation > 0,
                'EXISTING_WORKER_OWNER_AND_GENERATION_REQUIRED')
        self.owner, self.generation = owner, generation
        self.rpc_identity = snapshot(worker_identity)
        self.target = self.discovery = self.current_call = self.current_lease = None
        self.receipts, self.callbacks, self.native_recovery = [], [], []
        self.collection_errors = []
        self.projection_prefixes = {}
        self.callback_rows = {}
        self.phase_attempts, self.public_selected = set(), {}
        self.native_owner_started = self.native_owner_released = False
        self.recovery_synchronized_device = False
        self.fault_attempted = False
        super().__init__(*args, **kwargs)

    def _load_runtime(self, root, build_sha256):
        return GroupC1Runtime(QuantGroupNative.load(root, approved_build_sha256=build_sha256))

    def make_inventory(self, identity):
        return C1Inventory(self.model, self.candidates, session=self, worker_identity=identity,
            native=self.native_reader, folder=self.folder / 'inventory', row_lineage=self.row_lineage,
            row_transfer=self.projection_rows)

    def phase_name(self):
        return PHASES[len(self.rows)] if len(self.rows) < len(PHASES) else 'complete'

    def allow_nonfinite_output(self):
        return (self.phase_name() == 'fault' and self.active is not None and bool(self.active.get('decode'))
                and any(row['phase'] == 'fault' for row in self.receipts))

    def expect(self, request_ids):
        self.collect()
        require(self.installed and not self.aborted and not self.error and self.expected is None
                and self.phase_name() != 'complete', 'BOUNDED_IDLE_C1_GENERATION_REQUIRED')
        require(isinstance(request_ids, list) and len(request_ids) == len(set(request_ids)) == 8
                and all(isinstance(value, str) and value and value not in self.seen for value in request_ids),
                'EIGHT_FRESH_ACTUAL_REQUEST_IDS_REQUIRED')
        self.expected = {value: dict(logical_id=index, prompt=self.prompts[index]) for index, value in enumerate(request_ids)}
        self.decode_result = self.prefill_result = None
        self.forward_count = 0
        self.capture_started = False

    def pre(self, module, args, kwargs):
        super().pre(module, args, kwargs)
        self.active['phase'] = self.phase_name()

    def inventory_worker_identity(self):
        return dict(status='WORKER_IDENTIFIED', quarantined=False, worker=self._identity(),
                    identity_source='ACTUAL_PROVIDER_RPC_CONTEXT_WITH_LOCAL_LIFETIME_RECHECK')

    def _identity(self, *, recovering=False):
        # The provider was created from Runtime's current identity under its
        # non-reentrant RPC lock. Never recursively call the identity RPC here.
        from fault_injection.runtime_discovery import _process_start_identity
        import torch_npu
        actual = self.rpc_identity
        configured = self.worker.device
        require(actual['pid'] == os.getpid() and actual['process_start_identity'] == _process_start_identity()
                and self.worker.rank == actual['rank'] == 0 and self.worker.model_runner is self.runner
                and self.runner.model is self.model and threading.get_ident() == self.thread
                and configured.type == 'npu' and configured.index == actual['configured_logical_device']
                and configured.index == actual['logical_device'] == actual['observed_logical_device']
                and configured.index == self.native_reader.device == int(torch_npu.npu.current_device())
                and bool(actual['worker_start_identity']) and bool(actual['model_instance_id']),
                'C1_SAME_ACTUAL_TP1_WORKER_REQUIRED')
        return snapshot(actual)

    def provider_call_leases(self, state):
        # Registration/retained B buffers remain explicitly native_registered
        # and in raw host_bytes. A sealed target cannot self-arm: admission also
        # needs the current owned public scope and its exact live row proof.
        return state['pending'] + state['raw']['public_active'] + int(self.current_call is not None)

    def begin_observation(self):
        self._identity()
        require(not self.capture_started and self.current_call is None
                and not any(self.state()[name] for name in ('pending', 'executing')),
                'C1_CURRENT_NATIVE_CALL_NOT_IDLE')
        phase = self.phase_name()
        if phase == 'reference':
            require(self.target is not None and not self.native_owner_started, 'TWO_ACTUAL_CENSUSES_BEFORE_NATIVE_OWNER')
            owner_key = bytes.fromhex(digest(dict(owner=self.owner, worker=self.inventory.lifetime)))
            config = Config(owner_hi=int.from_bytes(owner_key[:8], 'little'), owner_lo=int.from_bytes(owner_key[8:16], 'little'),
                generation=self.generation, thread=threading.get_native_id(), device=self.native_reader.device,
                max_targets=1, max_calls=4, max_device_bytes=64 * 1024 * 1024, max_host_bytes=128 * 1024 * 1024)
            self.runtime.binding.install(config, self._observed, self._completed)
            self.native_owner_started = True
        if phase == 'fault':
            require([row['phase'] for row in self.receipts] == ['reference', 'lossless', 'qualification_off']
                    and set(self.projection_prefixes) == {'reference', 'lossless', 'qualification_off'}
                    and not self.fault_attempted, 'COMPLETE_SINGLE_SITE_REFERENCE_LOSSLESS_OFF_BEFORE_FAULT')
            # This scope contains exactly one independently proven site. The
            # full original inventory, including every unresolved site, remains
            # attached as evidence and is never relabeled all_eligible.
            scope = digest(dict(scope=SCOPE, target=self.target['identity'],
                discovery_sha256=self.discovery['sha256'], qualification=[dict(native=row['artifact_sha256'],
                    completed_projection_prefix=digest(self.projection_prefixes[row['phase']])) for row in self.receipts]))
            self.runtime.binding.seal([self.target['sha256']], scope, unresolved_candidates=0)
            self.fault_attempted = True
        self.capture_started = True
        if self.metadata_rows is not None:
            self.metadata_rows.begin(tuple(self.expected))

    def _select_target(self):
        report = self.inventory.report()
        require(len(report['native_forwards']) == 2 and len(report['forwards']) == 2
                and report['public_observer_complete'] and report['complete_healthy_replays']
                and all(row['complete_native_denominator'] for row in report['native_forwards']),
                'TWO_COMPLETE_ACTUAL_NATIVE_PUBLIC_CENSUSES_REQUIRED')
        self.discovery = dict(report=snapshot(report), sha256=digest(report))
        choices = []
        for forward in report['native_forwards']:
            by_module = {}
            for candidate in forward['candidates']:
                by_module.setdefault(candidate['module_object_identity'], []).append(candidate)
            choices.append(by_module)
        for first in report['native_forwards'][0]['candidates']:
            identity = first['module_object_identity']
            paired = choices[1].get(identity, [])
            if identity is None or len(choices[0][identity]) != 1 or len(paired) != 1:
                continue
            # This one reason can refer to an independent later path. Keep it
            # verbatim in the census; require the closed target/consumer prefix.
            if any(set(row['reasons']) - {'COMPLETE_REQUEST_ROW_OPERATOR_OBSERVATION_UNPROVEN'} for row in (first, paired[0])):
                continue
            public = [report['public_calls'][row['public_api_index']] for row in (first, paired[0])]
            projection = [[row for row in forward['projections'] if row['module_object_identity'] == identity]
                          for forward in report['forwards']]
            if any(len(rows) != 1 or not rows[0]['public_chain_verified'] for rows in projection):
                continue
            try:
                prefixes = [closed_projection_prefix(public[index], projection[index][0], report['native_forwards'][index])
                            for index in range(2)]
            except RuntimeError:
                continue
            fields = ('path', 'module_object_identity', 'quant_method_object_identity', 'method_source_sha256', 'weight_object_identity')
            if any(projection[0][0][name] != projection[1][0][name] for name in fields):
                continue
            if (public[0]['healthy_formals'] != public[1]['healthy_formals']
                    or not all(row['finite'] for row in public[0]['healthy_formals'].values())
                    or row_signature(public[0]['request_row_lineage']) != row_signature(public[1]['request_row_lineage'])):
                continue
            geometry = {name: stable_geometry(first['native']['parent'][name]) for name in ('input', 'quant', 'scale')}
            if geometry != {name: stable_geometry(paired[0]['native']['parent'][name]) for name in geometry}:
                continue
            module, method = self.inventory.by_id[identity][1:3]
            require(module.tp_size == 1 and module.tp_rank == 0, 'SINGLE_SITE_TP1_MODULE_REQUIRED')
            patch = next(item for item in self.inventory.patches if item[0] is method)
            function = getattr(patch[1], '__func__', patch[1])
            source = Path(inspect.getsourcefile(function))
            frozen = dict(scope=SCOPE, lifetime=self.inventory.lifetime, native_build=self.runtime.build_identity,
                projection={name: projection[0][0][name] for name in fields}, weight=metadata(module.weight),
                geometry=geometry, rows=row_signature(public[0]['request_row_lineage']))
            self.target = dict(identity=snapshot(frozen), sha256=digest(frozen), module=module, method=method,
                function=function, code=function.__code__, globals=function.__globals__, source=source,
                healthy_formals=snapshot(public[0]['healthy_formals']), qualification_prefixes=prefixes)
            return
        raise RuntimeError('NO_SINGLE_NATURAL_SITE_WITH_TWO_COMPLETE_INDEPENDENT_ROW_PROOFS')

    def _current_target(self, inventory, x):
        frame, target = inventory.frame, self.target
        if target is None or frame is None or frame['module'] is not target['module']:
            return None
        require(frame['row']['module_object_identity'] == target['identity']['projection']['module_object_identity']
                and frame['module'].quant_method is target['method'] and same_tensor(frame['input'], x)
                and metadata(frame['module'].weight) == target['identity']['weight']
                and target['function'].__code__ is target['code'] and target['function'].__globals__ is target['globals']
                and hashlib.sha256(target['source'].read_bytes()).hexdigest() == target['identity']['projection']['method_source_sha256'],
                'FROZEN_C1_MODULE_METHOD_WEIGHT_OR_INPUT_OWNER_CHANGED')
        proof = inventory.row_lineage(x)
        require(row_signature(proof) == target['identity']['rows']
                and proof['actual_rows']['request_ids'] == self.active['decode']['request_ids']
                and proof['forward_id'] == inventory.current['forward_id'], 'C1_CURRENT_FORWARD_REQUEST_ROWS_CHANGED')
        actual = native_snapshot(QuantGroupNative.tensor_view(x, 'input', self.native_reader.device))
        require(stable_geometry(actual) == target['identity']['geometry']['input'], 'C1_FROZEN_NATIVE_INPUT_GEOMETRY_CHANGED')
        return snapshot(proof)

    def invoke_dynamic(self, inventory, original, args, kwargs):
        x = args[0] if args else kwargs.get('input')
        proof = self._current_target(inventory, x)
        phase = self.phase_name()
        if proof is None:
            return original(*args, **kwargs)
        self.public_selected[phase] = self.public_selected.get(phase, 0) + 1
        require(self.public_selected[phase] == 1, 'C1_SELECTED_SITE_REPEATED_IN_ONE_ACTUAL_FORWARD')
        if phase not in MODES:
            return original(*args, **kwargs)
        require(self.current_call is None and phase not in self.phase_attempts, 'ONE_C1_ATTEMPT_PER_PHASE')
        context = dict(input=x, phase=phase, row_proof=proof, completions=[])
        self.current_call = context
        self.current_lease = None
        try:
            output, done = self.runtime.binding.invoke(x, lambda: original(*args, **kwargs))
            require(len(context['completions']) == 1 and context['completions'][0]['call'] == done['call'],
                    'ONE_REAL_SELECTED_COMPLETION_REQUIRED')
            inventory.native_call.update(completion=done, phase=phase, row_proof=proof)
            return output
        except BaseException as error:
            self.error = repr(error)
            inventory.native_call.update(selected_failure=repr(error), completions=context['completions'])
            raise
        finally:
            self.current_call = None

    def _observed(self, call):
        binding = self.runtime.binding
        raw = binding.raw(call.token)
        require(raw['kind'] == 1 and raw['parent'] == native_snapshot(call), 'INDEPENDENT_NATIVE_PARENT_BEFORE_C1_ADMISSION_REQUIRED')
        require(call.token not in self.callback_rows and len(self.callbacks) < 8192, 'BOUNDED_UNIQUE_NATIVE_PARENT_CALLBACK_REQUIRED')
        row = dict(call=native_snapshot(call), phase=self.phase_name(), selected=self.current_call is not None, completed=False)
        self.callbacks.append(row)
        self.callback_rows[call.token] = row
        context = self.current_call
        if context is None:
            return binding.pass_call(call)
        self._identity()
        phase = context['phase']
        require(phase == self.phase_name() and phase in MODES and phase not in self.phase_attempts
                and call.thread == threading.get_native_id() and call.device == self.native_reader.device
                and call.lifetime_token > 0 and not call.reserved, 'C1_ACTUAL_CURRENT_CALL_LIFETIME_REQUIRED')
        actual = native_snapshot(QuantGroupNative.tensor_view(context['input'], 'input', call.device))
        current_proof = self.row_observer._lookup(context['input'])
        require(current_proof is not None and current_proof['node']['node_id'] == context['row_proof']['node_id']
                and current_proof['metadata'] == context['row_proof']['tensor']
                and current_proof['rows'] == context['row_proof']['actual_rows'],
                'C1_ROW_PROOF_CHANGED_BETWEEN_PUBLIC_CALL_AND_NATIVE_ADMISSION')
        require(all(native_snapshot(call.input)[name] == value for name, value in actual.items() if name != 'descriptor')
                and all(stable_geometry(native_snapshot(getattr(call, name))) == self.target['identity']['geometry'][name]
                        for name in ('input', 'quant', 'scale')), 'C1_NATIVE_CURRENT_CALL_FORMALS_CHANGED')
        config = binding.config
        lease = Lease(mode=MODES[phase], owner_hi=config.owner_hi, owner_lo=config.owner_lo,
            generation=self.generation, thread=call.thread, target=key(self.target['sha256']),
            actual_row_proof=key(digest(dict(proof=context['row_proof'], actual_call=native_snapshot(call),
                target=self.target['sha256'], owner=self.owner, generation=self.generation))), expected=call)
        self.phase_attempts.add(phase)  # A rejected attempt cannot silently retry.
        self.current_lease = lease
        row['lease'] = native_snapshot(lease)
        return binding.admit(lease)

    def _completed(self, completion):
        done = native_snapshot(completion)
        row = self.callback_rows.get(completion.call.token)
        require(row is not None and not row['completed'] and row['call'] == done['call'], 'C1_NATIVE_COMPLETION_OWNER_CHANGED')
        row.update(completed=True, status=done['status'], admitted=done['admitted'])
        if not row['selected']:
            require(not done['admitted'], 'NON_TARGET_NATIVE_INTERVENTION_REFUSED')
            return 0
        context = self.current_call
        require(context is not None and self.current_lease is not None, 'LIVE_SELECTED_PUBLIC_CALL_REQUIRED')
        context['completions'].append(done)
        end = max(done['core']['reference']['execute_exit'], done['core']['effect']['execute_exit'])
        events = self.runtime.binding.raw_range(done['call']['token'], end)
        verify_completion(done, events, self.current_lease, released=False)
        return 0

    def logits(self, module, args, kwargs, output):
        decode = self.active is not None and bool(self.active.get('decode'))
        if self.row_observer.active:
            self.inventory.row_trace = self.row_observer.end()
        QuantServingSession.logits(self, module, args, kwargs, output)
        if decode:
            try:
                self.inventory.end(self.decode_result['logits_sha256'], model_output_finite=self.decode_result['logits_finite'])
            except BaseException as error:
                self.error = repr(error)
                self.inventory.fail(self.error)
                self.save()
                raise

    def _recover_native(self):
        binding = self.runtime.binding
        if binding.config is None:
            return
        require(self.current_call is None and self.inventory.frame is None, 'C1_PUBLIC_MODEL_CALL_STILL_ACTIVE')
        state = binding.query()
        require(not any(state[name] for name in ('public_active', 'native_stack_active', 'callback_active', 'completion_active')),
                'C1_ACTUAL_NATIVE_STACK_STILL_ACTIVE')
        if state['core']['installed']:
            if state['core']['targets']:
                require(state['core']['targets'] == 1 and self.target is not None, 'C1_NATIVE_TARGET_SET_CHANGED')
                binding.stop(self.target['sha256'])
                # Includes actual passthrough streams used by every other
                # projection; raw pending==0 alone is never device drain.
                import torch_npu
                torch_npu.npu.synchronize()
                self.recovery_synchronized_device = True
                require(binding.drain(self.target['sha256']) is True, 'C1_NATIVE_TARGET_DRAIN_PENDING')
                binding.recover()  # Native refuses any late public-storage restore.
            else:
                require(binding.cancel_empty() is True, 'C1_EMPTY_OWNER_DRAIN_PENDING')
        state = binding.query()
        require(not any(state['core'][name] for name in ('installed', 'pending_parent', 'device_bytes', 'host_bytes',
                    'owned_descriptors', 'owned_workspace_bytes', 'borrowed_workspace_bytes', 'quarantined_bytes'))
                and not any(state['census'][name] for name in ('preparing', 'pending', 'executing'))
                and not any(state[name] for name in ('quarantined', 'abi_exceptions', 'orphan_descriptors')),
                'C1_NATIVE_ZERO_RESOURCES_NOT_VERIFIED')
        binding.remove()
        self.native_owner_released = True
        self.native_recovery.append(dict(before_remove=state, after_remove=binding.query(),
            worker=self._identity(recovering=True), actual_global_device_synchronize=self.recovery_synchronized_device,
            empty_owner_uses_actual_native_observed_stream_drain=not self.phase_attempts, late_public_restore_requested=False))

    def collect(self):
        try:
            self._collect()
        except BaseException as error:
            self.collection_errors.append(dict(phase=self.phase_name(), error=repr(error)))
            self.save()
            raise

    def _collect(self):
        if self.expected is None or self.decode_result is None:
            return
        require(not self.error and self.active is None and self.prefill_result is not None and self.forward_count == 2,
                'COMPLETE_NATURAL_C1_PREFILL_AND_DECODE_REQUIRED')
        state = self.state()
        require(not state['pending'] and not state['executing'], 'C1_NATIVE_CALL_STILL_INFLIGHT')
        phase = self.phase_name()
        if self.rows:
            reference = self.rows[0]
            require(self.prefill_result['logits_sha256'] == reference['prefill']['logits_sha256'], 'C1_PREFILL_CHANGED_BEFORE_FAULT')
            if phase != 'fault':
                require(all(self.decode_result[name] == reference['decode'][name] for name in ('hidden_sha256', 'logits_sha256')),
                        'C1_FULL_HEALTHY_DECODE_NOT_RESTORED')
        if phase in MODES:
            current = [row for row in self.receipts if row['phase'] == phase]
            require(self.public_selected.get(phase) == 1 and len(current) == 1, 'ONE_COMPLETE_SELECTED_NATIVE_CALL_REQUIRED')
            forward = self.inventory.rows[-1]
            projections = [row for row in forward['projections'] if row['module_object_identity'] == id(self.target['module'])]
            require(len(projections) == 1 and projections[0]['public_chain_verified'], 'C1_ACTUAL_Q_SCALE_CONSUMER_NOT_PROVEN')
            public = [self.inventory.public_calls[index] for index in projections[0]['api_call_indices']
                      if self.inventory.public_calls[index]['operator'] == 'npu_dynamic_quant']
            require(len(public) == 1, 'C1_ONE_ACTUAL_PUBLIC_DYNAMIC_CALL_REQUIRED')
            self.projection_prefixes[phase] = closed_projection_prefix(public[0], projections[0], self.inventory.native_forwards[-1])
            classified = [event for event in self.inventory.native_forwards[-1]['native_events']
                          if event['kind'] == 2 and any(event['binding'].values())]
            require(len(classified) == (8 if phase == 'fault' else 4)
                    and all(event['binding']['parent_token'] == current[0]['completion']['call']['token'] for event in classified),
                    'C1_NON_TARGET_NATIVE_CLASSIFICATION_OR_MISSING_HELPER')
            require(all(row['completed'] and row.get('status') == 0 for row in self.callbacks),
                    'C1_NATIVE_PARENT_CALLBACK_COMPLETION_MISSING')
        if phase in ('reference', 'lossless', 'qualification_off', 'off'):
            matching = [row for row in self.inventory.public_calls[self.inventory.rows[-1]['public_start']:]
                        if row['operator'] == 'npu_dynamic_quant' and row['module_object_identity'] == id(self.target['module'])]
            require(len(matching) == 1 and not matching[0]['errors']
                    and matching[0].get('healthy_formals') == self.target['healthy_formals'],
                    'C1_ACTUAL_HEALTHY_INPUT_Q_SCALE_NOT_RESTORED')
        if phase == 'fault':
            self._recover_native()
        if phase == 'off':
            require(self.native_owner_released and self.public_selected.get(phase) == 1
                    and not any(self.state()[name] for name in ('armed', 'pending', 'executing', 'owned')),
                    'C1_SAME_WORKER_HEALTHY_OFF_AND_ZERO_OWNER_REQUIRED')
            self._identity()
            projections = [row for row in self.inventory.rows[-1]['projections'] if row['module_object_identity'] == id(self.target['module'])]
            require(len(projections) == 1, 'C1_OFF_ACTUAL_PROJECTION_REQUIRED')
            self.projection_prefixes[phase] = closed_projection_prefix(matching[0], projections[0], self.inventory.native_forwards[-1])
        self.rows.append(dict(phase=phase, prefill=self.prefill_result, decode=self.decode_result))
        self.seen.extend(self.expected)
        self.expected = None
        if phase == 'discovery_2':
            self._select_target()
        self.recovered = phase == 'off'
        self.save()

    def close(self, *, abort=False):
        if not abort:
            self.collect()
        self._recover_native()  # Keep all hooks/strong owners if this refuses.
        require('fault' not in self.phase_attempts or self.recovered,
                'C1_SAME_WORKER_HEALTHY_OFF_REQUIRED_BEFORE_OWNER_RELEASE')
        super().close(abort=abort)

    def report(self):
        value = super().report()
        accepted = bool(self.recovered and not self.error and not self.installed and self.runtime.binding.config is None
            and not any(value[name] for name in ('owned_hooks', 'owned_method_patches', 'owned_api_patches',
                'owned_row_dispatch_modes', 'owned_source_profile_callbacks', 'owned_early_dispatch_modes',
                'owned_nested_dispatch_modes', 'owned_live_metadata_records', 'observer_cleanup_pending')))
        value.update(schema='demoncat.quant.group_single_site_serving.v1', scope=SCOPE,
            status='SINGLE_SITE_C1_READY_FOR_REVIEW' if accepted else 'SINGLE_SITE_C1_PENDING',
            next_phase=self.phase_name(), owner=self.owner, generation=self.generation,
            selected_site=None if self.target is None else dict(identity=self.target['identity'], sha256=self.target['sha256']),
            selected_discovery_prefixes=[] if self.target is None else self.target['qualification_prefixes'],
            completed_projection_prefixes=self.projection_prefixes,
            discovery_sha256=None if self.discovery is None else self.discovery['sha256'],
            complete_discovery=str(self.folder / 'discovery.json') if self.discovery is not None else None,
            receipts=self.receipts, observed_owner_callbacks=self.callbacks, native_recovery=self.native_recovery,
            collection_errors=self.collection_errors,
            native_owner_installed=self.runtime.binding.config is not None,
            group_owner_installed=self.runtime.binding.config is not None,
            native_owner_released=self.native_owner_released, fault_attempted=self.fault_attempted,
            native_faults=sum(row['phase'] == 'fault' for row in self.receipts),
            functional_effect_proven=any(row['phase'] == 'fault' for row in self.receipts),
            c1=accepted, c2=False, c3=False, e5='NOT_ASSESSED',
            all_eligible_qualification=False, native_qualified_targets=int(accepted))
        return snapshot(value)

    def save(self):
        super().save()
        if self.discovery is not None:
            (self.folder / 'discovery.json').write_text(json.dumps(self.discovery, indent=2, allow_nan=False) + '\n')


def install_quant_group_c1_binding(worker, *, prompts, root, build_sha256, evidence_dir):
    """Inert deployment binding; Native install occurs only under QuantProvider."""
    from fault_injection.providers import register_runtime
    require(not hasattr(worker, '_dcat_quant_binding') and not hasattr(worker, '_demoncat_shared_scale_inventory'),
            'EXCLUSIVE_QUANT_DEPLOYMENT_BINDING_REQUIRED')
    require(os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING') == '0' and os.environ.get('TASK_QUEUE_ENABLE') == '0',
            'HOST_DRIVEN_SYNCHRONOUS_QUANT_REQUIRED')
    require_serial_eager(worker.model_runner, expected_max_seqs=8)
    require(isinstance(prompts, list) and len(prompts) == 8 and all(isinstance(row, list) and len(row) > 1
            and all(type(token) is int and token >= 0 for token in row) for row in prompts), 'EIGHT_FROZEN_PROMPTS_REQUIRED')
    for value in (worker, worker.model_runner):
        require(not any(name in vars(value) for name in ('execute_model', '_prepare_inputs')), 'CORE_OVERRIDE_REFUSED')
    folder = Path(evidence_dir)
    require(folder.is_absolute() and folder.resolve().is_relative_to('/evidence'), 'PRIVATE_QUANT_C1_EVIDENCE_REQUIRED')
    model = worker.model_runner.model
    def make_session(actual_worker, actual_model, *, owner, generation, worker_identity):
        require(actual_worker is worker and actual_model is model and worker.model_runner.model is model, 'FOREIGN_MODEL')
        return QuantGroupC1ServingSession(worker, model, owner=owner, generation=generation, worker_identity=worker_identity, root=root,
            build_sha256=build_sha256, folder=folder / 'runtime', prompts=prompts)
    def factory(actual_model):
        require(actual_model is model, 'FOREIGN_MODEL')
        return QuantGroupC1ServingBinding(make_session)
    register_runtime(MECHANISM, factory)
    worker._dcat_quant_binding = factory
    return dict(status='DEPLOYMENT_BOUND_NOT_ACCEPTED', scope=SCOPE, native_owner_installed=False,
                c1=False, c2=False, c3=False, e5='NOT_ASSESSED', core_methods_replaced=False)
