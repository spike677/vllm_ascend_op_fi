"""Bounded natural writer-v2 discovery and per-target B/L/F/OFF experiments.

The complete native census is always retained. A candidate experiment is not an
all-target C2/C3 certificate. Unknown public storage lifetimes stop the dependent
phase before a replacement is requested.
"""
from collections import Counter
import ctypes as C
import hashlib
import inspect
import json
import marshal
from pathlib import Path
import threading

from .residual26 import allocation_pair
from .writer26_group import Call, MODES, packet, key_words
from .writer26_inventory import (WriterLifetimeInventory, allocation_segments, correlate_lifetimes,
                                 digest, require, tensor_matches_native)


SINGLE_SITE_STAGE = 'c1_single_effects'
SINGLE_SITE_SCOPE = 'ONE_ACTUAL_QMM_SITE_FIXED_FRESH_PREFILL_NOT_ALL_ELIGIBLE_NOT_C2_C3'
_STABLE_NATIVE = ('cache', 'registration', 'device', 'dtype', 'format', 'dimensions',
    'residual_dimensions', 'output_bytes', 'residual_bytes', 'shape', 'strides', 'residual_shape', 'residual_strides')


def tensors(value):
    import torch
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in tensors(item)]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in tensors(item)]
    return []


def full_storage(tensor, address, size):
    return (tensor.is_contiguous() and tensor.storage_offset() == 0
        and tensor.untyped_storage().data_ptr() == address and tensor.untyped_storage().nbytes() == size
        and tensor.numel() * tensor.element_size() == size)


def hold_storage(tensor):
    """Own the actual Tensor and allocation, not a reusable pointer witness."""
    storage = tensor.untyped_storage()
    return dict(tensor=tensor, storage=storage, object_identity=id(tensor), storage_identity=storage._cdata,
        shape=tuple(tensor.shape), stride=tuple(tensor.stride()), dtype=tensor.dtype, device=tensor.device)


def held_storage(record, address, size):
    if record is None:
        return False
    tensor, storage = record['tensor'], record['storage']
    return (id(tensor) == record['object_identity'] and storage._cdata == record['storage_identity']
        and tensor.untyped_storage()._cdata == record['storage_identity']
        and storage.data_ptr() == address and storage.nbytes() == size and full_storage(tensor, address, size)
        and tuple(tensor.shape) == record['shape'] and tuple(tensor.stride()) == record['stride']
        and tensor.dtype == record['dtype'] and tensor.device == record['device'])


def actual_method(module):
    from demoncat.runtime.vllm_ascend.worker_extension.call_binding import original_before_collector
    function = original_before_collector(module).__func__
    source = Path(inspect.getsourcefile(function)).resolve(strict=True)
    require(function.__globals__ is vars(inspect.getmodule(function)), 'ACTUAL_MODULE_FORWARD_GLOBALS_CHANGED')
    return dict(path=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(marshal.dumps(function.__code__)).hexdigest(),
        callable_identity=id(function), module_class=type(module).__module__ + '.' + type(module).__qualname__)


def target_key(row):
    call = row['native_packet']
    signature = {key: call[key] for key in ('mechanism', 'device', 'dtype', 'format', 'dimensions',
                 'output_bytes', 'residual_bytes', 'shape')}
    if call['mechanism'] == 1:
        signature.update(cache=call['cache'], registration=call['registration'])
    return digest(dict(module=row['module_object_identity'], method=row['method'], signature=signature))


def classify_candidate(candidate):
    """No missing lineage may become an unsupported exclusion by convenience."""
    native, callback = candidate['native'], candidate.get('callback')
    if native['residual'] == native['residual_bytes'] == native['producer'] == 0 and native['flags'] in (1, 5):
        return 'UNSUPPORTED', ['NO_ACTUAL_PRECEDING_RESIDUAL_IN_THIS_FRESH_FORWARD']
    if callback is None:
        return 'UNVERIFIED', list(candidate['reasons'])
    reasons = list(candidate['reasons'])
    if callback.get('proven_nonadjacent'):
        allowed = {'LIVE_ALLOCATOR_BLOCKS_PROVEN_NOT_ADJACENT', 'PUBLIC_WRITER_OUTPUT_NOT_RETAINED',
                   'PUBLIC_CONSUMER_LIFETIME_NOT_PROVEN'}
        if set(reasons).issubset(allowed):
            return 'UNSUPPORTED', ['LIVE_ALLOCATOR_BLOCKS_PROVEN_NOT_ADJACENT']
    if not callback.get('writer_output_retained'):
        reasons.append('PUBLIC_WRITER_OUTPUT_NOT_RETAINED')
    if not callback.get('public_consumer_lifetime'):
        reasons.append('PUBLIC_CONSUMER_LIFETIME_NOT_PROVEN')
    if callback.get('fia_typed_before') is not None and (
            not callback.get('fia_typed_public_records') or
            (callback.get('writer_output_retained') or {}).get('evidence') != 'ACTUAL_PUBLIC_FIA_RETURN'):
        reasons.append('FIA_TYPED_ACTUAL_PUBLIC_PATH_UNVERIFIED')
    return ('OBSERVED_NOT_QUALIFIED', []) if not reasons else ('UNVERIFIED', sorted(set(reasons)))


class Writer26Experiment(WriterLifetimeInventory):
    """One current target per fixed fresh request, all native calls still observed.

    Stage inventory is write-free. Stage lossless tests every resolved candidate
    if the explicit target budget covers the full resolved set. Stage effects
    adds one F/OFF pair per such target, preserving every unresolved row and
    explicitly keeping C2/C3 false. This is a bounded candidate qualification
    entrance. The explicit c1_single_effects stage selects one actual resolved
    site and keeps the complete inventory, without granting group coverage.
    """
    def __init__(self, *args, stage='inventory', max_targets=8, **kwargs):
        super().__init__(*args, **kwargs)
        require(stage in ('inventory', 'lossless', 'effects', SINGLE_SITE_STAGE)
                and type(max_targets) is int and 1 <= max_targets <= 128,
                'BOUNDED_WRITER_EXPERIMENT_STAGE_REQUIRED')
        require(stage != SINGLE_SITE_STAGE or (not self.runtime.fia and self.runtime.mechanism == '26.residual_stomp'),
                'SINGLE_SITE_C1_EFFECTS_QMM_ONLY')
        self.stage, self.max_targets = stage, max_targets
        self.single_selection = self.single_qualification = None
        self.single_aliases = self._actual_aliases() if stage == SINGLE_SITE_STAGE else None
        if stage == SINGLE_SITE_STAGE:
            self.root_hooks = 0
        self.plan = [dict(phase='inventory_1', mode=None, target=None), dict(phase='inventory_2', mode=None, target=None)]
        self.targets, self.target_refs, self.output_refs = {}, {}, {}
        self.pending, self.consumer_inputs = {}, None
        self.pending_lifetimes = {}
        self.failure_events, self.last_native_receipt = [], None
        self.outcomes = []
        self.candidate_inventory = []
        self.blockers = []
        self.baseline = None
        self.initialized = False
        self.native_removed = False
        self.ordinal = 0
        from ..metadata_arrays import MetadataArraySession
        self.metadata = MetadataArraySession(getattr(self.runtime, 'metadata_arrays', None), self.runtime.build_identity)
        self.metadata_initial = None
        from .writer26_public import FiaPublicOutputObserver
        self.public_output = FiaPublicOutputObserver(self) if self.runtime.fia else None
        self.fia_typed = getattr(self.runtime, 'fia_typed', None)
        from .writer26_fia_typed import FiaStockBridge
        self.fia_bridge = FiaStockBridge(self) if self.fia_typed is not None else None

    @property
    def phase(self):
        return self.plan[len(self.rows)]['phase'] if len(self.rows) < len(self.plan) else 'complete'

    @property
    def entry(self):
        return self.plan[len(self.rows)]

    def _idle(self, registered):
        metadata = self.metadata.read()
        if self.metadata_initial is None:
            self.metadata_initial = metadata
        state = self.runtime.state()
        core = state['core']
        require(bool(core['installed']) is registered and not any(core[n] for n in ('pending', 'executing', 'device_bytes', 'descriptor_pending'))
                and not any(state[n] for n in ('callback_active', 'completion_active', 'descriptor', 'prepared_fia')),
                'WRITER_GROUP_NATIVE_TRANSACTION_NOT_IDLE')
        require(not core['error'] and not state['foreign_calls'] and not state['rejected_calls'],
                'WRITER_GROUP_NATIVE_ERROR_REQUIRES_EXPLICIT_RECOVERY')
        if self.fia_typed is not None:
            self.fia_typed.state(idle=True)
        return state

    def install(self):
        import torch
        self._thread()
        require(not self.installed and not self.initialized, 'ONE_WRITER_EXPERIMENT_PER_OWNER_REQUIRED')
        self._idle(False)
        self.reader.state()
        self.original_producer = torch.ops._C_ascend.npu_add_rms_norm_bias
        def producer(*args, **kwargs):
            return self._producer(args, kwargs) if self.current is not None else self.original_producer(*args, **kwargs)
        self.producer_observer = producer
        torch.ops._C_ascend.npu_add_rms_norm_bias = producer
        try:
            self.runtime.install(self._callback, self._completed)
            self.installed = self.initialized = True
            for _, module in self.modules.values():
                if module is not self.model:
                    self.handles.append(module.register_forward_pre_hook(self._pre, with_kwargs=True))
                    self.handles.append(module.register_forward_hook(self._post, with_kwargs=True, always_call=True))
            self.owned_hooks = len(self.handles)
        except BaseException:
            if self.runtime.state()['core']['installed']:
                self.runtime.close()
            torch.ops._C_ascend.npu_add_rms_norm_bias = self.original_producer
            for handle in self.handles:
                handle.remove()
            self.handles.clear()
            raise

    def begin(self, request_id, input_sha256):
        self._thread()
        if self.stage == SINGLE_SITE_STAGE:
            require(not self.failure, 'FAILED_SINGLE_SITE_CANNOT_RETRY_OR_SELECT_ANOTHER_TARGET')
            require(self._actual_aliases() == self.single_aliases, 'SINGLE_SITE_ACTUAL_MODULE_ALIASES_CHANGED')
        require(self.current is None and request_id not in self.requests and self.phase != 'complete',
                'FRESH_BOUNDED_WRITER_REQUEST_REQUIRED')
        if self.phase == 'health':
            self._before_health()
        self._idle(not self.native_removed)
        require(not self.retained, 'RESIDUAL_LIFETIME_ESCAPED_PREVIOUS_REQUEST')
        self.current = dict(phase=self.phase, request_id=request_id, input_sha256=input_sha256,
            native_before=self.reader.state(), callback_start=len(self.callbacks), producer_start=len(self.producers))
        if self.fia_typed is not None:
            self.current['fia_typed_before'] = self.fia_typed.state(idle=True)
        self.requests.add(request_id)
        self.stack = [self.model]
        require(not self.output_refs and not self.pending and not self.pending_lifetimes,
                'WRITER_LIFETIME_ESCAPED_ACTUAL_REQUEST')
        if self.phase == 'inventory_1':
            self.baseline_input = input_sha256
        require(input_sha256 == self.baseline_input, 'ACTUAL_FRESH_WORKLOAD_CHANGED')
        self.current.update(plan=dict(self.entry), admissions=0, completed=0)
        if self.entry['mode'] == 'fault':
            if self.stage == SINGLE_SITE_STAGE:
                require(self.single_qualification is not None and self.runtime.state()['core']['sealed'],
                        'SINGLE_SITE_REFERENCE_LOSSLESS_OFF_BEFORE_FAULT_REQUIRED')
            self.fault_started = True
        if self.public_output is not None and not self.native_removed:
            # The serving host has already started its optional collector.
            self.public_output.install()

    def _before_health(self):
        self.runtime.close()
        self.native_removed = True

    def _observe_callback(self, row, call):
        pass

    def _record_failure(self, reason, source, *, token=None):
        message = str(reason)
        self.failure_events.append(dict(source=source, reason=message, phase=self.phase, token=token))
        if self.failure is None:
            self.failure = message

    def fail(self, reason):
        self._record_failure(reason, 'request_unwind')
        for token, outcome in self.pending.items():
            try:
                receipt = self.runtime.receipt(token)
                outcome['last_native_receipt'] = receipt
                self.last_native_receipt = receipt
            except BaseException as error:
                outcome['native_receipt_read_failure'] = str(error)
        if self.current is not None:
            self.current.update(failure_events=list(self.failure_events), pending_outcomes=list(self.pending.values()),
                                last_native_receipt=self.last_native_receipt)
        # Always-call root/logits cleanup must not replace the earlier Native
        # receipt failure with the secondary missing-model-output exception.
        super().fail(self.failure)

    def _requested_mode(self, row):
        if (self.stage == SINGLE_SITE_STAGE and self.entry['mode'] is not None
                and row['target_key'] == self.entry['target']):
            frozen = self.targets[self.entry['target']]['second_native']
            require(self._actual_aliases() == self.single_aliases
                and all(row['independent_event'][key] == frozen[key] for key in _STABLE_NATIVE),
                'SINGLE_SITE_CURRENT_OBJECT_OR_NATIVE_FORMAL_CHANGED')
            row['single_site_current_binding_verified'] = True
        return self.entry['mode'], self.entry['target']

    def _expected_admissions(self):
        return 1

    def _outcome_complete(self, outcome, row):
        pass

    def _request_complete(self, row):
        if self.fia_typed is not None:
            after, records = self.fia_typed.delta(row['fia_typed_before'])
            row.update(fia_typed_after=after, fia_typed_records=records)
            for phase, kind in ((1, 'fia_before'), (2, 'fia_gws')):
                actual = [(item['call'], item['thread'], item['event_sequence']) for item in records if item['phase'] == phase]
                native = [(item['call'], item['thread'], item['sequence']) for item in row['native_events']
                          if item['kind_name'] == kind]
                require(actual == native, 'COMPLETE_TYPED_FIA_NATIVE_DENOMINATOR_CHANGED')
            require(sum(item['phase'] == 3 for item in records) == row['admissions'],
                    'FIA_EVERY_SELECTED_OWNED_EXECUTE_MUST_BE_OBSERVED')
        if self.stage == SINGLE_SITE_STAGE:
            row['complete_inventory'] = dict(native_writer_count=len(row['candidates']),
                native_events_sha256=digest(row['native_events']), candidates_sha256=digest(row['candidates']),
                actual_aliases_sha256=digest(self._actual_aliases()))
            require(self._actual_aliases() == self.single_aliases, 'SINGLE_SITE_ACTUAL_MODULE_ALIASES_CHANGED')
            if row['phase'] == 'health':
                self._check_denominator(row)

    def _check_denominator(self, row):
        require(len(row['candidates']) == len(self.rows[0]['candidates']), 'NATURAL_WRITER_COUNT_CHANGED_KEEP_FULL_DENOMINATOR')
        if self.stage == SINGLE_SITE_STAGE and row['plan']['mode'] is not None:
            selected = [candidate for candidate in row['candidates']
                if (candidate.get('callback') or {}).get('target_key') == self.single_selection['target_key']]
            require(len(selected) == 1 and classify_candidate(selected[0])[0] == 'OBSERVED_NOT_QUALIFIED'
                and selected[0] is row['candidates'][self.single_selection['inventory_index']],
                'SINGLE_SITE_ACTUAL_OCCURRENCE_OR_CONSUMER_CHANGED')

    def _actual_aliases(self):
        # Keep every registered alias. Module names are observations, never a
        # user-supplied selector, and duplicate aliases do not become targets.
        return [dict(path=name or '<root>', object_identity=id(module))
                for name, module in self.model.named_modules(remove_duplicate=False)]

    def _post(self, module, args, kwargs, output):
        if self.current is not None:
            for row in self.callbacks[self.current['callback_start']:]:
                if row.get('module_object_identity') != id(module) or row.get('writer_output_retained'):
                    continue
                if row.get('public_output_scope'):
                    # A refused real API return cannot be repaired by guessing
                    # that a later same-address module result is its output.
                    continue
                call = row['native_packet']
                found = {id(t): t for t in tensors(output) if full_storage(t, call['output'], call['output_bytes'])}
                if len(found) == 1:
                    tensor = next(iter(found.values()))
                    self.output_refs[call['token']] = tensor
                    row['writer_output_retained'] = dict(tensor_object_identity=id(tensor),
                        storage_address=tensor.untyped_storage().data_ptr(), bytes=tensor.untyped_storage().nbytes(),
                        shape=list(tensor.shape), stride=list(tensor.stride()), dtype=str(tensor.dtype))
        super()._post(module, args, kwargs, output)

    def _producer(self, args, kwargs):
        from demoncat.runtime.vllm_ascend.worker_extension.serving import complete_tensor
        inputs = tensors((args, kwargs))
        live = []
        for row in self.callbacks[self.current['callback_start']:]:
            call = row.get('native_packet')
            if not call or row.get('public_consumer_lifetime'):
                continue
            consumed = [t for t in inputs if full_storage(t, call['residual'], call['residual_bytes'])]
            if not consumed:
                continue
            retained = self.retained.get(call['residual'])
            writer = self.output_refs.get(call['token'])
            valid = (len({id(t) for t in consumed}) == 1 and retained is not None
                and retained['sequence'] == call['producer_sequence']
                and full_storage(retained['tensor'], call['residual'], call['residual_bytes'])
                and consumed[0].untyped_storage()._cdata == retained['tensor'].untyped_storage()._cdata
                and writer is not None and full_storage(writer, call['output'], call['output_bytes']))
            row['public_consumer_lifetime'] = bool(valid)
            if valid:
                row['public_consumer_binding'] = dict(producer_tensor_object_identity=id(retained['tensor']),
                    consumer_tensor_object_identity=id(consumed[0]),
                    residual_storage_identity=int(consumed[0].untyped_storage()._cdata),
                    writer_tensor_object_identity=id(writer),
                    same_residual_storage=True, writer_retained_through_actual_consumer=True)
            if call['token'] in self.pending:
                require(valid, 'QUALIFIED_WRITER_OR_RESIDUAL_PUBLIC_LIFETIME_CHANGED_BEFORE_CONSUMER')
                owned = self.pending_lifetimes[call['token']]
                require(held_storage(owned['residual'], call['residual'], call['residual_bytes']),
                        'PENDING_PRODUCER_STORAGE_OWNER_CHANGED_BEFORE_CONSUMER')
                owned.update(consumer=hold_storage(consumed[0]), writer=hold_storage(writer))
            live.append(row)
        self.consumer_inputs = inputs  # Strong references through the actual native callback.
        try:
            result = super()._producer(args, kwargs)
            for row in live:
                call = row['native_packet']
                if call['token'] in self.pending:
                    outcome = self.pending[call['token']]
                    require(outcome.get('released_receipt') is not None, 'ACTUAL_CONSUMER_DID_NOT_COMPLETE_NATIVE_LEASE')
                    residual = next(t for t in inputs if full_storage(t, call['residual'], call['residual_bytes']))
                    writer = self.output_refs[call['token']]
                    _, residual_after, _ = complete_tensor(residual)
                    _, output_sha, _ = complete_tensor(writer)
                    outcome.update(residual_after_sha256=residual_after, writer_output_sha256=output_sha,
                        public_consumer_lifetime=True)
                    require(outcome['residual_before_sha256'] == residual_after, 'FULL_LIVE_RESIDUAL_NOT_RESTORED')
                    if outcome['mode'] == 'reference':
                        self.target_refs[outcome['target']] = dict(output_sha256=output_sha, residual_sha256=residual_after)
                    elif outcome['mode'] != 'fault':
                        require(self.target_refs[outcome['target']] == dict(output_sha256=output_sha, residual_sha256=residual_after),
                                'LOSSLESS_OR_OFF_FULL_TARGET_BYTES_DIFFER')
                    self.current['completed'] += 1
                    self._outcome_complete(outcome, row)
                    self.outcomes.append(outcome)
                    del self.pending[call['token']]
                    del self.pending_lifetimes[call['token']]
                self.output_refs.pop(call['token'], None)
            return result
        except BaseException as error:
            self._record_failure(error, 'public_consumer')
            raise
        finally:
            self.consumer_inputs = None

    def _allocation(self, call, row):
        import torch_npu
        snapshot = torch_npu.npu.memory_snapshot()
        relevant = allocation_segments(snapshot, (call.residual, call.output))
        path = self.folder / ('allocator-' + str(call.independent_call) + '.json')
        path.write_text(json.dumps(relevant, indent=2, allow_nan=False) + '\n')
        row['allocator_segments'] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            complete_snapshot_sha256=digest(snapshot))
        try:
            row['allocation'] = allocation_pair(snapshot, call.residual, call.output, call.residual_bytes, call.output_bytes)
        except BaseException:
            blocks = []
            for segment in relevant:
                cursor = segment['address']
                for block in segment['blocks']:
                    start = block.get('address', cursor)
                    blocks.append((start, start + block['size'], block['state']))
                    cursor = start + block['size']
            matched = [[b for b in blocks if b[2] == 'active_allocated' and b[0] <= address and address + size <= b[1]]
                       for address, size in ((call.residual, call.residual_bytes), (call.output, call.output_bytes))]
            if all(len(v) == 1 for v in matched) and matched[0][0] != matched[1][0] and matched[0][0][1] != matched[1][0][0]:
                row['proven_nonadjacent'] = matched
                raise RuntimeError('LIVE_ALLOCATOR_BLOCKS_PROVEN_NOT_ADJACENT')
            raise RuntimeError('CURRENT_LIVE_ALLOCATOR_PAIR_UNVERIFIED')

    def _callback(self, pointer):
        call = Call.from_buffer_copy(pointer.contents)
        row = dict(native_packet=packet(call), original_continuations=0, claimed=0, errors=[], independent_call=None)
        observed = False
        try:
            if self.current is None:
                row['original_continuations'] = 1
                self.runtime.passthrough(call)
                return 0
            self._thread()
            self.callbacks.append(row)
            require(call.version == 2 and call.size == C.sizeof(Call) and call.thread == self.native_thread,
                    'ACTUAL_NATIVE_CALL_ABI_OR_THREAD_CHANGED')
            require(self.stack and id(self.stack[-1]) in self.modules, 'ACTUAL_WRITER_MODULE_UNPROVEN')
            module = self.stack[-1]
            row.update(module_object_identity=id(module), module_path=self.modules[id(module)][0], method=actual_method(module),
                actual_request_id=self.current['request_id'], token=call.token)
            if self.public_output is not None and self.public_output.frames:
                row['public_output_scope'] = self.public_output.frames[-1]['record']['api']
            state = self.reader.state()
            event = self.reader.read(state['stored_events'] - 1)
            require(event['kind_name'] == ('fia_before' if self.runtime.fia else 'qmm_enter')
                and event['call'] == call.independent_call and event['thread'] == call.thread
                and all(event[name] == getattr(call, name) for name in
                        ('output', 'output_bytes', 'residual', 'residual_bytes', 'cache', 'registration', 'device', 'dtype', 'format'))
                and event['producer'] == call.producer_sequence, 'INDEPENDENT_WRITER_EVENT_CHANGED')
            row.update(independent_call=call.independent_call, independent_event=event, target_key=target_key(row))
            if self.fia_typed is not None:
                # Keep the existing actual site identity if a fresh descriptor
                # check fails: that call is a failed eligible opportunity.
                row['fia_typed_before'] = self.fia_typed.current_before(row['native_packet'], event)
            retained = self.retained.get(call.residual)
            require(retained is not None and retained['request_id'] == self.current['request_id']
                    and retained['sequence'] == call.producer_sequence and tensor_matches_native(retained['tensor'], event, residual=True),
                    'CURRENT_PUBLIC_RESIDUAL_LIFETIME_UNPROVEN')
            row['producer_tensor_object_identity'] = id(retained['tensor'])
            if self.runtime.fia:
                require(call.output != call.residual and 0 < call.output_bytes <= call.residual_bytes,
                        'INDEPENDENT_FIA_DESTINATION_GEOMETRY_UNSUPPORTED')
            else:
                self._allocation(call, row)
            row['current_candidate_geometry_eligible'] = True
            self._observe_callback(row, call)
            observed = True
            mode, selected = self._requested_mode(row)
            if mode is not None and row['target_key'] == selected:
                require(self.current['admissions'] < self._expected_admissions() and selected in self.targets,
                        'ONE_QUALIFIED_WRITER_OCCURRENCE_PER_FIXED_REQUEST_REQUIRED')
                retained_by = self.targets[selected]['callback'].get('writer_output_retained', {})
                if retained_by.get('evidence') == 'ACTUAL_PUBLIC_FIA_RETURN':
                    require(self.public_output is not None and self.public_output.matches_callback(row),
                            'QUALIFIED_PUBLIC_FIA_CALL_SCOPE_CHANGED_BEFORE_ADMISSION')
                if self.fia_bridge is not None:
                    require(self.public_output is not None and self.public_output.matches_callback(row),
                            'TYPED_FIA_ACTUAL_PUBLIC_STACK_REQUIRED')
                    self.fia_bridge.before_admit(selected, row['fia_typed_before'], mode)
                from demoncat.runtime.vllm_ascend.worker_extension.serving import complete_tensor
                _, residual_sha, _ = complete_tensor(retained['tensor'])
                self.pending[call.token] = dict(mode=mode, target=selected, native_call=packet(call),
                    residual_before_sha256=residual_sha, request_id=self.current['request_id'])
                self.pending_lifetimes[call.token] = dict(residual=hold_storage(retained['tensor']),
                    producer_sequence=call.producer_sequence, request_id=self.current['request_id'])
                row['original_continuations'] = 1
                arguments = dict(typed_before=row['fia_typed_before']) if self.fia_typed is not None else {}
                self.pending[call.token]['lease'] = self.runtime.execute(call, mode, selected, row.get('allocation'), **arguments)
                row['claimed'] = 1
                self.current['admissions'] += 1
        except BaseException as error:
            row['errors'].append(str(error))
            if (self.current is not None and self.entry['mode'] is not None
                    and (row.get('target_key') == self.entry['target'] or
                         self.entry['target'] == '*' and row.get('target_key') in self.targets)):
                self._record_failure(error, 'native_admission', token=call.token)
        finally:
            if not observed and row.get('independent_event') is not None:
                try:
                    self._observe_callback(row, call)
                except BaseException as error:
                    self._record_failure(error, 'independent_observer', token=call.token)
            if not row['original_continuations']:
                row['original_continuations'] = 1
                try:
                    self.runtime.passthrough(call)
                except BaseException as error:
                    self._record_failure(error, 'native_passthrough', token=call.token)
        return 0 if self.failure is None else 1

    def _completed(self, pointer):
        receipt = None
        try:
            receipt = packet(pointer.contents)
            self.last_native_receipt = receipt
            outcome = self.pending[receipt['token']]
            outcome['observed_consumer_receipt'] = receipt  # Preserve failed receipts before any validation.
            call = outcome['native_call']
            require(self.consumer_inputs is not None and receipt['target'] == list(key_words(outcome['target']))
                and receipt['owner_hi'] == self.runtime.config.owner_hi and receipt['owner_lo'] == self.runtime.config.owner_lo
                and receipt['generation'] == self.runtime.config.generation and receipt['error'] == 0
                and receipt['mode'] == MODES[outcome['mode']] and receipt['mechanism'] == call['mechanism']
                and all(receipt[name] == call[name] for name in ('independent_call', 'cache', 'producer_sequence',
                    'output', 'output_bytes', 'residual', 'residual_bytes')) and receipt['stream'] != 0
                and (receipt['registration'] == call['registration'] if not self.runtime.fia else
                     receipt['registration'] != 0 and receipt['executor'] != 0)
                and receipt['writer_status'] == receipt['consumer_status'] == 0
                and receipt['writer_calls'] == receipt['consumer_calls'] == 1
                and all(receipt[key] == 1 for key in ('writer_complete', 'consumer_complete', 'output_readback_exact',
                    'padding_readback_exact', 'residual_restored_exact')), 'EXACT_NATIVE_WRITER_CONSUMER_RECEIPT_REQUIRED')
            if outcome['mode'] == 'fault':
                require(receipt['changed_residual_bytes'] > 0, 'ACTUAL_RESIDUAL_EFFECT_NOT_OBSERVED')
            elif outcome['mode'] == 'lossless':
                require(receipt['replacement'] not in (receipt['output'], receipt['residual'])
                        and receipt['reference_equal'] == receipt['lossless_qualified'] == 1,
                        'ACTUAL_TEMPORARY_LOSSLESS_OUTPUT_NOT_QUALIFIED')
            if self.fia_typed is not None:
                require(outcome.get('fia_public_typed') is not None
                    and (outcome['mode'] == 'fault' or outcome['fia_public_typed']['stock_equal'] is True),
                    'FIA_COMPLETED_PUBLIC_TYPED_AND_STOCK_BRIDGE_REQUIRED')
                _, snapshot = self.fia_typed.snapshot(receipt['token'])
                require(snapshot == outcome['fia_public_typed']['snapshots'],
                        'FIA_TYPED_SNAPSHOT_CHANGED_BEFORE_CONSUMER_RELEASE')
                outcome['fia_typed_consumer_snapshot'] = snapshot
            outcome['consumer_receipt'] = receipt
            outcome['released_receipt'] = self.runtime.acknowledge(receipt['token'])
            require(outcome['released_receipt']['released'] == 1, 'ACTUAL_NATIVE_WRITER_LEASE_NOT_RELEASED')
            return 0
        except BaseException as error:
            self._record_failure(error, 'native_completion', token=None if receipt is None else receipt['token'])
            return 1

    def _freeze(self):
        require(self.rows[0]['input_sha256'] == self.rows[1]['input_sha256']
                and self.rows[0]['output_sha256'] == self.rows[1]['output_sha256'], 'HEALTHY_COMPLETE_REQUEST_CHANGED')
        self.baseline = self.rows[0]['output_sha256']
        left, right = self.rows[0]['candidates'], self.rows[1]['candidates']
        require(len(left) == len(right) and len(right) > 0, 'COMPLETE_NATIVE_WRITER_DENOMINATOR_CHANGED')
        counts = Counter(c['callback'].get('target_key') for c in right if c.get('callback'))
        first_counts = Counter(c['callback'].get('target_key') for c in left if c.get('callback'))
        for index, (before, candidate) in enumerate(zip(left, right)):
            status, reasons = classify_candidate(candidate)
            previous_status, _ = classify_candidate(before)
            callback, previous = candidate.get('callback'), before.get('callback')
            key = callback.get('target_key') if callback else None
            if status != previous_status or (status == 'OBSERVED_NOT_QUALIFIED' and
                    (previous is None or previous.get('target_key') != key or counts[key] != 1)):
                status, reasons = 'UNVERIFIED', ['HEALTHY_TARGET_LIFETIME_OR_OCCURRENCE_CHANGED']
            if (self.stage == SINGLE_SITE_STAGE and status == 'OBSERVED_NOT_QUALIFIED'
                    and (first_counts[key] != 1 or
                         any(before['native'][name] != candidate['native'][name] for name in _STABLE_NATIVE))):
                status, reasons = 'UNVERIFIED', ['HEALTHY_SINGLE_SITE_OCCURRENCE_OR_NATIVE_FORMAL_CHANGED']
            row = dict(index=index, target_key=key, status=status, reasons=reasons,
                first_native=before['native'], second_native=candidate['native'], callback=callback)
            self.candidate_inventory.append(row)
            if status == 'OBSERVED_NOT_QUALIFIED':
                self.targets[key] = row
        if self.stage == 'inventory':
            self.plan.append(dict(phase='health', mode=None, target=None))
            return
        if not self.targets or (self.stage != SINGLE_SITE_STAGE and len(self.targets) > self.max_targets):
            self.blockers.append('EMPTY_OR_OVER_BUDGET_COMPLETE_RESOLVED_TARGET_SET')
            self.plan.append(dict(phase='health', mode=None, target=None))
            return
        if self.stage == SINGLE_SITE_STAGE:
            # Select from the complete, twice-observed set; never truncate it.
            # This ordering is deterministic within this actual Worker lifetime.
            key = min(self.targets)
            self.single_selection = dict(policy='MIN_ACTUAL_HEALTHY_RESOLVED_TARGET_KEY_NO_FALLBACK', target_key=key,
                inventory_index=self.targets[key]['index'], complete_native_count=len(self.candidate_inventory),
                complete_resolved_count=len(self.targets), inventory_sha256=digest(self.candidate_inventory),
                aliases_sha256=digest(self.single_aliases), planned_target_count=1,
                legacy_complete_plan_max_targets=self.max_targets, max_targets_is_selector=False)
        per_request_events = self.rows[1]['native_after']['total_events'] - self.rows[1]['native_before']['total_events']
        remaining_requests = self._planned_qualification_requests()
        available = self.rows[1]['native_after']['capacity'] - self.rows[1]['native_after']['stored_events']
        if per_request_events * remaining_requests > available:
            self.blockers.append('COMPLETE_TARGET_PLAN_EXCEEDS_INDEPENDENT_NATIVE_EVENT_CAPACITY')
            self.plan.append(dict(phase='health', mode=None, target=None))
            return
        if self.stage == SINGLE_SITE_STAGE:
            for phase, mode in (('reference', 'reference'), ('lossless', 'lossless'), ('qualification_off', 'off'),
                                ('fault', 'fault'), ('recovery_off', 'off')):
                self.plan.append(dict(phase='single_site_' + phase, mode=mode, target=self.single_selection['target_key']))
            self.plan.append(dict(phase='health', mode=None, target=None))
            return
        for index, key in enumerate(self.targets):
            for mode in ('reference', 'lossless', 'off'):
                self.plan.append(dict(phase='target_%03d_%s' % (index, mode), mode=mode, target=key))
        if self.stage == 'effects':
            # All resolved targets must qualify before any fault is requested.
            for index, key in enumerate(self.targets):
                self.plan.extend(dict(phase='target_%03d_%s' % (index, mode), mode=mode, target=key) for mode in ('fault', 'off'))
        self.plan.append(dict(phase='health', mode=None, target=None))

    def end(self, output_sha256, *, model_output_finite):
        try:
            self._end_request(output_sha256, model_output_finite=model_output_finite)
        except BaseException as error:
            if self.stage == SINGLE_SITE_STAGE:
                self._record_failure(error, 'single_site_request_end')
                self.save()
            raise

    def _end_request(self, output_sha256, *, model_output_finite):
        self._thread()
        require(self.current is not None and self.stack == [self.model], 'COMPLETE_ACTUAL_WRITER_ROOT_REQUIRED')
        row = self.current
        after, native = self.reader.delta(row['native_before'])
        callbacks = self.callbacks[row['callback_start']:]
        producers = self.producers[row['producer_start']:]
        row.update(native_after=after, native_events=native, callbacks=callbacks, producers=producers,
            candidates=correlate_lifetimes(native, callbacks, producers, self.runtime.mechanism),
            output_sha256=output_sha256, output_finite=model_output_finite)
        self.rows.append(row)
        self.current = None
        self.stack.clear()
        mode = row['plan']['mode']
        self._request_complete(row)
        require(after['qmm_runs'] - row['native_before']['qmm_runs'] == sum(e['kind_name'] == 'qmm_enter' for e in native),
                'NATIVE_WRITER_DENOMINATOR_LOSS')
        if mode is not None:
            require(row['admissions'] == row['completed'] == self._expected_admissions_for(row) and not self.failure and not self.pending,
                    'ACTUAL_CANDIDATE_WRITER_LIFETIME_OR_EFFECT_INCOMPLETE')
            self._check_denominator(row)
            if mode != 'fault':
                require(model_output_finite and output_sha256 == self.baseline, 'FULL_VOCAB_LOSSLESS_OR_OFF_CHANGED')
        else:
            require(model_output_finite and not self.failure, 'HEALTHY_WRITER_INVENTORY_FAILED')
            if row['phase'] == 'health':
                require(output_sha256 == self.baseline and self.native_removed, 'SAME_WORKER_FULL_VOCAB_AFTER_NATIVE_RELEASE_CHANGED')
        self._idle(not self.native_removed)
        if self.public_output is not None:
            self.public_output.close()
        self.output_refs.clear()
        row['metadata_arrays'] = self.metadata.latest
        self.retained.clear()
        if len(self.rows) == 2:
            self._freeze()
        if self.stage == 'effects' and self.phase != 'complete' and self.entry['mode'] == 'fault' and not self.runtime.state()['core']['sealed']:
            self.runtime.seal(tuple(self.targets))
        if self.stage == SINGLE_SITE_STAGE and row['phase'] == 'single_site_qualification_off':
            key = self.single_selection['target_key']
            require([outcome['mode'] for outcome in self.outcomes] == ['reference', 'lossless', 'off']
                and all(outcome['target'] == key and outcome['released_receipt']['released'] == 1
                        for outcome in self.outcomes), 'SINGLE_SITE_FRESH_B_L_OFF_REQUIRED')
            # The Native engine independently seals exactly the leases it has
            # qualified. Unselected census entries were passthrough, not leases.
            self.runtime.seal((key,))
            self.single_qualification = dict(target_key=key, request_ids=[outcome['request_id'] for outcome in self.outcomes],
                outcomes_sha256=digest(self.outcomes), owner=getattr(self.runtime, 'owner', None),
                native_owner=[self.runtime.config.owner_hi, self.runtime.config.owner_lo],
                generation=self.runtime.config.generation, lifetime_sha256=digest(self.lifetime),
                build=self.runtime.build_identity, baseline_output_sha256=self.baseline, scope=SINGLE_SITE_SCOPE,
                serializable_authority=False, qualification_granted=False)
        self.recovered = self.phase == 'complete' and row['phase'] == 'health' and output_sha256 == self.baseline
        self.save()

    def _expected_admissions_for(self, row):
        return 1

    def _planned_qualification_requests(self):
        if self.stage == SINGLE_SITE_STAGE:
            return 6  # B/L/OFF/F/OFF and one healthy request, full census each.
        return len(self.targets) * (5 if self.stage == 'effects' else 3) + 1

    def recover_live_pending(self):
        """Retry only while actual public output/residual tensors remain owned."""
        self.runtime.stop()
        self.runtime.drain()
        for token, outcome in list(self.pending.items()):
            call = outcome['native_call']
            owned = self.pending_lifetimes.get(token)
            writer = owned.get('writer') if owned is not None else None
            # The inventory advances its current producer after AddRms returns.
            # The selected lease independently holds the consumed allocation.
            require(owned is not None and owned['producer_sequence'] == call['producer_sequence']
                and owned['request_id'] == outcome['request_id']
                and held_storage(owned['residual'], call['residual'], call['residual_bytes'])
                and held_storage(owned.get('consumer'), call['residual'], call['residual_bytes'])
                and held_storage(writer, call['output'], call['output_bytes']),
                'PENDING_OFF_POINTER_LIVENESS_UNVERIFIED_DO_NOT_WRITE')
            self.runtime.restore(token)
            outcome['recovery_receipt'] = self.runtime.acknowledge(token)
            self.outcomes.append(outcome)
            del self.pending[token]
            del self.pending_lifetimes[token]

    def close(self):
        self._thread()
        require(not self.active, 'ACTIVE_WRITER_REQUEST_CANNOT_DETACH')
        if self.installed:
            if self.runtime.state()['core']['pending']:
                self.recover_live_pending()
            if not self.native_removed:
                self.runtime.close()
                self.native_removed = True
            if self.public_output is not None:
                self.public_output.close()
            import torch
            require(torch.ops._C_ascend.npu_add_rms_norm_bias is self.producer_observer, 'PUBLIC_PRODUCER_OWNER_CHANGED')
            for handle in reversed(self.handles):
                handle.remove()
                hooks = handle.hooks_dict_ref()
                require(hooks is None or handle.id not in hooks, 'WRITER_MODULE_HOOK_REMOVAL_UNCONFIRMED')
            self.handles.clear()
            torch.ops._C_ascend.npu_add_rms_norm_bias = self.original_producer
            self.installed = False
        self.output_refs.clear()
        self.retained.clear()
        # Native close verified zero pending/executing/resources above. Any
        # remaining diagnostic outcome owns no Native memory or restore right.
        self.pending_lifetimes.clear()
        if self.fia_bridge is not None:
            self.fia_bridge.close()
        self.save()

    def report(self):
        value = dict(schema='demoncat.writer26.candidate_experiment.v2', mechanism=self.runtime.mechanism,
            stage=self.stage, phase=self.phase, failure=self.failure, first_failure=self.failure,
            failure_events=self.failure_events, last_native_receipt=self.last_native_receipt, blockers=self.blockers,
            scope='SINGLE_FRESH_PREFILL_ONE_CURRENT_TARGET_PER_REQUEST_NOT_DECODE_NOT_C2',
            lifetime=self.lifetime, build=self.runtime.build_identity, rows=self.rows, candidates=self.candidate_inventory,
            metadata_arrays_initial=self.metadata_initial, metadata_arrays_latest=self.metadata.latest,
            resolved_candidate_count=len(self.targets), qualified_candidate_count=sum(1 for key in self.targets if
                any(r['target'] == key and r['mode'] == 'lossless' and r.get('released_receipt') for r in self.outcomes)),
            outcomes=self.outcomes, native_state=self.runtime.state(), owned_hooks=len(self.handles),
            pending_output_tensors=len(self.output_refs), pending_residual_tensors=len(self.retained),
            pending_per_call_leases=list(self.pending),
            pending_outcomes=list(self.pending.values()), pending_public_storage_owners=len(self.pending_lifetimes),
            public_output_observer=self.public_output.report() if self.public_output is not None else None,
            fia_typed_stock_bridge=self.fia_bridge.report() if self.fia_bridge is not None else None,
            owned_public_api_wrappers=len(self.public_output.owned) if self.public_output is not None else 0,
            healthy_repeated=len(self.rows) >= 2 and self.rows[0]['output_sha256'] == self.rows[1]['output_sha256'],
            recovered=self.recovered, installed=self.installed, native_removed=self.native_removed,
            native_qualified_targets=0, c1=False, c2=False, c3=False,
            remaining_gates=['REVIEW_COMPLETE_VALID_NATURAL_INVENTORY', 'COMMON_ALL_ELIGIBLE_SESSION_NATIVE_DISPATCH',
                            'SAME_WORKER_LIVE_C2_BEFORE_EVERY_MATCH'])
        if self.stage == SINGLE_SITE_STAGE:
            modes = [row['mode'] for row in self.outcomes]
            state = value['native_state']
            native_released = (not any(state['core'][name] for name in
                ('installed', 'pending', 'executing', 'device_bytes', 'descriptor_pending', 'error'))
                and not any(state[name] for name in ('callback_active', 'completion_active', 'descriptor',
                                                     'prepared_fia', 'foreign_calls', 'rejected_calls')))
            complete = (self.recovered and not self.installed and not self.failure and not self.blockers
                and self.single_qualification is not None and native_released
                and not self.pending and not self.pending_lifetimes and not self.handles and not self.root_hooks
                and not self.output_refs and not self.retained and len(self.rows) == 8
                and modes == ['reference', 'lossless', 'off', 'fault', 'off'])
            value.update(scope=SINGLE_SITE_SCOPE, alias_inventory=self.single_aliases,
                single_site_c1=dict(schema='demoncat.writer26.single_site_c1.v1', selection=self.single_selection,
                    reference_lossless_off=self.single_qualification, evidence_complete=complete,
                    native_released=native_released, owned_root_logits_hooks=self.root_hooks,
                    selected_native_admissions=sum(row['admissions'] for row in self.rows),
                    actual_fault_hits=sum(row['mode'] == 'fault' for row in self.outcomes),
                    whole_request_effect_observed=any(row['plan']['mode'] == 'fault'
                        and row['output_sha256'] != self.baseline for row in self.rows),
                    complete_native_counts=[len(row['candidates']) for row in self.rows],
                    all_eligible_qualification=False, qualification_granted=False, serializable_authority=False,
                    fallback_attempted=False, e5='UNVERIFIED_INDEPENDENT_RAW_B_F_COMPARISON_REQUIRED'))
        return json.loads(json.dumps(value, allow_nan=False))
