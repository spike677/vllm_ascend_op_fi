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
    return ('OBSERVED_NOT_QUALIFIED', []) if not reasons else ('UNVERIFIED', sorted(set(reasons)))


class Writer26Experiment(WriterLifetimeInventory):
    """One current target per fixed fresh request, all native calls still observed.

    Stage inventory is write-free. Stage lossless tests every resolved candidate
    if the explicit target budget covers the full resolved set. Stage effects
    adds one F/OFF pair per such target, preserving every unresolved row and
    explicitly keeping C2/C3 false. This is a bounded candidate qualification
    entrance, not a substitute for a future complete CoverageSession proof.
    """
    def __init__(self, *args, stage='inventory', max_targets=8, **kwargs):
        super().__init__(*args, **kwargs)
        require(stage in ('inventory', 'lossless', 'effects') and type(max_targets) is int and 1 <= max_targets <= 128,
                'BOUNDED_WRITER_EXPERIMENT_STAGE_REQUIRED')
        self.stage, self.max_targets = stage, max_targets
        self.plan = [dict(phase='inventory_1', mode=None, target=None), dict(phase='inventory_2', mode=None, target=None)]
        self.targets, self.target_refs, self.output_refs = {}, {}, {}
        self.pending, self.consumer_inputs = {}, None
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
        require(self.current is None and request_id not in self.requests and self.phase != 'complete',
                'FRESH_BOUNDED_WRITER_REQUEST_REQUIRED')
        if self.phase == 'health':
            self._before_health()
        self._idle(not self.native_removed)
        require(not self.retained, 'RESIDUAL_LIFETIME_ESCAPED_PREVIOUS_REQUEST')
        self.current = dict(phase=self.phase, request_id=request_id, input_sha256=input_sha256,
            native_before=self.reader.state(), callback_start=len(self.callbacks), producer_start=len(self.producers))
        self.requests.add(request_id)
        self.stack = [self.model]
        require(not self.output_refs and not self.pending, 'WRITER_LIFETIME_ESCAPED_ACTUAL_REQUEST')
        if self.phase == 'inventory_1':
            self.baseline_input = input_sha256
        require(input_sha256 == self.baseline_input, 'ACTUAL_FRESH_WORKLOAD_CHANGED')
        self.current.update(plan=dict(self.entry), admissions=0, completed=0)
        if self.entry['mode'] == 'fault':
            self.fault_started = True
        if self.public_output is not None and not self.native_removed:
            # The serving host has already started its optional collector.
            self.public_output.install()

    def _before_health(self):
        self.runtime.close()
        self.native_removed = True

    def _observe_callback(self, row, call):
        pass

    def _requested_mode(self, row):
        return self.entry['mode'], self.entry['target']

    def _expected_admissions(self):
        return 1

    def _outcome_complete(self, outcome, row):
        pass

    def _request_complete(self, row):
        pass

    def _check_denominator(self, row):
        require(len(row['candidates']) == len(self.rows[0]['candidates']), 'NATURAL_WRITER_COUNT_CHANGED_KEEP_FULL_DENOMINATOR')

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
                self.output_refs.pop(call['token'], None)
            return result
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
                from demoncat.runtime.vllm_ascend.worker_extension.serving import complete_tensor
                _, residual_sha, _ = complete_tensor(retained['tensor'])
                self.pending[call.token] = dict(mode=mode, target=selected, native_call=packet(call),
                    residual_before_sha256=residual_sha, request_id=self.current['request_id'])
                row['original_continuations'] = 1
                self.pending[call.token]['lease'] = self.runtime.execute(call, mode, selected, row.get('allocation'))
                row['claimed'] = 1
                self.current['admissions'] += 1
        except BaseException as error:
            row['errors'].append(str(error))
            if (self.current is not None and self.entry['mode'] is not None
                    and (row.get('target_key') == self.entry['target'] or
                         self.entry['target'] == '*' and row.get('target_key') in self.targets)):
                self.failure = str(error)
        finally:
            if not observed and row.get('independent_event') is not None:
                try:
                    self._observe_callback(row, call)
                except BaseException as error:
                    self.failure = str(error)
            if not row['original_continuations']:
                row['original_continuations'] = 1
                try:
                    self.runtime.passthrough(call)
                except BaseException as error:
                    self.failure = str(error)
        return 0 if self.failure is None else 1

    def _completed(self, pointer):
        try:
            receipt = packet(pointer.contents)
            outcome = self.pending[receipt['token']]
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
            outcome['consumer_receipt'] = receipt
            outcome['released_receipt'] = self.runtime.acknowledge(receipt['token'])
            require(outcome['released_receipt']['released'] == 1, 'ACTUAL_NATIVE_WRITER_LEASE_NOT_RELEASED')
            return 0
        except BaseException as error:
            self.failure = str(error)
            return 1

    def _freeze(self):
        require(self.rows[0]['input_sha256'] == self.rows[1]['input_sha256']
                and self.rows[0]['output_sha256'] == self.rows[1]['output_sha256'], 'HEALTHY_COMPLETE_REQUEST_CHANGED')
        self.baseline = self.rows[0]['output_sha256']
        left, right = self.rows[0]['candidates'], self.rows[1]['candidates']
        require(len(left) == len(right) and len(right) > 0, 'COMPLETE_NATIVE_WRITER_DENOMINATOR_CHANGED')
        counts = Counter(c['callback'].get('target_key') for c in right if c.get('callback'))
        for index, (before, candidate) in enumerate(zip(left, right)):
            status, reasons = classify_candidate(candidate)
            previous_status, _ = classify_candidate(before)
            callback, previous = candidate.get('callback'), before.get('callback')
            key = callback.get('target_key') if callback else None
            if status != previous_status or (status == 'OBSERVED_NOT_QUALIFIED' and
                    (previous is None or previous.get('target_key') != key or counts[key] != 1)):
                status, reasons = 'UNVERIFIED', ['HEALTHY_TARGET_LIFETIME_OR_OCCURRENCE_CHANGED']
            row = dict(index=index, target_key=key, status=status, reasons=reasons,
                first_native=before['native'], second_native=candidate['native'], callback=callback)
            self.candidate_inventory.append(row)
            if status == 'OBSERVED_NOT_QUALIFIED':
                self.targets[key] = row
        if self.stage == 'inventory':
            self.plan.append(dict(phase='health', mode=None, target=None))
            return
        if not self.targets or len(self.targets) > self.max_targets:
            self.blockers.append('EMPTY_OR_OVER_BUDGET_COMPLETE_RESOLVED_TARGET_SET')
            self.plan.append(dict(phase='health', mode=None, target=None))
            return
        per_request_events = self.rows[1]['native_after']['total_events'] - self.rows[1]['native_before']['total_events']
        remaining_requests = self._planned_qualification_requests()
        available = self.rows[1]['native_after']['capacity'] - self.rows[1]['native_after']['stored_events']
        if per_request_events * remaining_requests > available:
            self.blockers.append('COMPLETE_TARGET_PLAN_EXCEEDS_INDEPENDENT_NATIVE_EVENT_CAPACITY')
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
        self.recovered = self.phase == 'complete' and row['phase'] == 'health' and output_sha256 == self.baseline
        self.save()

    def _expected_admissions_for(self, row):
        return 1

    def _planned_qualification_requests(self):
        return len(self.targets) * (5 if self.stage == 'effects' else 3) + 1

    def recover_live_pending(self):
        """Retry only while actual public output/residual tensors remain owned."""
        self.runtime.stop()
        self.runtime.drain()
        for token, outcome in list(self.pending.items()):
            call = outcome['native_call']
            writer = self.output_refs.get(token)
            residual = self.retained.get(call['residual'])
            require(writer is not None and residual is not None
                and full_storage(writer, call['output'], call['output_bytes'])
                and full_storage(residual['tensor'], call['residual'], call['residual_bytes']),
                'PENDING_OFF_POINTER_LIVENESS_UNVERIFIED_DO_NOT_WRITE')
            self.runtime.restore(token)
            outcome['recovery_receipt'] = self.runtime.acknowledge(token)
            self.outcomes.append(outcome)
            del self.pending[token]

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
        self.save()

    def report(self):
        return dict(schema='demoncat.writer26.candidate_experiment.v2', mechanism=self.runtime.mechanism,
            stage=self.stage, phase=self.phase, failure=self.failure, blockers=self.blockers,
            scope='SINGLE_FRESH_PREFILL_ONE_CURRENT_TARGET_PER_REQUEST_NOT_DECODE_NOT_C2',
            lifetime=self.lifetime, build=self.runtime.build_identity, rows=self.rows, candidates=self.candidate_inventory,
            metadata_arrays_initial=self.metadata_initial, metadata_arrays_latest=self.metadata.latest,
            resolved_candidate_count=len(self.targets), qualified_candidate_count=sum(1 for key in self.targets if
                any(r['target'] == key and r['mode'] == 'lossless' and r.get('released_receipt') for r in self.outcomes)),
            outcomes=self.outcomes, native_state=self.runtime.state(), owned_hooks=len(self.handles),
            pending_output_tensors=len(self.output_refs), pending_residual_tensors=len(self.retained),
            pending_per_call_leases=list(self.pending),
            public_output_observer=self.public_output.report() if self.public_output is not None else None,
            owned_public_api_wrappers=len(self.public_output.owned) if self.public_output is not None else 0,
            healthy_repeated=len(self.rows) >= 2 and self.rows[0]['output_sha256'] == self.rows[1]['output_sha256'],
            recovered=self.recovered, installed=self.installed, native_removed=self.native_removed,
            native_qualified_targets=0, c1=False, c2=False, c3=False,
            remaining_gates=['REVIEW_COMPLETE_VALID_NATURAL_INVENTORY', 'COMMON_ALL_ELIGIBLE_SESSION_NATIVE_DISPATCH',
                            'SAME_WORKER_LIVE_C2_BEFORE_EVERY_MATCH'])
