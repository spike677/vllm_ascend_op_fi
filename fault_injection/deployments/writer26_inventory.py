"""Read-only all-writer lifetime census; no native qualification or admission."""
from collections import Counter
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import threading

from .residual26 import FCALL, QCALL, FiaContext, QmmContext, allocation_pair


KINDS = {1: 'qmm_enter', 2: 'qmm_exit', 3: 'fia_before', 4: 'fia_gws', 5: 'fia_enter',
         6: 'fia_exit', 7: 'add_rms_gws', 8: 'add_rms_enter', 9: 'add_rms_exit', 10: 'qmm_registration'}


class InventoryEvent(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ('version', 'size', 'kind', 'flags')]
    _fields_ += [(name, C.c_int32) for name in ('device', 'status')]
    _fields_ += [(name, C.c_uint64) for name in ('sequence', 'thread', 'call', 'registration', 'executor', 'cache',
                                               'producer', 'output', 'output_bytes', 'residual', 'residual_bytes')]
    _fields_ += [(name, C.c_int64) for name in ('rows', 'columns', 'residual_rows', 'residual_columns')]
    _fields_ += [(name, C.c_int32) for name in ('dtype', 'format')]
    _fields_ += [(name, C.c_uint32) for name in ('dimensions', 'residual_dimensions')]
    _fields_ += [(name, C.c_int64 * 8) for name in ('shape', 'strides', 'residual_shape', 'residual_strides')]


class InventoryState(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ('version', 'size', 'flags', 'reserved')]
    _fields_ += [(name, C.c_uint64) for name in ('capacity', 'total_events', 'stored_events', 'overflow_events', 'errors',
        'all_runs', 'qmm_runs', 'non_target_runs', 'qmm_enter', 'qmm_exit', 'fia_before', 'fia_gws', 'fia_enter', 'fia_exit',
        'add_rms_gws', 'add_rms_enter', 'add_rms_exit', 'qmm_registration')]


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def packet(value):
    return {name: list(getattr(value, name)) if isinstance(getattr(value, name), C.Array)
            else int(getattr(value, name)) for name, _ in value._fields_}


def tensor_matches_native(tensor, event, *, residual=False):
    """Address alone does not bind a public view to the complete native formal."""
    prefix = 'residual_' if residual else ''
    address = event['residual' if residual else 'output']
    size = event['residual_bytes' if residual else 'output_bytes']
    dimensions = event[prefix + 'dimensions']
    return (0 < dimensions <= 8 and tensor.storage_offset() == 0
        and tensor.is_contiguous() and tensor.untyped_storage().data_ptr() == address
        and tensor.untyped_storage().nbytes() == size
        and tensor.numel() * tensor.element_size() == size
        and list(tensor.shape) == event[prefix + 'shape'][:dimensions]
        and list(tensor.stride()) == event['residual_strides' if residual else 'strides'][:dimensions])


def allocation_segments(snapshot, addresses):
    """Preserve exact relevant allocator records, including unsuccessful adjacency."""
    require(isinstance(snapshot, list), 'ACTUAL_ALLOCATOR_SNAPSHOT_REQUIRED')
    rows = []
    for segment in snapshot:
        start = segment['address']
        end = start + sum(block['size'] for block in segment['blocks'])
        if any(start <= address < end for address in addresses):
            rows.append(segment)
    require(len(rows) <= 2, 'AMBIGUOUS_ALLOCATOR_SEGMENTS')
    return rows


class NativeWriterInventory:
    def __init__(self, library, sha256):
        source = Path(library).resolve(strict=True)
        require(hashlib.sha256(source.read_bytes()).hexdigest() == sha256, 'NATIVE_INVENTORY_BUILD_CHANGED')
        self.library_sha256 = sha256
        self.lib = C.CDLL(str(source), mode=os.RTLD_LAZY | os.RTLD_NOLOAD)
        self.state_fn = self.lib.dcat26_inventory_state_v1
        self.read_fn = self.lib.dcat26_inventory_read_v1
        self.state_fn.argtypes, self.state_fn.restype = [C.POINTER(InventoryState)], C.c_int
        self.read_fn.argtypes, self.read_fn.restype = [C.c_uint64, C.POINTER(InventoryEvent)], C.c_int
        require(C.sizeof(InventoryEvent) == 416 and C.sizeof(InventoryState) == 160, 'NATIVE_INVENTORY_ABI_SIZE_CHANGED')

    def state(self):
        value = InventoryState(version=1, size=C.sizeof(InventoryState))
        require(self.state_fn(C.byref(value)) == 0 and value.version == 1
                and value.size == C.sizeof(value), 'NATIVE_INVENTORY_STATE_ABI_CHANGED')
        row = packet(value)
        require(row['flags'] == 7 and row['capacity'] == 32768 and row['reserved'] == 0,
                'COMPLETE_NATIVE_TRACE_AND_TOPOLOGY_REQUIRED')
        require(row['overflow_events'] == row['errors'] == 0 and row['stored_events'] == row['total_events'],
                'NATIVE_INVENTORY_OVERFLOW_OR_ERROR')
        require(row['all_runs'] == row['qmm_runs'] + row['non_target_runs'], 'NATIVE_RUN_COUNTER_LOSS')
        return row

    def read(self, index):
        value = InventoryEvent(version=1, size=C.sizeof(InventoryEvent))
        require(self.read_fn(index, C.byref(value)) == 0 and value.version == 1 and value.size == C.sizeof(value),
                'NATIVE_INVENTORY_EVENT_ABI_CHANGED')
        row = packet(value)
        require(row['sequence'] == index + 1 and row['kind'] in KINDS, 'NATIVE_INVENTORY_EVENT_GAP')
        row['kind_name'] = KINDS[row['kind']]
        return row

    def delta(self, before):
        after = self.state()
        require(all(after[name] >= before[name] for name in before if name not in ('flags', 'version', 'size', 'reserved')),
                'NATIVE_INVENTORY_COUNTER_RESET')
        rows = [self.read(index) for index in range(before['stored_events'], after['stored_events'])]
        counts = Counter(row['kind_name'] for row in rows)
        require(all(counts[name] == after[name] - before[name] for name in KINDS.values()),
                'NATIVE_INVENTORY_DETAILS_DO_NOT_MATCH_COUNTERS')
        return after, rows


def correlate_lifetimes(native_rows, callback_rows, producer_rows, mechanism):
    """Keep every native writer, including unresolved module/allocation/consumer."""
    start_kind, finish_kind = (('fia_before', 'fia_exit') if mechanism == '26.fia_destination'
                               else ('qmm_enter', 'qmm_exit'))
    candidates = [row for row in native_rows if row['kind_name'] == start_kind]
    callbacks = {row['independent_call']: row for row in callback_rows if row.get('independent_call')}
    require(len(callbacks) == sum(bool(row.get('independent_call')) for row in callback_rows), 'DUPLICATE_WRITER_CALLBACK')
    results = []
    for row in candidates:
        reasons = []
        callback = callbacks.get(row['call'])
        if callback is None:
            reasons.append('NATIVE_WRITER_HAS_NO_OWNED_PUBLIC_CALLBACK')
        elif callback['errors']:
            reasons.extend(callback['errors'])
        endings = [end for end in native_rows if end['kind_name'] == finish_kind and end['call'] == row['call']]
        if len(endings) != 1 or endings[0]['status'] != 0:
            reasons.append('ACTUAL_WRITER_EXECUTION_NOT_COMPLETED')
        producers = [producer for producer in producer_rows if producer.get('native_producer') == row['producer']
                     and producer.get('output', {}).get('storage_addr') == row['residual']]
        if len(producers) != 1:
            reasons.append('ACTUAL_PUBLIC_PRODUCER_LIFETIME_UNPROVEN')
        elif producers[0]['errors']:
            reasons.extend(producers[0]['errors'])
        consumers = [end for end in native_rows if end['kind_name'] == 'add_rms_exit'
                     and end['sequence'] > row['sequence'] and end['residual'] == row['residual']
                     and end['residual_bytes'] == row['residual_bytes']]
        if not consumers or consumers[0]['status'] != 0 or consumers[0]['producer'] != row['producer'] + 1:
            reasons.append('NEXT_ACTUAL_NATIVE_CONSUMER_UNPROVEN')
        results.append(dict(native=row, callback=callback, execution=endings,
            public_producer=producers, next_consumer=consumers[:1], reasons=reasons,
            eligibility='OBSERVED_WRITER_LIFETIME_NOT_LOSSLESS_QUALIFIED' if not reasons else 'UNRESOLVED',
            native_qualified=False, c2=False, c3=False))
    return results


class WriterLifetimeInventory:
    def __init__(self, model, runtime, reader, *, worker_identity, folder):
        self.model, self.runtime, self.reader = model, runtime, reader
        self.worker_identity, self.folder = worker_identity, Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.thread = threading.get_ident()
        self.native_thread = threading.get_native_id()
        from fault_injection.runtime_discovery import _process_start_identity
        self.lifetime = dict(worker=worker_identity, pid=os.getpid(), native_thread=self.native_thread,
            process_start_identity=_process_start_identity(), model_object_identity=id(model),
            native_inventory_sha256=reader.library_sha256, native_writer_sha256=runtime.sha256)
        self.current = self.frame = None
        self.rows, self.requests, self.handles, self.stack = [], set(), [], []
        self.callbacks, self.producers, self.retained = [], [], {}
        self.modules = {id(module): (name or '<root>', module) for name, module in model.named_modules()}
        self.installed = self.recovered = self.fault_started = self.recovering = False
        self.failure = None
        self.original_producer = self.producer_observer = self.callback = None
        self.owned_hooks = 0

    @property
    def active(self):
        return self.current is not None

    @property
    def phase(self):
        return 'inventory_' + str(len(self.rows) + 1) if len(self.rows) < 2 else 'complete'

    def _thread(self):
        require(threading.get_ident() == self.thread, 'FOREIGN_WRITER_INVENTORY_THREAD')

    def _idle(self, registered):
        state = self.runtime.native_state()
        require(bool(state['registered']) is registered and not any(state[name] for name in (
            'prepared', 'queued_executors', 'pending_consumers', 'active_consumer', 'callback_active', 'environment_enabled')),
            'WRITER_NATIVE_LEASE_OR_ARM_PRESENT')
        return state

    def install(self):
        import torch
        self._thread()
        require(not self.installed, 'WRITER_INVENTORY_ALREADY_INSTALLED')
        self._idle(False)
        self.reader.state()
        self.original_producer = torch.ops._C_ascend.npu_add_rms_norm_bias
        def producer(*args, **kwargs):
            if self.current is None:
                return self.original_producer(*args, **kwargs)
            return self._producer(args, kwargs)
        self.producer_observer = producer
        torch.ops._C_ascend.npu_add_rms_norm_bias = producer
        self.callback = (FCALL if self.runtime.fia else QCALL)(self._callback)
        result = (self.runtime.install_fn(self.callback, self.runtime.session) if self.runtime.fia
                  else self.runtime.install_fn(self.callback))
        if result:
            torch.ops._C_ascend.npu_add_rms_norm_bias = self.original_producer
            raise RuntimeError('READONLY_WRITER_CALLBACK_INSTALL_FAILED')
        self.installed = True
        try:
            for _, module in self.modules.values():
                if module is self.model:
                    continue
                self.handles.append(module.register_forward_pre_hook(self._pre, with_kwargs=True))
                self.handles.append(module.register_forward_hook(self._post, with_kwargs=True, always_call=True))
            self.owned_hooks = len(self.handles)
            self._idle(True)
        except BaseException:
            self.close()
            raise

    def _pre(self, module, args, kwargs):
        if self.current is not None:
            self.stack.append(module)

    def _post(self, module, args, kwargs, output):
        if self.current is not None:
            require(self.stack and self.stack[-1] is module, 'ACTUAL_WRITER_MODULE_STACK_CHANGED')
            self.stack.pop()

    def begin(self, request_id, input_sha256):
        self._thread()
        self._idle(True)
        require(self.current is None and request_id not in self.requests and self.phase != 'complete',
                'FRESH_BOUNDED_WRITER_REQUEST_REQUIRED')
        require(not self.retained, 'RESIDUAL_LIFETIME_ESCAPED_PREVIOUS_REQUEST')
        self.current = dict(phase=self.phase, request_id=request_id, input_sha256=input_sha256,
            native_before=self.reader.state(), callback_start=len(self.callbacks), producer_start=len(self.producers))
        self.requests.add(request_id)
        self.stack = [self.model]

    def _producer(self, args, kwargs):
        import torch
        from demoncat.runtime.vllm_ascend.worker_extension.serving20 import live_tensor
        before = self.reader.state()
        inputs = [tensor for tensor in (*args, *kwargs.values()) if isinstance(tensor, torch.Tensor)]
        result = self.original_producer(*args, **kwargs)  # Exactly one actual producer/consumer.
        row = dict(module_object_identity=id(self.stack[-1]), module_path=self.modules[id(self.stack[-1])][0],
            request_id=self.current['request_id'], original_calls=1, errors=[])
        self.producers.append(row)
        try:
            _, native = self.reader.delta(before)
            row['native_events'] = native
        except BaseException as error:
            row['errors'].append(str(error))
            return result
        completed = [event for event in native if event['kind_name'] == 'add_rms_exit']
        if len(completed) != 1 or completed[0]['status'] != 0:
            row['errors'].append('ONE_COMPLETED_NATIVE_ADD_RMS_REQUIRED')
            return result
        actual = completed[0]
        outputs = [tensor for tensor in result if isinstance(tensor, torch.Tensor)] if isinstance(result, (tuple, list)) else [result]
        output = [tensor for tensor in outputs if isinstance(tensor, torch.Tensor)
                  and tensor.untyped_storage().data_ptr() == actual['output']]
        consumed = [tensor for tensor in inputs if tensor.untyped_storage().data_ptr() == actual['residual']]
        if len(output) != 1 or len(consumed) != 1:
            row['errors'].append('PUBLIC_NATIVE_PRODUCER_OR_CONSUMER_STORAGE_UNPROVEN')
            return result
        if not tensor_matches_native(output[0], actual) or not tensor_matches_native(consumed[0], actual, residual=True):
            row['errors'].append('PUBLIC_NATIVE_RESIDUAL_FORMAL_GEOMETRY_UNPROVEN')
            return result
        previous = self.retained.pop(actual['residual'], None)
        row.update(native_producer=actual['producer'], output=live_tensor(output[0]),
            consumed=live_tensor(consumed[0]), consumed_retained_producer=previous['sequence'] if previous else None)
        self.retained[actual['output']] = dict(tensor=output[0], sequence=actual['producer'],
            request_id=self.current['request_id'], public=row)
        return result

    def _callback(self, pointer, token=None):
        row = dict(original_continuations=0, claimed=None, errors=[], independent_call=None)
        try:
            p = pointer.contents
            token = p.token if self.runtime.fia else token
            if self.current is None:
                row['original_continuations'] = 1
                row['claimed'] = int(self.runtime.cont(token))
                if row['claimed'] != 0:
                    self.failure = 'READONLY_OUT_OF_SCOPE_WRITER_UNEXPECTEDLY_CLAIMED'
                return 0
            self._thread()
            self.callbacks.append(row)
            require(self.stack and self.stack[-1] is self.modules[id(self.stack[-1])][1], 'ACTUAL_WRITER_MODULE_REQUIRED')
            owner = self.stack[-1]
            row.update(module_object_identity=id(owner), module_path=self.modules[id(owner)][0],
                actual_request_id=self.current['request_id'], native_packet=packet(p), token=int(token))
            require((p.version == 1 and p.size == C.sizeof(FiaContext) and p.metadata_ok == 1)
                    if self.runtime.fia else p.abi_version == 1,
                    'NATIVE_WRITER_CALLBACK_ABI_CHANGED')
            state = self.reader.state()
            independent = self.reader.read(state['stored_events'] - 1)
            expected = 'fia_before' if self.runtime.fia else 'qmm_enter'
            output = p.output if self.runtime.fia else p.original_output
            require(independent['kind_name'] == expected and independent['thread'] == self.native_thread
                    and independent['device'] == p.device and independent['output'] == output
                    and independent['output_bytes'] == p.output_bytes and independent['residual'] == p.residual
                    and independent['residual_bytes'] == p.residual_bytes,
                    'WRITER_CALLBACK_NOT_BOUND_TO_INDEPENDENT_NATIVE_EVENT')
            row.update(independent_call=independent['call'], independent_event=independent)
            retained = self.retained.get(p.residual)
            require(retained is not None and retained['request_id'] == self.current['request_id']
                    and retained['sequence'] == independent['producer'], 'CURRENT_PUBLIC_RESIDUAL_LIFETIME_UNPROVEN')
            row.update(producer_tensor_object_identity=id(retained['tensor']), producer_sequence=retained['sequence'],
                       residual_retained_only_until_actual_consumer=True)
            require(tensor_matches_native(retained['tensor'], independent, residual=True),
                    'RETAINED_RESIDUAL_FORMAL_GEOMETRY_CHANGED')
            if self.runtime.fia:
                require(output != p.residual and 0 < p.output_bytes <= p.residual_bytes,
                        'INDEPENDENT_FIA_DESTINATION_GEOMETRY_UNSUPPORTED')
            else:
                import torch_npu
                snapshot = torch_npu.npu.memory_snapshot()
                row['allocator_snapshot_sha256'] = digest(snapshot)
                relevant = allocation_segments(snapshot, (p.residual, output))
                path = self.folder / ('allocator-' + str(independent['sequence']) + '.json')
                path.write_text(json.dumps(relevant, indent=2, allow_nan=False) + '\n')
                row['allocator_segments'] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    complete_snapshot_sha256=row['allocator_snapshot_sha256'], relevant_segment_count=len(relevant))
                row['allocation'] = allocation_pair(snapshot, p.residual, output, p.residual_bytes, p.output_bytes)
                row['allocation_lifetime_scope'] = 'ACTIVE_BLOCKS_AT_ACTUAL_NATIVE_WRITER_CALLBACK'
        except BaseException as error:
            row['errors'].append(str(error))
        finally:
            if token is not None and row['original_continuations'] == 0:
                row['original_continuations'] += 1
                row['claimed'] = int(self.runtime.cont(token))
                if row['claimed'] != 0:
                    self.failure = 'READONLY_NATIVE_WRITER_UNEXPECTEDLY_CLAIMED'
        return 0

    def end(self, output_sha256, *, model_output_finite):
        self._thread()
        require(self.current is not None and self.stack == [self.model], 'COMPLETE_ACTUAL_WRITER_FORWARD_REQUIRED')
        self._idle(True)
        row = self.current
        after, native = self.reader.delta(row['native_before'])
        require(after['qmm_runs'] - row['native_before']['qmm_runs'] == sum(e['kind_name'] == 'qmm_enter' for e in native),
                'NATIVE_QMM_WRITER_DENOMINATOR_INCOMPLETE')
        callbacks = self.callbacks[row['callback_start']:]
        producers = self.producers[row['producer_start']:]
        row.update(native_after=after, native_events=native, callbacks=callbacks, producers=producers,
            candidates=correlate_lifetimes(native, callbacks, producers, self.runtime.mechanism),
            output_sha256=output_sha256, output_finite=model_output_finite,
            terminal_residuals=[dict(address=address, producer=value['sequence']) for address, value in self.retained.items()])
        self.retained.clear()  # End of the actual request, never carry into decode.
        self.stack.clear()
        self.current = None
        self.rows.append(row)
        require(model_output_finite and not self.failure and all(callback['claimed'] == 0 for callback in callbacks),
                'READONLY_NATIVE_WRITER_CONTROL_FAILED')
        if len(self.rows) == 2:
            require(self.rows[0]['input_sha256'] == row['input_sha256']
                    and self.rows[0]['output_sha256'] == output_sha256, 'WHOLE_WORKER_HEALTHY_INVENTORY_NOT_REPEATABLE')
            self.recovered = True
        self.save()

    def close(self):
        self._thread()
        require(not self.active, 'ACTIVE_WRITER_INVENTORY_CANNOT_DETACH')
        if self.installed:
            self._idle(True)
            require(self.runtime.remove_fn(self.callback) == 0, 'READONLY_NATIVE_WRITER_REMOVE_FAILED')
            import torch
            require(torch.ops._C_ascend.npu_add_rms_norm_bias is self.producer_observer,
                    'PUBLIC_RESIDUAL_OBSERVER_OWNER_CHANGED')
            torch.ops._C_ascend.npu_add_rms_norm_bias = self.original_producer
            for handle in reversed(self.handles):
                handle.remove()
            self.handles.clear()
            self.owned_hooks = 0
            self.installed = False
        self._idle(False)
        self.retained.clear()
        self.save()

    def fail(self, reason):
        self.failure = str(reason)
        if self.current is not None:
            self.current['failure'] = self.failure
            try:
                after, native = self.reader.delta(self.current['native_before'])
                callbacks = self.callbacks[self.current['callback_start']:]
                producers = self.producers[self.current['producer_start']:]
                self.current.update(native_after=after, native_events=native, callbacks=callbacks, producers=producers,
                    candidates=correlate_lifetimes(native, callbacks, producers, self.runtime.mechanism))
            except BaseException as error:
                self.current['inventory_failure'] = str(error)
            (self.folder / 'failed_request.json').write_text(json.dumps(self.current, indent=2, allow_nan=False) + '\n')
        self.current = None
        self.stack.clear()
        self.save()

    def query(self):
        return {'qualification': self.report()}

    def report(self):
        return dict(status='READONLY_NATIVE_WRITER_INVENTORY' if not self.failure else 'INVENTORY_FAILED',
            mechanism=self.runtime.mechanism, phase=self.phase, error=self.failure,
            lifetime=self.lifetime, lifetime_sha256=digest(self.lifetime),
            scope='TWO_ACTUAL_SINGLE_REQUEST_FRESH_PREFILLS_NOT_DECODE',
            native_injections=0 if not self.failure and all(row['claimed'] == 0 for row in self.callbacks) else None,
            native_prepare_claims=sum(row['claimed'] == 1 for row in self.callbacks),
            qualified_native_targets=0, c1=False, c2=False, c3=False,
            healthy_repeated=self.recovered, owned_hooks=len(self.handles), installed=self.installed,
            rows=self.rows, pending_residual_tensor_references=len(self.retained),
            remaining_gates=['per-target lossless writer substitution and live allocation lease',
                'sealed all-target group admission', 'per-target effects and all-target OFF/drain/release',
                'same-Worker C2 proof before every_match'])

    def save(self):
        (self.folder / 'writer_inventory.json').write_text(json.dumps(self.report(), indent=2, allow_nan=False) + '\n')
