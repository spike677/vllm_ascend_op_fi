"""Observe the real decode metadata producer before the model pre-hook.

No cosine formula, equal-value shortcut, or replay creates producer authority.
The real cache index, full buffer write, metadata return and consumer alias must
all occur in the same owned decode interval.
"""
import hashlib
from pathlib import Path
import sys
import threading

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode, _get_current_dispatch_mode_stack

from .quant_row_provenance import exit_owned_mode, metadata, snapshot, tensor_values
from .quant_source_rows import source_bindings, _arguments, _resolve, _same_extent
from .serving_context import require


REVIEWED_METADATA = {
    'vllm_ascend.attention.mla_v1': (
        '17892014ecd431a1804fb81f5380509275e1440b41e1f9e0870ba8440c365516',
        {'AscendMLAMetadataBuilder.build_decode_metadata': 'builder'}),
    'vllm_ascend.ops.rotary_embedding': (
        '74653da7274de182e8497f7d6b07ea7c6deeeef21e710e34eed5a349b0f671e7',
        {'get_cos_and_sin_mla': 'rotary'}),
}


def _key(tensor):
    value = metadata(tensor)
    return (value['device'], value['storage_address'], value['storage_bytes'], value['storage_identity'])


def _same_bytes_view(left, right):
    a, b = metadata(left), metadata(right)
    return (all(a[k] == b[k] for k in ('storage_address', 'storage_bytes', 'storage_identity', 'offset', 'dtype', 'device'))
            and left.is_contiguous() and right.is_contiguous() and left.numel() == right.numel())


class MlaMetadataRows:
    def __init__(self, runner, bindings, sources, *, max_calls=8192, simulation=False):
        self.runner, self.sources, self.max_calls, self.simulation = runner, snapshot(sources), max_calls, simulation
        require(type(max_calls) is int and 0 < max_calls <= 32768, 'BOUNDED_EARLY_METADATA_OBSERVER_REQUIRED')
        self.bindings = {row['function'].__code__: row for row in bindings}
        self.callback = self._profile
        self.active = False
        self.mode = None
        self.rows, self.pending, self.epochs, self.events, self.errors = [], {}, {}, [], []
        self.scope = self.actual_positions = None
        self.total = self.overflow = 0
        self.released = False
        self.profile_released = True
        self.closing = False

    @classmethod
    def from_runtime(cls, runner):
        return cls(runner, *source_bindings(REVIEWED_METADATA))

    def _runner_in_stack(self, frame):
        while frame is not None:
            if frame.f_locals.get('self') is self.runner:
                return True
            frame = frame.f_back
        return False

    def begin(self, expected_ids):
        require(not self.active and sys.getprofile() is None and _get_current_dispatch_mode() is None,
                'EXCLUSIVE_EARLY_METADATA_OBSERVER_REQUIRED')
        require(len(expected_ids) == len(set(expected_ids)) == 8, 'EIGHT_ACTUAL_PENDING_REQUESTS_REQUIRED')
        require(all(hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() == row['sha256']
                    for row in self.sources.values()), 'EARLY_METADATA_SOURCE_CHANGED')
        require(all(row['function'].__code__ is code and row['function'].__globals__ is vars(row['module'])
                and _resolve(row['module'], row['qualname']) is row['function']
                for code, row in self.bindings.items()), 'EARLY_METADATA_LOADED_CODE_CHANGED')
        self.expected_ids, self.thread = tuple(expected_ids), threading.get_ident()
        self.rows, self.pending, self.epochs, self.events, self.errors = [], {}, {}, [], []
        self.scope = self.actual_positions = None
        self.total = self.overflow = 0
        self.released = False
        self.profile_released, self.closing = True, False
        self.mode_scope_inference = None
        self.active = True
        # RPC arms only the profile observer. A dispatch mode entered here and
        # removed inside execute_model's inference_mode guard leaves stale
        # Python dispatch TLS when that guard restores its entry state.
        try:
            sys.setprofile(self.callback)
        finally:
            self.profile_released = sys.getprofile() is not self.callback

    def _enter_producer_mode(self):
        require(self.mode is None and _get_current_dispatch_mode() is None,
                'EXCLUSIVE_EARLY_METADATA_PRODUCER_MODE_REQUIRED')
        parent = self
        class Observe(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return parent._operator(func, args, kwargs or {})
        self.mode = Observe()
        self.mode_scope_inference = torch.is_inference_mode_enabled()
        try:
            self.mode.__enter__()
        except BaseException:
            if not any(value is self.mode for value in _get_current_dispatch_mode_stack()):
                self.mode = None
            raise

    def _write(self, tensors):
        changes = [_key(tensor) for tensor in tensors]
        known = set(self.epochs).union(changes)
        for key in known:
            if any(key[0] == other[0] and key[1] < other[1] + other[2]
                   and other[1] < key[1] + key[2] for other in changes):
                self.epochs[key] = self.epochs.get(key, 0) + 1

    def _operator(self, func, args, kwargs):
        self.total += 1
        if self.total > self.max_calls:
            self.overflow += 1
            return func(*args, **kwargs)
        event = dict(sequence=self.total, operator=str(func), original_calls=0, returned=False)
        self.events.append(event)
        live = dict(event=event, function=func, args=args, kwargs=kwargs)
        try:
            require(threading.get_ident() == self.thread, 'EARLY_METADATA_FOREIGN_THREAD')
            schema = getattr(func, '_schema', None)
            inputs = tensor_values((args, kwargs))
            event.update(schema=str(schema), inputs=[metadata(t) for t in inputs])
            for tensor in inputs:
                self.epochs.setdefault(_key(tensor), 0)
            event['input_epochs'] = [self.epochs[_key(tensor)] for tensor in inputs]
            if schema is None or schema.is_mutable:
                self._write(inputs)  # Every alias touched by an unknown write.
            for row in self.pending.values():
                if row['role'] in ('builder', 'rotary'):
                    row['operations'].append(live)
        except BaseException as error:
            self.errors.append('EARLY_METADATA_OBSERVER:' + repr(error))
        event['original_calls'] = 1
        output = func(*args, **kwargs)
        live['output'] = output
        event['returned'] = True
        try:
            event['outputs'] = [metadata(t) for t in tensor_values(output)]
            event['output_epochs'] = [self.epochs.get(_key(t), 0) for t in tensor_values(output)]
        except BaseException as error:
            self.errors.append('EARLY_METADATA_OUTPUT_OBSERVER:' + repr(error))
        return output

    def _profile(self, frame, event, result):
        if event not in ('call', 'return') or frame.f_code not in self.bindings:
            return
        row = None
        try:
            binding = self.bindings[frame.f_code]
            require(self.active and threading.get_ident() == self.thread
                and frame.f_globals is vars(binding['module'])
                and _resolve(binding['module'], binding['qualname']) is binding['function']
                and binding['function'].__code__ is frame.f_code, 'EARLY_MLA_LOADED_PRODUCER_CHANGED')
            if event == 'call':
                require(self._runner_in_stack(frame), 'METADATA_PRODUCER_HAS_NO_ACTUAL_RUNNER_OWNER')
                if binding['role'] == 'builder' and self.mode is None:
                    self._enter_producer_mode()
                require(self.mode is not None, 'ACTUAL_BUILDER_MUST_START_METADATA_INTERVAL')
                row = dict(sequence=len(self.rows) + 1, role=binding['role'], input=dict(frame.f_locals),
                    source_sha256=binding['source_sha256'], operations=[], status='INFLIGHT')
                if row['role'] == 'rotary':
                    row['builder'] = next((value for value in reversed(list(self.pending.values()))
                                           if value['role'] == 'builder'), None)
                    row['module'] = binding['module']
                else:
                    positions = row['input']['common_attn_metadata'].positions
                    require(isinstance(positions, torch.Tensor), 'ACTUAL_COMMON_METADATA_POSITIONS_REQUIRED')
                    row.update(source_positions=positions, source_positions_metadata=metadata(positions),
                               source_positions_epoch=self.epochs.setdefault(_key(positions), 0))
                self.rows.append(row)
                self.pending[id(frame)] = row
            else:
                row = self.pending[id(frame)]
                if row['role'] == 'rotary':
                    self._rotary_return(row, result)
                else:
                    self._builder_return(row, result)
                row['status'] = 'OBSERVED'
        except BaseException as error:
            if row is not None:
                row.update(status='UNVERIFIED', reason=str(error))
            else:
                self.errors.append(type(error).__name__ + ':' + str(error))
        finally:
            if event == 'return' and id(frame) in self.pending:
                row = self.pending.pop(id(frame))
                row.pop('operations', None)

    def _builder_return(self, row, result):
        """Bind the reviewed two executed slices to the exact returned object."""
        source = row['source_positions']
        before = row['source_positions_metadata']
        epoch = row['source_positions_epoch']
        returned = getattr(result, 'input_positions', None)
        row['returned_positions_metadata'] = metadata(returned) if isinstance(returned, torch.Tensor) else None
        require(result is not None and isinstance(getattr(result, 'input_positions', None), torch.Tensor)
                and row['input']['common_attn_metadata'].positions is source
                and metadata(source) == before and self.epochs.get(_key(source), 0) == epoch,
                'COMMON_METADATA_POSITIONS_CHANGED_DURING_BUILDER')
        require(before['dtype'] == str(torch.int64) and len(before['shape']) == 1
                and before['shape'][0] >= len(self.expected_ids) and before['stride'] == [1],
                'REVIEWED_CONTIGUOUS_COMMON_POSITION_VECTOR_REQUIRED')
        # The pinned builder executes common[:num_actual_tokens].long(), then
        # middle[:num_decode_tokens]. Follow actual output/input identities
        # backwards. An unrelated later source slice cannot substitute for
        # either edge just because its storage and values happen to match.
        slices = [op for op in row['operations'] if op['function'] is torch.ops.aten.slice.Tensor]
        chain, output = [], returned
        for _ in range(2):
            matches = [op for op in slices if op.get('output') is output]
            require(len(matches) == 1, 'NO_OBSERVED_PREFIX_SLICE_TO_METADATA_POSITION_RETURN')
            op = matches[0]
            chain.insert(0, op)
            output = _arguments(op['function'], op['args'], op['kwargs']).get('self')
        require(output is source and chain[0]['event']['sequence'] < chain[1]['event']['sequence'],
                'ACTUAL_TWO_POSITION_SLICE_SOURCE_OR_ORDER_CHANGED')
        tensors, links = [source], []
        for op in chain:
            bound = _arguments(op['function'], op['args'], op['kwargs'])
            require(bound.get('self') is tensors[-1], 'ACTUAL_POSITION_SLICE_PARENT_CHANGED')
            require(all(type(bound.get(key)) is int for key in ('dim', 'start', 'end', 'step'))
                    and (bound['dim'], bound['start'], bound['end'], bound['step']) == (0, 0, len(self.expected_ids), 1),
                    'ACTUAL_METADATA_POSITION_PREFIX_SLICE_CHANGED')
            event, output = op['event'], op.get('output')
            require(event['original_calls'] == 1 and event['returned'] and isinstance(output, torch.Tensor)
                    and event['inputs'] == [metadata(tensors[-1])] and event['input_epochs'] == [epoch]
                    and len(event['outputs']) == 1 and event['output_epochs'] == [epoch],
                    'ACTUAL_METADATA_POSITION_SLICE_LIFETIME_CHANGED')
            raw_output, after = event['outputs'][0], metadata(output)
            # The dispatcher sees the returned view before PyTorch attaches
            # the base's version counter. Retain that raw snapshot, and bind
            # the completed same-object view to the unchanged source version.
            require(all(raw_output[name] == after[name] for name in after if name != 'version'),
                    'ACTUAL_METADATA_POSITION_SLICE_LIFETIME_CHANGED')
            require(all(after[name] == before[name] for name in ('device', 'dtype', 'storage_address',
                        'storage_identity', 'storage_bytes', 'offset', 'version', 'inference_version_unavailable'))
                    and after['shape'] == [len(self.expected_ids)] and after['stride'] == [1],
                    'ACTUAL_METADATA_POSITION_SLICE_EXTENT_CHANGED')
            require(self.epochs.get(_key(output), 0) == epoch, 'ACTUAL_METADATA_POSITION_SLICE_WRITE_EPOCH_CHANGED')
            tensors.append(output)
            links.append(dict(event=event['sequence'], schema=event['schema'], input=event['inputs'][0],
                dispatch_output=raw_output, completed_output=after, epoch=epoch,
                actual_arguments=[0, 0, len(self.expected_ids), 1]))
        middle, positions = tensors[1:]
        # The reviewed .long() may be an identity with no dispatcher event. If
        # an actual to.dtype event occurs, it must be this same no-copy int64
        # object; an equal-valued conversion cannot create row authority.
        conversions = []
        for candidate in row['operations']:
            if candidate['function'] is not torch.ops.aten.to.dtype:
                continue
            bound = _arguments(candidate['function'], candidate['args'], candidate['kwargs'])
            if bound.get('self') is not middle:
                continue
            require(bound.get('dtype') == torch.int64 and bound.get('copy') is False
                    and bound.get('non_blocking') is False and bound.get('memory_format') is None
                    and candidate.get('output') is middle and candidate['event']['returned']
                    and candidate['event']['original_calls'] == 1
                    and chain[0]['event']['sequence'] < candidate['event']['sequence'] < chain[1]['event']['sequence'],
                    'ACTUAL_METADATA_POSITION_CAST_CHANGED')
            conversions.append(candidate['event']['sequence'])
        op = chain[-1]
        row.update(output=result, positions=positions, positions_metadata=metadata(positions), positions_epoch=epoch,
            position_chain_tensors=tuple(tensors), position_chain_metadata=[metadata(tensor) for tensor in tensors],
            position_lineage=dict(source=before, source_epoch=epoch, slice_event=op['event']['sequence'],
                slice_schema=op['event']['schema'], actual_slice_arguments=[0, 0, len(self.expected_ids), 1],
                slice_dispatch_output=op['event']['outputs'][0], slice_output=metadata(positions), identity_cast_events=conversions,
                slice_chain=links, source_to_return_actual_slice_count=2,
                actual_return_object_verified=True, values_used_as_authority=False))

    def _rotary_return(self, row, result):
        require(row['builder'] is not None and row['input'].get('use_cache') is True
                and isinstance(result, tuple) and len(result) == 2, 'ACTUAL_CACHED_ROTARY_PRODUCER_REQUIRED')
        positions = row['input']['positions']
        module = row['module']
        operations = row['operations']
        indices = [op for op in operations if op['event']['operator'] == 'aten.index.Tensor']
        copies = [op for op in operations if op['event']['operator'] == 'aten.copy_.default']
        require(len(indices) == len(copies) == 2 and all(op['event']['returned'] for op in operations),
                'COMPLETE_ACTUAL_ROTARY_INDEX_AND_BUFFER_WRITES_REQUIRED')
        require(all(op['event']['operator'] in {'aten.index.Tensor', 'aten.unsqueeze.default',
                    'aten.slice.Tensor', 'aten.copy_.default'} for op in operations),
                'UNREVIEWED_ROTARY_PRODUCER_OPERATOR')
        links = []
        for ordinal, (cache_name, buffer_name) in enumerate((('_cos_cache', '_cos_mla'), ('_sin_cache', '_sin_mla'))):
            index, copy = indices[ordinal], copies[ordinal]
            cache, buffer = getattr(module, cache_name), getattr(module, buffer_name)
            require(index['args'][0] is cache and len(index['args'][1]) == 1 and index['args'][1][0] is positions
                    and _same_bytes_view(copy['args'][1], index['output'])
                    and _same_extent(result[ordinal], copy['args'][0])
                    and _key(result[ordinal]) == _key(buffer) and result[ordinal].storage_offset() == 0,
                    'ROTARY_INDEX_OUTPUT_BUFFER_OBJECT_CHAIN_CHANGED')
            links.append(dict(cache=metadata(cache), buffer=metadata(buffer), index_event=index['event']['sequence'],
                copy_event=copy['event']['sequence'], output=metadata(result[ordinal]),
                epoch=self.epochs.get(_key(result[ordinal]), 0)))
        row.update(output=result, positions=positions, positions_metadata=metadata(positions),
                   positions_epoch=self.epochs.get(_key(positions), 0), links=links)

    def close(self, *, scope=None, actual_positions=None):
        require(self.active and not self.pending and threading.get_ident() == self.thread
                and sys.getprofile() is (None if self.profile_released else self.callback),
                'EARLY_METADATA_OBSERVER_OWNER_OR_INFLIGHT_CHANGED')
        self.closing = True
        if self.mode is not None:
            require(torch.is_inference_mode_enabled() == self.mode_scope_inference,
                    'EARLY_METADATA_INFERENCE_SCOPE_CHANGED_RETAIN_OWNER')
        if not self.profile_released:
            sys.setprofile(None)
            self.profile_released = sys.getprofile() is None
            require(self.profile_released, 'EARLY_METADATA_PROFILE_NOT_RELEASED')
        if self.mode is not None:
            removed, error = exit_owned_mode(self.mode, 'EARLY_METADATA_OBSERVER_OWNER_OR_INFLIGHT_CHANGED')
            if removed:
                self.mode = None
            if error is not None:
                raise error
        require(_get_current_dispatch_mode() is None, 'EARLY_METADATA_MODE_STACK_NOT_RESTORED')
        self.active = False
        if scope is not None:
            require(set(scope['request_ids']) == set(self.expected_ids) and isinstance(actual_positions, torch.Tensor),
                    'ACTUAL_MODEL_ROWS_DIFFER_FROM_PENDING_METADATA_REQUESTS')
            self.scope, self.actual_positions = snapshot(scope), actual_positions
            self.actual_position_metadata = metadata(actual_positions)
        return self.report()

    def consumer(self, decode, graph):
        """Return a detached proof only for a live actual producer/consumer join."""
        require(not self.active and not self.errors and not self.overflow and self.scope is not None
                and metadata(self.actual_positions) == self.actual_position_metadata,
                'ACTUAL_COMPLETE_METADATA_INTERVAL_REQUIRED')
        require(graph.epochs.get(graph.storage_key(metadata(self.actual_positions)), 0) == 0,
                'ACTUAL_MODEL_POSITIONS_WRITTEN_AFTER_SCHEDULER_BINDING')
        builders = [row for row in self.rows if row['role'] == 'builder' and row['status'] == 'OBSERVED'
                    and row['output'] is decode and decode.input_positions is row['positions']
                    and metadata(row['positions']) == row['positions_metadata']
                    and row['input']['common_attn_metadata'].positions is row['source_positions']
                    and metadata(row['source_positions']) == row['source_positions_metadata']
                    and all(metadata(tensor) == expected for tensor, expected in
                            zip(row['position_chain_tensors'], row['position_chain_metadata']))
                    and self.epochs.get(_key(row['source_positions']), 0) == row['source_positions_epoch']
                    and graph.epochs.get(graph.storage_key(metadata(row['source_positions'])), 0) == 0
                    and self.epochs.get(_key(row['positions']), 0) == row['positions_epoch']
                    and _same_extent(row['positions'], self.actual_positions)]
        require(len(builders) == 1, 'ACTUAL_DECODE_METADATA_RETURN_AND_POSITION_OWNER_REQUIRED')
        for row in reversed(self.rows):
            if row['role'] != 'rotary' or row['status'] != 'OBSERVED' or row.get('builder') is not builders[0]:
                continue
            if (metadata(row['positions']) != row['positions_metadata']
                    or self.epochs.get(_key(row['positions']), 0) != row['positions_epoch']
                    or graph.epochs.get(graph.storage_key(metadata(row['positions'])), 0) != 0
                    or not _same_extent(row['positions'], decode.input_positions)):
                continue
            valid = all(_same_extent(target, observed) and metadata(observed) == link['output']
                and self.epochs.get(_key(observed), 0) == link['epoch']
                and graph.epochs.get(graph.storage_key(metadata(target)), 0) == 0
                for target, observed, link in zip((decode.cos, decode.sin), row['output'], row['links']))
            if valid:
                return snapshot(dict(verified=True, actual_rows=self.scope, builder_sequence=builders[0]['sequence'],
                    producer_sequence=row['sequence'], source_sha256=row['source_sha256'], links=row['links'],
                    position_lineage=builders[0]['position_lineage'],
                    positions=metadata(row['positions']), complete_index_values_inferred=False,
                    actual_index_operand_identity_verified=True, native_qualified=False))
        raise RuntimeError('NO_LIVE_ACTUAL_ROTARY_PRODUCER_FOR_CURRENT_METADATA')

    def watched_storages(self):
        return snapshot([metadata(tensor) for row in self.rows if row['status'] == 'OBSERVED'
                         for tensor in tensor_values((row.get('output'), row.get('positions'), row.get('position_chain_tensors')))])

    def release(self):
        require(not self.active and not self.pending, 'METADATA_PRODUCER_STILL_INFLIGHT')
        self.rows = self.report()['boundaries']
        self.scope = self.actual_positions = None
        self.released = True

    def report(self):
        return snapshot(dict(schema='demoncat.quant.early_mla_metadata.v1', active=self.active,
            owned_profile_callbacks=int(not self.profile_released), owned_dispatch_modes=int(self.mode is not None),
            cleanup_pending=self.active and self.closing,
            sources=self.sources, simulation=self.simulation, errors=self.errors, events=self.events,
            actual_calls=self.total, retained_calls=len(self.events), missing_calls=self.overflow,
            live_producer_records=0 if self.released else len(self.rows),
            boundaries=[{key: value for key, value in row.items() if key not in
                ('input', 'operations', 'output', 'positions', 'source_positions', 'position_chain_tensors', 'module', 'builder')} for row in self.rows],
            complete=not self.active and not self.pending and not self.errors and not self.overflow
                and all(event['original_calls'] == 1 and event['returned'] for event in self.events),
            model_or_operator_reexecution=False, native_qualified=False))
