"""Bounded actual row provenance; opaque operations never inherit row labels.

This is an observation graph, not Native qualification. Inference tensors have
no PyTorch version counter; their report says so explicitly and all dispatcher
writes invalidate storage aliases through an independent observed-write epoch.
"""
import hashlib
import json
import sys
import threading
import time
import weakref

import torch
from torch.utils._python_dispatch import (
    TorchDispatchMode, _get_current_dispatch_mode, _get_current_dispatch_mode_stack,
)

from .serving_context import require


def tensor_values(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in tensor_values(item)]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in tensor_values(item)]
    return []


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def snapshot(value):
    """Detach every nested JSON value at the observation/public API boundary."""
    return json.loads(json.dumps(value, allow_nan=False))


def exit_owned_mode(mode, refusal):
    """Exit only the current owned mode and report an exit that popped then raised.

    Callers retain the handle until the real stack confirms its removal. An
    exception before removal is retryable; a foreign mode is never popped.
    """
    require(_get_current_dispatch_mode() is mode, refusal)
    error = None
    try:
        mode.__exit__(None, None, None)
    except BaseException as caught:
        error = caught
    removed = not any(value is mode for value in _get_current_dispatch_mode_stack())
    require(removed or error is not None, refusal + ':EXIT_DID_NOT_REMOVE_MODE')
    return removed, error


def same_mode_stack(expected):
    actual = _get_current_dispatch_mode_stack()
    return len(actual) == len(expected) and all(a is b for a, b in zip(actual, expected))


def metadata(tensor):
    storage = tensor.untyped_storage()
    try:
        version = tensor._version
    except RuntimeError:
        version = None
    return dict(object_id=id(tensor), storage_address=storage.data_ptr(), storage_bytes=storage.nbytes(),
        storage_identity=storage._cdata, offset=tensor.storage_offset(), shape=list(tensor.shape),
        stride=list(tensor.stride()), dtype=str(tensor.dtype), device=str(tensor.device), version=version,
        inference_version_unavailable=version is None)


class RequestRowProvenance(TorchDispatchMode):
    _PRESERVE = {'clone.default', 'detach.default', 'alias.default', '_to_copy.default'}
    _RESHAPE = {'view.default', '_unsafe_view.default', 'reshape.default'}
    _POINTWISE = {'add.Tensor', 'sub.Tensor', 'mul.Tensor', 'div.Tensor', 'neg.default', 'abs.default',
        'exp.default', 'log.default', 'rsqrt.default', 'sqrt.default', 'sigmoid.default', 'silu.default',
        'gelu.default', 'tanh.default', 'relu.default', 'reciprocal.default', 'pow.Tensor_Scalar'}
    _VIEWS = {'transpose.int', 'permute.default', 'slice.Tensor', 'select.int',
              'split.Tensor', 'split_with_sizes.default', 'cat.default'}

    def __init__(self, *, max_calls=16384):
        super().__init__()
        require(type(max_calls) is int and 0 < max_calls <= 65536, 'BOUNDED_ROW_OBSERVER_REQUIRED')
        self.max_calls = max_calls
        self.thread = threading.get_ident()
        self.allowed = {}
        for name in self._PRESERVE | self._RESHAPE | self._POINTWISE | self._VIEWS:
            namespace, overload = name.split('.')
            try:
                operator = getattr(getattr(torch.ops.aten, namespace), overload)
                self.allowed[operator] = name
            except AttributeError:
                pass
        self.active = False
        self.forward_id = None
        self.events, self.nodes, self.proofs, self.epochs = [], [], {}, {}
        self.parameters = {}
        self.total = self.overflow = 0
        self.errors = []
        self.boundaries = None
        self.watched_storages = []
        self.call_stack = []
        self.ending = False
        self.mode_released = True
        self.base_modes = ()

    def begin(self, forward_id):
        require(not self.active and threading.get_ident() == self.thread, 'ONE_CURRENT_ROW_OBSERVER_REQUIRED')
        self.forward_id = forward_id
        self.events, self.nodes, self.proofs, self.epochs = [], [], {}, {}
        self.parameters = {}
        self.total = self.overflow = 0
        self.errors = []
        self.watched_storages = []
        self.call_stack = []
        self.ending, self.mode_released = False, False
        self.base_modes = tuple(_get_current_dispatch_mode_stack())
        self.active = True
        try:
            self.__enter__()
        except BaseException:
            self.mode_released = not any(value is self for value in _get_current_dispatch_mode_stack())
            raise

    @staticmethod
    def storage_key(meta):
        return (meta['device'], meta['storage_address'], meta['storage_bytes'], meta['storage_identity'])

    def _lookup(self, tensor):
        record = self.proofs.get(id(tensor))
        if record is None or record['tensor']() is not tensor:
            return None
        actual = metadata(tensor)
        if (digest(record['rows']) != record['node']['rows_sha256'] or digest(record['node']) != record['node_sha256']
                or actual != record['metadata'] or self.epochs.get(self.storage_key(actual), 0) != record['write_epoch']):
            self.proofs.pop(id(tensor), None)
            return None
        return record

    def _bind(self, tensor, *, rows, parents, operator, evidence, row_axis=0):
        rows, evidence = snapshot(rows), snapshot(evidence)
        meta = metadata(tensor)
        require(tensor.ndim >= 2 and type(row_axis) is int and 0 <= row_axis < tensor.ndim
                and tensor.shape[row_axis] == len(rows['request_ids']), 'ACTUAL_ROW_BOUNDARY_GEOMETRY_CHANGED')
        node = dict(node_id=len(self.nodes) + 1, operator=operator, metadata=meta, parents=list(parents),
            rows_sha256=digest(rows), row_axis=row_axis, evidence=evidence,
            write_epoch=self.epochs.get(self.storage_key(meta), 0))
        self.nodes.append(node)
        self.proofs[id(tensor)] = dict(tensor=weakref.ref(tensor), metadata=meta, rows=rows, node=node,
            node_sha256=digest(node), write_epoch=node['write_epoch'])
        return snapshot(node)

    def seed(self, tensor, proof):
        require(self.active and not self.ending and not self.errors and not self.overflow and proof.get('verified') is True,
                'ACTUAL_VERIFIED_ROW_SEED_REQUIRED')
        rows = proof['actual_rows']
        ids = rows['request_ids']
        require(isinstance(ids, list) and len(ids) == len(set(ids)) == tensor.shape[0]
                and all(isinstance(key, str) and key for key in ids), 'ACTUAL_ORDERED_REQUEST_ROWS_REQUIRED')
        self._bind(tensor, rows=rows, parents=(), operator='ACTUAL_SCHEDULER_EMBEDDING_RMSNORM', evidence=proof)

    def register_broadcast_parameter(self, module, name, tensor, *, kind):
        require(self.active and not self.ending and kind == '_parameters' and isinstance(tensor, torch.nn.Parameter)
                and not tensor.requires_grad and getattr(module, kind).get(name) is tensor,
                'ACTUAL_FROZEN_MODEL_PARAMETER_OWNER_REQUIRED')
        meta = metadata(tensor)
        self.parameters[id(tensor)] = dict(tensor=weakref.ref(tensor), owner=weakref.ref(module), name=name, kind=kind,
            metadata=meta, write_epoch=self.epochs.get(self.storage_key(meta), 0))

    def _broadcast_parameter(self, tensor, output_shape):
        record = self.parameters.get(id(tensor))
        if record is None or record['tensor']() is not tensor or record['owner']() is None:
            return False
        meta = metadata(tensor)
        return (getattr(record['owner'](), record['kind']).get(record['name']) is tensor
            and meta == record['metadata'] and self.epochs.get(self.storage_key(meta), 0) == record['write_epoch']
            and (tensor.ndim < len(output_shape) or tensor.ndim == len(output_shape) and tensor.shape[0] == 1))

    def _query(self, tensor, *, visible):
        healthy = self.boundaries is None or not self.boundaries.errors
        record = self._lookup(tensor) if self.active and not self.ending and healthy and visible and not self.errors and not self.overflow else None
        if record is None or record['node']['row_axis'] != 0:
            return dict(verified=False, reason='NO_COMPLETE_ACTUAL_ROW_PRESERVING_OPERATOR_CHAIN',
                shape_is_not_request_row_proof=True, tensor=metadata(tensor))
        return snapshot(dict(verified=True, source='ACTUAL_DISPATCHER_ROW_PRESERVING_CHAIN', forward_id=self.forward_id,
            node_id=record['node']['node_id'], actual_rows=record['rows'], tensor=record['metadata'],
            observed_write_epoch=record['write_epoch'], native_qualified=False,
            observation_prefix=dict(observer_id=id(self), forward_id=self.forward_id,
                observed_at_monotonic_ns=time.monotonic_ns(), actual_calls=self.total, retained_calls=len(self.events),
                missing_calls=self.overflow, errors=len(self.errors),
                source_errors=0 if self.boundaries is None else len(self.boundaries.errors),
                parent_source_sequences=[row['sequence'] for row in getattr(self.boundaries, 'stack', ())])))

    def query(self, tensor):
        return self._query(tensor, visible=self._visible_context())

    def _native_query_health(self):
        require(self.active and not self.ending and not self.mode_released
                and threading.get_ident() == self.thread and not self.errors and not self.overflow
                and self.total == len(self.events) <= self.max_calls, 'NATIVE_ROW_OBSERVER_NOT_CURRENT_AND_HEALTHY')
        source = self.boundaries
        if source is not None:
            require(source.graph is self and source.active and not source.ending and not source.errors
                    and source.thread == self.thread and not source.profile_released
                    and sys.getprofile() is source.callback, 'NATIVE_ROW_SOURCE_OWNER_OR_HEALTH_CHANGED')
            source._verify_bindings()

    def capture_native_input_context(self):
        """Hold actual outer frames; this is not a reusable row authorization."""
        self._native_query_health()
        require(self._visible_context(), 'NATIVE_ROW_PUBLIC_CONTEXT_NOT_VISIBLE')
        modes = tuple(_get_current_dispatch_mode_stack())
        source = self.boundaries
        frames = tuple(getattr(source, 'operator_stack', ()))
        require(len(frames) == len(self.call_stack)
                and all(frame['func'] is func for frame, func in zip(frames, self.call_stack)),
                'NATIVE_ROW_PUBLIC_SOURCE_FRAME_CONTEXT_CHANGED')
        require(modes and (modes[-1] is self or source is not None
                and any(modes[-1] is mode for mode in source.owned_modes)), 'NATIVE_ROW_DISPATCH_OWNER_CHANGED')
        return dict(observer=self, forward_id=self.forward_id, operators=tuple(self.call_stack), modes=modes,
            source=source, source_operators=frames,
            source_parents=tuple(getattr(source, 'stack', ())))

    def query_native_input(self, tensor, *, public_proof, context, operator, schema):
        """Recheck the proof inside one actual, read-only Native input dispatch.

        The caller resolves the exact supported OpOverload, never a name alias.
        Its pending dispatcher frame may be the sole addition to the captured
        public stack. No errors or opaque frames are hidden to obtain a proof.
        """
        self._native_query_health()
        source = self.boundaries
        require(context['observer'] is self and context['forward_id'] == self.forward_id
                and context['source'] is source, 'NATIVE_ROW_OBSERVER_FORWARD_OR_SOURCE_CHANGED')
        outer = context['operators']
        require(len(self.call_stack) in (len(outer), len(outer) + 1)
                and all(actual is old for actual, old in zip(self.call_stack, outer)),
                'NATIVE_ROW_OUTER_DISPATCH_CONTEXT_CHANGED')
        frames = tuple(getattr(source, 'operator_stack', ()))
        parents = tuple(getattr(source, 'stack', ()))
        require(len(frames) == len(self.call_stack) and len(parents) == len(context['source_parents'])
                and all(actual is old for actual, old in zip(frames, context['source_operators']))
                and all(actual is old and actual['status'] == 'INFLIGHT'
                        for actual, old in zip(parents, context['source_parents']))
                and all(frame['func'] is func for frame, func in zip(frames, self.call_stack)),
                'NATIVE_ROW_SOURCE_FRAME_CONTEXT_CHANGED')
        opaque = [func for func in outer if not str(func).startswith(('aten.', 'prims.'))]
        require(not opaque or source is not None and source.permits_public_rows(opaque),
                'NATIVE_ROW_OUTER_OPAQUE_CONTEXT_NOT_VISIBLE')
        added = len(self.call_stack) != len(outer)
        expected_modes = context['modes'][:-1] if added else context['modes']
        require(same_mode_stack(expected_modes), 'NATIVE_ROW_CURRENT_DISPATCH_OWNER_CHANGED')
        dispatch = None
        if added:
            require(isinstance(operator, torch._ops.OpOverload) and self.call_stack[-1] is operator
                    and str(operator._schema) == schema, 'NATIVE_ROW_UNREVIEWED_ADDITIONAL_DISPATCH')
            actual_schema = operator._schema
            formals = actual_schema.arguments
            frame, event = frames[-1], frames[-1]['event']
            require(not actual_schema.is_mutable and len(formals) > 0 and str(formals[0].type) == 'Tensor'
                    and all(arg.alias_info is None for arg in formals)
                    and len(actual_schema.returns) == 2
                    and all(str(arg.type) == 'Tensor' and arg.alias_info is None for arg in actual_schema.returns),
                    'NATIVE_ROW_INPUT_OPERATOR_SCHEMA_CHANGED')
            names = [arg.name for arg in formals]
            args, kwargs = frame['args'], frame['kwargs']
            require(len(args) <= len(formals) and not any(arg.kwarg_only for arg in formals[:len(args)])
                    and not set(kwargs).difference(names) and not set(names[:len(args)]).intersection(kwargs),
                    'NATIVE_ROW_INPUT_OPERATOR_ARGUMENTS_CHANGED')
            bound = {arg.name: arg.default_value for arg in formals if arg.has_default_value()}
            bound.update(zip(names, args))
            bound.update(kwargs)
            require(set(bound) == set(names) and bound[names[0]] is tensor
                    and all(arg.has_default_value() and not isinstance(bound[arg.name], torch.Tensor)
                        and bound[arg.name] == arg.default_value for arg in formals[1:]),
                    'NATIVE_ROW_INPUT_OPERATOR_ARGUMENTS_CHANGED')
            require(event['status'] == 'PENDING' and event['original_calls'] == 1 and not event['mutable']
                    and 0 < event['sequence'] <= len(self.events) and self.events[event['sequence']-1] is event
                    and event['sequence'] > public_proof['observation_prefix']['actual_calls']
                    and event['schema'] == schema and event['inputs'] == [metadata(tensor)],
                    'NATIVE_ROW_CURRENT_DISPATCH_FRAME_UNPROVEN')
            dispatch = dict(sequence=event['sequence'], operator=str(operator), schema_sha256=digest(schema),
                            actual_input=metadata(tensor))
        fresh = self._query(tensor, visible=True)  # Visibility was checked above, including the sole Native frame.
        require(fresh.get('verified') is True and public_proof.get('verified') is True
                and all(fresh[name] == public_proof[name] for name in
                        ('forward_id', 'node_id', 'actual_rows', 'tensor', 'observed_write_epoch'))
                and fresh['observation_prefix']['observer_id'] == public_proof['observation_prefix']['observer_id']
                and fresh['observation_prefix']['forward_id'] == public_proof['observation_prefix']['forward_id'] == self.forward_id
                and fresh['observation_prefix']['parent_source_sequences'] == public_proof['observation_prefix']['parent_source_sequences']
                and public_proof['observation_prefix']['actual_calls'] == public_proof['observation_prefix']['retained_calls']
                and not any(public_proof['observation_prefix'][key] for key in ('missing_calls', 'errors', 'source_errors'))
                and fresh['observation_prefix']['observed_at_monotonic_ns'] >= public_proof['observation_prefix']['observed_at_monotonic_ns']
                and fresh['observation_prefix']['actual_calls'] >= public_proof['observation_prefix']['actual_calls'],
                'NATIVE_ROW_PROOF_CHANGED_AFTER_PUBLIC_QUERY')
        fresh['native_input_dispatch'] = dispatch
        return fresh

    def projection(self, tensor, output, *, evidence):
        """Only called after exact Native-DQ and public owned-QMM chain checks."""
        record = self._lookup(tensor) if self.active and not self.ending and self._visible_context() and not self.errors and not self.overflow else None
        if record is None or record['node']['row_axis'] != 0:
            return self.query(output)
        require(evidence.get('complete_native_dynamic_formals') is True and evidence.get('owned_public_qmm_chain') is True
                and tensor.ndim == output.ndim == 2 and output.shape[0] == tensor.shape[0]
                and output.device == tensor.device and output.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr(),
                'EXACT_NATIVE_DYNAMIC_QUANT_AND_OWNED_PUBLIC_QMM_REQUIRED')
        self._bind(output, rows=record['rows'], parents=(record['node']['node_id'],),
            operator='ACTUAL_NATIVE_DYNAMIC_QUANT_TO_OWNED_PUBLIC_QMM', evidence=evidence)
        return self.query(output)

    def _visible_context(self):
        opaque = [func for func in self.call_stack if not str(func).startswith(('aten.', 'prims.'))]
        return not opaque or self.boundaries is not None and self.boundaries.permits_public_rows(opaque)

    def _invalidate(self, values):
        written = [metadata(t) for t in {id(t): t for t in values}.values()]
        keys = {self.storage_key(meta) for meta in written}
        # Independent Tensor/Storage wrappers can alias the same device bytes
        # without sharing either _version or Storage._cdata. Invalidate every
        # overlapping live proof, not just aliases with one Storage object.
        for record in list(self.proofs.values()) + list(self.parameters.values()):
            meta = record['metadata']
            if any(meta['device'] == change['device'] and
                   meta['storage_address'] < change['storage_address'] + change['storage_bytes'] and
                   change['storage_address'] < meta['storage_address'] + meta['storage_bytes'] for change in written):
                keys.add(self.storage_key(meta))
        for meta in self.watched_storages:
            if any(meta['device'] == change['device'] and
                   meta['storage_address'] < change['storage_address'] + change['storage_bytes'] and
                   change['storage_address'] < meta['storage_address'] + meta['storage_bytes'] for change in written):
                keys.add(self.storage_key(meta))
        for key in keys:
            self.epochs[key] = self.epochs.get(key, 0) + 1
        return [dict(device=key[0], address=key[1], bytes=key[2], storage_identity=key[3],
                     write_epoch=self.epochs[key]) for key in sorted(keys)]

    def _parents(self, name, args, kwargs, outputs):
        inputs = tensor_values((args, kwargs))
        if not inputs or not outputs or any(t.ndim < 2 for t in outputs):
            return None
        first = inputs[0]
        proof = self._lookup(first)
        if proof is None or proof['node']['row_axis'] != 0 or first.ndim < 2:
            return None
        count = len(proof['rows']['request_ids'])
        if any(t.shape[0] != count for t in outputs):
            return None
        records = [proof]
        if name in self._PRESERVE:
            return records if all(tuple(t.shape) == tuple(first.shape) for t in outputs) else None
        if name in self._RESHAPE:
            return records if first.is_contiguous() and all(t.is_contiguous() and t.numel() == first.numel() for t in outputs) else None
        if name in self._POINTWISE:
            shape = tuple(outputs[0].shape)
            if tuple(first.shape) != shape:
                return None
            for tensor in inputs[1:]:
                if tensor.ndim == 0 or self._broadcast_parameter(tensor, shape):
                    continue  # No request axis, or a current owned parameter broadcast.
                other = self._lookup(tensor)
                if (tuple(tensor.shape) != shape or other is None or other['node']['row_axis'] != 0
                        or other['rows'] != proof['rows']):
                    return None
                records.append(other)
            return records
        if name == 'transpose.int':
            dims = (args[1] % first.ndim, args[2] % first.ndim)
            return records if 0 not in dims or dims[0] == dims[1] else None
        if name == 'permute.default':
            return records if args[1][0] % first.ndim == 0 else None
        if name in ('slice.Tensor', 'select.int', 'split.Tensor', 'split_with_sizes.default'):
            position = 2 if name.startswith('split') else 1
            dim = (args[position] if len(args) > position else kwargs.get('dim', 0)) % first.ndim
            if dim != 0:
                return records
            if name == 'slice.Tensor':
                start = args[2] if len(args) > 2 else kwargs.get('start', 0)
                end = args[3] if len(args) > 3 else kwargs.get('end', count)
                step = args[4] if len(args) > 4 else kwargs.get('step', 1)
                return records if start == 0 and (end is None or end >= count) and step == 1 else None
            return None
        if name == 'cat.default':
            dim = (args[1] if len(args) > 1 else kwargs.get('dim', 0)) % first.ndim
            if dim == 0:
                return None
            records = [self._lookup(t) for t in inputs]
            return records if all(record and record['node']['row_axis'] == 0
                                  and record['rows'] == proof['rows'] for record in records) else None
        return None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        self.total += 1
        if self.total > self.max_calls:
            self.overflow += 1
            return func(*args, **kwargs)
        event = dict(sequence=self.total, operator=str(func), original_calls=0, status='PENDING',
                     row_preserving=False, reason='UNREVIEWED_OPERATOR_OR_INPUT_ROW_PROVENANCE')
        self.events.append(event)
        schema = getattr(func, '_schema', None)
        boundary = None
        try:
            require(threading.get_ident() == self.thread, 'FOREIGN_ROW_OBSERVER_THREAD')
            event.update(schema=str(schema), schema_sha256=digest(str(schema)),
                inputs=[metadata(t) for t in tensor_values((args, kwargs))])
            mutable = schema is None or schema.is_mutable
            event['mutable'] = mutable
            if self.boundaries is not None:
                boundary = self.boundaries.before_operator(func, args, kwargs, event)
            if mutable:
                written = (boundary['written'] if boundary is not None and boundary.get('written') is not None
                           else tensor_values((args, kwargs)))
                event['invalidated_storages'] = self._invalidate(written)
                event['reason'] = 'UNKNOWN_OR_MUTATING_OPERATOR_INVALIDATES_STORAGE_ALIASES'
        except BaseException as error:
            self.errors.append(type(error).__name__ + ':' + str(error))
            mutable = True
        try:
            event['original_calls'] = 1
            self.call_stack.append(func)
            try:
                output = func(*args, **kwargs)
            finally:
                self.call_stack.pop()
        except BaseException as error:
            event.update(status='ORIGINAL_FAILED', exception=repr(error))
            if self.boundaries is not None:
                self.boundaries.failed_operator(boundary, error)
            raise
        try:
            outputs = tensor_values(output)
            event['outputs'] = [metadata(t) for t in outputs]
            name = self.allowed.get(func)
            parents = self._parents(name, args, kwargs, outputs) if name and not mutable else None
            accepted = (self.boundaries.after_operator(boundary, output)
                        if self.boundaries is not None else set())
            if parents is not None and not self.errors:
                event['broadcast_parameters'] = [dict(module_object_identity=id(self.parameters[id(t)]['owner']()),
                    parameter_name=self.parameters[id(t)]['name'], metadata=self.parameters[id(t)]['metadata'],
                    write_epoch=self.parameters[id(t)]['write_epoch']) for t in tensor_values((args, kwargs))
                    if self._broadcast_parameter(t, tuple(outputs[0].shape))]
                for tensor in outputs:
                    self._bind(tensor, rows=parents[0]['rows'], parents=[p['node']['node_id'] for p in parents],
                        operator=str(func), evidence=dict(dispatch_sequence=event['sequence'], schema_sha256=event['schema_sha256']))
                event.update(row_preserving=True, reason=None)
            else:
                for tensor in outputs:
                    if id(tensor) not in accepted:
                        self.proofs.pop(id(tensor), None)
                if accepted:
                    event.update(row_preserving=True, reason=None)
            event['status'] = 'OBSERVED'
        except BaseException as error:
            self.errors.append(type(error).__name__ + ':' + str(error))
            event['status'] = 'ROW_OBSERVER_ERROR'
        return output

    def end(self):
        require(self.active and threading.get_ident() == self.thread, 'CURRENT_ROW_OBSERVER_REQUIRED')
        require(not self.call_stack, 'ROW_OBSERVER_CALLS_STILL_INFLIGHT')
        self.ending = True
        # Torch temporarily removes this outer mode during the opaque call,
        # then restores it above any nested modes left by a failed cleanup.
        # Remove only our outer mode first so the retained nested owners can
        # subsequently be inspected and retried without touching foreign ones.
        if not self.mode_released:
            removed, error = exit_owned_mode(self, 'ROW_DISPATCH_MODE_OWNER_CHANGED')
            self.mode_released = removed
            if error is not None:
                raise error
        if self.boundaries is not None and self.boundaries.active:
            self.boundaries.end()
        require(same_mode_stack(self.base_modes), 'ROW_DISPATCH_MODE_STACK_NOT_RESTORED')
        self.active = False
        return self.report()

    def report(self):
        boundary_report = self.boundaries.report() if self.boundaries is not None else None
        return snapshot(dict(schema='demoncat.quant.actual_row_provenance.v1', forward_id=self.forward_id,
            scope='ACTUAL_DECODE_THREAD_INCLUDING_READONLY_OBSERVER_READBACKS',
            actual_calls=self.total, retained_calls=len(self.events), missing_calls=self.overflow, errors=self.errors,
            complete=not self.active and not self.errors and not self.overflow and (boundary_report is None or boundary_report['complete'])
                and all(e['original_calls'] == 1 and e['status'] == 'OBSERVED' for e in self.events),
            events=self.events, nodes=self.nodes, active=self.active, native_qualified=False,
            owned_dispatch_modes=int(not self.mode_released), cleanup_pending=self.active and self.ending,
            unknown_edges=[e['sequence'] for e in self.events if not e['row_preserving']],
            source_boundaries=boundary_report))
