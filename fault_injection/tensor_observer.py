"""Read-only natural tensor census, independent of fault admission and counters.

This observes actual eager dispatch and module returns. Structural names and
source clues are inventory hints, never authorization to inject. No tensor is
copied, converted, written, or reconstructed by this observer.
"""
from collections import Counter
from contextlib import nullcontext
from contextvars import ContextVar
import copy
import hashlib
import inspect
from pathlib import Path
import threading
import weakref


_METADATA_DIAGNOSTICS = ContextVar('dcat_tensor_observer_metadata_diagnostics', default=None)


def tensors(value):
    import torch
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)


def metadata(tensor):
    # Tensor.stride/storage_offset are public APIs too. The observer's own
    # metadata queries must not become model construct nodes. A context-local
    # guard also covers the deployment observer callbacks using this helper;
    # the actual model operator always executes outside this tiny window.
    diagnostics = _METADATA_DIAGNOSTICS.get()
    with diagnostics() if diagnostics is not None else nullcontext():
        row = dict(object_id=id(tensor), storage_id=int(tensor.untyped_storage()._cdata),
                   shape=list(tensor.shape), stride=list(tensor.stride()),
                   offset=tensor.storage_offset(), dtype=str(tensor.dtype), device=str(tensor.device))
        if tensor.device.type == 'npu':
            import torch_npu
            row['actual_format'] = int(torch_npu.get_npu_format(tensor))
        return row


def callable_identity(module):
    function = getattr(module.forward, '__func__', module.forward)
    row = dict(module_type=type(module).__module__ + '.' + type(module).__qualname__,
               module_object_id=id(module), callable_object_id=id(function))
    try:
        source = inspect.getsource(function)
        row.update(file=inspect.getsourcefile(function), first_line=inspect.getsourcelines(function)[1],
                   source_sha256=hashlib.sha256(source.encode()).hexdigest(), source=source)
    except (OSError, TypeError):
        row['source_unavailable'] = True
    return row


def _dispatch_parameter(value, depth=0):
    if type(value) is float and not float('-inf') < value < float('inf'):
        return {'nonfinite': repr(value)}
    if value is None or type(value) in (bool, int, float):
        return value
    if type(value) is str:
        return {'type': 'str', 'length': len(value)}
    if type(value) in (list, tuple) and len(value) <= 32 and depth < 4:
        return [_dispatch_parameter(item, depth + 1) for item in value]
    return {'type': type(value).__name__}


class NaturalTensorObserver:
    """Bounded full eager census; does not depend on any injection implementation."""
    def __init__(self, model, *, max_events=100000, on_module_enter=None, on_operator_enter=None,
                 diagnostics_context=None, record_all_operators=False, operator_context=None):
        if type(max_events) is not int or max_events < 1:
            raise ValueError('A finite event budget is required')
        self.model, self.max_events = model, max_events
        self.events, self.forwards, self.stack, self.handles = [], [], [], []
        self.counts = Counter()
        self.module_inventory = []
        self.outputs, self.views = [], []
        self._dispatch_outputs = {}
        self.mode = None
        self.thread = None
        self.closed = False
        self.dropped = 0
        self.on_module_enter = on_module_enter
        self.on_operator_enter = on_operator_enter
        self.diagnostics_context = diagnostics_context
        self.record_all_operators = record_all_operators
        self.operator_context = operator_context
        self._metadata_token = None
        self._owner_thread = None

    def _emit(self, row):
        if len(self.events) >= self.max_events:
            self.dropped += 1
            raise RuntimeError('INDEPENDENT_TENSOR_OBSERVER_BUDGET_EXHAUSTED')
        row.update(sequence=len(self.events), forward=len(self.forwards)-1,
                   path=self.stack[-1][0] if self.stack else '<outside>')
        self.events.append(row)
        return row

    def _pre(self, path, module, role):
        def enter(actual, args, kwargs):
            if self.closed:
                return
            if actual is self.model:
                if self.stack or self.mode is not None:
                    raise RuntimeError('CONCURRENT_OR_RECURSIVE_OBSERVER_FORWARD')
                self.thread = threading.get_ident()
                self.forwards.append(dict(index=len(self.forwards), status='running',
                                          input=[metadata(x) for x in tensors((args, kwargs))]))
                self.outputs.clear()
                self.views.clear()
                self._dispatch_outputs.clear()
                self._open_mode()
            if self.mode is not None:
                if threading.get_ident() != self.thread:
                    raise RuntimeError('OBSERVER_THREAD_CHANGED')
                self.stack.append((path, actual, role))
                if self.on_module_enter is not None:
                    self.on_module_enter(path, actual, args, kwargs)
        return enter

    def _post(self, module):
        def leave(actual, args, kwargs, output):
            if not self.stack or self.stack[-1][1] is not actual:
                return
            path, _, role = self.stack[-1]
            if role and output is not None:
                row = self._emit(dict(kind='module_return', role=role,
                    input=[metadata(x) for x in tensors((args, kwargs))],
                    output=[metadata(x) for x in tensors(output)], consumers=[]))
                for tensor in tensors(output):
                    self.outputs.append((weakref.ref(tensor), row))
            self.stack.pop()
            if actual is self.model:
                self.forwards[-1].update(status='returned' if output is not None else 'failed',
                    output=[metadata(x) for x in tensors(output)])
                from ._scoped_dispatch import _close_owned_mode
                _close_owned_mode(self, 'mode', 'OBSERVER_MODE_OWNERSHIP_CHANGED')
        return leave

    def _open_mode(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        observer = self
        class Census(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return observer._dispatch(func, args, kwargs or {})
        self.mode = Census()
        self.mode.__enter__()

    def _dispatch(self, func, args, kwargs):
        schema = getattr(func, '_schema', None)
        name = getattr(schema, 'name', str(func))
        self.counts[str(func)] += 1
        inputs = tuple(tensors((args, kwargs)))
        if self.on_operator_enter is not None:
            self.on_operator_enter(func, args, kwargs, tuple(self.stack))
        input_storages = {int(t.untyped_storage()._cdata) for t in inputs}
        data_reader = name in ('aten::mul', 'aten::add', 'aten::linear', 'aten::mm', 'aten::matmul',
            'aten::bmm', 'aten::cat', 'aten::convolution', 'aten::conv3d', 'npu::npu_apply_rotary_pos_emb',
            'npu::npu_rotary_mul', 'npu::npu_quant_matmul', 'npu::npu_dynamic_quant')
        output_readers = [(ref, row) for ref, row in self.outputs if ref() is not None
            and int(ref().untyped_storage()._cdata) in input_storages] if data_reader else []
        view_copies = [(ref, base, row) for ref, base, row in self.views
            if name == 'aten::copy_' and args and ref() is args[0]]
        base_readers = [(base, row) for ref, base, row in self.views if data_reader and row['writes']
            and base() is not None and int(base().untyped_storage()._cdata) in input_storages]
        wanted = (self.record_all_operators or name in ('aten::convolution', 'aten::conv3d', 'aten::rms_norm',
            'npu::npu_rms_norm', 'npu::npu_add_rms_norm', 'npu::npu_moe_token_unpermute',
            'npu::npu_moe_token_permute', 'aten::index_select', 'aten::index')
            or 'rms_norm' in name or 'rotary' in name or 'moe' in name)
        row = self._emit(dict(kind='operator', operator=str(func), schema=str(schema),
            input=[metadata(x) for x in inputs])) if wanted else None
        if self.record_all_operators:
            producers = set()
            for item in inputs:
                match = self._dispatch_outputs.get(id(item))
                if match is not None and match[0]() is item:
                    producers.add(match[1])
            row.update(namespace=name.partition('::')[0] if '::' in name else None,
                       parameters=dict(positional=[_dispatch_parameter(value) for value in args],
                                       keyword={key: _dispatch_parameter(value) for key, value in kwargs.items()}),
                       input_producer_sequences=sorted(producers))
        with self.operator_context(row) if self.operator_context is not None and row is not None else nullcontext():
            result = func(*args, **kwargs)
        if row is not None:
            outputs = tuple(tensors(result))
            row['output'] = [metadata(x) for x in outputs]
            row['original_returned'] = True
            if name in ('aten::convolution', 'aten::conv3d'):
                values = {item.name: item.default_value for item in schema.arguments}
                values.update(zip((item.name for item in schema.arguments), args)); values.update(kwargs)
                row['conv_geometry'] = {key: values[key] for key in
                    ('stride','padding','dilation','groups','transposed','output_padding') if key in values}
                owners = [(path,module) for path,module,_ in self.stack
                    if getattr(module,'weight',None) is values.get('weight')]
                row['weight_owner_paths'] = [path for path,_ in owners]
            if self.record_all_operators:
                for item in outputs:
                    self._dispatch_outputs[id(item)] = (weakref.ref(item), row['sequence'])
            if 'rms_norm' in name:
                owners = [(path, module) for path, module, _ in self.stack
                    if any(getattr(module, 'weight', None) is item for item in inputs)]
                row['weight_owner_paths'] = [path for path, _ in owners]
                row['weight_owner_sources'] = [dict(path=path, **callable_identity(module))
                                                for path, module in owners]
            if name in ('aten::index_select', 'aten::index'):
                owners = [(path, key) for path, module, _ in self.stack
                    for key, buffer in module.named_buffers(recurse=False)
                    if inputs and buffer is inputs[0]]
                row['registered_buffer_owners'] = owners
        for ref, linked in output_readers:
            linked['consumers'].append(dict(operator=str(func), path=self.stack[-1][0], returned=True))
        for ref, base, linked in view_copies:
            linked['writes'].append(dict(operator=str(func), path=self.stack[-1][0],
                returned=True, destination_is_observed_view=True))
        for base, linked in base_readers:
            linked['consumers'].append(dict(operator=str(func), path=self.stack[-1][0], returned=True))
        if name in ('aten::as_strided', 'aten::select', 'aten::slice') and inputs:
            views = tuple(tensors(result))
            if len(views) == 1:
                base, view = inputs[0], views[0]
                meta = metadata(base)
                # Retain real non-ND layouts (including base NCDHW=30) as well
                # as public as_strided observations. A layout code other than
                # 0/2 is an inventory condition, not private-format eligibility.
                if meta.get('actual_format') not in (None, 0, 2) or name == 'aten::as_strided':
                    linked = self._emit(dict(kind='view', operator=str(func), schema=str(schema), base=meta,
                        view=metadata(view), aliases_base=view.untyped_storage()._cdata ==
                        base.untyped_storage()._cdata, writes=[], consumers=[]))
                    self.views.append((weakref.ref(view), weakref.ref(base), linked))
        return result

    def __enter__(self):
        import torch
        if self._metadata_token is not None or self.closed:
            raise RuntimeError('TENSOR_OBSERVER_CANNOT_BE_REENTERED')
        self._owner_thread = threading.get_ident()
        self._metadata_token = _METADATA_DIAGNOSTICS.set(self.diagnostics_context)
        rms_norm = getattr(torch.nn, 'RMSNorm', ())
        for path, module in self.model.named_modules():
            identity = callable_identity(module)
            source = identity.get('source', '')
            role = ('conv3d' if isinstance(module, torch.nn.Conv3d) else
                    'conv3d_structural_candidate' if getattr(getattr(module,'weight',None),'ndim',None) == 5
                    and isinstance(getattr(module,'kernel_size',None),(tuple,list))
                    and len(module.kernel_size) == 3 else
                    'rmsnorm_source_candidate' if isinstance(module, rms_norm)
                    or ('rsqrt' in source and 'mean' in source) or 'rms_norm' in source else
                    'rope_table_owner' if hasattr(module, 'cos_sin_cache') else None)
            if role:
                self.module_inventory.append(dict(path=path or '<root>', role=role, **identity))
            for callback, pre in ((self._pre(path or '<root>', module, role), True),
                                  (self._post(module), False)):
                register = module.register_forward_pre_hook if pre else module.register_forward_hook
                options = dict(with_kwargs=True)
                if not pre:
                    options['always_call'] = True
                handle = register(callback, **options)
                self.handles.append((handle, callback))
        return self

    def close(self):
        if self._metadata_token is not None and threading.get_ident() != self._owner_thread:
            raise RuntimeError('TENSOR_OBSERVER_METADATA_OWNER_THREAD_CHANGED')
        if self.stack or self.mode is not None:
            raise RuntimeError('OBSERVER_FORWARD_STILL_INFLIGHT')
        for handle, callback in reversed(self.handles[:]):
            hooks = handle.hooks_dict_ref()
            if hooks is not None and handle.id in hooks and hooks[handle.id] is not callback:
                raise RuntimeError('OBSERVER_HOOK_OWNERSHIP_CHANGED')
            try:
                handle.remove()
            finally:
                if hooks is None or handle.id not in hooks:
                    self.handles.remove((handle, callback))
            if hooks is not None and handle.id in hooks:
                raise RuntimeError('OBSERVER_HOOK_REMOVAL_UNCONFIRMED')
        self.outputs.clear()
        self.views.clear()
        self._dispatch_outputs.clear()
        if self._metadata_token is not None:
            _METADATA_DIAGNOSTICS.reset(self._metadata_token)
            self._metadata_token = None
        self.closed = True

    def report(self):
        return dict(schema='demoncat.natural_tensor_census.v1', source='independent_workload_observer',
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            module_inventory=copy.deepcopy(self.module_inventory), forwards=copy.deepcopy(self.forwards),
            events=copy.deepcopy(self.events), operator_counts=dict(self.counts),
            dropped_calls=self.dropped, complete=bool(self.forwards) and not self.dropped
                and all(x['status'] == 'returned' for x in self.forwards),
            observer_hooks=len(self.handles), inflight=int(self.mode is not None),
            qualification_granted=False, tensor_mutations=0,
            limits=['Eager current-thread dispatch only; opaque/fused internals are not inferred.',
                    'Structural/source candidates are not fault eligibility.',
                    'A source lookup is not RoPE without actual owner and consumer evidence.'])

    def __exit__(self, *exc):
        self.close()
        return False
