"""Independent final Torch Conv3d arguments and live first-consumer lineage.

Install below the padding adapter. This is not an ACLNN/launch observer and
never grants Native qualification. Retained objects prevent storage-id reuse
from manufacturing descendants or consumers.
"""
import copy
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import sys
import threading

from ._scoped_dispatch import _close_owned_mode
from .conv3d_natural import conv_arguments, tensor_identity
from .rope_sites import ModuleSites
from .tensor_callable import TensorCallableGuard
from .tensor_observer import metadata, tensors


def require(condition, reason):
    if not condition:
        raise RuntimeError('VISION17_' + reason)


_DIAGNOSTICS = ContextVar('vision17_owned_diagnostics', default=False)


@contextmanager
def vision17_diagnostics():
    """Shared exclusion for owned readbacks, independent of arm/admission state."""
    token=_DIAGNOSTICS.set(True)
    try:yield
    finally:_DIAGNOSTICS.reset(token)


class Vision17Trace:
    ALIAS = {'aten::view', 'aten::reshape', 'aten::_unsafe_view', 'aten::flatten', 'aten::transpose',
             'aten::permute', 'aten::squeeze', 'aten::unsqueeze', 'aten::slice', 'aten::select',
             'aten::alias', 'aten::detach'}
    COPY = {'aten::clone', 'aten::_to_copy'}
    SHAPE_FACTORIES = {'aten::zeros_like', 'aten::ones_like', 'aten::empty_like', 'aten::full_like'}
    READERS = {'aten::add', 'aten::mul', 'aten::linear', 'aten::mm', 'aten::matmul', 'aten::addmm',
               'aten::bmm', 'aten::convolution', 'aten::conv3d', 'aten::cat'}

    def __init__(self, model, *, simulation=False, diagnostics_context=None, max_calls=16, max_edges=128):
        import torch
        require(type(simulation) is bool and type(max_calls) is int and 0 < max_calls <= 64
                and type(max_edges) is int and 0 < max_edges <= 4096, 'FINITE_SCOPE_REQUIRED')
        self.model, self.simulation = model, simulation
        self.diagnostics_context = diagnostics_context
        self.max_calls, self.max_edges = max_calls, max_edges
        self.sites = ModuleSites(model)
        self.owners = {path: module for path, module in self.sites.nodes.items()
                       if isinstance(getattr(module, 'weight', None), torch.Tensor) and module.weight.ndim == 5}
        self.guards = {id(module): TensorCallableGuard(module) for module in self.owners.values()}
        self.hooks, self.stack, self.rows, self.records, self.forwards = [], [], [], [], []
        self.mode = None; self.closed = False; self.thread = None; self.diagnostic = False
        self.entry_error = None; self.errors = []

    def identity(self, value):
        if value is None:
            return None
        require(self.simulation or value.device.type == 'npu', 'ACTUAL_NPU_TENSOR_REQUIRED')
        if value.device.type == 'npu':
            import torch
            torch.npu.synchronize()
        return tensor_identity(value.detach().cpu().clone())

    def diagnostic_scope(self):
        return self.diagnostics_context() if self.diagnostics_context else nullcontext()

    def enter_module(self, module, args, kwargs):
        if module is self.model:
            require(not self.stack, 'RECURSIVE_OR_CONCURRENT_SCOPE')
            self.entry_error = sys.exc_info()[1]
            self.forwards.append(dict(index=len(self.forwards), returned=False))
        elif not self.stack:
            return
        require(threading.get_ident() == self.thread, 'THREAD_CHANGED')
        path = self.sites.resolve(module, self.stack)
        self.stack.append((path, module))

    def leave_module(self, module, args, kwargs, output):
        if self.stack and self.stack[-1][1] is module:
            self.stack.pop()
        if module is self.model and self.forwards:
            self.forwards[-1]['returned'] = output is not None and sys.exc_info()[1] is self.entry_error

    def dispatch(self, func, args, kwargs):
        if not self.stack or self.diagnostic or _DIAGNOSTICS.get():
            return func(*args, **kwargs)
        require(threading.get_ident() == self.thread, 'THREAD_CHANGED')
        name = getattr(getattr(func, '_schema', None), 'name', str(func))
        inputs = tuple(tensors((args, kwargs)))
        linked = []
        for record in self.records:
            if record['forward'] != len(self.forwards)-1 or record['row']['consumer'] is not None:
                continue
            exact = [(position, held) for position, item in enumerate(inputs)
                     for held in record['lineage'] if item is held]
            if exact:
                linked.append((record, exact))
        values = conv_arguments(func, args, kwargs)
        row = None
        if values is not None:
            require(len(self.rows) < self.max_calls, 'CALL_BUDGET_EXHAUSTED')
            owners = [(path, module) for path, module in self.stack if getattr(module, 'weight', None) is values['weight']]
            require(len(owners) == 1 and self.owners.get(owners[0][0]) is owners[0][1], 'ACTUAL_WEIGHT_OWNER_AMBIGUOUS')
            path, owner = owners[0]; self.guards[id(owner)].verify()
            self.diagnostic = True
            try:
                with self.diagnostic_scope():
                    before = [self.identity(values.get(k)) for k in ('input', 'weight', 'bias')]
                    row = dict(sequence=len(self.rows), forward=len(self.forwards)-1, path=path,
                        module_object_id=id(owner), operator=str(func), schema=str(func._schema),
                        actual_input_metadata=[metadata(t) for t in (values['input'], values['weight'])],
                        actual_geometry={k:values[k] for k in ('stride','padding','dilation','groups','transposed','output_padding') if k in values},
                        inputs_before=before, returned=False, lineage_edges=[], consumer=None,
                        final_torch_native_dispatch_arguments=True, aclnn_launch_arguments=False)
                    self.rows.append(row)
            finally:
                self.diagnostic = False
        result = func(*args, **kwargs)
        outputs = tuple(tensors(result))
        self.diagnostic = True
        try:
            with self.diagnostic_scope():
                if row is not None:
                    require(len(outputs) == 1, 'ONE_CONV_OUTPUT_REQUIRED')
                    row.update(returned=True, inputs_after=[self.identity(values.get(k)) for k in ('input','weight','bias')],
                               output=self.identity(outputs[0]), output_metadata=metadata(outputs[0]))
                    row['inputs_unchanged'] = row['inputs_before'] == row['inputs_after']
                    self.records.append(dict(row=row,forward=len(self.forwards)-1,lineage=[outputs[0]]))
                for record, exact in linked:
                    if name in self.ALIAS | self.COPY:
                        require(len(record['row']['lineage_edges']) < self.max_edges, 'LINEAGE_BUDGET_EXHAUSTED')
                        require(outputs, 'EMPTY_LINEAGE_RETURN')
                        if name in self.ALIAS:
                            require(all(any(out.untyped_storage()._cdata == held.untyped_storage()._cdata
                                            for _,held in exact) for out in outputs), 'UNPROVEN_ALIAS_RETURN')
                        record['row']['lineage_edges'].append(dict(operator=str(func), returned=True,
                            actual_input_object_ids=[id(held) for _,held in exact],
                            output_metadata=[metadata(out) for out in outputs],
                            relation='ACTUAL_RETURNED_ALIAS' if name in self.ALIAS else 'ACTUAL_RETURNED_COPY'))
                        record['lineage'].extend(out for out in outputs if not any(out is held for held in record['lineage']))
                    elif name not in self.SHAPE_FACTORIES:
                        # Do not skip an unknown first consumer and later call a
                        # supported operation the first numerical reader.
                        record['row']['consumer'] = dict(operator=str(func), returned=True,
                            supported_numerical_consumer=name in self.READERS,
                            path=self.stack[-1][0], actual_input_positions=sorted(set(i for i,_ in exact)),
                            actual_input_metadata=[metadata(inputs[i]) for i in sorted(set(i for i,_ in exact))],
                            consumed_values=[self.identity(inputs[i]) for i in sorted(set(i for i,_ in exact))],
                            outputs=[self.identity(out) for out in outputs],
                            binding='EXACT_LIVE_OBJECT_FROM_OBSERVED_RETURN_CHAIN',
                            downstream_model_propagation_proven=False)
        finally:
            self.diagnostic = False
        return result

    def __enter__(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        require(self.mode is None and not self.closed, 'SINGLE_USE_OBSERVER_REQUIRED')
        self.thread = threading.get_ident(); owner = self
        class LowerObserver(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return owner.dispatch(func,args,kwargs or {})
        try:
            for module in self.model.modules():
                for method, options in ((module.register_forward_pre_hook,dict(with_kwargs=True)),
                                        (module.register_forward_hook,dict(with_kwargs=True,always_call=True))):
                    callback = self.enter_module if method.__name__ == 'register_forward_pre_hook' else self.leave_module
                    self.hooks.append((method(callback,**options),callback))
            self.mode=LowerObserver();self.mode.__enter__()
            return self
        except BaseException:
            self.close();raise

    def close(self):
        require(self.thread is None or threading.get_ident() == self.thread, 'CLEANUP_THREAD_CHANGED')
        require(not self.stack, 'ACTUAL_ENCODER_STILL_ACTIVE')
        if self.mode is not None:
            _close_owned_mode(self,'mode','VISION17_FOREIGN_DISPATCH_SCOPE')
        for handle, callback in reversed(self.hooks[:]):
            hooks=handle.hooks_dict_ref()
            require(hooks is not None and (handle.id not in hooks or hooks[handle.id] is callback), 'HOOK_OWNER_CHANGED')
            try:
                handle.remove()
            finally:
                if handle.id not in hooks:self.hooks.remove((handle,callback))
        self.records.clear();self.closed=True

    def report(self):
        complete = bool(self.rows) and all(f['returned'] for f in self.forwards) and all(
            r['returned'] and r['inputs_unchanged'] and r['consumer'] is not None
            and r['consumer']['supported_numerical_consumer'] for r in self.rows)
        return dict(observed_calls=copy.deepcopy(self.rows), structural_candidates=list(self.owners),
                    resolved_candidates=[r['path'] for r in self.rows if r['returned']], qualified_targets=[],
                    forwards=copy.deepcopy(self.forwards), torch_dispatch_chain_complete=complete,
                    aclnn_launch_arguments=False, device_payload_observed=False, native_qualification=False,
                    simulation=self.simulation, hooks=len(self.hooks), owned_mode=self.mode is not None,
                    held_lineage_tensors=sum(len(r['lineage']) for r in self.records),
                    c1=False,c2=False,c3=False,e5=False)

    def __exit__(self,*args):
        self.close()
