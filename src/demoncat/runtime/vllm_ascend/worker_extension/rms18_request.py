"""One ordinary RMS return and its real consumer, inside an actual request.

No full-model return census, group qualification, replay, file or msprobe dump.
Only the first supported normal return is read back. If it has no actual
consumer this bounded task refuses rather than searching all model layers.
"""
from contextlib import nullcontext
import sys
import uuid

from fault_injection.rmsnorm_return import RMSNormReturnSession
from fault_injection.tensor_observer import tensors
from .serving import ServingReplaySession, complete_tensor
from .serving_context import require
from .providers.tensor_modules import _capture, _identity


class _OneRMSReturn(RMSNormReturnSession):
    def __init__(self, task):
        self.task = task
        self.chosen = self.actual = None
        self.done = False
        super().__init__(task.model, arm='fault' if task.phase == 'fault' else 'baseline',
            simulation=task.binding.simulation, passthrough_unselected=True,
            observe_post_root_consumers=True, admission=self._admit)

    def _wanted(self, module):
        if self.done: return False
        if self.task.target_module is not None:
            return module is self.task.target_module
        return self.chosen is None or module is self.chosen

    def dispatch(self, func, args, kwargs):
        # Other RMS calls execute directly after/before the single selected
        # return. Do not build pending events or copy their values.
        name = getattr(getattr(func, '_schema', None), 'name', '')
        native = name.split('::', 1)[0] in ('npu', '_C_ascend') and name.rsplit('::', 1)[-1] in (
            'npu_rms_norm', 'npu_add_rms_norm', 'npu_add_rms_norm_bias')
        if native and self.done:
            return self._read_call(func, args, kwargs)
        if native and self.task.target_module is not None:
            inputs = tuple(tensors((args, kwargs)))
            if not any(value is self.task.target_module.weight for value in inputs):
                return self._read_call(func, args, kwargs)
        return super().dispatch(func, args, kwargs)

    def _return(self, module, args, output):
        if not self._wanted(module):
            self.pending.pop(id(module), None)
            return None
        return super()._return(module, args, output)

    def _proxy_return(self, module, args, output):
        if id(module) not in self.pending and self.eager.get(id(module)) is None:
            return None
        self.chosen = module
        values = tuple(tensors(output))
        require(bool(values) and bool(tuple(tensors(args))), 'RMS18_POSITIONAL_ACTUAL_RETURN_INPUT_REQUIRED')
        weight = module.weight
        _, output_sha, finite = complete_tensor(values[0])
        _, weight_sha, weight_finite = complete_tensor(weight)
        require(finite and weight_finite, 'RMS18_FINITE_ACTUAL_RETURN_WEIGHT_REQUIRED')
        self.actual = dict(module_object_id=id(module), weight_object_id=id(weight),
            weight_storage=int(weight.untyped_storage()._cdata), weight_sha256=weight_sha,
            input=_identity(_capture(args)), original_return_sha256=output_sha,
            shape=list(values[0].shape), dtype=str(values[0].dtype), device=str(values[0].device))
        result = super()._proxy_return(module, args, output)
        require(len(self.events) == 1, 'RMS18_ONE_RETURN_OBSERVATION_REQUIRED')
        self.events[0]['actual'] = self.actual
        self.done = True
        return result

    def _admit(self, event):
        return self.task.admit(event, self.actual)

    def _read_call(self, func, args, kwargs):
        if self.events and self.events[0].get('consumers'):
            return func(*args, **kwargs)
        result = super()._read_call(func, args, kwargs)
        if self.events and self.events[0].get('consumers'):
            self.returns.clear()
        return result

    def dispatch_outside_root(self, func, args, kwargs):
        if self.task.session.logits_running:
            return self._read_call(func, args, kwargs)
        # Permit last-row view/copy lineage between root and real logits, but
        # arbitrary host arithmetic is not an actual model consumer.
        name = getattr(getattr(func, '_schema', None), 'name', '')
        views = ('aten::view', 'aten::reshape', 'aten::_unsafe_view', 'aten::slice', 'aten::select',
                 'aten::index', 'aten::index_select', 'aten::clone', 'aten::_to_copy', 'aten::squeeze',
                 'aten::unsqueeze', 'aten::flatten', 'aten::transpose', 'aten::permute')
        return self._read_call(func, args, kwargs) if name in views else func(*args, **kwargs)


class Rms18TaskRequest(ServingReplaySession):
    def __init__(self, provider):
        self.provider = provider
        self.require = provider.require
        super().__init__(provider.worker, provider.model, provider, None, provider._ticket[0],
            prompt_token_ids=list(provider.binding.prompt), evidence_dir=None, persist_evidence=False)
        self.callbacks = {}
        self.window = None
        self.root_running = self.logits_running = False
        self.request_error = None
        self.entry_error = None
        self.root_post_handle = self.completion_handle = None

    def make_window(self):
        return _OneRMSReturn(self.provider)

    def _same_worker(self):
        super()._same_worker()
        self.provider._context()

    def _add(self, module, callback, pre=False):
        register = module.register_forward_pre_hook if pre else module.register_forward_hook
        handle = register(callback, with_kwargs=True, **({} if pre else {'always_call': True}))
        self.handles.append(handle)
        self.callbacks[handle.id] = callback
        return handle

    def _remove(self, handle):
        hooks = handle.hooks_dict_ref()
        self.require(hooks is not None and (handle.id not in hooks or hooks[handle.id] is self.callbacks[handle.id]),
                'RMS18_OWNED_HOOK_CHANGED')
        try: handle.remove()
        finally:
            if handle.id not in hooks:
                self.handles.remove(handle)
                self.callbacks.pop(handle.id, None)
        self.require(handle.id not in hooks, 'RMS18_HOOK_REMOVAL_UNCONFIRMED')

    def install(self):
        self._same_worker()
        self.require(not self.installed and not self.handles, 'RMS18_HOOKS_ALREADY_INSTALLED')
        try:
            self._add(self.model, self._root_pre, True)
            self._add(self.processor, self._logits_pre, True)
            self.installed = True
        except BaseException:
            self.close()
            raise

    def expect(self, request_id):
        self._same_worker()
        self.readback()
        self.require(self.provider.phase != 'complete', 'RMS18_TASK_PHASES_EXHAUSTED')
        self.require(self.installed and self.expected is self.current is self.window is None
                and not self.request_error and self.provider.phase != 'complete', 'RMS18_PREVIOUS_REQUEST_NOT_CLOSED')
        self.require(isinstance(request_id, str) and 0 < len(request_id) <= 256
                and request_id not in self.provider.requests, 'FRESH_ENQUEUED_ID_REQUIRED')
        self.expected = request_id

    def _diagnostics(self):
        return self.window.diagnostics() if self.window else nullcontext()

    def _failure(self, error):
        if self.request_error is None: self.request_error = type(error).__name__ + ': ' + str(error)
        self.provider.fail(error)

    def _root_pre(self, module, args, kwargs):
        try:
            self.require(not self.root_running and not self.logits_running, 'RMS18_CONCURRENT_ROOT_REFUSED')
            own = [h.id for h in self.handles if h.hooks_dict_ref() is module._forward_pre_hooks]
            self.require(list(module._forward_pre_hooks) == own, 'RMS18_ROOT_PRE_HOOK_ORDER_UNREVIEWED')
            self.entry_error = sys.exc_info()[1]
            ServingReplaySession._root_pre(self, module, args, kwargs)
            self.current.update(request_id=self.current['actual_internal_request_id'],
                invocation=self.provider.code_prefix.lower() + '-task:' + uuid.uuid4().hex, input=_identity(_capture((args, kwargs))))
            self.provider.begin_actual(self.current)
            if self.provider.phase in ('normal', 'fault'):
                self.window = self.make_window()
                self.window.__enter__()
            self.root_post_handle = self._add(self.model, self._root_post)
            self.completion_handle = self._add(self.processor, self._logits_post)
            if self.window:
                # Newly installed root pre-hooks are absent from PyTorch's
                # current snapshot; enter exactly its owned root callback once.
                for handle, callback in getattr(self.window,'entry_hooks',self.window.hooks):
                    if handle.hooks_dict_ref() is module._forward_pre_hooks:
                        result=(callback(module,args,kwargs) if handle.id in module._forward_pre_hooks_with_kwargs
                                else callback(module,args))
                        self.require(result is None, 'RMS18_ROOT_ARGUMENTS_CHANGED')
            self.root_running = True
        except BaseException as error:
            self._failure(error)
            raise

    def _root_post(self, module, args, kwargs, output):
        try:
            actual_error = sys.exc_info()[1]
            if actual_error is not None and actual_error is not self.entry_error:
                self._failure(actual_error)  # Keep the first producer/model failure.
            self._same_worker()
            self.require(self.current is not None and output is not None and sys.exc_info()[1] is self.entry_error
                    and next(reversed(module._forward_hooks)) == self.root_post_handle.id,
                    'RMS18_ORIGINAL_ROOT_RETURN_UNCONFIRMED')
            self.require(output.ndim == 2 and output.shape[0] == len(self.prompt),
                    'COMPLETE_PREFILL_HIDDEN_STATES_REQUIRED')
            with self._diagnostics():
                _, full, finite = complete_tensor(output)
                _, self.hidden_digest, _ = complete_tensor(output[-1:].contiguous())
            self.require(finite, 'RMS18_HIDDEN_NOT_FINITE')
            self.current.update(hidden_sha256=full, root_returned=True)
        except BaseException as error:
            self._failure(error)
            raise
        finally:
            self.root_running = False

    def _logits_pre(self, module, args, kwargs):
        try:
            self._same_worker()
            self.require(self.current is not None and self.current['root_returned'] and not self.logits_running
                    and not self.request_error, 'RMS18_COMPLETED_ROOT_BEFORE_LOGITS_REQUIRED')
            if self.boundary_guard: self.boundary_guard()
            actual = self.logits_binding.bind(module, args, kwargs)
            self.require(actual.get('lm_head') is self.head and actual.get('embedding_bias') is None,
                    'ACTUAL_LOGITS_HEAD_CHANGED')
            with self._diagnostics(): _, digest, finite = complete_tensor(actual['hidden_states'])
            self.require(finite and digest == self.hidden_digest, 'RMS18_LOGITS_ROWS_CHANGED')
            self.logits_running = True
        except BaseException as error:
            self._failure(error)
            raise

    def _logits_post(self, module, args, kwargs, output):
        try:
            self._same_worker()
            self.require(self.logits_running and output is not None and sys.exc_info()[1] is self.entry_error
                    and next(reversed(module._forward_hooks)) == self.completion_handle.id,
                    'RMS18_ORIGINAL_LOGITS_RETURN_UNCONFIRMED')
            self.require(list(output.shape) == self.boundary_report['output_contract'],
                    'COMPLETE_GLOBAL_VOCAB_OUTPUT_REQUIRED')
            with self._diagnostics(): _, digest, finite = complete_tensor(output)
            self.require(finite, 'RMS18_LOGITS_NOT_FINITE')
            self.current['output'] = dict(hidden=self.current['hidden_sha256'], logits=digest)
            self.current['mechanism'] = self.window.report() if self.window else None
            if self.window:
                self.window.close()
                self.window = None
            for handle in (self.completion_handle, self.root_post_handle): self._remove(handle)
            self.completion_handle = self.root_post_handle = None
            self.provider.end_actual(self.current)
            self.current = None
        except BaseException as error:
            self._failure(error)
            raise
        finally:
            self.logits_running = False

    def close_window(self):
        self._same_worker()
        self.require(not self.root_running and not self.logits_running, 'RMS18_ACTUAL_FORWARD_STILL_ACTIVE')
        self.require(self.current is None or self.request_error, 'RMS18_ACTUAL_REQUEST_NOT_FINISHED')
        if self.window:
            self.window.close()
            self.window = None
        for name in ('completion_handle', 'root_post_handle'):
            handle = getattr(self, name)
            if handle is not None:
                self._remove(handle)
                setattr(self, name, None)
        self.current = self.expected = None

    def close(self):
        self.close_window()
        for handle in reversed(self.handles[:]): self._remove(handle)
        self.installed = False

    def readback(self):
        self._same_worker()
        for handle in self.handles:
            self.require(handle.hooks_dict_ref() is not None
                    and handle.hooks_dict_ref().get(handle.id) is self.callbacks[handle.id], 'RMS18_OWNED_HOOK_CHANGED')
        state = self.window.runtime_state() if self.window else dict(ownership_verified=True, leases=0)
        self.require(state['ownership_verified'], 'RMS18_OWNED_DISPATCH_CHANGED')
        return dict(installed=self.installed, owned_hooks=len(self.handles) + (getattr(self.window,'owned_hook_count',len(self.window.hooks)) if self.window else 0),
            inflight=int(self.root_running or self.logits_running), leases=state['leases'],
            pending=self.expected is not None or self.current is not None)
