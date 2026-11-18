"""Actual-forward lifetime for the existing natural Tensor and MoE sessions.

RPCs queue only an expected ID. Owned dispatch/profile scopes begin in the real
root pre-hook and end at its real logits consumer. Existing algorithms and
their independent observers remain responsible for calls, effects and recovery.
"""
import sys
import uuid

from .serving_context import require
from .serving_tensor import ServingTensorGroup
from .serving_moe import ServingMoEGroup
from .providers.tensor_modules import _capture


class _ActualRequest:
    def __init__(self, *args, **kwargs):
        self.deferred_request = self.completion_handle = self.request_error = None
        self.request_entry_error = None
        self.root_active = False
        self.callbacks_active = 0
        super().__init__(*args, **kwargs)
        self.provider.owned_host_hooks = lambda: len(self.handles) + int(self.root_post_handle is not None) + int(
            self.completion_handle is not None)

    def _context(self, *, verify_targets=True):
        super()._context(verify_targets=verify_targets)
        handle = self.completion_handle
        if handle is not None:
            hooks = handle.hooks_dict_ref()
            require(hooks is not None and hooks.get(handle.id) is self._callbacks.get(handle.id),
                    'TENSOR_COMPLETION_HOOK_OWNER_CHANGED')

    def _remove_hook(self, handle):
        try:
            return super()._remove_hook(handle)
        finally:
            hooks = handle.hooks_dict_ref()
            if hooks is not None and handle.id not in hooks:
                if self.root_post_handle is handle:
                    self.root_post_handle = None
                if self.completion_handle is handle:
                    self.completion_handle = None

    def _failed(self, error):
        if self.request_error is None:
            self.request_error = type(error).__name__ + ': ' + str(error)

    def expect(self, request_id, phase, arm=None):
        self._context()
        require(self.deferred_request is None and self.expected is None and self.current is None
                and self.adapter is None and self.observer is None and not self.request_error,
                'PREVIOUS_TENSOR_REQUEST_NOT_COLLECTED')
        require(isinstance(request_id, str) and request_id and request_id not in self.seen,
                'FRESH_REAL_ENQUEUED_REQUEST_REQUIRED')
        require(arm in (None, 'baseline', 'fault'), 'EXPLICIT_TENSOR_ARM_REQUIRED')
        if not self.handles:
            self._install_context()
        self.deferred_request = (request_id, phase, arm)
        return dict(expected=request_id, phase=phase, arm=arm, deferred_until_actual_forward=True)

    def _begin_request(self, request_id, phase, arm):
        self._context()
        self.expected, self.phase = request_id, phase
        self.provider.calls = []
        self.provider.current_invocation = 'worker-tensor-forward:' + uuid.uuid4().hex
        self.provider.current_request_ids = (request_id,)
        self.provider._running = True
        if phase.endswith('off'):
            self.provider.enabled = False
        if arm is not None:
            self.adapter = self._make_adapter(arm)
            self.adapter.__enter__()
            self.provider.active_adapter = self.adapter
        # Retain the object before __enter__: partial installation is owned
        # cleanup, not an unreferenced successful construction.
        self.observer = self._make_observer()
        self.observer.__enter__()
        callback = self._post
        self.root_post_handle = self.model.register_forward_hook(callback, with_kwargs=True, always_call=True)
        self._callbacks[self.root_post_handle.id] = callback

    def _pre(self, module, args, kwargs):
        self.callbacks_active += 1
        try:
            require(self.deferred_request is not None and not self.root_active,
                    'UNSCHEDULED_OR_CONCURRENT_TENSOR_FORWARD')
            self.root_active = True
            self.request_entry_error = sys.exc_info()[1]
            before = dict(module._forward_pre_hooks)
            own = [key for key, callback in before.items() if callback == self._pre]
            require(module is self.model and len(own) == 1 and set(before) == set(own),
                    'TENSOR_ROOT_PRE_HOOK_ORDER_UNREVIEWED')
            request = self.deferred_request
            self.deferred_request = None
            self._begin_request(*request)
            # PyTorch snapshotted pre-hooks before entering this callback.
            # Enter newly installed owned hooks once, in their actual order.
            for key, callback in tuple(module._forward_pre_hooks.items()):
                if key == own[0]:
                    super()._pre(module, args, kwargs)
                elif key not in before:
                    result = (callback(module, args, kwargs) if key in module._forward_pre_hooks_with_kwargs
                              else callback(module, args))
                    require(result is None, 'TENSOR_OWNED_PRE_HOOK_CHANGED_ARGUMENTS')
            callback = self._collect_returned_request
            self.completion_handle = self.processor.register_forward_hook(callback, with_kwargs=True, always_call=True)
            self._callbacks[self.completion_handle.id] = callback
        except BaseException as error:
            self._failed(error)
            # A failure before the post-hook exists never entered model.forward.
            if self.root_post_handle is None:
                self.root_active = False
            raise
        finally:
            self.callbacks_active -= 1

    def _post(self, module, args, kwargs, output):
        self.callbacks_active += 1
        try:
            require(sys.exc_info()[1] is self.request_entry_error
                    and next(reversed(module._forward_hooks)) == self.root_post_handle.id,
                    'TENSOR_ROOT_RETURN_NOT_CONFIRMED')
            return super()._post(module, args, kwargs, output)
        except BaseException as error:
            self._failed(error)
            raise
        finally:
            self.root_active = False
            self.callbacks_active -= 1

    def _collect_returned_request(self, module, args, kwargs, output):
        self.callbacks_active += 1
        try:
            require(not self.request_error and output is not None and sys.exc_info()[1] is self.request_entry_error
                    and next(reversed(module._forward_hooks)) == self.completion_handle.id,
                    'ACTUAL_TENSOR_LOGITS_RETURN_REQUIRED')
            with self._diagnostics():
                self.current['logits'] = _capture(output)
            self._remove_hook(self.completion_handle)
            super().collect()
        except BaseException as error:
            self._failed(error)
            raise
        finally:
            self.callbacks_active -= 1

    def abort(self):
        require(not self.root_active and not self.callbacks_active, 'TENSOR_ACTUAL_FORWARD_STILL_ACTIVE')
        if self.completion_handle is not None:
            self._remove_hook(self.completion_handle)
        self.deferred_request = None
        return super().abort()

    def close(self):
        require(self.deferred_request is None and self.completion_handle is None
                and not self.root_active and not self.callbacks_active, 'TENSOR_ACTUAL_REQUEST_STILL_OWNED')
        return super().close()


class ActualTensorRequest(_ActualRequest, ServingTensorGroup):
    pass


class ActualMoERequest(_ActualRequest, ServingMoEGroup):
    pass
