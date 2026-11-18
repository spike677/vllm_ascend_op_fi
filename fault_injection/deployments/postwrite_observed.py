"""Enter the existing postwrite observer around an already executing request.

This seam never invokes a model or chooses a target. PostwriteProvider retains
the candidate, Native lease, finite-byte checks and recovery authority.
"""
import threading

from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

from .postwrite import _ObserveDispatch, PostwriteRefused


def require(value, reason):
    if not value:
        raise PostwriteRefused(reason)


class PostwriteObservedForward:
    def __init__(self, provider, phase, *, manual_modules, guard):
        self.provider, self.phase, self.guard = provider, phase, guard
        self.manual = dict(manual_modules)
        self.observer = _ObserveDispatch(provider, phase)
        self.handles, self.callbacks = [], {}
        self.mode_active = self.locked = self.active = self.returned = False
        self.failure = None

    def _context(self):
        require(threading.get_ident() == self.provider._thread, 'POSTWRITE_SYNCHRONOUS_OWNER_REQUIRED')
        self.guard()

    def _enter(self, name, module):
        self._context()
        require(self.active and not self.failure, 'POSTWRITE_LATE_OBSERVER_CALLBACK')
        self.provider._stack.append((name, module))
        self.observer.inventory.module_entered(name, module)

    def _leave(self, module):
        self._context()
        require(self.active and self.provider._stack and self.provider._stack[-1][1] is module,
                'POSTWRITE_OBSERVED_MODULE_STACK_CHANGED')
        self.provider._stack.pop()

    def enter_manual(self, module):
        require(module in self.manual, 'POSTWRITE_MANUAL_BOUNDARY_NOT_BOUND')
        self._enter(self.manual[module], module)

    def leave_manual(self, module):
        require(module in self.manual, 'POSTWRITE_MANUAL_BOUNDARY_NOT_BOUND')
        self._leave(module)

    def start(self):
        self._context()
        p = self.provider
        require(not self.locked and not p.live_lease_count and not p._stack,
                'POSTWRITE_RECOVERY_REQUIRED')
        require(p._lock.acquire(False), 'POSTWRITE_SYNCHRONOUS_OWNER_REQUIRED')
        self.locked, self.active = True, True
        p._last_observer, p._last_intervention = self.observer, None
        p._last_native_lease_receipt = None
        for name, module in p.model.named_modules():
            if module in self.manual:
                continue
            def pre(current, inputs, name=name):
                self._enter(name, current)
            def post(current, inputs, output):
                self._leave(current)
            for callback, register, options in ((pre, module.register_forward_pre_hook, {}),
                    (post, module.register_forward_hook, dict(always_call=True))):
                handle = register(callback, **options)
                self.handles.append(handle)
                self.callbacks[handle.id] = callback
        try:
            self.observer.__enter__()
        finally:
            self.mode_active = any(mode is self.observer for mode in _get_current_dispatch_mode_stack())

    def exit_mode(self):
        self._context()
        if not self.mode_active:
            return
        stack = _get_current_dispatch_mode_stack()
        require(stack and stack[-1] is self.observer, 'POSTWRITE_FOREIGN_DISPATCH_MODE_ON_TOP')
        try:
            self.observer.__exit__(None, None, None)
        finally:
            if not any(mode is self.observer for mode in _get_current_dispatch_mode_stack()):
                self.mode_active = False

    def _remove_hook(self, handle):
        hooks = handle.hooks_dict_ref()
        require(hooks is not None and (handle.id not in hooks or
            hooks.get(handle.id) is self.callbacks[handle.id]), 'POSTWRITE_OBSERVER_HOOK_OWNER_CHANGED')
        try:
            handle.remove()
        finally:
            if handle.id not in hooks:
                self.callbacks.pop(handle.id, None)
                self.handles.remove(handle)
        require(handle.id not in hooks, 'POSTWRITE_OBSERVER_HOOK_REMOVE_UNCONFIRMED')

    def finish(self, output):
        self._context()
        require(self.active and not self.provider._stack and not self.returned,
                'POSTWRITE_COMPLETE_OBSERVED_FORWARD_REQUIRED')
        self.returned = True
        self.exit_mode()
        result = self.provider._observed_result(self.observer, output, self.phase)
        self.close(unwound=True)
        return result

    def close(self, *, unwound):
        self._context()
        require(unwound, 'POSTWRITE_ACTUAL_FORWARD_STILL_ACTIVE')
        self.active = False
        self.exit_mode()
        for handle in reversed(self.handles[:]):
            self._remove_hook(handle)
        self.provider._stack.clear()
        if self.locked:
            self.provider._lock.release()
            self.locked = False
        # Existing Native owner/generation/storage guards perform recovery.
        self.provider.recover()

    def readback(self):
        self._context()
        for handle in self.handles:
            hooks = handle.hooks_dict_ref()
            require(hooks is not None and hooks.get(handle.id) is self.callbacks[handle.id],
                    'POSTWRITE_OBSERVER_HOOK_OWNER_CHANGED')
        require(self.mode_active == any(mode is self.observer for mode in _get_current_dispatch_mode_stack()),
                'POSTWRITE_DISPATCH_MODE_READBACK_CHANGED')
        return dict(owned_hooks=len(self.handles), dispatch_mode_present=self.mode_active,
                    lock_retained=self.locked, actual_forward_returned=self.returned)
