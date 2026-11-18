"""Owned eager-dispatch scope shared by the rebuilt 8 and 22 adapters."""
from contextlib import contextmanager, nullcontext
import copy
import sys
import threading

from .core import UnsupportedCall

_LEASE = threading.Lock()


class ScopedDispatch:
    def __init__(self, model, *, arm='fault', simulation=False, selected_call=None,
                 diagnostics_context=None):
        if arm not in ('baseline', 'fault') or type(simulation) is not bool:
            raise ValueError('Explicit arm and simulation flag required')
        self.model, self.arm, self.simulation = model, arm, simulation
        self.selected_call = selected_call
        self.thread = None
        self.hooks = []
        self.stack = []
        self.events = []
        self.forwards = []
        self.root_entries = []
        self.mode = None
        self.active = self.closed = self.locked = False
        self.diagnostic = False
        self.temps = []
        self.devices = set()
        self.errors = []
        self.diagnostics_context = diagnostics_context

    @contextmanager
    def diagnostics(self):
        before = self.diagnostic
        self.diagnostic = True
        try:
            with self.diagnostics_context() if self.diagnostics_context else nullcontext():
                yield
        finally: self.diagnostic = before

    def synchronize(self):
        if 'npu' in self.devices:
            import torch
            torch.npu.synchronize()

    def cpu(self, tensor):
        self.devices.add(tensor.device.type)
        self.synchronize()
        return tensor.detach().cpu().clone()

    def _before(self, path, module):
        def enter(actual, args):
            if not self.active: return
            if threading.get_ident() != self.thread:
                raise UnsupportedCall('Concurrent model calls are not isolated by this adapter')
            if actual is self.model:
                if self.stack:
                    self.root_entries.append(None)
                    self.errors.append('Recursive root forward')
                    raise UnsupportedCall('Recursive model call refused')
                self.root_entries.append((len(self.forwards), sys.exc_info()[1]))
                self.forwards.append(dict(index=len(self.forwards), status='running'))
            elif not self.stack:
                return
            self.stack.append((path, module))
        return enter

    def _after(self, module):
        def leave(actual, args, output):
            if not self.active or threading.get_ident() != self.thread: return
            if actual is self.model:
                entry = self.root_entries.pop() if self.root_entries else None
                if entry is None: return
                index, entering_error = entry
                self.forwards[index]['status'] = ('returned' if output is not None
                    and sys.exc_info()[1] is entering_error else 'no_output_or_exception')
            if self.stack and self.stack[-1][1] is actual:
                self.stack.pop()
        return leave

    def __enter__(self):
        if self.active or self.closed or not _LEASE.acquire(blocking=False):
            raise RuntimeError('Another scoped dispatcher is active or this session was closed')
        self.locked = True
        self.thread = threading.get_ident()
        try:
            from torch.utils._python_dispatch import TorchDispatchMode
            parent = self
            class Observe(TorchDispatchMode):
                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    if not parent.active or parent.diagnostic:
                        return func(*args, **(kwargs or {}))
                    if not parent.stack:
                        return parent.dispatch_outside_root(func, args, kwargs or {})
                    return parent.dispatch(func, args, kwargs or {})
            for path, module in self.model.named_modules():
                pre = self._before(path or '<root>', module)
                post = self._after(module)
                self.hooks.append((module.register_forward_pre_hook(pre, prepend=module is self.model), pre))
                self.hooks.append((module.register_forward_hook(post, always_call=True), post))
            self.mode = Observe()
            self.mode.__enter__()
            self.active = True
            return self
        except BaseException:
            self.close()
            raise

    def event(self, operator):
        event = dict(path=self.stack[-1][0], ordinal=len(self.events), operator=operator,
                     forward=len(self.forwards)-1, status='observed')
        self.events.append(event)
        return event

    def dispatch_outside_root(self, func, args, kwargs):
        # Default scopes never inspect or inject unrelated post-root work.
        return func(*args, **kwargs)

    def is_selected(self, event):
        return self.selected_call is None or self.selected_call == (event['path'], event['ordinal'])

    def runtime_state(self):
        from torch.utils._python_dispatch import _get_current_dispatch_mode
        owned = all(handle.hooks_dict_ref() is not None
                    and handle.hooks_dict_ref().get(handle.id) is callback
                    for handle, callback in self.hooks)
        if self.mode is not None:
            owned = owned and threading.get_ident() == self.thread and _get_current_dispatch_mode() is self.mode
        return dict(hook_owned=bool(self.hooks or self.mode), ownership_verified=owned,
                    enabled=self.active, inflight=int(bool(self.stack)), leases=int(self.locked),
                    cleanup_confirmed=self.closed and not self.locked)

    def report(self):
        return dict(mechanism=self.mechanism, arm=self.arm, simulation=self.simulation,
                    events=copy.deepcopy(self.events), forwards=copy.deepcopy(self.forwards),
                    errors=list(self.errors), execution_devices=sorted(self.devices),
                    npu_validation_claimed=False, model_acceptance_claimed=False,
                    coverage='OBSERVED_BOUNDED_SCOPE_NOT_PUBLIC_ALL_ELIGIBLE')

    def close(self):
        if self.closed: return
        if self.locked and threading.get_ident() != self.thread:
            raise RuntimeError('Dispatcher cleanup must run on the owning thread')
        if self.stack or self.root_entries:
            raise RuntimeError('Model forward is still in flight')
        self.active = False
        self.synchronize()
        if self.mode is not None:
            from torch.utils._python_dispatch import _get_current_dispatch_mode
            if _get_current_dispatch_mode() is not self.mode:
                raise RuntimeError('Foreign dispatcher is above the owned mode')
            self.mode.__exit__(None,None,None)
            self.mode = None
        for handle, callback in reversed(self.hooks[:]):
            hooks = handle.hooks_dict_ref()
            if hooks is not None and handle.id in hooks and hooks[handle.id] is not callback:
                raise RuntimeError('Foreign model hook replaced an owned callback')
            handle.remove()
            if hooks is not None and handle.id in hooks:
                raise RuntimeError('Owned dispatcher hook removal is unconfirmed')
            self.hooks.remove((handle, callback))
        self.temps.clear()
        self.closed = True
        if self.locked:
            _LEASE.release(); self.locked = False

    def __exit__(self, *exc):
        self.close()
        return False
