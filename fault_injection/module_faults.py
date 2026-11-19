"""Structural adapters reconstructed from documented effect proxies.

These are new local implementations, not byte-identical historical workers.
No NPU/model-quality validation is inherited from the historical evidence.
"""
from __future__ import annotations

import threading
import copy
import sys
import hashlib
from pathlib import Path
import types

from .core import UnsupportedCall

_LEASE = threading.Lock()

# Reviewed actual collector on 124. Its instance wrapper does not set
# __wrapped__; matching a class/name alone would wrongly admit custom forwards.
_REVIEWED_MSPROBE_FORWARD_HASHES = {
    '4876e89f7f726cd552fc1bf0e7b7745a898461deec2cb2abf33b02deba7329d7',
}
_MSPROBE_MODULE = 'msprobe.pytorch.dump.module_dump.module_processor'


def _original_standard_forward(module):
    """Follow only the reviewed collector's exact code/closure; keep it installed."""
    actual = module.forward
    seen = set()
    while getattr(actual,'__func__',None) is None:
        if id(actual) in seen or len(seen)>8:
            return None
        seen.add(id(actual))
        collector = sys.modules.get(_MSPROBE_MODULE)
        source = getattr(collector,'__file__',None)
        installer = getattr(collector,'wrap_forward_with_hook_safety',None)
        if (not isinstance(actual,types.FunctionType) or not isinstance(installer,types.FunctionType)
                or not source or Path(source).suffix!='.py'):
            return None
        try:
            digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        except OSError:
            return None
        if digest not in _REVIEWED_MSPROBE_FORWARD_HASHES:
            return None
        if Path(installer.__code__.co_filename).resolve()!=Path(source).resolve():
            return None
        codes = [c for c in installer.__code__.co_consts
                 if isinstance(c,types.CodeType) and c.co_name=='wrapped_forward']
        if len(codes)!=1 or actual.__code__ is not codes[0]:
            return None
        cells = dict(zip(actual.__code__.co_freevars,actual.__closure__ or ()))
        try:
            if cells['module'].cell_contents is not module:
                return None
            actual = cells['original_forward'].cell_contents
        except (KeyError,ValueError):
            return None
    return actual if getattr(actual,'__self__',None) is module else None


def discover_modules(model, fault):
    import torch
    if fault == '17.spatial_padding':
        test = lambda m: (isinstance(m, torch.nn.Conv3d)
            and getattr(_original_standard_forward(m), '__func__', None) is torch.nn.Conv3d.forward
            and getattr(m._conv_forward, '__func__', None) is torch.nn.Conv3d._conv_forward)
    elif fault == '18.api_half':
        test = lambda m: (isinstance(m, torch.nn.RMSNorm)
            and getattr(_original_standard_forward(m), '__func__', None) is torch.nn.RMSNorm.forward)
    else:
        raise ValueError(fault)
    return [(name or '<root>', module) for name, module in model.named_modules() if test(module)]


class ModuleFaultSession:
    """All eligible actual calls, without hardcoded model names or paths.

    Fault17 changes only the native Conv3d padding argument. Fault18 is an
    explicitly named module-return gain proxy; it is not native Reduce error.
    This adapter keeps evidence in memory. The unified session facade persists
    report() when evidence_dir is configured; it is not an msprobe capture.
    """
    _cleanup_retryable = True

    def __init__(self, model, fault, *, arm='fault', unsupported='error', evidence_dir=None,
                 _selected_call=None):
        if arm not in ('baseline', 'fault') or unsupported not in ('error', 'passthrough'):
            raise ValueError('Invalid arm/unsupported policy')
        self.model, self.fault, self.arm, self.unsupported = model, fault, arm, unsupported
        self.evidence_dir = evidence_dir
        self.candidates = discover_modules(model, fault)
        self.events, self.handles, self.stack = [], [], []
        self.active = self.closed = False
        self._locked = False
        self.owner = None
        self.root_active = False
        self.root_entries, self.forwards, self.rejected_roots = [], [], []
        self.forward_index = None
        self.original = self.wrapper = None
        self.cleanup_errors = []
        # Deployment-selected actual call, never a user-supplied layer name.
        # The ordinary attach facade retains its documented all-call behavior.
        self.selected_call = _selected_call
        self.owned_callbacks = {}

    def _own_hook(self, module, callback, *, pre=False, **kwargs):
        method = module.register_forward_pre_hook if pre else module.register_forward_hook
        handle = method(callback, **kwargs)
        self.handles.append(handle)
        self.owned_callbacks[handle.id] = callback
        return handle

    def runtime_state(self):
        """Read the real hook dictionaries, without treating hooks as native code."""
        owned = True
        for handle in self.handles:
            ref = getattr(handle, 'hooks_dict_ref', None)
            hooks = ref() if callable(ref) else None
            if hooks is None or hooks.get(handle.id) is not self.owned_callbacks.get(handle.id):
                owned = False
        if self.wrapper is not None:
            import torch.nn.functional as F
            owned = owned and F.conv3d is self.wrapper
        return dict(hook_owned=bool(self.handles or self.wrapper), ownership_verified=owned,
                    enabled=self.active, inflight=int(self.root_active),
                    leases=int(self._locked), cleanup_confirmed=self.closed and not self._locked,
                    owner_thread=self.owner)

    def __enter__(self):
        if self.active or self.closed:
            raise RuntimeError('Session cannot be reused')
        if not self.candidates:
            raise UnsupportedCall('No structurally supported modules; no hook was installed')
        if not _LEASE.acquire(blocking=False):
            raise RuntimeError('Another module proxy session is active')
        self.owner = threading.get_ident()
        self._locked = True
        try:
            self.active = True
            self._own_hook(self.model, self._root_enter, pre=True, with_kwargs=True, prepend=True)
            for path, module in self.candidates:
                self._own_hook(module, self._pre(path), pre=True, with_kwargs=True)
                self._own_hook(module, self._post(path), with_kwargs=True, always_call=True)
            self._own_hook(self.model, self._root_leave, with_kwargs=True, always_call=True)
            if self.fault == '17.spatial_padding':
                import torch.nn.functional as F
                self.original = F.conv3d
                self.wrapper = self._conv
                F.conv3d = self.wrapper
            return self
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                raise RuntimeError('Module installation failed: '+repr(error)+'; cleanup failed: '+repr(cleanup)) from error
            raise

    def _root_enter(self, module, args, kwargs):
        if not self.active:
            return
        if threading.get_ident() != self.owner:
            self.rejected_roots.append(dict(reason='Concurrent root call', thread=threading.get_ident()))
            raise UnsupportedCall('Concurrent or recursive model execution is unsupported')
        if self.root_active:
            # always_call hooks run even when this pre-hook raises. A rejected
            # nested invocation must not clear the owning outer invocation.
            self.root_entries.append(None)
            self.rejected_roots.append(dict(reason='Recursive root call', forward=self.forward_index))
            raise UnsupportedCall('Concurrent or recursive model execution is unsupported')
        self.forward_index = len(self.forwards)
        self.forwards.append(dict(index=self.forward_index, status='running'))
        self.root_entries.append((self.forward_index, sys.exc_info()[1]))
        self.root_active = True

    def _root_leave(self, module, args, kwargs, output):
        if threading.get_ident() == self.owner and self.root_entries:
            entry = self.root_entries.pop()
            if entry is None:
                return
            accepted, entering_exception = entry
            failed = sys.exc_info()[1] is not entering_exception
            self.forwards[accepted]['status'] = 'returned' if output is not None and not failed else 'no_output_or_exception'
            self.root_active = False
            self.forward_index = None

    def _pre(self, path):
        def enter(module, args, kwargs):
            if not self.root_active:
                return
            if threading.get_ident() != self.owner:
                raise UnsupportedCall('Concurrent model execution is unsupported')
            event = dict(path=path, ordinal=len(self.events), forward=self.forward_index,
                         status='observed', reason=None, original_execution_verified=False,
                         module_object_id=id(module), module_type=type(module).__module__+'.'+type(module).__qualname__,
                         module_forward_identity=id(getattr(module.forward,'__func__',module.forward)))
            self.events.append(event)
            self.stack.append((module, event))
        return enter

    def _post(self, path):
        def leave(module, args, kwargs, output):
            if threading.get_ident() != self.owner or not self.stack or self.stack[-1][0] is not module:
                return None
            _, event = self.stack.pop()
            if output is None:
                event.update(status='failed', reason='Module returned no output or raised')
                return None
            if self.fault == '18.api_half':
                import torch
                if not isinstance(output, torch.Tensor) or not output.is_floating_point() or module.training:
                    event.update(status='unsupported', reason='RMSNorm inference floating tensor output required')
                    if self.unsupported == 'error':
                        raise UnsupportedCall(event['reason'])
                    event['status'] = 'passthrough'
                    return None
                selected = self.selected_call is None or self.selected_call == (path, event['ordinal'])
                injecting = self.arm == 'fault' and selected
                event.update(status='applied' if injecting else 'baseline', original_execution_verified=True,
                             output_shape=list(output.shape), dtype=str(output.dtype),
                             boundary='module_return_gain_proxy', gain=0.5 if injecting else 1.0)
                from .actions import scale_value
                return scale_value(output, 0.5) if injecting else None
            if event['status'] == 'observed':
                event.update(status='unsupported', reason='No public Functional.conv3d call observed')
                if self.unsupported == 'error':
                    raise UnsupportedCall(event['reason'])
            return None
        return leave

    @staticmethod
    def _triple(value):
        return (value,)*3 if isinstance(value, int) else tuple(value)

    def _conv(self, *args, **kwargs):
        if threading.get_ident() != self.owner or not self.stack:
            return self.original(*args, **kwargs)
        module, event = self.stack[-1]
        if event['status'] != 'observed':
            raise UnsupportedCall('Multiple convolutions in one selected module need their own adapter')
        names = ('input','weight','bias','stride','padding','dilation','groups')
        values = dict(bias=None, stride=1, padding=0, dilation=1, groups=1)
        if len(args) > len(names) or set(kwargs)-set(names) or set(names[:len(args)]) & set(kwargs):
            raise UnsupportedCall('Unrecognized Conv3d arguments')
        values.update(zip(names, args)); values.update(kwargs)
        x, w = values['input'], values['weight']
        stride, padding, dilation = (self._triple(values[k]) for k in ('stride','padding','dilation'))
        kernel = tuple(w.shape[2:])
        # Preserve historical p=(0,7,7), and admit only mathematically proven
        # shape-preserving patch geometry. Batch/channel counts are discovered.
        eligible = (not module.training and x.ndim == 5 and w is module.weight
            and values['bias'] is module.bias and tuple(x.shape[2:]) == kernel
            and kernel[1:] == (16,16) and stride == kernel and padding == (0,0,0)
            and dilation == (1,1,1) and values['groups'] == 1 and module.padding_mode == 'zeros')
        if not eligible:
            event.update(status='unsupported', reason='Only zero-padded, nonoverlap 16x16 spatial patch Conv3d is supported')
            if self.unsupported == 'error':
                raise UnsupportedCall(event['reason'])
            event['status'] = 'passthrough'
            return self.original(*args, **kwargs)
        selected = self.selected_call is None or self.selected_call == (event['path'], event['ordinal'])
        injecting = self.arm == 'fault' and selected
        chosen = (0,7,7) if injecting else padding
        native_kwargs = dict(values, padding=chosen)
        output = self.original(**native_kwargs)
        expected = (x.shape[0], w.shape[0], 1, 1, 1)
        if tuple(output.shape) != expected:
            event.update(status='failed', reason='Unexpected native Conv3d shape')
            raise UnsupportedCall(event['reason'])
        event.update(status='applied' if injecting else 'baseline', original_execution_verified=True,
                     input_shape=list(x.shape), weight_shape=list(w.shape), output_shape=list(output.shape),
                     native_calls=1, padding=list(chosen), boundary='Functional.conv3d.arguments')
        return output

    def report(self):
        paths = [p for p, _ in self.candidates]
        rows = {p: [e for e in self.events if e['path'] == p] for p in paths}
        by_forward = []
        for forward in self.forwards:
            current = {p: [e for e in rows[p] if e['forward'] == forward['index']] for p in paths}
            covered = bool(paths) and forward['status']=='returned' and all(
                calls and all(e['status'] in ('applied','baseline') for e in calls) for calls in current.values())
            by_forward.append(dict(forward, coverage={p:len(calls) for p,calls in current.items()},
                                   all_discovered_candidates_covered=covered))
        complete = bool(by_forward) and not self.rejected_roots and all(
            row['all_discovered_candidates_covered'] for row in by_forward)
        return dict(fault=self.fault, arm=self.arm, candidate_modules=len(paths),
                    coverage={p:len(v) for p,v in rows.items()}, events=copy.deepcopy(self.events),
                    forwards=by_forward, rejected_root_calls=copy.deepcopy(self.rejected_roots),
                    cleanup_errors=list(self.cleanup_errors),
                    all_discovered_candidates_covered=complete, npu_validation_claimed=False,
                    implementation='new_local_documented_proxy_reimplementation',
                    evidence_persistence='in_memory_adapter_report; unified facade owns optional JSON persistence; not msprobe',
                    limits=['No inherited historical msprobe alignment or model quality claim.',
                            'Coverage requires every discovered candidate in every completed root forward; inactive branches remain unobserved.',
                            'Unobserved/custom module implementations remain unsupported.'])

    def assert_coverage(self):
        result = self.report()
        if not result['all_discovered_candidates_covered']:
            raise UnsupportedCall('Missing or rejected calls; inspect report()')
        return result

    def close(self):
        if self.closed:
            return
        if self._locked and threading.get_ident() != self.owner:
            raise RuntimeError('Session must be closed on its owning thread')
        if self.root_active or self.stack or self.root_entries:
            raise RuntimeError('Module session cannot close inside an active model forward')
        # Stop admission while retaining ownership of anything not yet restored.
        self.active = False
        errors = []
        if self.wrapper is not None:
            try:
                import torch.nn.functional as F
                if F.conv3d is self.wrapper:
                    F.conv3d = self.original
                elif F.conv3d is not self.original:
                    raise RuntimeError('Foreign conv3d wrapper replaced this session; not overwritten')
                self.wrapper = None
            except BaseException as exc:
                errors.append(exc)
        remaining = []
        for handle in reversed(self.handles):
            try:
                ref = getattr(handle, 'hooks_dict_ref', None)
                hooks = ref() if callable(ref) else None
                key = getattr(handle, 'id', None)
                if hooks is not None and key in hooks and hooks[key] is not self.owned_callbacks.get(key):
                    raise RuntimeError('Foreign module hook replaced an owned callback')
                handle.remove()
            except BaseException as exc:
                errors.append(exc)
                remaining.append(handle)
        self.handles = list(reversed(remaining))
        if errors:
            self.cleanup_errors.extend(f'{type(error).__name__}: {error}' for error in errors)
            raise RuntimeError('Module cleanup failed: ' + '; '.join(str(error) for error in errors)) from errors[0]
        self.closed = True
        if self._locked:
            _LEASE.release()
            self._locked = False

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if exc_value is not None:
                raise RuntimeError('Model failed: ' + repr(exc_value) + '; module cleanup failed: ' + repr(cleanup)) from exc_value
            raise
        return False
