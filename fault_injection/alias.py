"""Automatic model-scoped private-format view alias detachment (fault 15).

Only the public torch.as_strided entry is intercepted. Actual writes and base
consumers are observed separately; discovering a view never certifies a write
path. Historical frozen probes and their narrower acceptance remain unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import functools
import sys
import threading

from .core import UnsupportedCall

_LEASE = threading.Lock()
# Conservative data-reading consumers. Metadata-only factories and in-place
# writers do not establish that downstream arithmetic consumed old base data.
_READ_CONSUMERS = {'aten::add', 'aten::sub', 'aten::mul', 'aten::div', 'aten::mm',
                   'aten::bmm', 'aten::matmul', 'aten::addmm', 'aten::linear'}


class NpuFormatProvider:
    """Real format conversion only. There is deliberately no CPU fallback."""
    simulation = False

    def format(self, tensor):
        if tensor.device.type != 'npu':
            return None
        import torch_npu
        return int(torch_npu.get_npu_format(tensor))

    def convert(self, tensor):
        import torch_npu
        return torch_npu.npu_format_cast(tensor, 2)

    def synchronize(self, tensor):
        if tensor.device.type == 'npu':
            import torch
            torch.npu.synchronize(tensor.device)


def _storage(tensor, provider):
    storage = tensor.untyped_storage()
    return {'storage_id': int(storage._cdata), 'storage_bytes': storage.nbytes(),
            'offset': tensor.storage_offset(), 'shape': list(tensor.shape),
            'stride': list(tensor.stride()), 'dtype': str(tensor.dtype),
            'device': str(tensor.device), 'actual_format': provider.format(tensor),
            'tensor_object_id': id(tensor)}


class AliasSession:
    """Bind eligible views at their actual active model-module call boundary.

    `_format_provider` is a testing/provider seam, never a user layer selector.
    CPU providers must declare simulation=True; reports cannot grant NPU proof.
    Base/temporary/view objects remain alive for the duration of this object.
    """
    _cleanup_retryable = True

    def __init__(self, model, arm='fault', unsupported='error', evidence_dir=None,
                 *, _format_provider=None):
        if arm not in ('baseline', 'fault') or unsupported not in ('error', 'passthrough'):
            raise ValueError('Invalid arm or unsupported policy')
        if not callable(getattr(model, 'named_modules', None)):
            raise TypeError('Expected a model instance exposing named_modules')
        self.model, self.arm, self.unsupported = model, arm, unsupported
        self.provider = _format_provider or NpuFormatProvider()
        if not hasattr(self.provider, 'simulation'):
            raise TypeError('Format provider must disclose simulation status')
        self.evidence_dir = evidence_dir
        self._modules = [(p or '<root>', m) for p, m in model.named_modules()]
        self._hooks, self._records, self._events = [], [], []
        self._tls = threading.local()
        self._owner_thread = None
        self._active = self._locked = self._closed = False
        self._original = self._wrapper = self._mode = None
        self._mode_cleanup_uncertain = False
        self._cleanup_errors = []
        self._rejected_root_calls = []
        self._forwards = []

    @contextmanager
    def _diagnostic(self):
        old = getattr(self._tls, 'diagnostic', False)
        self._tls.diagnostic = True
        try:
            yield
        finally:
            self._tls.diagnostic = old

    def _cpu(self, tensor):
        self.provider.synchronize(tensor)
        return tensor.detach().cpu().clone()

    def _pre(self, path, module):
        def enter(actual, args):
            if not self._active or actual is not module:
                return
            if threading.get_ident() != self._owner_thread:
                if actual is self.model:
                    self._rejected_root_calls.append(dict(reason='Concurrent root call', thread=threading.get_ident()))
                    raise UnsupportedCall('Model forward must run on the alias session owning thread')
                return
            stack = getattr(self._tls, 'stack', [])
            if actual is self.model:
                entries = getattr(self._tls, 'root_entries', [])
                if stack:
                    # PyTorch executes always_call post-hooks after a rejected
                    # pre-hook. Do not let that post-hook pop the outer root.
                    entries.append(None)
                    self._tls.root_entries = entries
                    self._rejected_root_calls.append(dict(reason='Recursive root call', forward=self._tls.forward_index))
                    raise UnsupportedCall('Recursive model forward cannot be isolated')
                self._tls.forward_index = len(self._forwards)
                self._forwards.append(dict(index=self._tls.forward_index, status='running'))
                entries.append((self._tls.forward_index, sys.exc_info()[1]))
                self._tls.root_entries = entries
            elif not stack:
                return  # Direct calls to a child outside the owning model are untouched.
            stack.append((path, actual))
            self._tls.stack = stack
        return enter

    def _post(self, module):
        def leave(actual, args, output):
            if not self._active or threading.get_ident() != self._owner_thread:
                return
            if actual is self.model:
                entries = getattr(self._tls, 'root_entries', [])
                entry = entries.pop() if entries else None
                if entry is None:
                    return
                index, entering_exception = entry
                failed = sys.exc_info()[1] is not entering_exception
                self._forwards[index]['status'] = 'returned' if output is not None and not failed else 'no_output_or_exception'
            stack = getattr(self._tls, 'stack', [])
            if stack and stack[-1][1] is actual:
                stack.pop()
        return leave

    def _in_scope(self):
        return (self._active and threading.get_ident() == self._owner_thread
                and bool(getattr(self._tls, 'stack', []))
                and not getattr(self._tls, 'diagnostic', False))

    def __enter__(self):
        if self._active or self._closed:
            raise RuntimeError('Session cannot be re-entered')
        if not _LEASE.acquire(blocking=False):
            raise RuntimeError('Another alias session owns the public API')
        self._locked = True
        try:
            import torch
            from torch.utils._python_dispatch import TorchDispatchMode
            session = self

            class Observer(TorchDispatchMode):
                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    return session._observe(func, args, kwargs or {})

            self._owner_thread = threading.get_ident()
            for path, module in self._modules:
                self._hooks.append(module.register_forward_pre_hook(self._pre(path, module), prepend=module is self.model))
                self._hooks.append(module.register_forward_hook(self._post(module), always_call=True))
            self._original = torch.as_strided

            @functools.wraps(self._original)
            def wrapper(input, size, stride, storage_offset=None):
                before = len(self._events)
                try:
                    return self._dispatch(input, size, stride, storage_offset)
                except BaseException as exc:
                    if len(self._events) > before:
                        self._events[-1].update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
                    raise

            self._wrapper = wrapper
            torch.as_strided = wrapper
            mode = Observer()
            mode.__enter__()
            self._mode = mode
            self._active = True
            return self
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                raise RuntimeError('Alias installation failed: '+repr(error)+'; cleanup failed: '+repr(cleanup)) from error
            raise

    def _dispatch(self, base, size, stride, offset):
        if not self._in_scope():
            return self._original(base, size, stride, offset)
        import torch
        path = self._tls.stack[-1][0]
        event = dict(path=path, forward=self._tls.forward_index, status='observed',
                     size=list(size), stride=list(stride), offset=offset)
        self._events.append(event)
        with self._diagnostic():
            fmt = self.provider.format(base)
            event['eligible'] = fmt == 29
            if base.device.type != 'npu' and not self.provider.simulation and fmt is not None:
                raise UnsupportedCall('A production format provider cannot classify CPU storage as NPU')
            matched = (fmt == 29 and base.dtype == torch.bfloat16 and tuple(base.shape) == (16, 16)
                       and tuple(base.stride()) == (16, 1) and base.storage_offset() == 0
                       and tuple(size) == (8, 8) and tuple(stride) == (16, 1) and offset == 17)
            if not matched:
                # Ordinary CPU/other non-private views are not fault candidates.
                event.update(status='unsupported' if fmt == 29 else 'unmatched', actual_format=fmt)
                if fmt == 29 and self.unsupported == 'error':
                    raise UnsupportedCall('Private view is outside the pinned NZ29/BF16 geometry')
                result = self._original(base, size, stride, offset)
                if fmt == 29:
                    event['status'] = 'passthrough'
                return result
            before = self._cpu(base)
            argument = self.provider.convert(base) if self.arm == 'fault' else base
            converted_values = self._cpu(argument)
            if not torch.equal(before, converted_values):
                raise UnsupportedCall('Format conversion changed logical base values')
            if self.arm == 'fault' and argument.untyped_storage()._cdata == base.untyped_storage()._cdata:
                raise UnsupportedCall('Format conversion did not create independent storage')
            if self.arm == 'fault' and self.provider.format(argument) not in (0, 2):
                raise UnsupportedCall('Conversion did not produce a base format')
            # This is the only injected factor and the only original view call.
            view = self._original(argument, size, stride, offset)
            view_before = self._cpu(view)
            indices = torch.tensor([[17 + i * 16 + j for j in range(8)] for i in range(8)])
            if not torch.equal(view_before, before.flatten()[indices]):
                raise UnsupportedCall('Native returned view has incorrect initial logical values')
            shares_argument = view.untyped_storage()._cdata == argument.untyped_storage()._cdata
            if not shares_argument:
                raise UnsupportedCall('Native view does not alias its actual construction argument')
            event.update(status='routed', actual_format=fmt, initial_values_correct=True,
                         base=_storage(base,self.provider), view=_storage(view,self.provider), temporary=_storage(argument,self.provider),
                         aliases_original_base=view.untyped_storage()._cdata == base.untyped_storage()._cdata,
                         writes=[], consumers=[], unclassified_base_uses=[], observation_errors=[])
            self._records.append(dict(base=base, argument=argument, view=view, before=before,
                                      indices=indices, event=event))
            return view

    def _observe(self, func, args, kwargs):
        if not self._in_scope():
            return func(*args, **kwargs)
        involved = [r for r in self._records if r['event']['forward'] == self._tls.forward_index
                    and any(arg is r['base'] or arg is r['view'] for arg in args)]
        try:
            return self._observe_impl(func, args, kwargs)
        except BaseException as exc:
            # Routing a view is not a successful observed write/consumer when
            # the real operation or the independent observer fails afterwards.
            for record in involved:
                record['event']['status'] = 'failed'
                record['event']['observation_errors'].append(dict(
                    operator=str(func), error=f'{type(exc).__name__}: {exc}'))
            raise

    def _observe_impl(self, func, args, kwargs):
        name = str(func)
        schema = getattr(func, '_schema', None)
        data_reader = getattr(schema, 'name', None) in _READ_CONSUMERS
        active = [r for r in self._records if r['event']['forward'] == self._tls.forward_index]
        copies = [r for r in active if name == 'aten.copy_.default' and args and args[0] is r['view']]
        base_uses = [r for r in active if r['event']['writes'] and any(arg is r['base'] for arg in args)]
        consumers = base_uses if data_reader else []
        if not data_reader:
            for record in base_uses:
                record['event']['unclassified_base_uses'].append(dict(
                    operator=name, reason='Not a verified data-reading consumer; metadata/write/view use is insufficient'))
        copy_inputs = []
        with self._diagnostic():
            for r in copies:
                # Expected values from the actual source, independently from the
                # copy_ return and without consulting injection counters.
                value = self._cpu(args[1]).to(r['view'].dtype).expand(r['view'].shape).clone()
                copy_inputs.append((r, value, self._cpu(r['base'])))
            consumer_inputs = [(r, self._cpu(r['base'])) for r in consumers]
        result = func(*args, **kwargs)
        with self._diagnostic():
            import torch
            for r, values, before_copy in copy_inputs:
                after, written = self._cpu(r['base']), self._cpu(r['view'])
                expected = before_copy.clone()
                expected.flatten()[r['indices']] = values
                target = r['indices'].flatten()
                outside = torch.ones(before_copy.numel(), dtype=torch.bool); outside[target] = False
                r['event']['writes'].append(dict(operator=name, view_write_correct=torch.equal(written, values),
                    base_matches_expected_update=torch.equal(after, expected),
                    base_target_retains_prewrite_values=torch.equal(after.flatten()[target], before_copy.flatten()[target]),
                    write_changes_target_values=not torch.equal(values, before_copy.flatten()[r['indices']]),
                    non_target_unchanged=torch.equal(after.flatten()[outside], before_copy.flatten()[outside])))
            for r, actual in consumer_inputs:
                r['event']['consumers'].append(dict(operator=name, original_base_object=True,
                    base_value_snapshot_matches_after_call=torch.equal(actual, self._cpu(r['base'])),
                    consumer_result_reference_verified=None))
        return result

    def report(self):
        routed = [e for e in self._events if e['status'] == 'routed']
        eligible = [e for e in self._events if e.get('eligible')]
        return dict(schema='demoncat.alias_auto.v1', mechanism='15.alias_detach', arm=self.arm,
                    format_provider_simulation=bool(self.provider.simulation),
                    candidate_modules=[p for p, _ in self._modules], events=copy.deepcopy(self._events),
                    eligible_calls=len(eligible), routed_calls=len(routed),
                    all_observed_eligible_calls_routed=bool(eligible) and len(routed) == len(eligible)
                        and not self._rejected_root_calls and not any(e['status']=='failed' for e in self._events)
                        and bool(self._forwards) and all(f['status']=='returned' for f in self._forwards),
                    rejected_root_calls=copy.deepcopy(self._rejected_root_calls),
                    forwards=copy.deepcopy(self._forwards),
                    write_observed=any(e['writes'] for e in routed),
                    original_base_consumer_observed=any(e['consumers'] for e in routed),
                    unclassified_base_uses=sum(len(e.get('unclassified_base_uses',[])) for e in self._events),
                    cleanup_errors=list(self._cleanup_errors),
                    mode_cleanup_pending=self._mode_cleanup_uncertain,
                    npu_validation_claimed=False, model_acceptance_claimed=False,
                    limits=['Only public torch.as_strided interception; Tensor/ATen/native internals are not intercepted.',
                            'No observed eligible call is not successful injection. Missing writes/consumers stay unproven.',
                            'Only an explicit set of data-reading operators establishes a base consumer; other uses remain unclassified.',
                            'Dispatch observations are separate from msprobe statistics and do not select its first difference.',
                            'Arbitrary consumer output correctness needs a separate reference; no model quality claim.'])

    def assert_coverage(self):
        report = self.report()
        if not report['all_observed_eligible_calls_routed']:
            raise UnsupportedCall('No complete eligible public as_strided coverage; inspect report')
        return report

    def close(self):
        if self._closed:
            return
        if self._locked and threading.get_ident() != self._owner_thread:
            raise RuntimeError('AliasSession must be closed on its owning thread')
        if getattr(self._tls, 'stack', []) or getattr(self._tls, 'root_entries', []):
            raise RuntimeError('Alias session cannot close inside an active model forward')
        self._active = False
        errors = []
        if self._mode_cleanup_uncertain:
            errors.append(RuntimeError('Dispatch mode cleanup unconfirmed; process restart required'))
        elif self._mode is not None:
            try:
                self._mode.__exit__(None, None, None)
                self._mode = None
            except BaseException as exc:
                errors.append(exc)
                # __exit__ may already have popped the thread dispatch stack.
                # Retrying it could remove a different owner's mode.
                self._mode_cleanup_uncertain = True
        if self._wrapper is not None:
            try:
                import torch
                if torch.as_strided is self._wrapper:
                    torch.as_strided = self._original
                elif torch.as_strided is not self._original:
                    raise RuntimeError('Public torch.as_strided binding changed externally')
                self._wrapper = None
            except BaseException as exc:
                errors.append(exc)
        remaining = []
        for handle in reversed(self._hooks):
            try:
                handle.remove()
            except BaseException as exc:
                errors.append(exc)
                remaining.append(handle)
        self._hooks = list(reversed(remaining))
        if errors:
            self._cleanup_errors.extend(f'{type(error).__name__}: {error}' for error in errors)
            raise RuntimeError('Alias cleanup failed: ' + '; '.join(str(error) for error in errors)) from errors[0]
        self._closed = True
        if self._locked:
            _LEASE.release()
            self._locked = False

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
