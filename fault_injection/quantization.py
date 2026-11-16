"""Automatic dynamic-quant module binding; native arithmetic stays in the old backend.

Model structure is only discovery. A deployment provider must supply the real
scheduler mapping and independently established healthy-input contract. A
Linear's leading dimension is never interpreted as request count by itself.
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any

from .core import UnsupportedCall

_LEASE = threading.Lock()
_VERIFIED_LOAD = object()
_CONTROL_SHA = '5ce739c4272840f110044cfe898827fe403b5730273258e8ba0eda437685b38c'
_NATIVE_SHA = '895e635d4674dae95c03869bb9da4d66617519ea056f5e2b4c671bf7e3c9675c'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _chain(module):
    method = getattr(module, 'quant_method', None)
    names, seen = [], set()
    while method is not None and len(names) < 8 and id(method) not in seen:
        seen.add(id(method))
        names.append(type(method).__module__ + '.' + type(method).__name__)
        if type(method).__name__ == 'AscendW8A8DynamicLinearMethod':
            return method, names
        method = getattr(method, 'quant_method', None)
    return None, names


def _discover(model):
    if not callable(getattr(model, 'named_modules', None)):
        raise TypeError('model must expose named_modules()')
    candidates, rejected = [], []
    for path, module in model.named_modules():
        if not hasattr(module, 'quant_method'):
            continue
        inner, chain = _chain(module)
        reason = None
        if inner is None:
            reason = 'No recognized dynamic W8A8 method in actual wrapper chain'
        elif not callable(getattr(getattr(module, 'quant_method', None), 'apply', None)):
            reason = 'Outer quantization method has no callable apply'
        elif str(getattr(getattr(module, 'weight', None), 'dtype', None)) != 'torch.int8':
            reason = 'Dynamic W8A8 candidate requires actual INT8 weight'
        elif type(getattr(module, 'tp_size', None)) is not int or type(getattr(module, 'tp_rank', None)) is not int:
            reason = 'Tensor-parallel size/rank metadata unavailable'
        elif module.tp_size != 2 or module.tp_rank not in (0, 1):
            reason = 'Old native contract is TP2; other parallel layouts remain unvalidated'
        if reason:
            rejected.append(dict(path=path or '<root>', method_chain=chain, reason=reason))
        else:
            candidates.append((path or '<root>', module, module.quant_method, chain))
    return candidates, rejected


def inspect_quantization(model):
    candidates, rejected = _discover(model)
    return dict(schema='demoncat.dynamic_quant.discovery.v1',
        candidates=[dict(path=p, method_chain=c, tp_size=m.tp_size, tp_rank=m.tp_rank,
                         status='candidate_not_observed') for p,m,_,c in candidates],
        unsupported_modules=rejected, scheduler_mapping_proven=False,
        native_binding_proven=False, new_npu_validation_claimed=False)


@dataclass(frozen=True)
class QuantizationCall:
    path: str
    module: Any
    input: Any
    model: Any
    forward_index: int


def _validate_rows(call, context):
    """Check actual mapping supplied by the runner, not inferred batch labels."""
    x = call.input
    if not isinstance(context, dict) or context.get('model') is not call.model:
        raise UnsupportedCall('Runner context must bind the actual model object')
    if (type(context.get('rank')) is not int or type(context.get('world_size')) is not int
            or context['rank'] != call.module.tp_rank or context['world_size'] != 2):
        raise UnsupportedCall('Runner TP/rank disagrees with the actual module')
    ids, counts = context.get('request_ids'), context.get('counts')
    rows, positions = context.get('row_request_indices'), context.get('row_token_positions')
    if (not isinstance(ids, list) or not ids or any(not isinstance(v, str) or not v for v in ids)
            or len(set(ids)) != len(ids) or not isinstance(counts,list)
            or any(type(n) is not int for n in counts) or counts != [1] * len(ids)
            or not isinstance(rows,list) or any(type(n) is not int for n in rows)
            or rows != list(range(len(ids))) or context.get('pure_decode') is not True):
        raise UnsupportedCall('Need verified one-row-per-request pure decode mapping')
    if (len(ids) != x.shape[0] or not isinstance(positions, list) or len(positions) != len(ids)
            or any(type(v) is not int or v < 0 for v in positions)):
        raise UnsupportedCall('Actual tensor rows do not match scheduler mapping')
    digest = context.get('baseline_input_sha256')
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in '0123456789abcdef' for c in digest)
            or not isinstance(context.get('healthy_run_id'), str) or not context['healthy_run_id']):
        raise UnsupportedCall('Independent healthy input identity is required')
    return dict(request_ids=list(ids), row_token_positions=list(positions), rank=context['rank'],
                healthy_run_id=context['healthy_run_id'], baseline_input_sha256=digest)


class PinnedQuantRuntime:
    """Deployment bridge to unchanged BackendControl, never a numeric proxy.

    context_provider(call) is installed by the runner integration and must
    certify real scheduler rows and a prior healthy input. It receives automatic
    object/path bindings; end users do not provide layer names. The existing
    provider remains a required deployment component, not fabricated metadata.
    """
    native = True
    test_only = False

    def __init__(self, backend, context_provider, build_identity, *, _load_token=None):
        if _load_token is not _VERIFIED_LOAD:
            raise UnsupportedCall('Construct PinnedQuantRuntime through the verified native loader')
        if not callable(context_provider) or not all(getattr(backend, 'resolver_checks', {}).values()):
            raise UnsupportedCall('Verified backend and live runner context provider required')
        if set(getattr(backend, 'resolver_checks', {})) != {
                'aclnnDynamicQuantV2GetWorkspaceSize','aclnnDynamicQuantV2','CanUsePTACache'}:
            raise UnsupportedCall('Backend resolver ABI evidence incomplete')
        self.backend, self.context_provider = backend, context_provider
        self.build_identity, self.poisoned = build_identity, False
        self._last_verified_receipt = None

    @classmethod
    def from_existing_backend(cls, source_root, *, context_provider, approved_build_sha256):
        root = Path(source_root).resolve(strict=True)
        control, source, build = root/'backend_control.py', root/'native_backend.cpp', root/'runtime/build.json'
        if _sha(control) != _CONTROL_SHA or _sha(source) != _NATIVE_SHA or _sha(build) != approved_build_sha256:
            raise UnsupportedCall('Existing quantization source/build identity differs from deployment approval')
        data = json.loads(build.read_bytes())
        library = root/'runtime/vendor/op_api/lib/libcust_opapi.so'
        expected = data.get('files', {})
        if (expected.get('native_backend.cpp') != _NATIVE_SHA or
                expected.get('runtime/vendor/op_api/lib/libcust_opapi.so') != _sha(library)):
            raise UnsupportedCall('Native library differs from its approved build receipt')
        if os.environ.get('DCAT_DQ_V2_ENABLE') != '1':
            raise UnsupportedCall('Start a dedicated process through the existing native vendor launcher')
        spec = importlib.util.spec_from_file_location('_demoncat_pinned_quant_control', control)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return cls(module.BackendControl(), context_provider, dict(build_sha256=approved_build_sha256,
                   library_sha256=_sha(library), control_sha256=_CONTROL_SHA), _load_token=_VERIFIED_LOAD)

    def context(self, call):
        if self.poisoned:
            raise UnsupportedCall('Previous native failure poisoned this runtime; use a fresh process')
        if str(call.input.device).split(':')[0] != 'npu':
            raise UnsupportedCall('Native DynamicQuant requires NPU tensors; no CPU fallback')
        self.backend.validate_input(call.input)
        return self.context_provider(call)

    @staticmethod
    def input_sha256(x):
        import torch
        return hashlib.sha256(x.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()

    def arm(self, call, fault):
        self._last_verified_receipt = None
        try:
            self.backend.arm_validated(call.input, fault=fault)
        except BaseException:
            # The session will attempt finish even if arm raised after native
            # state changed. No subsequent call may reuse this uncertain state.
            self.poisoned = True
            raise

    @staticmethod
    def _valid_counts(record, fault):
        expected = dict(prepare_hits=1, execute_hits=1, fault_executions=int(fault),
                        api_version=2, prepare_errors=0, execution_errors=0)
        return (isinstance(record, dict) and record == expected
                and all(type(value) is int for value in record.values()))

    def finish(self, *, fault, failed=False):
        try:
            record = self.backend.finish()
            if failed or not self._valid_counts(record, fault):
                raise UnsupportedCall('Native quantization did not execute exactly one matching V2 call')
            self._last_verified_receipt = record
            return record
        except BaseException:
            self.poisoned = True
            raise

    def proves_native_receipt(self, record, *, fault):
        return (not self.poisoned and record is self._last_verified_receipt
                and self._valid_counts(record, fault))


class QuantizationSession:
    _cleanup_retryable = True

    def __init__(self, model, arm='fault', unsupported='error', evidence_dir=None, runtime=None):
        if arm not in ('baseline','fault') or unsupported not in ('error','passthrough'):
            raise ValueError('Invalid arm or unsupported policy')
        self.model, self.arm, self.unsupported = model, arm, unsupported
        # Runner/deployment installs the provider. No model path configuration.
        self.runtime = runtime if runtime is not None else getattr(model, '_demoncat_quant_runtime', None)
        self.candidates, self.rejected = _discover(model)
        self.events, self._patches, self._hooks = [], [], []
        self._local, self._active, self._locked, self._closed = threading.local(), False, False, False
        self._forward_index = -1
        self._owner = None
        self._poisoned = False
        self._native_state_uncertain = False
        self._cleanup_errors = []
        self._forwards, self._root_entries, self._rejected_roots = [], [], []

    def __enter__(self):
        if self._active or self._closed:
            raise RuntimeError('Session cannot be reused')
        if not self.candidates:
            raise UnsupportedCall('No dynamic W8A8 quantization candidates in this model')
        if self.rejected and self.unsupported == 'error':
            raise UnsupportedCall('Unsupported quantization modules: ' + ', '.join(r['path'] for r in self.rejected))
        if self.runtime is None:
            raise UnsupportedCall('Quantization runner/native provider not installed; discovery alone cannot arm a fault')
        for name in ('context','input_sha256','arm','finish'):
            if not callable(getattr(self.runtime, name, None)):
                raise UnsupportedCall('Incomplete runtime provider: ' + name)
        if not (getattr(self.runtime,'native',False) is True or getattr(self.runtime,'test_only',False) is True):
            raise UnsupportedCall('Provider must declare native or explicit CPU-test-only mode')
        if getattr(self.runtime,'native',False) is True and getattr(self.runtime,'test_only',False) is True:
            raise UnsupportedCall('A CPU test provider cannot simultaneously claim native execution')
        if getattr(self.runtime,'native',False) is True and (threading.current_thread() is not threading.main_thread()
                                                          or threading.active_count() != 1):
            raise UnsupportedCall('Native process-scoped backend requires a dedicated single-request process')
        if not _LEASE.acquire(blocking=False):
            raise RuntimeError('Another quantization session owns the process scope')
        self._locked, self._owner = True, threading.get_ident()
        try:
            by_module = {id(m):(p,m,method) for p,m,method,_ in self.candidates}
            groups = {id(method):method for _,_,method,_ in self.candidates}
            for method in groups.values():
                original = method.apply
                had_local = 'apply' in vars(method)
                saved_local = vars(method).get('apply')
                @functools.wraps(original)
                def wrapped(module, *args, _original=original, _method=method, **kwargs):
                    entry = by_module.get(id(module))
                    if not self._active or not getattr(self._local, 'inside', False) or entry is None:
                        return _original(module, *args, **kwargs)
                    if (threading.get_ident() != self._owner or entry[1] is not module or entry[2] is not _method
                            or getattr(module,'quant_method',None) is not _method):
                        raise UnsupportedCall('Quantization object or thread scope mismatch')
                    if getattr(self._local, 'quant_active', False):
                        raise UnsupportedCall('Nested quantization arm scopes are unsupported')
                    x = args[0] if args else kwargs.get('x')
                    return self._execute(QuantizationCall(entry[0],module,x,self.model,self._forward_index),
                                         lambda: _original(module,*args,**kwargs))
                self._patches.append((method, original, wrapped, had_local, saved_local))
                method.apply = wrapped
            self._hooks.append(self.model.register_forward_pre_hook(self._enter_forward, prepend=True))
            self._hooks.append(self.model.register_forward_hook(self._leave_forward, always_call=True))
            self._active = True
            return self
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                raise RuntimeError('Quantization installation failed: '+repr(error)+'; cleanup failed: '+repr(cleanup)) from error
            raise

    def _enter_forward(self, module, args):
        if not self._active:
            return
        if threading.get_ident() != self._owner:
            self._rejected_roots.append(dict(reason='Concurrent root call', thread=threading.get_ident()))
            raise UnsupportedCall('Concurrent/reentrant model execution is unsupported')
        if getattr(self._local,'inside',False):
            self._root_entries.append(None)
            self._rejected_roots.append(dict(reason='Recursive root call', forward=self._forward_index))
            raise UnsupportedCall('Concurrent/reentrant model execution is unsupported')
        self._forward_index = len(self._forwards)
        self._forwards.append(dict(index=self._forward_index, status='running'))
        self._root_entries.append((self._forward_index, sys.exc_info()[1]))
        self._local.inside = True

    def _leave_forward(self, module, args, output):
        if not self._active or threading.get_ident() != self._owner or not self._root_entries:
            return
        entry = self._root_entries.pop()
        if entry is None:
            return
        index, entering_exception = entry
        failed = sys.exc_info()[1] is not entering_exception
        self._forwards[index]['status'] = 'returned' if output is not None and not failed else 'no_output_or_exception'
        self._local.inside = False

    def _execute(self, call, original):
        event = dict(path=call.path, forward_index=call.forward_index, status='observed',
                     provider_native_declared=getattr(self.runtime,'native',False) is True,
                     native_execution_proven=False, native_hit_proven=False)
        self.events.append(event)
        x = call.input
        try:
            if self._poisoned:
                raise RuntimeError('Quantization session poisoned by a previous arm/execution failure; create a fresh process')
            import torch
            if (str(getattr(getattr(call.module,'weight',None),'dtype',None)) != 'torch.int8'
                    or call.module.tp_size != 2 or call.module.tp_rank not in (0,1)):
                raise UnsupportedCall('Actual quantization weight or parallel contract changed')
            if not isinstance(x,torch.Tensor) or x.ndim != 2 or not x.is_contiguous() or x.storage_offset() != 0:
                raise UnsupportedCall('Expected actual contiguous zero-offset [T,H] tensor')
            if x.dtype not in (torch.float16,torch.bfloat16) or not (1 <= x.shape[0] <= 64 and
                    1 <= x.shape[1] <= 65536 and x.numel() <= 1048576):
                raise UnsupportedCall('Input exceeds old native dtype/shape contract')
            if not bool(torch.isfinite(x).all()) or not bool((x.float().abs().amax(dim=-1)>0).all()):
                raise UnsupportedCall('Nonfinite input or zero rows are unsupported')
            rows = _validate_rows(call, self.runtime.context(call))
            before = self.runtime.input_sha256(x)
            if before != rows['baseline_input_sha256']:
                raise UnsupportedCall('Current input differs from independently certified healthy input')
            event.update(shape=list(x.shape), dtype=str(x.dtype), row_contract=rows, input_before=before)
        except UnsupportedCall as exc:
            event.update(status='unsupported',reason=str(exc))
            if self.unsupported == 'passthrough':
                event['status'] = 'passthrough'
                return original()
            raise
        except BaseException as exc:
            event.update(status='failed',reason=repr(exc))
            raise
        self._local.quant_active = True
        armed = False
        try:
            try:
                # A native arm may change state before its reply is lost.
                self._native_state_uncertain = getattr(self.runtime, 'native', False) is True
                self.runtime.arm(call, fault=self.arm == 'fault')
            except BaseException as error:
                self._poisoned = True
                try:
                    self.runtime.finish(fault=self.arm == 'fault', failed=True)
                    self._native_state_uncertain = False
                except BaseException as cleanup:
                    raise RuntimeError('Quantization arm failed: '+repr(error)+'; cleanup failed: '+repr(cleanup)) from error
                raise
            armed = True
            try:
                output = original()
            except BaseException as error:
                try:
                    self.runtime.finish(fault=self.arm == 'fault', failed=True)
                    self._native_state_uncertain = False
                except BaseException as cleanup:
                    raise RuntimeError('Original quantization call failed: '+repr(error)+'; native cleanup failed: '+repr(cleanup)) from error
                raise
            receipt = self.runtime.finish(fault=self.arm == 'fault', failed=False)
            self._native_state_uncertain = False
            after = self.runtime.input_sha256(x)
            if after != before:
                raise UnsupportedCall('Native selected quantization unexpectedly changed its input')
            # A provider's declaration is not a launch/execution proof. Only
            # this pinned loader and its validated, identity-bound counters
            # qualify; all other provider receipts stay explicit observations.
            proven = (type(self.runtime) is PinnedQuantRuntime and
                      self.runtime.proves_native_receipt(receipt, fault=self.arm == 'fault'))
            event.update(status='routed',input_after=after,native_receipt=receipt,
                         native_execution_proven=proven,
                         native_hit_proven=proven and self.arm == 'fault',
                         test_only=getattr(self.runtime,'test_only',False) is True)
            return output
        except BaseException as exc:
            self._poisoned = True
            event.update(status='failed', armed=armed, reason=repr(exc))
            raise
        finally:
            self._local.quant_active = False

    def report(self):
        modules = {}
        for path,_,_,_ in self.candidates:
            events = [e for e in self.events if e['path']==path]
            modules[path] = dict(observed_calls=len(events), routed_calls=sum(e['status']=='routed' for e in events),
                                 native_proven_calls=sum(e['native_hit_proven'] for e in events))
        per_forward = {str(index): {path: sum(e['forward_index']==index and e['path']==path and e['status']=='routed'
                                             for e in self.events) for path in modules}
                       for index in range(self._forward_index+1)}
        return dict(schema='demoncat.dynamic_quant.coverage.v1', modules=modules, events=list(self.events),
                    session_poisoned=self._poisoned,
                    native_cleanup_pending=self._native_state_uncertain,
                    cleanup_errors=list(self._cleanup_errors),
                    per_forward=per_forward,
                    forwards=[dict(row) for row in self._forwards],
                    rejected_root_calls=[dict(row) for row in self._rejected_roots],
                    unsupported_modules=self.rejected,
                    all_candidates_routed=bool(modules) and not self.rejected and all(
                        v['observed_calls']>0 and v['observed_calls']==v['routed_calls'] for v in modules.values())
                        and not self._rejected_roots and not self._poisoned
                        and bool(per_forward) and all(all(count>0 for count in row.values()) for row in per_forward.values())
                        and all(row['status']=='returned' for row in self._forwards),
                    new_npu_validation_claimed=False,
                    limits=['Provider supplies true scheduler mapping; tensor shape alone is insufficient.',
                            'Changed inputs at later layers may fail the prior healthy-input contract; they are not silently admitted.',
                            'CPU-test-only routing is not native injection; historical Qwen acceptance is not transferred.'])

    def assert_coverage(self):
        report = self.report()
        if not report['all_candidates_routed']:
            raise UnsupportedCall('Quantization coverage incomplete; inspect report()')
        return report

    def close(self):
        if self._closed:
            return
        if self._locked and threading.get_ident() != self._owner:
            raise RuntimeError('Quantization session must be closed on its owning thread')
        if (self._root_entries or getattr(self._local, 'inside', False)
                or getattr(self._local, 'quant_active', False)):
            raise RuntimeError('Quantization session cannot close inside an active model forward')
        self._active = False
        errors = []
        remaining = []
        for hook in reversed(self._hooks):
            try: hook.remove()
            except BaseException as exc:
                errors.append(str(exc))
                remaining.append(hook)
        self._hooks = list(reversed(remaining))
        remaining = []
        for method, original, wrapper, had_local, saved in reversed(self._patches):
            try:
                if getattr(method,'apply',None) is wrapper:
                    if had_local:
                        method.apply = saved
                    else:
                        delattr(method,'apply')
                elif not ((had_local and vars(method).get('apply') is saved)
                          or (not had_local and 'apply' not in vars(method)
                              and getattr(method, 'apply', None) == original)):
                    raise RuntimeError('Quantization apply replaced by another owner')
            except BaseException as exc:
                errors.append(str(exc))
                remaining.append((method, original, wrapper, had_local, saved))
        self._patches = list(reversed(remaining))
        if self._native_state_uncertain:
            # The legacy provider has no independently acknowledged recovery
            # API. Restoring Python wrappers does not prove native disarm.
            errors.append('Native quantization cleanup unconfirmed; process restart required')
        if errors:
            self._cleanup_errors.extend(errors)
            raise RuntimeError('; '.join(errors))
        self._closed = True
        if self._locked:
            _LEASE.release(); self._locked = False

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if exc_value is not None:
                raise RuntimeError('Quantization scope failed: '+repr(exc_value)+'; restore failed: '+repr(cleanup)) from exc_value
            raise
        return False
