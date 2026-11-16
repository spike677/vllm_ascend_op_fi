"""Automatic model ownership for native mechanisms with runtime proof gates.

Module hooks identify ownership only. They do not infer CANN addresses, arm a
buffer proxy, or reconstruct a page table. A version-specific provider must
observe the *actual native boundary* and preserve the existing native checks.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any, Callable

from .core import Decision, UnsupportedCall

_LEASE = threading.Lock()
_STACK = ContextVar('demoncat_native_module_stack', default=())
_ROOTS = ContextVar('demoncat_native_model_roots', default=())

# These are independent contracts, not aliases for one output mutation.
REQUIREMENTS = {
    '20.buffer_substitute': ('abi_handshake', 'semantic_slot', 'live_storage', 'input_transform'),
    '26.residual_stomp': ('abi_handshake', 'semantic_slot', 'live_storage', 'allocation_adjacency', 'consumer_lifetime'),
    '26.fia_destination': ('abi_handshake', 'semantic_slot', 'live_storage', 'writer_destination', 'consumer_lifetime'),
    '33.paged_decode': ('build_identity', 'native_tiling', 'page_table_mapping', 'kv_cache_lifetime', 'request_mapping'),
    '39.matmul_range': ('abi_handshake', 'semantic_slot', 'live_storage', 'finite_input_range'),
    '39.output_proxy': ('abi_handshake', 'semantic_slot', 'live_storage', 'output_proxy_bounds'),
}
OPERATORS = {
    '20.buffer_substitute': {'aclnnQuantMatmulWeightNz', 'aclnnMatmulWeightNz', 'aclnnMatmul'},
    '26.residual_stomp': {'aclnnQuantMatmulWeightNz'},
    '26.fia_destination': {'aclnnFusedInferAttentionScore', 'aclnnFusedInferAttentionScoreV2',
                           'torch_npu.npu_fused_infer_attention_score.out'},
    '33.paged_decode': {'aclnnIncreFlashAttention', 'aclnnFusedInferAttentionScore',
                       'torch_npu.npu_incre_flash_attention', 'torch_npu.npu_fused_infer_attention_score'},
    '39.matmul_range': {'aclnnMatmul'},
    '39.output_proxy': {'aclnnMatmul'},
}
TARGET_KINDS = {
    '20.buffer_substitute': {'linear', 'quantized_linear_candidate'},
    '26.residual_stomp': {'quantized_linear_candidate'},
    '26.fia_destination': {'shared_attention', 'native_attention_candidate'},
    '33.paged_decode': {'shared_attention', 'native_attention_candidate'},
    '39.matmul_range': {'linear'},
    '39.output_proxy': {'linear'},
}


def _module_kinds(module):
    import torch
    kinds = []
    if isinstance(module, torch.nn.Linear):
        kinds.append('linear')
    method, seen = getattr(module, 'quant_method', None), set()
    while method is not None and id(method) not in seen and len(seen) < 8:
        seen.add(id(method))
        if callable(getattr(method, 'apply', None)):
            kinds.append('quantized_linear_candidate')
            break
        method = getattr(method, 'quant_method', None)
    # A concrete native paged implementation object supplies runtime metadata.
    impl = getattr(module, 'impl', None)
    if (impl is not None and callable(getattr(impl, 'forward', None))
            and all(hasattr(module, name) for name in ('num_heads', 'num_kv_heads', 'head_size'))):
        kinds.append('native_attention_candidate')
    from .attention import _resolve
    resolved, _ = _resolve(module)
    if resolved is not None:
        kinds.append('shared_attention')
    return kinds


def _candidates(model):
    if not callable(getattr(model, 'named_modules', None)):
        raise TypeError('model must expose named_modules()')
    rows = []
    for path, module in model.named_modules():
        kinds = _module_kinds(module)
        if kinds:
            rows.append((path or '<root>', module, kinds))
    return rows


def inspect_native(model):
    return {'schema': 'demoncat.native.discovery.v1',
            'candidates': [dict(path=path, type=type(module).__qualname__, kinds=kinds,
                                native_slot_verified=False, lifecycle_verified=False)
                           for path, module, kinds in _candidates(model)],
            'mechanisms': {key: dict(required_proofs=list(value),
                                    integration='native_provider_required', native_validated=False)
                           for key, value in REQUIREMENTS.items()},
            'native_calls_observed': 0,
            'limits': ['Structural candidates do not prove native slots, page maps, allocations or consumers.',
                       'No module-name or leading tensor dimension is interpreted as a request mapping.']}


@dataclass(frozen=True)
class NativeProof:
    """Provider receipt bound to this native invocation, not a user boolean.

    Receipt content is provider-supplied evidence. Hash checks establish byte
    identity and call binding, not independent hardware truth or authenticity.
    """
    kind: str
    path: str
    sha256: str


@dataclass(frozen=True)
class NativeCall:
    owner: Any
    call_id: str
    operator: str
    arguments: tuple = ()
    kwargs: dict = field(default_factory=dict)
    proofs: tuple[NativeProof, ...] = ()


def validate_proofs(call, fault):
    if not isinstance(call.call_id, str) or not call.call_id:
        raise UnsupportedCall('Native call identity is missing')
    if call.operator not in OPERATORS[fault]:
        raise UnsupportedCall(f'{fault} cannot target native operator {call.operator}')
    kinds, receipts = set(), []
    for proof in call.proofs:
        if not isinstance(proof, NativeProof) or proof.kind in kinds:
            raise UnsupportedCall('Duplicate or invalid native proof')
        path = Path(proof.path)
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 8*1024*1024:
            raise UnsupportedCall('Native proof must be a bounded regular file')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != proof.sha256:
            raise UnsupportedCall('Native proof identity changed')
        try:
            receipt = json.loads(data)
        except (ValueError, UnicodeError) as exc:
            raise UnsupportedCall('Native proof is not JSON') from exc
        if not isinstance(receipt, dict) or not (
            receipt.get('schema') == 'demoncat.native.proof.v1'
            and receipt.get('call_id') == call.call_id
            and receipt.get('mechanism') == fault
            and receipt.get('operator') == call.operator
            and receipt.get('owner_id') == id(call.owner) and receipt.get('pid') == os.getpid()
            and receipt.get('kind') == proof.kind
            and receipt.get('status') == 'verified'
            and isinstance(receipt.get('details'), dict) and receipt['details']):
            raise UnsupportedCall('Native proof is stale, mismatched or incomplete')
        kinds.add(proof.kind)
        receipts.append(dict(kind=proof.kind, path=str(path), sha256=proof.sha256))
    missing = set(REQUIREMENTS[fault]) - kinds
    if missing:
        raise UnsupportedCall('Missing native proofs: ' + ', '.join(sorted(missing)))
    return receipts


class NativeSession:
    """Bind an internal native provider to discovered live model objects.

    Provider protocol:
      install(callback, candidates, mechanism, arm) -> removal callable
      check(call, mechanism, arm) -> Decision
      execute(call, invoke_once, mechanism, arm) -> original model-visible result
      finish(call, mechanism, arm) -> native observation receipt
      abort(call, mechanism, arm) -> idempotently release this call's native state

    install must be inert for writes and invoke callback only at the real native
    boundary. execute must retain the legacy ABI/storage/tiling checks and undo
    its own control state on error. The Python bridge never guesses addresses.
    """
    def __init__(self, model, fault, runtime=None, arm='fault', unsupported='error', evidence_dir=None):
        if fault not in REQUIREMENTS:
            raise ValueError('No native bridge contract for ' + str(fault))
        if arm not in ('baseline', 'fault') or unsupported not in ('error', 'passthrough'):
            raise ValueError('Invalid arm or unsupported policy')
        self.model, self.fault, self.runtime = model, fault, runtime
        self.arm, self.unsupported, self.evidence_dir = arm, unsupported, evidence_dir
        self._candidates = [row for row in _candidates(model) if set(row[2]) & TARGET_KINDS[fault]]
        self._by_id = {id(m): (path, m, kinds) for path, m, kinds in self._candidates}
        self._handles, self._events, self._seen = [], [], set()
        self._forwards, self._invocations, self._rejected_roots = [], [], []
        self._invocation_exceptions = {}
        self._root_entries, self._cleanup_errors = [], []
        self._owner = self._forward_index = None
        self._root_active = False
        self._remove = None
        self._active = self._locked = self._closed = False
        self._cleanup_retryable = True
        self._closing = False
        self._pending_aborts = []
        self._installation_unknown = False

    def __enter__(self):
        if self._active or self._closed:
            raise RuntimeError('Session cannot be reentered')
        if self.runtime is None:
            raise UnsupportedCall(f'{self.fault}: native provider not adapted for this deployment; '
                                  'module discovery cannot replace ABI, storage or page-table proof')
        for name in ('install', 'check', 'execute', 'finish', 'abort'):
            if not callable(getattr(self.runtime, name, None)):
                raise UnsupportedCall('Incomplete native provider: ' + name)
        if not self._candidates:
            raise UnsupportedCall('No native structural candidates discovered')
        if not _LEASE.acquire(blocking=False):
            raise UnsupportedCall('Another native session owns the process boundary')
        self._locked = True
        self._owner = threading.get_ident()
        try:
            # Enter before pre-existing local pre-hooks and leave after the
            # candidate hooks, including when the root is itself a candidate.
            self._handles.append(self.model.register_forward_pre_hook(self._enter_root, prepend=True))
            for _path, module, _kinds in self._candidates:
                self._handles.append(module.register_forward_pre_hook(self._enter_module))
                self._handles.append(module.register_forward_hook(self._exit_module, always_call=True))
            self._handles.append(self.model.register_forward_hook(self._exit_root, always_call=True))
            self._installation_unknown = True
            self._remove = self.runtime.install(self._dispatch, tuple(self._candidates), self.fault, self.arm)
            if not callable(self._remove):
                raise UnsupportedCall('Native provider did not supply an ownership-scoped removal function')
            self._installation_unknown = False
            self._active = True
            return self
        except BaseException:
            self.close()
            raise

    def _enter_root(self, module, args):
        if self._closing:
            raise UnsupportedCall('Native cleanup is pending; model execution is quarantined')
        if not self._active:
            return
        if threading.get_ident() != self._owner:
            self._rejected_roots.append(dict(reason='Concurrent root call', thread=threading.get_ident()))
            raise UnsupportedCall('Concurrent or recursive native model execution is unsupported')
        if self._root_active:
            # always_call hooks also run after this pre-hook raises. Its marker
            # prevents that rejected invocation from popping the outer scope.
            self._root_entries.append(None)
            self._rejected_roots.append(dict(reason='Recursive root call', forward=self._forward_index))
            raise UnsupportedCall('Concurrent or recursive native model execution is unsupported')
        self._forward_index = len(self._forwards)
        self._forwards.append(dict(index=self._forward_index, status='running', errors=[]))
        self._root_entries.append(dict(index=self._forward_index, roots=_ROOTS.get(), stack=_STACK.get(),
                                       exception=sys.exc_info()[1]))
        self._root_active = True
        _ROOTS.set(_ROOTS.get() + (id(self),))

    def _exit_root(self, module, args, output):
        if not self._active or threading.get_ident() != self._owner or not self._root_entries:
            return
        entry = self._root_entries.pop()
        if entry is None:
            return
        forward = self._forwards[entry['index']]
        current_error = sys.exc_info()[1]
        failed = current_error is not entry['exception']
        forward['status'] = 'returned' if output is not None and not failed else 'no_output_or_exception'
        if failed:
            forward['errors'].append(f'{type(current_error).__name__}: {current_error}')
        if _ROOTS.get() != entry['roots'] + (id(self),):
            forward['errors'].append('Native model root ownership changed unexpectedly')
        if _STACK.get() != entry['stack']:
            forward['errors'].append('Unclosed native candidate scope at root exit')
        # Always restore the accepted outer invocation, even if a module hook
        # or provider raised. A later forward cannot inherit its scope.
        _ROOTS.set(entry['roots'])
        _STACK.set(entry['stack'])
        for invocation in self._invocations:
            if invocation['forward'] == entry['index']:
                self._invocation_exceptions.pop(invocation['index'], None)
        self._root_active = False
        self._forward_index = None

    def _enter_module(self, module, args):
        if not self._in_root():
            return
        index = len(self._invocations)
        self._invocations.append(dict(index=index, path=self._by_id[id(module)][0],
                                      forward=self._forward_index, status='running'))
        self._invocation_exceptions[index] = sys.exc_info()[1]
        _STACK.set(_STACK.get() + ((id(self), module, index),))

    def _exit_module(self, module, args, output):
        if not self._in_root():
            return
        stack = _STACK.get()
        if not stack or stack[-1][:2] != (id(self), module):
            self._forwards[self._forward_index]['errors'].append('Native candidate scope did not close in order')
            return
        index = stack[-1][2]
        failed = sys.exc_info()[1] is not self._invocation_exceptions.pop(index, None)
        self._invocations[index]['status'] = 'returned' if output is not None and not failed else 'no_output_or_exception'
        _STACK.set(stack[:-1])

    def _in_root(self):
        return (self._active and threading.get_ident() == self._owner and self._root_active
                and self._root_entries and self._root_entries[-1] is not None
                and id(self) in _ROOTS.get())

    def _dispatch(self, call, invoke_original):
        if not callable(invoke_original):
            if self._in_root():
                self._forwards[self._forward_index]['errors'].append('Missing actual original invocation')
            raise TypeError('Actual original invocation is required')
        if not isinstance(call, NativeCall):
            if self._in_root():
                self._forwards[self._forward_index]['errors'].append('Provider did not report a typed actual call')
            raise UnsupportedCall('Native provider did not report a typed actual call')
        entry = self._by_id.get(id(call.owner))
        if (not self._in_root() or entry is None or entry[1] is not call.owner
                or not _STACK.get() or _STACK.get()[-1][:2] != (id(self), call.owner)):
            return invoke_original()
        path = entry[0]
        event = dict(path=path, call_id=call.call_id, operator=call.operator,
                     forward=self._forward_index, module_invocation=_STACK.get()[-1][2],
                     status='observed', reason=None, actual_original_calls=0,
                     native_hit_proven=False, native_launch_verified=False)
        self._events.append(event)
        if call.call_id in self._seen:
            event.update(status='failed', reason='Native call identity reused')
            raise UnsupportedCall(event['reason'])
        self._seen.add(call.call_id)
        def once():
            if event['actual_original_calls']:
                raise UnsupportedCall('Provider attempted to execute the original more than once')
            event['actual_original_calls'] += 1
            return invoke_original()
        execution_started = False
        try:
            decision = self.runtime.check(call, self.fault, self.arm)
            if not isinstance(decision, Decision):
                raise UnsupportedCall('Native provider check must return Decision')
            if not decision.supported:
                event.update(status='unsupported', reason=decision.reason)
                if self.unsupported == 'error':
                    raise UnsupportedCall(decision.reason)
                result = once()
                event['status'] = 'passthrough'
                return result
            event['proofs'] = validate_proofs(call, self.fault)
            execution_started = True
            result = self.runtime.execute(call, once, self.fault, self.arm)
            if event['actual_original_calls'] != 1:
                raise UnsupportedCall('Provider did not execute the original exactly once')
            observation = self.runtime.finish(call, self.fault, self.arm)
            if not isinstance(observation, dict) or observation.get('call_id') != call.call_id:
                raise UnsupportedCall('Native provider returned a missing or mismatched observation')
            event.update(status='routed', provider_observation=observation,
                         provider_scope=getattr(self.runtime, 'validation_scope', 'unspecified'))
            # A provider's assertion is retained verbatim, not upgraded to an
            # independent launch or physical-root-cause certificate here.
            return result
        except BaseException as exc:
            if event['status'] == 'observed':
                event.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
            if execution_started:
                try:
                    self.runtime.abort(call, self.fault, self.arm)
                    event['abort_completed'] = True
                except BaseException as cleanup:
                    event['abort_completed'] = False
                    event['cleanup_error'] = f'{type(cleanup).__name__}: {cleanup}'
                    self._pending_aborts.append((call, event))
                    self._closing = True
                    raise RuntimeError('Native execution and scoped cleanup failed: ' + str(cleanup)) from exc
            raise

    def report(self):
        modules = {path: dict(candidate_kinds=kinds,
                             observed_calls=sum(e['path'] == path for e in self._events),
                             routed_calls=sum(e['path'] == path and e['status'] == 'routed' for e in self._events))
                   for path, _module, kinds in self._candidates}
        invocations = []
        for invocation in self._invocations:
            current = [e for e in self._events if e['module_invocation'] == invocation['index']]
            invocations.append(dict(invocation, observed_calls=len(current),
                                    routed_calls=sum(e['status'] == 'routed' for e in current),
                                    all_observed_calls_routed=(invocation['status'] == 'returned'
                                        and bool(current) and all(e['status'] == 'routed' for e in current))))
        forwards = []
        for forward in self._forwards:
            current = [i for i in invocations if i['forward'] == forward['index']]
            by_path = {path: [i for i in current if i['path'] == path] for path in modules}
            missing = [path for path, calls in by_path.items() if not calls]
            unobserved = [path for path, calls in by_path.items()
                          if not calls or any(not i['observed_calls'] for i in calls)]
            covered = (forward['status'] == 'returned' and not forward['errors']
                       and bool(current) and not missing
                       and all(i['all_observed_calls_routed'] for i in current))
            forwards.append(dict(forward, candidate_invocations={p: len(c) for p, c in by_path.items()},
                                 unexecuted_candidates=missing, unobserved_candidates=unobserved,
                                 all_discovered_candidates_covered=covered))
        complete = (bool(forwards) and not self._rejected_roots and not self._cleanup_errors
                    and all(f['all_discovered_candidates_covered'] for f in forwards))
        return dict(schema='demoncat.native.coverage.v1', mechanism=self.fault,
                    active=self._active, candidates=modules, events=copy.deepcopy(self._events),
                    forwards=copy.deepcopy(forwards), module_invocations=invocations,
                    rejected_root_calls=copy.deepcopy(self._rejected_roots),
                    cleanup_errors=list(self._cleanup_errors),
                    cleanup_pending=self._closing, pending_native_aborts=len(self._pending_aborts),
                    installation_cleanup_unknown=self._installation_unknown, closed=self._closed,
                    all_discovered_candidates_covered=complete,
                    all_observed_supported_calls_routed=bool(self._events) and all(e['status'] == 'routed' for e in self._events),
                    unobserved_candidates=[p for p, row in modules.items() if not row['observed_calls']],
                    native_fault_hit_proven=False, npu_validation_claimed=False,
                    limits=['Provider evidence is call-bound but not independently certified by this bridge.',
                            'No native callbacks is not successful injection or complete model coverage.',
                            'Every accepted forward and every candidate invocation needs observed routed callbacks.',
                            'Inactive dynamic branches remain unexecuted, not covered by a previous forward.',
                            'A None model/module result is conservatively no-output-or-exception, not a coverage pass.',
                            'Structural candidates may be unrelated to the selected native mechanism.'])

    def assert_coverage(self):
        report = self.report()
        if not report['all_discovered_candidates_covered']:
            raise UnsupportedCall('Native coverage incomplete; see unobserved/unsupported/failed calls')
        return report

    def close(self):
        if self._locked and threading.get_ident() != self._owner:
            raise RuntimeError('Native session must be closed on its owning thread')
        if self._root_active:
            raise RuntimeError('Native session cannot close inside an active model forward')
        self._closing = True
        if self._installation_unknown:
            raise RuntimeError('Native provider installation did not return owned removal; cleanup unknown, process restart required')
        errors = []
        pending = []
        for call, event in self._pending_aborts:
            try:
                self.runtime.abort(call, self.fault, self.arm)
                # Retain the original failure; recovery is a separate fact.
                event['abort_recovery_completed'] = True
            except BaseException as exc:
                pending.append((call, event))
                errors.append(exc)
        self._pending_aborts = pending
        if errors:
            self._cleanup_errors.extend(f'{type(e).__name__}: {e}' for e in errors)
            raise RuntimeError('Native scoped cleanup failed; owner retry required: '+ '; '.join(str(e) for e in errors)) from errors[0]
        if self._remove is not None:
            try:
                self._remove()
            except BaseException as exc:
                # Provider removal may be a synchronous native drain. Preserve
                # the capability and hooks until it confirms successful return.
                self._cleanup_errors.append(f'{type(exc).__name__}: {exc}')
                raise RuntimeError('Native cleanup failed; owner retry required: '+str(exc)) from exc
            self._remove = None
        failed_handles = []
        for handle in reversed(self._handles):
            try:
                handle.remove()
            except BaseException as exc:
                errors.append(exc)
                failed_handles.append(handle)
        self._handles = list(reversed(failed_handles))
        if errors:
            self._cleanup_errors.extend(f'{type(e).__name__}: {e}' for e in errors)
            raise RuntimeError('Native cleanup failed: ' + '; '.join(str(e) for e in errors)) from errors[0]
        self._active, self._closed, self._closing = False, True, False
        self._root_entries.clear()
        self._invocation_exceptions.clear()
        self._forward_index = None
        _STACK.set(tuple(row for row in _STACK.get() if row[0] != id(self)))
        _ROOTS.set(tuple(row for row in _ROOTS.get() if row != id(self)))
        if self._locked:
            _LEASE.release()
            self._locked = False

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if exc_value is not None:
                raise RuntimeError('Native scope failed: '+repr(exc_value)+'; restore failed: '+repr(cleanup)) from exc_value
            raise
        return False
