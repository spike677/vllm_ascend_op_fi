"""Bounded observation window installed on real modules, not an RPC-thread mode."""
from __future__ import annotations

import copy
import hashlib
import json
import threading
import uuid
from collections import Counter

from .surface import ModuleSurfaceAdapter, is_tensor, tensor_paths
from .quantization import QuantizationResolver


def page(records, offset=0, limit=50, max_bytes=262144):
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 200:
        raise ValueError('INVALID_PAGE')
    selected, size = [], 0
    for row in records[offset:offset + limit]:
        amount = len(json.dumps(row, ensure_ascii=False, allow_nan=False).encode())
        if size + amount > max_bytes:
            if not selected:
                raise ValueError('RECORD_EXCEEDS_RPC_BUDGET; narrow target metadata')
            break
        selected.append(row)
        size += amount
    next_offset = offset + len(selected)
    return copy.deepcopy({'items': selected, 'total': len(records),
                           'next_offset': next_offset if next_offset < len(records) else None})


class InspectSession:
    def __init__(self, model, ranks, runtime, *, epoch=None, predicate=is_tensor, quant_contracts=()):
        self.epoch = epoch or uuid.uuid4().hex
        self.runtime = copy.deepcopy(runtime)
        self.fingerprint = hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()
        self.surface = ModuleSurfaceAdapter(model, ranks, predicate)
        self.quant = QuantizationResolver(quant_contracts)
        self.lock = threading.RLock()
        self.handles, self.events = [], []
        self.active = False
        self.timer = None
        self.session_id = None
        self.window = None
        self.counts = Counter()
        self.stop_reason = 'NOT_STARTED'
        self.terminal_stop_reason = None
        self.errors = []
        self.selected_names = []

    def check_epoch(self, epoch):
        if epoch != self.epoch:
            raise ValueError('STALE_SERVICE_EPOCH')

    def status(self):
        with self.lock:
            return {'service_epoch': self.epoch, 'runtime_fingerprint': self.fingerprint,
                    'runtime': copy.deepcopy(self.runtime), 'ranks': self.surface.ranks,
                    'owned_layers': self.surface.owned_layers, 'active': self.active,
                    'session': self.session_id, 'event_count': len(self.events),
                    'installed_handles': len(self.handles), 'stop_reason': self.stop_reason,
                    'terminal_stop_reason': self.terminal_stop_reason,
                    'module_call_counts': dict(self.counts),
                    'enumeration_complete': True,
                    'observation_status': 'OBSERVED' if self.events else
                        'HOOK_INSTALLED_BUT_NOT_OBSERVED' if self.session_id else 'NOT_STARTED',
                    'errors': self.errors[:10], 'mutation_supported_by_inspect': False,
                    'native_backend': 'NATIVE_BACKEND_UNAVAILABLE'}

    def list_modules(self, filters=None, offset=0, limit=50):
        filters = filters or {}
        allowed = {'layer_index', 'module_name', 'module_class', 'logical_role'}
        if set(filters) - allowed:
            raise ValueError('UNSUPPORTED_MODULE_FILTER')
        rows = [r for r in self.surface.records if all(r.get(k) == v for k, v in filters.items())]
        with self.lock:
            observed = {e['module_name'] for e in self.events}
            rows = [{**r, 'observed': bool(observed.intersection(r['aliases']))} for r in rows]
            return page(rows, offset, limit)

    def describe_target(self, module_name):
        r = copy.deepcopy(self.surface.by_name[module_name])
        r['quantization'] = self.quant.resolve(self.surface.modules[module_name], r, self.fingerprint)
        return page([r], 0, 1)['items'][0]

    def observe_start(self, *, service_epoch, module_names, request_window, exclusive,
                      ttl_seconds=60, max_events=256, max_bytes=1048576, samples_per_module=2):
        self.check_epoch(service_epoch)
        if exclusive is not True or not isinstance(request_window, str) or not 1 <= len(request_window) <= 128:
            raise ValueError('EXCLUSIVE_SINGLE_REQUEST_WINDOW_REQUIRED')
        for value, ceiling in ((ttl_seconds, 300), (max_events, 2048), (max_bytes, 2097152),
                               (samples_per_module, 8)):
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError('OBSERVATION_BUDGET_OUT_OF_RANGE')
        if len(module_names) > 512:
            raise ValueError('TOO_MANY_TARGET_MODULES')
        with self.lock:
            if self.active or self.handles:
                raise ValueError('OBSERVATION_ALREADY_ACTIVE_OR_CLEANUP_FAILED')
            modules = self.surface.selected(module_names)
            self.session_id = uuid.uuid4().hex
            self.window = request_window
            self.events, self.counts, self.errors = [], Counter(), []
            self.terminal_stop_reason = None
            self.max_events, self.max_bytes, self.samples_per_module = max_events, max_bytes, samples_per_module
            self.bytes_used = 0
            self.selected_names = list(module_names)
            self.active = True
            self.stop_reason = 'ACTIVE'
            try:
                for name, module in modules:
                    def hook(_module, args, kwargs, name=name):
                        self._observe(name, args, kwargs)
                        return None
                    self.handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
                self.timer = threading.Timer(ttl_seconds, self.stop, kwargs={'reason': 'TTL_EXPIRED'})
                self.timer.daemon = True
                self.timer.start()
            except Exception:
                self.stop(reason='INSTALL_FAILED')
                raise
            return self.status()

    def _observe(self, name, args, kwargs):
        with self.lock:
            if not self.active:
                return
            self.counts[name] += 1
            if self.counts[name] > self.samples_per_module:
                return
            try:
                from .model_descriptor import annotate_call
                parameters = tensor_paths(args, kwargs, self.surface.predicate)
                annotate_call(self.surface.modules[name], args, kwargs, parameters)
                event = {'session': self.session_id, 'service_epoch': self.epoch,
                         'runtime_fingerprint': self.fingerprint, 'origin': 'LIVE',
                         'request_window': self.window, 'request_identity_level': 'USER_ASSERTED_EXCLUSIVE_WINDOW',
                         'request_count_limit': 1, 'scheduler_request_identity': None,
                         'request_limit_enforcement': 'USER_EXCLUSIVE_WINDOW_CONTRACT_ONLY; not scheduler enforcement',
                         'module_name': name, 'module_aliases': self.surface.by_name[name]['aliases'],
                         'layer_index': self.surface.by_name[name]['layer_index'],
                         'ranks': self.surface.by_name[name]['ranks'], 'invocation': self.counts[name],
                         'phase': 'UNKNOWN', 'surface': 'module_args',
                         'parameters': parameters}
                cost = len(json.dumps(event, allow_nan=False).encode())
                if len(self.events) >= self.max_events or self.bytes_used + cost > self.max_bytes:
                    self.stop(reason='METADATA_BUDGET_REACHED')
                    return
                self.events.append(event)
                self.bytes_used += cost
                if len(self.events) >= self.max_events:
                    self.stop(reason='EVENT_BUDGET_REACHED')
            except Exception as error:
                self.errors.append(type(error).__name__ + ':' + str(error)[:200])
                self.stop(reason='OBSERVER_ERROR')
                # Observation may not break or replace the model invocation.

    def observe_poll(self, session, offset=0, limit=50):
        with self.lock:
            if session != self.session_id:
                raise ValueError('UNKNOWN_OBSERVATION_SESSION')
            return {**self.status(), **page(self.events, offset, limit)}

    def stop(self, session=None, reason='EXPLICIT_STOP'):
        with self.lock:
            if session is not None and session != self.session_id:
                raise ValueError('UNKNOWN_OBSERVATION_SESSION')
            self.active = False
            if self.timer:
                self.timer.cancel()
                self.timer = None
            self.stop_reason = reason
            if reason not in ('EXPLICIT_STOP', 'ACTIVE'):
                self.terminal_stop_reason = reason
            pending = []
            for handle in self.handles:
                try:
                    handle.remove()
                except Exception as error:
                    pending.append(handle)
                    self.errors.append('REMOVE_FAILED:' + type(error).__name__)
            self.handles = pending
            return self.status()
