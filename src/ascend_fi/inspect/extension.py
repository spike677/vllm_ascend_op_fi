"""Single prefixed RPC; lazy init compatible with vLLM worker mixins."""
from __future__ import annotations

import importlib.metadata
import os
import threading

from .session import InspectSession, page
from .surface import stored
from .plans import module_candidates, export_plan


_INIT_LOCK = threading.RLock()
ACTIONS = frozenset({'status', 'list_modules', 'describe_target', 'describe_model',
                     'auto_targets', 'observe_start',
                     'observe_poll', 'observe_stop', 'list_candidates', 'export_plan'})


def worker_model(worker):
    for parent in (worker, stored(worker, 'worker')):
        runner = stored(parent, 'model_runner')
        model = stored(runner, 'model')
        if model is not None:
            return model
    raise ValueError('WORKER_MODEL_UNAVAILABLE')


def rank_context(worker):
    result = {}
    for role, field, env in [('global', 'rank', 'RANK'), ('local', 'local_rank', 'LOCAL_RANK'),
                             ('tp', 'tp_rank', None), ('pp', 'pp_rank', None),
                             ('ep', 'ep_rank', None), ('dp', 'dp_rank', None)]:
        value, source = stored(worker, field), 'stored_worker.' + field
        if type(value) is not int:
            raw = os.environ.get(env, '') if env else ''
            value = int(raw) if raw.isdigit() else None
            source = 'env:' + env if value is not None else 'UNKNOWN'
        result[role] = {'value': value, 'source': source}
    return result


def runtime_context():
    result = {'pid': os.getpid()}
    for name in ('torch', 'torch_npu', 'vllm', 'vllm-ascend'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = 'UNKNOWN'
    # No guessed CANN version or physical/logical device equality.
    result['CANN'] = os.environ.get('ASCEND_FI_INSPECT_CANN_VERSION', 'UNKNOWN')
    result['CANN_source'] = 'deployment declaration; runtime verification required'
    return result


class InspectWorkerExtension:
    def demoncat_inspect(self, action='status', **kwargs):
        if os.environ.get('VLLM_SERVER_DEV_MODE') != '1':
            return {'ok': False, 'error': 'DEVELOPER_MODE_REQUIRED'}
        if action not in ACTIONS:
            return {'ok': False, 'error': 'ACTION_NOT_ALLOWED'}
        try:
            with _INIT_LOCK:
                session = stored(self, '_demoncat_inspect_session_v1')
                if session is not None and worker_model(self) is not session.surface.model:
                    session.stop(reason='MODEL_REPLACED')
                    if session.handles:
                        raise ValueError('OLD_MODEL_OBSERVER_CLEANUP_FAILED')
                    session = None
                if session is None:
                    session = InspectSession(worker_model(self), rank_context(self), runtime_context())
                    setattr(self, '_demoncat_inspect_session_v1', session)
            targets = kwargs.pop('target_service_epochs', None)
            if targets is not None:
                if not isinstance(targets, list) or not targets or len(targets) > 64:
                    raise ValueError('INVALID_TARGET_SERVICE_EPOCHS')
                if session.epoch not in targets:
                    return {'ok': True, 'skipped': True, 'service_epoch': session.epoch,
                            'worker': session.surface.ranks, 'reason': 'NOT_SELECTED_WORKER'}
                if action in ('observe_start', 'export_plan'):
                    kwargs.setdefault('service_epoch', session.epoch)
            sessions = kwargs.pop('session_by_epoch', None)
            if sessions is not None:
                kwargs['session'] = sessions[session.epoch]
            available = all(callable(getattr(type(self), method, None)) for method in
                            ('fake_requant_install', 'fake_requant_arm', 'fake_requant_remove', 'fake_requant_query'))
            if action == 'status':
                if kwargs:
                    raise ValueError('UNKNOWN_STATUS_ARGUMENT')
                value = {**session.status(), 'reference_backend_available': available, 'actions': sorted(ACTIONS)}
            elif action in ('describe_model', 'auto_targets'):
                from .model_descriptor import describe_model
                from .target_resolver import resolve_targets
                descriptor, modules = describe_model(session, self)
                if action == 'describe_model':
                    value = {'descriptor': descriptor, **page(modules, **kwargs)}
                else:
                    with session.lock:
                        report = resolve_targets(descriptor, modules, session.events, session.status())
                    targets_page = page(report.pop('targets'), **kwargs)
                    value = {**report, **targets_page}
            elif action == 'list_modules':
                value = session.list_modules(**kwargs)
            elif action == 'describe_target':
                value = session.describe_target(**kwargs)
            elif action == 'observe_start':
                if available:
                    query = self.fake_requant_query()
                    if query.get('enabled') is not False or query.get('installed') is not False:
                        raise ValueError('EXISTING_FAULT_OR_HOOK_STATE_NOT_CLEAN; inspect will not remove it')
                value = session.observe_start(**kwargs)
            elif action == 'observe_poll':
                value = session.observe_poll(**kwargs)
            elif action == 'observe_stop':
                value = session.stop(**kwargs)
            elif action == 'list_candidates':
                rows = module_candidates(session, available)
                filters = kwargs.pop('filters', {})
                allowed = {'module_name', 'logical_role', 'capability_id', 'backend', 'origin', 'dtype', 'layer_index'}
                if set(filters) - allowed:
                    raise ValueError('UNSUPPORTED_CANDIDATE_FILTER')
                rows = [r for r in rows if all((r['observed'].get(k) if k in ('dtype', 'layer_index') else r.get(k)) == v
                                             for k, v in filters.items())]
                value = page(rows, **kwargs)
            else:
                ids = kwargs.pop('candidate_ids')
                choices = {c['candidate_id']: c for c in module_candidates(session, available)}
                if len(ids) != len(set(ids)) or any(cid not in choices for cid in ids):
                    raise ValueError('UNKNOWN_OR_DUPLICATE_CANDIDATE')
                value = export_plan([choices[cid] for cid in ids], list(session.surface.modules), **kwargs)
                session.check_epoch(value['service_epoch'])
                if value['runtime_fingerprint'] != session.fingerprint:
                    raise ValueError('STALE_RUNTIME')
            return {'ok': True, 'worker': session.surface.ranks, 'service_epoch': session.epoch, 'result': value}
        except Exception as error:
            return {'ok': False, 'error': type(error).__name__ + ':' + str(error)[:500]}
