"""Worker-local lease validation and delegation to the selected provider."""

from copy import deepcopy
from dataclasses import dataclass
import json
import re
import secrets
import threading

from .identity import EXTENSION_VERSION, IdentityTracker


_TICKET = {'tenant_id', 'attach_nonce', 'worker'}
_TARGET = {'mechanism', 'implementation_version', 'owner'}
_RESTORE = {'abort', 'disable', 'clean', 'unregister', 'recover'}
_ACTIONS = {'prepare', 'register', 'check', 'execute', 'enable', 'advance', 'finish'} | _RESTORE


class Rejected(ValueError):
    """Machine-readable refusal before an action is dispatched."""


class NativePreloadNotAttached(RuntimeError):
    """Provider proved preload absence before installing or arming anything."""


def _resolve_provider(mechanism, implementation_version, worker, model, identity):
    from .providers import resolve_provider

    return resolve_provider(mechanism, implementation_version, worker, model, identity)


def _provider_capabilities():
    from .providers import provider_capabilities

    return provider_capabilities()


def _text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 128 or '\0' in value:
        raise Rejected('INVALID_SCOPE_FIELD')
    return value


def _check_shape(value, required, optional=()):
    if (not isinstance(value, dict) or not required <= value.keys()
            or value.keys() - required - set(optional)):
        raise Rejected('INVALID_PAYLOAD')


def _json(value):
    # Canonical serialization also distinguishes a Boolean from an integer.
    return json.dumps(value, allow_nan=False, sort_keys=True)


def _provider_error(error):
    details = {'error_type': type(error).__name__}
    code = getattr(error, 'code', None)
    if isinstance(code, str) and re.fullmatch(r'[A-Z][A-Z0-9_]{0,79}', code):
        details['provider_code'] = code
    report = getattr(error, 'prerequisites', None)
    if isinstance(report, dict) and report.get('schema') == 'demoncat.prerequisites.v1':
        encoded = _json(report)
        if len(encoded.encode('utf-8')) <= 32768:
            details['prerequisites'] = json.loads(encoded)
    return details


@dataclass
class _Tenant:
    tenant_id: str
    attach_nonce: str
    identity: dict

    def ticket(self):
        return dict(tenant_id=self.tenant_id, attach_nonce=self.attach_nonce,
                    worker=deepcopy(self.identity))


@dataclass
class _Lease:
    tenant: _Tenant
    mechanism: str
    implementation_version: str
    owner: str
    generation: int
    native_owner: str
    provider: object
    phase: str = 'PREPARING'
    attempted: bool = False
    recovery_required: bool = False
    replay_pending: bool = False

    def ticket(self):
        return dict(self.tenant.ticket(), mechanism=self.mechanism,
                    implementation_version=self.implementation_version,
                    owner=self.owner, generation=self.generation)


class WorkerFaultRuntime:
    """One RPC owner table for one live Worker. It owns no host lifecycle methods."""

    def __init__(self, pid):
        self.pid = pid
        self._identity = IdentityTracker(pid)
        self._lock = threading.Lock()
        self._tenants = {}
        self._leases = {}
        self._census = None
        self._generation = 0
        self._identity_invalidated = False

    def invoke(self, worker, action, payload=None):
        if not self._lock.acquire(blocking=False):
            return {'status': 'REJECTED', 'reason': 'WORKER_RPC_BUSY'}
        try:
            context = self._identity.read(worker)
            self._observe_context(context)
            if action == 'worker_identity':
                return {'status': 'WORKER_IDENTIFIED' if context.ready else 'WORKER_CONTEXT_NOT_READY',
                        'worker': context.worker, 'quarantined': self._quarantined()}
            if action == 'capabilities':
                try:
                    providers = _provider_capabilities()
                    from .registry import worker_core_manifest
                    core_manifest = worker_core_manifest()
                    from .ordinary_admission import ordinary_route_report
                    for row in core_manifest['mechanisms']:
                        row['ordinary_task_prerequisites'] = ordinary_route_report(row['mechanism'],worker,context.model)
                    resolver_status = 'AVAILABLE'
                except ModuleNotFoundError as error:
                    if (error.name or '').split('.')[0] not in ('fault_injection', 'demoncat_fault39'):
                        raise
                    providers, resolver_status = [], 'DEPENDENCY_UNAVAILABLE'
                    core_manifest = None
                return {'status': 'CAPABILITIES', 'worker': context.worker,
                        'extension_version': EXTENSION_VERSION, 'providers': providers,
                        'worker_core_manifest': core_manifest,
                        'provider_resolver_status': resolver_status,
                        'native_state_requires_prepared_ticket': True, 'one_attempt_per_lease': True,
                        'quarantined': self._quarantined()}
            if action == 'attach':
                return self._attach(context, payload)
            if action == 'detach':
                if self._census is not None:
                    _check_shape(payload, _TICKET)
                    tenant = self._tenant(context, payload)
                    if self._census['tenant'] is tenant:
                        recovery = self._census_action(worker, context, 'cleanup', self._census_ticket())
                        if self._census is not None:
                            return dict(status='RECOVERY_REQUIRED', ticket=tenant.ticket(),
                                        recoveries=[], census_recoveries=[recovery])
                        result = self._detach(context, payload)
                        result['census_recoveries'] = [recovery]
                        return result
                return self._detach(context, payload)
            if isinstance(action, str) and action.startswith('census_'):
                return self._census_action(worker, context, action.removeprefix('census_'), payload)
            if action == 'query':
                return self._query(context, payload)
            if action == 'discover':
                _check_shape(payload, _TICKET)
                self._tenant(context, payload)
                if self._leases or self._census or self._quarantined():
                    raise Rejected('DISCOVERY_REQUIRES_IDLE_WORKER_OWNER_TABLE')
                from .discovery import discover_worker
                try:
                    report = discover_worker(worker, context.model, context.worker)
                except Exception as error:
                    after = self._identity.read(worker)
                    self._observe_context(after)
                    return {'status': 'DISCOVERY_FAILED', 'reason': 'READ_ONLY_DISCOVERY_UNAVAILABLE',
                            **_provider_error(error), 'qualification_granted': False}
                after = self._identity.read(worker)
                self._observe_context(after)
                self._expect_worker(after, payload['worker'])
                return {'status': 'DISCOVERY_SNAPSHOT', 'discovery': report}
            if isinstance(action, str) and action in _ACTIONS:
                return self._dispatch(worker, context, action, payload)
            raise Rejected('UNSUPPORTED_ACTION')
        except Rejected as error:
            if str(error) == 'RPC_NOT_TARGETED':
                return {'status': 'RPC_NOT_TARGETED'}
            return {'status': 'REJECTED', 'reason': str(error)}
        finally:
            self._lock.release()

    def _observe_context(self, context):
        # Losing visibility on a different RPC thread does not itself show a
        # model replacement. A changed live context invalidates all tickets.
        live = context.worker
        keys = ('rank', 'configured_logical_device', 'model_instance_id')
        changed = [tenant for tenant in self._tenants.values() if any(
            tenant.identity[key] != live[key] for key in keys
        ) or (context.ready and _json(tenant.identity) != _json(live))]
        if (self._leases or self._census) and changed:
            self._identity_invalidated = True
        elif changed:
            for tenant in changed:
                del self._tenants[tenant.tenant_id]

    def _expect_worker(self, context, expected):
        if not isinstance(expected, dict) or type(expected.get('rank')) is not int:
            raise Rejected('INVALID_WORKER_IDENTITY')
        live = context.worker
        if live['rank'] is not None and live['rank'] != expected['rank']:
            raise Rejected('RPC_NOT_TARGETED')
        if not context.ready:
            raise Rejected('WORKER_CONTEXT_NOT_READY')
        try:
            exact_match = _json(expected) == _json(live)
        except (TypeError, ValueError, RecursionError):
            raise Rejected('INVALID_WORKER_IDENTITY') from None
        if not exact_match:
            raise Rejected('WORKER_IDENTITY_MISMATCH')
        if self._identity_invalidated:
            raise Rejected('WORKER_IDENTITY_CHANGED_WITH_OWNED_RESOURCES')

    def _attach(self, context, payload):
        _check_shape(payload, {'tenant_id', 'worker'})
        self._expect_worker(context, payload['worker'])
        tenant_id = _text(payload['tenant_id'])
        if tenant_id in self._tenants:
            raise Rejected('TENANT_ALREADY_ATTACHED')
        if self._quarantined():
            raise Rejected('WORKER_RECOVERY_REQUIRED')
        if self._census is not None:
            raise Rejected('WORKER_CENSUS_IN_PROGRESS')
        if any(lease.replay_pending for lease in self._leases.values()):
            raise Rejected('WORKER_REPLAY_IN_PROGRESS')
        tenant = _Tenant(tenant_id, secrets.token_hex(16), deepcopy(context.worker))
        self._tenants[tenant_id] = tenant
        return {'status': 'ATTACHED', 'ticket': tenant.ticket()}

    def _tenant(self, context, payload):
        self._expect_worker(context, payload['worker'])
        tenant = self._tenants.get(_text(payload['tenant_id']))
        nonce = payload['attach_nonce']
        if (tenant is None or not isinstance(nonce, str) or not nonce.isascii()
                or not secrets.compare_digest(tenant.attach_nonce, nonce)):
            raise Rejected('STALE_ATTACHMENT')
        return tenant

    def _lease(self, tenant, payload):
        mechanism = _text(payload['mechanism'])
        version = _text(payload['implementation_version'])
        owner = _text(payload['owner'])
        generation = payload.get('generation')
        lease = self._leases.get(mechanism)
        if (lease is None or type(generation) is not int
                or lease.generation != generation):
            raise Rejected('STALE_GENERATION')
        if lease.tenant is not tenant or lease.owner != owner:
            raise Rejected('OWNER_MISMATCH')
        if lease.implementation_version != version:
            raise Rejected('PROVIDER_IDENTITY_MISMATCH')
        return lease

    @staticmethod
    def _fresh(lease):
        try:
            state = lease.provider.query(owner=lease.native_owner, generation=lease.generation)
            if isinstance(state,dict):
                from .deployment_authorization import acceptance_use
                use=acceptance_use(lease.provider,'query')
                if use is not None:state=dict(state,deployment_acceptance_use=use)
                from .ordinary_evidence import bound_state
                state=bound_state(state,ordinary=getattr(lease.provider,'ordinary_only',False))
                if getattr(lease.provider,'_deployment_admission',None) is not None:
                    from fault_injection.request_plan import task_state
                    state=task_state(lease.mechanism,state)
                if state.get('evidence_limit_error'):lease.recovery_required=True
            if not isinstance(state, dict):
                return None
            boundary = getattr(lease.provider, 'execution_boundary', 'native')
            if boundary == 'native':
                # Existing Native admission is unchanged. A Python receipt is
                # never accepted in place of an actual loaded ABI readback.
                if (state.get('native_loaded') is not True
                        or type(state.get('native_registered')) is not bool
                        or state.get('execution_boundary', 'native') != 'native'):
                    return None
            elif boundary == 'python_hook':
                if (state.get('execution_boundary') != boundary
                        or state.get('state_source') != 'OWNED_PYTHON_HOOKS'
                        or state.get('hook_state_available') is not True
                        or type(state.get('hook_owned')) is not bool
                        or state.get('owner') != lease.native_owner
                        or type(state.get('generation')) is not int
                        or state['generation'] != lease.generation
                        or state.get('native_loaded', False) is not False
                        or state.get('native_registered', False) is not False):
                    return None
            else:
                return None
            if any(type(state.get(key)) is not bool for key in ('registered', 'enabled')):
                return None
            if any(type(state.get(key)) is not int or state[key] < 0 for key in ('inflight', 'leases')):
                return None
            pending = state.get('replay_pending', False)
            failed = state.get('replay_failed', False)
            if type(pending) is not bool or type(failed) is not bool:
                return None
            lease.replay_pending = pending
            if failed:
                lease.recovery_required = True
            return json.loads(_json(state))
        except BaseException as error:
            lease.recovery_required = True
            if not isinstance(error, Exception):
                raise
            return None

    @staticmethod
    def _quiet(state):
        return (state is not None and state['enabled'] is False
                and state['inflight'] == 0 and state['leases'] == 0)

    @classmethod
    def _inert(cls, state):
        return (cls._quiet(state) and state['registered'] is False
                and (state['hook_owned'] is False
                     if state.get('execution_boundary') == 'python_hook'
                     else state['native_registered'] is False))

    @staticmethod
    def _state_reply(state):
        # Retain the v1 native field for its callers without labeling an owned
        # Python hook as a Native execution or build verification.
        return {'provider_state': state,
                'native': (None if state is not None and
                           state.get('execution_boundary') == 'python_hook' else state)}

    @staticmethod
    def _unavailable(lease):
        return ('PROVIDER_STATE_UNAVAILABLE'
                if getattr(lease.provider, 'execution_boundary', 'native') != 'native'
                else 'NATIVE_STATE_UNAVAILABLE')

    def _quarantined(self):
        # Providers may share streams, allocators or process-level interposers.
        # An unresolved fault cannot be bypassed by choosing another mechanism
        # or another tenant on this same Worker.
        return self._identity_invalidated or bool(self._census and self._census['uncertain']) or any(
            lease.recovery_required for lease in self._leases.values())

    def _census_ticket(self):
        entry = self._census
        return dict(entry['tenant'].ticket(), owner=entry['owner'], generation=entry['generation'])

    def _census_action(self, worker, context, action, payload):
        if action not in ('start', 'query', 'stop', 'cleanup'):
            raise Rejected('UNSUPPORTED_CENSUS_ACTION')
        _check_shape(payload, _TICKET | {'owner'} | (set() if action == 'start' else {'generation'}),
                     ('settings',) if action == 'start' else ())
        tenant = self._tenant(context, payload)
        owner = _text(payload['owner'])
        if action == 'start':
            if self._census is not None or self._leases or self._quarantined():
                raise Rejected('CENSUS_REQUIRES_IDLE_WORKER_OWNER_TABLE')
            settings = payload.get('settings', {})
            _check_shape(settings, {'phase', 'expected'}, ('expected_max_seqs', 'max_events'))
            if type(settings.get('max_events', 10000)) is not int or not 1 <= settings.get('max_events', 10000) <= 100000:
                raise Rejected('CENSUS_EVENT_BUDGET_REQUIRED')
            from .call_census import WorkerCallCensus
            try:
                capture = WorkerCallCensus(worker, context.worker, **settings)
            except (ValueError, TypeError, RuntimeError) as error:
                return dict(status='CENSUS_REJECTED', reason='CENSUS_BINDING_INVALID', **_provider_error(error))
            self._generation += 1
            # Retain ownership before installing hooks, including partial start.
            self._census = dict(tenant=tenant, owner=owner, generation=self._generation,
                                capture=capture, uncertain=False)
        entry = self._census
        if (entry is None or (action != 'start' and
                (type(payload['generation']) is not int or payload['generation'] != entry['generation']))):
            raise Rejected('STALE_CENSUS_GENERATION')
        if entry['tenant'] is not tenant or entry['owner'] != owner:
            raise Rejected('CENSUS_OWNER_MISMATCH')
        ticket = self._census_ticket()
        try:
            report = (entry['capture'].start() if action == 'start' else
                      getattr(entry['capture'], action)(worker, context.worker))
            if action in ('cleanup', 'stop'):
                # Confirm current owned resources independently of the action's
                # returned success message, as for provider native readback.
                report = entry['capture'].query(worker, context.worker)
            report = json.loads(_json(report))
            if (not isinstance(report, dict) or type(report.get('releasable')) is not bool
                    or report.get('qualification_granted') is not False
                    or _json(report.get('worker')) != _json(context.worker)):
                raise RuntimeError('CENSUS_READBACK_UNAVAILABLE')
            if report['releasable'] and not (
                    report.get('identity_valid') is True and report.get('stop_requested') is True
                    and report.get('resource_cleanup_confirmed') is True
                    and report.get('cleanup_complete') is True
                    and report.get('inflight') is False
                    and type(report.get('observer_hooks')) is int and report['observer_hooks'] == 0
                    and report.get('foreign_hooks') is False
                    and report.get('metadata_scope_active') is False):
                raise RuntimeError('CENSUS_RELEASE_READBACK_CONTRADICTORY')
        except BaseException as error:
            entry['uncertain'] = True
            if not isinstance(error, Exception):
                raise
            return dict(status='CENSUS_RECOVERY_REQUIRED', action=action, ticket=ticket,
                        recovery_required=True, **_provider_error(error))
        released = action == 'cleanup' and report['releasable']
        if released:
            self._census = None
        elif action == 'cleanup':
            entry['uncertain'] = True
        return dict(status='CENSUS_RELEASED' if released else 'CENSUS_SNAPSHOT', action=action,
                    ticket=ticket, census=report, recovery_required=bool(self._census and entry['uncertain']),
                    qualification_granted=False)

    def _query(self, context, payload):
        _check_shape(payload, _TICKET | _TARGET | {'generation'})
        lease = self._lease(self._tenant(context, payload), payload)
        state = self._fresh(lease)
        if not self._quiet(state):
            lease.recovery_required = True
        return {'status': 'QUERY_OK' if state is not None else self._unavailable(lease),
                'ticket': lease.ticket(), 'phase': lease.phase,
                'recovery_required': lease.recovery_required, **self._state_reply(state),
                'replay_pending': lease.replay_pending,
                'quarantined': self._quarantined()}

    def _dispatch(self, worker, context, action, payload):
        required = _TICKET | _TARGET | ({'generation'} if action != 'prepare' else set())
        _check_shape(payload, required, ('settings',))
        settings = payload.get('settings', {})
        if not isinstance(settings, dict):
            raise Rejected('INVALID_PROVIDER_SETTINGS')
        try:
            settings = json.loads(_json(settings))
        except (TypeError, ValueError, RecursionError):
            raise Rejected('INVALID_PROVIDER_SETTINGS') from None
        tenant = self._tenant(context, payload)
        if self._census is not None:
            raise Rejected('WORKER_CENSUS_IN_PROGRESS')
        if action == 'prepare':
            mechanism = _text(payload['mechanism'])
            version = _text(payload['implementation_version'])
            owner = _text(payload['owner'])
            if mechanism in self._leases:
                raise Rejected('MECHANISM_ALREADY_OWNED')
            if self._quarantined():
                raise Rejected('WORKER_RECOVERY_REQUIRED')
            if any(item.replay_pending for item in self._leases.values()):
                raise Rejected('WORKER_REPLAY_IN_PROGRESS')
            try:
                provider = _resolve_provider(mechanism, version, worker, context.model, deepcopy(context.worker))
                if not all(callable(getattr(provider, name, None)) for name in ('dispatch', 'query')):
                    raise TypeError('Provider does not satisfy the worker contract')
            except NativePreloadNotAttached:
                return {'status': 'NATIVE_PRELOAD_NOT_ATTACHED', 'action': action}
            except Exception as error:
                # Resolver must be inert. Ownership starts before prepare, which can install resources.
                return dict(status='PROVIDER_UNAVAILABLE', **_provider_error(error))
            self._generation += 1
            lease = _Lease(tenant, mechanism, version, owner, self._generation,
                           secrets.token_hex(16), provider)
            self._leases[mechanism] = lease
        else:
            lease = self._lease(tenant, payload)
        return self._run(lease, action, settings)

    def _run(self, lease, action, settings):
        if any(item is not lease and item.replay_pending for item in self._leases.values()):
            raise Rejected('WORKER_REPLAY_IN_PROGRESS')
        if lease.recovery_required and action not in _RESTORE:
            raise Rejected('RECOVERY_REQUIRED')
        if self._quarantined() and action not in _RESTORE:
            raise Rejected('WORKER_RECOVERY_REQUIRED')
        expected = {'register': 'PREPARED', 'check': 'REGISTERED',
                    'execute': 'CHECKED', 'enable': 'CHECKED', 'advance': 'ATTEMPTED', 'finish': 'ATTEMPTED'}
        if action in ('execute', 'enable') and lease.attempted:
            raise Rejected('ATTEMPT_CONSUMED')
        if action in expected and lease.phase != expected[action]:
            raise Rejected('INVALID_ACTION_FOR_PHASE')
        if action in ('execute', 'enable'):
            # Count a disputed or failed execute exactly once.
            lease.attempted = True
            lease.phase = 'ATTEMPTED'
        result = {'ticket': lease.ticket(), 'action': action}
        try:
            from .deployment_authorization import acceptance_use
            use=acceptance_use(lease.provider,action)
            if use is not None:
                result['deployment_acceptance_use']=use
                if action not in _RESTORE and not use['scope_valid']:
                    raise RuntimeError('DEPLOYMENT_ACCEPTANCE_SCOPE_CHANGED')
            reply = lease.provider.dispatch(action, settings, owner=lease.native_owner,
                                            generation=lease.generation)
            if not isinstance(reply, dict):
                raise TypeError('Provider action must return a dictionary')
            from .ordinary_evidence import bound_state
            reply = bound_state(reply,ordinary=getattr(lease.provider,'ordinary_only',False))
            reply = json.loads(_json(reply))
        except BaseException as error:
            if action == 'prepare' and isinstance(error, NativePreloadNotAttached):
                del self._leases[lease.mechanism]
                return {'status': 'NATIVE_PRELOAD_NOT_ATTACHED', 'action': action}
            lease.recovery_required = True
            if not isinstance(error, Exception):
                # Preserve interruption semantics, but do not let a surviving
                # host reuse this Worker after a disputed native action.
                raise
            return dict(result, status='PROVIDER_ACTION_FAILED', phase=lease.phase,
                        recovery_required=True, **_provider_error(error))

        state = self._fresh(lease)
        verified = state is not None
        if action in ('clean', 'unregister', 'prepare'):
            verified = self._inert(state)
        elif action in ('register', 'check'):
            verified = self._quiet(state) and state['registered'] is True
        elif action in ('execute', 'enable', 'advance', 'finish', 'disable', 'abort', 'recover'):
            verified = self._quiet(state)
        if state is not None and state.get('evidence_limit_error') and action not in _RESTORE:
            verified = False
        if state is not None and state.get('replay_failed', False):
            verified = False
        if action in ('finish', 'disable', 'abort', 'clean', 'unregister') and lease.replay_pending:
            verified = False
        if not verified:
            lease.recovery_required = True
            return dict(result, status='RECOVERY_REQUIRED' if state is not None else self._unavailable(lease),
                        **self._state_reply(state), phase=lease.phase, recovery_required=True, provider=reply)

        lease.phase = {'prepare': 'PREPARED', 'register': 'REGISTERED', 'check': 'CHECKED',
                       'finish': 'FINISHED',
                       'recover': 'ATTEMPTED',
                       'clean': 'CLEANED', 'unregister': 'UNREGISTERED',
                       'disable': 'STOPPED', 'abort': 'STOPPED'}.get(action, lease.phase)
        if action == 'unregister':
            del self._leases[lease.mechanism]
        if action in _RESTORE:
            lease.recovery_required = False
        return dict(result, status='ACTION_COMPLETED', **self._state_reply(state), phase=lease.phase,
                    recovery_required=lease.recovery_required, provider=reply)

    def _detach(self, context, payload):
        _check_shape(payload, _TICKET)
        tenant = self._tenant(context, payload)
        recoveries = [self._run(lease, 'unregister', {}) for lease in tuple(self._leases.values())
                      if lease.tenant is tenant]
        if any(row['status'] != 'ACTION_COMPLETED' for row in recoveries):
            return {'status': 'RECOVERY_REQUIRED', 'ticket': tenant.ticket(), 'recoveries': recoveries}
        del self._tenants[tenant.tenant_id]
        return {'status': 'DETACHED', 'recoveries': recoveries}
