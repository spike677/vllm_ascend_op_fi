"""Client routing over the existing Worker RPC ownership state machine.

The serving runtime owns Workers and scheduler work. This module starts no
process, replaces no model method, installs no native library, and retries no
fault. The catalog/binding HostContracts remain separate deployment contracts.
"""

from copy import deepcopy
import json
import threading
import uuid

from .config import FaultConfig
from .host_receipt_ledger import ReceiptLedger, ReceiptLedgerError


HOST_API = 'demoncat.worker_core.host_adapter.v1'
_ATTACH = {'tenant_id', 'attach_nonce', 'worker'}
_LEASE = _ATTACH | {'mechanism', 'implementation_version', 'owner', 'generation'}
_RESTORE = {'disable', 'abort', 'recover', 'clean', 'unregister'}
_METHODS = {'demoncat_attach', 'demoncat_fault_rpc', 'demoncat_fault_query',
            'demoncat_detach', 'demoncat_discover', 'demoncat_census'}
_FAILURES = {'PROVIDER_ACTION_FAILED', 'NATIVE_STATE_UNAVAILABLE',
             'PROVIDER_STATE_UNAVAILABLE', 'RECOVERY_REQUIRED'}
_SCOPE_FAILURES = {'WORKER_IDENTITY_MISMATCH', 'WORKER_IDENTITY_CHANGED_WITH_OWNED_RESOURCES',
                   'STALE_ATTACHMENT', 'STALE_GENERATION', 'OWNER_MISMATCH'}


def _canonical(value):
    return json.dumps(value, allow_nan=False, sort_keys=True)


def _copy(value):
    return json.loads(_canonical(value))


class HostContractError(RuntimeError):
    """A host route or response cannot be established without guessing."""

    def __init__(self, reason, response=None):
        super().__init__(reason)
        self.response = response


class HostAdapter:
    """Adapt an *existing* executor's ``collective_rpc`` method.

    ``select_worker(config, identities)`` is trusted host routing policy. It
    must return one complete identity from the supplied tuple. Without policy,
    exactly one Worker is required; rank zero is never implicitly selected.

    ``prepare_binding(identity, config)`` may provision an inert, reviewed
    deployment binding, then return action settings as ``{'prepare': {...},
    'register': {...}}``. It must not arm callbacks or acquire fault resources;
    those belong exclusively to the existing provider lease. Omit the hook
    when the deployment binding is already installed. Providers still verify
    the binding and all qualification conditions at their normal RPC stages.
    """

    api = HOST_API

    def __init__(self, executor, *, select_worker=None, prepare_binding=None, timeout=30):
        if not callable(getattr(executor, 'collective_rpc', None)):
            raise TypeError('An existing collective_rpc executor is required')
        if any(value is not None and not callable(value)
               for value in (select_worker, prepare_binding)):
            raise TypeError('Host policies must be callable')
        if type(timeout) not in (int, float) or not 0 < timeout < float('inf'):
            raise ValueError('A finite positive RPC timeout is required')
        self.executor = executor
        self._select = select_worker
        self._binding = prepare_binding
        self.timeout = timeout

    def _broadcast(self, method, args=()):
        rows = self.executor.collective_rpc(method, timeout=self.timeout, args=args)
        if (not isinstance(rows, (list, tuple)) or not rows
                or any(not isinstance(row, dict) or not isinstance(row.get('status'), str)
                       for row in rows)):
            raise HostContractError('INVALID_RPC_RESPONSE')
        return _copy(list(rows))

    def discover_workers(self):
        """Read a complete, unambiguous identity census without attaching."""
        rows = self._broadcast('demoncat_worker_identity')
        # A broadcast cannot safely distinguish a missing/duplicate rank. Do
        # not send mutations before every recipient has an established identity.
        if any(row['status'] != 'WORKER_IDENTIFIED' for row in rows):
            raise HostContractError('WORKER_CONTEXT_NOT_READY')
        identities = [row.get('worker') for row in rows]
        if any(not isinstance(worker, dict) or type(worker.get('rank')) is not int
               or worker['rank'] < 0 for worker in identities):
            raise HostContractError('INVALID_WORKER_IDENTITY')
        if len({worker['rank'] for worker in identities}) != len(identities):
            raise HostContractError('AMBIGUOUS_BROADCAST_RANK')
        return rows

    def route(self, config):
        rows = self.discover_workers()
        identities = [row['worker'] for row in rows]
        if self._select is None:
            if len(identities) != 1:
                raise HostContractError('TRUSTED_WORKER_ROUTING_REQUIRED')
            selected = identities[0]
        else:
            selected = self._select(config, tuple(deepcopy(identities)))
        matches = [row for row in rows if _canonical(row['worker']) == _canonical(selected)]
        if len(matches) != 1:
            raise HostContractError('SELECTED_WORKER_NOT_DISCOVERED')
        if matches[0].get('quarantined') is not False:
            raise HostContractError('WORKER_RECOVERY_REQUIRED')
        return _copy(selected)

    def capabilities(self, worker):
        rows = self._broadcast('demoncat_capabilities')
        matches = [row for row in rows if row['status'] == 'CAPABILITIES'
                   and _canonical(row.get('worker')) == _canonical(worker)]
        if len(matches) != 1:
            raise HostContractError('CAPABILITY_WORKER_IDENTITY_MISMATCH')
        return matches[0]

    def prepare_binding(self, worker, config):
        settings = {} if self._binding is None else self._binding(_copy(worker), config)
        if (not isinstance(settings, dict) or set(settings) - {'prepare', 'register'}
                or any(not isinstance(value, dict) for value in settings.values())):
            raise HostContractError('INVALID_HOST_BINDING_SETTINGS')
        return _copy(settings)

    def invoke(self, worker, method, *args):
        if method not in _METHODS:
            raise HostContractError('UNSUPPORTED_WORKER_METHOD')
        if (not args or not isinstance(args[-1], dict)
                or _canonical(args[-1].get('worker')) != _canonical(worker)):
            raise HostContractError('RPC_WORKER_IDENTITY_MISMATCH')
        rows = self._broadcast(method, tuple(_copy(list(args))))
        targeted = [row for row in rows if row['status'] != 'RPC_NOT_TARGETED']
        if len(targeted) != 1:
            raise HostContractError('RPC_TARGET_NOT_UNIQUE', rows)
        return targeted[0]


class WorkerFaultClient:
    """One automatic route and one attempted lease; Runtime owns lifecycle.

    Fault actions are explicit: prepare/register/check/enable (or execute),
    advance/finish where serving replay requires them, then disable/unregister
    and detach. No context manager performs hidden cleanup or retries.
    """

    def __init__(self, host, config, *, receipt_mode='full'):
        if type(receipt_mode) is not str or receipt_mode not in ('full', 'cursor'):
            raise ValueError('receipt_mode must be full or cursor')
        if getattr(host, 'api', None) != HOST_API or not all(
                callable(getattr(host, name, None))
                for name in ('route', 'capabilities', 'prepare_binding', 'invoke')):
            raise TypeError('An explicit Worker Core HostAdapter is required')
        if isinstance(config, dict):
            config = FaultConfig.from_dict(config)
        if not isinstance(config, FaultConfig):
            raise TypeError('Worker FaultConfig is required')
        self.host, self.config = host, config
        self.owner = 'worker-fault-' + uuid.uuid4().hex
        self.tenant_id = 'worker-client-' + uuid.uuid4().hex
        self._worker = self._attachment = self._lease = None
        self._census = None
        self._capabilities = None
        self._settings = {}
        self._attach_started = self._prepare_started = False
        self._uncertain = self._released = self._pending = False
        self._check_completed = False
        self._history = []
        self._lock = threading.RLock()
        self._receipt_mode = receipt_mode
        self._receipt_ledger = ReceiptLedger(self.owner, self.tenant_id) if receipt_mode == 'cursor' else None
        self._receipt_streaming = False

    def _receipt_idle(self):
        if self._receipt_streaming:
            raise ReceiptLedgerError('RECEIPT_EXPORT_IN_PROGRESS')

    def _record(self, action, reply):
        self._receipt_idle()
        self._history.append({'action': action, 'response': _copy(reply)})
        return self._action_snapshot()

    def _accepted_record(self, action, reply, *, uncertain=True):
        # An accepted cursor-mode RPC is already authoritative. Local receipt
        # failures must never become another RPC failure or a fictitious owner.
        if self._receipt_mode == 'cursor':
            return self._record(action, reply)
        try:  # Preserve the default full-mode exception path as well as values.
            return self._record(action, reply)
        except BaseException as error:
            return self._failure(action, error, uncertain=uncertain, response=reply)

    def _action_snapshot(self):
        if self._receipt_mode == 'full':
            return self._snapshot()
        try:
            ledger = self._receipt_index()
        except Exception as error:
            return self._receipt_ledger.incomplete_control(
                self._snapshot_fields(), error, index_complete=False)
        try:
            return ledger.control(self._snapshot_fields(), record_error=True)
        except Exception as error:
            return ledger.incomplete_control(self._snapshot_fields(), error, index_complete=True)

    def _reject(self, action, reason):
        return self._record(action, {'status': 'HOST_REJECTED', 'reason': reason})

    def _failure(self, action, error, *, uncertain, response=None):
        self._uncertain |= uncertain
        reply = {'status': 'RPC_UNCERTAIN' if uncertain else 'HOST_REJECTED',
                 'reason': str(error) if isinstance(error, HostContractError) else 'HOST_CALL_FAILED',
                 'error_type': type(error).__name__}
        from .runtime import _provider_error
        reply.update(_provider_error(error))
        message=str(error)
        reply['error_message']=message[:2048]
        reply['error_message_truncated']=len(message)>2048
        observed = response if response is not None else getattr(error, 'response', None)
        if observed is not None:
            try:
                reply['observed_response'] = _copy(observed)
            except (TypeError, ValueError, RecursionError):
                reply['observed_response_unserializable'] = True
        result = self._record(action, reply)
        if not isinstance(error, Exception):
            raise error
        return result

    def _attachment_matches(self, ticket):
        return (isinstance(ticket, dict) and set(ticket) == _ATTACH
                and ticket['tenant_id'] == self.tenant_id
                and isinstance(ticket['attach_nonce'], str) and bool(ticket['attach_nonce'])
                and ticket['attach_nonce'].isascii()
                and _canonical(ticket['worker']) == _canonical(self._worker))

    def _lease_matches(self, ticket):
        if not isinstance(ticket, dict) or set(ticket) != _LEASE or self._attachment is None:
            return False
        return (_canonical({key: ticket[key] for key in _ATTACH}) == _canonical(self._attachment)
                and ticket['mechanism'] == self.config.mechanism
                and ticket['implementation_version'] == self.config.implementation_version
                and ticket['owner'] == self.owner
                and type(ticket['generation']) is int and ticket['generation'] > 0
                and (self._lease is None or _canonical(ticket) == _canonical(self._lease)))

    def _attach(self):
        self._receipt_idle()
        if self._attachment is not None:
            return None
        if self._attach_started:
            return self._reject('attach', 'ATTACHMENT_ALREADY_ATTEMPTED')
        try:
            self._worker = self.host.route(self.config)
            self._capabilities = self.host.capabilities(self._worker)
        except BaseException as error:
            return self._failure('attach', error, uncertain=False)
        self._attach_started = True  # Persist intent before sending the RPC.
        reply = None
        try:
            reply = self.host.invoke(self._worker, 'demoncat_attach',
                                     {'tenant_id': self.tenant_id, 'worker': self._worker})
            if reply.get('status') == 'ATTACHED':
                if not self._attachment_matches(reply.get('ticket')):
                    raise HostContractError('ATTACHMENT_TICKET_MISMATCH')
                self._attachment = _copy(reply['ticket'])
            elif reply.get('status') != 'REJECTED':
                raise HostContractError('INVALID_ATTACH_RESPONSE')
        except BaseException as error:
            return self._failure('attach', error, uncertain=True, response=reply)
        return self._accepted_record('attach', reply)

    def attach(self):
        with self._lock:
            result = self._attach()
            return self._action_snapshot() if result is None else result

    def prepare(self):
        with self._lock:
            self._receipt_idle()
            if self._census is not None:
                return self._reject('prepare', 'CENSUS_CLEANUP_REQUIRED')
            if self._prepare_started or self._uncertain:
                return self._reject('prepare', 'NEW_CLIENT_REQUIRED_AFTER_PREPARE_ATTEMPT')
            result = self._attach()
            if self._attachment is None:
                return result
            providers = self._capabilities.get('providers', [])
            if not any(isinstance(row, dict) and row.get('mechanism') == self.config.mechanism
                       and row.get('implementation_version') == self.config.implementation_version
                       for row in providers):
                return self._reject('prepare', 'WORKER_PROVIDER_NOT_ADAPTED')
            self._prepare_started = True
            try:
                self._settings = self.host.prepare_binding(self._worker, self.config)
            except BaseException as error:
                return self._failure('prepare_binding', error, uncertain=True)
            payload = dict(self._attachment, mechanism=self.config.mechanism,
                           implementation_version=self.config.implementation_version, owner=self.owner)
            return self._dispatch('prepare', payload)

    def _accept(self, action, reply):
        status = reply.get('status')
        if 'ticket' in reply:
            if not self._lease_matches(reply['ticket']):
                raise HostContractError('LEASE_TICKET_MISMATCH')
            self._lease = _copy(reply['ticket'])
        elif status in {'ACTION_COMPLETED', 'QUERY_OK'} | _FAILURES:
            raise HostContractError('LEASE_TICKET_MISSING')
        if status not in {'ACTION_COMPLETED', 'QUERY_OK', 'REJECTED',
                          'PROVIDER_UNAVAILABLE', 'NATIVE_PRELOAD_NOT_ATTACHED'} | _FAILURES:
            raise HostContractError('INVALID_WORKER_RESPONSE')
        if status in {'PROVIDER_UNAVAILABLE', 'NATIVE_PRELOAD_NOT_ATTACHED'} and action != 'prepare':
            raise HostContractError('INVALID_PREPARE_ONLY_RESPONSE')
        if status == 'QUERY_OK' and action != 'query':
            raise HostContractError('UNEXPECTED_QUERY_RESPONSE')
        if status == 'ACTION_COMPLETED' and reply.get('action') != action:
            raise HostContractError('RPC_ACTION_MISMATCH')
        if status == 'ACTION_COMPLETED' and reply.get('recovery_required') is not False:
            raise HostContractError('CONTRADICTORY_ACTION_RECOVERY')
        if status in _FAILURES or reply.get('recovery_required') is True:
            self._uncertain = True
        if status == 'REJECTED' and reply.get('reason') in _SCOPE_FAILURES:
            self._uncertain = True
        if status == 'ACTION_COMPLETED':
            if action == 'check':
                self._check_completed = True
            if action in _RESTORE:
                self._uncertain = False
            if action == 'unregister':
                self._lease, self._released = None, True
        state = reply.get('provider_state') or reply.get('native')
        if isinstance(state, dict):
            self._pending = (state.get('replay_pending') is True
                             or state.get('recovery_validation_pending') is True)

    def _dispatch(self, action, payload=None, settings=None):
        self._receipt_idle()
        if action != 'prepare':
            if self._lease is None:
                return self._reject(action, 'LEASE_TICKET_REQUIRED_USE_DETACH_FOR_UNCERTAIN_PREPARE')
            if self._uncertain and action not in _RESTORE and action != 'query':
                return self._reject(action, 'RECOVERY_REQUIRED')
            payload = _copy(self._lease)
        action_settings = self._settings.get(action, {}) if settings is None else settings
        if action_settings:
            payload = dict(payload, settings=_copy(action_settings))
        method, args = ('demoncat_fault_query', (payload,)) if action == 'query' else (
            'demoncat_fault_rpc', (action, payload))
        reply = None
        try:
            reply = self.host.invoke(self._worker, method, *args)
            self._accept(action, reply)
        except BaseException as error:
            return self._failure(action, error, uncertain=True, response=reply)
        return self._accepted_record(action, reply)

    def _action(self, action):
        with self._lock:
            return self._dispatch(action)

    def register(self):
        return self._action('register')

    def check(self):
        return self._action('check')

    def execute(self):
        return self._action('execute')

    def enable(self):
        return self._action('enable')

    def advance(self, request_id):
        # This is the host's real scheduled request ID, not target qualification.
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError('An actual scheduled request ID is required')
        with self._lock:
            return self._dispatch('advance', settings={'request_id': request_id})

    def finish(self):
        return self._action('finish')

    def advance_batch(self, request_ids):
        """Forward actual enqueue IDs; the provider validates its batch scope."""
        if (not isinstance(request_ids, (list, tuple)) or not 1 <= len(request_ids) <= 256
                or any(not isinstance(value, str) or not value.strip() for value in request_ids)
                or len(set(request_ids)) != len(request_ids)):
            raise ValueError('A bounded list of distinct actual request IDs is required')
        with self._lock:
            return self._dispatch('advance', settings={'request_ids': list(request_ids)})

    def begin_decode_capture(self):
        """Observe the host's next decode step; never advance its scheduler."""
        with self._lock:
            return self._dispatch('advance', settings={'capture_decode': True})

    def collect_replay(self):
        """Collect completed provider observations without advancing inference."""
        with self._lock:
            return self._dispatch('advance', settings={'collect': True})

    def disable(self):
        return self._action('disable')

    def abort(self):
        return self._action('abort')

    def recover(self):
        return self._action('recover')

    def clean(self):
        return self._action('clean')

    def unregister(self):
        return self._action('unregister')

    def query(self):
        return self._action('query')

    def discover(self):
        """Read-only C0 inventory from extensions implementing discovery."""
        with self._lock:
            self._receipt_idle()
            if self._uncertain:
                return self._reject('discover', 'RECOVERY_REQUIRED')
            result = self._attach()
            if self._attachment is None:
                return result
            reply = None
            try:
                reply = self.host.invoke(self._worker, 'demoncat_discover', self._attachment)
                if reply.get('status') == 'DISCOVERY_SNAPSHOT':
                    report = reply.get('discovery')
                    if (not isinstance(report, dict)
                            or _canonical(report.get('worker')) != _canonical(self._worker)
                            or report.get('read_only') is not True
                            or report.get('qualification_granted') is not False):
                        raise HostContractError('DISCOVERY_SCOPE_MISMATCH')
                elif reply.get('status') not in ('DISCOVERY_FAILED', 'REJECTED'):
                    raise HostContractError('INVALID_DISCOVERY_RESPONSE')
                if reply.get('status') == 'REJECTED' and reply.get('reason') in _SCOPE_FAILURES:
                    self._uncertain = True
            except BaseException as error:
                return self._failure('discover', error, uncertain=self._lease is not None, response=reply)
            return self._accepted_record('discover', reply, uncertain=self._lease is not None)

    def detach(self):
        """Explicit same-tenant cleanup, including a lost prepare response."""
        with self._lock:
            self._receipt_idle()
            if self._attachment is None:
                return self._reject('detach', 'ATTACHMENT_TICKET_REQUIRED')
            reply = None
            try:
                reply = self.host.invoke(self._worker, 'demoncat_detach', self._attachment)
                status = reply.get('status')
                if status in ('DETACHED', 'RECOVERY_REQUIRED'):
                    census_recoveries = reply.get('census_recoveries', [])
                    if not isinstance(census_recoveries, list) or len(census_recoveries) > 1:
                        raise HostContractError('INVALID_CENSUS_DETACH_RECOVERY')
                    if self._census is not None and not census_recoveries:
                        raise HostContractError('CENSUS_DETACH_RECOVERY_MISSING')
                    for row in census_recoveries:
                        self._accept_census('cleanup', row)
                    if status == 'DETACHED' and self._census is not None:
                        raise HostContractError('CENSUS_DETACH_NOT_RELEASED')
                    recoveries = reply.get('recoveries')
                    if not isinstance(recoveries, list) or len(recoveries) > 1:
                        raise HostContractError('INVALID_DETACH_RECOVERY')
                    if self._lease is not None and not recoveries:
                        raise HostContractError('DETACH_RECOVERY_MISSING')
                    if status == 'RECOVERY_REQUIRED' and _canonical(reply.get('ticket')) != _canonical(self._attachment):
                        raise HostContractError('DETACH_ATTACHMENT_MISMATCH')
                    for row in recoveries:
                        if not isinstance(row, dict) or not self._lease_matches(row.get('ticket')):
                            raise HostContractError('DETACH_OWNER_MISMATCH')
                        self._lease = _copy(row['ticket'])
                    if status == 'DETACHED':
                        if any(row.get('status') != 'ACTION_COMPLETED'
                               or row.get('action') != 'unregister' or row.get('phase') != 'UNREGISTERED'
                               or row.get('recovery_required') is not False for row in recoveries):
                            raise HostContractError('DETACH_RECOVERY_NOT_CONFIRMED')
                        self._attachment = self._lease = None
                        self._uncertain = self._pending = False
                        self._released = True
                    else:
                        self._uncertain = True
                elif status == 'REJECTED':
                    self._uncertain = True
                else:
                    raise HostContractError('INVALID_DETACH_RESPONSE')
            except BaseException as error:
                return self._failure('detach', error, uncertain=True, response=reply)
            return self._accepted_record('detach', reply)

    def _snapshot_fields(self):
        recovery = ('REQUIRED' if self._uncertain else 'LEASE_RELEASE_CONFIRMED' if self._released
                    else 'PENDING' if self._pending else 'NOT_ASSESSED')
        response = self._history[-1]['response'] if self._history else {'status': 'NEW'}
        return dict(
            schema='demoncat.worker_core.host_receipt.v1', status=response['status'],
            config=self.config.as_dict(), mechanism=self.config.mechanism,
            worker=self._worker, owner=self.owner, tenant_id=self.tenant_id,
            attachment=self._attachment, ticket=self._lease, census_ticket=self._census,
            ownership_retained=bool(self._attachment or self._lease or self._census or self._uncertain),
            capability_enumeration=self._capabilities,
            provider_check='COMPLETED' if self._check_completed else 'NOT_ASSESSED',
            assessments={'C2': 'NOT_ASSESSED', 'C3': 'NOT_ASSESSED', 'E5': 'NOT_ASSESSED',
                         'Recovery': {'status': recovery, 'healthy_output': 'NOT_ASSESSED'}},
            response=response, history=self._history,
            inner_framework_integration_verified=False,
        )

    def _snapshot(self):
        return _copy(self._snapshot_fields())

    def snapshot(self):
        """Copy the receipt ledger; this is not a fresh Worker readback."""
        with self._lock:
            return self._snapshot()

    def _receipt_index(self):
        if self._receipt_mode != 'cursor' or self._receipt_ledger is None:
            raise ReceiptLedgerError('RECEIPT_CURSOR_MODE_REQUIRED')
        try:
            self._receipt_ledger.synchronize(self.owner, self.tenant_id, self._history)
        except BaseException as error:
            code = error.code if isinstance(error, ReceiptLedgerError) else 'RECEIPT_INDEX_FAILED'
            self._receipt_ledger.note_error(code, error_type=type(error).__name__)
            if not isinstance(error, Exception):
                raise
            raise ReceiptLedgerError(code, error_type=type(error).__name__) from error
        return self._receipt_ledger

    def receipt_cursor(self):
        """Return a live ledger position, not permission to issue a Worker action."""
        with self._lock:
            return self._receipt_index().cursor()

    def status_receipt(self):
        """Read bounded Host control state without copying history or making RPCs."""
        with self._lock:
            return self._receipt_index().control(self._snapshot_fields())

    def receipt_batch(self, cursor, *, limit=4):
        """Copy at most four complete records / 8 MiB; never skip an oversized row."""
        with self._lock:
            return self._receipt_index().batch(cursor, self._history, limit)

    def stream_receipt(self, cursor, sink):
        """Export one original record to a fresh binary sink, in <=64 KiB writes."""
        with self._lock:
            self._receipt_idle()
            ledger = self._receipt_index()
            parts, size, metadata = ledger.record_parts(cursor, self._history)
            self._receipt_streaming = True
            try:
                return ledger.write(sink, parts, size, metadata)
            finally:
                self._receipt_streaming = False

    def stream_snapshot(self, sink, *, cursor):
        """Export the full legacy v1 receipt once, without cloning all history."""
        with self._lock:
            self._receipt_idle()
            ledger = self._receipt_index()
            parts, size, metadata = ledger.snapshot_parts(cursor, self._snapshot_fields(), self._history)
            self._receipt_streaming = True
            try:
                return ledger.write(sink, parts, size, metadata)
            finally:
                self._receipt_streaming = False

    def _accept_census(self, action, reply):
        status = reply.get('status')
        if status in ('REJECTED', 'CENSUS_REJECTED'):
            if reply.get('reason') in _SCOPE_FAILURES:
                self._uncertain = True
            return
        if status not in ('CENSUS_SNAPSHOT', 'CENSUS_RELEASED', 'CENSUS_RECOVERY_REQUIRED'):
            raise HostContractError('INVALID_CENSUS_RESPONSE')
        ticket = reply.get('ticket')
        if (not isinstance(ticket, dict) or set(ticket) != _ATTACH | {'owner', 'generation'}
                or not self._attachment_matches({key: ticket[key] for key in _ATTACH})
                or ticket['owner'] != self.owner or type(ticket['generation']) is not int
                or ticket['generation'] < 1
                or (self._census is not None and _canonical(ticket) != _canonical(self._census))
                or reply.get('action') != action):
            raise HostContractError('CENSUS_TICKET_MISMATCH')
        self._census = _copy(ticket)
        if status == 'CENSUS_RELEASED':
            report = reply.get('census', {})
            if (action != 'cleanup' or reply.get('recovery_required') is not False
                    or report.get('releasable') is not True
                    or report.get('resource_cleanup_confirmed') is not True
                    or report.get('qualification_granted') is not False):
                raise HostContractError('CENSUS_RELEASE_NOT_CONFIRMED')
            self._census = None
            self._uncertain = False
        elif status == 'CENSUS_RECOVERY_REQUIRED' or reply.get('recovery_required') is True:
            self._uncertain = True

    def census_start(self, *, phase, expected, expected_max_seqs=1, max_events=10000):
        """Observe one naturally enqueued request; never run a model here."""
        with self._lock:
            self._receipt_idle()
            if self._census is not None or self._lease is not None or self._uncertain:
                return self._reject('census_start', 'IDLE_OWNERSHIP_REQUIRED')
            result = self._attach()
            if self._attachment is None:
                return result
            settings = dict(phase=phase, expected=expected,
                            expected_max_seqs=expected_max_seqs, max_events=max_events)
            return self._census_action('start', dict(self._attachment, owner=self.owner, settings=settings))

    def _census_action(self, action, payload=None):
        self._receipt_idle()
        if payload is None:
            if self._census is None:
                return self._reject('census_' + action, 'CENSUS_TICKET_REQUIRED')
            payload = self._census
        reply = None
        try:
            reply = self.host.invoke(self._worker, 'demoncat_census', action, _copy(payload))
            self._accept_census(action, reply)
        except BaseException as error:
            return self._failure('census_' + action, error, uncertain=True, response=reply)
        return self._accepted_record('census_' + action, reply)

    def census_query(self):
        with self._lock:
            return self._census_action('query')

    def census_stop(self):
        with self._lock:
            return self._census_action('stop')

    def census_cleanup(self):
        with self._lock:
            return self._census_action('cleanup')
