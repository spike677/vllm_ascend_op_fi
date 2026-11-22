"""Main-project facade over existing WorkerFaultClient leases.

Trusted host callbacks own ordinary inference, scheduling and canaries. This
module creates no Worker and imports no experiment. User input is never copied
into a deployment profile. Scheduler hooks carry only internal request IDs.
"""
from copy import deepcopy
import threading
import uuid
from .host import HostAdapter, WorkerFaultClient, _GroupIdentity
from .config import FaultConfig
from .continuous_binding import digest
from fault_injection.product_contract import RequestScope


def released_resources(cleanup):
    """Require observed provider counters and Runtime owner release on every rank."""
    unregister, detach = cleanup.get('unregister', []), cleanup.get('detach', [])
    if not unregister or len(unregister) != len(detach):
        return False
    for row, released in zip(unregister, detach):
        state = row.get('response', {}).get('provider_state', {})
        if not (row.get('status') == 'ACTION_COMPLETED'
                and all(type(state.get(k)) is int and state[k] == 0 for k in ('leases', 'owned_hooks', 'inflight'))
                and state.get('registered') is False and state.get('replay_pending') is False
                and state.get('recovery', {}).get('resources_zero') is True
                and released.get('status') == 'DETACHED' and released.get('ownership_retained') is False
                and released.get('worker') == row.get('worker')):
            return False
    return True


class ProductClient:
    @classmethod
    def from_deployment(cls, executor, path, *, operator, run_canary, drain, timeout=120):
        """Trusted host loads all deployment fields; caller supplies no user input.

        run_canary receives (deployment_canary_config, before_enqueue_execution).
        The existing service owns tokenization, image preprocessing and execution.
        Worker bootstrap independently checks this file against its live model.
        """
        from .product_config import load_product_deployment
        deployment = load_product_deployment(path)
        client = cls(executor, FaultConfig.from_dict(deployment['fault']),
                     profile=deployment['profile'], operator=operator, drain=drain, timeout=timeout,
                     run_canary=lambda before: run_canary(deepcopy(deployment['canary']), before))
        client.deployment = deployment
        return client

    def __init__(self, executor, config, *, profile, operator, run_canary, drain, timeout=120):
        if not isinstance(config, FaultConfig) or not callable(run_canary) or not callable(drain):
            raise TypeError('TRUSTED_PRODUCT_HOST_CALLBACKS_REQUIRED')
        self.executor, self.config = executor, config
        self.profile, self.operator = profile, operator
        self.run_canary, self.drain = run_canary, drain
        self.timeout = timeout
        self.host = HostAdapter(executor, timeout=timeout)
        self.identity = _GroupIdentity('worker-group-' + uuid.uuid4().hex, 'worker-group-client-' + uuid.uuid4().hex, 1)
        self.clients = []
        self.members = []
        self.requests = []
        self.baseline = None
        self.enabled = self.bound = False
        self.inflight = None
        self.recovery = None
        self.canary_evidence = None
        self.cleanup_evidence = None
        self.deployment = None
        self._lock = threading.RLock()

    def _binding(self, action, **settings):
        rows=self.executor.collective_rpc('demoncat_product_binding', timeout=self.timeout,
            args=(action, dict(operator=self.operator, profile=self.profile, **settings)))
        if not rows or any(type(r.get('rank')) is not int for r in rows) or sorted(r['rank'] for r in rows)!=list(range(len(self.members))):
            raise RuntimeError('PRODUCT_BINDING_RANK_SET_CHANGED')
        return sorted(rows,key=lambda r:r['rank'])

    def attach(self):
        if self.clients:
            raise RuntimeError('PRODUCT_ALREADY_ATTACHED')
        self.members = sorted([r['worker'] for r in self.host.discover_workers()], key=lambda r: r['rank'])
        if [r['rank'] for r in self.members] != list(range(len(self.members))):
            raise RuntimeError('PRODUCT_COMPLETE_RANK_GROUP_REQUIRED')
        if self.deployment is not None and len(self.members) != self.deployment['tp']:
            raise RuntimeError('PRODUCT_DEPLOYMENT_WORLD_SIZE_MISMATCH')
        for member in self.members:
            host = HostAdapter(self.executor, select_worker=lambda c, rows, m=member: m, timeout=self.timeout)
            self.clients.append(WorkerFaultClient(host, self.config, _group_identity=self.identity, receipt_mode='latest'))
        return dict(status='PRODUCT_ATTACHED', world_size=len(self.members), product_qualified=False)

    def capabilities(self):
        if not self.clients:
            self.attach()  # census only; acquires no binding, permit, hook or lease
        return self._binding('capabilities')

    def _all(self, action):
        rows = [getattr(client, action)() for client in self.clients]
        expected = 'QUERY_OK' if action == 'query' else 'DETACHED' if action == 'detach' else 'ACTION_COMPLETED'
        if len(rows) != len(self.members) or not all(r['status'] == expected for r in rows):
            raise RuntimeError('PRODUCT_ALL_RANK_' + action.upper() + '_FAILED:' + repr(rows))
        compact=[]
        for row in rows:
            state=deepcopy(row.get('response',{}))
            state.pop('provider',None)
            if action in ('disable','unregister','detach') and isinstance(state.get('provider_state'),dict):
                state['provider_state'].pop('last_request',None)
            compact.append(dict(status=row['status'],worker=row.get('worker'),response=state,
                ownership_retained=row.get('ownership_retained'),ticket=row.get('ticket'),attachment=row.get('attachment')))
        return compact

    def _canary(self):
        def before(ids):
            if not isinstance(ids, list) or len(ids) != 1:
                raise RuntimeError('PRODUCT_CANARY_ACTUAL_INTERNAL_ID_REQUIRED')
            self._binding('canary', request_ids=ids)
        self.run_canary(before)  # must wait for actual engine completion
        result = self._binding('canary')
        if len(result) != len(self.members):
            raise RuntimeError('PRODUCT_CANARY_RANK_MISSING')
        return result

    def enable(self, scope=None):
        with self._lock:
            if not self.clients or self.bound:
                raise RuntimeError('PRODUCT_NEW_ATTACHMENT_REQUIRED')
            policy = RequestScope(**(scope or {}))
            contract = dict(owner=self.identity.owner, group_generation=1, members=self.members,
                members_sha256=digest(self.members), mechanism=self.config.mechanism,
                tp=len(self.members), injection_policy='ALL_RANK_EVERY_ELIGIBLE_CALL')
            self.bound = True  # retain cleanup responsibility even if one rank rejects attach
            self._binding('attach', group_contract=contract, scope=dict(mode=policy.mode, count=policy.count))
            for unused in range(6):
                observations = self._canary()
                if all(row['warmup_valid'] for row in observations):
                    break
            else:
                raise RuntimeError('PRODUCT_CANARY_UNSTABLE_IN_SIX')
            self.baseline = [deepcopy(row['observations'][-1]['output']) for row in observations]
            self.canary_evidence = dict(before=deepcopy(observations), after=None)
            for action in ('prepare', 'register', 'check', 'enable'):
                self._all(action)
            self.enabled = True
            return self.status()

    def before_request(self, internal_request_id):
        """Main-project scheduler hook, after enqueue and before any execution."""
        with self._lock:
            if not self.enabled:
                return dict(status='FAULT_DISABLED')
            if self.inflight is not None:
                raise RuntimeError('PRODUCT_SERIAL_HOST_SCHEDULER_REQUIRED')
            self.inflight = internal_request_id
            rows = [c.advance(internal_request_id) for c in self.clients]
            if not all(r['status'] == 'ACTION_COMPLETED' for r in rows):
                self.enabled = False
                raise RuntimeError('PRODUCT_REQUEST_ARM_FAILED:' + repr(rows))
            return dict(status='REQUEST_OBSERVER_ARMED', request_id=internal_request_id)

    def after_request(self, internal_request_id):
        """Result-export hook; invoke only after the engine completed this ID."""
        with self._lock:
            if self.inflight != internal_request_id:
                raise RuntimeError('PRODUCT_COMPLETION_ID_MISMATCH')
            rows = self._all('collect_replay')
            effects = [r['response']['provider_state']['last_request'] for r in rows]
            if any(e['request_id'] != internal_request_id or not e['completed'] for e in effects):
                raise RuntimeError('PRODUCT_RESULT_REQUEST_MAPPING_CHANGED')
            record = dict(request_id=internal_request_id, ranks=effects)
            self.requests.append(record)
            self.inflight = None
            return deepcopy(record)

    def status(self):
        return dict(enabled=self.enabled, inflight=self.inflight, requests=deepcopy(self.requests),
                    recovery=deepcopy(self.recovery), canary=deepcopy(self.canary_evidence),
                    cleanup=deepcopy(self.cleanup_evidence), product_qualified=False)

    def disable(self):
        with self._lock:
            self.enabled = False  # no new request admission while draining
        self.drain()
        with self._lock:
            if self.inflight is not None:
                raise RuntimeError('PRODUCT_DRAIN_DID_NOT_COMPLETE_RESULT_HOOK')
            self.cleanup_evidence = {action:self._all(action) for action in ('disable','unregister','detach')}
            self._binding('reset_canary_observer')
            actual = self._canary()
            self.canary_evidence['after'] = deepcopy(actual)
            same = all(r['observations'][-1]['output'] == reference and r['observations'][-1]['all_finite']
                       for r, reference in zip(actual, self.baseline))
            bytes_ok = all(r['bytes_restored'] and r['coverage']['failed'] == 0
                           for req in self.requests for r in req['ranks'])
            zero = released_resources(self.cleanup_evidence)
            self.recovery = dict(bytes_restored=bytes_ok, resources_zero=zero, canary_exact=same,
                                 restored=bytes_ok and zero and same)
            if not self.recovery['restored']:
                raise RuntimeError('PRODUCT_RECOVERY_NOT_PROVEN')
            return self.status()

    def detach(self):
        if self.enabled or self.inflight is not None or not self.recovery or not self.recovery['restored']:
            raise RuntimeError('PRODUCT_CONFIRMED_RECOVERY_REQUIRED')
        self._binding('detach')
        self.bound = False
        return dict(status='PRODUCT_DETACHED', recovery=deepcopy(self.recovery))
