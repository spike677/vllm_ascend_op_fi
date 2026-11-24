"""Main-project facade over existing WorkerFaultClient leases.

The Worker discovers user requests; the trusted host only sends canaries. This
module creates no Worker and imports no experiment. User input is never copied
into a deployment profile. No host request-start or request-completion hook is required.
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

        run_canary receives deployment_canary_config and sends ordinary HTTP inference.
        The existing service owns tokenization, image preprocessing and execution.
        Worker bootstrap independently checks this file against its live model.
        """
        from .product_config import load_product_deployment
        deployment = load_product_deployment(path)
        client = cls(executor, FaultConfig.from_dict(deployment['fault']),
                     profile=deployment['profile'], operator=operator, drain=drain, timeout=timeout,
                     run_canary=lambda: run_canary(deepcopy(deployment['canary'])))
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
        self._enable_failure_pending = False
        self.cursor = 0
        self.recovery = None
        self.canary_evidence = None
        self.cleanup_evidence = None
        self.deployment = None
        self.registration = None
        self._lock = threading.RLock()

    def _binding(self, action, **settings):
        rows=self.executor.collective_rpc('demoncat_product_binding', timeout=self.timeout,
            args=(action, dict(operator=self.operator, profile=self.profile, **settings)))
        if not rows or any(type(r.get('rank')) is not int for r in rows) or sorted(r['rank'] for r in rows)!=list(range(len(self.members))):
            raise RuntimeError('PRODUCT_BINDING_RANK_SET_CHANGED')
        rows = sorted(rows,key=lambda r:r['rank'])
        errors = [r for r in rows if r.get('status') in ('PRODUCT_BINDING_REFUSED', 'PRODUCT_BINDING_ERROR')]
        if errors:
            error_type = ValueError if all(r['status']=='PRODUCT_BINDING_REFUSED' for r in errors) else RuntimeError
            raise error_type('PRODUCT_BINDING_' + action.upper() + '_FAILED:' + repr(errors))
        return rows

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
            refused = {'PROVIDER_UNAVAILABLE', 'NATIVE_PRELOAD_NOT_ATTACHED', 'REJECTED', 'HOST_REJECTED'}
            def contract_refusal(row):
                if row['status']==expected:return True
                if row['status'] not in refused:return False
                reply=row.get('response',{})
                # Provider construction and host transport bugs remain exit 5.
                return ('error_type' not in reply or reply.get('provider_code') is not None
                        or reply['error_type'] in ('ValueError','PermissionError','ContextRefused','HostContractError'))
            error_type = ValueError if all(contract_refusal(r) for r in rows) else RuntimeError
            raise error_type('PRODUCT_ALL_RANK_' + action.upper() + '_FAILED:' + repr(rows))
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
        self._binding('arm_canary')
        self.run_canary()
        result = self._binding('canary')
        if len(result) != len(self.members):
            raise RuntimeError('PRODUCT_CANARY_RANK_MISSING')
        return result

    def enable(self, scope=None):
        with self._lock:
            # A second enable must never roll back an already active session.
            if not self.clients or self.bound:
                raise ValueError('PRODUCT_NEW_ATTACHMENT_REQUIRED')
            if self.recovery and self.recovery.get('enable_rollback') and self.recovery.get('resources_zero'):
                self.clients = [WorkerFaultClient(HostAdapter(self.executor,
                    select_worker=lambda c, rows, m=member: m, timeout=self.timeout), self.config,
                    _group_identity=self.identity, receipt_mode='latest') for member in self.members]
                self.recovery = None
            try:
                return self._enable(scope)
            except Exception as error:
                if self.bound:
                    self._enable_failure_pending = True
                    try:
                        self._rollback_enable()
                    except Exception as cleanup_error:
                        raise RuntimeError('PRODUCT_ENABLE_RECOVERY_REQUIRED:' + str(error)
                                           + '; cleanup=' + str(cleanup_error)) from error
                raise

    def _rollback_enable(self):
        """Use Runtime tenant cleanup, including prepare without a lease ticket.

        Every rank is attempted even if another rejects cleanup. Ownership stays
        retained until actual detach acknowledgements and binding removal agree.
        """
        self.enabled = False
        self.drain()
        rows = []
        for client in self.clients:
            try:
                state = client.snapshot()
                if state['attachment'] is None and not state['ownership_retained']:
                    rows.append(dict(status='NOT_ATTACHED', resources_zero=True))
                else:
                    rows.append(client.detach())
            except Exception as error:
                rows.append(dict(status='CLEANUP_FAILED', reason=str(error)))
        self.cleanup_evidence = dict(enable_rollback=rows)
        if not all(r.get('status')=='NOT_ATTACHED' or
                   (r.get('status')=='DETACHED' and r.get('ownership_retained') is False) for r in rows):
            raise RuntimeError('PRODUCT_ENABLE_RUNTIME_CLEANUP_INCOMPLETE')
        removed = self._binding('rollback', owner=self.identity.owner)
        self.cleanup_evidence['binding_rollback'] = removed
        if not all(r.get('status')=='PRODUCT_BINDING_ROLLED_BACK' and r.get('resources_zero') is True for r in removed):
            raise RuntimeError('PRODUCT_ENABLE_BINDING_CLEANUP_INCOMPLETE')
        self.bound = False
        self._enable_failure_pending = False
        # Cleanup is not a successful product acceptance or a canary result.
        self.recovery = dict(resources_zero=True, enable_rollback=True,
                             canary_exact=None, restored=False)

    def _enable(self, scope=None):
        with self._lock:
            if not self.clients or self.bound:
                raise RuntimeError('PRODUCT_NEW_ATTACHMENT_REQUIRED')
            policy = RequestScope(**(scope or {}))
            contract = dict(owner=self.identity.owner, group_generation=1, members=self.members,
                members_sha256=digest(self.members), mechanism=self.config.mechanism,
                tp=len(self.members), injection_policy='ALL_RANK_EVERY_ELIGIBLE_CALL')
            self.bound = True  # retain cleanup responsibility even if one rank rejects attach
            attached=self._binding('attach', group_contract=contract, scope=dict(mode=policy.mode))
            if self.config.mechanism.startswith('external.'):
                sources={r.get('external_source_sha256') for r in attached}
                if len(sources)!=1 or None in sources:
                    raise ValueError('EXTERNAL_RANK_SOURCE_IDENTITY_MISMATCH')
            if any(r.get('mode')=='PRODUCT_REGISTERED' for r in attached):
                ids={r.get('registration_sha256') for r in attached}
                if any(r.get('mode')!='PRODUCT_REGISTERED' for r in attached) or len(ids)!=1 or None in ids:
                    raise RuntimeError('PRODUCT_REGISTRATION_RANK_GROUP_MISMATCH')
                self.registration=ids.pop()
            for unused in range(6):
                observations = self._canary()
                from .batch_canary import stable
                batch=self.config.mechanism=='quant.batch_shared_scale'
                if all(stable(row['observations'],3) if batch else row['warmup_valid'] for row in observations):
                    break
            else:
                raise RuntimeError('PRODUCT_CANARY_UNSTABLE_IN_SIX')
            self.baseline = [deepcopy(row['observations'][-1]['output']) for row in observations]
            self.canary_evidence = dict(before=deepcopy(observations), after=None)
            for action in ('prepare', 'register', 'check', 'enable'):
                self._all(action)
            self.enabled = True
            return self.status()

    def status(self):
        if self.enabled:
            self._collect_status()
        return dict(enabled=self.enabled, requests=deepcopy(self.requests),
                    recovery=deepcopy(self.recovery), canary=deepcopy(self.canary_evidence),
                    cleanup=deepcopy(self.cleanup_evidence), product_qualified=self.registration is not None,
                    registration_sha256=self.registration)

    def _collect_status(self):
        while True:
            rows=self._binding('status',cursor=self.cursor)
            if any(r['failure'] for r in rows):
                raise RuntimeError('PRODUCT_WORKER_OBSERVER_FAILED:'+repr(rows))
            counts=[len(r['requests']) for r in rows]
            if len(set(counts))!=1 or len({r['cursor'] for r in rows})!=1:
                raise RuntimeError('PRODUCT_RANK_REQUEST_COMPLETION_MISMATCH')
            if counts[0]:
                effects=[r['requests'][0] for r in rows]
                ids={r['request_id'] for r in effects}
                if len(ids)!=1:raise RuntimeError('PRODUCT_RANK_INTERNAL_ID_MISMATCH')
                self.requests.append(dict(request_id=effects[0]['request_id'],ranks=effects))
                self.cursor=rows[0]['cursor']
            if not any(r['more'] for r in rows):break

    def disable(self):
        with self._lock:
            if self._enable_failure_pending:
                self._rollback_enable()
                return self.status()
            if not self.bound and self.recovery and self.recovery.get('enable_rollback') and self.recovery.get('resources_zero'):
                return self.status()
            self.enabled = False  # Worker admission ends at the disable RPC control boundary
        self.drain()
        with self._lock:
            stopped=self._all('disable')
            self._collect_status()
            self.cleanup_evidence = dict(disable=stopped,unregister=self._all('unregister'),detach=self._all('detach'))
            self._binding('reset_canary_observer')
            actual = self._canary()
            if self.config.mechanism=='quant.batch_shared_scale':
                actual = self._canary()
            self.canary_evidence['after'] = deepcopy(actual)
            same = all(r['observations'][-1]['output'] == reference and r['observations'][-1]['all_finite']
                       for r, reference in zip(actual, self.baseline))
            if self.config.mechanism=='quant.batch_shared_scale':
                from .batch_canary import stable
                same=all(stable(r['observations'],2,b['observations'][-1]) for r,b in zip(actual,self.canary_evidence['before']))
            bytes_ok = all(r['bytes_restored'] and r['coverage']['failed'] == 0
                           for req in self.requests for r in req['ranks'])
            zero = released_resources(self.cleanup_evidence)
            self.recovery = dict(bytes_restored=bytes_ok, resources_zero=zero, canary_exact=same,
                                 restored=bytes_ok and zero and same)
            if not self.recovery['restored']:
                raise RuntimeError('PRODUCT_RECOVERY_NOT_PROVEN')
            return self.status()

    def detach(self):
        if not self.bound and self.recovery and self.recovery.get('enable_rollback') and self.recovery.get('resources_zero'):
            return dict(status='PRODUCT_DETACHED', recovery=deepcopy(self.recovery))
        if self.enabled or not self.recovery or not self.recovery['restored']:
            raise RuntimeError('PRODUCT_CONFIRMED_RECOVERY_REQUIRED')
        self._binding('detach')
        self.bound = False
        return dict(status='PRODUCT_DETACHED', recovery=deepcopy(self.recovery))
