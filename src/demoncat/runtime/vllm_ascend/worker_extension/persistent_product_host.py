"""Trusted v2 host facade; uses the existing all-rank Runtime lifecycle.

No v1 ProductClient or binding mutation. The deployment owns the HTTP ingress
and real backend drain. Device fences and byte evidence come from Workers.
This facade does not grant registration or install a provider from user input.
"""
from copy import deepcopy
from .external_persistent_runtime import PersistentFaultConfig
from .persistent_ingress import PersistentIngress
from .product_host import ProductClient, released_resources
from .serving_context import require


class PersistentProductClient(ProductClient):
    def __init__(self,executor,config,*,operator,ingress,run_canary,
                 canary_request,control_boundary,timeout=120):
        require(type(config) is PersistentFaultConfig and type(ingress) is PersistentIngress,
                'EXTERNAL_V2_TRUSTED_CONFIG_AND_INGRESS_REQUIRED')
        require(callable(control_boundary),'EXTERNAL_V2_WORKER_BOUNDARY_REQUIRED')
        super().__init__(executor,config,profile=config.mechanism,operator=operator,
                         run_canary=run_canary,drain=lambda:None,timeout=timeout)
        self.ingress=ingress;self.canary_request=deepcopy(canary_request)
        self.control_boundary=control_boundary

    def _pause(self):
        snapshot=self.ingress.pause_and_drain(self.identity.owner,timeout=self.timeout)
        # This is a trusted deployment callback, not a flag supplied by a user.
        # Each Worker must verify its own idle runner and complete its stream.
        rows=self.control_boundary(self.identity.owner,snapshot)
        require(type(rows) is list and len(rows)==len(self.members)
                and sorted(r.get('rank',-1) for r in rows)==list(range(len(self.members)))
                and all(r.get('device_readers_complete') is True and r.get('inflight')==0
                        and r.get('owner')==self.identity.owner for r in rows),
                'EXTERNAL_V2_ALL_RANK_DRAIN_UNPROVEN')
        return rows

    def _canary(self):
        with self.ingress.canary_request(self.identity.owner,self.canary_request):
            result=self.run_canary()
        self._pause()
        require(type(result) is list and len(result)==len(self.members),
                'EXTERNAL_V2_CANARY_ALL_RANKS_REQUIRED')
        rows=sorted(result,key=lambda r:r.get('rank',-1))
        require([r.get('rank') for r in rows]==list(range(len(self.members)))
                and all(r.get('all_finite') is True and isinstance(r.get('output'),dict)
                        and r['output'] for r in rows),'EXTERNAL_V2_CANARY_EVIDENCE_REQUIRED')
        return deepcopy(rows)

    def capabilities(self):
        if not self.clients:self.attach()
        # Runtime discovery includes trusted v2 registrations; no v1 profile RPC.
        return self.host.discover_workers()

    def enable(self,scope=None):
        with self._lock:
            require(scope in (None,{}, {'mode':'all'}),'EXTERNAL_V2_ALL_REQUESTS_ONLY')
            require(self.clients and not self.bound,'PRODUCT_NEW_ATTACHMENT_REQUIRED')
            self.bound=True
            try:
                self._pause();previous=None
                for unused in range(6):
                    current=self._canary()
                    if current==previous:break
                    previous=current
                else:raise RuntimeError('PRODUCT_CANARY_UNSTABLE_IN_SIX')
                self.baseline=current;self.canary_evidence=dict(before=current,after=None)
                for action in ('prepare','register','check','enable'):self._all(action)
                # Reopen only after every rank committed. No conditional hit
                # invokes a collective; this is the common host control boundary.
                self.enabled=True;self.ingress.resume(self.identity.owner)
                return self.status()
            except Exception:
                self._enable_failure_pending=True
                self._rollback_enable()
                raise

    def _rollback_enable(self):
        self._pause();rows=[]
        for client in self.clients:
            try:
                state=client.snapshot()
                rows.append(client.detach() if state['attachment'] is not None or state['ownership_retained']
                            else dict(status='NOT_ATTACHED',ownership_retained=False))
            except Exception as error:rows.append(dict(status='CLEANUP_FAILED',reason=str(error)))
        self.cleanup_evidence=dict(enable_rollback=rows)
        require(all(r.get('status') in ('DETACHED','NOT_ATTACHED') and r.get('ownership_retained') is False
                    for r in rows),'EXTERNAL_V2_RECOVERY_REQUIRED_INGRESS_REMAINS_CLOSED')
        self.ingress.release(self.identity.owner)
        self.enabled=self.bound=self._enable_failure_pending=False
        self.recovery=dict(resources_zero=True,enable_rollback=True,canary_exact=None,restored=False)

    def status(self):
        rows=self._all('query') if self.enabled else []
        return dict(enabled=self.enabled,ranks=rows,recovery=deepcopy(self.recovery),
                    canary=deepcopy(self.canary_evidence),cleanup=deepcopy(self.cleanup_evidence),
                    ingress=self.ingress.snapshot(),product_qualified=False)

    def disable(self):
        with self._lock:
            if self._enable_failure_pending:
                self._rollback_enable();return self.status()
            require(self.bound,'PRODUCT_NEW_ATTACHMENT_REQUIRED')
            self._pause()
            # Any exception retains the closed ingress and existing ownership.
            stopped=self._all('disable');self.enabled=False
            states=[r['response']['provider_state'] for r in stopped]
            require(all(s.get('recovery',{}).get('bytes_restored') is True for s in states),
                    'EXTERNAL_V2_BYTES_NOT_RESTORED')
            self.cleanup_evidence=dict(disable=stopped,unregister=self._all('unregister'),detach=self._all('detach'))
            actual=self._canary();self.canary_evidence['after']=actual
            same=actual==self.baseline;zero=released_resources(self.cleanup_evidence)
            self.recovery=dict(bytes_restored=True,resources_zero=zero,canary_exact=same,restored=same and zero)
            require(self.recovery['restored'],'PRODUCT_RECOVERY_NOT_PROVEN')
            return self.status()

    def detach(self):
        with self._lock:
            require(not self.enabled and self.recovery and self.recovery.get('restored') is True,
                    'PRODUCT_CONFIRMED_RECOVERY_REQUIRED')
            self.ingress.release(self.identity.owner);self.bound=False
            return dict(status='PRODUCT_DETACHED',recovery=deepcopy(self.recovery))
