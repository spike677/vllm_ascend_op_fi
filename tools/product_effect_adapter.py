"""Evidence adapter for the existing ProductClient (public and external v1).

HCCL preload adapters use the same probe protocol, but must supply their own
audited reader of actual communication/restore receipts; this class grants no
HCCL qualification and does not invent missing counts.
"""
from tools.product_effect_probe import COUNTERS, require
from demoncat.runtime.vllm_ascend.worker_extension.product_host import released_resources


class ProductSession:
    def __init__(self, client):
        self.client = client
        self.owner = client.identity.owner
        self.worker_identity = sorted([r['worker'] for r in client.host.discover_workers()], key=lambda r:r['rank'])

    def attach(self): return self.client.attach()
    def enable(self): return self.client.enable()
    def status(self): return self.client.status()
    def disable(self):
        self.stopped = self.client.disable()
        return self.stopped
    def detach(self): return self.client.detach()

    def evidence(self):
        d = self.stopped; before, after = d['canary']['before'], d['canary']['after']
        require(released_resources(d['cleanup']), 'RUNTIME_RESOURCE_PROOF_MISSING')
        states = {r['worker']['rank']: r['response']['provider_state'] for r in d['cleanup']['unregister']}
        result = []
        for rank in range(len(self.worker_identity)):
            rows = [next(r for r in request['ranks'] if r['rank']==rank) for request in d['requests']]
            require(rows and all(r['completed'] for r in rows), 'ACTUAL_COMPLETED_REQUESTS_REQUIRED')
            frames = {}
            batch = any('frames' in r for r in rows)
            counts = dict.fromkeys(COUNTERS, 0)
            elements = 0
            for r in rows:
                require(r['resources'] == dict(leases=0,hooks=0,inflight=0) and r['bytes_restored'], 'REQUEST_RECOVERY_MISSING')
                if batch:
                    for frame in r['frames']:
                        key=frame['index']
                        require(key not in frames or frames[key]==frame, 'BATCH_SHARED_FRAME_CONTRADICTION')
                        frames[key]=frame
                else:
                    for k in COUNTERS: counts[k] += r['coverage'][k]
                    require(type(r.get('changed_elements')) is int, 'CHANGED_ELEMENTS_MISSING')
                    elements += r['changed_elements']
                for s in r['sites']:
                    if not s['eligible']: continue
                    if batch:
                        require(s['eligible']==s['consumed']==s['restored']==s['reference_passed'], 'BATCH_QMM_PROOF_MISSING')
                    else:
                        require(s.get('consumer_operator') and s['consumed'] and s['restored'], 'CONSUMER_UNPROVEN')
                        require(s.get('before_sha256') and s['before_sha256']==s.get('restored_sha256'), 'REGION_BYTES_UNPROVEN')
            if batch:
                # No duplicated per-request participation counts. Frame elements
                # must be supplied by the real observer, never inferred from calls.
                for frame in frames.values():
                    for k in COUNTERS: counts[k] += frame['coverage'][k]
                    require(type(frame.get('changed_elements')) is int, 'BATCH_FRAME_CHANGED_ELEMENTS_MISSING')
                    elements += frame['changed_elements']
            b = next(x for x in before if x['rank']==rank)['observations']
            a = next(x for x in after if x['rank']==rank)['observations']
            require(len(b)>=2 and a and all(x['all_finite'] for x in b[-2:]+a[-1:])
                    and b[-2]['output']==b[-1]['output']==a[-1]['output'], 'FULL_CANARY_NOT_EXACT')
            receipt=next(r for r in d['cleanup']['unregister'] if r['worker']['rank']==rank)
            require(receipt['attachment']['tenant_id']==self.client.identity.tenant_id, 'RANK_TENANT_MISMATCH')
            result.append(dict(rank=rank,owner=self.owner,counts=counts,changed_elements=elements,
                provider_owner=states[rank]['owner'],
                request_ids=[r['request_id'] for r in rows], evidence_paths=[
                    'disabled.requests.*.ranks', 'disabled.cleanup.unregister', 'disabled.canary'],
                consumer=True,bytes_restored=True,resources_zero=True,canary_exact=True))
        return dict(ranks=result,logits_changed=None)


class ProductAdapter:
    """Trusted deployment supplies a client factory and its existing HTTP sender."""
    def __init__(self, client_factory, infer, identity):
        self.client_factory, self.sender, self.actual_identity = client_factory, infer, identity
    def identity(self): return self.actual_identity
    def new_session(self): return ProductSession(self.client_factory())
    def infer(self, prompts, sampling): return self.sender(prompts, sampling)
