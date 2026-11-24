"""V2 region primitives, not a product lifecycle or an executable registration.

Trusted adapters must supply real ingress/drain/allocation/page observations.
The existing Runtime owns owner/generation and group rollback. No naked address,
unobserved cache rebuild, implicit model reload or cross-rank collective here.
"""
from dataclasses import dataclass
import torch
from fault_injection.engines.region_lease import RegionLeases, identity, exact, version
from .serving_context import require


@dataclass(frozen=True)
class PersistentRegion:
    name: str
    tensor: object
    region_kind: str
    allocation_generation: object
    observe_generation: object
    consumers: tuple
    # Explicit declaration from the reviewed adapter; None is unknown, not empty.
    derived_copies: object
    request_generation: object = None
    observe_request_generation: object = None


class PersistentRegions:
    """Per-rank region operations under the existing owner/generation lease.

    Holds strong refs and snapshots until an observed safe restore boundary.
    Failure leaves active leases intact. Public provider registration remains off.
    """
    def __init__(self, leases, *, boundary, maximum_bytes):
        require(type(leases) is RegionLeases, 'V2_EXISTING_REGION_LEASES_REQUIRED')
        require(callable(boundary) and type(maximum_bytes) is int and maximum_bytes>0, 'V2_BOUNDARY_BUDGET_REQUIRED')
        self.leases=leases; self.boundary=boundary; self.maximum_bytes=maximum_bytes
        self.entries={}; self.events=[]; self.reader_observations=0

    def _boundary(self):
        b=self.boundary()
        require(b.get('dedicated_instance') is True and b.get('accepting') is False
                and b.get('inflight')==0 and b.get('device_readers_complete') is True,
                'V2_EXCLUSIVE_DRAINED_DEVICE_BOUNDARY_REQUIRED')
        return b

    def _current(self, entry):
        r=entry['region']
        require(r.observe_generation()==r.allocation_generation, 'V2_ALLOCATION_GENERATION_CHANGED_NO_WRITEBACK')
        require(identity(r.tensor)==entry['identity'], 'V2_STORAGE_CHANGED_NO_WRITEBACK')
        if r.region_kind=='kv_cache':
            require(r.observe_request_generation()==r.request_generation, 'V2_PAGE_OWNER_CHANGED_NO_WRITEBACK')
        return r

    def prepare(self, regions):
        b=self._boundary(); require(not self.entries,'V2_REGIONS_ALREADY_PREPARED')
        require(regions and all(type(r) is PersistentRegion for r in regions),'V2_TYPED_REGIONS_REQUIRED')
        names={r.name for r in regions}; require(len(names)==len(regions),'V2_DUPLICATE_REGION')
        total=sum(r.tensor.numel()*r.tensor.element_size() for r in regions)
        require(total<=self.maximum_bytes,'V2_BACKUP_BUDGET_EXCEEDED')
        # Validate all descriptors before acquiring anything.
        keys=[]
        for r in regions:
            require(r.region_kind in ('persistent_weight','kv_cache'),'V2_REGION_KIND_UNSUPPORTED')
            require(r.allocation_generation is not None and callable(r.observe_generation)
                    and r.observe_generation()==r.allocation_generation,'V2_ALLOCATION_UNPROVEN')
            require(r.consumers and all(isinstance(x,str) and x for x in r.consumers),'V2_CONSUMER_PATHS_REQUIRED')
            require(type(r.derived_copies) is tuple and set(r.derived_copies)<=names,'V2_DERIVED_COPIES_UNPROVEN')
            if r.region_kind=='kv_cache':
                require(b.get('prefix_cache') is False and b.get('kv_connector') is False
                        and b.get('offload') is False and b.get('shared_pages') is False,'V2_PRIVATE_KV_ONLY')
                require(r.request_generation is not None and callable(r.observe_request_generation)
                        and r.observe_request_generation()==r.request_generation,'V2_PAGE_OWNER_UNPROVEN')
            k=identity(r.tensor)
            require(not any(k[:2]==p[:2] and max(k[2],p[2])<min(k[3],p[3]) for p in keys),'V2_OVERLAPPING_REGIONS')
            keys.append(k)
        for r,k in zip(regions,keys):
            token=self.leases.acquire(r.tensor)
            self.entries[r.name]=dict(region=r,identity=k,token=token,readers=set(),events=[])
        return dict(regions=len(self.entries),backup_bytes=total)

    def write(self,name,value):
        entry=self.entries[name]
        event=dict(region=name,region_kind=entry['region'].region_kind,observed=1,eligible=1,attempted=1,applied=0,
                   changed=0,changed_elements=0,
                   consumed=0,restored=0,failed=0,restore_kind=None)
        entry['events'].append(event);self.events.append(event)
        try:
            self._boundary();r=self._current(entry)
            require(type(value) is torch.Tensor and value.shape==r.tensor.shape and value.dtype==r.tensor.dtype
                    and value.device==r.tensor.device,'V2_EXACT_VALUE_LAYOUT_REQUIRED')
            require(all(e['consumed'] and not e['failed'] for e in entry['events'][:-1]),
                    'V2_PRIOR_WRITE_READERS_INCOMPLETE')
            lease=self.leases._get(entry['token'])
            before=r.tensor.detach().view(torch.uint8).clone()
            with torch.no_grad():
                if lease['written']:
                    # Keep the original backup for this allocation generation.
                    # Rewrites need their own reader completion before another write.
                    require(version(r.tensor)==lease['version'] and exact(r.tensor,lease['expected']),
                            'V2_INTERVENING_WRITE_NO_OVERWRITE')
                    r.tensor.copy_(value);self.leases.complete(lease['stream'])
                    lease.update(version=version(r.tensor),expected=value.detach().clone(),consumed=False)
                else:
                    self.leases.write(entry['token'],value)
            entry['readers'].clear()
            changed=int((before.reshape(-1,r.tensor.element_size())!=r.tensor.detach().view(torch.uint8).reshape(-1,r.tensor.element_size())).any(dim=1).sum().item())
            event.update(applied=1,changed=int(changed>0),changed_elements=changed)
        except Exception:
            event['failed']=1
            raise

    def consumed(self,name,consumer,*,request_generation):
        entry=self.entries[name];r=self._current(entry)
        require(consumer in r.consumers and request_generation is not None,'V2_ACTUAL_READER_ASSOCIATION_REQUIRED')
        if r.region_kind=='kv_cache':require(request_generation==r.request_generation,'V2_CROSS_REQUEST_KV_READER')
        require(entry['events'] and all(e['applied'] and not e['failed'] for e in entry['events']),'V2_NO_WRITE_TO_CONSUME')
        self.leases.consumed(entry['token'])  # device completion on the held stream
        entry['readers'].add(consumer);self.reader_observations+=1
        entry['events'][-1]['consumed']=int(set(r.consumers)<=entry['readers'])

    def observe_kv_legal_write(self,name,*,request_generation):
        """Only a reviewed KV writer boundary may record its own completed write.

        Preserves current bytes as the expected state, never performs restoration
        or changes allocation ownership. Restore remains forbidden until request
        end and before page release/reuse, with device readers complete.
        """
        entry=self.entries[name];r=self._current(entry)
        require(r.region_kind=='kv_cache' and request_generation==r.request_generation,'V2_KV_WRITER_OWNER_MISMATCH')
        lease=self.leases._get(entry['token'])
        self.leases.complete(lease['stream'])
        lease['expected']=r.tensor.detach().clone();lease['version']=version(r.tensor)
        entry['legal_writes']=entry.get('legal_writes',0)+1

    def restore(self,name):
        b=self._boundary();entry=self.entries[name];r=self._current(entry)
        require(set(r.consumers)<=entry['readers'],'V2_ALL_CONSUMER_PATHS_UNPROVEN')
        if r.region_kind=='kv_cache':
            require(r.request_generation in b.get('ended_request_generations',())
                    and b.get('before_page_release') is True,'V2_KV_SAFE_END_REQUIRED')
        try:
            with torch.no_grad():self.leases.restore(entry['token'])
        except Exception:
            for e in entry['events']:e['failed']=1
            raise
        for e in entry['events']:e.update(restored=1,restore_kind='BYTE_EXACT_SAME_ALLOCATION_GENERATION')
        del self.entries[name]

    def release_unwritten(self,name):
        self._boundary();entry=self.entries[name];self._current(entry)
        require(not entry['events'],'V2_WRITTEN_REGION_REQUIRES_RESTORE')
        self.leases.consumed(entry['token']);self.leases.release_unmodified(entry['token'])
        del self.entries[name]

    def snapshot(self):
        return dict(leases=len(self.entries),reader_observations=self.reader_observations,
                    events=[dict(x) for x in self.events],qualification_granted=False,
                    public_route='CLOSED_PENDING_REAL_ADAPTER_AND_A2_GROUP_ACCEPTANCE')
