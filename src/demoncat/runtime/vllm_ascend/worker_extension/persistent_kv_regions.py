"""Request-owned KV region leases; same-stream writer/read/end boundaries.

No global idle flag is invented during inference. A trusted adapter supplies
actual page ownership and completion. The existing Runtime owns the scope.
"""
import torch
from fault_injection.engines.region_lease import RegionLeases,identity,exact,version
from .serving_context import require


class RequestKVRegions:
    def __init__(self,leases,*,owner_observation,maximum_bytes):
        require(type(leases) is RegionLeases and callable(owner_observation),'KV_REAL_LEASE_AND_OWNER_REQUIRED')
        require(type(maximum_bytes) is int and maximum_bytes>0,'KV_BACKUP_BUDGET_REQUIRED')
        self.leases=leases;self.owner_observation=owner_observation;self.maximum_bytes=maximum_bytes
        self.entries={};self.events=[];self.reads=0

    def current(self,entry,*,ending=False):
        observed=self.owner_observation()
        require(observed['request_id']==entry['request_id'] and observed['request_generation']==entry['request_generation']
                and entry['page'] in observed['pages'],'KV_PAGE_OWNER_CHANGED_NO_WRITEBACK')
        require(observed.get('before_free') is ending,'KV_WRONG_BOUNDARY')
        require(identity(entry['tensor'])==entry['identity'],'KV_STORAGE_CHANGED_NO_WRITEBACK')
        return self.leases._get(entry['token'])

    def write_after_producer(self,name,tensor,value,*,page):
        require(name not in self.entries,'KV_REGION_ALREADY_WRITTEN')
        observed=self.owner_observation()
        require(observed.get('before_free') is False and page in observed['pages'],'KV_ACTUAL_PRIVATE_PAGE_REQUIRED')
        used=sum(e['tensor'].numel()*e['tensor'].element_size() for e in self.entries.values())
        event=dict(name=name,observed=1,eligible=1,attempted=1,applied=0,changed=0,changed_elements=0,
                   consumed=0,restored=0,failed=0,readers=0)
        self.events.append(event)
        try:
            require(used+tensor.numel()*tensor.element_size()<=self.maximum_bytes,'KV_BACKUP_BUDGET_EXCEEDED')
            require(isinstance(value,torch.Tensor) and value.shape==tensor.shape and value.dtype==tensor.dtype
                    and value.device==tensor.device,'KV_VALUE_LAYOUT_MISMATCH')
            token=self.leases.acquire(tensor)
            entry=dict(tensor=tensor,identity=identity(tensor),token=token,page=page,
                       request_id=observed['request_id'],request_generation=observed['request_generation'],event=event)
            self.entries[name]=entry
            changed=int((tensor.view(torch.uint8).reshape(-1,tensor.element_size())!=value.contiguous().view(torch.uint8).reshape(-1,tensor.element_size())).any(1).sum())
            with torch.no_grad():self.leases.write(token,value)
            event.update(applied=1,changed=int(changed>0),changed_elements=changed)
        except Exception:event['failed']=1;raise

    def acknowledge_disjoint_writer(self,name):
        e=self.entries[name];lease=self.current(e)
        self.leases.complete(lease['stream'])
        # A different KV slot can advance the shared tensor version. Accept
        # that version only when every byte of our held region is unchanged.
        require(exact(e['tensor'],lease['expected']),'KV_REGION_LEGAL_UPDATE_NO_WRITEBACK')
        lease['version']=version(e['tensor'])

    def consumed(self,name):
        e=self.entries[name];lease=self.current(e)
        require(exact(e['tensor'],lease['expected']),'KV_CONSUMER_VALUE_MISMATCH')
        self.leases.consumed(e['token']);e['event']['consumed']=1;e['event']['readers']+=1;self.reads+=1

    def restore_before_free(self):
        for name in tuple(self.entries):
            e=self.entries[name];lease=self.current(e,ending=True)
            require(exact(e['tensor'],lease['expected']),'KV_END_UNEXPECTED_LEGAL_UPDATE')
            if e['event']['consumed']!=1:
                require(self.owner_observation().get('request_status')=='FINISHED_ABORTED','KV_CONSUMER_NOT_OBSERVED')
                # Actual scheduler abort before page free, same stream complete.
                # Restore without inventing a consumer or a successful hit.
                self.leases.complete(lease['stream'])
                with torch.no_grad():e['tensor'].copy_(lease['backup'])
                self.leases.complete(lease['stream'])
                require(exact(e['tensor'],lease['backup']),'KV_ABORT_BYTES_NOT_RESTORED')
                del self.leases.active[e['token']]
                e['event'].update(restored=1,restore_kind='ABORT_BEFORE_CONSUMER_BYTE_ROLLBACK')
                del self.entries[name];continue
            # Other rows may have been written since the last observed reader.
            lease['version']=version(e['tensor'])
            with torch.no_grad():self.leases.restore(e['token'])
            e['event'].update(restored=1,restore_kind='BYTE_EXACT_SAME_REQUEST_GENERATION_BEFORE_PAGE_FREE');del self.entries[name]
