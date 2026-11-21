"""Per-hit live-object leases for ordered torch rewrites, not a task lifecycle.

Strong tensor/storage references prevent allocator reuse during a lease. A
different storage/shape or an intervening in-place mutation prevents writeback.
Only contiguous regions and an explicitly completed same-stream consumer are
supported. The owner must retain failed leases for explicit recovery review.
"""
import torch


def version(t):
    # vLLM inference tensors intentionally have no version counter. They need
    # byte comparison against the held expected value before restoration.
    try:return t._version
    except RuntimeError:return None


def exact(a,b):return torch.equal(a.view(torch.uint8),b.view(torch.uint8))


def identity(t):
    if not isinstance(t,torch.Tensor) or not t.is_contiguous() or t.numel()==0:
        raise ValueError('CONTIGUOUS_LIVE_TENSOR_REGION_REQUIRED')
    storage=t.untyped_storage()
    start=t.storage_offset()*t.element_size();end=start+t.numel()*t.element_size()
    if end>storage.nbytes():raise ValueError('REGION_OUTSIDE_LIVE_STORAGE')
    return (str(t.device),storage._cdata,start,end,tuple(t.shape),t.dtype,
            storage.data_ptr(),storage.nbytes())


class RegionLeases:
    def __init__(self,owner,generation,*,stream,complete):
        if not owner or type(generation) is not int or generation<1:
            raise ValueError('LEASE_SCOPE_REQUIRED')
        self.owner=owner;self.generation=generation
        self.stream=stream;self.complete=complete;self.active={};self.sequence=0

    def acquire(self,t):
        key=identity(t)
        for lease in self.active.values():
            other=lease['identity']
            if key[:2]==other[:2] and max(key[2],other[2])<min(key[3],other[3]):
                raise ValueError('OVERLAPPING_LIVE_REGION_LEASE')
        current=self.stream();backup=t.detach().clone()
        self.complete(current)  # backup completion, before any modification
        self.sequence+=1
        self.active[self.sequence]=dict(tensor=t,storage=t.untyped_storage(),identity=key,
            backup=backup,expected=backup,stream=current,version=version(t),written=False,consumed=False)
        return self.sequence

    def _get(self,token):
        lease=self.active[token]
        if identity(lease['tensor'])!=lease['identity']:
            raise RuntimeError('LEASE_OBJECT_STORAGE_CHANGED_NO_WRITEBACK')
        if self.stream()!=lease['stream']:
            raise RuntimeError('LEASE_STREAM_CHANGED')
        return lease

    def write(self,token,value):
        lease=self._get(token);t=lease['tensor']
        if lease['written'] or version(t)!=lease['version'] or not exact(t,lease['expected']):
            raise RuntimeError('LEASE_INTERVENING_MUTATION')
        t.copy_(value);self.complete(lease['stream'])
        lease.update(written=True,version=version(t),expected=value.detach().clone())

    def consumed(self,token):
        lease=self._get(token)
        self.complete(lease['stream'])
        lease['consumed']=True

    def observe_output_copy(self,token,expected):
        """Trusted site observes its returned value copied into a held output.

        Not a write primitive: the current bytes must equal the known producer
        value. It is used before the real consumer, never after arbitrary work.
        """
        lease=self._get(token)
        if lease['written'] or not exact(lease['tensor'],expected):
            raise RuntimeError('PRODUCER_OUTPUT_COPY_NOT_PROVEN')
        lease.update(written=True,version=version(lease['tensor']),expected=expected.detach().clone())

    def restore(self,token):
        lease=self._get(token);t=lease['tensor']
        if not lease['consumed'] or version(t)!=lease['version'] or not exact(t,lease['expected']):
            raise RuntimeError('CONSUMER_INCOMPLETE_OR_LEGAL_UPDATE_NO_WRITEBACK')
        t.copy_(lease['backup']);self.complete(lease['stream'])
        if not torch.equal(t.view(torch.uint8),lease['backup'].view(torch.uint8)):
            raise RuntimeError('REGION_BYTES_NOT_RESTORED')
        del self.active[token]

    def count(self):return len(self.active)
