"""Bounded same-stream sample snapshots, verified at a shared forward boundary.

CPU-tested optimization candidate, not enabled by the released Batch binding.
Does not defer input restoration, release object leases or grant acceptance.
"""
import numpy as np
import torch
from .quant_reference import compare_samples


class DeferredQuantReference:
    def __init__(self,*,stream,complete,max_calls=512,max_bytes=32*1024*1024,sample_limit=1024):
        if any(type(n) is not int or n<1 for n in (max_calls,max_bytes,sample_limit)) or sample_limit<2:
            raise ValueError('INVALID_REFERENCE_QUEUE_BUDGET')
        self.stream=stream;self.complete=complete;self.max_calls=max_calls
        self.max_bytes=max_bytes;self.sample_limit=sample_limit;self.rows=[];self.bytes=0

    def enqueue(self,key,x,q,scales):
        if any(r['key']==key for r in self.rows):raise ValueError('DUPLICATE_REFERENCE_CALL')
        if x.ndim!=2 or min(x.shape)<1 or not x.is_floating_point() or q.shape!=x.shape or q.dtype!=torch.int8:
            raise ValueError('INVALID_REFERENCE_LAYOUT')
        if scales.shape!=(x.shape[0],) or scales.dtype!=torch.float32 or x.device!=q.device or x.device!=scales.device:
            raise ValueError('INVALID_REFERENCE_SCALE_LAYOUT')
        current=self.stream()
        if self.rows and (current!=self.rows[0]['stream'] or x.device!=self.rows[0]['device']):
            raise ValueError('REFERENCE_STREAM_OR_DEVICE_CHANGED')
        n=x.numel();count=min(n,self.sample_limit-1)+1
        estimate=count*(4+1+8)+scales.numel()*4+4
        # Reserve both retained snapshots and the equally-sized packed device
        # buffers. This budget excludes the operator's transient FP32 x/abs.
        if len(self.rows)>=self.max_calls or 2*(self.bytes+estimate)>self.max_bytes:
            raise ValueError('REFERENCE_SNAPSHOT_BUDGET_EXCEEDED')
        flat=x.detach().float().reshape(-1);maximum,index=flat.abs().max(dim=0)
        grid=torch.linspace(0,n-1,count-1,device=x.device).to(torch.int64)
        indexes=torch.cat((grid,index.reshape(1)))
        # No CPU conversion/item/unique here. Snapshots cannot alias live buffers.
        row=dict(key=key,total=n,stream=current,device=x.device,
            values=flat[indexes].clone(),q=q.detach().reshape(-1)[indexes].clone(),
            scales=scales.detach().clone(),maximum=maximum.reshape(1).clone(),indexes=indexes.clone())
        self.rows.append(row);self.bytes+=estimate

    def flush(self):
        if not self.rows:return []
        if self.stream()!=self.rows[0]['stream']:raise ValueError('REFERENCE_STREAM_CHANGED_BEFORE_FLUSH')
        # Pack on the same stream before one completion fence. Three transfers
        # serve the entire forward. Validation failures retain snapshots.
        floats=torch.cat([t for r in self.rows for t in (r['values'],r['scales'],r['maximum'])])
        codes=torch.cat([r['q'] for r in self.rows]);indices=torch.cat([r['indexes'] for r in self.rows])
        self.complete(self.rows[0]['stream'])
        floats=floats.cpu().numpy();codes=codes.cpu().numpy();indices=indices.cpu().numpy()
        result=[];fo=qo=0
        for r in self.rows:
            count=r['values'].numel();size=r['scales'].numel()
            # Same distinct deterministic grid + extremum as immediate reference.
            _,first=np.unique(indices[qo:qo+count],return_index=True);first.sort()
            values=floats[fo:fo+count][first];actual_q=codes[qo:qo+count][first]
            scales=floats[fo+count:fo+count+size];maximum=floats[fo+count+size]
            report=compare_samples(values,actual_q,scales,maximum,total_elements=r['total'])
            result.append(dict(key=r['key'],**report));fo+=count+size+1;qo+=count
        self.rows.clear();self.bytes=0
        return result

    def pending(self):return len(self.rows)
