"""Bounded real KV growth after the one already qualified Fault33 length.

No new fault geometry is admitted: only KV16384 uses the existing audited
selector. Following lengths use stock paged IFA, with exact valid-page checks.
This HF route is not a common vLLM Worker provider qualification.
"""
import math
import os
import time
import threading
import uuid
from contextlib import contextmanager

from ..attention import AttentionSession
from ..core import Decision,UnsupportedCall
from ..native import NativeCall
from .paged33 import Paged33Runtime,pack_pages,digest


def pack_partial_pages(x,block=128):
    import torch
    if x.ndim!=4 or type(block) is not int or block<=0:raise UnsupportedCall('Invalid page geometry')
    b,h,length,d=x.shape
    if length<1:raise UnsupportedCall('Empty KV is unsupported')
    count=(length+block-1)//block;tail=count*block-length
    padded=torch.nn.functional.pad(x,(0,0,0,tail)) if tail else x
    pages=pack_pages(padded,block)
    mapping=[list(range(i*count,(i+1)*count)) for i in range(b)]
    for i,ids in enumerate(mapping):
        inverse=pages[ids[0]:ids[-1]+1].permute(1,0,2,3).reshape(h,count*block,d)
        if not torch.equal(inverse[:,:length],x[i]):raise UnsupportedCall('Valid KV page inverse differs')
        if tail and not bool((inverse[:,length:]==0).all()):raise UnsupportedCall('Tail padding is not zero')
    return pages,mapping,tail


class ContinuousPaged33Runtime(Paged33Runtime):
    validation_scope='HF actual token/KV growth; fault geometry unchanged; following calls use stock paged IFA'

    def __init__(self,model,*,bundle,evidence_dir,decode_steps=4):
        if type(decode_steps) is not int or not 1<=decode_steps<=8:raise UnsupportedCall('Bounded 1..8 decode steps required')
        super().__init__(model,bundle=bundle,evidence_dir=evidence_dir)
        self.decode_steps=decode_steps;self.previous_length=None;self.previous_cache=None;self.followups=[]
        self.stock_route=AttentionSession(model,self)
        self.stock_started=False

    def root(self,module,args,kwargs):
        ids=kwargs.get('input_ids',args[0] if args else None);cache=kwargs.get('past_key_values')
        if ids is None or tuple(ids.shape)!=(8,1) or cache is None or not hasattr(cache,'layers'):
            raise UnsupportedCall('Actual B8 single-token input and DynamicCache required')
        length=cache.get_seq_length()
        expected=16383 if self.previous_length is None else self.previous_length+1
        if length!=expected or length>=16383+self.decode_steps:raise UnsupportedCall('KV trajectory changed or exceeded')
        if self.previous_cache is not None and cache is not self.previous_cache:raise UnsupportedCall('Live cache identity changed')
        self.previous_length=length;self.previous_cache=cache;self.cache=cache;self.tokens=ids.detach().cpu().tolist()

    def install(self,callback,candidates,mechanism,arm):
        if mechanism!='33.paged_decode':raise UnsupportedCall('Wrong paged33 mechanism')
        self.callback,self.arm=callback,arm
        self.root_handle=self.model.register_forward_pre_hook(self.root,with_kwargs=True)
        try:self.route.__enter__()
        except BaseException:self.root_handle.remove();raise
        def remove():
            self.route.close();self.root_handle.remove()
            if self.pending is not None or any(k.startswith('FAULT33_PA_') for k in os.environ):
                raise RuntimeError('Native scope unresolved')
        return remove

    @contextmanager
    def stock_followups(self):
        # Separate lifetime: later lengths are real normal calls, not admitted
        # fault targets. Never downgrade NativeSession's per-forward coverage.
        if (self.stock_started or self.previous_length!=16383 or not self.calls
                or not self.route._closed or self.pending is not None
                or any(k.startswith('FAULT33_PA_') for k in os.environ)):
            raise UnsupportedCall('Completed and closed first-decode fault scope required')
        self.stock_started=True
        handle=self.model.register_forward_pre_hook(self.root,with_kwargs=True)
        try:
            with self.stock_route:yield self.stock_route
        finally:handle.remove()

    def check(self,call,mechanism=None,arm=None):
        if isinstance(call,NativeCall):return super().check(call,mechanism,arm)
        g=call.geometry()
        if g.get('kv_length')==16384:return super().check(call,mechanism,arm)
        import torch
        try:
            expected=dict(batch=8,query_heads=16,kv_heads=8,query_length=1,
                kv_length=self.previous_length+1,qk_dim=128,value_dim=128)
            if not 16385<=g.get('kv_length',0)<=16383+self.decode_steps or g!=expected:
                raise UnsupportedCall('Unreviewed following-decode geometry')
            if any(t.dtype!=torch.float16 or t.device.type!='npu' for t in (call.query,call.key,call.value)):
                raise UnsupportedCall('FP16 NPU required')
            if call.dropout or call.kwargs.get('output_attentions') or call.kwargs.get('sliding_window'):
                raise UnsupportedCall('Attention attributes unadapted')
            if not torch.are_deterministic_algorithms_enabled() or call.scaling is None or not math.isclose(float(call.scaling),128**-.5,abs_tol=1e-12,rel_tol=0):
                raise UnsupportedCall('Determinism or scale differs')
            if call.attention_mask is not None and not bool((call.attention_mask==0).all()):
                raise UnsupportedCall('Masked decode unsupported')
            index=getattr(call.module,'layer_idx',None)
            if type(index) is not int or not 0<=index<len(self.cache.layers):raise UnsupportedCall('Cache layer unbound')
            for a,b in ((call.key,self.cache.layers[index].keys),(call.value,self.cache.layers[index].values)):
                if a.untyped_storage()._cdata!=b.untyped_storage()._cdata or a.storage_offset()!=b.storage_offset() or a.shape!=b.shape or a.stride()!=b.stride():
                    raise UnsupportedCall('Actual KV cache storage mismatch')
            return Decision(True,'Following length uses stock paged IFA; not fault eligible')
        except (UnsupportedCall,AttributeError,TypeError) as error:return Decision(False,str(error))

    def run(self,call):
        if call.geometry()['kv_length']==16384:
            if self.stock_started:raise UnsupportedCall('Fault geometry re-entered after recovery')
            return super().run(call)
        import json
        import torch,torch_npu
        if self.pending is not None or any(k.startswith('FAULT33_PA_') for k in os.environ):raise UnsupportedCall('Following call has foreign fault scope')
        pk,mapping,tail=pack_partial_pages(call.key);pv,mapping_v,tail_v=pack_partial_pages(call.value)
        if mapping!=mapping_v or tail!=tail_v:raise UnsupportedCall('Key/value page map differs')
        table=torch.tensor(mapping,device=call.query.device,dtype=torch.int32)
        before={k:digest(x) for k,x in (('q',call.query),('key',pk),('value',pv))}
        cid=uuid.uuid4().hex;length=call.geometry()['kv_length'];torch.npu.synchronize()
        marker=dict(role='stock_followup',call_id=cid,kv_length=length,pid=os.getpid(),
            tid=threading.get_native_id(),block_size=128,deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        begin=time.monotonic_ns();print(json.dumps(dict(fault33='enter',monotonic_ns=begin,**marker)),flush=True)
        result=torch_npu.npu_incre_flash_attention(call.query,pk,pv,block_table=table,
            actual_seq_lengths=[length]*8,num_heads=16,num_key_value_heads=8,input_layout='BNSD',
            scale_value=call.scaling,block_size=128,inner_precise=0)
        torch.npu.synchronize();end=time.monotonic_ns()
        print(json.dumps(dict(fault33='exit',monotonic_ns=end,**marker)),flush=True)
        after={k:digest(x) for k,x in (('q',call.query),('key',pk),('value',pv))}
        if before!=after or table.cpu().tolist()!=mapping:raise RuntimeError('Stock IFA modified inputs')
        self.followups.append(dict(call_id=cid,cache_layer=call.module.layer_idx,kv_length=length,
            tokens=self.tokens,table=mapping,tail_padding=tail,inputs_unchanged=True,
            input_hashes=before,output_sha256=digest(result),policy_applied=False,
            scope='STOCK_PAGED_FOLLOWUP_NOT_NEW_FAULT_QUALIFICATION'))
        return result.transpose(1,2).contiguous(),None
