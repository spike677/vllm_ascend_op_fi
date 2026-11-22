"""Source-bound real KV span write between ScatterPaKvCache and paged FIA.

Only decode with a proven same-request writer/consumer is eligible. Native
transport owns the byte backup, write and restore; no output rewrite.
"""
import hashlib,inspect
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode,_disable_current_modes
from .gqa28 import SOURCE_SHA
from .mla28 import require,fingerprint
from ..engines.region_lease import exact
from ..deployments.postwrite_boundaries import tensor_geometry


def arguments(func,args,kwargs):
    return {a.name:args[i] if i<len(args) else kwargs.get(a.name,a.default_value)
            for i,a in enumerate(func._schema.arguments)}


class KVMode(TorchDispatchMode):
    def __init__(self,site,name,impl,role,meta=None,event=None):
        super().__init__();self.site=site;self.name=name;self.impl=impl;self.role=role;self.meta=meta;self.event=event;self.calls=0

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {};op=func._schema.name.split('::')[-1]
        if self.role=='writer' and op=='npu_scatter_pa_kv_cache':
            a=arguments(func,args,kwargs);result=func(*args,**kwargs)
            with _disable_current_modes():
                require(a.get('cache_mode')=='Norm','KV41_WRITER_LAYOUT_UNREVIEWED')
                cache=a['value_cache'];slots=a['slot_mapping'].cpu().tolist()
                require(cache is self.impl.value_cache and a['key_cache'] is self.impl.key_cache,'KV41_WRITER_CACHE_OWNER')
                source=a['value'].view(len(slots),cache.shape[-2],cache.shape[-1])
                require(all(exact(cache.view(-1,*cache.shape[-2:])[slot],source[i]) for i,slot in enumerate(slots)),
                    'KV41_ACTUAL_WRITER_VALUES_NOT_CONFIRMED')
                self.site.writers[self.name]=dict(returned=True,key_cache=tensor_geometry(a['key_cache']),
                    value_cache=tensor_geometry(cache),slots=slots,stream=torch.npu.current_stream(),
                    request_ids=list(self.site.runner.input_batch.req_ids))
                self.calls+=1
            return result
        if self.role!='consumer' or op!='npu_fused_infer_attention_score':return func(*args,**kwargs)
        with _disable_current_modes():
            self.calls+=1;a=arguments(func,args,kwargs);e=self.event
            # torch_npu's Tensor[] arguments still refer to the actual cache views.
            for k in ('key','value'):
                if isinstance(a.get(k),(tuple,list)):
                    require(len(a[k])==1,'KV41_MULTI_CACHE_ARGUMENT');a[k]=a[k][0]
            count=self.meta.actual_seq_lengths_q[-1];length=self.meta.seq_lens_list[0]
            if count!=1 or length<=1:
                e.update(eligible=False,reason='PREFILL_DIRECT_KV_NOT_PAGED_DECODE')
                return func(*args,**kwargs)
            writer=self.site.writers.pop(self.name,None)
            require(writer is not None and writer['request_ids']==e['request_ids']
                and writer['stream']==torch.npu.current_stream(),'KV41_CURRENT_REQUEST_WRITER_REQUIRED')
            from demoncat.runtime.vllm_ascend.worker_extension.paged33_audit import audit_paged_call
            scheduler=dict(request_count=1,counts=[1],token_row_count=1,row_token_positions=[length-1])
            proof=audit_paged_call(op,a,impl=self.impl,metadata=self.meta,scheduler=scheduler,writer=writer)
            cache=self.impl.value_cache;slot=proof['slots'][0];region=cache.view(-1,cache.shape[-2]*cache.shape[-1])[slot]
            e.update(eligible=self.site.fault,writer='npu_scatter_pa_kv_cache',consumer_operator=op,
                slots=proof['slots'],page_mapping=proof['used_pages'],region_elements=region.numel())
            if not self.site.fault:return func(*args,**kwargs)
            before=fingerprint(region);token=self.site.native.acquire(region)
            indices=[h*cache.shape[-1]+c for h in range(cache.shape[-2]) for c in range(cache.shape[-1]-2,cache.shape[-1])]
            changed=self.site.native.apply(token,5,indices)
            e.update(applied=True,changed=changed,before_sha256=before,after_sha256=fingerprint(region))
        result=func(*args,**kwargs)
        with _disable_current_modes():
            self.site.native.consumed(token);e['consumed']=True
            self.site.native.restore(token);e.update(restored=True,restored_sha256=fingerprint(region))
            require(e['restored_sha256']==before,'KV41_NATIVE_RESTORE_BYTES_DIFFER')
        return result


class NativeKV41Site:
    def __init__(self,model,runner,*,fault,native):
        self.model=model;self.runner=runner;self.fault=fault;self.native=native
        self.methods=[];self.hooks=[];self.sites=[];self.events=[];self.writers={};self.excluded=[];self.uncovered=[]

    @property
    def pending(self):return self.native.count()

    def install(self):
        for name,module in self.model.named_modules():
            impl=getattr(module,'impl',None)
            if impl is None or type(impl).__name__!='AscendAttentionBackendImpl':continue
            require(hashlib.sha256(Path(inspect.getsourcefile(type(impl))).read_bytes()).hexdigest()==SOURCE_SHA,'KV41_ATTENTION_SOURCE_UNREVIEWED')
            require(impl.sliding_window is None and impl.sinks is None and impl.kv_sharing_target_layer_name is None,'KV41_SHARED_OR_SLIDING_CACHE_UNREVIEWED')
            self.sites.append(name)
            for method in ('reshape_and_cache','do_kv_cache_update','forward_impl'):
                require(method not in vars(impl),'KV41_ALREADY_OWNED')
                original=getattr(impl,method)
                def wrapper(*args,_name=name,_impl=impl,_method=method,_original=original,**kwargs):
                    if _method!='forward_impl':
                        mode=KVMode(self,_name,_impl,'writer')
                        with mode:result=_original(*args,**kwargs)
                        require(mode.calls==1,'KV41_ACTUAL_WRITER_NOT_OBSERVED')
                        return result
                    require(not kwargs and len(args)==6,'KV41_FORWARD_BOUNDARY_CHANGED')
                    ids=list(self.runner.input_batch.req_ids);require(len(ids)==1,'KV41_SINGLE_REQUEST')
                    meta=args[4];require(meta.causal and len(meta.seq_lens_list)==1,'KV41_CAUSAL_METADATA')
                    e=dict(site=_name,request_ids=ids,observed=True,eligible=False,applied=False,changed=0,consumed=False,restored=False,failed=False)
                    self.events.append(e);mode=KVMode(self,_name,_impl,'consumer',meta,e)
                    with mode:result=_original(*args)
                    require(mode.calls==1,'KV41_ACTUAL_FIA_CONSUMER_NOT_OBSERVED')
                    self.writers.pop(_name,None)
                    return result
                setattr(impl,method,wrapper);self.methods.append((impl,method,wrapper))
        require(self.sites,'KV41_NO_REAL_ATTENTION_SITES')
        return self.sites

    def close(self):
        require(not self.native.count(),'KV41_LIVE_NATIVE_LEASE_REMAINS')
        require(all(vars(o).get(n) is f for o,n,f in self.methods),'KV41_WRAPPER_OWNER_CHANGED')
        for o,n,_ in reversed(self.methods):delattr(o,n)
        self.methods.clear();self.writers.clear()
