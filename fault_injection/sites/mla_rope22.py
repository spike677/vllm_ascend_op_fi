"""MLA wrong rotary lookup rows at actual Q and fused KV consumers.

The current metadata positions select the normal rows; position+1 selects the
fault rows. Metadata buffers are leased only around real MLA preprocessing.
Neither shared rotary tables nor positions nor the kernel are modified.
"""
import hashlib,inspect
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode,_disable_current_modes
from .mla28 import SOURCE_SHA,require,fingerprint

ROTARY_SHA='7f041448559229932b804872dbb9d77a8a83c5c206e5aa52e30b58daf5cf4d38'


class MLAConsumerMode(TorchDispatchMode):
    def __init__(self,cos,sin,prefill):
        super().__init__();self.cos=cos;self.sin=sin;self.prefill=prefill;self.calls=[];self.q=None;self.k=None
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {};name=func._schema.name
        if name not in ('npu::npu_interleave_rope','npu::npu_kv_rmsnorm_rope_cache'):return func(*args,**kwargs)
        # Source-reviewed signatures use cos/sin at 1/2 (Q) and 2/3 (KV).
        q=name=='npu::npu_interleave_rope';offset=1 if q else 2
        values=[args[i] if i<len(args) else kwargs.get(a.name,a.default_value) for i,a in enumerate(func._schema.arguments)]
        require(values[offset] is self.cos and values[offset+1] is self.sin,'MLA_ROPE_ACTUAL_CONSUMER_INPUT_CHANGED')
        require(self.calls==([] if q else ['q']),'MLA_ROPE_CONSUMER_ORDER_CHANGED')
        result=func(*args,**kwargs);self.calls.append('q' if q else 'k')
        if q:self.q=result
        else:self.k=result[2] if self.prefill else result[0]
        return result


class MLARope22Site:
    def __init__(self,model,runner,*,fault,region_leases):
        self.model=model;self.runner=runner;self.fault=fault;self.region_leases=region_leases
        self.sites=[];self.events=[];self.methods=[];self.hooks=[];self.forward_calls=[];self.uncovered=[];self.excluded=[]
    @property
    def pending(self):return self.region_leases.count()
    def install(self):
        from vllm_ascend.ops import rotary_embedding as rotary
        require(hashlib.sha256(Path(rotary.__file__).read_bytes()).hexdigest()==ROTARY_SHA,'UNREVIEWED_MLA_ROTARY_SOURCE')
        self.rotary=rotary
        self.hooks.append(self.model.register_forward_pre_hook(lambda *a:self.forward_calls.append({})))
        seen=set()
        for name,module in self.model.named_modules():
            impl=getattr(module,'impl',None)
            if impl is None or type(impl).__name__!='AscendMLAImpl' or id(impl) in seen:continue
            seen.add(id(impl));self.sites.append(name)
            require(hashlib.sha256(Path(inspect.getsourcefile(type(impl))).read_bytes()).hexdigest()==SOURCE_SHA,'UNREVIEWED_MLA_SOURCE')
            require(impl.use_mla_rope and not impl.fa_quant_layer and impl.speculative_config is None,'MLA_ROPE_VARIANT_UNSUPPORTED')
            for method in ('mla_preprocess_prefill','mla_preprocess_decode'):
                require(method not in vars(impl),'MLA_ROPE_ALREADY_WRAPPED')
                original=getattr(impl,method)
                def wrapped(*args,_name=name,_method=method,_original=original,**kwargs):
                    require(not kwargs and len(args)==4,'MLA_PREPROCESS_SIGNATURE_CHANGED')
                    return self.call(_name,_method,_original,args)
                setattr(impl,method,wrapped);self.methods.append((impl,method,wrapped))
        require(self.sites,'NO_REAL_MLA_ROTARY_SITES');return self.sites
    def call(self,name,method,original,args):
        require(self.pending==0 and self.forward_calls,'MLA_ROPE_OVERLAP')
        ids=list(self.runner.input_batch.req_ids);require(len(ids)==1,'MLA_ROPE_SINGLE_ACTUAL_REQUEST')
        prefill=method=='mla_preprocess_prefill';meta=args[3];part=meta.prefill if prefill else meta.decode
        require((meta.num_prefills==1 and meta.num_decodes==0) if prefill else (meta.num_prefills==0 and meta.num_decodes==1),
            'MLA_ROPE_MIXED_REQUEST_MAPPING_UNSUPPORTED')
        positions=part.input_positions;cos,sin=part.cos,part.sin
        require(positions.ndim==1 and positions.numel()==meta.num_actual_tokens and cos.is_contiguous() and sin.is_contiguous(),
            'MLA_ROPE_REAL_METADATA_LAYOUT')
        calls=self.forward_calls[-1];calls[name]=calls.get(name,0)+1
        e=dict(site=name,request_ids=ids,eligible=True,applied=False,changed=0,consumed=False,restored=False,failed=False)
        self.events.append(e)
        with _disable_current_modes():
            tables=(self.rotary._cos_cache,self.rotary._sin_cache)
            require(all(isinstance(t,torch.Tensor) and t.shape[0]>1 for t in tables),'MLA_ROTARY_TABLES_UNAVAILABLE')
            expected=[t[positions].unsqueeze(1).unsqueeze(2) for t in tables]
            require(all(torch.equal(x,y) and x.dtype==y.dtype for x,y in zip((cos,sin),expected)),'MLA_ROPE_LOOKUP_NOT_PROVEN')
            if self.fault:
                shifted=(positions+1).remainder(tables[0].shape[0]);wrong=[t[shifted].unsqueeze(1).unsqueeze(2) for t in tables]
                tokens=[self.region_leases.acquire(t) for t in (cos,sin)]
                e.update(before_sha256=[fingerprint(t) for t in (cos,sin)],
                    changed=sum(int((x!=y).sum().item()) for x,y in zip((cos,sin),wrong)),
                    rotary_rows_changed=int(((cos!=wrong[0])|(sin!=wrong[1])).reshape(positions.numel(),-1).any(-1).sum().item()),
                    position_rows_changed=int((positions!=shifted).sum().item()))
                for token,value in zip(tokens,wrong):self.region_leases.write(token,value)
                e['applied']=True
        mode=MLAConsumerMode(cos,sin,prefill)
        with mode:result=original(*args)
        require(mode.calls==['q','k'],'MLA_ROPE_REAL_CONSUMERS_NOT_OBSERVED')
        with _disable_current_modes():
            if self.fault:
                for token in tokens:self.region_leases.consumed(token);self.region_leases.restore(token)
                e.update(consumed=True,restored=True,restored_sha256=[fingerprint(t) for t in (cos,sin)])
                require(e['restored_sha256']==e['before_sha256'],'MLA_ROTARY_BYTES_NOT_RESTORED')
            e.update(q_output_sha256=fingerprint(mode.q),k_output_sha256=fingerprint(mode.k),consumer_calls=mode.calls)
        return result
    def close(self):
        require(not self.pending,'MLA_ROPE_LEASES_REMAIN')
        for impl,name,wrapper in reversed(self.methods):
            require(vars(impl).get(name) is wrapper,'MLA_ROPE_WRAPPER_OWNER_CHANGED');delattr(impl,name)
        self.methods.clear()
        for h in self.hooks:h.remove()
        self.hooks.clear()
