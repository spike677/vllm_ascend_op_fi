"""Native input-content overwrite before actual DynamicQuant and original QMM.

Uses an actual live decoder residual as source. No pointer/executor patching,
no temporary invented source data, and no projection output replacement.
"""
import hashlib,inspect
from pathlib import Path
import torch
from .mla28 import require,fingerprint
from .batch_quant import SOURCE_SHA
from .native_fia26 import DECODER_SHA
from .norm18 import NORM_SHA


def quantized_read(native,source,x,original,args,kwargs,event,*,fault):
    require(x.ndim==source.ndim==2 and x.shape[0]==source.shape[0] and min(x.shape[1],source.shape[1])>=64,
        'INPUT20_ACTUAL_ROW_GEOMETRY')
    require(x.dtype==source.dtype==torch.bfloat16 and x.is_contiguous() and source.is_contiguous(),
        'INPUT20_BASE_BF16_MATRIX_REQUIRED')
    require(x.untyped_storage()._cdata!=source.untyped_storage()._cdata,'INPUT20_DISTINCT_LIVE_SOURCE_REQUIRED')
    target=x[-1,:64];origin=source[-1,:64]
    event.update(source_sha256=fingerprint(origin),region_elements=64,consumer_operator='npu_dynamic_quant',
        writer='native_owned_object_copy')
    if fault:
        dst=native.acquire(target);src=native.acquire(origin);event['before_sha256']=fingerprint(target)
        event.update(applied=True,changed=native.copy(dst,src),after_sha256=fingerprint(target))
        require(event['after_sha256']==event['source_sha256'],'INPUT20_NATIVE_COPY_MISMATCH')
    result=original(x,*args,**kwargs)
    q,scale=result
    require(q.dtype==torch.int8 and q.shape==x.shape and scale.dtype==torch.float32 and scale.shape==(x.shape[0],),
        'INPUT20_DYNAMIC_QUANT_RESULT_CHANGED')
    require(all(t.untyped_storage()._cdata!=x.untyped_storage()._cdata for t in result),'INPUT20_CONSUMER_ALIASES_INPUT')
    if fault:
        native.consumed(dst);native.restore(dst);native.restore(src)
        event.update(restored=True,restored_sha256=fingerprint(target))
        require(event['before_sha256']==event['restored_sha256'],'INPUT20_RESTORE_DIFFER')
    return result


class NativeInput20Site:
    def __init__(self,model,runner,*,fault,native):
        self.model=model;self.runner=runner;self.fault=fault;self.native=native
        self.events=[];self.sites=[];self.methods=[];self.hooks=[];self.uncovered=[];self.excluded=[]
        self.context=None;self.parents=[];self.sources={};self.api=None
    @property
    def pending(self):return self.native.count()+int(self.context is not None)
    def install(self):
        import torch_npu
        self.api=torch_npu;self.original_quant=torch_npu.npu_dynamic_quant;self.original_qmm=torch_npu.npu_quant_matmul
        require(not getattr(self.original_quant,'_dcat_site',False),'INPUT20_QUANT_ALREADY_OWNED')
        decoders={n:m for n,m in self.model.named_modules() if hasattr(m,'input_layernorm') and hasattr(m,'post_attention_layernorm')}
        for name,decoder in decoders.items():
            require(hashlib.sha256(Path(inspect.getsourcefile(decoder.forward)).read_bytes()).hexdigest()==DECODER_SHA
                and not decoder.use_attn_reduce_scatter_for_moe,'INPUT20_DECODER_UNREVIEWED')
            def enter(mod,args,kwargs,_name=name):
                require(not self.parents and not self.sources,'INPUT20_DECODER_OVERLAP');self.parents.append(_name)
            def leave(mod,args,result,_name=name):
                require(self.parents==[_name] and self.context is None,'INPUT20_DECODER_INCOMPLETE')
                self.parents.pop();self.sources.pop(_name,None)
            self.hooks.extend([decoder.register_forward_pre_hook(enter,with_kwargs=True),decoder.register_forward_hook(leave)])
            for norm in (decoder.input_layernorm,decoder.post_attention_layernorm):
                require(hashlib.sha256(Path(inspect.getsourcefile(type(norm))).read_bytes()).hexdigest()==NORM_SHA,'INPUT20_NORM_UNREVIEWED')
                def observed(mod,args,result,_name=name):
                    require(self.parents==[_name],'INPUT20_SOURCE_OUTSIDE_REQUEST')
                    self.sources[_name]=result[1] if isinstance(result,tuple) else args[0]
                self.hooks.append(norm.register_forward_hook(observed))
        seen=set()
        for name,module in self.model.named_modules():
            scheme=getattr(getattr(module,'quant_method',None),'quant_method',None)
            if type(scheme).__name__!='AscendW8A8DynamicLinearMethod':continue
            parents=[n for n in decoders if name.startswith(n+'.')]
            if not parents:
                self.excluded.append(dict(site=name,reason='NO_REVIEWED_DECODER_RESIDUAL'));continue
            require(len(parents)==1 and id(scheme) not in seen,'INPUT20_SHARED_SCHEME');seen.add(id(scheme))
            require(hashlib.sha256(Path(inspect.getsourcefile(type(scheme))).read_bytes()).hexdigest()==SOURCE_SHA
                and 'apply' not in vars(scheme),'INPUT20_QUANT_SOURCE_OR_OWNER_CHANGED')
            original=scheme.apply;parent=parents[0]
            def call(layer,x,*args,_name=name,_parent=parent,_original=original,**kwargs):
                require(self.context is None and self.parents==[_parent] and _parent in self.sources,'INPUT20_CURRENT_SOURCE_MISSING')
                e=dict(site=_name,request_ids=list(self.runner.input_batch.req_ids),eligible=True,
                    applied=False,changed=0,consumed=False,restored=False,failed=False,qmm_calls=0)
                require(len(e['request_ids'])==1,'INPUT20_SINGLE_REQUEST')
                self.events.append(e);self.context=dict(x=x,source=self.sources[_parent],event=e,buffers=None,stream=torch.npu.current_stream())
                result=_original(layer,x,*args,**kwargs)
                require(e['qmm_calls']==1 and self.context['buffers'] is not None,'INPUT20_REAL_QMM_MISSING')
                e['consumed']=True;self.context=None
                return result
            scheme.apply=call;self.methods.append((scheme,call));self.sites.append(name)
        require(self.sites,'INPUT20_NO_REAL_QUANTIZED_PROJECTIONS')
        def quant(x,*args,**kwargs):
            ctx=self.context
            if ctx is None:
                self.uncovered.append(dict(operator='npu_dynamic_quant',reason='OUTSIDE_INPUT20_SCHEME'))
                return self.original_quant(x,*args,**kwargs)
            require(ctx['x'] is x and ctx['buffers'] is None,'INPUT20_QUANT_CALL_MULTIPLICITY')
            result=quantized_read(self.native,ctx['source'],x,self.original_quant,args,kwargs,ctx['event'],fault=self.fault)
            ctx['buffers']=result;return result
        def qmm(x,*args,**kwargs):
            ctx=self.context
            if ctx is None:return self.original_qmm(x,*args,**kwargs)
            require(ctx['buffers'] is not None and x is ctx['buffers'][0] and kwargs.get('pertoken_scale') is ctx['buffers'][1]
                and torch.npu.current_stream()==ctx['stream'],'INPUT20_REAL_QMM_INPUT_JOIN')
            out=self.original_qmm(x,*args,**kwargs)
            event=torch.npu.Event();event.record(ctx['stream']);event.synchronize();ctx['event']['qmm_calls']+=1
            return out
        quant._dcat_site=True;self.quant=quant;self.qmm=qmm
        torch_npu.npu_dynamic_quant=quant;torch_npu.npu_quant_matmul=qmm
        return self.sites
    def close(self):
        require(not self.pending and not self.parents and not self.sources,'INPUT20_PENDING_NATIVE_REGIONS')
        require(self.api.npu_dynamic_quant is self.quant and self.api.npu_quant_matmul is self.qmm,'INPUT20_API_OWNER_CHANGED')
        self.api.npu_dynamic_quant=self.original_quant;self.api.npu_quant_matmul=self.original_qmm
        for scheme,fn in self.methods:
            require(vars(scheme).get('apply') is fn,'INPUT20_SCHEME_OWNER_CHANGED');delattr(scheme,'apply')
        for h in self.hooks:h.remove()
        self.methods.clear();self.hooks.clear()
