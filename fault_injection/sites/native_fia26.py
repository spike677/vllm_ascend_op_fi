"""FIA bytes into the live residual before its nonmutating add-rms consumer.

The original FIA and norm still execute. This does not redirect their pointers
or assert an upstream kernel overflow. Unknown in-place consumers refuse.
"""
import hashlib
import inspect
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
from .gqa28 import SOURCE_SHA
from .norm18 import NORM_SHA
from .mla28 import require, fingerprint
from .native_kv41 import arguments
from .projection39 import tensors, contains_element

DECODER_SHA='b642c10eb68978ca0df25f92ea866add08b31e654d99e78eaac0195d8bc6c74b'


class CaptureFIA(TorchDispatchMode):
    def __init__(self):super().__init__();self.outputs=[]
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        result=func(*args,**(kwargs or {}))
        if func._schema.name=='npu::npu_fused_infer_attention_score':
            require(isinstance(result,(list,tuple)) and isinstance(result[0],torch.Tensor),'FIA26_NATIVE_RETURN_CHANGED')
            self.outputs.append(result[0])
        return result


class ResidualRead(TorchDispatchMode):
    def __init__(self,site,record,residual):
        super().__init__();self.site=site;self.record=record;self.residual=residual;self.calls=0
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {}
        if func._schema.name not in ('npu::npu_add_rms_norm','_C_ascend::npu_add_rms_norm_bias'):
            return func(*args,**kwargs)
        with _disable_current_modes():
            require(self.calls==0,'FIA26_MULTIPLE_RESIDUAL_CONSUMERS')
            e=self.record;source=e['source'];target=self.residual[-1,:64]
            actual=arguments(func,args,kwargs)
            matches=[]
            for a in func._schema.arguments:
                if any(contains_element(t,target[:1]) for t in tensors(actual[a.name])):
                    require(a.alias_info is None or not a.alias_info.is_write,'FIA26_INPLACE_CONSUMER_UNSUPPORTED')
                    matches.append(a.name)
            require(len(matches)==1,'FIA26_RESIDUAL_ACTUAL_INPUT_NOT_UNIQUE')
            require(fingerprint(source)==e['source_sha256'],'FIA26_SOURCE_CHANGED_BEFORE_DESTINATION')
            require(source.dtype==target.dtype==torch.bfloat16 and source.shape==target.shape
                and source.device==target.device,'FIA26_SOURCE_DESTINATION_GEOMETRY')
            e.update(consumer_operator=func._schema.name,consumer_argument=matches[0],region_elements=target.numel())
            native=self.site.native
            if self.site.fault:
                e['before_sha256']=fingerprint(target);dst=native.acquire(target);src=native.acquire(source)
                e.update(applied=True,changed=native.copy(dst,src),after_sha256=fingerprint(target))
                require(e['after_sha256']==e['source_sha256'],'FIA26_NATIVE_CONTENT_COPY_FAILED')
        result=func(*args,**kwargs)
        with _disable_current_modes():
            # The actual kernel must return fresh residual/output storage. A
            # normal in-place update is never erased by the original snapshot.
            require(all(t.untyped_storage()._cdata!=self.residual.untyped_storage()._cdata for t in tensors(result)),
                'FIA26_CONSUMER_OUTPUT_ALIASES_OLD_RESIDUAL')
            self.calls+=1;e['consumed']=True
            if self.site.fault:
                native.consumed(dst);native.restore(dst);native.restore(src)
                e.update(restored=True,restored_sha256=fingerprint(target))
                require(e['before_sha256']==e['restored_sha256'],'FIA26_RESIDUAL_RESTORE_DIFFER')
        return result


class NativeFIA26Site:
    def __init__(self,model,runner,*,fault,native):
        self.model=model;self.runner=runner;self.fault=fault;self.native=native
        self.sites=[];self.events=[];self.hooks=[];self.methods=[];self.live={};self.residuals={};self.uncovered=[];self.excluded=[]
    @property
    def pending(self):return self.native.count()+len(self.live)
    def install(self):
        for name,decoder in self.model.named_modules():
            if getattr(decoder,'layer_type',None)!='full_attention':continue
            require(hashlib.sha256(Path(inspect.getsourcefile(decoder.forward)).read_bytes()).hexdigest()==DECODER_SHA,
                'FIA26_DECODER_SOURCE_UNREVIEWED')
            require(not decoder.use_attn_reduce_scatter_for_moe and not decoder.layer_scale,'FIA26_DECODER_LAYOUT_UNREVIEWED')
            attention=decoder.self_attn;impl=attention.attn.impl;norm=decoder.post_attention_layernorm
            require(hashlib.sha256(Path(inspect.getsourcefile(type(impl))).read_bytes()).hexdigest()==SOURCE_SHA
                and hashlib.sha256(Path(inspect.getsourcefile(type(norm))).read_bytes()).hexdigest()==NORM_SHA,
                'FIA26_SOURCE_UNREVIEWED')
            self.sites.append(name)
            def residual(mod,args,result,_name=name):
                require(isinstance(result,tuple) and len(result)==2,'FIA26_REAL_RESIDUAL_REQUIRED')
                require(_name not in self.residuals,'FIA26_PREVIOUS_RESIDUAL_REMAINS');self.residuals[_name]=result[1]
            self.hooks.append(decoder.input_layernorm.register_forward_hook(residual))
            require('forward_impl' not in vars(impl),'FIA26_ALREADY_WRAPPED')
            original=impl.forward_impl
            def fia(*args,_name=name,_original=original,**kwargs):
                require(_name not in self.live and _name in self.residuals,'FIA26_REQUEST_CALL_ORDER')
                mode=CaptureFIA()
                with mode:result=_original(*args,**kwargs)
                require(len(mode.outputs)==1,'FIA26_REAL_FIA_NOT_OBSERVED')
                t=mode.outputs[0]
                require(t.is_contiguous() and t.shape[0]>0,'FIA26_OUTPUT_LAYOUT')
                source=t.reshape(t.shape[0],-1)[-1,:64]
                require(source.numel()==64,'FIA26_OUTPUT_REGION_TOO_SMALL')
                e=dict(site=_name,request_ids=list(self.runner.input_batch.req_ids),eligible=True,
                    applied=False,changed=0,consumed=False,restored=False,failed=False,
                    source_sha256=fingerprint(source),writer='npu_fused_infer_attention_score')
                require(len(e['request_ids'])==1,'FIA26_SINGLE_REQUEST')
                self.events.append(e);self.live[_name]=dict(e,source=source,event=e)
                return result
            impl.forward_impl=fia;self.methods.append((impl,'forward_impl',fia))
            require('forward' not in vars(norm),'FIA26_NORM_ALREADY_WRAPPED')
            original_norm=norm.forward
            def consume(*args,_name=name,_original=original_norm,**kwargs):
                require(not kwargs and len(args)==2 and _name in self.live,'FIA26_NORM_BOUNDARY_CHANGED')
                residual=self.residuals[_name]
                require(args[1] is residual and residual.ndim==2 and residual.is_contiguous(),'FIA26_RESIDUAL_IDENTITY')
                record=self.live[_name];mode=ResidualRead(self,record,residual)
                with mode:result=_original(*args)
                require(mode.calls==1,'FIA26_REAL_RESIDUAL_CONSUMER_MISSING')
                record['event'].update({k:v for k,v in record.items() if k not in ('source','event')})
                del self.live[_name];del self.residuals[_name]
                return result
            norm.forward=consume;self.methods.append((norm,'forward',consume))
        require(self.sites,'FIA26_NO_ACTUAL_FULL_ATTENTION')
        return self.sites
    def close(self):
        require(not self.pending and not self.residuals,'FIA26_LIVE_NATIVE_RESIDUAL_LEASE')
        require(all(vars(o).get(n) is f for o,n,f in self.methods),'FIA26_WRAPPER_OWNER_CHANGED')
        for o,n,_ in reversed(self.methods):delattr(o,n)
        for h in self.hooks:h.remove()
        self.methods.clear();self.hooks.clear()
