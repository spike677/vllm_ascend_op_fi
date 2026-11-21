"""Pinned Gemma norm and fused Q/K normalization, actual dispatch consumers.

Half one actual output element per norm call on every rank. Fused Q/K outputs
are separate normalization sites. No weights or third-party kernels modified.
"""
import hashlib,inspect
from pathlib import Path
import torch
from torch.utils._python_dispatch import _disable_current_modes
from .projection39 import ProjectionConsumerMode
from .mla28 import require

NORM_SHA='b173bf93aa86da125c02b8e2e3b3711ae8128742b7b0a4ffb2a7232433ed6755'
ATTENTION_SHA='cf7ec7aae45bc83c5a08020210c22b698e8f8cd2629179a85ac7d3e1d1419127'


class NormConsumerMode(ProjectionConsumerMode):
    def __init__(self,site):super().__init__(site.region_leases);self.site=site
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        result=super().__torch_dispatch__(func,types,args,kwargs)
        if func._schema.name=='vllm::triton_split_qkv_rmsnorm_mrope':
            require(len(self.site.parents)==1,'FUSED_NORM_PARENT_UNPROVEN')
            require(isinstance(result,(tuple,list)) and len(result)==4,'FUSED_NORM_OUTPUT_CHANGED')
            for suffix,t in zip(('q_norm','k_norm'),result[:2]):
                self.site.output(self.site.parents[0]+'.fused.'+suffix,t)
        return result


class Norm18Site:
    def __init__(self,model,runner,*,fault,region_leases):
        self.model=model;self.runner=runner;self.fault=fault;self.region_leases=region_leases
        self.events=[];self.sites=[];self.hooks=[];self.uncovered=[];self.excluded=[]
        self.forward_calls=[];self.active=False;self.parents=[];self.mode=NormConsumerMode(self)
    @property
    def pending(self):return len(self.mode.pending)
    def output(self,name,t):
        require(self.active and self.forward_calls,'NORM_OUTSIDE_MODEL')
        require(t.dtype==torch.bfloat16 and t.numel()>0 and t.ndim>=1,'NORM_OUTPUT_LAYOUT')
        calls=self.forward_calls[-1];calls[name]=calls.get(name,0)+1
        with _disable_current_modes():
            # One element; slicing preserves actual storage even for strided Q/K.
            region=torch.as_strided(t,(1,),(1,),t.storage_offset()+(t.shape[0]-1)*t.stride(0))
            e=dict(site=name,request_ids=list(self.runner.input_batch.req_ids),eligible=True,
                applied=False,changed=0,consumed=False,restored=False,failed=False,effect='scale_half')
            self.events.append(e)
            self.mode.watch(region,e,inject=self.fault,replacement=lambda x:x*0.5)
    def install(self):
        def enter(*unused):
            require(not self.pending and not self.parents,'NORM_PREVIOUS_FORWARD_NOT_CONSUMED')
            self.forward_calls.append({})
            if not self.active:self.mode.__enter__();self.active=True
        self.hooks.append(self.model.register_forward_pre_hook(enter))
        visual=getattr(self.model,'visual',None);visual_ids={id(m) for m in visual.modules()} if visual is not None else set()
        for name,module in self.model.named_modules():
            if id(module) in visual_ids:continue
            if type(module).__name__=='AscendGemmaRMSNorm':
                require(hashlib.sha256(Path(inspect.getsourcefile(type(module))).read_bytes()).hexdigest()==NORM_SHA,'UNREVIEWED_NORM_SOURCE')
                self.sites.append(name)
                def output(mod,args,result,_name=name):self.output(_name,result[0] if isinstance(result,tuple) else result)
                self.hooks.append(module.register_forward_hook(output))
            if all(hasattr(module,a) for a in ('q_norm','k_norm','qkv_proj','rotary_emb','attn')):
                # The plugin installs AscendQwen3NextAttention.forward onto the
                # original class; class-name matching misses this actual call.
                require(hashlib.sha256(Path(inspect.getsourcefile(module.forward)).read_bytes()).hexdigest()==ATTENTION_SHA,'UNREVIEWED_FUSED_NORM_SOURCE')
                self.sites.extend(name+'.fused.'+s for s in ('q_norm','k_norm'))
                def before(mod,args,_name=name):self.parents.append(_name)
                def after(mod,args,out,_name=name):
                    require(self.parents==[_name],'FUSED_NORM_PARENT_ORDER');self.parents.pop()
                self.hooks.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
        require(self.sites,'NO_ACTUAL_NORM_CANDIDATES')
        return self.sites
    def close(self):
        require(not self.pending and not self.parents and self.region_leases.count()==0,'NORM_REGIONS_REMAIN')
        if self.active:self.mode.__exit__(None,None,None);self.active=False
        for h in self.hooks:h.remove()
        self.hooks.clear()
