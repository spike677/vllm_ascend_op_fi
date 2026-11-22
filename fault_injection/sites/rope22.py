"""Wrong rotary row at the real fused mRoPE lookup input, all actual calls.

The original positions and shared rotary cache are never changed. The transient
cos/sin input is leased, populated from position+1 and consumed by the original
fused operator, then restored after same-stream device completion.
"""
import hashlib,inspect
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode,_disable_current_modes
from .mla28 import require,fingerprint
from .norm18 import ATTENTION_SHA


class RopeMode(TorchDispatchMode):
    def __init__(self,site):super().__init__();self.site=site
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {};site=self.site
        if func._schema.name!='vllm::triton_split_qkv_rmsnorm_mrope':return func(*args,**kwargs)
        require(len(site.parents)==1,'FUSED_ROPE_PARENT_UNPROVEN')
        name,module,positions=site.parents[0]
        values={a.name:(args[i] if i<len(args) else kwargs.get(a.name)) for i,a in enumerate(func._schema.arguments)}
        cos=values['cos_sin'];cache=module.rotary_emb.cos_sin_cache
        require(positions.ndim==2 and positions.shape[0]==3 and positions.dtype in (torch.int32,torch.int64),'MROPE_THREE_POSITION_ROWS_REQUIRED')
        require(cos.is_contiguous() and cache.shape[0]>1 and cos.dtype==torch.bfloat16,'MROPE_ACTUAL_LOOKUP_LAYOUT')
        calls=site.forward_calls[-1];calls[name]=calls.get(name,0)+1
        e=dict(site=name,request_ids=list(site.runner.input_batch.req_ids),eligible=True,
            applied=False,changed=0,consumed=False,restored=False,failed=False)
        site.events.append(e)
        with _disable_current_modes():
            expected=cache[positions].to(device=cos.device,dtype=cos.dtype)
            require(torch.equal(expected,cos),'MROPE_ACTUAL_LOOKUP_NOT_PROVEN')
            if site.fault:
                shifted=(positions+1).remainder(cache.shape[0])
                wrong=cache[shifted].to(device=cos.device,dtype=cos.dtype)
                token=site.region_leases.acquire(cos)
                e.update(nonfinite_before=int((~torch.isfinite(cos)).sum().item()),nonfinite_after=int((~torch.isfinite(wrong)).sum().item()),
                    before_sha256=fingerprint(cos),changed=int((cos!=wrong).sum().item()),
                    rotary_rows_changed=int((cos!=wrong).reshape(-1,cos.shape[-1]).any(-1).sum().item()),
                    position_rows_changed=int((positions!=shifted).sum().item()))
                site.region_leases.write(token,wrong);e['applied']=True
        # Exactly one original fused call, with its real transient lookup input.
        result=func(*args,**kwargs)
        with _disable_current_modes():
            if site.fault:
                site.region_leases.consumed(token);e['consumed']=True
                site.region_leases.restore(token)
                e.update(restored=True,restored_sha256=fingerprint(cos),consumer_operator=func._schema.name)
                require(e['before_sha256']==e['restored_sha256'],'MROPE_LOOKUP_RESTORE_FAILED')
            e.update(q_output_sha256=fingerprint(result[0]),k_output_sha256=fingerprint(result[1]))
        return result


class Rope22Site:
    def __init__(self,model,runner,*,fault,region_leases):
        self.model=model;self.runner=runner;self.fault=fault;self.region_leases=region_leases
        self.events=[];self.sites=[];self.hooks=[];self.uncovered=[];self.excluded=[]
        self.parents=[];self.forward_calls=[];self.active=False;self.mode=RopeMode(self)
    @property
    def pending(self):return self.region_leases.count()
    def install(self):
        def enter(*unused):
            require(not self.active and not self.pending,'ROPE_FORWARD_OVERLAP')
            self.forward_calls.append({});self.mode.__enter__();self.active=True
        def leave(*unused):
            require(not self.parents and not self.pending,'ROPE_FORWARD_INCOMPLETE')
            self.mode.__exit__(None,None,None);self.active=False
        self.hooks.extend([self.model.register_forward_pre_hook(enter),self.model.register_forward_hook(leave)])
        for name,module in self.model.named_modules():
            if not all(hasattr(module,a) for a in ('q_norm','k_norm','qkv_proj','rotary_emb','attn')):continue
            require(hashlib.sha256(Path(inspect.getsourcefile(module.forward)).read_bytes()).hexdigest()==ATTENTION_SHA,'UNREVIEWED_FUSED_ROPE_SOURCE')
            self.sites.append(name)
            def before(mod,args,kwargs,_name=name):
                positions=args[0] if args else kwargs.get('positions')
                require(isinstance(positions,torch.Tensor),'ACTUAL_ROPE_POSITIONS_REQUIRED')
                self.parents.append((_name,mod,positions))
            def after(mod,args,out,_name=name):
                require(len(self.parents)==1 and self.parents[0][0]==_name,'ROPE_PARENT_ORDER');self.parents.pop()
            self.hooks.extend([module.register_forward_pre_hook(before,with_kwargs=True),module.register_forward_hook(after)])
        require(self.sites,'NO_ACTUAL_FUSED_ROPE_SITES');return self.sites
    def close(self):
        require(not self.parents and not self.pending,'ROPE_REGION_RESOURCES_REMAIN')
        if self.active:self.mode.__exit__(None,None,None);self.active=False
        for h in self.hooks:h.remove()
        self.hooks.clear()
