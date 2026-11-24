"""Deployment-owned Conv3d output example, not a yellow mechanism or grant.

The adapter reads inputs/weights but rewrites only the transient actual output.
Image/request admissibility remains in ProductRequest and the pinned validator.
"""
import torch
from torch.utils._python_dispatch import _disable_current_modes
from fault_injection.sites.norm18_variance import FullNormConsumerMode
from fault_injection.sites.mla28 import fingerprint, require


class ExternalVisionGainSite:
    def __init__(self, model, runner, *, fault, region_leases):
        self.model=model; self.runner=runner; self.fault=fault; self.region_leases=region_leases
        self.events=[]; self.sites=[]; self.forward_calls=[]; self.uncovered=[]; self.parents=[]
        self.hooks=[]; self.active=False; self.mode=FullNormConsumerMode(self)

    @property
    def pending(self):return len(self.mode.pending)

    def install(self):
        visual=getattr(self.model,'visual',None)
        proj=getattr(getattr(visual,'patch_embed',None),'proj',None)
        require(isinstance(proj,torch.nn.Conv3d), 'EXTERNAL_ACTUAL_PATCH_CONV3D_REQUIRED')
        self.sites=['visual.patch_embed.proj']
        def enter(*unused):
            require(not self.active and not self.pending, 'EXTERNAL_VISION_FORWARD_OVERLAP')
            self.forward_calls.append({}); self.mode.__enter__(); self.active=True
        def output(module,args,value):
            require(self.active and isinstance(value,torch.Tensor) and value.is_contiguous()
                    and value.dtype==torch.bfloat16, 'EXTERNAL_CONV_OUTPUT_LAYOUT_UNPROVEN')
            with _disable_current_modes():
                readonly=[t for t in args if isinstance(t,torch.Tensor)]+list(module.parameters())
                require(all(value.untyped_storage()._cdata != t.untyped_storage()._cdata for t in readonly),
                        'EXTERNAL_CONV_OUTPUT_ALIASES_READONLY_OBJECT')
                before=[fingerprint(t) for t in readonly]
                region=value.view(-1)
                name=self.sites[0]; calls=self.forward_calls[-1];calls[name]=calls.get(name,0)+1
                event=dict(site=name,request_ids=list(self.runner.input_batch.req_ids),eligible=True,
                    applied=False,changed=0,changed_elements=0,consumed=False,restored=False,failed=False,
                    effect='EXTERNAL_TRANSIENT_CONV_OUTPUT_GAIN',shape=list(value.shape))
                self.events.append(event)
                def replacement(t):
                    result=t*.75
                    event['changed_elements']=int((t.view(torch.uint8).reshape(-1,t.element_size()) !=
                        result.view(torch.uint8).reshape(-1,t.element_size())).any(1).sum().item())
                    return result
                self.mode.watch(region,event,inject=self.fault,replacement=replacement)
                require([fingerprint(t) for t in readonly]==before,'EXTERNAL_READONLY_INPUT_OR_WEIGHT_CHANGED')
                event['readonly_inputs_weights_unchanged']=True
        self.hooks=[visual.register_forward_pre_hook(enter),proj.register_forward_hook(output)]
        return self.sites

    def finish_forward(self):
        require(self.active and not self.pending and self.region_leases.count()==0,
                'EXTERNAL_VISION_REAL_CONSUMER_PENDING')
        self.mode.__exit__(None,None,None);self.active=False

    def close(self):
        require(not self.pending and self.region_leases.count()==0,'EXTERNAL_VISION_LEASES_REMAIN')
        if self.active:self.finish_forward()
        for h in self.hooks:h.remove()
        self.hooks.clear()
