"""Deployment-owned Conv3d output example, not a yellow mechanism or grant.

The adapter reads inputs/weights but rewrites only the transient actual output.
Image/request admissibility remains in ProductRequest and the pinned validator.
"""
import torch
from torch.utils._python_dispatch import _disable_current_modes
from fault_injection.sites.norm18_variance import FullNormConsumerMode
from fault_injection.sites.mla28 import fingerprint, require
from fault_injection.conv3d_natural import conv_arguments


class ActualConvConsumerMode(FullNormConsumerMode):
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        values=conv_arguments(func,args,kwargs or {})
        module=self.site.projection
        if values is None or module is None:
            return super().__torch_dispatch__(func,types,args,kwargs)
        require(self.site.producer is None,'EXTERNAL_MULTIPLE_CONV_PRODUCERS')
        x,w,b=values['input'],values['weight'],values['bias']
        require(w is module.weight and b is getattr(module,'bias',None)
                and x.ndim==5 and not values.get('transposed',False),
                'EXTERNAL_CONV_LIVE_PARAMETER_OWNER_UNPROVEN')
        with _disable_current_modes():
            readonly=[t for t in (x,w,b) if isinstance(t,torch.Tensor)]
            before=[fingerprint(t) for t in readonly]
        result=super().__torch_dispatch__(func,types,args,kwargs)
        self.site.producer=dict(output=result,readonly=readonly,before=before,
            operator=func._schema.name,input_shape=list(x.shape),weight_shape=list(w.shape))
        return result


class ExternalVisionGainSite:
    def __init__(self, model, runner, *, fault, region_leases):
        self.model=model; self.runner=runner; self.fault=fault; self.region_leases=region_leases
        self.events=[]; self.sites=[]; self.forward_calls=[]; self.uncovered=[]; self.parents=[]
        self.hooks=[]; self.active=False; self.mode=ActualConvConsumerMode(self)
        self.projection=None; self.producer=None

    @property
    def pending(self):return len(self.mode.pending)

    def install(self):
        visual=getattr(self.model,'visual',None)
        proj=getattr(getattr(visual,'patch_embed',None),'proj',None)
        # vLLM Conv3dLayer is a CustomOp, not an nn.Conv3d subclass. A 5-D
        # weight only discovers the candidate; actual dispatch proves Conv3d.
        require(isinstance(proj,torch.nn.Module) and isinstance(getattr(proj,'weight',None),torch.Tensor)
                and proj.weight.ndim==5, 'EXTERNAL_ACTUAL_PATCH_CONV3D_REQUIRED')
        self.sites=['visual.patch_embed.proj']
        def enter(*unused):
            require(not self.active and not self.pending, 'EXTERNAL_VISION_FORWARD_OVERLAP')
            self.forward_calls.append({}); self.mode.__enter__(); self.active=True
        def output(module,args,value):
            require(self.active and isinstance(value,torch.Tensor) and value.is_contiguous()
                    and value.dtype==torch.bfloat16, 'EXTERNAL_CONV_OUTPUT_LAYOUT_UNPROVEN')
            producer=self.producer
            require(producer is not None and producer['output'] is value,
                    'EXTERNAL_ACTUAL_CONV_RETURN_UNPROVEN')
            with _disable_current_modes():
                readonly=producer['readonly'];before=producer['before']
                require(all(value.untyped_storage()._cdata != t.untyped_storage()._cdata for t in readonly),
                        'EXTERNAL_CONV_OUTPUT_ALIASES_READONLY_OBJECT')
                require([fingerprint(t) for t in readonly]==before,'EXTERNAL_CONV_MUTATED_READONLY_INPUT')
                region=value.view(-1)
                name=self.sites[0]; calls=self.forward_calls[-1];calls[name]=calls.get(name,0)+1
                event=dict(site=name,request_ids=list(self.runner.input_batch.req_ids),eligible=True,
                    applied=False,changed=0,changed_elements=0,consumed=False,restored=False,failed=False,
                    effect='EXTERNAL_TRANSIENT_CONV_OUTPUT_GAIN',shape=list(value.shape),
                    native_operator=producer['operator'],input_shape=producer['input_shape'],
                    weight_shape=producer['weight_shape'],native_return_identity=True)
                self.events.append(event)
                def replacement(t):
                    result=t*.75
                    event['changed_elements']=int((t.view(torch.uint8).reshape(-1,t.element_size()) !=
                        result.view(torch.uint8).reshape(-1,t.element_size())).any(1).sum().item())
                    return result
                self.mode.watch(region,event,inject=self.fault,replacement=replacement)
                require([fingerprint(t) for t in readonly]==before,'EXTERNAL_READONLY_INPUT_OR_WEIGHT_CHANGED')
                event['readonly_inputs_weights_unchanged']=True
            self.projection=None;self.producer=None
        def before_projection(module,args):
            require(self.projection is None and self.producer is None,'EXTERNAL_CONV_CALL_OVERLAP')
            self.projection=module
        self.hooks=[visual.register_forward_pre_hook(enter),proj.register_forward_pre_hook(before_projection),
                    proj.register_forward_hook(output)]
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
