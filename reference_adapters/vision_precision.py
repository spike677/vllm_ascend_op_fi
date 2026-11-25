"""Library16/32 Conv3d precision proxies using the original image path."""
import torch
from torch.utils._python_dispatch import _disable_current_modes
from experiments.worker_core_v11_20261118.external_input_examples import BaseSite, InputReadMode, changed
from fault_injection.conv3d_natural import conv_arguments
from fault_injection.sites.mla28 import require, fingerprint


class ConvMode(InputReadMode):
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {};values=conv_arguments(func,args,kwargs);site=self.site
        if values is None or site.projection is None:
            return super().__torch_dispatch__(func,types,args,kwargs)
        require(site.produced is None,'REFERENCE_MULTIPLE_CONV_CALLS')
        x,w,b=values['input'],values['weight'],values['bias']
        require(w is site.projection.weight and b is getattr(site.projection,'bias',None)
            and x.dtype==w.dtype==torch.bfloat16 and not values.get('transposed',False),
            'REFERENCE_CONV_OWNER_OR_SEMANTICS')
        with _disable_current_modes():
            readonly=[t for t in (x,w,b) if isinstance(t,torch.Tensor)]
            before=[fingerprint(t) for t in readonly]
            event=site.event('visual.patch_embed.proj',x)
            event.update(effect='PHENOMENON_PROXY_NOT_ORIGINAL_MECHANISM',
                named_variant=site.variant,native_operator=func._schema.name,
                input_shape=list(x.shape),weight_shape=list(w.shape),native_return_identity=True)
            token=None
            if site.input_only and site.fault:
                value=site.transform_input(x)
                require(x.is_contiguous(),'REFERENCE_CONV_INPUT_NOT_CONTIGUOUS')
                token=site.region_leases.acquire(x)
                event.update(before_sha256=fingerprint(x),changed_elements=changed(x,value))
                site.region_leases.write(token,value)
                event.update(applied=True,changed=int(event['changed_elements']>0),after_sha256=fingerprint(x))
        result=func(*args,**kwargs)
        with _disable_current_modes():
            if token is not None:
                site.region_leases.consumed(token)
                event.update(consumed=True,consumer_operator=func._schema.name,consumer_sha256=fingerprint(x))
                site.region_leases.restore(token)
                event.update(restored=True,restored_sha256=fingerprint(x))
            require([fingerprint(t) for t in readonly]==before,'REFERENCE_CONV_READONLY_CHANGED')
            event.update(readonly_inputs_weights_unchanged=True,shape=list(result.shape))
            require(result.is_contiguous(),'REFERENCE_CONV_OUTPUT_NOT_CONTIGUOUS')
            if site.input_only:
                trace={}
                site.encoder_traces.append((event,trace))
                self.watch(result.view(-1),trace,inject=False)
            else:
                value=torch.nn.functional.conv3d(x.half(),w.half(),None if b is None else b.half(),
                    values['stride'],values['padding'],values['dilation'],values['groups']).to(result.dtype)
                require(value.shape==result.shape,'REFERENCE_CONV_RECOMPUTE_SHAPE')
                def replacement(t):
                    event['changed_elements']=changed(t,value.reshape(-1))
                    return value.reshape(-1)
                self.watch(result.view(-1),event,inject=site.fault,replacement=replacement)
                require([fingerprint(t) for t in readonly]==before,'REFERENCE_CONV_RECOMPUTE_MUTATED_INPUT')
            site.produced=result
        return result


class VisionPrecisionSite(BaseSite):
    input_only=False
    variant='row16_conv3d_compute_fp16'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.mode=ConvMode(self);self.projection=None;self.produced=None;self.encoder_traces=[]
    @property
    def pending(self):return len(self.mode.pending)+self.region_leases.count()
    def install(self):
        visual=getattr(self.model,'visual',None)
        projection=getattr(getattr(visual,'patch_embed',None),'proj',None)
        require(isinstance(projection,torch.nn.Module) and getattr(projection,'weight',None) is not None
            and projection.weight.ndim==5,'REFERENCE_CONV3D_PATCH_REQUIRED')
        def before(mod,args):
            require(self.active and self.projection is None,'REFERENCE_CONV_CALL_ORDER')
            self.projection=mod;self.produced=None
        def after(mod,args,result):
            require(result is self.produced,'REFERENCE_ACTUAL_CONV_RETURN_REQUIRED')
            self.projection=None;self.produced=None
        self.hooks=[visual.register_forward_pre_hook(self.enter),
            projection.register_forward_pre_hook(before),projection.register_forward_hook(after)]
        self.sites=['visual.patch_embed.proj']
        return self.sites
    def finish_forward(self):
        for event,trace in self.encoder_traces:
            require(trace.get('consumed') is True,'REFERENCE_ENCODER_CONSUMER_MISSING')
            event['encoder_consumer_operator']=trace['consumer_operator']
        self.encoder_traces.clear()
        super().finish_forward()


class ConvInputFp16RoundTrip(VisionPrecisionSite):
    input_only=True
    variant='row32_conv3d_input_fp16'
    def transform_input(self,x):return x.half().to(x.dtype)


class ConvInputE4M3(ConvInputFp16RoundTrip):
    variant='row32_conv3d_input_e4m3'
    format_name='e4m3'
    def transform_input(self,x):
        from reference_adapters.precision_rules import fp8_like
        return fp8_like(x,self.format_name)


class ConvInputE5M2(ConvInputE4M3):
    variant='row32_conv3d_input_e5m2'
    format_name='e5m2'


FACTORIES={'row16_conv3d_compute_fp16':VisionPrecisionSite,
           'row32_conv3d_input_fp16':ConvInputFp16RoundTrip,
           'row32_conv3d_input_e4m3':ConvInputE4M3,
           'row32_conv3d_input_e5m2':ConvInputE5M2}
