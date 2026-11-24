"""Actual DynamicQuant argument substitution; no Q/scale result rewriting."""
import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
from experiments.worker_core_v11_20261118.external_input_examples import BaseSite, changed
from fault_injection.sites.mla28 import fingerprint, require


class QuantArgumentMode(TorchDispatchMode):
    def __init__(self,site):super().__init__();self.site=site

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {};site=self.site
        if func._schema.name!='npu::npu_dynamic_quant':return func(*args,**kwargs)
        # The reviewed overload has a Tensor as its first positional argument.
        require(args and func._schema.arguments[0].type.kind()=='TensorType',
                'EXTERNAL_QUANT_ARGUMENT_SCHEMA_CHANGED')
        x=args[0]
        require(x.dtype==torch.bfloat16 and x.ndim==2 and x.is_contiguous()
                and x.shape[0]>0 and x.shape[1]>=2,'EXTERNAL_QUANT_ARGUMENT_LAYOUT')
        with _disable_current_modes():
            event=site.event('npu.dynamic_quant:operand0',x)
            original=fingerprint(x);event['original_input_sha256']=original
            operand=x.detach().clone() if site.fault else x
            token=None
            if site.fault:
                region=operand[-1,:2];token=site.region_leases.acquire(region)
                value=region*.75
                event.update(before_sha256=fingerprint(region),changed_elements=changed(region,value))
                site.region_leases.write(token,value)
                event.update(applied=True,changed=int(event['changed_elements']>0),after_sha256=fingerprint(region))
            event['argument_substitution']=site.fault
            result=func(operand,*args[1:],**kwargs)
            site.region_leases.complete(site.region_leases.stream())
            require(isinstance(result,(tuple,list)) and len(result)>=2
                    and result[0].dtype==torch.int8,'EXTERNAL_ACTUAL_QUANT_RESULT_UNPROVEN')
            event.update(consumed=True,consumer_operator=func._schema.name,
                         same_stream_completed=True,result_forwarded_unmodified=True)
            if token is not None:
                site.region_leases.consumed(token);site.region_leases.restore(token)
                event.update(restored=True,restored_sha256=fingerprint(region))
            require(fingerprint(x)==original,'EXTERNAL_QUANT_ORIGINAL_INPUT_MUTATED')
            event['original_input_unchanged']=True
            return result


class ExternalQuantArgumentSite(BaseSite):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.mode=QuantArgumentMode(self)
    def install(self):
        self.sites=['npu.dynamic_quant:operand0']
        self.hooks.append(self.model.register_forward_pre_hook(self.enter));return self.sites
