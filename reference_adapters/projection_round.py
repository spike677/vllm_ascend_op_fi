"""Library34: truncate real dequantized down_proj output before fused add/norm."""
import torch
from torch.utils._python_dispatch import _disable_current_modes
from experiments.worker_core_v11_20261118.external_input_examples import BaseSite, changed
from reference_adapters.hidden_rows import HiddenReadMode
from reference_adapters.precision_rules import truncate_bf16
from fault_injection.sites.mla28 import require


class FusedConsumerMode(HiddenReadMode):
    def complete_consumers(self, matched, name):
        # Do not report a view, an all-reduce, or a generic add as the requested
        # fused consumer. The base mode invokes this after the real operation.
        if matched:
            require(name == '_C_ascend::npu_add_rms_norm_bias',
                    'REFERENCE_FUSED_ADD_NORM_CONSUMER_REQUIRED:'+name)
        return super().complete_consumers(matched,name)


class ProjectionMantissa5(BaseSite):
    fraction_bits=5
    variant='row34_projection_mantissa5'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.mode=FusedConsumerMode(self)
        self.projections={}
    @property
    def pending(self):return len(self.mode.pending)
    def install(self):
        self.hooks.append(self.model.register_forward_pre_hook(self.enter))
        visual=getattr(self.model,'visual',None)
        excluded={id(m) for m in visual.modules()} if visual is not None else set()
        for name,module in self.model.named_modules():
            if id(module) in excluded or not all(hasattr(module,k) for k in
                    ('gate_up_proj','down_proj','act_fn')):continue
            self.sites.append(name+'.down_proj')
            def projection(mod,args,result,_owner=id(module)):
                self.projections[_owner]=result[0] if isinstance(result,tuple) else result
            def output(mod,args,result,_name=name):
                require(self.active,'REFERENCE_PROJECTION_OUTSIDE_FORWARD')
                t=result[0] if isinstance(result,tuple) else result
                require(t is self.projections.pop(id(mod),None),
                        'REFERENCE_DOWN_PROJ_REAL_RETURN_REQUIRED')
                require(isinstance(t,torch.Tensor) and t.dtype==torch.bfloat16
                    and t.ndim==2 and t.is_contiguous() and t.numel()>0,
                    'REFERENCE_DEQUANTIZED_BF16_ND_REQUIRED')
                with _disable_current_modes():
                    event=self.event(_name+'.down_proj',t)
                    event.update(named_variant=self.variant,fraction_bits=self.fraction_bits,
                        effect='PHENOMENON_PROXY_NOT_ORIGINAL_MECHANISM',
                        target='whole_dequantized_projection_output',covered_elements=t.numel(),
                        actual_projection_return_identity=True)
                    def replacement(region):
                        value=truncate_bf16(region,self.fraction_bits)
                        event['changed_elements']=changed(region,value)
                        return value
                    self.mode.watch(t.view(-1),event,inject=self.fault,replacement=replacement)
            self.hooks.extend([module.down_proj.register_forward_hook(projection),
                               module.register_forward_hook(output)])
        require(self.sites,'REFERENCE_NO_NATURAL_DOWN_PROJ_BOUNDARY')
        return self.sites


class ProjectionMantissa3(ProjectionMantissa5):
    fraction_bits=3
    variant='row34_projection_mantissa3'


FACTORIES={'row34_projection_mantissa5':ProjectionMantissa5,
           'row34_projection_mantissa3':ProjectionMantissa3}
