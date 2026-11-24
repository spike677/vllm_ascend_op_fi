"""Library3 fixed gate-rounding proxy; original fused-kernel replay is not claimed."""
import torch
from torch.utils._python_dispatch import _disable_current_modes
from experiments.worker_core_v11_20261118.external_input_examples import BaseSite, InputReadMode, changed
from fault_injection.sites.mla28 import require, fingerprint


def recompute(x):
    require(x.ndim == 2 and x.shape[-1] % 2 == 0 and x.dtype == torch.bfloat16,
        'REFERENCE_SWIGLU_LAYOUT')
    gate, up = x.float().chunk(2, -1)
    gate = (gate * torch.sigmoid(gate)).half().float()
    return (gate * up).to(x.dtype)


class SwigluMode(InputReadMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = super().__torch_dispatch__(func, types, args, kwargs)
        site=self.site
        if site.activation is not None and any(k in func._schema.name.lower() for k in ('swiglu','silu_and_mul')):
            require(site.native is None, 'REFERENCE_MULTIPLE_SWIGLU_PRODUCERS')
            require(args and args[0] is site.activation, 'REFERENCE_SWIGLU_INPUT_IDENTITY')
            site.native=(func._schema.name,result)
        return result


class GateFp16Round(BaseSite):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.mode=SwigluMode(self);self.activation=None;self.native=None
    @property
    def pending(self):return len(self.mode.pending)
    def install(self):
        self.hooks.append(self.model.register_forward_pre_hook(self.enter))
        for name,module in self.model.named_modules():
            if not all(hasattr(module,k) for k in ('gate_up_proj','down_proj','act_fn')):continue
            self.sites.append(name+'.act_fn')
            def before(mod,args):
                require(self.active and self.activation is None and len(args)==1,'REFERENCE_SWIGLU_ORDER')
                self.activation=args[0];self.input_sha=fingerprint(args[0]);self.native=None
            def after(mod,args,result,_name=name):
                require(self.native is not None and self.native[1] is result,'REFERENCE_NATIVE_SWIGLU_REQUIRED')
                require(result.is_contiguous() and result.dtype==torch.bfloat16,'REFERENCE_SWIGLU_OUTPUT')
                with _disable_current_modes():
                    event=self.event(_name+'.act_fn',result)
                    event.update(named_variant='row3_gate_fp16_round',effect='PHENOMENON_PROXY_NOT_ORIGINAL_MECHANISM',
                        target='all_gate_elements',native_operator=self.native[0],input_sha256=self.input_sha,
                        gate_rule='silu_fp32_then_fp16_round_then_fp32_mul_up_then_bf16')
                    value=recompute(self.activation)
                    require(fingerprint(self.activation)==self.input_sha,'REFERENCE_SWIGLU_INPUT_CHANGED')
                    def replacement(t):
                        event['changed_elements']=changed(t,value.reshape(-1))
                        return value.reshape(-1)
                    self.mode.watch(result.view(-1),event,inject=self.fault,replacement=replacement)
                self.activation=None;self.native=None
            self.hooks.extend([module.act_fn.register_forward_pre_hook(before),module.act_fn.register_forward_hook(after)])
        require(self.sites,'REFERENCE_NO_SWIGLU_BOUNDARY')
        return self.sites


FACTORIES={'row3_gate_fp16_round':GateFp16Round}
