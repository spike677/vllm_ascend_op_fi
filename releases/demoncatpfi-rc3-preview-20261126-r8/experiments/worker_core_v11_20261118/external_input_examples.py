"""Trusted transient input sites; examples, not yellow mechanism implementations.

No shared product route is changed. Factories are registered through ExternalSite.
The dispatch variant substitutes an actual operand, never edits an op's result.
"""
import json
import os
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
from fault_injection.sites.mla28 import require, fingerprint
from fault_injection.sites.norm18_variance import FullNormConsumerMode


def changed(a, b):
    return int((a.view(torch.uint8).reshape(-1, a.element_size()) !=
                b.view(torch.uint8).reshape(-1, b.element_size())).any(1).sum().item())


class InputReadMode(FullNormConsumerMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func._schema.name == 'vllm::triton_split_qkv_rmsnorm_mrope':
            require(not self.pending, 'EXTERNAL_INPUT_UNCONSUMED_BEFORE_FUSION')
            return func(*args, **(kwargs or {}))
        return super().__torch_dispatch__(func, types, args, kwargs)


class BaseSite:
    def __init__(self, model, runner, *, fault, region_leases):
        self.model=model; self.runner=runner; self.fault=fault; self.region_leases=region_leases
        self.events=[]; self.sites=[]; self.uncovered=[]; self.forward_calls=[]
        self.hooks=[]; self.parents=[]; self.active=False; self.input_calls={}

    @property
    def pending(self):return self.region_leases.count()

    def enter(self, *unused):
        require(not self.active and not self.pending, 'EXTERNAL_INPUT_FORWARD_OVERLAP')
        self.forward_calls.append({}); self.mode.__enter__(); self.active=True

    def event(self, name, x):
        calls=self.forward_calls[-1]; calls[name]=calls.get(name, 0)+1
        event=dict(site=name,request_ids=list(self.runner.input_batch.req_ids),
            eligible=True,applied=False,changed=0,changed_elements=0,consumed=False,
            restored=False,failed=False,input_shape=list(x.shape),target='last_row_first_two_columns')
        self.events.append(event); return event

    def finish_forward(self):
        require(self.active and not self.pending, 'EXTERNAL_INPUT_REAL_CONSUMER_PENDING')
        self.mode.__exit__(None,None,None); self.active=False

    def close(self):
        require(not self.pending, 'EXTERNAL_INPUT_LEASES_REMAIN')
        if self.active:self.finish_forward()
        for h in self.hooks:h.remove()
        self.hooks.clear()
        # Optional acceptance-only sidecar, not a product registration input.
        directory=os.environ.get('DCAT_EXTERNAL_SITE_EVIDENCE')
        if directory and self.fault:
            rank=torch.distributed.get_rank()
            with (Path(directory)/('external-input-rank'+str(rank)+'.jsonl')).open('a') as f:
                f.write(json.dumps(dict(rank=rank,kind=type(self).__name__,events=self.events))+'\n')


class ExternalModuleInputSite(BaseSite):
    """Mutate a real projection input until its first full-region value reader."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs); self.mode=InputReadMode(self)

    def install(self):
        self.hooks.append(self.model.register_forward_pre_hook(self.enter))
        visual=getattr(self.model,'visual',None)
        excluded={id(m) for m in visual.modules()} if visual is not None else set()
        for name, module in self.model.named_modules():
            if id(module) in excluded or not all(hasattr(module,k) for k in
                    ('quant_method','input_size','output_size')):continue
            weight=getattr(module,'weight',None)
            if not isinstance(weight,torch.Tensor) or weight.ndim != 2:continue
            self.sites.append(name)
            def before(mod,args,_name=name):
                require(self.active and args, 'EXTERNAL_MODULE_INPUT_OUTSIDE_FORWARD')
                x=args[0]
                require(isinstance(x,torch.Tensor) and x.dtype==torch.bfloat16 and
                        x.ndim==2 and x.is_contiguous() and x.shape[0]>0 and x.shape[1]>=2,
                        'EXTERNAL_MODULE_INPUT_LAYOUT_UNSUPPORTED')
                with _disable_current_modes():
                    event=self.event(_name,x); region=x[-1,:2]
                    event['whole_input_before']=fingerprint(x)
                    self.input_calls[id(mod)]=event
                    def value(t):
                        result=t*.75; event['changed_elements']=changed(t,result); return result
                    self.mode.watch(region,event,inject=self.fault,replacement=value)
            def after(mod,args,result):
                require(not self.mode.pending, 'EXTERNAL_MODULE_INPUT_NO_VALUE_CONSUMER')
                with _disable_current_modes():
                    event=self.input_calls.pop(id(mod))
                    event['whole_input_restored']=fingerprint(args[0])==event['whole_input_before']
                    require(event['whole_input_restored'], 'EXTERNAL_WHOLE_INPUT_NOT_RESTORED')
            self.hooks.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
        require(self.sites, 'EXTERNAL_NO_PROJECTION_INPUT_CANDIDATE')
        return self.sites


class ArgumentMode(TorchDispatchMode):
    def __init__(self,site):super().__init__(); self.site=site

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {}; site=self.site
        if func != torch.ops.aten.add.Tensor or len(args)<2:
            return func(*args,**kwargs)
        x,y=args[:2]
        if not (isinstance(x,torch.Tensor) and isinstance(y,torch.Tensor) and
                x.dtype==y.dtype==torch.bfloat16 and x.ndim==2 and x.shape==y.shape
                and x.is_contiguous() and x.shape[0]>0 and x.shape[1]>=2):
            return func(*args,**kwargs)
        with _disable_current_modes():
            event=site.event('aten.add.Tensor:operand0',x)
            original=fingerprint(x); event['original_input_sha256']=original
            operand=x.detach().clone() if site.fault else x
            token=None
            if site.fault:
                region=operand[-1,:2]; token=site.region_leases.acquire(region)
                value=region*.75
                event.update(before_sha256=fingerprint(region),changed_elements=changed(region,value))
                site.region_leases.write(token,value)
                event.update(applied=True,changed=int(event['changed_elements']>0),after_sha256=fingerprint(region))
            event['argument_substitution']=site.fault
            # Invoke exactly the intercepted real op, retaining every other argument.
            result=func(operand,y,*args[2:],**kwargs)
            site.region_leases.complete(site.region_leases.stream())
            event.update(consumed=True,consumer_operator='aten::add',same_stream_completed=True)
            if token is not None:
                site.region_leases.consumed(token); site.region_leases.restore(token)
                event.update(restored=True,restored_sha256=fingerprint(region))
            require(fingerprint(x)==original, 'EXTERNAL_DISPATCH_ORIGINAL_INPUT_MUTATED')
            event['original_input_unchanged']=True
            return result


class ExternalDispatchInputSite(BaseSite):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs); self.mode=ArgumentMode(self)

    def install(self):
        self.sites=['aten.add.Tensor:operand0']
        self.hooks.append(self.model.register_forward_pre_hook(self.enter))
        return self.sites
