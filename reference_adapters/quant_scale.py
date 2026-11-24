"""Library rows 9/10: transient scale error proxies, not original kernel defects.

Native DynamicQuant supplies the scale. Requantize the same x using a fixed
scale multiplier, preserving signed int8/FP32 layouts. The original QMM reads
these buffers; restore only after the owning quantized linear apply completes.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes

from experiments.worker_core_v11_20261118.external_input_examples import BaseSite, changed
from fault_injection.sites.mla28 import require, fingerprint


def requantize(x, scale, factor):
    """CPU-FP32 reference arithmetic, fixed factor supplied only by factory class."""
    require(factor in (.25, .5, 2., 4.), 'REFERENCE_FIXED_SCALE_VARIANT_REQUIRED')
    require(x.ndim == 2 and scale.shape == (x.shape[0],), 'REFERENCE_SCALE_LAYOUT')
    host_x=x.detach().float().cpu(); host_s=scale.detach().float().cpu()*factor
    require(bool(torch.isfinite(host_x).all()) and bool(torch.isfinite(host_s).all())
            and bool((host_s>=0).all()), 'REFERENCE_QUANT_NONFINITE_OR_NEGATIVE_SCALE')
    zero=host_s==0
    require(not bool((host_x[zero]!=0).any()), 'REFERENCE_ZERO_SCALE_NONZERO_ROW')
    divisor=torch.where(zero,torch.ones_like(host_s),host_s)
    raw=torch.round(host_x/divisor[:,None])
    q=raw.clamp(-127,127).to(torch.int8)
    return q.to(x.device),host_s.to(scale.device),dict(
        saturated_elements=int((q.abs().to(torch.int16)==127).sum()),
        clipped_elements=int((raw.abs()>127).sum()),zero_elements=int((q==0).sum()),
        rounding='nearest_even',signed_range=[-127,127],scale_multiplier=factor)


def argument(func,args,kwargs,name):
    for i,a in enumerate(func._schema.arguments):
        if a.name==name:return args[i] if i<len(args) else kwargs.get(name)
    return None


class QuantMode(TorchDispatchMode):
    quant_name='npu::npu_dynamic_quant'
    qmm_name='npu::npu_quant_matmul'
    def __init__(self,site):super().__init__();self.site=site
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {};site=self.site;ctx=site.context;name=func._schema.name
        if name==self.quant_name:
            result=func(*args,**kwargs)
            if ctx is None:
                site.uncovered.append(dict(operator=name,reason='OUTSIDE_DISCOVERED_DYNAMIC_LINEAR'))
                return result
            require(ctx.get('buffers') is None and args and args[0] is ctx['input'],
                    'REFERENCE_QUANT_INPUT_OR_MULTIPLICITY')
            require(isinstance(result,(tuple,list)) and len(result)==2,'REFERENCE_QUANT_RETURN_SCHEMA')
            q,s=result;x=args[0];e=ctx['event']
            require(x.dtype==torch.bfloat16 and q.dtype==torch.int8 and s.dtype==torch.float32
                    and q.shape==x.shape and s.shape==(x.shape[0],)
                    and q.is_contiguous() and s.is_contiguous(),'REFERENCE_QUANT_DTYPE_OR_LAYOUT')
            ctx['buffers']=(q,s)
            with _disable_current_modes():
                e.update(q_shape=list(q.shape),scale_shape=list(s.shape),input_sha256=fingerprint(x),
                         before_sha256=[fingerprint(q),fingerprint(s)],
                         native_zero_elements=int((q==0).sum().item()))
                if site.fault:
                    nq,ns,stats=requantize(x,s,site.factor);e.update(stats)
                    e['changed_elements']=changed(q,nq)+changed(s,ns)
                    ctx['tokens']=[site.region_leases.acquire(t) for t in (q,s)]
                    for token,value in zip(ctx['tokens'],(nq,ns)):site.region_leases.write(token,value)
                    e.update(applied=True,changed=int(e['changed_elements']>0),
                             after_sha256=[fingerprint(q),fingerprint(s)])
            return result
        if name==self.qmm_name and ctx is not None:
            require(ctx.get('buffers') is not None,'REFERENCE_QMM_WITHOUT_QUANT')
            q,s=ctx['buffers'];actual=argument(func,args,kwargs,'pertoken_scale')
            require(args[0] is q and actual is s,'REFERENCE_QMM_EXACT_Q_SCALE_REQUIRED')
            with _disable_current_modes():
                hashes=[fingerprint(q),fingerprint(s)]
                require(hashes==ctx['event'].get('after_sha256',ctx['event']['before_sha256']),
                        'REFERENCE_QMM_VALUE_CHANGED')
            result=func(*args,**kwargs)
            ctx['event']['qmm_reads']+=1
            ctx['event'].update(consumer_operator=name,consumer_sha256=hashes)
            return result
        return func(*args,**kwargs)


class ScaleSite(BaseSite):
    factor=None
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.mode=QuantMode(self);self.context=None;self.methods=[]
    def install(self):
        self.hooks.append(self.model.register_forward_pre_hook(self.enter))
        seen=set()
        for name,module in self.model.named_modules():
            scheme=getattr(getattr(module,'quant_method',None),'quant_method',None)
            if type(scheme).__name__!='AscendW8A8DynamicLinearMethod':continue
            require(id(scheme) not in seen and 'apply' not in vars(scheme),'REFERENCE_SCHEME_OWNER_CONFLICT')
            seen.add(id(scheme));original=scheme.apply
            source=hashlib.sha256(Path(inspect.getsourcefile(type(scheme))).read_bytes()).hexdigest()
            def apply(layer,x,*args,_name=name,_original=original,_sha=source,**kwargs):
                require(self.active and self.context is None,'REFERENCE_NESTED_QUANT_APPLY')
                e=self.event(_name,x);e.update(named_variant=type(self).__name__,
                    effect='PHENOMENON_PROXY_NOT_ORIGINAL_MECHANISM',target='all_actual_quant_rows',
                    qmm_reads=0,scheme_source_sha256=_sha)
                ctx=dict(input=x,event=e);self.context=ctx
                try:
                    result=_original(layer,x,*args,**kwargs)
                    require(ctx.get('buffers') is not None and e['qmm_reads']>0,
                            'REFERENCE_QUANT_CONSUMER_MISSING')
                    with _disable_current_modes():
                        self.region_leases.complete(self.region_leases.stream())
                        e['consumed']=True
                        require(fingerprint(x)==e['input_sha256'],'REFERENCE_INPUT_CHANGED')
                        if self.fault:
                            for token in ctx['tokens']:
                                self.region_leases.consumed(token);self.region_leases.restore(token)
                            e.update(restored=True,restored_sha256=[fingerprint(t) for t in ctx['buffers']])
                            require(e['restored_sha256']==e['before_sha256'],'REFERENCE_QUANT_RESTORE')
                    return result
                except BaseException:
                    e['failed']=True;raise
                finally:self.context=None
            scheme.apply=apply;self.methods.append((scheme,apply));self.sites.append(name)
        require(self.sites,'REFERENCE_NO_SEPARATE_DYNAMIC_QUANT_LINEAR')
        return self.sites
    def close(self):
        require(self.context is None and not self.pending,'REFERENCE_QUANT_PENDING')
        for scheme,wrapper in self.methods:
            require(vars(scheme).get('apply') is wrapper,'REFERENCE_SCHEME_OWNER_CHANGED')
            del scheme.apply
        self.methods.clear();super().close()


class SaturationHalf(ScaleSite):factor=.5
class SaturationQuarter(ScaleSite):factor=.25
class SqueezeDouble(ScaleSite):factor=2.
class SqueezeQuadruple(ScaleSite):factor=4.


FACTORIES={'row9_scale_half':SaturationHalf,'row9_scale_quarter':SaturationQuarter,
           'row10_scale_double':SqueezeDouble,'row10_scale_quadruple':SqueezeQuadruple}
