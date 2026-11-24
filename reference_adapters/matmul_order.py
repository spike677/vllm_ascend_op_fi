"""Library19/34 BF16 arithmetic proxies. Quantized operands are not dequantized.

Only an observed mm/addmm/linear whose real BF16 weight belongs to the model is
eligible. Fixed last row / first 64 columns bounds CPU reference cost. Library34
requires an actual bias-bearing fused producer; no synthetic residual is added.
"""
import torch
from torch.utils._python_dispatch import _disable_current_modes
from experiments.worker_core_v11_20261118.external_input_examples import BaseSite, InputReadMode, changed
from fault_injection.sites.mla28 import require, fingerprint


def reference(left, right, bias=None, *, early_round=False):
    left=left.detach().float().cpu();right=right.detach().float().cpu()
    bias=None if bias is None else bias.detach().float().cpu()
    blocks=list(range(0,left.numel(),128))
    def accumulate(order):
        total=torch.zeros(right.shape[1],dtype=torch.float32)
        for start in order:
            total += left[start:start+128] @ right[start:start+128]
        return total
    full=accumulate(blocks)
    alternate=accumulate(reversed(blocks))
    if early_round:
        require(bias is not None,'REFERENCE_REAL_BIAS_REQUIRED')
        alternate=full.to(torch.bfloat16).float()+bias
    elif bias is not None:alternate=alternate+bias
    if bias is not None:full=full+bias
    return full.to(torch.bfloat16),alternate.to(torch.bfloat16)


class MatmulMode(InputReadMode):
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {};name=func._schema.name;site=self.site
        if name not in ('aten::mm','aten::addmm','aten::linear','npu::npu_linear'):
            return super().__torch_dispatch__(func,types,args,kwargs)
        values={a.name:a.default_value for a in func._schema.arguments}
        values.update(zip((a.name for a in func._schema.arguments),args));values.update(kwargs)
        if name=='aten::mm':x,w=args[:2];bias=None
        elif name=='aten::addmm':
            x,w=values['mat1'],values['mat2'];bias=values['self']
            if values.get('alpha',1)!=1 or values.get('beta',1)!=1:
                return super().__torch_dispatch__(func,types,args,kwargs)
        else:
            x,w=args[:2];bias=values.get('bias')
        if not isinstance(w,torch.Tensor) or w.untyped_storage()._cdata not in site.weights:
            return super().__torch_dispatch__(func,types,args,kwargs)
        if x.ndim!=2 or w.ndim!=2 or x.dtype!=torch.bfloat16 or w.dtype!=torch.bfloat16:
            return super().__torch_dispatch__(func,types,args,kwargs)
        if site.early_round and bias is None:
            return super().__torch_dispatch__(func,types,args,kwargs)
        # Same object/storage must be the actual dispatch operand, including a transpose view.
        with _disable_current_modes():
            matrix=w.t() if name in ('aten::linear','npu::npu_linear') else w
            require(matrix.shape[0]==x.shape[1],'REFERENCE_BF16_MATMUL_LAYOUT')
            width=min(64,matrix.shape[1])
            if bias is not None:
                require(bias.ndim==1 and bias.numel()==matrix.shape[1],'REFERENCE_MATMUL_BIAS_LAYOUT')
            before=fingerprint(w)
            normal,alternate=reference(x[-1],matrix[:,:width],None if bias is None else bias[:width],early_round=site.early_round)
        result=super().__torch_dispatch__(func,types,args,kwargs)
        require(result.ndim==2 and result.dtype==torch.bfloat16 and result.is_contiguous(),
            'REFERENCE_BF16_MATMUL_RETURN')
        with _disable_current_modes():
            require(fingerprint(w)==before,'REFERENCE_MATMUL_WEIGHT_CHANGED')
            observed=result[-1,:width].detach().float().cpu()
            tolerance=torch.maximum(observed.abs(),normal.float().abs())*(2**-7)+1e-6
            require(bool(((observed-normal.float()).abs()<=tolerance).all()),'REFERENCE_MATMUL_CONTROL_MISMATCH')
            event=site.event(site.weights[w.untyped_storage()._cdata],result)
            event.update(named_variant=site.variant,effect='PHENOMENON_PROXY_NOT_ORIGINAL_MECHANISM',
                native_operator=name,native_return_identity=True,shape=list(result.shape),
                input_shape=list(x.shape),weight_shape=list(w.shape),readonly_inputs_weights_unchanged=True,
                target='last_row_first_64_columns',k_block=128,reverse_k_order=not site.early_round,
                early_bf16_before_real_bias=site.early_round,control_passed=True,
                control_max_abs=float((observed-normal.float()).abs().max()),weight_sha256=before,
                bf16_operands=True,quantized_operand_recomputed=False)
            def replacement(region):
                value=alternate.to(region.device)
                event['changed_elements']=changed(region,value)
                return value
            self.watch(result[-1,:width],event,inject=site.fault,replacement=replacement)
        return result


class ReverseSplitK(BaseSite):
    early_round=False
    variant='row19_bf16_reverse_splitk'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.mode=MatmulMode(self);self.weights={}
    @property
    def pending(self):return len(self.mode.pending)
    def install(self):
        root=self.model.visual if self.early_round else self.model
        visual=getattr(self.model,'visual',None)
        excluded={id(m) for m in visual.modules()} if visual is not None and not self.early_round else set()
        for name,module in root.named_modules():
            w=getattr(module,'weight',None)
            if id(module) in excluded or not isinstance(w,torch.Tensor) or w.ndim!=2 or w.dtype!=torch.bfloat16:continue
            if not (hasattr(module,'quant_method') or isinstance(module,torch.nn.Linear)):continue
            self.weights[w.untyped_storage()._cdata]=name
        require(self.weights,'REFERENCE_NO_REAL_BF16_WEIGHT')
        self.sites=list(self.weights.values())
        self.hooks=[root.register_forward_pre_hook(self.enter)]
        return self.sites


class EarlyBiasRound(ReverseSplitK):
    early_round=True
    variant='row34_bf16_before_bias'


FACTORIES={'row19_bf16_reverse_splitk':ReverseSplitK,'row34_bf16_before_bias':EarlyBiasRound}
