"""Column-sharded projection range proxy; post-reduction outputs are refused."""
import hashlib,inspect
from pathlib import Path
import torch
from .projection39 import Projection39Site
from .mla28 import require
from .batch_quant import SOURCE_SHA as QUANT_SHA
from fault_injection.engines.matmul_range import recompute_cell


class Matmul39Site(Projection39Site):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.captured={};self.schemes=[]

    def classify(self,module,args,t):
        ok,reason=super().classify(module,args,t)
        if not ok:return ok,reason
        from vllm.model_executor.layers.linear import ColumnParallelLinear
        if not isinstance(module,ColumnParallelLinear) or module.gather_output:
            return False,'ONLY_LOCAL_COLUMN_SHARDS_NO_COLLECTIVE'
        if t.ndim!=2 or len(args)<1 or args[0].ndim!=2:
            return False,'MATRIX_PROJECTION_ONLY'
        scheme=getattr(module.quant_method,'quant_method',module.quant_method)
        if type(scheme).__name__ not in ('AscendW8A8DynamicLinearMethod','UnquantizedLinearMethod','AscendUnquantizedLinearMethod'):
            return False,'UNREVIEWED_PROJECTION_ARITHMETIC'
        if type(scheme).__name__!='AscendW8A8DynamicLinearMethod':
            return False,'UNQUANTIZED_REFERENCE_CPU_ONLY_NOT_NPU_QUALIFIED'
        return True,None

    def install(self):
        import torch_npu
        from vllm.model_executor.layers.linear import ColumnParallelLinear
        seen=set()
        for name,module in self.model.named_modules():
            if not isinstance(module,ColumnParallelLinear) or module.gather_output:continue
            scheme=getattr(module.quant_method,'quant_method',module.quant_method)
            if type(scheme).__name__!='AscendW8A8DynamicLinearMethod':continue
            require(id(scheme) not in seen,'MATMUL_SHARED_SCHEME');seen.add(id(scheme))
            require(hashlib.sha256(Path(inspect.getsourcefile(type(scheme))).read_bytes()).hexdigest()==QUANT_SHA,'MATMUL_QUANT_SOURCE_CHANGED')
            require('apply' not in vars(scheme),'MATMUL_SCHEME_ALREADY_OWNED')
            original=scheme.apply
            def apply(layer,x,*args,_original=original,**kwargs):
                real=torch_npu.npu_quant_matmul;records=[]
                require(not getattr(real,'_dcat_site',False),'MATMUL_QMM_ALREADY_OWNED')
                def qmm(q,w,s,*extra,**kw):
                    require(not extra,'MATMUL_QMM_POSITIONAL_SCHEMA_CHANGED')
                    out=real(q,w,s,**kw);records.append((q,w,s,kw,out));return out
                qmm._dcat_site=True;torch_npu.npu_quant_matmul=qmm
                try:out=_original(layer,x,*args,**kwargs)
                finally:
                    require(torch_npu.npu_quant_matmul is qmm,'MATMUL_QMM_OWNER_CHANGED')
                    torch_npu.npu_quant_matmul=real
                require(len(records)==1,'MATMUL_CHUNKED_QMM_NOT_SUPPORTED')
                self.captured[id(layer)]=records[0]
                return out
            scheme.apply=apply;self.schemes.append((scheme,apply))
        return super().install()

    def replacement(self,module,args,t,event):
        scheme=getattr(module.quant_method,'quant_method',module.quant_method)
        if type(scheme).__name__=='AscendW8A8DynamicLinearMethod':
            require(id(module) in self.captured,'MATMUL_REAL_QMM_INPUTS_MISSING')
            q,w,ws,kw,out=self.captured.pop(id(module))
            require(out is t and q.dtype==w.dtype==torch.int8 and w.shape==(q.shape[1],t.shape[1]),'MATMUL_QMM_RETURN_OR_LAYOUT_CHANGED')
            ps=kw.get('pertoken_scale');bias=kw.get('bias')
            require(ps is not None and ps.shape==(q.shape[0],) and ws.numel()==t.shape[1] and bias is None,'MATMUL_QMM_SCALE_OR_BIAS_UNSUPPORTED')
            left,right=q[-1],w[:,0]
            scale=float(ps[-1].item())*float(ws.reshape(-1)[0].item());b=0.
            event['real_qmm_inputs']=True
        else:
            x=args[0];w=module.weight
            require(w.shape==(t.shape[1],x.shape[1]),'MATMUL_WEIGHT_LAYOUT_CHANGED')
            left,right=x[-1],w[0];scale=1.
            bias=module.bias if not module.skip_bias_add else None
            b=0. if bias is None else float(bias[0].item())
            event['real_qmm_inputs']=False
        full,faulty,info=recompute_cell(left,right,scale=scale,bias=b)
        observed=float(t[-1,0].item());reference=float(full.item())
        # One BF16 spacing plus FP32 scaling tolerance; this gate validates
        # operands/layout before deliberately changing the K range.
        tolerance=max(abs(reference),abs(observed))*2**-7+1e-6
        require(abs(observed-reference)<=tolerance,'MATMUL_FULL_RANGE_CONTROL_MISMATCH')
        event.update(range_reference=info,full_range_reference=reference,native_cell=observed,
                     full_range_control_passed=True,non_target_exact=True)
        value=faulty.to(device=t.device,dtype=t.dtype).reshape(1)
        require(torch.isfinite(value).all().item(),'MATMUL_FAULT_NONFINITE')
        return lambda region:value

    def close(self):
        super().close()
        for scheme,wrapped in reversed(self.schemes):
            require(vars(scheme).get('apply') is wrapped,'MATMUL_SCHEME_OWNER_CHANGED');delattr(scheme,'apply')
        self.schemes.clear();self.captured.clear()
