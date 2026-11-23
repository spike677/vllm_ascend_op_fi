"""Read-only operator census for V4-0731; no fault permit or qualification.

Eligible here means a typed candidate with a scheduler request mapping. It does
not claim a validated injection/consumer/recovery adapter.
"""
import inspect
from collections import Counter
import torch
from torch.utils._python_dispatch import TorchDispatchMode

PATTERNS={
 '18':('rms_norm','rmsnorm'),
 '22':('inplace_partial_rotary_mul','partial_rotary'),
 '39op':('quant_matmul_weight_nz','quant_matmul','matmul','mm'),
 '39mr':('quant_matmul_weight_nz','quant_matmul','matmul','mm'),
 'batch':('dynamic_quant','dynamic_quant_v2'),
 '41':('compressor','reshape_and_cache','scatter_pa_kv_cache','mla_cache'),
 '8':('moe_distribute_combine_v2',),
 '28':('sparse_attn_sharedkv',),
}


def tensors(value):
    if isinstance(value,torch.Tensor):yield value
    elif isinstance(value,(list,tuple)):
        for x in value:yield from tensors(x)
    elif isinstance(value,dict):
        for x in value.values():yield from tensors(x)


def match(name,patterns):
    leaf=name.split('::')[-1].lower()
    return any(leaf==p or (p!='mm' and p in leaf) for p in patterns)


def reason(mechanism,args,out,requests):
    inputs=list(tensors(args));outputs=list(tensors(out))
    if not requests:return 'ACTUAL_SCHEDULER_REQUEST_JOIN_MISSING'
    if not outputs:return 'NO_TENSOR_OUTPUT'
    if mechanism in ('39op','39mr'):
        if outputs[0].dtype!=torch.bfloat16:return 'OUTPUT_NOT_BF16'
        if outputs[0].ndim!=2 or not outputs[0].is_contiguous():return 'OUTPUT_NOT_CONTIGUOUS_2D'
    if mechanism=='batch' and not any(t.dtype==torch.int8 for t in outputs):return 'NO_INT8_QUANT_OUTPUT'
    if mechanism=='8' and (outputs[0].ndim!=2 or outputs[0].shape[0]<2):return 'NO_MULTI_ROW_OUTPUT'
    if mechanism=='8':return 'COMBINE_OUTPUT_ROW_JOIN_UNPROVEN'
    if mechanism in ('28','41'):return 'SPARSE_COMPRESSED_LAYOUT_ADAPTER_REQUIRES_RUNTIME_JOIN'
    if not inputs:return 'NO_TENSOR_INPUT'
    return None


class Dsv4ObserveMode(TorchDispatchMode):
    def __init__(self,report,*,requests,phase,call_path=True):
        super().__init__();self.report=report;self.requests=requests;self.phase=phase;self.call_path=call_path
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {};out=func(*args,**kwargs)
        name=func._schema.name
        matched=[m for m,patterns in PATTERNS.items() if match(name,patterns)]
        if not matched:return out
        path=[]
        if self.call_path:
            for frame in inspect.stack(context=0)[1:18]:
                file=frame.filename.replace('\\','/')
                if '/vllm' in file:path.append(dict(file=file[file.index('/vllm')+1:],function=frame.function,line=frame.lineno))
        for m in matched:
            why=reason(m,(args,kwargs),out,self.requests);site=(name,self.phase,tuple((v['file'],v['function'],v['line']) for v in path))
            rows=self.report[m];row=rows.setdefault(site,dict(operator=name,phase=self.phase,call_path=path,observed=0,eligible=0,reasons=Counter(),request_ids=set(),input_shapes=[],output_shapes=[]))
            row['observed']+=1;row['request_ids'].update(self.requests)
            if why:row['reasons'][why]+=1
            else:row['eligible']+=1
            if not row['input_shapes']:
                row['input_shapes']=[dict(shape=list(t.shape),dtype=str(t.dtype)) for t in tensors((args,kwargs))]
                row['output_shapes']=[dict(shape=list(t.shape),dtype=str(t.dtype)) for t in tensors(out)]
        return out


def empty_report():return {m:{} for m in PATTERNS}


def export_report(report,rank):
    result={}
    for m,sites in report.items():
        rows=[]
        for row in sites.values():
            rows.append(dict(row,reasons=dict(row['reasons']),request_ids=sorted(row['request_ids'])))
        result[m]=dict(observed=sum(r['observed'] for r in rows),eligible=sum(r['eligible'] for r in rows),sites=rows,
            unavailable_reason=None if rows else 'NO_MATCHED_PUBLIC_DISPATCH_CALL_OBSERVED',
            injection_adapter_validated=False)
    return dict(schema='demoncat.dsv4_0731.observe.v1',rank=rank,mechanisms=result,
        read_only=True,injected_calls=0,eligibility_scope='TYPED_CANDIDATE_ONLY_NOT_ADMISSION',qualification_granted=False)
