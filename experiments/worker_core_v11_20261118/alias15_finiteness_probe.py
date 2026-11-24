"""Acceptance-only read-only observers; never initialize or alter an activation."""
import json
import os
from pathlib import Path
import torch
from fault_injection.sites.mla28 import fingerprint


def install(worker):
    runner=worker.model_runner; model=runner.model; handles=[]
    path=Path('/evidence')/('alias15-finite-rank'+str(worker.rank)+'.jsonl')
    def record(kind, name, tensor):
        if not isinstance(tensor,torch.Tensor):return
        row=dict(rank=worker.rank,kind=kind,site=name,request_ids=list(runner.input_batch.req_ids),
                 shape=list(tensor.shape),sha256=fingerprint(tensor),
                 nonfinite=int((~torch.isfinite(tensor)).sum().item()),elements=tensor.numel())
        with path.open('a') as stream:stream.write(json.dumps(row)+'\n')
    def hidden(mod,args,result):
        record('model_hidden','model',result)
    handles.append(model.register_forward_hook(hidden))
    original_logits=model.compute_logits
    def logits(*args,**kwargs):
        value=original_logits(*args,**kwargs)
        record('model_logits','model',value)
        return value
    model.compute_logits=logits
    modules=dict(model.named_modules())
    for name,module in modules.items():
        impl=getattr(module,'impl',None)
        if impl is None or type(impl).__name__!='AscendAttentionBackendImpl':continue
        parent=modules.get(name.rsplit('.',1)[0]);consumer=getattr(parent,'o_proj',None)
        if consumer is None:continue
        def output(mod,args,value,_name=name):record('attention_base_return',_name,value)
        def projection(mod,args,_name=name):record('projection_input',_name,args[0])
        handles.extend([module.register_forward_hook(output),consumer.register_forward_pre_hook(projection)])
    worker._alias15_readonly_diagnostic_handles=handles
    if os.environ.get('DCAT_ALIAS15_ALLOCATOR_STRESS') == '1':
        from alias15_allocator_stress import install as install_stress
        install_stress(worker)
    return dict(rank=worker.rank,readonly_hooks=len(handles),changes_model_values=False)
