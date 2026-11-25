"""Bounded per-call D1 evidence; no synthetic EP row join or conditional collectives."""
from collections import Counter
import hashlib
import inspect
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from fault_injection.sites.dsv4_observer import PATTERNS, match, reason, tensors
from experiments.worker_core_v11_20261118.dsv4_probe_support import scheduler_rows


class CallJournal(TorchDispatchMode):
    def __init__(self,state,runner,scheduler,rank,groups,modules,*,requests,phase):
        super().__init__()
        self.state,self.runner,self.scheduler=state,runner,scheduler
        self.rank,self.groups,self.modules=rank,groups,modules
        self.requests,self.phase=requests,phase

    def tensor(self,value,path,*,capture_indices=False):
        if isinstance(value,torch.Tensor):
            row=dict(path=path,object_id=id(value),shape=list(value.shape),dtype=str(value.dtype),
                     stride=list(value.stride()),device=str(value.device))
            try:row.update(data_ptr=value.data_ptr(),storage_offset=value.storage_offset())
            except RuntimeError:row['data_ptr']=None
            if capture_indices and value.dtype in (torch.int16,torch.int32,torch.int64,torch.bool):
                if value.numel()<=self.state['integer_budget_remaining']:
                    data=value.detach().cpu().contiguous();raw=data.numpy().tobytes()
                    row.update(values=data.reshape(-1).tolist(),sha256=hashlib.sha256(raw).hexdigest())
                    self.state['integer_budget_remaining']-=value.numel()
                else:row['values']=None;row['reason']='INTEGER_CAPTURE_BUDGET_EXCEEDED';self.state['truncated']=True
            yield row
        elif isinstance(value,(list,tuple)):
            for i,item in enumerate(value):yield from self.tensor(item,path+'/'+str(i),capture_indices=capture_indices)
        elif isinstance(value,dict):
            for key,item in value.items():yield from self.tensor(item,path+'/'+str(key),capture_indices=capture_indices)

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {};name=func._schema.name
        moe=any(x in name.lower() for x in ('moe_distribute_dispatch','moe_distribute_combine','moe_init_routing','moe_finalize_routing'))
        matched=[m for m,p in PATTERNS.items() if match(name,p)]
        selected=moe or bool(matched)
        before=None
        if selected and len(self.state['calls'])<self.state['max_calls']:
            try:
                named={a.name:(args[i] if i<len(args) else kwargs.get(a.name)) for i,a in enumerate(func._schema.arguments)}
                before=list(self.tensor(named,'input',capture_indices=moe))
            except Exception as error:self.state['errors'].append(dict(stage='operator_input',operator=name,error=repr(error)))
        # Exactly once. Original operator errors propagate unchanged.
        out=func(*args,**kwargs)
        if not selected:return out
        if len(self.state['calls'])>=self.state['max_calls']:
            self.state['truncated']=True;return out
        try:self.record(name,str(func._schema),args,kwargs,out,matched,before,moe)
        except Exception as error:self.state['errors'].append(dict(stage='operator_observation',operator=name,error=repr(error)))
        return out

    def record(self,name,schema,args,kwargs,out,matched,inputs,moe):
        mapping=scheduler_rows(self.runner,self.scheduler);layers=[];path=[]
        for frame in inspect.stack(context=0)[1:24]:
            obj=frame.frame.f_locals.get('self')
            if id(obj) in self.modules:layers.append(self.modules[id(obj)])
            file=frame.filename.replace('\\','/')
            if '/vllm' in file:path.append(dict(file=file[file.index('/vllm')+1:],function=frame.function,line=frame.lineno))
        for m in matched:
            why=reason(m,(args,kwargs),out,self.requests)
            key=(name,self.phase,tuple((v['file'],v['function'],v['line']) for v in path))
            row=self.state['report'][m].setdefault(key,dict(operator=name,phase=self.phase,call_path=path,observed=0,eligible=0,reasons=Counter(),request_ids=set(),input_shapes=[],output_shapes=[]))
            row['observed']+=1;row['request_ids'].update(self.requests)
            if why:row['reasons'][why]+=1
            else:row['eligible']+=1
            if not row['input_shapes']:
                row['input_shapes']=[dict(shape=list(t.shape),dtype=str(t.dtype)) for t in tensors((args,kwargs))]
                row['output_shapes']=[dict(shape=list(t.shape),dtype=str(t.dtype)) for t in tensors(out)]
        outputs=list(self.tensor(out,'output',capture_indices=moe));seq=len(self.state['calls'])+1
        row=dict(sequence=seq,step=self.state['steps'],operator=name,schema=schema,
                 layer=layers[0] if layers else None,layer_path=layers,phase=self.phase,
                 rank=self.rank,groups=self.groups,request_ids=list(self.requests),input_rows=mapping,
                 inputs=inputs,outputs=outputs,padding=None,valid_output_rows=None,
                 combine_output_row_join='COMBINE_OUTPUT_ROW_JOIN_UNPROVEN' if 'combine' in name.lower() else None)
        # Preserve explicit group/routing attributes without filling omitted defaults.
        row['scalar_keyword_arguments']={k:v for k,v in kwargs.items() if v is None or type(v) in (int,float,bool,str)}
        row['scalar_positional_arguments']={str(i):v for i,v in enumerate(args) if v is None or type(v) in (int,float,bool,str)}
        if 'dispatch' in name.lower():self.state['dispatches'].append(row)
        if 'combine' in name.lower():
            keys={(x.get('data_ptr'),x.get('sha256')) for x in inputs or [] if x.get('sha256') and x.get('data_ptr')}
            row['dispatch_candidates']=[d['sequence'] for d in self.state['dispatches'] if d['step']==row['step'] and d['layer']==row['layer']
                and keys.intersection((x.get('data_ptr'),x.get('sha256')) for x in d['outputs'] if x.get('sha256'))]
            row['join_reason']='EP_CROSS_RANK_ROUTING_AND_PADDING_SEMANTICS_REQUIRE_REVIEW'
        self.state['calls'].append(row)
