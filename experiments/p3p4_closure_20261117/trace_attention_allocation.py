"""Read-only line tracing of actual empty -> view -> FIA destination copy.

No tensor initialization, API replacement, filtering of statistics, or fault
injection. This auxiliary observer records full-byte hashes and finite counts;
it cannot retroactively prove the storage history of an older process.
"""
import ast
import hashlib
import inspect
from pathlib import Path
import sys
import weakref


def tensor_record(x):
    import torch
    if not isinstance(x,torch.Tensor) or x.device.type!='npu' or not x.is_contiguous():
        raise RuntimeError('CONTIGUOUS_REAL_NPU_TENSOR_REQUIRED')
    if x.numel()*x.element_size()>16*1024**2:raise RuntimeError('OBSERVATION_BUDGET_EXCEEDED')
    cpu=x.detach().cpu()
    storage=x.untyped_storage()
    return dict(storage_identity=int(storage._cdata),data_ptr=x.data_ptr(),storage_bytes=storage.nbytes(),
        offset=x.storage_offset(),shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype),
        bytes_sha256=hashlib.sha256(cpu.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        elements=x.numel(),nan=int(torch.isnan(cpu).sum()),inf=int(torch.isinf(cpu).sum()),
        finite=int(torch.isfinite(cpu).sum()),deterministic=torch.are_deterministic_algorithms_enabled(),
        fill_uninitialized_memory=bool(torch.utils.deterministic.fill_uninitialized_memory))


class AttentionAllocationTrace:
    def __init__(self,model):
        from fault_injection.native import _candidates
        self.model=model;self.sources={};self.allocations=[];self.copies=[];self.frames={}
        self.active=False;self.roots=0;self.error=None;self.latest={}
        for _,module,kinds in _candidates(model):
            if 'native_attention_candidate' not in kinds:continue
            self.add_source(inspect.unwrap(module.forward),'allocation')
            # Discover the real implementation's copy statement; backend
            # method names differ across inspected vLLM-Ascend versions.
            for _,method in inspect.getmembers(type(module.impl),inspect.isfunction):
                self.add_source(inspect.unwrap(method),'copy')
        if not self.sources:raise RuntimeError('NO_INSPECTABLE_NATURAL_ATTENTION_SOURCE')

    def add_source(self,fn,kind):
        code=getattr(getattr(fn,'__func__',fn),'__code__',None)
        if code is None:raise RuntimeError('PYTHON_SOURCE_BOUNDARY_REQUIRED')
        path=Path(code.co_filename);text=path.read_text();tree=ast.parse(text)
        source_lines,first_line=inspect.getsourcelines(fn)
        last_line=first_line+len(source_lines)-1
        lines=[]
        for node in ast.walk(tree):
            if not isinstance(node,ast.Assign) or len(node.targets)!=1:continue
            if not first_line <= node.lineno <= last_line:continue
            target=node.targets[0]
            if kind=='allocation' and isinstance(target,ast.Name) and target.id=='output':
                if isinstance(node.value,ast.Call) and ast.unparse(node.value.func) in ('torch.empty','torch.empty_like'):
                    lines.append(node.lineno)
            if kind=='copy' and isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and target.value.id=='output':
                if isinstance(node.value,ast.Subscript) and isinstance(node.value.value,ast.Name) and node.value.value.id=='attn_output':
                    lines.append(node.lineno)
        if lines:self.sources[code]=dict(kind=kind,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            statements={line:text.splitlines()[line-1].strip() for line in lines})

    def begin(self,module,args):
        if self.active or sys.gettrace() is not None:raise RuntimeError('FOREIGN_OR_RECURSIVE_TRACE')
        self.roots+=1;self.active=True;self.frames={};self.latest={}
        sys.settrace(self.trace)

    def end(self,module,args,output):
        if not self.active:return
        if sys.gettrace() is not self.trace_callback and sys.gettrace()!=self.trace:
            raise RuntimeError('TRACE_OWNER_CHANGED')
        sys.settrace(None);self.active=False;self.frames={};self.latest={}
        if self.error:raise RuntimeError(self.error)

    def trace(self,frame,event,arg):
        source=self.sources.get(frame.f_code)
        if source is None:return None
        try:
            fid=id(frame);state=self.frames.setdefault(fid,{})
            if state.get('pending') and event in ('line','return'):
                pending=state.pop('pending')
                record=tensor_record(frame.f_locals['output'])
                if source['kind']=='allocation':
                    index=len(self.allocations)
                    row=dict(index=index,root=self.roots,source=source,line=pending['line'],after_empty=record)
                    self.allocations.append(row)
                    self.latest[record['storage_identity']]=(index,weakref.ref(frame.f_locals['output']))
                else:
                    row=pending['row'];row['after_copy']=record
                    row['finite_after_copy']=record['finite']==record['elements']
                    row['copy_matches_rhs']=record['bytes_sha256']==row['rhs']['bytes_sha256']
                    self.copies.append(row)
            if event=='line' and frame.f_lineno in source['statements']:
                if source['kind']=='allocation':state['pending']=dict(line=frame.f_lineno)
                else:
                    pre=tensor_record(frame.f_locals['output']);n=frame.f_locals['num_tokens']
                    rhs=tensor_record(frame.f_locals['attn_output'][:n])
                    link=self.latest.get(pre['storage_identity'])
                    origin=self.allocations[link[0]] if link else None
                    before=origin['after_empty'] if origin else None
                    row=dict(root=self.roots,source=source,line=frame.f_lineno,before_copy=pre,rhs=rhs,
                        allocation_index=link[0] if link else None,
                        same_allocation_bytes_before_copy=bool(before and before['bytes_sha256']==pre['bytes_sha256']),
                        same_storage=bool(before and all(before[k]==pre[k] for k in ('storage_identity','data_ptr','storage_bytes','offset'))),
                        full_logical_overwrite=pre['elements']==rhs['elements'])
                    state['pending']=dict(line=frame.f_lineno,row=row)
            if event=='return':self.frames.pop(fid,None)
        except BaseException as error:
            self.error=repr(error);raise
        return self.trace_callback

    def install(self):
        self.trace_callback=self.trace
        self.pre=self.model.register_forward_pre_hook(self.begin,prepend=True)
        self.post=self.model.register_forward_hook(self.end,always_call=True)

    def close(self):
        if self.active:raise RuntimeError('ACTIVE_TRACE_CANNOT_UNLOAD')
        self.pre.remove();self.post.remove()
        return dict(schema='demoncat.attention.allocation_trace.v1',actual_roots=self.roots,
            allocation_count=len(self.allocations),copy_count=len(self.copies),allocations=self.allocations,copies=self.copies,
            error=self.error,model_core_methods_replaced=False,tensor_values_modified=False,
            limitations=['Separate observed run; not historical process pointer evidence',
                'torch.empty does not establish the previous allocator contents or their writer',
                'Readback synchronizes execution and can affect timing/allocation reuse'])
