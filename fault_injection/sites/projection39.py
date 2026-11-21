"""Continuous BF16 projection return proxy with a real dispatch consumer.

One last-row element per actual projection return, on every rank. A tensor
view is not a value consumer. A consumer that writes the protected element is
refused, rather than restoring over a legitimate in-place update.
"""
import torch
from torch.utils._python_dispatch import TorchDispatchMode,_disable_current_modes
from .mla28 import require,fingerprint
from fault_injection.engines.region_lease import exact


def tensors(value):
    if isinstance(value,torch.Tensor):yield value
    elif isinstance(value,(tuple,list)):
        for item in value:yield from tensors(item)
    elif isinstance(value,dict):
        for item in value.values():yield from tensors(item)


def element_index(t,region):
    """Exact membership for nonoverlapping strided layouts; no data read."""
    if t.device!=region.device or t.untyped_storage()._cdata!=region.untyped_storage()._cdata:return None
    require(t.dtype==region.dtype,'PROJECTION_CONSUMER_DTYPE_REINTERPRETATION')
    offset=region.storage_offset()-t.storage_offset()
    if offset<0 or t.numel()==0:return None
    dimensions=sorted(((stride,size,dim) for dim,(size,stride) in enumerate(zip(t.shape,t.stride())) if size>1),reverse=True)
    span=1
    for stride,size,dim in reversed(dimensions):
        require(stride>=span,'PROJECTION_CONSUMER_OVERLAPPING_LAYOUT')
        span+=(size-1)*stride
    coordinates=[0]*t.ndim
    for stride,size,dim in dimensions:
        index,offset=divmod(offset,stride)
        if index>=size:return None
        coordinates[dim]=index
    return tuple(coordinates) if offset==0 else None


def contains_element(t,region):return element_index(t,region) is not None


class ProjectionConsumerMode(TorchDispatchMode):
    def __init__(self,leases):
        super().__init__();self.leases=leases;self.pending=[]

    def watch(self,region,event,*,inject=True):
        if not inject:
            self.pending.append(dict(region=region,token=None,event=event));return
        with _disable_current_modes():
            token=self.leases.acquire(region)
            event.update(before_sha256=fingerprint(region),already_positive_inf=bool(torch.isposinf(region).all().item()),
                         upstream_nonfinite=bool((~torch.isfinite(region)).any().item()))
            self.leases.write(token,torch.full_like(region,torch.inf))
            event.update(applied=True,changed=int(not event['already_positive_inf']),after_sha256=fingerprint(region))
        self.pending.append(dict(region=region,token=token,event=event))

    def complete_consumers(self,matched,operator):
        for lease in matched:
            e=lease['event'];e.update(consumed=True,consumer_operator=operator)
            if lease['token'] is not None:
                self.leases.consumed(lease['token'])
                self.leases.restore(lease['token'])
                e.update(restored=True,restored_sha256=fingerprint(lease['region']))
                require(e['before_sha256']==e['restored_sha256'],'PROJECTION_RESTORE_BYTES_DIFFER')
            self.pending=[item for item in self.pending if item is not lease]

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {};schema=func._schema
        # Alias-producing operations are views, never consumption evidence.
        is_view=any(r.alias_info is not None for r in schema.returns) and not schema.is_mutable
        metadata_only=schema.name in ('aten::empty_like','aten::zeros_like','aten::ones_like',
            'aten::full_like','aten::new_empty','aten::new_empty_strided','aten::new_zeros',
            'aten::new_ones','aten::new_full','aten::is_pinned')
        matched=[]
        for lease in self.pending:
            reads=False
            for index,argument in enumerate(schema.arguments):
                value=args[index] if index<len(args) else kwargs.get(argument.name)
                overlap=any(contains_element(t,lease['region']) for t in tensors(value))
                if overlap:
                    require(argument.alias_info is None or not argument.alias_info.is_write,
                            'PROJECTION_CONSUMER_WRITES_PROTECTED_REGION')
                    reads=True
            if reads:
                trace=lease['event'].setdefault('observed_read_ops',[])
                if schema.name not in trace and len(trace)<20:trace.append(schema.name)
            if reads and not is_view and not metadata_only:matched.append(lease)
        result=func(*args,**kwargs)
        # NPU exposes aten::contiguous with alias-return schema even when it
        # allocates/copies. A distinct storage AND exact coordinate copy are
        # required; a no-op contiguous remains a view, not consumption proof.
        if schema.name=='aten::contiguous' and isinstance(result,torch.Tensor):
            source=args[0]
            if result.untyped_storage()._cdata!=source.untyped_storage()._cdata:
                require(result.shape==source.shape and result.dtype==source.dtype,'CONTIGUOUS_COPY_LAYOUT_CHANGED')
                for lease in self.pending:
                    index=element_index(source,lease['region'])
                    if index is not None:
                        copied=result[tuple(slice(i,i+1) for i in index)].reshape(1)
                        require(exact(copied,lease['region']),'CONTIGUOUS_REAL_COPY_NOT_PROVEN')
                        if not any(item is lease for item in matched):matched.append(lease)
        self.complete_consumers(matched,schema.name)
        return result


class Projection39Site:
    def __init__(self,model,runner,*,fault,region_leases):
        self.model=model;self.runner=runner;self.fault=fault;self.region_leases=region_leases
        self.events=[];self.sites=[];self.hooks=[];self.uncovered=[];self.active=False
        self.excluded=[];self.forward_calls=[];self.patches=[]
        self.mode=ProjectionConsumerMode(region_leases)

    @property
    def pending(self):return len(self.mode.pending)

    def install(self):
        from vllm.model_executor.layers.linear import LinearBase
        import torch_npu
        def enter(*unused):
            require(not self.active and not self.pending,'PROJECTION_MODE_OVERLAP')
            self.forward_calls.append({})
            self.mode.__enter__();self.active=True
        def leave(*unused):
            require(self.active and not self.pending,'PROJECTION_RETURN_WITHOUT_REAL_CONSUMER:'+str([
                dict(site=p['event'].get('site'),shape=p['event'].get('shape'),
                     observed_read_ops=p['event'].get('observed_read_ops')) for p in self.mode.pending]))
            self.mode.__exit__(None,None,None);self.active=False
        self.hooks.extend([self.model.register_forward_pre_hook(enter),self.model.register_forward_hook(leave)])
        visual=getattr(self.model,'visual',None)
        visual_ids={id(m) for m in visual.modules()} if visual is not None else set()
        for name,module in self.model.named_modules():
            if not isinstance(module,LinearBase):continue
            if id(module) in visual_ids:
                self.excluded.append(dict(site=name,reason='VISUAL_BRANCH_FORBIDDEN_BY_PURE_TEXT_OBSERVER'))
                continue
            self.sites.append(name)
            def output(mod,args,result,_name=name):
                require(self.active,'PROJECTION_OUTSIDE_REAL_MODEL_FORWARD')
                calls=self.forward_calls[-1];calls[_name]=calls.get(_name,0)+1
                t=result[0] if isinstance(result,tuple) else result
                require(isinstance(t,torch.Tensor) and t.ndim==2 and t.is_contiguous(),'PROJECTION_OUTPUT_LAYOUT_CHANGED')
                with _disable_current_modes():
                    eligible=t.dtype==torch.bfloat16 and torch_npu.get_npu_format(t)==2
                    event=dict(site=_name,request_ids=list(self.runner.input_batch.req_ids),observed=True,
                        eligible=eligible,applied=False,changed=0,consumed=False,restored=False,failed=False,
                        reason=None if eligible else 'OUTPUT_NOT_BF16_ND',shape=list(t.shape),dtype=str(t.dtype))
                    self.events.append(event)
                    if eligible:self.mode.watch(t[-1:,:1].reshape(1),event,inject=self.fault)
            self.hooks.append(module.register_forward_hook(output))
        require(self.sites,'NO_LINEAR_PROJECTION_SITES')
        return self.sites

    def close(self):
        require(not self.pending and self.region_leases.count()==0,'PROJECTION_REGION_LEASES_REMAIN')
        if self.active:self.mode.__exit__(None,None,None);self.active=False
        for obj,name,original,wrapped in reversed(self.patches):
            require(getattr(obj,name) is wrapped,'PROJECTION_CONSUMER_OWNER_CHANGED')
            setattr(obj,name,original)
        self.patches.clear()
        for hook in self.hooks:hook.remove()
        self.hooks.clear()
