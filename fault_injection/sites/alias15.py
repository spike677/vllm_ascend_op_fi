"""ND output-view alias detachment at an actual impl writer. No private-format claim."""
import hashlib,inspect,threading
from pathlib import Path
import torch

def require(ok,why):
    if not ok:raise RuntimeError(why)

def storage(t):return (str(t.device),t.untyped_storage()._cdata)
def sha(t):return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

class Alias15Site:
    def __init__(self,model,runner,*,fault,region_leases):
        from .gqa28 import SOURCE_SHA
        import torch_npu
        self.model=model;self.runner=runner;self.fault=fault;self.region_leases=region_leases
        self.format_reader=torch_npu.get_npu_format;self.source_sha=SOURCE_SHA
        self.complete=lambda:region_leases.complete(region_leases.stream())
        self.methods=[];self.hooks=[];self.events=[];self.live={};self.active=None;self.uncovered=[]
        self.tid=threading.get_ident();self.sites=[];self.original_empty=None;self.original_mul=None

    @property
    def pending(self):return len(self.live)+self.region_leases.count()

    def install(self):
        modules=dict(self.model.named_modules())
        for name,module in modules.items():
            impl=getattr(module,'impl',None)
            if impl is None or type(impl).__name__!='AscendAttentionBackendImpl':continue
            parent=modules.get(name.rsplit('.',1)[0]);consumer=getattr(parent,'o_proj',None)
            if consumer is None:continue
            actual=hashlib.sha256(Path(inspect.getsourcefile(type(impl))).read_bytes()).hexdigest()
            require(actual==self.source_sha,'ALIAS_WRITER_SOURCE_UNREVIEWED')
            require('forward' not in vars(module) and 'forward' not in vars(impl),'ALIAS_BOUNDARY_ALREADY_WRAPPED')
            original=module.forward;writer=impl.forward;signature=inspect.signature(writer)
            def outer(*args,_name=name,_original=original,**kwargs):
                require(self.active is None and _name not in self.live,'ALIAS_OUTPUT_OVERLAP')
                state=dict(site=_name,allocations=[],writer=None);self.active=state
                try:
                    result=_original(*args,**kwargs)
                    event=state['writer']
                    require(event is not None,'ALIAS_WRITER_NOT_OBSERVED')
                    require(storage(result)==storage(state['base']),'ALIAS_RETURN_BASE_STORAGE_CHANGED')
                    event['returned_view_same_storage']=True
                    self.live[_name]=state
                    return result
                finally:self.active=None
            def write(*args,_name=name,_writer=writer,_signature=signature,**kwargs):
                require(self.active is not None and self.active['site']==_name,'ALIAS_WRITER_OUTSIDE_MODULE')
                bound=_signature.bind(*args,**kwargs);view=bound.arguments['output']
                candidates=[t for t in self.active['allocations'] if t is not view and storage(t)==storage(view)]
                require(len(candidates)==1,'ALIAS_OUTPUT_BASE_ALLOCATION_NOT_UNIQUE')
                base=candidates[0]
                require(base.is_contiguous() and view.is_contiguous() and base.numel()==view.numel() and base.storage_offset()==view.storage_offset()==0,'ALIAS_OUTPUT_EXACT_INTERVAL_REQUIRED')
                self.complete();before=sha(base)
                event=dict(site=_name,base_shape=list(base.shape),view_shape=list(view.shape),dtype=str(base.dtype),
                           format=self.format_reader(base),base_is_view_object=base is view,
                           before_sha256=before,writer='AscendAttentionBackendImpl.forward(output=actual_view)',
                           writer_source_sha256=self.source_sha,consumer=type(consumer).__module__+'.'+type(consumer).__name__,
                           initial_all_finite=bool(torch.isfinite(base).all().item()),injection_enabled=self.fault,
                           request_ids=list(self.runner.input_batch.req_ids),eligible=True,applied=False,changed=0,changed_elements=0,consumed=False,restored=False,failed=False)
                require(event['format']==2 and base.dtype==torch.bfloat16,'ALIAS15_ND_BF16_REQUIRED')
                require(len(event['request_ids'])==1,'ALIAS15_SINGLE_REQUEST_REQUIRED')
                if self.fault:
                    token=self.region_leases.acquire(base)
                    temporary=view.detach().clone()
                    require(storage(temporary)!=storage(base) and sha(temporary)==before,'ALIAS15_INITIAL_VALUE_OR_STORAGE')
                    bound.arguments['output']=temporary
                    self.active.update(lease=token,temporary=temporary)
                    result=_writer(*bound.args,**bound.kwargs);self.complete()
                    require(storage(result)==storage(temporary),'ALIAS15_WRITER_RETURN_NOT_TEMPORARY')
                    require(sha(base)==before,'ALIAS15_ORIGINAL_BASE_WAS_WRITTEN')
                    # The actual writer supplies the intended value; never recompute
                    # or create a synthetic copy_ path for the model.
                    count=int((base.view(-1).view(torch.uint8).reshape(-1,base.element_size())!=temporary.reshape(-1).view(torch.uint8).reshape(-1,base.element_size())).any(dim=1).sum().item())
                    event.update(applied=True,changed=int(count>0),changed_elements=count,
                        intended_sha256=sha(temporary),detached_storage=True,initial_view_exact=True)
                else:
                    result=_writer(*args,**kwargs);self.complete()
                event.update(after_sha256=sha(base),view_after_sha256=sha(view),finite=bool(torch.isfinite(base).all().item()))
                require(event['after_sha256']==event['view_after_sha256'],'ALIAS_BASE_RETURN_VIEW_CHANGED')
                event['base_changed']=before!=event['after_sha256']
                self.events.append(event);self.active.update(base=base,writer=event)
                return result
            module.forward=outer;impl.forward=write;self.methods.extend([(module,outer),(impl,write)])
            def pre(mod,args,_name=name):
                require(_name in self.live,'ALIAS_OUTPUT_CONSUMER_WITHOUT_WRITER')
                state=self.live[_name];x=args[0];event=state['writer']
                expected=state.get('gate_result',state['base'])
                require(storage(x)==storage(expected) and x.numel()==expected.numel(),'ALIAS_CONSUMER_STORAGE_INTERVAL_MISMATCH')
                self.complete();event['consumer_sha256']=sha(x)
                require(event['consumer_sha256']==event.get('gate_result_sha256',event['after_sha256']),'ALIAS_CONSUMER_VALUE_MISMATCH')
                event['consumer_input_same_storage']=storage(x)==storage(state['base'])
                event['projection_reads_exact_produced_value']=True
            def post(mod,args,out,_name=name):
                state=self.live[_name];self.complete();event=state['writer']
                event.update(device_consumer_completed=True,consumed=True,consumer_operator='gate_mul_then_o_proj' if 'gate_result' in state else 'o_proj')
                if self.fault:
                    self.region_leases.consumed(state['lease']);self.region_leases.release_unmodified(state['lease'])
                    event.update(restored=True,restored_sha256=sha(state['base']),restore_kind='UNCHANGED_BASE_BYTES_TEMPORARY_RELEASED')
                    require(event['before_sha256']==event['restored_sha256'],'ALIAS15_BASE_RECOVERY_MISMATCH')
                del self.live[_name]
            self.hooks.extend([consumer.register_forward_pre_hook(pre),consumer.register_forward_hook(post)])
            self.sites.append(name)
        require(self.sites,'NO_NATURAL_ATTENTION_OUTPUT_SITE')
        self.original_empty=torch.empty
        def empty(*args,**kwargs):
            t=self.original_empty(*args,**kwargs)
            if threading.get_ident()==self.tid and self.active is not None:
                self.active['allocations'].append(t)
            return t
        self.empty_wrapper=empty;torch.empty=empty
        self.original_mul=torch.Tensor.__mul__
        def mul(x,other):
            matches=[]
            if threading.get_ident()==self.tid and isinstance(x,torch.Tensor):
                matches=[s for s in self.live.values() if storage(x)==storage(s['base'])]
            if not matches:return self.original_mul(x,other)
            require(len(matches)==1,'ALIAS_GATE_BASE_AMBIGUOUS')
            state=matches[0];event=state['writer'];self.complete()
            require(sha(x)==event['after_sha256'],'ALIAS_GATE_INPUT_CHANGED')
            result=self.original_mul(x,other);self.complete()
            event.update(direct_consumer='torch.Tensor.__mul__',direct_consumer_same_storage=True,
                         gate_input_sha256=sha(x),gate_result_sha256=sha(result),gate_device_read_completed=True)
            state['gate_result']=result
            return result
        self.mul_wrapper=mul;torch.Tensor.__mul__=mul
        return list(self.sites)

    def close(self):
        require(not self.pending and self.active is None,'ALIAS_OUTPUT_CONSUMER_PENDING')
        if self.original_empty is not None:
            require(torch.empty is self.empty_wrapper,'ALIAS_EMPTY_OWNER_CHANGED');torch.empty=self.original_empty
        if self.original_mul is not None:
            require(torch.Tensor.__mul__ is self.mul_wrapper,'ALIAS_MUL_OWNER_CHANGED');torch.Tensor.__mul__=self.original_mul
        for obj,wrapper in reversed(self.methods):
            require(obj.forward is wrapper,'ALIAS_METHOD_OWNER_CHANGED');del obj.forward
        for h in self.hooks:h.remove()
        self.methods.clear();self.hooks.clear()
