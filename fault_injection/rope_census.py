"""Source-bound zero-injection census of the real opaque fused RoPE reader.

It reads actual positions/cache/Q/K and enters a nested dispatch observer only
inside the reviewed Python reader. It neither replaces a model method nor
changes an argument. Cache construction predates installation and is therefore
source-bound, explicitly not claimed as an observed producer invocation.
"""
from collections import Counter
from contextlib import contextmanager, nullcontext
import hashlib
from pathlib import Path
import sys
import threading

from .conv3d_natural import tensor_identity
from .tensor_callable import TensorCallableGuard
from .tensor_observer import callable_identity, metadata, tensors
from .moe_census import describe


ASCEND_SOURCE = '74653da7274de182e8497f7d6b07ea7c6deeeef21e710e34eed5a349b0f671e7'
BASE_SOURCE = '32e2145231012f98c8c613cc37dfe621996a3508af77ee7e0265baefde5c1139'


def same_view(left, right):
    return (left.untyped_storage()._cdata == right.untyped_storage()._cdata
        and left.storage_offset() == right.storage_offset() and left.shape == right.shape
        and left.stride() == right.stride() and left.dtype == right.dtype and left.device == right.device)


class NaturalRoPETrace:
    def __init__(self,model,reader,owners,sources,*,guards=None,simulation=False,
                 diagnostics_context=None,max_events=10000):
        if not simulation and not guards:
            raise RuntimeError('SOURCE_BOUND_ACTUAL_ROPE_GUARDS_REQUIRED')
        self.model,self.reader,self.owners,self.sources=model,reader,owners,sources
        self.reader_code=reader.__code__
        self.guards=guards or {};self.simulation=simulation
        self.diagnostics_context=diagnostics_context;self.max_events=max_events
        self.rows,self.native_rows,self.errors,self.hooks=[],[],[],[]
        self.pending={};self.consumer_pending={};self.returned=[];self.active=False
        self.callback=self._profile;self.request_id=None;self.root_positions=None
        self.expected_positions=None;self.scheduler=None;self.thread=None

    @classmethod
    def from_runtime(cls,model):
        sources={}
        for name,expected in (('vllm_ascend.ops.rotary_embedding',ASCEND_SOURCE),
                ('vllm.model_executor.layers.rotary_embedding.base',BASE_SOURCE)):
            namespace=sys.modules.get(name);path=Path(getattr(namespace,'__file__',''))
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise RuntimeError('UNREVIEWED_ACTUAL_ROPE_SOURCE:'+name)
            sources[name]=dict(path=str(path),sha256=expected)
        namespace=sys.modules['vllm_ascend.ops.rotary_embedding']
        reader=namespace.rope_forward_oot
        if (reader.__globals__ is not vars(namespace) or Path(reader.__code__.co_filename).resolve()!=
                Path(namespace.__file__).resolve() or reader.__closure__):
            raise RuntimeError('ACTUAL_FUSED_ROPE_READER_OWNER_CHANGED')
        owners={};guards={}
        for path,module in model.named_modules():
            if not hasattr(module,'cos_sin_cache'):continue
            path=path or '<root>'
            if type(module) is not namespace.AscendRotaryEmbedding:
                raise RuntimeError('UNADAPTED_STRUCTURAL_ROPE_CACHE_OWNER:'+path)
            guard=TensorCallableGuard(module,extra_methods=('_compute_cos_sin_cache','_compute_inv_freq'))
            if guard._functions.get('_forward_method') is not namespace.AscendRotaryEmbedding.forward_oot:
                raise RuntimeError('ACTUAL_FUSED_ROPE_DISPATCH_NOT_BOUND:'+path)
            owners[path]=module;guards[path]=guard
        if not owners:raise RuntimeError('ACTUAL_FUSED_ROPE_CACHE_OWNER_REQUIRED')
        sources['reader']=dict(function=reader.__module__+'.'+reader.__qualname__,
            first_line=reader.__code__.co_firstlineno,has_triton=namespace.HAS_TRITON)
        return cls(model,reader,owners,sources,guards=guards)

    @contextmanager
    def diagnostics(self):
        from torch.utils._python_dispatch import _disable_current_modes
        with self.diagnostics_context() if self.diagnostics_context else nullcontext():
            # Only observer-owned reads are outside model dispatch/statistics.
            # The original reader/native operator executes after this ends.
            with _disable_current_modes():yield

    def set_request(self,request_id,positions,expected_positions,scheduler):
        if self.request_id is not None or self.pending:
            raise RuntimeError('ROPE_REQUEST_ALREADY_INFLIGHT')
        if not request_id or not isinstance(expected_positions,list):
            raise RuntimeError('ACTUAL_ROPE_SCHEDULER_IDENTITY_REQUIRED')
        self.request_id=request_id;self.root_positions=positions
        self.expected_positions=list(expected_positions);self.scheduler=scheduler
        self.returned.clear()

    def end_request(self):
        if self.pending:raise RuntimeError('ROPE_READER_STILL_INFLIGHT')
        self.request_id=None;self.root_positions=None;self.expected_positions=None;self.scheduler=None
        self.returned.clear()

    def _owner(self,frame):
        while frame is not None:
            owner=frame.f_locals.get('self')
            for path,module in self.owners.items():
                if owner is module:return path,module
            frame=frame.f_back
        raise RuntimeError('ACTUAL_ROPE_MODULE_CALLER_MISSING')

    def _cpu(self,tensor):return tensor.detach().cpu().clone()

    def _open_native(self,row,actual):
        from torch.utils._python_dispatch import TorchDispatchMode
        parent=self
        class Observe(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                if len(parent.native_rows)>=parent.max_events:raise RuntimeError('ROPE_NATIVE_CENSUS_BUDGET')
                schema=getattr(func,'_schema',None);values={arg.name:arg.default_value for arg in schema.arguments}
                values.update(zip((arg.name for arg in schema.arguments),args));values.update(kwargs or {})
                with parent.diagnostics():
                    inputs=tuple(tensors((args,kwargs or {})))
                    native=dict(sequence=len(parent.native_rows),reader_sequence=row['sequence'],
                        request_id=parent.request_id,operator=str(func),schema=str(schema),
                        arguments={key:describe(value) for key,value in values.items()},original_calls=0)
                    if 'rotary' in schema.name:
                        native['reader_argument_join']=dict(
                            positions_exact_object=any(t is actual['positions'] for t in inputs),
                            cache_exact_object=any(t is actual['cos_sin_cache'] for t in inputs),
                            query_actual_storage=any(same_view(t,actual['query']) for t in inputs),
                            key_actual_storage=any(same_view(t,actual['key']) for t in inputs))
                parent.native_rows.append(native)
                result=func(*args,**(kwargs or {}))
                with parent.diagnostics():
                    native.update(original_calls=1,original_returned=True,
                        output=[metadata(t) for t in tensors(result)])
                return result
        mode=Observe();mode.__enter__();return mode

    def _profile(self,frame,event,result):
        if event not in ('call','return') or frame.f_code is not self.reader_code:return
        try:
            if threading.get_ident()!=self.thread or self.request_id is None:
                raise RuntimeError('ROPE_READER_OUTSIDE_REAL_OWNED_REQUEST')
            if self.reader.__code__ is not self.reader_code:
                raise RuntimeError('ROPE_READER_CODE_CHANGED')
            if event=='call':
                if len(self.rows)>=self.max_events:raise RuntimeError('ROPE_READER_CENSUS_BUDGET')
                path,module=self._owner(frame)
                if path in self.guards:self.guards[path].verify()
                actual={key:frame.f_locals[key] for key in
                    ('positions','query','key','cos_sin_cache','head_size','rotary_dim','is_neox_style','offsets')}
                with self.diagnostics():
                    import torch
                    positions=actual['positions'];cache=actual['cos_sin_cache']
                    if (cache is not module.cos_sin_cache or not same_view(positions,self.root_positions)
                        or positions.ndim!=1 or positions.dtype not in (torch.int32,torch.int64)
                        or cache.ndim!=2 or not self.simulation and positions.device.type!='npu'):
                        raise RuntimeError('ACTUAL_ROPE_POSITIONS_CACHE_OWNER_BINDING_MISMATCH')
                    positions_cpu=self._cpu(positions);cache_cpu=self._cpu(cache)
                    values=positions_cpu.tolist()
                    if values!=self.expected_positions:raise RuntimeError('ACTUAL_ROPE_POSITIONS_DIFFER_FROM_SCHEDULER')
                    legal=bool(values) and min(values)>=0 and max(values)<cache.shape[0]
                    distinct=len(values)>1 and values[0]!=values[-1]
                    row=dict(sequence=len(self.rows),request_id=self.request_id,model_path=path,
                        scheduler=self.scheduler,actual_positions=values,positions=metadata(positions),
                        cache=metadata(cache),cache_sha256=tensor_identity(cache_cpu)['sha256'],
                        query_before=tensor_identity(self._cpu(actual['query'])),
                        key_before=tensor_identity(self._cpu(actual['key'])),
                        arguments={key:describe(actual[key]) for key in ('head_size','rotary_dim','is_neox_style','offsets')},
                        legal_positions=legal,distinct_first_last_positions=distinct,
                        returned=False,consumers=[],cache_creation_observed=False)
                    if legal and distinct:
                        row['candidate_rows']={str(index):tensor_identity(cache_cpu[index]) for index in (values[0],values[-1])}
                        row['candidate_rows_different']=not torch.equal(cache_cpu[values[0]],cache_cpu[values[-1]])
                    self.rows.append(row)
                mode=self._open_native(row,actual)
                self.pending[id(frame)]=(row,mode,actual,positions_cpu,cache_cpu)
            elif id(frame) in self.pending:
                from torch.utils._python_dispatch import _get_current_dispatch_mode
                row,mode,actual,positions_cpu,cache_cpu=self.pending[id(frame)]
                if _get_current_dispatch_mode() is not mode:raise RuntimeError('ROPE_NESTED_MODE_OWNERSHIP_CHANGED')
                mode.__exit__(None,None,None);del self.pending[id(frame)]
                with self.diagnostics():
                    import torch
                    output=tuple(tensors(result))
                    row.update(returned=len(output)==2,output=[metadata(t) for t in output],
                        complete_output=[tensor_identity(self._cpu(t)) for t in output],
                        actual_positions_unchanged=torch.equal(self._cpu(actual['positions']),positions_cpu),
                        complete_cache_unchanged=torch.equal(self._cpu(actual['cos_sin_cache']),cache_cpu))
                    if len(output)==2:
                        row['output_aliases_actual_qk']=[same_view(output[0],actual['query']),same_view(output[1],actual['key'])]
                        self.returned.append((row,output))
        except BaseException as exc:
            self.errors.append(repr(exc))

    def _consumer(self,path):
        def pre(module,args,kwargs):
            if not self.active or self.request_id is None:return
            with self.diagnostics():
                inputs=tuple(tensors((args,kwargs)))
                observed=[]
                for row,output in self.returned:
                    # Require both actual returned Q and K views at one real
                    # downstream module; no equality-by-value or guessed layer.
                    if all(any(same_view(t,value) for t in inputs) for value in output):
                        consumer=dict(path=path,actual_query_and_key_views=True,returned=False,
                            input=[metadata(t) for t in inputs],callable=callable_identity(module))
                        row['consumers'].append(consumer);observed.append(consumer)
                self.consumer_pending.setdefault(id(module),[]).append(observed)
        return pre

    def _consumer_return(self,module,args,kwargs,output):
        pending=self.consumer_pending.get(id(module))
        if not pending:return
        observed=pending.pop()
        if not pending:del self.consumer_pending[id(module)]
        with self.diagnostics():
            for row in observed:
                row.update(returned=output is not None,output=[metadata(t) for t in tensors(output)])

    def __enter__(self):
        if self.active or sys.getprofile() is not None:raise RuntimeError('FOREIGN_ROPE_PROFILE_OWNER_OR_ACTIVE_TRACE')
        self.thread=threading.get_ident()
        try:
            for path,module in self.model.named_modules():
                callback=self._consumer(path or '<root>')
                self.hooks.append((module.register_forward_pre_hook(callback,with_kwargs=True),callback))
                callback=self._consumer_return
                self.hooks.append((module.register_forward_hook(callback,with_kwargs=True,always_call=True),callback))
            self.active=True;sys.setprofile(self.callback);return self
        except BaseException:
            for handle,_ in self.hooks:handle.remove()
            self.hooks.clear();raise

    def close(self):
        if not self.active:return
        if threading.get_ident()!=self.thread or sys.getprofile() is not self.callback:
            raise RuntimeError('ROPE_PROFILE_OWNERSHIP_CHANGED')
        if self.pending or self.consumer_pending:raise RuntimeError('ROPE_READER_OR_CONSUMER_STILL_INFLIGHT')
        for handle,callback in self.hooks:
            hooks=handle.hooks_dict_ref()
            if hooks is not None and handle.id in hooks and hooks[handle.id] is not callback:
                raise RuntimeError('ROPE_CONSUMER_HOOK_OWNERSHIP_CHANGED')
        sys.setprofile(None);self.active=False
        for handle,callback in reversed(self.hooks):
            handle.remove()
        self.hooks.clear();self.returned.clear()

    def report(self):
        native=[row for row in self.native_rows if 'rotary' in row['schema']]
        per_request={request:dict(Counter(row['model_path'] for row in self.rows if row['request_id']==request))
                     for request in {row['request_id'] for row in self.rows}}
        structural=bool(per_request) and all(set(counts)==set(self.owners) and
                    all(count==1 for count in counts.values()) for counts in per_request.values())
        joins=all(len([item for item in native if item['reader_sequence']==row['sequence']])==1
                  for row in self.rows) and all(all(item['reader_argument_join'].values()) for item in native)
        complete=(not self.active and not self.pending and not self.consumer_pending and not self.errors
            and bool(self.rows) and structural and joins
            and all(row['returned'] and row['actual_positions_unchanged'] and row['complete_cache_unchanged']
                    and any(item['returned'] for item in row['consumers']) for row in self.rows)
            and all(row.get('original_returned') and row['original_calls']==1 for row in self.native_rows))
        return dict(complete=bool(complete),simulation=self.simulation,sources=self.sources,
            actual_dispatch={path:guard.report() for path,guard in self.guards.items()},
            structural_cache_owners=list(self.owners),readers=self.rows,native_operators=self.native_rows,
            structural_coverage_complete=bool(structural),native_reader_joins_complete=bool(joins),
            structural_calls_per_request=per_request,
            observed_readers=len(self.rows),observed_native_rotary_calls=len(native),
            readers_per_request=dict(Counter(row['request_id'] for row in self.rows)),errors=self.errors,
            mutations=0,tensor_values_read=True,operator_reexecution=False,model_methods_replaced=False,
            qualification_granted=False,cache_creation_observed=False,
            proposed_variant='LEGAL_POSITIONS_INPUT_WRONG_ROW_NOT_UINT32_OVERFLOW',
            variant_status='NOT_IMPLEMENTED_OR_QUALIFIED_BY_THIS_ZERO_INJECTION_CENSUS')

    def __exit__(self,*exc):self.close();return False
