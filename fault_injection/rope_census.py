"""Source-bound read-only census of the real opaque fused RoPE reader.

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
from .rope_sites import ModuleSites
from .rope_triton_trace import TritonRoPELaunchTrace, TRITON_ROPE_SOURCE, TRITON_JIT_SOURCE


ASCEND_SOURCE = '74653da7274de182e8497f7d6b07ea7c6deeeef21e710e34eed5a349b0f671e7'
BASE_SOURCE = '32e2145231012f98c8c613cc37dfe621996a3508af77ee7e0265baefde5c1139'


def same_view(left, right):
    return (left.untyped_storage()._cdata == right.untyped_storage()._cdata
        and left.storage_offset() == right.storage_offset() and left.shape == right.shape
        and left.stride() == right.stride() and left.dtype == right.dtype and left.device == right.device)


class NaturalRoPETrace:
    def __init__(self,model,reader,owners,sources,*,guards=None,simulation=False,
                 diagnostics_context=None,max_events=10000,outer_operator=None,triton_kernel=None,
                 position_policy='exact'):
        if not simulation and not guards:
            raise RuntimeError('SOURCE_BOUND_ACTUAL_ROPE_GUARDS_REQUIRED')
        self.sites=ModuleSites(model)
        self.model,self.reader,self.owners,self.sources=model,reader,self.sites.owner_sites(owners),sources
        self.reader_code=reader.__code__
        self.guards=guards or {};self.simulation=simulation
        self.diagnostics_context=diagnostics_context;self.max_events=max_events
        self.rows,self.native_rows,self.errors,self.hooks=[],[],[],[]
        self.pending={};self.consumer_pending={};self.returned=[];self.active=False
        self.callback=self._profile;self.request_id=None;self.root_positions=None
        self.expected_positions=None;self.scheduler=None;self.thread=None
        self.stack=[];self.outer_mode=None;self.outer_pending=[];self.outer_rows=[]
        self.root_rows=[];self.request_ids=set()
        self.outer_operator=outer_operator
        self.triton=TritonRoPELaunchTrace(triton_kernel,self) if triton_kernel is not None else None
        self._guard_by_owner={id(owners[path]):guard for path,guard in self.guards.items()}
        if position_policy not in ('exact','exact_or_last_from_first'):
            raise ValueError('Explicit read-only position policy required')
        self.position_policy=position_policy

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
        kernel=None
        if namespace.HAS_TRITON:
            for name,expected in (('vllm_ascend.ops.triton.rope',TRITON_ROPE_SOURCE),
                                  ('triton.runtime.jit',TRITON_JIT_SOURCE)):
                mod=sys.modules.get(name);path=Path(getattr(mod,'__file__',''))
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                    raise RuntimeError('UNREVIEWED_ACTUAL_ROPE_TRITON_SOURCE:'+name)
                sources[name]=dict(path=str(path),sha256=expected)
            mod=sys.modules['vllm_ascend.ops.triton.rope'];kernel=mod._triton_rope
            if (namespace.rope_forward_triton is not mod.rope_forward_triton
                    or kernel.fn.__globals__ is not vars(mod)
                    or kernel.run.__func__ is not sys.modules['triton.runtime.jit'].JITFunction.run):
                raise RuntimeError('ACTUAL_TRITON_ROPE_DISPATCH_OWNER_CHANGED')
        return cls(model,reader,owners,sources,guards=guards,
                   outer_operator='vllm::npu_rotary_embedding',triton_kernel=kernel)

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
        if not request_id or request_id in self.request_ids or not isinstance(expected_positions,list):
            raise RuntimeError('ACTUAL_ROPE_SCHEDULER_IDENTITY_REQUIRED')
        self.request_ids.add(request_id)
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
            for path,module in reversed(self.stack):
                if owner is module and self.owners.get(path) is module:
                    self.sites.verify(path,module)
                    return path,module
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
        try:
            if self.triton is not None:self.triton.profile(frame,event,result)
            if event not in ('call','return') or frame.f_code is not self.reader_code:return
            if threading.get_ident()!=self.thread or self.request_id is None:
                raise RuntimeError('ROPE_READER_OUTSIDE_REAL_OWNED_REQUEST')
            if self.reader.__code__ is not self.reader_code:
                raise RuntimeError('ROPE_READER_CODE_CHANGED')
            if event=='call':
                if len(self.rows)>=self.max_events:raise RuntimeError('ROPE_READER_CENSUS_BUDGET')
                path,module=self._owner(frame)
                if id(module) in self._guard_by_owner:self._guard_by_owner[id(module)].verify()
                actual={key:frame.f_locals[key] for key in
                    ('positions','query','key','cos_sin_cache','head_size','rotary_dim','is_neox_style','offsets')}
                with self.diagnostics():
                    import torch
                    positions=actual['positions'];cache=actual['cos_sin_cache']
                    if (cache is not module.cos_sin_cache
                        or positions.ndim!=1 or positions.dtype not in (torch.int32,torch.int64)
                        or cache.ndim!=2 or not self.simulation and positions.device.type!='npu'):
                        raise RuntimeError('ACTUAL_ROPE_POSITIONS_CACHE_OWNER_BINDING_MISMATCH')
                    positions_cpu=self._cpu(positions);cache_cpu=self._cpu(cache)
                    values=positions_cpu.tolist()
                    root_cpu=self._cpu(self.root_positions)
                    if root_cpu.tolist()!=self.expected_positions:
                        raise RuntimeError('ROOT_ROPE_POSITIONS_DIFFER_FROM_SCHEDULER')
                    expected_wrong=list(self.expected_positions)
                    if expected_wrong:expected_wrong[-1]=expected_wrong[0]
                    exact=values==self.expected_positions and same_view(positions,self.root_positions)
                    wrong=(self.position_policy=='exact_or_last_from_first' and len(values)>1
                        and self.expected_positions[0]!=self.expected_positions[-1] and values==expected_wrong
                        and positions.untyped_storage()._cdata!=self.root_positions.untyped_storage()._cdata)
                    if not exact and not wrong:raise RuntimeError('ACTUAL_ROPE_POSITIONS_CACHE_OWNER_BINDING_MISMATCH')
                    legal=bool(values) and min(values)>=0 and max(values)<cache.shape[0]
                    distinct=len(values)>1 and values[0]!=values[-1]
                    row=dict(sequence=len(self.rows),request_id=self.request_id,model_path=path,
                        module_object_id=id(module),cache_lifetime_object_id=id(cache),
                        active_parent_path=self.stack[-2][0] if len(self.stack)>1 else None,
                        scheduler=self.scheduler,actual_positions=values,scheduler_positions=list(self.expected_positions),
                        positions=metadata(positions),
                        cache=metadata(cache),cache_sha256=tensor_identity(cache_cpu)['sha256'],
                        query_before=tensor_identity(self._cpu(actual['query'])),
                        key_before=tensor_identity(self._cpu(actual['key'])),
                        non_target_qk_before=[tensor_identity(self._cpu(actual[name])[:-1]) for name in ('query','key')],
                        arguments={key:describe(actual[key]) for key in ('head_size','rotary_dim','is_neox_style','offsets')},
                        legal_positions=legal,distinct_first_last_positions=distinct,
                        independent_position_transform='last_from_first' if wrong else 'identity',
                        root_positions_unchanged=False,
                        returned=False,consumers=[],cache_creation_observed=False)
                    candidate_positions=self.expected_positions
                    if legal and len(candidate_positions)>1 and candidate_positions[0]!=candidate_positions[-1]:
                        row['candidate_rows']={str(index):tensor_identity(cache_cpu[index])
                            for index in (candidate_positions[0],candidate_positions[-1])}
                        row['candidate_rows_different']=not torch.equal(cache_cpu[candidate_positions[0]],cache_cpu[candidate_positions[-1]])
                    self.rows.append(row)
                    if self.outer_operator is not None:
                        if len(self.outer_pending)!=1:raise RuntimeError('ACTUAL_OUTER_ROPE_CALL_REQUIRED')
                        outer,values=self.outer_pending[-1]
                        outer['reader_sequences'].append(row['sequence'])
                        outer['reader_argument_join']={name:values[name] is actual[name]
                            for name in ('positions','query','key','cos_sin_cache')}
                        outer['reader_scalar_join']={name:values[name]==actual[name]
                            for name in ('head_size','rotary_dim','is_neox_style')}
                mode=self._open_native(row,actual)
                self.pending[id(frame)]=(row,mode,actual,positions_cpu,cache_cpu,root_cpu)
            elif id(frame) in self.pending:
                from torch.utils._python_dispatch import _get_current_dispatch_mode
                row,mode,actual,positions_cpu,cache_cpu,root_cpu=self.pending[id(frame)]
                if _get_current_dispatch_mode() is not mode:raise RuntimeError('ROPE_NESTED_MODE_OWNERSHIP_CHANGED')
                mode.__exit__(None,None,None);del self.pending[id(frame)]
                with self.diagnostics():
                    import torch
                    output=tuple(tensors(result))
                    row.update(returned=len(output)==2,output=[metadata(t) for t in output],
                        complete_output=[tensor_identity(self._cpu(t)) for t in output],
                        non_target_output=[tensor_identity(self._cpu(t)[:-1]) for t in output],
                        actual_positions_unchanged=torch.equal(self._cpu(actual['positions']),positions_cpu),
                        root_positions_unchanged=torch.equal(self._cpu(self.root_positions),root_cpu),
                        complete_cache_unchanged=torch.equal(self._cpu(actual['cos_sin_cache']),cache_cpu))
                    if len(output)==2:
                        row['output_aliases_actual_qk']=[same_view(output[0],actual['query']),same_view(output[1],actual['key'])]
                        self.returned.append((row,output))
        except BaseException as exc:
            self.errors.append(repr(exc))

    def _consumer(self,path):
        def pre(module,args,kwargs):
            if not self.active or self.request_id is None:return
            if threading.get_ident()!=self.thread:
                self.errors.append('ROPE_MODEL_THREAD_CHANGED');raise RuntimeError(self.errors[-1])
            try:
                if module is self.model and self.stack:raise RuntimeError('ROPE_RECURSIVE_ROOT_REFUSED')
                actual_path=self.sites.resolve(module,self.stack)
            except BaseException as exc:
                self.consumer_pending.setdefault(id(module),[]).append(None)
                self.errors.append(repr(exc));raise
            if actual_path is None:return
            self.stack.append((actual_path,module))
            root_row=None
            if module is self.model:
                root_row=dict(request_id=self.request_id,status='running')
                self.root_rows.append(root_row)
            with self.diagnostics():
                inputs=tuple(tensors((args,kwargs)))
                observed=[]
                for row,output in self.returned:
                    # Require both actual returned Q and K views at one real
                    # downstream module; no equality-by-value or guessed layer.
                    if all(any(same_view(t,value) for t in inputs) for value in output):
                        consumer=dict(path=actual_path,actual_query_and_key_views=True,returned=False,
                            input=[metadata(t) for t in inputs],callable=callable_identity(module))
                        row['consumers'].append(consumer);observed.append(consumer)
                self.consumer_pending.setdefault(id(module),[]).append((observed,sys.exc_info()[1],root_row))
        return pre

    def _consumer_return(self,module,args,kwargs,output):
        if threading.get_ident()!=self.thread:return
        pending=self.consumer_pending.get(id(module))
        if not pending:return
        entry=pending.pop()
        if not pending:del self.consumer_pending[id(module)]
        if entry is None:return
        observed,entering_error,root_row=entry
        if not self.stack or self.stack[-1][1] is not module:raise RuntimeError('ROPE_LIVE_SCOPE_MISMATCH')
        self.stack.pop()
        returned=output is not None and sys.exc_info()[1] is entering_error
        if root_row is not None:root_row['status']='returned' if returned else 'failed'
        with self.diagnostics():
            for row in observed:
                row.update(returned=returned,output=[metadata(t) for t in tensors(output)])

    def _open_outer(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        parent=self
        class Outer(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                schema=getattr(func,'_schema',None)
                if not parent.stack or getattr(schema,'name',None)!=parent.outer_operator:
                    return func(*args,**(kwargs or {}))
                if parent.outer_pending:raise RuntimeError('REENTRANT_OUTER_ROPE_CALL')
                values={arg.name:arg.default_value for arg in schema.arguments}
                values.update(zip((arg.name for arg in schema.arguments),args));values.update(kwargs or {})
                row=dict(sequence=len(parent.outer_rows),request_id=parent.request_id,
                    model_path=parent.stack[-1][0],operator=str(func),schema=str(schema),
                    reader_sequences=[],original_calls=0,original_returned=False)
                parent.outer_rows.append(row);parent.outer_pending.append((row,values))
                try:
                    output=func(*args,**(kwargs or {}))
                    row.update(original_calls=1,original_returned=True)
                    return output
                finally:parent.outer_pending.pop()
        mode=Outer();mode.__enter__();self.outer_mode=mode

    def __enter__(self):
        if self.active or sys.getprofile() is not None:raise RuntimeError('FOREIGN_ROPE_PROFILE_OWNER_OR_ACTIVE_TRACE')
        self.thread=threading.get_ident()
        try:
            for path,module in self.model.named_modules():
                callback=self._consumer(path or '<root>')
                self.hooks.append((module.register_forward_pre_hook(callback,with_kwargs=True),callback))
                callback=self._consumer_return
                self.hooks.append((module.register_forward_hook(callback,with_kwargs=True,always_call=True),callback))
            self.active=True
            if self.outer_operator is not None:self._open_outer()
            sys.setprofile(self.callback);return self
        except BaseException:
            if sys.getprofile() is self.callback:sys.setprofile(None)
            if self.outer_mode is not None:
                self.outer_mode.__exit__(None,None,None);self.outer_mode=None
            self.active=False
            for handle,_ in self.hooks:handle.remove()
            self.hooks.clear();raise

    def close(self):
        if not self.active and not self.hooks and self.outer_mode is None:return
        if threading.get_ident()!=self.thread or self.active and sys.getprofile() is not self.callback:
            raise RuntimeError('ROPE_PROFILE_OWNERSHIP_CHANGED')
        if (self.pending or self.consumer_pending or self.stack or self.outer_pending
                or self.triton and (self.triton.pending or self.triton.launch_frames)):
            raise RuntimeError('ROPE_READER_OR_CONSUMER_STILL_INFLIGHT')
        for handle,callback in self.hooks:
            hooks=handle.hooks_dict_ref()
            if hooks is not None and handle.id in hooks and hooks[handle.id] is not callback:
                raise RuntimeError('ROPE_CONSUMER_HOOK_OWNERSHIP_CHANGED')
        if self.outer_mode is not None:
            from torch.utils._python_dispatch import _get_current_dispatch_mode
            if _get_current_dispatch_mode() is not self.outer_mode:raise RuntimeError('ROPE_OUTER_MODE_OWNERSHIP_CHANGED')
        if self.active:sys.setprofile(None);self.active=False
        if self.outer_mode is not None:
            self.outer_mode.__exit__(None,None,None);self.outer_mode=None
        for handle,callback in reversed(self.hooks[:]):
            handle.remove()
            hooks=handle.hooks_dict_ref()
            if hooks is not None and handle.id in hooks:
                raise RuntimeError('ROPE_CONSUMER_HOOK_REMOVAL_UNCONFIRMED')
            self.hooks.remove((handle,callback))
        self.returned.clear()

    def report(self):
        native=[row for row in self.native_rows if 'rotary' in row['schema']]
        per_request={request:dict(Counter(row['model_path'] for row in self.rows if row['request_id']==request))
                     for request in {row['request_id'] for row in self.rows}}
        structural=bool(per_request) and all(set(counts)==set(self.owners) and
                    all(count==1 for count in counts.values()) for counts in per_request.values())
        nested_joins=all(len([item for item in native if item['reader_sequence']==row['sequence']])==1
                  for row in self.rows) and all(all(item['reader_argument_join'].values()) for item in native)
        triton_rows=self.triton.rows if self.triton else []
        triton_joins=bool(triton_rows) and not self.triton.pending and not self.triton.launch_frames and all(
            len([item for item in triton_rows if item['reader_sequence']==row['sequence']])==1
            for row in self.rows) and all(self.triton.complete(row) for row in triton_rows)
        outer_joins=self.outer_operator is None or (len(self.outer_rows)==len(self.rows) and all(
            row['original_returned'] and row['original_calls']==1 and len(row['reader_sequences'])==1
            and all(row.get('reader_argument_join',{}).values())
            and all(row.get('reader_scalar_join',{}).values()) for row in self.outer_rows))
        joins=(triton_joins if self.triton is not None else nested_joins) and outer_joins
        complete=(not self.active and not self.hooks and self.outer_mode is None
            and not self.pending and not self.consumer_pending and not self.errors
            and bool(self.rows) and structural and joins
            and len(self.root_rows)==len(self.request_ids) and all(row['status']=='returned' for row in self.root_rows)
            and all(row['returned'] and row['actual_positions_unchanged'] and row['root_positions_unchanged'] and row['complete_cache_unchanged']
                    and any(item['returned'] for item in row['consumers']) for row in self.rows)
            and all(row.get('original_returned') and row['original_calls']==1 for row in self.native_rows))
        return dict(complete=bool(complete),simulation=self.simulation,sources=self.sources,
            actual_dispatch={path:guard.report() for path,guard in self.guards.items()},
            structural_cache_owners=list(self.owners),readers=self.rows,native_operators=self.native_rows,
            outer_operators=self.outer_rows,outer_reader_joins_complete=bool(outer_joins),
            triton_launches=triton_rows,triton_reader_joins_complete=bool(triton_joins),
            nested_torch_rotary_joins_complete=bool(nested_joins),
            native_join_evidence_kind='SOURCE_BOUND_TRITON_RUNTIME_CALL' if self.triton else 'TORCH_NATIVE_OPERATOR',
            raw_device_launch_payload_observed=False,
            position_policy=self.position_policy,
            root_forwards=self.root_rows,
            structural_coverage_complete=bool(structural),native_reader_joins_complete=bool(joins),
            structural_calls_per_request=per_request,
            observed_readers=len(self.rows),observed_native_rotary_calls=len(native),
            readers_per_request=dict(Counter(row['request_id'] for row in self.rows)),errors=self.errors,
            mutations=0,tensor_values_read=True,operator_reexecution=False,model_methods_replaced=False,
            qualification_granted=False,cache_creation_observed=False,
            proposed_variant='LEGAL_POSITIONS_INPUT_WRONG_ROW_NOT_UINT32_OVERFLOW',
            variant_status='SEPARATE_INPUT_ADAPTER_NOT_QUALIFIED_BY_THIS_READ_ONLY_OBSERVER')

    def __exit__(self,*exc):self.close();return False
