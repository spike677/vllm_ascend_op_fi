"""Live first-layer TND input binding, without hooks, repacking or admission.

The caller supplies its actual forward context and its observed cache-writer
receipt. This helper checks their live object/storage joins and keeps the
objects until explicit release. It is not a request mapper, Native owner, C1
proof, or permission to select a tile. No serialized report can recreate it.
"""
import copy
import os
import threading

from .paged33_arch22_bridge import validate_normal_call
from .paged33_audit import audit_paged_call, _flat_alias, _indices, _meta
from .paged33_boundary_probe import (FIRST_LAYER, tensor_snapshot, logical_cache_snapshot,
                                     scalar_snapshot)
from .serving_context import require


def _thread():
    return os.getpid(), threading.get_ident(), threading.get_native_id()


def _value_signature(snapshot):
    # Different requests may allocate a new query/mask/table object. Each one
    # is independently bound before/after its real call, including storage.
    return dict(metadata={k:snapshot['metadata'][k] for k in
        ('shape','stride','storage_offset','dtype','device')},
        bytes=snapshot['bytes'],sha256=snapshot['sha256'])


class LogicalPagedInput:
    """One natural call; retains failures and actual objects until release()."""
    def __init__(self, *, worker, module, module_path, forward_context, metadata,
                 actual, scheduler, writer, owner, generation, allow_cpu_test=False, target_scope=None):
        require(owner is not None and not isinstance(owner,(str,int,float,bool,dict,list,tuple))
                and type(generation) is int and generation>0, 'PAGED_LIVE_OWNER_REQUIRED')
        if target_scope is None:
            require(module_path == FIRST_LAYER, 'PAGED_FIRST_LAYER_SCOPE_ONLY')
        else:
            from .paged33_normal_targets import NormalTargetScope
            require(type(target_scope) is NormalTargetScope, 'PAGED_TRUSTED_NORMAL_TARGET_SCOPE_REQUIRED')
            target_scope.check(worker, module_path, module, owner, generation)
        self.target_scope = target_scope
        self.owner,self.generation,self.thread = owner,generation,_thread()
        self.worker,self.runner = worker,worker.model_runner
        self.model,self.module,self.impl = self.runner.model,module,module.impl
        self.module_path,self.context,self.metadata = module_path,forward_context,metadata
        self.actual,self.scheduler,self.writer = actual,scheduler,writer
        self.cpu = allow_cpu_test
        self.tensors = {k:v for k,v in actual.items() if hasattr(v,'untyped_storage')}
        self.caches = (self.impl.key_cache,self.impl.value_cache)
        self.table,self.slots = metadata.block_tables,metadata.slot_mapping
        self.pool = forward_context.attn_metadata
        self.virtual_engine = forward_context.virtual_engine
        self.cache_container = module.kv_cache[self.virtual_engine]
        self.request = copy.deepcopy(scheduler)
        self.writer_before = copy.deepcopy(writer)
        self.calls,self.inflight,self.complete,self.closed = 0,False,False,False
        self.first_error,self._sealed,self.result = None,None,None
        self.identity=dict(owner_object_id=id(owner),generation=generation,worker_object_id=id(worker),
            model_object_id=id(self.model),module_object_id=id(module),impl_object_id=id(self.impl),
            pid=self.thread[0],python_thread_ident=self.thread[1],native_tid=self.thread[2])
        self.before = self._snapshot(owner,generation)

    def _identity(self,owner,generation):
        if self.target_scope is not None:
            self.target_scope.check(self.worker,self.module_path,self.module,owner,generation)
        require(not self.closed and owner is self.owner and type(generation) is int
                and generation == self.generation and _thread() == self.thread,
                'PAGED_OWNER_GENERATION_OR_THREAD_DIFFERS')
        require(self.worker.model_runner is self.runner and self.runner.model is self.model
                and self.model.get_submodule(self.module_path) is self.module
                and self.module.impl is self.impl
                and self.impl.key_cache is self.caches[0] and self.impl.value_cache is self.caches[1],
                'PAGED_WORKER_MODEL_OR_LIVE_CACHE_REPLACED')
        require(self.context.attn_metadata is self.pool
                and self.pool[self.module.layer_name] is self.metadata
                and self.context.no_compile_layers[self.module.layer_name] is self.module
                and self.context.virtual_engine == self.virtual_engine
                and self.module.kv_cache[self.virtual_engine] is self.cache_container,
                'PAGED_ACTUAL_FORWARD_CONTEXT_REPLACED')
        require(self.metadata.block_tables is self.table and self.metadata.slot_mapping is self.slots
                and all(self.actual.get(k) is v for k,v in self.tensors.items())
                and self.scheduler == self.request and self.writer == self.writer_before,
                'PAGED_ARGUMENT_REQUEST_OR_WRITER_REPLACED')
        require(self.writer.get('module_path') == self.module_path,
                'PAGED_WRITER_FROM_DIFFERENT_MODULE')
        require(len(self.cache_container) == 2, 'PAGED_LIVE_CACHE_PAIR_REQUIRED')
        _flat_alias(self.caches[0],self.cache_container[0])
        _flat_alias(self.caches[1],self.cache_container[1])

    def _snapshot(self,owner,generation):
        self._identity(owner,generation)
        boundary = audit_paged_call('npu_fused_infer_attention_score',self.actual,
            impl=self.impl,metadata=self.metadata,scheduler=self.scheduler,writer=self.writer,
            allow_cpu_test=self.cpu)
        validate_normal_call(self.actual,boundary,allow_cpu_test=self.cpu)
        require(self.scheduler['request_count'] == 1
                and len(self.scheduler['request_ids']) == 1
                and type(self.scheduler['request_ids'][0]) is str and self.scheduler['request_ids'][0],
                'PAGED_ONE_ACTUAL_REQUEST_REQUIRED')
        table_values = _indices(self.actual['block_table'],'complete_block_table')
        require(all(type(page) is int and 0<=page<self.caches[0].shape[0] for page in table_values[0]),
                'PAGED_UNUSED_OR_USED_PAGE_OUT_OF_BOUNDS')
        snapshots = {name:tensor_snapshot(self.actual[name]) for name in ('query','atten_mask','block_table')}
        require(all(value['finite'] is True for value in snapshots.values()),'PAGED_INPUT_NONFINITE')
        length = boundary['sequence_lengths'][0]
        logical = {name:logical_cache_snapshot(cache,boundary['used_pages'],length)
                   for name,cache in zip(('key','value'),self.caches)}
        require(all(value['finite'] is True for value in logical.values()),'PAGED_LOGICAL_KV_NONFINITE')
        scalars = {name:scalar_snapshot(value) for name,value in self.actual.items() if name not in self.tensors}
        signature = dict(supplied_arguments=sorted(self.actual),scalars=scalars,
            tensors={name:_value_signature(value) for name,value in snapshots.items()},
            cache_geometry={name:{key:_meta(cache)[key] for key in
                ('shape','stride','storage_offset','dtype','device','storage_nbytes')}
                for name,cache in zip(('key','value'),self.caches)},
            block_table_values=table_values,writer_consumer_slot=list(boundary['slots']),
            logical_kv={name:{key:value[key] for key in ('shape','bytes','sha256')}
                        for name,value in logical.items()})
        # Recheck identity after readbacks; snapshots include object/storage,
        # versions when present, and complete valid logical K/V byte hashes.
        self._identity(owner,generation)
        return dict(signature=signature,boundary=boundary,tensors=snapshots,logical_kv=logical,
                    argument_metadata={name:dict(_meta(value),object_id=id(value)) for name,value in self.tensors.items()})

    def execute(self,invoke_once,*,owner,generation):
        """Use only around the already occurring API call, with its original closure."""
        self._identity(owner,generation)
        require(self.calls == 0 and not self.inflight and self.first_error is None,
                'PAGED_ORIGINAL_CALL_ALREADY_ATTEMPTED')
        self.inflight = True
        try:
            require(self._snapshot(owner,generation) == self.before,'PAGED_INPUT_CHANGED_BEFORE_ORIGINAL')
            self.calls = 1  # Before entering the original, including uncertain return.
            result = invoke_once()
            self.result = result
            after = self._snapshot(owner,generation)
            require(after == self.before,'PAGED_INPUT_CHANGED_DURING_ORIGINAL')
            import torch
            require(type(result) is tuple and len(result) == 2
                    and all(isinstance(value,torch.Tensor) for value in result)
                    and list(result[0].shape) == [1,16,128]
                    and str(result[0].dtype) == 'torch.float16'
                    and list(result[1].shape) == [0] and str(result[1].dtype) == 'torch.float32'
                    and all(value.device == self.actual['query'].device for value in result),
                    'PAGED_ACTUAL_RETURN_CONTAINER_OR_GEOMETRY_DIFFERS')
            outputs = [tensor_snapshot(value) for value in result]
            require(all(value['finite'] is True for value in outputs),'PAGED_OUTPUT_NONFINITE')
            self._sealed = dict(before=self.before,after=after,outputs=outputs,request_id=self.request['request_ids'][0])
            self.complete = True
            return result
        except BaseException as error:
            if self.first_error is None:self.first_error=repr(error)
            raise
        finally:
            self.inflight = False

    def report(self):
        return copy.deepcopy(dict(schema='demoncat.paged33.live_logical_input.v1',
            actual_original_calls=self.calls,complete=self.complete,inflight=self.inflight,
            closed=self.closed,first_error=self.first_error,source_bound=True,
            full_input_arrays_preserved=False,operator_input_repacked=False,
            simulation=self.cpu,qualification_granted=False,identity=self.identity,evidence=self._sealed))

    def release(self,*,owner,generation):
        require(owner is self.owner and type(generation) is int and generation==self.generation
                and _thread()==self.thread and not self.inflight,'PAGED_RELEASE_OWNER_OR_INFLIGHT')
        # This object owns references only. It must never remove another
        # owner's hooks, wrapper, Native scope, cache or forward context.
        if not self.closed:
            self.worker=self.runner=self.model=self.module=self.impl=self.context=self.metadata=None
            self.actual=self.scheduler=self.writer=self.pool=self.cache_container=self.result=None
            self.tensors={};self.caches=();self.table=self.slots=None
            self.closed=True
        return dict(released=True,first_error=self.first_error,owned_hooks=0,owned_native_scopes=0)


def compare_logical_inputs(previous,current,*,owner,generation):
    """Compare two completed live bindings; never normalize a serialized flag."""
    require(type(previous) is LogicalPagedInput and type(current) is LogicalPagedInput,
            'PAGED_ACTUAL_BINDING_OBJECTS_REQUIRED')
    require(type(generation) is int and generation>0,'PAGED_ACTUAL_GENERATION_REQUIRED')
    for item in (previous,current):
        require(not item.closed and item.complete and not item.inflight and item.first_error is None
                and item.owner is owner and item.generation==generation and _thread()==item.thread,
                'PAGED_INCOMPLETE_OR_FOREIGN_BINDING')
    require(previous is not current and previous.thread==current.thread
            and all(getattr(previous,key) is getattr(current,key) for key in
                    ('worker','runner','model','module','impl'))
            and all(a is b for a,b in zip(previous.caches,current.caches))
            and previous._sealed['request_id'] != current._sealed['request_id'],
            'PAGED_SAME_WORKER_CACHE_TWO_REQUESTS_REQUIRED')
    old,new = (item._sealed['before']['signature'] for item in (previous,current))
    old_used,new_used = (item._sealed['before']['boundary']['used_pages'][0] for item in (previous,current))
    require(len(old_used)==len(new_used),'PAGED_LOGICAL_LENGTH_DIFFERS')
    count=len(old_used)
    require(old['block_table_values'][0][:count]==old_used and new['block_table_values'][0][:count]==new_used,
            'PAGED_USED_PAGE_TABLE_JOIN_DIFFERS')
    normalized=[]
    for signature in (old,new):
        value=copy.deepcopy(signature)
        value['block_table_values'][0][:count]=list(range(count))
        value['tensors']['block_table']['sha256']='PROVEN_USED_PAGE_RELOCATION_ONLY'
        value['writer_consumer_slot']=['PROVEN_CURRENT_LOGICAL_POSITION_SLOT']
        normalized.append(value)
    require(normalized[0] == normalized[1],'PAGED_LOGICAL_INPUT_DIFFERS_BEYOND_PAGE_RELOCATION')
    return dict(raw_input_exact=old==new,logical_input_equivalent=True,
        used_page_count=count,previous_used_pages=list(old_used),current_used_pages=list(new_used),
        raw_table_sha256=[value['tensors']['block_table']['sha256'] for value in (old,new)],
        writer_consumer_slots=[value['writer_consumer_slot'] for value in (old,new)],
        allowed_difference_fields=['used_page_ids','full_page_table_sha256','proven_writer_consumer_slot'],
        request_ids=[item._sealed['request_id'] for item in (previous,current)],
        full_logical_kv_hashes_verified=True,full_input_arrays_preserved=False,
        source_bound=True,scope='FIRST_LAYER_TND_Q1_PAGE16_FP16_ONLY',qualification_granted=False)
