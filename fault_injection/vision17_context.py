"""Actual single-image Worker request identity at the encoder's live call.

This source-bound adapter reads an existing scheduled call; it does not replace
execute_model, construct request IDs, switch Conv branches or invoke inference.
"""
import hashlib
import inspect
from pathlib import Path
import sys
import threading

from .vision17_trace import require, vision17_diagnostics
from .conv3d_natural import tensor_identity
from .rope_sites import ModuleSites


SOURCES={
    'vllm.model_executor.models.qwen3_vl':'5fffbdd4cc1515ddca8c47d295cfbad7707a73ab65026ec476ca941ae4b02b6b',
    'vllm.v1.worker.gpu_model_runner':'e164efbb988bc23ddff83842d27930a8cc904104d3c74d28c8c9369535b08f01',
    'vllm_ascend.worker.model_runner_v1':'8d35f26a9db8f63038158c139a43f92e3e970117bb4a2b43c82fd828e349b981',
    'vllm.model_executor.layers.conv':'5151c6947141a39eab2f314aedb384db65a40c40785b7c69309a78a58b713a37',
    'vllm_ascend.ops.conv':'cc288c542410ea50bf9f112d25db2493f54a2c522f27bdba3f9afc137740e295',
}


class MethodPin:
    def __init__(self,owner,name,*,simulation=False):
        self.owner,self.name,self.simulation=owner,name,simulation
        method=getattr(owner,name);self.function=getattr(method,'__func__',None)
        require(self.function is not None and getattr(method,'__self__',None) is owner,'ACTUAL_BOUND_METHOD_REQUIRED:'+name)
        self.code=self.function.__code__;self.path=Path(self.code.co_filename).resolve()
        self.digest=hashlib.sha256(self.path.read_bytes()).hexdigest()
        if not simulation:
            require(SOURCES.get(self.function.__module__)==self.digest
                and self.function.__globals__ is vars(sys.modules[self.function.__module__]),'UNREVIEWED_ACTUAL_METHOD_SOURCE:'+name)
    def verify(self):
        actual=getattr(self.owner,self.name)
        require(getattr(actual,'__self__',None) is self.owner and getattr(actual,'__func__',None) is self.function
                and self.function.__code__ is self.code
                and hashlib.sha256(self.path.read_bytes()).hexdigest()==self.digest,'ACTUAL_METHOD_CHANGED:'+self.name)
    def report(self):
        return dict(method=self.name,module=self.function.__module__,qualname=self.function.__qualname__,
                    source_sha256=self.digest,first_line=self.code.co_firstlineno)


def _frame(pin):
    current=inspect.currentframe().f_back
    try:
        while current is not None:
            if current.f_code is pin.code and current.f_locals.get('self') is pin.owner:
                return current
            current=current.f_back
        raise RuntimeError('VISION17_ACTUAL_EXECUTING_FRAME_REQUIRED:'+pin.name)
    finally:
        del current


def _cpu(tensor):
    if tensor.device.type=='npu':
        import torch
        torch.npu.synchronize()
    return tensor.detach().cpu().clone()


class Vision17RequestContext:
    def __init__(self,worker,*,simulation=False):
        require(type(simulation) is bool,'EXPLICIT_SIMULATION_FLAG_REQUIRED')
        self.worker,self.runner,self.model=worker,worker.model_runner,worker.model_runner.model
        self.visual=self.model.visual;self.simulation=simulation;self.thread=threading.get_ident()
        self.visual_sites=ModuleSites(self.visual)
        self.pins={name:MethodPin(owner,name,simulation=simulation) for owner,name in (
            (self.runner,'_execute_mm_encoder'),(self.runner,'_batch_mm_inputs_from_scheduler'),
            (self.model,'embed_multimodal'),(self.model,'_process_image_input'))}
        self.held=[];self.rows=[]
        if not simulation:self.verify_runtime()

    def verify_runtime(self):
        require(self.worker.model_runner is self.runner and self.runner.model is self.model
                and self.model.visual is self.visual and threading.get_ident()==self.thread,'WORKER_MODEL_LIFETIME_CHANGED')
        for pin in self.pins.values():pin.verify()
        for path,module in self.visual_sites.nodes.items():self.visual_sites.verify(path,module)
        if self.simulation:return
        from .tensor_callable import TensorCallableGuard
        import torch
        config=self.runner.vllm_config;parallel=config.parallel_config
        require(all(getattr(parallel,k)==1 for k in ('tensor_parallel_size','pipeline_parallel_size','data_parallel_size'))
                and self.runner.pcp_size==self.runner.dcp_size==1 and self.runner.use_cp is False
                and parallel.enable_dbo is False and self.runner.speculative_config is None
                and self.runner.use_async_scheduling is False and config.scheduler_config.max_num_seqs==1
                and config.scheduler_config.enable_chunked_prefill is False
                and config.cache_config.enable_prefix_caching is False
                and config.model_config.enforce_eager is True and config.model_config.is_encoder_decoder is False
                and self.runner.is_multimodal_model is True and self.runner.lora_config is None
                and self.runner.is_multimodal_pruning_enabled is False and self.model.use_data_parallel is False,
                'REVIEWED_SINGLE_IMAGE_EAGER_SCOPE_REQUIRED')
        conv_owners=[module for module in self.visual.modules() if isinstance(getattr(module,'weight',None),torch.Tensor)
                     and module.weight.ndim==5]
        require(conv_owners,'NO_STRUCTURAL_CONV3D_CANDIDATES')
        for module in conv_owners:
            pin=MethodPin(module,'_forward_method')
            require(pin.function.__qualname__=='AscendConv3dLayer.forward_oot'
                    and pin.function.__module__=='vllm_ascend.ops.conv','ACTUAL_ASCEND_CONV_BRANCH_REQUIRED')
            implementation=MethodPin(module,'_forward_conv')
            require(implementation.function.__qualname__=='Conv3dLayer._forward_conv'
                    and implementation.function.__module__=='vllm.model_executor.layers.conv',
                    'ACTUAL_CONV3D_IMPLEMENTATION_REQUIRED')
            TensorCallableGuard(module).verify()

    def snapshot(self,actual_visual,args,kwargs,*,expected_request_id,expected_prompt):
        import torch
        self.verify_runtime()
        require(actual_visual is self.visual and len(args)==1 and set(kwargs)=={'grid_thw'},'ACTUAL_VISUAL_CALL_REQUIRED')
        executing={name:_frame(self.pins[name]) for name in ('_execute_mm_encoder','embed_multimodal','_process_image_input')}
        try:
            call=executing['_execute_mm_encoder'].f_locals;image_call=executing['_process_image_input'].f_locals
            embed=executing['embed_multimodal'].f_locals;scheduler=call['scheduler_output']
            require(call.get('model') is self.model and call.get('modality')=='image'
                    and call.get('num_items')==1 and call.get('current_item_idx')==0,'ACTUAL_SINGLE_IMAGE_GROUP_REQUIRED')
            request_id=expected_request_id
            require(isinstance(request_id,str) and request_id and set(self.runner.requests)=={request_id}
                    and self.runner.input_batch.req_ids==[request_id]
                    and self.runner.input_batch.req_id_to_index=={request_id:0},'ACTUAL_REQUEST_SET_MISMATCH')
            request=self.runner.requests[request_id]
            require(request.req_id==request_id and request.num_computed_tokens==0 and request.output_token_ids==[]
                    and request.prompt_embeds is None and request.lora_request is None
                    and request.prompt_token_ids==list(expected_prompt),'ACTUAL_FRESH_MM_PROMPT_REQUIRED')
            require(scheduler.scheduled_encoder_inputs=={request_id:[0]}
                    and scheduler.num_scheduled_tokens=={request_id:len(expected_prompt)}
                    and scheduler.total_num_scheduled_tokens==len(expected_prompt)
                    and not scheduler.scheduled_spec_decode_tokens,'ACTUAL_ENCODER_SCHEDULE_MISMATCH')
            require(len(request.mm_features)==1,'ONE_ACTUAL_IMAGE_FEATURE_REQUIRED')
            feature=request.mm_features[0];position=feature.mm_position
            require(feature.modality=='image' and feature.data is not None and isinstance(feature.identifier,str)
                    and feature.identifier and type(position.offset) is int and type(position.length) is int
                    and 0<=position.offset<position.offset+position.length<=len(expected_prompt)
                    and position.is_embed is None,'UNCACHED_IMAGE_FEATURE_AND_PLACEHOLDER_REQUIRED')
            require(call.get('mm_hashes')==[feature.identifier]
                    and len(call.get('mm_kwargs',[]))==1 and call['mm_kwargs'][0][0]=='image'
                    and call['mm_kwargs'][0][1] is feature.data
                    and len(call.get('mm_lora_refs',[]))==1 and call['mm_lora_refs'][0][0]==request_id
                    and call['mm_lora_refs'][0][1] is position,'ACTUAL_MM_ITEM_REQUEST_MAPPING_REQUIRED')
            batched=call['mm_kwargs_batch'];image_input=image_call['image_input']
            require(set(batched)=={'pixel_values','image_grid_thw'} and image_input['type']=='pixel_values'
                    and embed['kwargs']['pixel_values'] is batched['pixel_values']
                    and embed['kwargs']['image_grid_thw'] is batched['image_grid_thw']
                    and image_input['pixel_values'] is batched['pixel_values']
                    and image_input['image_grid_thw'] is batched['image_grid_thw']
                    and image_call['pixel_values'] is args[0] and image_call['grid_thw'] is kwargs['grid_thw'],
                    'ACTUAL_BATCH_PARSE_VISUAL_ARGUMENT_JOIN_FAILED')
            with vision17_diagnostics():
                pixels=_cpu(args[0]);grid=_cpu(kwargs['grid_thw'])
                original_pixels=feature.data['pixel_values'].data
                original_grid=feature.data['image_grid_thw'].data
                require(original_pixels.device.type=='cpu' and original_grid.device.type=='cpu',
                        'ACTUAL_PROCESSED_CPU_FEATURE_REQUIRED')
                require(torch.equal(pixels,original_pixels.to(dtype=pixels.dtype))
                        and torch.equal(grid,original_grid.reshape(1,3).to(dtype=grid.dtype)),
                        'ACTUAL_PROCESSED_IMAGE_BYTES_CHANGED')
                require(grid.shape==(1,3) and pixels.ndim==2 and int(grid.prod())==pixels.shape[0]
                        and pixels.shape[0]//self.visual.spatial_merge_size**2==position.length,
                        'ACTUAL_PATCH_PLACEHOLDER_COUNT_MISMATCH')
                row=dict(request_id=request_id,request_object_id=id(request),feature_object_id=id(feature),
                    encoder_input_index=0,feature_identifier=feature.identifier,placeholder=dict(offset=position.offset,length=position.length),
                    scheduler_object_id=id(scheduler),actual_visual_object_id=id(actual_visual),
                    mm_kwargs_item_object_id=id(feature.data),actual_batched_object_id=id(batched),
                    pixel_values=tensor_identity(pixels),image_grid_thw=tensor_identity(grid),
                    prompt_token_ids=list(expected_prompt),actual_frame_sources={k:p.report() for k,p in self.pins.items()},
                    single_image_request_join_proven=True,standard_worker_qualified=False,simulation=self.simulation)
            self.held.extend((request,feature,position,feature.data,batched,args[0],kwargs['grid_thw']))
            self.rows.append(row)
            return row
        finally:
            executing.clear()

    def close(self):
        self.held.clear()
