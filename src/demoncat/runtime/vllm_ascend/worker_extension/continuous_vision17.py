"""Source-bound Qwen3.8 image requests; shares the existing task lifecycle."""
import json
from copy import deepcopy
from .serving_context import require
from .continuous_binding import ContinuousRequest,digest

VISION_SOURCES={
 'vllm/model_executor/models/qwen3_5.py':'da738a2017741b8df66cbdb2ce5301c5cdec87ffacb2279f85245640eb33c496',
 'vllm/model_executor/models/qwen3_vl.py':'4f31566372fca55b708cffcf24229bd4481d5e9f4a77994ee3a71ba29f85936d',
 'vllm/model_executor/layers/conv.py':'d6264c05c69aeffb2451d39f5f1b09924dde1b006cf4674f0e7060e35c9ad971',
 'vllm_ascend/ops/conv.py':'a68d3a61e13880cb94988c29cf2702b19dc08c09f3fa89aeb1c82da7f0af9365',
 'vllm/v1/worker/gpu_model_runner.py':'81b7627fbe81f7aaa2f77b4bf085faa353c69d03662ebfe369536a9773bb70d0',
}


def validate_visual_model(runner):
    import torch
    from .v026_text_observer import PROFILES,_verify_function,_parallel_report
    model=runner.model;config=runner.vllm_config;mm=config.model_config.multimodal_config
    require((type(model).__module__,type(model).__name__)==('vllm.model_executor.models.qwen3_5','Qwen3_5ForConditionalGeneration'),
            'VISION17_EXACT_MODEL_REQUIRED')
    require((type(runner).__module__,type(runner).__name__)==('vllm_ascend.worker.model_runner_v1','NPUModelRunner'),
            'VISION17_EXACT_RUNNER_REQUIRED')
    require(str(runner.device).split(':')[0]=='npu' and not model.training and runner.use_async_scheduling is False
        and runner.speculative_config is None and config.model_config.enforce_eager is True
        and config.cache_config.enable_prefix_caching is False and config.kv_transfer_config is None
        and runner.lora_config is None,'VISION17_SERIAL_FRESH_EAGER_REQUIRED')
    require(mm is not None and mm.get_limit_per_prompt('image')==1 and mm.get_limit_per_prompt('video')==0
        and mm.mm_processor_cache_gb==0 and mm.mm_encoder_tp_mode=='weights'
        and model.use_data_parallel is False and runner.supports_mm_inputs is True
        and runner.enable_prompt_embeds is False and runner.uses_mrope is True,
        'VISION17_ACTUAL_UNCACHED_IMAGE_BRANCH_REQUIRED')
    rows=json.loads(PROFILES['v026_installed_c9c9432b'].read_bytes())['files']
    rows=rows+[dict(path=k,sha256=v) for k,v in VISION_SOURCES.items()]
    for f in (runner.execute_model,runner._preprocess,runner.sample_tokens,runner._execute_mm_encoder,
              runner._batch_mm_inputs_from_scheduler,model.forward,model.embed_multimodal,
              model.embed_input_ids,model.visual.forward,model.visual.patch_embed.forward,
              model.visual.patch_embed.proj._forward_method,model.visual.patch_embed.proj._forward_conv):
        _verify_function(f,rows)
    _parallel_report(runner,rows,'v026_installed_c9c9432b')
    conv=model.visual.patch_embed.proj
    require(tuple(conv.weight.shape[2:])==(2,16,16) and conv.weight.dtype==torch.bfloat16
        and tuple(conv.padding)==(0,0,0),'VISION17_LIVE_BF16_PATCH_REQUIRED')
    require(config.scheduler_config.enable_chunked_prefill is True and config.scheduler_config.max_num_seqs==1,
            'VISION17_SINGLE_IMAGE_SCHEDULING_REQUIRED')
    return 'qwen35_visual'


class ContinuousVisionRequest(ContinuousRequest):
    def install(self):
        import torch
        from fault_injection.sites.mla28 import fingerprint
        from fault_injection.sites.vision17 import Vision17Site
        self.binding.check();require(len(self.ids)==1,'VISION17_ONE_ACTUAL_REQUEST')
        self.encoder_calls=0;self.encoder_output=None;self.embedding_calls=0;self.in_encoder=False
        self.feature=None;self.current_stream=None
        def patch(obj,name,fn):
            require(name not in vars(obj),'VISION17_METHOD_ALREADY_OWNED:'+name)
            original=getattr(obj,name)
            def wrapper(*args,**kwargs):
                self.inside+=1
                try:return fn(original,*args,**kwargs)
                except BaseException as e:self.error=repr(e);raise
                finally:self.inside-=1
            setattr(obj,name,wrapper);self.patches.append((obj,name,wrapper))
        def execute(original,scheduler,intermediate_tensors=None):
            if scheduler.total_num_scheduled_tokens==0:
                require(not scheduler.num_scheduled_tokens and not scheduler.scheduled_encoder_inputs,'VISION17_BAD_CLEANUP')
                return original(scheduler,intermediate_tensors)
            require(self.samples<4 and intermediate_tensors is None and self.runner.execute_model_state is None,'VISION17_EXECUTE_ORDER')
            self.scheduler=scheduler;self.model_output=None
            result=original(scheduler,intermediate_tensors);state=self.runner.execute_model_state
            require(result is None and state is not None and state.scheduler_output is scheduler
                and state.hidden_states is self.model_output and state.logits.shape[0]==1,'VISION17_MODEL_RETURN_ASSOCIATION')
            self.rows.append(dict(hidden=fingerprint(state.hidden_states),logits=fingerprint(state.logits),
                finite=bool(torch.isfinite(state.hidden_states).all().item() and torch.isfinite(state.logits).all().item())))
            return result
        def preprocess(original,scheduler,total,intermediate_tensors=None):
            rid=self.ids[0];prompt=self.prompts[rid];req=self.runner.requests[rid]
            require(scheduler is self.scheduler and intermediate_tensors is None and self.runner.input_batch.req_ids==[rid]
                and self.runner.input_batch.req_id_to_index=={rid:0} and req.req_id==rid
                and req.prompt_token_ids==list(prompt) and req.prompt_embeds is None and req.lora_request is None,
                'VISION17_ACTUAL_REQUEST_MAPPING_CHANGED')
            expected=len(prompt) if self.samples==0 else 1
            require(total==scheduler.total_num_scheduled_tokens==expected and scheduler.num_scheduled_tokens=={rid:expected}
                and not scheduler.scheduled_spec_decode_tokens
                and int(self.runner.input_batch.num_computed_tokens_cpu[0])==(0 if self.samples==0 else len(prompt)+self.samples-1),
                'VISION17_SINGLE_CHUNK_OR_DECODE_MAPPING')
            if self.samples==0:
                require(len(req.mm_features)==1 and scheduler.scheduled_encoder_inputs=={rid:[0]},'VISION17_REAL_ENCODER_SCHEDULE_REQUIRED')
                feature=req.mm_features[0];position=feature.mm_position
                require(feature.modality=='image' and feature.data is not None and feature.identifier not in self.runner.encoder_cache
                    and 0<=position.offset<position.offset+position.length<=len(prompt),'VISION17_UNCACHED_IMAGE_FEATURE_REQUIRED')
                self.feature=feature
            else:require(not scheduler.scheduled_encoder_inputs and self.encoder_calls==1,'VISION17_UNEXPECTED_DECODE_ENCODER')
            value=original(scheduler,total,intermediate_tensors)
            require(len(value)==6 and value[3] is None and value[4]=={} and value[5] is None,'VISION17_PREPROCESS_EXTRA_INPUT')
            self.preprocessed=value;return value
        def encoder(original,scheduler):
            require(not self.in_encoder and scheduler is self.scheduler,'VISION17_ENCODER_ORDER')
            if not scheduler.scheduled_encoder_inputs:
                require(self.samples>0 and self.encoder_calls==1
                    and self.runner._batch_mm_inputs_from_scheduler(scheduler)==([],[],[]),
                    'VISION17_EMPTY_ENCODER_ONLY_AFTER_PREFILL')
                count=self.encoder_calls;result=original(scheduler)
                require(result==[] and self.encoder_calls==count,'VISION17_EMPTY_ENCODER_CALLED_VISUAL')
                return result
            require(self.samples==0,'VISION17_EXTRA_REAL_ENCODER_IN_DECODE')
            hashes,kwargs,refs=self.runner._batch_mm_inputs_from_scheduler(scheduler)
            require(hashes==[self.feature.identifier] and len(kwargs)==1 and kwargs[0][0]=='image'
                and kwargs[0][1] is self.feature.data and len(refs)==1 and refs[0][0]==self.ids[0]
                and refs[0][1] is self.feature.mm_position,'VISION17_ENCODER_REQUEST_JOIN')
            self.in_encoder=True
            try:result=original(scheduler)
            finally:self.in_encoder=False
            cached=self.runner.encoder_cache[self.feature.identifier]
            require(self.encoder_output is not None and cached.untyped_storage()._cdata==self.encoder_output.untyped_storage()._cdata,
                    'VISION17_ENCODER_CACHE_NOT_REAL_OUTPUT')
            return result
        def visual_enter(module,args,kwargs):
            require(self.in_encoder and self.encoder_calls==0 and module is self.model.visual,'VISION17_VISUAL_CALL_OUTSIDE_REQUEST')
            require(len(args)==1 and isinstance(args[0],torch.Tensor),'VISION17_REAL_PIXELS_REQUIRED')
            self.encoder_calls+=1
        def visual_output(module,args,value):self.encoder_output=value
        def embedding(original,input_ids,*args,**kwargs):
            if self.samples==0:
                embeds=kwargs.get('multimodal_embeddings');require(embeds is not None and len(embeds)==1,'VISION17_ONE_REAL_ENCODER_EMBEDDING')
                require(self.encoder_output is not None and embeds[0].untyped_storage()._cdata==self.encoder_output.untyped_storage()._cdata,
                        'VISION17_EMBEDDING_CONSUMER_STORAGE_CHANGED')
                self.embedding_calls+=1
            return original(input_ids,*args,**kwargs)
        def model_enter(module,args,kwargs):
            require(not args and self.preprocessed is not None and self.encoder_calls==1 and self.embedding_calls>=1,'VISION17_MODEL_BEFORE_ENCODER')
            expected=dict(zip(('input_ids','inputs_embeds','positions','intermediate_tensors'),self.preprocessed[:4]))
            require(set(kwargs)==set(expected) and all(kwargs[k] is v for k,v in expected.items()),'VISION17_LIVE_MODEL_INPUTS_CHANGED')
        def model_output(module,args,value):self.model_output=value
        def sample(original,*args,**kwargs):
            require(len(self.rows)==self.samples+1,'VISION17_SAMPLE_ORDER')
            result=original(*args,**kwargs);require(set(result.req_ids)==set(self.ids) and self.runner.execute_model_state is None,'VISION17_SAMPLE_IDS_CHANGED')
            self.samples+=1;return result
        def complete(stream):
            event=torch.npu.Event();event.record(stream);event.synchronize()
        try:
            patch(self.runner,'execute_model',execute);patch(self.runner,'_preprocess',preprocess)
            patch(self.runner,'_execute_mm_encoder',encoder);patch(self.runner,'sample_tokens',sample)
            patch(self.model,'embed_input_ids',embedding)
            self.hooks.extend([self.model.visual.register_forward_pre_hook(visual_enter,with_kwargs=True),
                self.model.visual.register_forward_hook(visual_output),self.model.register_forward_pre_hook(model_enter,with_kwargs=True),
                self.model.register_forward_hook(model_output)])
            self.site=Vision17Site(self.model.visual,fault=self.phase=='fault',references=self.binding.visual_references,
                request_ids=lambda:self.runner.input_batch.req_ids,stream=torch.npu.current_stream,complete=complete)
            self.site.install()
        except BaseException:self.close();raise

    def collect(self):
        require(self.inside==0 and self.samples==4 and self.error is None and self.encoder_calls==1 and self.embedding_calls>=1,
                'VISION17_FOUR_TOKEN_IMAGE_REQUEST_INCOMPLETE')
        events=self.site.events
        require(events and len(self.site.forward_calls)==1 and set(self.site.forward_calls[0])==set(self.site.sites),
                'VISION17_ACTUAL_PATCH_CALLS_INCOMPLETE')
        require(all(e['request_ids']==self.ids and e['consumed'] and e['restored'] and not e['failed']
                    and e['before_sha256']==e['restored_sha256'] for e in events),'VISION17_CONSUMER_OR_ARGUMENT_RECOVERY_FAILED')
        counts=dict(observed=len(events),eligible=0,attempted=0,applied=0,changed=0,consumed=0,restored=0,failed=0)
        if self.phase=='fault':
            counts.update(eligible=len(events),attempted=len(events),applied=sum(e['applied'] for e in events),
                changed=sum(e['changed'] for e in events),consumed=len(events),restored=len(events))
            require(counts['applied']==len(events) and counts['changed']>0,'VISION17_NO_ACTUAL_CHANGED_CALL')
        result=dict(phase=self.phase,request_ids=self.ids,output=deepcopy(self.rows),coverage=counts,
            all_finite=all(r['finite'] for r in self.rows),site_count=len(self.site.sites),events_sha256=digest(events),
            same_stream_verified=True,bytes_restored=True,restore_kind='CALL_ARGUMENT_ONLY_NO_STORAGE_OVERWRITE',
            observation_complete=True,uncovered_sites=[],inapplicable_reasons=[],
            vision_effect=dict(encoder_calls=self.encoder_calls,embedding_consumers=self.embedding_calls,
                changed_outputs=sum(e['changed'] for e in events),consumer_operators=sorted({e['consumer_operator'] for e in events}),
                actual_padding=[e['actual_padding'] for e in events]))
        self.close();result['resources']=dict(leases=0,hooks=0,inflight=0);return result
