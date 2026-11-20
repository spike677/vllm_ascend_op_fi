"""Source-bound 0.26 pure-text execution observation. Never grants a fault permit.

Observe an already scheduled fresh prefill through execute -> sample. The model,
its embeddings, logits and scheduler objects are returned unmodified. No worker
or request is created. New-stack ordinary bindings remain separately gated.
"""
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading
from .serving_context import require, cpu_list, tensor_metadata
from .v026_inspection import PROFILE, inspect_single_chunk, parallel_assessment
PROFILES = {"v026": PROFILE, "preferred_a2_85a2bc5e": PROFILE.with_name("preferred_a2_source_review.json")}


def _prefix(actual, buffer, shape):
    a,b=tensor_metadata(actual),tensor_metadata(buffer)
    require(a['storage_identity']==b['storage_identity'] and a['storage_offset']==b['storage_offset']
            and a['device']==b['device'] and a['dtype']==b['dtype'] and a['shape']==shape
            and a['stride']==b['stride'], 'V026_ACTUAL_BUFFER_PREFIX_DIFFERS')
    return a


class TextPrefillTrace:
    """Same-call facts, populated only at real boundaries by the observer below."""
    def __init__(self, runner, expected):
        self.runner=runner;self.model=runner.model;self.expected=deepcopy(expected)
        self.pid=os.getpid();self.thread=threading.get_ident()
        self.scheduler=None;self.events=[];self.mapping=None;self.preprocessed=None
        self.visual_calls=0;self.embedding_calls=0;self.model_calls=0
        self.state=None;self.first_error=None;self.finished=False
    def guard(self):
        require((os.getpid(),threading.get_ident())==(self.pid,self.thread)
                and self.runner.model is self.model and not self.finished,
                'V026_TRACE_OWNER_OR_MODEL_CHANGED')
    def enter(self,scheduler):
        self.guard();require(not self.events and self.runner.execute_model_state is None,'V026_EXECUTE_ORDER')
        self.scheduler=scheduler;self.events.append('execute_enter')
    def before_preprocess(self,scheduler,total,intermediate):
        self.guard();require(self.events==['execute_enter'] and scheduler is self.scheduler
                            and intermediate is None,'V026_PREPROCESS_ORDER')
        report=inspect_single_chunk(self.runner,scheduler,self.expected)
        require(report['status']=='matched','V026_SINGLE_CHUNK_NOT_CORROBORATED:'+json.dumps(report))
        require(total==sum(map(len,self.expected.values())),'V026_PADDING_UNSUPPORTED')
        for request_id in self.expected:
            req=self.runner.requests[request_id]
            require(req.req_id==request_id and req.prompt_embeds is None and req.lora_request is None
                    and req.output_token_ids==[],'V026_NON_TEXT_OR_PREVIOUS_GENERATION')
        self.mapping=report;self.events.append('preprocess_enter')
    def embedding(self,input_ids,kwargs):
        self.guard();require(self.events==['execute_enter','preprocess_enter']
                            and self.embedding_calls==0,'V026_EMBEDDING_ORDER')
        require(kwargs.get('multimodal_embeddings') is not None
                and len(kwargs['multimodal_embeddings'])==0,'V026_MULTIMODAL_EMBEDDINGS_PRESENT')
        n=sum(map(len,self.expected.values()))
        _prefix(input_ids,self.runner.input_ids.gpu,[n])
        ordered=[token for request_id in self.runner.input_batch.req_ids for token in self.expected[request_id]]
        require(cpu_list(self.runner.input_ids.cpu)[:n]==ordered,'V026_EMBED_INPUT_CPU_MIRROR_DIFFERS')
        self.embedding_calls+=1
    def after_preprocess(self,result):
        self.guard();require(self.embedding_calls==1 and len(result)==6,'V026_EMBEDDING_NOT_OBSERVED')
        ids,embeds,positions,intermediate,kwargs,ec=result;n=sum(map(len,self.expected.values()))
        require(intermediate is None and kwargs=={} and ec is None,'V026_PREPROCESS_EXTRA_INPUTS')
        # _prepare_mm_inputs may retain input_ids; if present it must be the
        # same source-pinned runner prefix, not arbitrary replacement IDs.
        if ids is not None:_prefix(ids,self.runner.input_ids.gpu,[n])
        _prefix(embeds,self.runner.inputs_embeds.gpu,[n,self.runner.inputs_embeds.gpu.shape[1]])
        _prefix(positions,self.runner.mrope_positions.gpu,[3,n])
        expected_positions=[p for request_id in self.runner.input_batch.req_ids for p in range(len(self.expected[request_id]))]
        require([row[:n] for row in cpu_list(self.runner.mrope_positions.cpu)]==[expected_positions]*3,
                'V026_PURE_TEXT_MROPE_CPU_MIRROR_DIFFERS')
        self.preprocessed=result;self.events.append('preprocess_return')
    def model_enter(self,module,args,kwargs):
        self.guard();require(module is self.model and not args and self.model_calls==0
                            and self.events[-1:] == ['preprocess_return'],'V026_MODEL_ORDER')
        expected=dict(zip(('input_ids','inputs_embeds','positions','intermediate_tensors'),self.preprocessed[:4]))
        require(set(kwargs)==set(expected) and all(kwargs[k] is v for k,v in expected.items()),
                'V026_MODEL_INPUT_NOT_PREPROCESSED_OBJECT')
        self.model_calls+=1;self.events.append('model_enter')
    def model_return(self,module,args,kwargs,output):
        self.guard();require(self.events[-1:] == ['model_enter'] and module is self.model and output is not None,
                            'V026_MODEL_RETURN_ORDER')
        self.events.append('model_return')
    def visual(self,*_args,**_kwargs):
        self.visual_calls+=1
        raise RuntimeError('V026_VISUAL_EXECUTION_FORBIDDEN_IN_TEXT_PROBE')
    def execute_return(self,result):
        self.guard();state=self.runner.execute_model_state
        require(result is None and self.events[-1:]==['model_return'] and state is not None
                and state.scheduler_output is self.scheduler and state.logits is not None
                and state.hidden_states is not None and state.sample_hidden_states is not None,
                'V026_EXECUTE_STATE_NOT_CURRENT_REQUEST')
        require(state.logits.shape[0]==len(self.expected) and self.visual_calls==0,'V026_LOGITS_ROW_OR_VISUAL_MISMATCH')
        self.state=state;self.events.append('execute_return_state_pending')
    def before_sample(self):
        self.guard();require(self.events[-1:]==['execute_return_state_pending']
                            and self.runner.execute_model_state is self.state,'V026_SAMPLE_STATE_REPLACED')
        self.events.append('sample_enter')
    def after_sample(self,result):
        self.guard();require(self.events[-1:]==['sample_enter'] and self.runner.execute_model_state is None,
                            'V026_SAMPLE_STATE_NOT_CONSUMED')
        require(list(result.req_ids)==self.runner.input_batch.req_ids
                and set(result.req_ids)==set(self.expected),'V026_SAMPLE_REQUEST_MAPPING_DIFFERS')
        self.events.append('sample_return');self.finished=True
    def snapshot(self):
        return dict(schema='demoncat.v026.text_prefill_trace.v1',events=list(self.events),
            request_ids=list(self.expected),mapping=deepcopy(self.mapping),visual_calls=self.visual_calls,
            embedding_calls=self.embedding_calls,model_calls=self.model_calls,
            status='OBSERVED' if self.finished and self.first_error is None else 'INCOMPLETE_OR_FAILED',
            visual_execution_excluded=self.finished and self.visual_calls==0 and self.first_error is None,
            model_input_association_verified=self.finished and self.first_error is None,
            association_basis='SOURCE_PINNED_EMBEDDING_COPY_AND_SAME_LIVE_MODEL_ARGUMENT_OBJECTS',
            device_input_values_compared=False,first_error=self.first_error,
            execution_authorized=False,qualification_granted=False)


def _verify_function(function,rows):
    function=inspect.unwrap(getattr(function,'__func__',function))
    path=Path(inspect.getsourcefile(function)).resolve(strict=True)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    require(any(path.as_posix().endswith('/'+row['path']) and digest==row['sha256'] for row in rows),
            'V026_UNREVIEWED_ACTUAL_FUNCTION:'+str(path))
    return dict(path=str(path),sha256=digest)


def _parallel_report(runner, rows):
    from vllm.distributed.parallel_state import get_pcp_group, get_dcp_group, get_tp_group
    sources=[_verify_function(f,rows) for f in (get_pcp_group,get_dcp_group,get_tp_group)]
    config=runner.vllm_config.parallel_config
    report=parallel_assessment(runner,pcp_group=get_pcp_group(),dcp_group=get_dcp_group(),
        tp_group=get_tp_group(),expected_tp=config.tensor_parallel_size)
    require(report['status']=='matched','NEWSTACK_PARALLEL_GROUPS_UNPROVEN:'+json.dumps(report))
    return dict(assessment=report,sources=sources)


class SourceBoundTextObserver:
    """Trusted deployment inspection, not a public-config entry or fault provider."""
    def __init__(self,runner,expected_prompts,*,source_profile="v026"):
        require(source_profile in PROFILES,"NEWSTACK_UNREVIEWED_SOURCE_PROFILE")
        self.source_profile=source_profile;self.parallel=None
        self.runner=runner;self.trace=TextPrefillTrace(runner,expected_prompts)
        self.patches=[];self.hooks=[];self.active=False;self.inside=0;self.sources=[]
    def install(self):
        require(not self.active and not self.patches,'V026_OBSERVER_SINGLE_INSTALL')
        runner=self.runner;model=runner.model
        require(type(runner).__module__=='vllm_ascend.worker.model_runner_v1'
                and type(runner).__name__=='NPUModelRunner'
                and type(model).__module__=='vllm.model_executor.models.qwen3_5'
                and type(model).__name__=='Qwen3_5ForConditionalGeneration', 'V026_EXACT_REVIEWED_RUNNER_AND_MODEL')
        require(str(runner.device).split(':')[0]=='npu','V026_ACTUAL_NPU_RUNNER_REQUIRED')
        require(runner.use_async_scheduling is False and runner.speculative_config is None
                and runner.vllm_config.model_config.enforce_eager is True,'V026_EAGER_SYNC_NON_SPEC_REQUIRED')
        mm=runner.vllm_config.model_config.multimodal_config
        require(mm is not None and mm.get_limit_per_prompt('image')==0 and mm.get_limit_per_prompt('video')==0,
                'V026_TEXT_ONLY_MODALITY_LIMITS_REQUIRED')
        require(runner.supports_mm_inputs is True and runner.enable_prompt_embeds is False
                and runner.uses_mrope is True, 'V026_REVIEWED_MM_TEXT_BRANCH_REQUIRED')
        rows=json.loads(PROFILES[self.source_profile].read_bytes())['files']
        self.parallel=_parallel_report(runner,rows)
        self.sources.append(_verify_function(mm.get_limit_per_prompt,rows))
        if self.source_profile=='preferred_a2_85a2bc5e':
            # The preferred runner overrides _preprocess but delegates the real
            # embedding/position preparation to its GPUModelRunner base.
            parent=super(type(runner),runner)
            self.sources.append(_verify_function(parent._preprocess,rows))
        for function in (runner.execute_model,runner.sample_tokens,runner._preprocess,runner._model_forward,
                         model.forward,model.embed_input_ids):
            self.sources.append(_verify_function(function,rows))
        def patch(obj,name,handler):
            require(name not in vars(obj),'V026_EXISTING_INSTANCE_METHOD_OVERRIDE:'+name)
            original=getattr(obj,name)
            def wrapper(*args,**kwargs):
                self.inside+=1
                try:return handler(original,*args,**kwargs)
                except BaseException as error:
                    self.trace.first_error=self.trace.first_error or repr(error);raise
                finally:self.inside-=1
            setattr(obj,name,wrapper);self.patches.append((obj,name,wrapper))
        def execute(original,scheduler_output,intermediate_tensors=None):
            self.trace.enter(scheduler_output)
            result=original(scheduler_output,intermediate_tensors)
            self.trace.execute_return(result);return result
        def preprocess(original,scheduler_output,num_input_tokens,intermediate_tensors=None):
            self.trace.before_preprocess(scheduler_output,num_input_tokens,intermediate_tensors)
            result=original(scheduler_output,num_input_tokens,intermediate_tensors)
            self.trace.after_preprocess(result);return result
        def embed(original,input_ids,*args,**kwargs):
            require(not args,'V026_EMBED_EXTRA_POSITIONAL_ARGUMENTS')
            self.trace.embedding(input_ids,kwargs);return original(input_ids,**kwargs)
        def sample(original,*args,**kwargs):
            self.trace.before_sample();result=original(*args,**kwargs)
            self.trace.after_sample(result);return result
        try:
            patch(runner,'execute_model',execute);patch(runner,'_preprocess',preprocess)
            patch(model,'embed_input_ids',embed);patch(runner,'sample_tokens',sample)
            self.hooks.append(model.register_forward_pre_hook(self.trace.model_enter,with_kwargs=True))
            self.hooks.append(model.register_forward_hook(self.trace.model_return,with_kwargs=True))
            visual=getattr(model,'visual',None)
            require(visual is not None,'V026_ACTUAL_VISUAL_MODULE_REQUIRED')
            for module in visual.modules():
                self.hooks.append(module.register_forward_pre_hook(self.trace.visual,with_kwargs=True))
            self.active=True;return self
        except BaseException:
            self.close();raise
    def close(self):
        require(self.inside==0,'V026_OBSERVER_STILL_EXECUTING')
        require(all(vars(obj).get(name) is wrapper for obj,name,wrapper in self.patches),
                'V026_OBSERVER_REPLACED_BY_FOREIGN_OWNER')
        for hook in reversed(self.hooks):hook.remove()
        self.hooks.clear()
        for obj,name,_ in reversed(self.patches):delattr(obj,name)
        self.patches.clear();self.active=False
    def snapshot(self):
        return dict(self.trace.snapshot(),source_profile=self.source_profile,parallel=deepcopy(self.parallel),sources=deepcopy(self.sources),owned_hooks=len(self.hooks),
                    owned_wrappers=len(self.patches),inflight=self.inside)
