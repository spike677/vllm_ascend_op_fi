"""Read-only, source-bound B8 prefill + first-decode mapping. No fault authority."""
from copy import deepcopy
from .v026_text_observer import TextPrefillTrace,SourceBoundTextObserver,_prefix
from .serving_context import require,cpu_list
from .serving import complete_tensor

class FirstDecodeTrace(TextPrefillTrace):
    def __init__(self,runner,expected):
        super().__init__(runner,expected)
        require(len(expected)==8 and not self.embedding_branch,'B8_REVIEWED_TOKEN_BRANCH_REQUIRED')
        self.rows=[];self.sampled={};self.input_values=None;self.output_hashes=None

    def before_preprocess(self,scheduler,total,intermediate):
        if not self.rows:return super().before_preprocess(scheduler,total,intermediate)
        self.guard()
        require(len(self.rows)==1 and self.events==['execute_enter'] and scheduler is self.scheduler
                and intermediate is None,'B8_FIRST_DECODE_ORDER')
        ids=self.runner.input_batch.req_ids
        require(len(ids)==8 and set(ids)==set(self.expected)
                and scheduler.num_scheduled_tokens==dict.fromkeys(self.expected,1)
                and total==scheduler.total_num_scheduled_tokens==8
                and scheduler.scheduled_new_reqs==[]
                and set(scheduler.scheduled_cached_reqs.req_ids)==set(ids)
                and not scheduler.scheduled_encoder_inputs and not scheduler.scheduled_spec_decode_tokens,
                'B8_COMPLETE_CACHED_SINGLE_TOKEN_DECODE_REQUIRED')
        tokens=[];positions=[]
        for index,rid in enumerate(ids):
            req=self.runner.requests[rid];prompt=self.expected[rid]
            require(self.runner.input_batch.req_id_to_index[rid]==index
                    and int(self.runner.input_batch.num_computed_tokens_cpu[index])==len(prompt)
                    and req.req_id==rid and req.prompt_token_ids==prompt
                    and req.output_token_ids==[self.sampled[rid]]
                    and req.mm_features==[] and req.prompt_embeds is None and req.lora_request is None,
                    'B8_DECODE_PREFIX_OR_ROW_MAPPING_CHANGED')
            tokens.append(self.sampled[rid]);positions.append(len(prompt))
        self.mapping=dict(request_ids=list(ids),input_ids=tokens,row_token_positions=positions,
            prefixes=[self.expected[rid]+[self.sampled[rid]] for rid in ids],single_chunk_prefill_proven=True)
        self.events.append('preprocess_enter')

    def after_preprocess(self,result):
        if not self.rows:return super().after_preprocess(result)
        self.guard();require(len(result)==6,'B8_DECODE_PREPROCESS_ARITY')
        ids,embeds,positions,intermediate,kwargs,ec=result
        require(ids is not None and embeds is None and intermediate is None and kwargs=={} and ec is None,
                'B8_DECODE_EXTRA_INPUTS')
        _prefix(ids,self.runner.input_ids.gpu,[8]);_prefix(positions,self.runner.mrope_positions.gpu,[3,8])
        require(cpu_list(self.runner.input_ids.cpu)[:8]==self.mapping['input_ids']
                and [r[:8] for r in cpu_list(self.runner.mrope_positions.cpu)]==[self.mapping['row_token_positions']]*3,
                'B8_DECODE_CPU_MIRROR_CHANGED')
        self.preprocessed=result;self.events.append('preprocess_return')

    def model_enter(self,module,args,kwargs):
        super().model_enter(module,args,kwargs)
        ids=self.runner.input_batch.req_ids
        tokens=([t for rid in ids for t in self.expected[rid]] if not self.rows else self.mapping['input_ids'])
        positions=([p for rid in ids for p in range(len(self.expected[rid]))] if not self.rows else self.mapping['row_token_positions'])
        actual,ih,finite=complete_tensor(kwargs['input_ids']);pos,ph,pfinite=complete_tensor(kwargs['positions'])
        require(finite and pfinite and actual.tolist()==tokens and pos.tolist()==[positions]*3,'B8_DEVICE_INPUT_BYTES_CHANGED')
        self.input_values=dict(request_ids=list(ids),input_ids=tokens,positions=positions,input_sha256=ih,positions_sha256=ph)

    def execute_return(self,result):
        super().execute_return(result)
        hashes={}
        for name in ('hidden_states','sample_hidden_states','logits'):
            _,sha,finite=complete_tensor(getattr(self.state,name));require(finite,'B8_NONFINITE_OUTPUT')
            hashes[name]=sha
        self.output_hashes=hashes

    def after_sample(self,result):
        super().after_sample(result)
        generated=result.sampled_token_ids
        require(len(generated)==8 and all(len(t)==1 and type(t[0]) is int for t in generated),'B8_ONE_ACTUAL_SAMPLE_PER_REQUEST')
        ids=list(result.req_ids)
        if not self.rows:
            self.sampled={rid:generated[i][0] for i,rid in enumerate(ids)}
        self.rows.append(dict(phase='prefill' if not self.rows else 'first_decode',
            input=deepcopy(self.input_values),output=deepcopy(self.output_hashes),
            generated=dict(zip(ids,deepcopy(generated))),trace=super().snapshot()))
        if len(self.rows)==1:
            self.events=[];self.finished=False;self.embedding_calls=self.model_calls=0
            self.mapping=self.preprocessed=self.state=self.scheduler=None
        else:require(len(self.rows)==2,'B8_EXTRA_DECODE_FORBIDDEN')

    def snapshot(self):
        result=super().snapshot()
        return dict(result,schema='demoncat.v026.b8_first_decode_trace.v1',rows=deepcopy(self.rows),
            status='OBSERVED' if len(self.rows)==2 and self.finished and self.first_error is None else 'INCOMPLETE_OR_FAILED',
            fault_calls=0,qualification_granted=False)

class SourceBoundGenerationObserver(SourceBoundTextObserver):
    def __init__(self,runner,expected_prompts,*,source_profile='v026_installed_c9c9432b'):
        require(source_profile=='v026_installed_c9c9432b','B8_ONLY_REVIEWED_026_PROFILE')
        super().__init__(runner,expected_prompts,source_profile=source_profile)
        self.trace=FirstDecodeTrace(runner,expected_prompts)
