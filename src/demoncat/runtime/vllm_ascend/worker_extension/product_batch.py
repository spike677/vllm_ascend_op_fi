"""Real scheduler batches on the existing product Runtime and region leases.

No host request callbacks or synthetic request IDs. Per-request coverage counts
participation in shared batch calls; frame IDs allow exact de-duplication.
"""
from copy import deepcopy
from .serving_context import require
from .continuous_binding import ContinuousRequest,digest
from fault_injection.product_contract import COUNTERS,request_range


def actual_forward_coverage(events, calls, candidates):
    """Compare actual apply entries, not a census of dormant module objects."""
    from collections import Counter
    require(calls and len(candidates)==len(set(candidates)) and set(calls)<=set(candidates),
            'PRODUCT_BATCH_ACTUAL_APPLY_CENSUS_INVALID')
    require(Counter(e['site'] for e in events)==Counter(calls),
            'PRODUCT_BATCH_SITE_FORWARD_COVERAGE')
    return dict(actual_apply_calls=dict(Counter(calls)),
        not_called=[dict(site=s,reason='NOT_CALLED_IN_ACTUAL_FORWARD') for s in candidates if s not in calls])


def classify_batch(rows,scheduled):
    """Rows come from the live runner after preprocessing state update."""
    if not rows or set(scheduled)!={r['id'] for r in rows}:return False,'ACTUAL_BATCH_MAPPING_UNPROVEN'
    if any(r['reasons'] for r in rows):return False,'BATCH_CONTAINS_UNSUPPORTED_REQUEST'
    if any(scheduled[r['id']]!=1 or r['computed']<r['prompt_length'] for r in rows):return False,'PREFILL_OR_MIXED_BATCH_NOT_INJECTED'
    if len(rows)<2:return False,'DECODE_BATCH_SIZE_ONE'
    if len(rows)>8:return False,'BATCH_EXCEEDS_VALIDATED_EIGHT'
    return True,None


def summarize_events(events):
    counts=dict.fromkeys(COUNTERS,0);sites={}
    for e in events:
        eligible=bool(e['eligible']);counts['observed']+=1;counts['eligible']+=eligible;counts['attempted']+=eligible
        for k in ('applied','consumed','restored','failed'):counts[k]+=bool(e.get(k))
        counts['changed']+=e.get('changed',0)>0
        if eligible:
            require(e['applied'] and e['consumed'] and e['restored'] and not e['failed']
                and e['before_sha256']==e['restored_sha256'] and e['reference_passed']
                and e['consumer_operator']=='torch_npu.npu_quant_matmul','PRODUCT_BATCH_CALL_PROOF_FAILED')
        row=sites.setdefault(e['site'],dict(site=e['site'],calls=0,eligible=0,consumed=0,restored=0,reference_passed=0,changed_elements=0))
        row['calls']+=1;row['eligible']+=eligible
        for k in ('consumed','restored'):row[k]+=bool(e.get(k))
        row['reference_passed']+=eligible and bool(e.get('reference_passed'))
        row['changed_elements']+=e.get('changed',0)
    return dict(coverage=counts,sites=list(sites.values()),events_sha256=digest(events),
        changed_elements=sum(e.get('changed',0) for e in events),coverage_unit='REQUEST_PARTICIPATION_IN_ACTUAL_BATCH_CALLS')


class AutomaticBatches:
    def __init__(self,binding,*,canary=False):
        self.binding=binding;self.runner=binding.worker.model_runner;self.model=binding.model;self.canary=canary
        self.session=None;self.records=[];self.seen=set();self.states={};self.canary_rows={}
        self.completed_canary=False;self.inside=0;self.error=None;self.patches=[];self.hooks=[]
        self.control_refusal=None
        self.frame=None;self.frame_index=0;self.model_output=None;self.preprocessed=None
        self.request=ContinuousRequest(binding,[],'normal' if canary else 'fault');self.site=None

    def install(self):
        self.binding.check()
        require(self.runner.vllm_config.scheduler_config.max_num_seqs==8,'PRODUCT_BATCH_MAX_NUM_SEQS_EIGHT_REQUIRED')
        def patch(name,handler):
            require(name not in vars(self.runner),'PRODUCT_BATCH_BOUNDARY_OWNED:'+name)
            original=getattr(self.runner,name)
            def wrapped(*args,**kwargs):
                self.inside+=1
                try:return handler(original,*args,**kwargs)
                except BaseException as e:self.error=repr(e);raise
                finally:self.inside-=1
            setattr(self.runner,name,wrapped);self.patches.append((name,wrapped))
        self.request.install_site();self.site=self.request.site;self.site.call_policy=self.policy
        patch('execute_model',self.execute);patch('_preprocess',self.preprocess);patch('sample_tokens',self.sample)
        def enter(module,args,kwargs):
            require(self.preprocessed is not None and not args,'PRODUCT_BATCH_FORWARD_ORDER')
            expected=dict(zip(('input_ids','inputs_embeds','positions','intermediate_tensors'),self.preprocessed[:4]))
            require(set(kwargs)==set(expected) and all(kwargs[k] is v for k,v in expected.items()),'PRODUCT_BATCH_REAL_MODEL_INPUTS')
        def out(module,args,value):self.model_output=value
        self.hooks=[self.model.register_forward_pre_hook(enter,with_kwargs=True),self.model.register_forward_hook(out)]

    def policy(self):
        require(self.frame is not None,'PRODUCT_BATCH_OUTSIDE_SCHEDULER')
        return dict(eligible=self.frame['eligible'],reason=self.frame['reason'],method='_forward_decode' if self.frame['decode'] else '_forward_prefill',batch_frame=self.frame['index'])

    def execute(self,original,scheduler,intermediate_tensors=None):
        self.binding.check();require(self.error is None,'PRODUCT_BATCH_OBSERVER_FAILED')
        # Canary contract errors belong to the control plane. Reject before
        # preprocessing/model execution, disarm only our non-injecting observer,
        # and let the real inference continue exactly once on its original path.
        # Model/kernel exceptions below are never swallowed or retried.
        if self.canary:
            ids=list(scheduler.num_scheduled_tokens)
            refusal=self.control_refusal
            if any(r in self.states for r in getattr(scheduler,'finished_req_ids',set())):
                refusal='PRODUCT_BATCH_CANARY_COMPLETION'
            elif ids:
                if self.completed_canary:refusal='PRODUCT_CANARY_EXTRA_REQUEST_BEFORE_DISARM'
                elif len(ids)!=1 or any(r!=ids[0] for r in self.states):
                    refusal='PRODUCT_BATCH_CANARY_REQUIRES_SERIAL_B1'
                elif ids[0] not in self.states:
                    new=[r for r in scheduler.scheduled_new_reqs if r.req_id==ids[0]]
                    if len(new)!=1 or tuple(getattr(new[0],'prompt_token_ids',()) or ())!=self.binding.prompts[len(self.canary_rows)]:
                        refusal='PRODUCT_BATCH_CANARY_PROMPT_ORDER'
            if refusal:
                self.control_refusal=refusal
                require(self.site is None or self.site.pending==0,'PRODUCT_BATCH_CONSUMERS_PENDING')
                self.request.close()
                for hook in self.hooks:hook.remove()
                self.hooks.clear()
                for name,wrapper in reversed(self.patches):
                    require(vars(self.runner).get(name) is wrapper,'PRODUCT_BATCH_WRAPPER_OWNER_CHANGED')
                    delattr(self.runner,name)
                self.patches.clear()
                self.states.clear();self.frame=None
                return original(scheduler,intermediate_tensors)
        for rid in getattr(scheduler,'finished_req_ids',set()):
            if rid in self.states:self.finish(rid,'SCHEDULER_FINISHED_REQUEST')
        if not scheduler.num_scheduled_tokens:return original(scheduler,intermediate_tensors)
        require(not self.completed_canary,'PRODUCT_CANARY_EXTRA_REQUEST_BEFORE_DISARM')
        self.scheduler=scheduler;self.new_ids={r.req_id for r in scheduler.scheduled_new_reqs}
        self.preprocessed=None;self.model_output=None;begin=len(self.site.events)
        forward_begin=len(self.site.forward_calls);uncovered_begin=len(self.site.uncovered)
        result=original(scheduler,intermediate_tensors)
        require(self.frame is not None,'PRODUCT_BATCH_PREPROCESS_NOT_OBSERVED')
        events=self.site.events[begin:];proof=summarize_events(events)
        require(len(self.site.forward_calls)==forward_begin+1,'PRODUCT_BATCH_MODEL_FORWARD_COUNT')
        site_coverage=actual_forward_coverage(events,self.site.forward_calls[-1],self.site.sites)
        site_coverage['uncovered_calls']=deepcopy(self.site.uncovered[uncovered_begin:])
        state=self.runner.execute_model_state;ids=self.frame['ids']
        require(result is None and state is not None and state.scheduler_output is scheduler
            and state.hidden_states is self.model_output and state.logits.shape[0]==len(ids),'PRODUCT_BATCH_RESULT_MAPPING')
        if self.canary:
            import torch
            from fault_injection.sites.mla28 import fingerprint
        offset=0
        for i,rid in enumerate(ids):
            count=scheduler.num_scheduled_tokens[rid];row=self.states[rid]
            row['events'].extend(events)
            row['frames'].append(dict(index=self.frame['index'],request_ids=list(ids),scheduled_counts=dict(scheduler.num_scheduled_tokens),
                eligible=self.frame['eligible'],reason=self.frame['reason'],site_coverage=deepcopy(site_coverage),**deepcopy(proof)))
            if self.canary:
                hidden=state.hidden_states[offset:offset+count];logits=state.logits[i]
                row['output'].append(dict(hidden=fingerprint(hidden),logits=fingerprint(logits),finite=bool(torch.isfinite(hidden).all().item() and torch.isfinite(logits).all().item())))
                pos=self.runner.input_batch.req_id_to_index[rid]
                row.setdefault('canary_schedule',[]).append(dict(batch_size=len(ids),prompt_index=len(self.canary_rows),prompt_sha256=digest(list(row['prompt'])),prompt_length=len(row['prompt']),computed=int(self.runner.input_batch.num_computed_tokens_cpu[pos]),scheduled=count))
            offset+=count
        return result

    def preprocess(self,original,scheduler,total,intermediate_tensors=None):
        require(scheduler is self.scheduler and intermediate_tensors is None,'PRODUCT_BATCH_SCHEDULER_CHANGED')
        batch=self.runner.input_batch;ids=list(batch.req_ids)
        require(set(ids)==set(scheduler.num_scheduled_tokens) and len(ids)==len(set(ids))
            and total==sum(scheduler.num_scheduled_tokens.values())==scheduler.total_num_scheduled_tokens,'PRODUCT_BATCH_TOKEN_MAPPING')
        rows=[]
        for rid in ids:
            req=self.runner.requests[rid];index=batch.req_id_to_index[rid]
            computed=int(batch.num_computed_tokens_cpu[index]);prompt=req.prompt_token_ids
            require(type(prompt) is list and all(type(t) is int for t in prompt),'PRODUCT_BATCH_PROMPT_UNPROVEN')
            if rid not in self.states:
                require(rid not in self.seen,'PRODUCT_BATCH_COMPLETED_ID_REUSED');self.seen.add(rid)
                features=getattr(req,'mm_features',None)
                facts=dict(request_count=1,prompt_length=len(prompt),max_tokens=getattr(req.sampling_params,'max_tokens',None),
                    single_chunk=rid in self.new_ids and computed==0 and scheduler.num_scheduled_tokens[rid]==len(prompt),fresh_cache=computed==0,
                    speculative=bool(scheduler.scheduled_spec_decode_tokens),lora=getattr(req,'lora_request',None) is not None,prompt_embeds=getattr(req,'prompt_embeds',None) is not None,
                    image_count=sum(f.modality=='image' for f in features) if isinstance(features,list) else None,
                    video_count=sum(f.modality=='video' for f in features) if isinstance(features,list) else None)
                reasons=request_range(facts)['reasons']
                if rid not in self.new_ids:reasons.append('REQUEST_STARTED_BEFORE_ENABLE')
                if self.canary:require(tuple(prompt) in self.binding.prompts,'PRODUCT_BATCH_CANARY_INPUT_MISMATCH')
                self.states[rid]=dict(prompt=tuple(prompt),facts=facts,reasons=reasons,samples=0,events=[],frames=[],output=[],external_request_id=getattr(req,'external_req_id',None))
            row=self.states[rid]
            require(tuple(prompt)==row['prompt'],'PRODUCT_BATCH_REQUEST_INPUT_CHANGED')
            if row['samples'] and not row['reasons']:
                require(computed==len(prompt)+row['samples']-1,'PRODUCT_BATCH_COMPUTED_TOKEN_MAPPING_CHANGED')
            rows.append(dict(id=rid,reasons=row['reasons'],computed=computed,prompt_length=len(prompt)))
        if self.canary:
            require(len(ids)==1 and len(self.states)==1,'PRODUCT_BATCH_CANARY_REQUIRES_SERIAL_B1')
            require(self.states[ids[0]]['prompt']==self.binding.prompts[len(self.canary_rows)],'PRODUCT_BATCH_CANARY_PROMPT_ORDER')
        eligible,reason=classify_batch(rows,scheduler.num_scheduled_tokens)
        self.frame=dict(index=self.frame_index,ids=ids,eligible=eligible and not self.canary,reason=reason,decode=all(r['computed']>=r['prompt_length'] for r in rows))
        # Canary still observes decode batch membership, but never injects.
        self.frame['eligible_candidate']=eligible;self.frame_index+=1
        result=original(scheduler,total,intermediate_tensors)
        require(len(result)==6 and result[3] is None and result[4]=={} and result[5] is None,'PRODUCT_BATCH_EXTRA_INPUT')
        self.preprocessed=result;return result

    def sample(self,original,*args,**kwargs):
        result=original(*args,**kwargs)
        if self.frame is None:return result
        require(result.req_ids==self.frame['ids'] and self.runner.execute_model_state is None,'PRODUCT_BATCH_SAMPLE_ASSOCIATION')
        for rid in self.frame['ids']:
            row=self.states[rid];row['samples']+=1
            if self.frame['eligible_candidate']:row['decode_batch_seen']=True
            limit=row['facts']['max_tokens']
            if type(limit) is int and row['samples']>=limit:self.finish(rid,'ACTUAL_SAMPLES_REACHED_REQUEST_MAX_TOKENS')
        self.frame=None;return result

    def finish(self,rid,reason,*,completed=True):
        row=self.states.pop(rid);proof=summarize_events(row['events'])
        reasons=list(row['reasons'])
        if not self.canary and proof['coverage']['eligible']==0:reasons+=sorted({f['reason'] for f in row['frames'] if f['reason']})
        record=dict(request_id=rid,rank=self.binding.identity['rank'],external_request_id=row['external_request_id'],completed=completed,completion_source=reason,
            status='NOT_INJECTED' if reasons else 'OBSERVED',reasons=reasons,**proof,bytes_restored=proof['coverage']['failed']==0,
            prompt_sha256=digest(list(row['prompt'])),range_facts=row['facts'],observed_model_steps=row['samples'],
            frames=[{k:v for k,v in f.items() if k!='sites'} for f in row['frames']],resources=dict(leases=0,hooks=0,inflight=0),
            resource_scope='REQUEST_REGIONS_RELEASED_OBSERVER_REMAINS_SCOPE_OWNED',model_baseline_comparison='NOT_PERFORMED_FOR_USER_REQUEST')
        self.records.append(record)
        if self.canary:
            key=row['prompt']
            if key in self.canary_rows or row['samples']!=4 or not completed or row['reasons']:
                self.control_refusal='PRODUCT_BATCH_CANARY_COMPLETION'
                return
            self.canary_rows[key]=row
            if len(self.canary_rows)==len(self.binding.prompts):
                output=[self.canary_rows[p]['output'] for p in self.binding.prompts]
                from .batch_canary import PROTOCOL
                self.binding.warmups.append(dict(output=output,all_finite=all(v['finite'] for steps in output for v in steps),canary_protocol=PROTOCOL,schedule=[self.canary_rows[p]['canary_schedule'] for p in self.binding.prompts]))
                self.completed_canary=True

    def close(self):
        require(self.inside==0 and (self.site is None or self.site.pending==0),'PRODUCT_BATCH_CONSUMERS_PENDING')
        for rid in list(self.states):self.finish(rid,'OBSERVER_DISABLED_AT_CONTROL_BOUNDARY',completed=False)
        self.request.close()
        for hook in self.hooks:hook.remove()
        self.hooks.clear()
        for name,wrapper in reversed(self.patches):
            require(vars(self.runner).get(name) is wrapper,'PRODUCT_BATCH_WRAPPER_OWNER_CHANGED');delattr(self.runner,name)
        self.patches.clear()

    def snapshot(self):
        return dict(requests=deepcopy(self.records),request_count=len(self.seen),active_request_id=next(iter(self.states),None),
            leases=self.site.pending if self.site else 0,owned_hooks=len(self.patches)+len(self.hooks)+(len(self.site.methods)+len(self.site.hooks) if self.site else 0),inflight=self.inside,failure=self.error)
