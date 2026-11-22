"""Trusted source-bound continuous binding; no user selectors or permission grant.

The existing Runtime owns task leases. This object holds only deployment facts
and the per-request observer. The original Native research route stays closed.
"""
from dataclasses import dataclass,field
from copy import deepcopy
import hashlib,json,os,threading
from pathlib import Path
from .serving_context import require

ATTRIBUTE='_demoncat_continuous_bindings'
SUPPORTED=('20.buffer_substitute','26.fia_destination','41.tail_column','17.spatial_padding','33.paged_decode','28.fia.denominator_replay','quant.batch_shared_scale','39.output_proxy','39.matmul_range','18.api_half','22.wrong_row','8.restore_offset')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@dataclass
class ContinuousBinding:
    worker: object
    model: object
    mechanism: str
    prompts: tuple
    group: dict
    identity: dict
    arithmetic_device: str
    pid: int
    thread: int
    reference_mode: str='immediate'
    native_library: tuple|None=None
    visual_references: dict=field(default_factory=dict)
    leases: set=field(default_factory=set)
    warmups: list=field(default_factory=list)
    warmup_request: object=None

    @property
    def warmup_valid(self):
        return (len(self.warmups)>=2 and all(r['all_finite'] for r in self.warmups[-2:])
                and self.warmups[-1]['output']==self.warmups[-2]['output'])

    def check(self):
        require(self.pid==os.getpid() and self.thread==threading.get_ident()
                and self.worker.model_runner.model is self.model,'CONTINUOUS_BINDING_IDENTITY_CHANGED')
        require(self.group['members'][self.identity['rank']]==self.identity,'CONTINUOUS_GROUP_MEMBER_CHANGED')


def current_binding(worker,model,mechanism):
    bindings=getattr(worker,ATTRIBUTE,{})
    b=bindings.get(mechanism) if type(bindings) is dict else None
    if type(b) is not ContinuousBinding or b.worker is not worker or b.model is not model:return None
    b.check();return b


def install_continuous_binding(worker,*,mechanism,prompts,group_contract,arithmetic_device='input',reference_mode='immediate',native_library=None):
    require(mechanism in SUPPORTED,'CONTINUOUS_ADAPTER_NOT_IMPLEMENTED')
    require(arithmetic_device in ('input','cpu'),'CONTINUOUS_ARITHMETIC_UNREVIEWED')
    require(reference_mode=='immediate' or (reference_mode=='forward' and mechanism=='quant.batch_shared_scale'
        and arithmetic_device=='input'),'CONTINUOUS_REFERENCE_MODE_UNREVIEWED')
    require(type(prompts) is list and (2<=len(prompts)<=8 if mechanism.startswith('quant.') else len(prompts)==1)
        and all(type(p) is list and 2<=len(p)<=1024 and all(type(t) is int and t>=0 for t in p) for p in prompts),
        'CONTINUOUS_BOUNDED_TOKEN_PROMPTS_REQUIRED')
    runner=worker.model_runner;model=runner.model
    from .continuous_model import validate_continuous_model
    # Validate exact pinned execute/preprocess/sample/model sources and real
    # parallel groups, without installing hooks or creating any NPU call.
    if mechanism=='17.spatial_padding':
        from .continuous_vision17 import validate_visual_model
        family=validate_visual_model(runner)
    else:family=validate_continuous_model(runner,{'pending-'+str(i):p for i,p in enumerate(prompts)})
    require(family!='deepseek_v2' or mechanism in ('28.fia.denominator_replay','8.restore_offset','18.api_half','39.output_proxy','22.wrong_row'),'MLA_CONTINUOUS_SITE_NOT_ADAPTED')
    require(not model.training and not runner.vllm_config.cache_config.enable_prefix_caching
        and runner.vllm_config.kv_transfer_config is None,'CONTINUOUS_FRESH_CACHE_INFERENCE_REQUIRED')
    if mechanism in ('41.tail_column','26.fia_destination','20.buffer_substitute'):
        require(family=='qwen35' and type(native_library) in (list,tuple) and len(native_library)==2
            and all(type(x) is str and x for x in native_library),'KV41_TRUSTED_NATIVE_BUILD_REQUIRED')
        require(Path(native_library[0]).is_file() and hashlib.sha256(Path(native_library[0]).read_bytes()).hexdigest()==native_library[1],
            'KV41_NATIVE_BUILD_IDENTITY_MISMATCH')
    else:require(native_library is None,'NATIVE_LIBRARY_NOT_APPLICABLE')
    identity=worker.demoncat_worker_identity()['worker'];c=deepcopy(group_contract)
    require(type(c) is dict and c.get('mechanism')==mechanism and c.get('injection_policy')=='ALL_RANK_EVERY_ELIGIBLE_CALL'
        and c.get('tp')==runner.vllm_config.parallel_config.tensor_parallel_size
        and c.get('members_sha256')==digest(c.get('members'))
        and [m['rank'] for m in c['members']]==list(range(c['tp']))
        and c['members'][identity['rank']]==identity,'CONTINUOUS_ACTUAL_GROUP_REQUIRED')
    bindings=getattr(worker,ATTRIBUTE,{})
    require(type(bindings) is dict and not bindings,'CONTINUOUS_BINDING_ALREADY_INSTALLED')
    b=ContinuousBinding(worker,model,mechanism,tuple(tuple(p) for p in prompts),c,identity,
        arithmetic_device,os.getpid(),threading.get_ident(),reference_mode=reference_mode,native_library=tuple(native_library) if native_library else None)
    bindings[mechanism]=b;setattr(worker,ATTRIBUTE,bindings)
    return dict(status='CONTINUOUS_BOUND_NOT_QUALIFIED',mechanism=mechanism,qualification_granted=False)


def remove_continuous_binding(worker,mechanism):
    b=current_binding(worker,worker.model_runner.model,mechanism)
    require(b is not None and not b.leases and b.warmup_request is None,'CONTINUOUS_OWNERS_MUST_BE_RELEASED')
    del getattr(worker,ATTRIBUTE)[mechanism]


def observe_continuous_warmup(worker,mechanism,*,request_ids=None):
    """Trusted observer for already-enqueued normal requests, never inference."""
    b=current_binding(worker,worker.model_runner.model,mechanism)
    require(b is not None and not b.leases,'CONTINUOUS_WARMUP_REQUIRES_IDLE_BINDING')
    if request_ids is not None:
        require(b.warmup_request is None and len(b.warmups)<6,'CONTINUOUS_WARMUP_BUDGET')
        b.warmup_request=make_continuous_request(b,request_ids,'warmup');b.warmup_request.install()
    else:
        require(b.warmup_request is not None,'CONTINUOUS_WARMUP_NOT_STARTED')
        b.warmups.append(b.warmup_request.collect());b.warmup_request=None
    return dict(warmup_valid=b.warmup_valid,attempts=len(b.warmups),
                observations=deepcopy(b.warmups),qualification_granted=False)


def make_continuous_request(binding,ids,phase):
    if binding.mechanism=='17.spatial_padding':
        from .continuous_vision17 import ContinuousVisionRequest
        return ContinuousVisionRequest(binding,ids,phase)
    return ContinuousRequest(binding,ids,phase)


class ContinuousRequest:
    """Observe actual four-step scheduling. Never drives inference itself."""
    def __init__(self,binding,ids,phase):
        self.binding=binding;self.runner=binding.worker.model_runner;self.model=binding.model
        self.ids=list(ids);self.phase=phase;self.rows=[];self.patches=[];self.hooks=[]
        self.samples=0;self.inside=0;self.error=None;self.site=None;self.scheduler=None
        self.prompts=dict(zip(self.ids,binding.prompts));self.model_output=None;self.preprocessed=None

    def install(self):
        import torch
        from .v026_inspection import inspect_single_chunk
        from fault_injection.sites.mla28 import fingerprint
        self.binding.check()
        require(len(self.ids)==len(self.binding.prompts) and len(set(self.ids))==len(self.ids)
            and all(type(i) is str and i for i in self.ids),'CONTINUOUS_ACTUAL_IDS_REQUIRED')
        def patch(obj,name,handler):
            require(name not in vars(obj),'CONTINUOUS_METHOD_ALREADY_WRAPPED:'+name)
            original=getattr(obj,name)
            def wrapper(*args,**kwargs):
                self.inside+=1
                try:return handler(original,*args,**kwargs)
                except BaseException as e:self.error=repr(e);raise
                finally:self.inside-=1
            setattr(obj,name,wrapper);self.patches.append((obj,name,wrapper))
        def execute(original,scheduler,intermediate_tensors=None):
            self.binding.check()
            if scheduler.total_num_scheduled_tokens==0:
                require(not scheduler.num_scheduled_tokens and not scheduler.scheduled_new_reqs
                    and not scheduler.scheduled_encoder_inputs and not scheduler.scheduled_spec_decode_tokens,
                    'CONTINUOUS_NONEMPTY_CLEANUP')
                return original(scheduler,intermediate_tensors)
            require(self.samples<4 and self.runner.execute_model_state is None and intermediate_tensors is None,
                    'CONTINUOUS_EXECUTE_ORDER')
            self.scheduler=scheduler;self.model_output=None
            result=original(scheduler,intermediate_tensors)
            state=self.runner.execute_model_state
            require(result is None and state is not None and state.scheduler_output is scheduler
                and state.hidden_states is self.model_output,'CONTINUOUS_EXECUTE_ASSOCIATION')
            require(state.logits is not None and state.logits.shape[0]==len(self.ids),'CONTINUOUS_LOGITS_MAPPING')
            self.rows.append(dict(request_ids=list(self.runner.input_batch.req_ids),
                phase='prefill' if self.samples==0 else 'decode',hidden=fingerprint(state.hidden_states),
                logits=fingerprint(state.logits),finite=bool(torch.isfinite(state.hidden_states).all().item()
                    and torch.isfinite(state.logits).all().item())))
            return result
        def preprocess(original,scheduler,total,intermediate_tensors=None):
            require(scheduler is self.scheduler and intermediate_tensors is None,'CONTINUOUS_PREPROCESS_ASSOCIATION')
            if self.samples==0:
                report=inspect_single_chunk(self.runner,scheduler,self.prompts)
                require(report['status']=='matched','CONTINUOUS_SINGLE_CHUNK_NOT_PROVEN:'+json.dumps(report))
            else:
                require(scheduler.num_scheduled_tokens=={i:1 for i in self.ids}
                    and scheduler.total_num_scheduled_tokens==len(self.ids)
                    and not scheduler.scheduled_encoder_inputs and not scheduler.scheduled_spec_decode_tokens,
                    'CONTINUOUS_DECODE_SCHEDULE_CHANGED')
                require(set(self.runner.input_batch.req_ids)==set(self.ids),'CONTINUOUS_DECODE_IDS_CHANGED')
                for rid,prompt in self.prompts.items():
                    req=self.runner.requests[rid]
                    index=self.runner.input_batch.req_id_to_index[rid]
                    computed=int(self.runner.input_batch.num_computed_tokens_cpu[index])
                    require(req.prompt_token_ids==list(prompt) and not req.mm_features
                        and computed==len(prompt)+self.samples-1,'CONTINUOUS_DECODE_PREFIX_CHANGED')
            require(total==scheduler.total_num_scheduled_tokens,'CONTINUOUS_PADDING_NOT_SUPPORTED')
            result=original(scheduler,total,intermediate_tensors)
            require(len(result)==6 and result[3] is None and result[4]=={} and result[5] is None,
                    'CONTINUOUS_PREPROCESS_EXTRA_INPUT')
            self.preprocessed=result
            return result
        def model_enter(module,args,kwargs):
            require(module is self.model and not args and self.preprocessed is not None,
                    'CONTINUOUS_MODEL_INPUT_ORDER')
            expected=dict(zip(('input_ids','inputs_embeds','positions','intermediate_tensors'),self.preprocessed[:4]))
            require(set(kwargs)==set(expected) and all(kwargs[k] is v for k,v in expected.items()),
                    'CONTINUOUS_MODEL_INPUT_OBJECTS_CHANGED')
        def sample(original,*args,**kwargs):
            require(len(self.rows)==self.samples+1 and self.runner.execute_model_state is not None,
                    'CONTINUOUS_SAMPLE_ORDER')
            result=original(*args,**kwargs)
            require(self.runner.execute_model_state is None and set(result.req_ids)==set(self.ids),
                    'CONTINUOUS_SAMPLE_IDS_CHANGED')
            self.samples+=1;return result
        def model_output(module,args,value):
            require(isinstance(value,torch.Tensor),'CONTINUOUS_TENSOR_HIDDEN_REQUIRED')
            self.model_output=value
        def visual(*unused,**kwargs):raise RuntimeError('CONTINUOUS_VISUAL_CALL_FORBIDDEN')
        try:
            patch(self.runner,'execute_model',execute);patch(self.runner,'_preprocess',preprocess)
            patch(self.runner,'sample_tokens',sample)
            self.hooks.append(self.model.register_forward_pre_hook(model_enter,with_kwargs=True))
            self.hooks.append(self.model.register_forward_hook(model_output))
            visual_module=getattr(self.model,'visual',None)
            if visual_module is not None:self.hooks.append(visual_module.register_forward_pre_hook(visual))
            if self.binding.mechanism in ('18.api_half','22.wrong_row','8.restore_offset'):
                from fault_injection.sites.norm18 import Norm18Site
                from fault_injection.sites.rope22 import Rope22Site
                from fault_injection.sites.mla_rope22 import MLARope22Site
                from fault_injection.sites.moe8 import MoE8Site
                from fault_injection.engines.region_lease import RegionLeases
                def complete(stream):
                    event=torch.npu.Event();event.record(stream);event.synchronize()
                leases=RegionLeases(self.binding.group['owner'],self.binding.group['group_generation'],
                    stream=torch.npu.current_stream,complete=complete)
                site_type={'18.api_half':Norm18Site,'22.wrong_row':Rope22Site,'8.restore_offset':MoE8Site}[self.binding.mechanism]
                if self.binding.mechanism=='22.wrong_row' and type(self.model).__name__=='DeepseekV2ForCausalLM':site_type=MLARope22Site
                self.site=site_type(self.model,self.runner,fault=self.phase=='fault',region_leases=leases)
            elif self.binding.mechanism in ('41.tail_column','26.fia_destination','20.buffer_substitute'):
                from fault_injection.engines.native_regions import NativeRegions
                from fault_injection.sites.native_kv41 import NativeKV41Site
                from fault_injection.sites.native_fia26 import NativeFIA26Site
                from fault_injection.sites.native_input20 import NativeInput20Site
                native=NativeRegions(*self.binding.native_library,self.binding.group['owner'],self.binding.group['group_generation'])
                site_type={'41.tail_column':NativeKV41Site,'26.fia_destination':NativeFIA26Site,'20.buffer_substitute':NativeInput20Site}[self.binding.mechanism]
                self.site=site_type(self.model,self.runner,fault=self.phase=='fault',native=native)
            elif self.binding.mechanism in ('39.output_proxy','39.matmul_range'):
                from fault_injection.sites.projection39 import Projection39Site
                from fault_injection.engines.region_lease import RegionLeases
                def complete(stream):
                    event=torch.npu.Event();event.record(stream);event.synchronize()
                leases=RegionLeases(self.binding.group['owner'],self.binding.group['group_generation'],
                    stream=torch.npu.current_stream,complete=complete)
                from fault_injection.sites.matmul39 import Matmul39Site
                site_type=Matmul39Site if self.binding.mechanism=='39.matmul_range' else Projection39Site
                self.site=site_type(self.model,self.runner,fault=self.phase=='fault',region_leases=leases)
            elif self.binding.mechanism.startswith('quant.'):
                from fault_injection.sites.batch_quant import BatchQuantSite
                from fault_injection.engines.region_lease import RegionLeases
                def complete(stream):
                    event=torch.npu.Event();event.record(stream);event.synchronize()
                leases=RegionLeases(self.binding.group['owner'],self.binding.group['group_generation'],
                    stream=torch.npu.current_stream,complete=complete)
                self.site=BatchQuantSite(self.model,self.runner,fault=self.phase=='fault',
                    arithmetic_device=self.binding.arithmetic_device,region_leases=leases,reference_mode=self.binding.reference_mode)
            else:
                from fault_injection.sites.gqa28 import GQA28Site
                from fault_injection.sites.mla28 import MLA28Site
                from fault_injection.engines.region_lease import RegionLeases
                def complete(stream):
                    event=torch.npu.Event();event.record(stream);event.synchronize()
                leases=RegionLeases(self.binding.group['owner'],self.binding.group['group_generation'],
                    stream=torch.npu.current_stream,complete=complete)
                site_type=MLA28Site if type(self.model).__name__=='DeepseekV2ForCausalLM' else GQA28Site
                if self.binding.mechanism=='33.paged_decode':
                    from fault_injection.sites.paged33 import Paged33Site
                    site_type=Paged33Site
                self.site=site_type(self.model,self.runner,fault=self.phase=='fault',region_leases=leases)
            self.site.install()
        except BaseException:
            self.close();raise

    def collect(self):
        require(self.inside==0 and self.samples==4 and self.error is None,'CONTINUOUS_REQUEST_NOT_COMPLETE')
        events=self.site.events
        actual_sites={e['site'] for e in events}
        if self.binding.mechanism in ('39.output_proxy','39.matmul_range','18.api_half','22.wrong_row','8.restore_offset'):
            calls=self.site.forward_calls
            require(len(calls)==4 and all(calls) and sum(sum(c.values()) for c in calls)==len(events),
                    'CONTINUOUS_ACTUAL_FORWARD_COVERAGE_INCOMPLETE')
        else:
            require(len(events)==4*len(self.site.sites),'CONTINUOUS_CALL_COVERAGE_INCOMPLETE')
        for e in events:
            actual=e.get('request_ids',[e.get('request_id')])
            require(set(actual)==set(self.ids),'CONTINUOUS_SITE_REQUEST_MAPPING')
        counts={k:0 for k in ('observed','eligible','attempted','applied','changed','consumed','restored','failed')}
        counts['observed']=len(events)
        if self.phase=='fault':
            eligible=[e for e in events if e.get('eligible',True)]
            counts.update(eligible=len(eligible),attempted=len(eligible))
            for e in eligible:
                for k in ('applied','consumed','restored','failed'):counts[k]+=int(bool(e.get(k,False)))
                counts['changed']+=int(e.get('changed',0)>0)
                require(e.get('before_sha256')==e.get('restored_sha256'),'CONTINUOUS_RESTORE_HASH_MISMATCH')
                if self.binding.mechanism=='39.matmul_range':
                    require(e.get('full_range_control_passed') is True,'CONTINUOUS_MATMUL_REFERENCE_MISSING')
                if self.binding.mechanism.startswith('quant.'):
                    require(e.get('reference_passed') is True,'CONTINUOUS_QUANT_REFERENCE_FAILED')
                elif self.binding.mechanism in ('28.fia.denominator_replay','33.paged_decode'):
                    require(e.get('non_target_exact') is True,'CONTINUOUS_NON_TARGET_CHANGED')
            require(counts['eligible']==counts['applied']==counts['consumed']==counts['restored']
                    and counts['changed']>0 and counts['failed']==0,'CONTINUOUS_ELIGIBLE_CALL_FAILED')
        require(not getattr(self.site,'uncovered',[]),'CONTINUOUS_UNCOVERED_SITE')
        output=[{k:r[k] for k in ('hidden','logits','finite')} for r in self.rows]
        result=dict(phase=self.phase,request_ids=self.ids,output=output,coverage=counts,
            all_finite=all(r['finite'] for r in self.rows),site_count=len(actual_sites),
            structural_candidate_count=len(self.site.sites),
            structural_candidates_without_call=sorted(set(self.site.sites)-actual_sites),
            actual_forward_calls=getattr(self.site,'forward_calls',None),
            events_sha256=digest(events),same_stream_verified=True,bytes_restored=True,
            observation_complete=True,uncovered_sites=list(getattr(self.site,'uncovered',[])),
            excluded_structural_sites=list(getattr(self.site,'excluded',[])),
            inapplicable_reasons=[dict(site=e['site'],reason=e['reason']) for e in events if e.get('eligible') is False and e.get('reason')],
            partition_effect=dict(calls=sum(bool(e.get('partition_reference')) for e in events),
                selected_changed_elements=sum(e.get('partition_reference',{}).get('selected_changed_elements',0) for e in events),
                pairs=[dict(site=e['site'],page_mapping=e['page_mapping'],**e['partition_reference']) for e in events if e.get('partition_reference')]),
            matmul_effect=dict(reference_controls=sum(bool(e.get('full_range_control_passed')) for e in events),
                rules=sorted({e['range_reference']['rule'] for e in events if e.get('range_reference')}),
                actual_quantized_calls=sum(bool(e.get('real_qmm_inputs')) for e in events)),
            projection_effect=dict(already_positive_inf_calls=sum(bool(e.get('already_positive_inf')) for e in events),
                upstream_nonfinite_calls=sum(bool(e.get('upstream_nonfinite')) for e in events),
                consumer_operators=sorted({e['consumer_operator'] for e in events if e.get('consumer_operator')})),
            moe_effect=dict(actual_producer_joins=sum(bool(e.get('actual_producer_join')) for e in events),
                native_combine_calls=sum(e.get('native_calls',0) for e in events),
                single_group_calls=sum(e.get('token_rows')==1 for e in events),
                request_rows_verified=sum(len(e.get('row_request_ids',[])) for e in events)),
            rotary_effect=dict(rows_changed=sum(e.get('rotary_rows_changed',0) for e in events),
                position_rows_changed=sum(e.get('position_rows_changed',0) for e in events),
                qk_output_fingerprints=[dict(site=e['site'],q=e['q_output_sha256'],k=e['k_output_sha256'])
                    for e in events if e.get('q_output_sha256')]),
            native_memory_effect=dict(calls=sum(e.get('applied',False) for e in events),
                writer_operators=sorted({e['writer'] for e in events if e.get('writer')}),
                consumer_operators=sorted({e['consumer_operator'] for e in events if e.get('consumer_operator')}),
                library_sha256=self.binding.native_library[1] if self.binding.native_library else None,
                regions=[{k:e[k] for k in ('site','slots','page_mapping','region_elements','changed','before_sha256','after_sha256','restored_sha256','consumed','restored','source_sha256','consumer_argument','qmm_calls') if k in e}
                    for e in events if e.get('writer') and e.get('applied')]),
            quant_reference=dict(sampled_elements=sum(e.get('sampled_elements',0) for e in events),
                q_sample_differences=sum(e.get('q_sample_differences',0) for e in events),
                backend=self.binding.arithmetic_device,mode=self.binding.reference_mode))
        if self.binding.mechanism=='20.buffer_substitute':
            from fault_injection.engines.native_evidence import compact_content_regions
            effect=result['native_memory_effect']
            effect['region_proofs']=compact_content_regions(effect.pop('regions'))
        ref=result['quant_reference']
        ref['q_sample_difference_fraction']=(ref['q_sample_differences']/ref['sampled_elements']
                                            if ref['sampled_elements'] else None)
        ref['reference_scope']='SAMPLED' if ref['sampled_elements'] else 'FULL_CPU' if self.binding.arithmetic_device=='cpu' else 'NOT_APPLIED'
        self.close();result['resources']=dict(leases=0,hooks=0,inflight=0)
        return result

    def close(self):
        require(self.inside==0,'CONTINUOUS_OBSERVER_STILL_EXECUTING')
        if self.site is not None:
            self.site.close();self.site=None
        require(all(vars(o).get(n) is w for o,n,w in self.patches),'CONTINUOUS_WRAPPER_OWNER_CHANGED')
        for h in self.hooks:h.remove()
        self.hooks.clear()
        for o,n,_ in reversed(self.patches):delattr(o,n)
        self.patches.clear()
