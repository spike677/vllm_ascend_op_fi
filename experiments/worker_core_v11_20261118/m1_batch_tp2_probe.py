"""Actual Qwen W8A8 TP2 ordinary B8; two owners on one Worker group."""
import argparse,hashlib,json,traceback,time
from pathlib import Path
from experiments.worker_core_v11_20261118 import probe_quant_ordinary as single
from experiments.c2c3_postwrite_20261117.natural41_probe import actual_worker
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.multi_rank import MultiRankTaskClient

def need(v,why):
    if not v:raise RuntimeError(why)

def setup(wrapper,identity,contract,build,manifest):
    from demoncat.runtime.vllm_ascend.worker_extension.quant_tp2 import install_quant_tp2_binding
    from demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance import install_blue_acceptance_permit
    w=actual_worker(wrapper)
    if w.demoncat_worker_identity()['worker']!=identity:return dict(status='NOT_TARGETED')
    result=install_quant_tp2_binding(w,root='/quant',build_sha256=build,group_contract=contract)
    install_blue_acceptance_permit(w,mechanisms=['quant.batch_shared_scale'],attempt=contract['owner'],source_manifest_sha256=manifest)
    return dict(status='BOUND',worker=identity,result=result)

def commit(wrapper,decision):
    from demoncat.runtime.vllm_ascend.worker_extension.quant_tp2 import commit_quant_tp2_role
    return commit_quant_tp2_role(actual_worker(wrapper),decision)

def remove(wrapper):
    from demoncat.runtime.vllm_ascend.worker_extension.serving_quant import remove_quant_binding
    from demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance import remove_blue_acceptance_permit
    w=actual_worker(wrapper);result=remove_quant_binding(w);remove_blue_acceptance_permit(w)
    del w._dcat_quant_ordinary_model
    return dict(result,worker=w.demoncat_worker_identity())

def health_install(wrapper):
    from vllm.distributed import get_tp_group
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from demoncat.runtime.vllm_ascend.worker_extension.multi_rank_prerequisites import tp2_eager_assessment
    from demoncat.runtime.vllm_ascend.worker_extension.serving_context import discover_logits_boundary
    from experiments.worker_core_v11_20261118.quant_health_observer import HealthHashes
    w=actual_worker(wrapper);r=w.model_runner
    need(not tp2_eager_assessment(r,get_tp_group(),w.rank,expected_max_seqs=8)['unmet'],'HEALTH_TP2_DEPLOYMENT_REQUIRED')
    need(not hasattr(w,'_dcat_quant_health_probe'),'HEALTH_ALREADY_ACTIVE')
    b=discover_logits_boundary(r.model,actual_logits_processor_type=LogitsProcessor,actual_vocab_size=r.vllm_config.model_config.hf_config.vocab_size)
    need(b['processor'].use_all_gather is True,'HEALTH_ALL_RANK_LOGITS_REQUIRED')
    o=HealthHashes(r.model,b['processor']);o.install();w._dcat_quant_health_probe=o
    return dict(rank=w.rank,installed=True)

def health_remove(wrapper,completed):
    from experiments.worker_core_v11_20261118.quant_health_observer import remove
    return remove(wrapper,completed)

def healthy(llm,prompts,params):
    need(not llm.llm_engine.has_unfinished_requests() and llm.reset_prefix_cache(reset_running_requests=False),'HEALTH_IDLE')
    llm.collective_rpc(health_install,timeout=120);complete=False
    try:
        outputs=llm.generate(prompts,params,use_tqdm=False);complete=True
    finally:hashes=llm.collective_rpc(health_remove,args=(complete,),timeout=120)
    return dict(tokens=[list(o.outputs[0].token_ids) for o in outputs],rank_hashes=hashes)

def review(states):
    need(set(states)=={0,1},'ALL_RANK_STATES');out={};ids=[]
    selected=states[0]['multi_rank']['selected_rank']
    for rank,state in states.items():
        s=state['serving'];rows=s['rows'];need([r['phase'] for r in rows]==list(single.PHASES),'SIX_REAL_BATCHES')
        n,f,o=rows[0],rows[4],rows[5];ids.append([r['decode']['decode']['request_ids'] for r in rows])
        need(n['projection']['hashes']==o['projection']['hashes'],'RANK_LOCAL_Q_SCALE_OFF_EXACT')
        need(all(n['decode'][k]==o['decode'][k] for k in ('hidden_sha256','logits_sha256')),'ALL_RANK_HIDDEN_LOGITS_OFF')
        need(s['actual_fault_hits']==int(rank==selected),'ONE_NATIVE_INJECTOR')
        if rank==selected:
            need(s['fault_effect']['quant']['changed_elements']>0 and s['fault_effect']['scale']['changed_elements']>0,'ACTUAL_Q_SCALE_EFFECT')
        else:need(s['guard_boundary_unchanged'] and f['projection']['hashes']==n['projection']['hashes'],'GUARD_LOCAL_BOUNDARY_CHANGED')
        need(not state['leases'] and not state['inflight'] and not state['native_registered'] and s['recovery_proven'],'ALL_RANK_RELEASE')
        out[rank]=dict(selected_site=s['selected_site']['path'],native_hits=s['actual_fault_hits'],q_scale_off_exact=True,
            hidden_changed=n['decode']['hidden_sha256']!=f['decode']['hidden_sha256'],logits_changed=n['decode']['logits_sha256']!=f['decode']['logits_sha256'])
    need(ids[0]==ids[1],'RANK_REQUEST_IDS_DIFFER');need(out[1-selected]['hidden_changed'],'CROSS_RANK_PROPAGATION_NOT_OBSERVED')
    return out

def main():
    from vllm import LLM,SamplingParams
    from demoncat.runtime.vllm_ascend.worker_extension.serving_context import actual_driver_request_map
    parser=argparse.ArgumentParser();parser.add_argument('--build-sha256',required=True);a=parser.parse_args()
    out=Path('/evidence');report=dict(status='STARTED',scopes=[],qualification_granted=False,combination='A2/CANN8.5.1/vLLM-Ascend0.18/Qwen3-0.6B W8A8/TP2/B8',collective_boundary_observed=False)
    def save():(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    started=time.monotonic();client=None;llm=None
    try:
        source=Path('/source');m=source/'source_manifest.json';manifest=hashlib.sha256(m.read_bytes()).hexdigest()
        for r in json.loads(m.read_bytes())['files']:need(hashlib.sha256((source/r['path']).read_bytes()).hexdigest()==r['sha256'],'FROZEN_SOURCE_CHANGED')
        report['source_manifest_sha256']=manifest;report['model']=single.model_identity('/model');save()
        from experiments.worker_core_v11_20261118.quant_spawn_bootstrap import worker_spawn_vendor_order
        with worker_spawn_vendor_order('/quant',a.build_sha256) as bootstrap:
            llm=LLM(model='/model',load_format='safetensors',dtype='bfloat16',tensor_parallel_size=2,distributed_executor_backend='mp',enforce_eager=True,max_model_len=128,max_num_seqs=8,max_num_batched_tokens=256,num_gpu_blocks_override=64,enable_prefix_caching=False,enable_chunked_prefill=False,async_scheduling=False,gpu_memory_utilization=.55,seed=42,trust_remote_code=False,disable_log_stats=True,worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        report['spawn_vendor_environment']=bootstrap;save()
        prompts=[dict(prompt_token_ids=llm.get_tokenizer().encode(t,add_special_tokens=True)) for t in single.PROMPTS]
        params=SamplingParams(temperature=0,max_tokens=2,ignore_eos=True);seen=set();held=None
        for index in range(2):
            scope=dict(index=index,phases=[]);report['scopes'].append(scope);scope['health_before']=healthy(llm,prompts,params);save()
            def bind(identity,config,contract):
                replies=llm.collective_rpc(setup,args=(identity,contract,a.build_sha256,manifest),timeout=120)
                need(sum(r['status']=='BOUND' for r in replies)==1,'ONE_CURRENT_WORKER_BINDING');return {}
            def roles(decision):scope['role_commit']=llm.collective_rpc(commit,args=(decision,),timeout=120);save()
            client=MultiRankTaskClient(llm,FaultConfig('quant','batch_shared_scale'),prepare_binding=bind,commit_roles=roles,timeout=120)
            def action(name,*args):
                result=getattr(client,name)(*args);scope['last_action']=result;save()
                need(result['status']==('QUERY_OK' if name=='query' else 'DETACHED' if name=='detach' else 'ACTION_COMPLETED'),'GROUP_'+name+':'+str(result['status']))
                return result['response']['provider_state']
            def batch(phase):
                need(not llm.llm_engine.has_unfinished_requests() and llm.reset_prefix_cache(reset_running_requests=False),'IDLE_BATCH')
                ids=list(llm.enqueue(prompts,params,use_tqdm=False));mapping=actual_driver_request_map(llm.llm_engine.output_processor)
                need(len(ids)==8 and len(set(ids))==8 and not set(ids)&seen and set(ids)=={r['internal_request_id'] for r in mapping},'ACTUAL_BATCH_MAPPING')
                seen.update(ids);need(action('advance_batch',ids)['next_phase']==phase,'PHASE_CHANGED')
                outputs=llm.wait_for_completion(use_tqdm=False)
                need(len(outputs)==8 and {o.request_id for o in outputs}=={r['external_request_id'] for r in mapping},'COMPLETED_BATCH_MAPPING')
                state=action('query');scope['phases'].append(dict(phase=phase,ids=ids,mapping=mapping,tokens=[list(o.outputs[0].token_ids) for o in outputs]));save();return state
            for name in ('prepare','register','check','enable'):action(name)
            members=client.contract()['members'];need(held is None or held==members,'WORKERS_RESTARTED');held=members
            for phase in single.PHASES:
                batch(phase)
                if phase=='fault':scope['explicit_recover']=action('recover')
            final=action('finish');scope['independent_checks']=review(final['rank_states']);scope['final_states']=final['rank_states']
            action('unregister');action('detach');scope['owner']=client.identity.owner;scope['ownership_retained']=client.snapshot()['ownership_retained']
            need(not scope['ownership_retained'],'OWNERSHIP_REMAIN');scope['removed']=llm.collective_rpc(remove,timeout=120)
            scope['health_after']=healthy(llm,prompts,params);need(scope['health_before']==scope['health_after'],'ALL_RANK_EXTERNAL_HEALTH_NOT_EXACT')
            scope['status']='TWO_RANK_SCOPE_REVIEW_PASSED';client=None;save()
        need(report['scopes'][0]['owner']!=report['scopes'][1]['owner'],'OWNER_REUSED')
        report['status']='TP2_BATCH_TWO_SCOPES_READY_FOR_REVIEW'
    except BaseException as e:
        report.update(status='FAILED_WITH_EVIDENCE',error=repr(e),traceback=traceback.format_exc())
        if client:
            try:report['failure_recover']=client.recover();report['owned_failure_snapshot']=client.snapshot()
            except BaseException as cleanup:report['cleanup_error']=repr(cleanup)
    finally:report['elapsed_seconds']=time.monotonic()-started;save()
    return 0 if report['status']=='TP2_BATCH_TWO_SCOPES_READY_FOR_REVIEW' else 2

if __name__=='__main__':raise SystemExit(main())
