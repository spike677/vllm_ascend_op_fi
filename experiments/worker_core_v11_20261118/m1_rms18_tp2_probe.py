"""Actual TP2 standard workers, two scopes; no simulated NPU acceptance."""
import hashlib,json,traceback
from pathlib import Path
from experiments.c2c3_postwrite_20261117.natural41_probe import actual_worker,output
from experiments.c2c3_postwrite_20261117.model_identity import vllm_safetensors_identity
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.multi_rank import MultiRankTaskClient
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import read_enqueued_identity

def require(value,reason):
    if not value:raise RuntimeError(reason)

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def setup(wrapper,identity,contract,prompt,weights,manifest):
    from demoncat.runtime.vllm_ascend.worker_extension.rms18_tp2 import install_rms18_tp2_binding
    from demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance import install_blue_acceptance_permit
    worker=actual_worker(wrapper)
    if worker.demoncat_worker_identity().get('worker')!=identity:return dict(status='NOT_TARGETED')
    result=install_rms18_tp2_binding(worker,group_contract=contract,prompt_token_ids=prompt,
        weight_manifest=weights,workload_sha256=digest(prompt),environment_sha256=digest(dict(source=manifest)))
    install_blue_acceptance_permit(worker,mechanisms=['18.api_half'],attempt=contract['owner'],source_manifest_sha256=manifest)
    return dict(status='BOUND',worker=identity,binding=result)

def commit(wrapper,decision):
    from demoncat.runtime.vllm_ascend.worker_extension.rms18_tp2 import commit_rms18_tp2_role
    return commit_rms18_tp2_role(actual_worker(wrapper),decision)

def remove(wrapper):
    from demoncat.runtime.vllm_ascend.worker_extension.serving_tensor_binding import remove_tensor_serving_binding
    from demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance import remove_blue_acceptance_permit
    worker=actual_worker(wrapper)
    result=remove_tensor_serving_binding(worker,'18.api_half');remove_blue_acceptance_permit(worker)
    return dict(result,worker=worker.demoncat_worker_identity()['worker'])

def verify(states):
    require(set(states)=={0,1},'BOTH_RANKS_REQUIRED')
    result={}
    for rank,state in states.items():
        rows=state['serving']['actual_requests'];require([r['phase'] for r in rows]==['normal','fault','off','health'],'FOUR_REAL_REQUESTS')
        n,f,o,h=rows
        require(n['output']==o['output']==h['output'] and state['serving']['recovered'],'ALL_RANK_EXACT_RECOVERY')
        events=f['mechanism']['events'];require(len(events)==1,'ONE_SELECTED_RETURN')
        require(events[0]['applied']==(rank==0),'ONE_INJECTOR_ONLY')
        require(events[0]['return_readback_exact'] and events[0]['consumers'],'ACTUAL_RETURN_AND_CONSUMER')
        if rank==1:require(f['guard_boundary_unchanged'],'GUARD_UNCHANGED')
        else:require(events[0]['output_changed'],'INJECTOR_EFFECT')
        result[rank]=dict(selected_path=state['serving']['selection']['path'],applied=events[0]['applied'],
            hidden_changed=n['output']['hidden']!=f['output']['hidden'],logits_changed=n['output']['logits']!=f['output']['logits'],
            exact_off=True,exact_health=True,normal=n['output'],fault=f['output'])
    require([r['request_id'] for r in states[0]['serving']['actual_requests']]==[r['request_id'] for r in states[1]['serving']['actual_requests']],'RANK_REQUEST_ALIGNMENT')
    require(result[1]['hidden_changed'],'GUARD_RANK_PROPAGATION_REQUIRED')
    return result

def main():
    from vllm import LLM,SamplingParams
    out=Path('/evidence/probe');out.mkdir()
    report=dict(status='STARTED',scopes=[],qualification_granted=False,combination='A2/0.18/CANN8.5.1/Qwen3-0.6B BF16/TP2',
        collective_boundary_observed=False,propagation_scope='per-rank full hidden/global logits, no unique collective attribution')
    def save():(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        root=Path('/source');manifest=root/'source_manifest.json';manifest_sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
        for row in json.loads(manifest.read_bytes())['files']:
            require(hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256'],'SOURCE_BYTES_CHANGED')
        report['source_manifest_sha256']=manifest_sha
        report['weights']=vllm_safetensors_identity('/model')
        llm=LLM(model='/model',dtype='bfloat16',load_format='safetensors',trust_remote_code=False,
            tensor_parallel_size=2,distributed_executor_backend='mp',enforce_eager=True,
            max_model_len=128,max_num_seqs=1,max_num_batched_tokens=128,num_gpu_blocks_override=64,
            enable_prefix_caching=False,enable_chunked_prefill=False,async_scheduling=False,
            gpu_memory_utilization=.35,seed=42,disable_log_stats=True,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        prompt=llm.get_tokenizer().encode('The capital of France is',add_special_tokens=True)
        params=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True);seen=[]
        for index in range(2):
            scope=dict(index=index,requests=[],status='STARTED');report['scopes'].append(scope);save()
            def bind(identity,config,contract):
                rows=llm.collective_rpc(setup,args=(identity,contract,prompt,report['weights'],manifest_sha),timeout=120)
                require(sum(r['status']=='BOUND' for r in rows)==1,'ONE_TARGETED_BINDING')
                scope.setdefault('bindings',[]).extend(r for r in rows if r['status']=='BOUND');return {}
            def roles(decision):
                scope['role_commit']=llm.collective_rpc(commit,args=(decision,),timeout=120);save()
            client=MultiRankTaskClient(llm,FaultConfig(18,'api_half'),prepare_binding=bind,commit_roles=roles,timeout=120)
            def action(name,*args):
                row=getattr(client,name)(*args);scope['last_action']=row;save()
                require(row['status']==('QUERY_OK' if name=='query' else 'DETACHED' if name=='detach' else 'ACTION_COMPLETED'),'GROUP_ACTION_FAILED:'+name)
                return row['response']['provider_state']
            def request(phase):
                require(not llm.llm_engine.has_unfinished_requests(),'IDLE_QUEUE_REQUIRED')
                require(llm.reset_prefix_cache(reset_running_requests=False) is True,'CACHE_RESET_REQUIRED')
                pending=list(llm.enqueue([dict(prompt_token_ids=prompt)],params,use_tqdm=False))
                mapping=read_enqueued_identity(llm,pending,seen_internal_request_ids=seen);seen.extend(pending)
                require(len(pending)==1,'ONE_REQUEST_REQUIRED');state=action('advance',pending[0])
                require(state['next_phase']==phase,'GROUP_PHASE_CHANGED')
                results=llm.wait_for_completion(use_tqdm=False)
                require(len(results)==1 and results[0].request_id==mapping['external_request_id'],'COMPLETED_REQUEST_MAPPING')
                scope['requests'].append(dict(phase=phase,mapping=mapping,output=output(results[0])));save()
                return action('query')
            try:
                for name in ('prepare','register','check','enable'):action(name)
                for phase in ('normal','fault','off','health'):state=request(phase)
                require(state['complete'],'ALL_RANK_COMPLETE');scope['checks']=verify(state['rank_states'])
                scope['final_states']=state['rank_states'];final=action('finish')
                require(all(not any(s[k] for k in ('owned_hooks','leases','inflight','replay_pending')) for s in final['rank_states'].values()),'ALL_RANK_RESOURCE_ZERO')
                action('disable');action('unregister');action('detach')
                scope['remove']=llm.collective_rpc(remove,timeout=120)
                scope['status']='TWO_RANK_SCOPE_READY_FOR_REVIEW'
            except BaseException as error:
                scope.update(status='FAILED',error=repr(error),traceback=traceback.format_exc());save()
                try:
                    recovered=action('recover')
                    phase=recovered['next_phase']
                    if phase=='off':request('off');request('health')
                    action('unregister');action('detach')
                except BaseException as recovery:scope['recovery_error']=repr(recovery)
                raise
            finally:
                snapshot=client.snapshot();scope.update(group=snapshot['group'],ownership_retained=snapshot['ownership_retained'])
                (out/('scope'+str(index)+'_host.json')).write_text(json.dumps(snapshot,indent=2));save()
        a,b=report['scopes'];require(a['group']['members']==b['group']['members'] and a['group']['owner']!=b['group']['owner'],'SAME_WORKERS_TWO_OWNERS')
        report['status']='TP2_RMS18_TWO_SCOPES_READY_FOR_INDEPENDENT_REVIEW'
    except BaseException as error:report.update(status='FAILED',error=repr(error),traceback=traceback.format_exc())
    save();print(json.dumps(dict(status=report['status'],error=report.get('error'))),flush=True)
    return 0 if report['status']=='TP2_RMS18_TWO_SCOPES_READY_FOR_INDEPENDENT_REVIEW' else 2
if __name__=='__main__':raise SystemExit(main())
