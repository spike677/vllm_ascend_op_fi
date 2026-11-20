"""0.26 Qwen3.5-family ordinary RMS18 TP2, paused enqueue, two real owners."""
import hashlib,json,traceback
from pathlib import Path
from experiments.c2c3_postwrite_20261117.natural41_probe import actual_worker,output
from experiments.c2c3_postwrite_20261117.model_identity import vllm_safetensors_identity
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.multi_rank import MultiRankTaskClient
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import actual_driver_request_map

def read_enqueued_identity(llm,ids,*,seen_internal_request_ids=()):
    require(llm.llm_engine.is_sleeping() is True,'SCHEDULER_MUST_REMAIN_PAUSED')
    rows=actual_driver_request_map(llm.llm_engine.output_processor)
    require(len(ids)==len(rows)==1 and rows[0]['internal_request_id']==ids[0]
        and ids[0] not in seen_internal_request_ids,'REAL_PAUSED_REQUEST_ID_REQUIRED')
    return rows[0]


def require(value,reason):
    if not value:raise RuntimeError(reason)

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def setup(wrapper,identity,contract,prompt,weights,manifest,native):
    from demoncat.runtime.vllm_ascend.worker_extension.new_stack_rms18 import install_rms18_newstack_binding as install_rms18_tp2_binding
    from demoncat.runtime.vllm_ascend.worker_extension.rope22_tp2 import install_rope22_tp2_binding
    from demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance import install_blue_acceptance_permit
    worker=actual_worker(wrapper)
    if worker.demoncat_worker_identity().get('worker')!=identity:return dict(status='NOT_TARGETED')
    if contract['mechanism']=='39.output_proxy':
        from fault_injection.deployments.postwrite import NativePostwriteLibrary
        from demoncat.runtime.vllm_ascend.worker_extension.serving39_output import install_fault39_output_serving_binding
        transport=NativePostwriteLibrary(native['path'],native['sha256'])
        result=install_fault39_output_serving_binding(worker,transport=transport,prompt_token_ids=prompt,runtime_only=True,group_contract=contract)
    else:
        installer={'18.api_half':install_rms18_tp2_binding,'22.wrong_row':install_rope22_tp2_binding}[contract['mechanism']]
        result=installer(worker,group_contract=contract,prompt_token_ids=prompt,
            weight_manifest=weights,workload_sha256=digest(prompt),environment_sha256=digest(dict(source=manifest)))
    install_blue_acceptance_permit(worker,mechanisms=[contract['mechanism']],attempt=contract['owner'],source_manifest_sha256=manifest)
    return dict(status='BOUND',worker=identity,binding=result)

def commit(wrapper,decision):
    from demoncat.runtime.vllm_ascend.worker_extension.new_stack_rms18 import commit_newstack_rms18_role as commit_rms18_tp2_role
    from demoncat.runtime.vllm_ascend.worker_extension.rope22_tp2 import commit_rope22_tp2_role
    from demoncat.runtime.vllm_ascend.worker_extension.output39_tp2 import commit_output39_tp2_role
    fn={'39.output_proxy':commit_output39_tp2_role,'18.api_half':commit_rms18_tp2_role,'22.wrong_row':commit_rope22_tp2_role}[decision['group']['mechanism']]
    return fn(actual_worker(wrapper),decision)

def remove(wrapper,mechanism):
    from demoncat.runtime.vllm_ascend.worker_extension.serving_tensor_binding import remove_tensor_serving_binding
    from demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance import remove_blue_acceptance_permit
    worker=actual_worker(wrapper)
    if mechanism=='39.output_proxy':
        from demoncat.runtime.vllm_ascend.worker_extension.serving39_output import remove_fault39_output_serving_binding
        result=remove_fault39_output_serving_binding(worker)
    else:result=remove_tensor_serving_binding(worker,mechanism)
    remove_blue_acceptance_permit(worker)
    return dict(result,worker=worker.demoncat_worker_identity()['worker'])

def verify_output39(states):
    require(set(states)=={0,1},'BOTH_RANKS_REQUIRED');results={};ids=[]
    selected=states[0]['multi_rank']['selected_rank']
    for rank,state in states.items():
        rows=state['evidence']['actual_requests'];require([r['phase'] for r in rows]==['normal','fault','off','health'],'FOUR_REAL_REQUESTS')
        n,f,o,h=rows;ids.append([r['actual_internal_request_id'] for r in rows]);e=f['report']
        require(n['hidden_logits_sha256']==o['hidden_logits_sha256']==h['hidden_logits_sha256'],'ALL_RANK_EXACT_RECOVERY')
        require(e['applied']==int(rank==selected) and e['recovered_storage_exact'] and e['consumers'],'ACTUAL_TARGET_AND_CONSUMER')
        if rank!=selected:require(e['guard_boundary_unchanged'] and e['native_lease_receipt'] is None,'GUARD_NOT_READ_ONLY')
        else:require(e['effect']['changed_elements']==1 and e['effect']['finite_to_positive_inf']==1,'NATIVE_EFFECT_REQUIRED')
        require(n['hidden_logits_sha256']!=f['hidden_logits_sha256'],'RANK_PROPAGATION_REQUIRED')
        results[rank]=dict(path=e['target']['module_path'],applied=e['applied'],combined_hidden_logits_changed=True,off_health_exact=True)
    require(ids[0]==ids[1],'RANK_REQUEST_ALIGNMENT');return results

def verify(states):
    selected=states[0]['multi_rank']['selected_rank'];require(selected in (0,1),'AUTOMATIC_RANK_REQUIRED')
    require(set(states)=={0,1},'BOTH_RANKS_REQUIRED')
    result={}
    for rank,state in states.items():
        rows=state['serving']['actual_requests'];require([r['phase'] for r in rows]==['normal','fault','off','health'],'FOUR_REAL_REQUESTS')
        n,f,o,h=rows
        require(n['output']==o['output']==h['output'] and state['serving']['recovered'],'ALL_RANK_EXACT_RECOVERY')
        events=f['mechanism']['events'];require(len(events)==1,'ONE_SELECTED_RETURN')
        require(events[0]['applied']==(rank==selected),'ONE_INJECTOR_ONLY')
        require(events[0]['return_readback_exact'] and events[0]['consumers'],'ACTUAL_RETURN_AND_CONSUMER')
        if rank!=selected:require(f['guard_boundary_unchanged'],'GUARD_UNCHANGED')
        else:require(events[0]['output_changed'],'INJECTOR_EFFECT')
        result[rank]=dict(selected_path=state['serving']['selection']['path'],applied=events[0]['applied'],
            hidden_changed=n['output']['hidden']!=f['output']['hidden'],logits_changed=n['output']['logits']!=f['output']['logits'],
            exact_off=True,exact_health=True,normal=n['output'],fault=f['output'])
    require([r['request_id'] for r in states[0]['serving']['actual_requests']]==[r['request_id'] for r in states[1]['serving']['actual_requests']],'RANK_REQUEST_ALIGNMENT')
    require(result[1-selected]['hidden_changed'],'GUARD_RANK_PROPAGATION_REQUIRED')
    return result

def main():
    from vllm import LLM,SamplingParams
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--mechanism',choices=['18.api_half'],default='18.api_half');parser.add_argument('--recover-after-fault',action='store_true');parser.add_argument('--native-library');parser.add_argument('--native-sha256');options=parser.parse_args();mechanism=options.mechanism
    native=dict(path=options.native_library,sha256=options.native_sha256)
    if mechanism=='39.output_proxy':require(all(native.values()),'EXPLICIT_NATIVE_BUILD_REQUIRED')
    fault_id,variant=mechanism.split('.',1);status='TP2_'+{'18':'RMS18','22':'ROPE22','39':'OUTPUT39'}[fault_id]+'_TWO_SCOPES_READY_FOR_INDEPENDENT_REVIEW'
    out=Path('/evidence/probe');out.mkdir()
    report=dict(status='STARTED',mechanism=mechanism,explicit_group_recovery=options.recover_after_fault,scopes=[],qualification_granted=False,combination='A2/0.26.0+empty/CANN9.1.0/Qwen3.8 W8A8 community/TP2',
        collective_boundary_observed=False,propagation_scope='per-rank full hidden/global logits, no unique collective attribution')
    def save():(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        root=Path('/source');manifest=root/'source_manifest.json';manifest_sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
        for row in json.loads(manifest.read_bytes())['files']:
            require(hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256'],'SOURCE_BYTES_CHANGED')
        report['source_manifest_sha256']=manifest_sha
        report['weights']=vllm_safetensors_identity('/model')
        from demoncat.runtime.vllm_ascend.worker_extension.v026_text_observer import PROFILES
        import inspect
        package=Path(inspect.getsourcefile(LLM)).resolve().parents[1]
        pins=json.loads(PROFILES['v026_installed_c9c9432b'].read_bytes())['files']
        for name in ('entrypoints/llm.py','v1/engine/llm_engine.py','v1/engine/core_client.py','v1/engine/core.py'):
            sha=hashlib.sha256((package/name).read_bytes()).hexdigest()
            require(any(row['path']=='vllm/'+name and row['sha256']==sha for row in pins),'HOST_SCHEDULER_SOURCE_NOT_REVIEWED')
        llm=LLM(model='/model',dtype='bfloat16',load_format='safetensors',trust_remote_code=False,
            tensor_parallel_size=2,distributed_executor_backend='mp',enforce_eager=True,
            max_model_len=1024,max_num_seqs=1,max_num_batched_tokens=1024,
            enable_prefix_caching=False,enable_chunked_prefill=True,async_scheduling=False,
            gpu_memory_utilization=.8,seed=42,disable_log_stats=True,limit_mm_per_prompt={'image':0,'video':0},
            quantization='ascend',speculative_config=None,kv_transfer_config=None,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        prompt=llm.get_tokenizer().encode('The capital of France is',add_special_tokens=True)
        params=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True);seen=[]
        for index in range(2):
            scope=dict(index=index,requests=[],status='STARTED');report['scopes'].append(scope);save()
            def bind(identity,config,contract):
                rows=llm.collective_rpc(setup,args=(identity,contract,prompt,report['weights'],manifest_sha,native),timeout=120)
                require(sum(r['status']=='BOUND' for r in rows)==1,'ONE_TARGETED_BINDING')
                scope.setdefault('bindings',[]).extend(r for r in rows if r['status']=='BOUND');return {}
            def roles(decision):
                scope['role_commit']=llm.collective_rpc(commit,args=(decision,),timeout=120);save()
            client=MultiRankTaskClient(llm,FaultConfig(int(fault_id),variant),prepare_binding=bind,commit_roles=roles,timeout=120)
            def action(name,*args):
                row=getattr(client,name)(*args);scope['last_action']=row;save()
                require(row['status']==('QUERY_OK' if name=='query' else 'DETACHED' if name=='detach' else 'ACTION_COMPLETED'),'GROUP_ACTION_FAILED:'+name)
                return row['response']['provider_state']
            def request(phase):
                require(not llm.llm_engine.has_unfinished_requests(),'IDLE_QUEUE_REQUIRED')
                require(llm.reset_prefix_cache(reset_running_requests=False) is True,'CACHE_RESET_REQUIRED')
                llm.sleep(level=0,mode='wait')
                require(llm.llm_engine.is_sleeping() is True,'PAUSE_REQUIRED')
                pending=list(llm.enqueue([dict(prompt_token_ids=prompt)],params,use_tqdm=False))
                mapping=read_enqueued_identity(llm,pending,seen_internal_request_ids=seen);seen.extend(pending)
                require(len(pending)==1,'ONE_REQUEST_REQUIRED');state=action('advance',pending[0])
                require(state['next_phase']==phase,'GROUP_PHASE_CHANGED')
                llm.wake_up(tags=['scheduling'])
                results=llm.wait_for_completion(use_tqdm=False)
                require(len(results)==1 and results[0].request_id==mapping['external_request_id'],'COMPLETED_REQUEST_MAPPING')
                scope['requests'].append(dict(phase=phase,mapping=mapping,output=output(results[0])));save()
                return action('query')
            try:
                for name in ('prepare','register','check','enable'):action(name)
                for phase in ('normal','fault','off','health'):
                    state=request(phase)
                    if phase=='fault' and options.recover_after_fault:action('recover')
                require(state['complete'],'ALL_RANK_COMPLETE');scope['checks']=(verify_output39 if mechanism=='39.output_proxy' else verify)(state['rank_states'])
                scope['final_states']=state['rank_states'];final=action('finish')
                require(all(not any(s[k] for k in ('owned_hooks','leases','inflight','replay_pending')) for s in final['rank_states'].values()),'ALL_RANK_RESOURCE_ZERO')
                action('disable');action('unregister');action('detach')
                scope['remove']=llm.collective_rpc(remove,args=(mechanism,),timeout=120)
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
        report['status']=status
    except BaseException as error:report.update(status='FAILED',error=repr(error),traceback=traceback.format_exc())
    save();print(json.dumps(dict(status=report['status'],error=report.get('error'))),flush=True)
    return 0 if report['status']==status else 2
if __name__=='__main__':raise SystemExit(main())
