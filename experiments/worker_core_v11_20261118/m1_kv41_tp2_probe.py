"""TP2 KV41 six real requests, two owners, full hidden/logits observers."""
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

def setup(wrapper,identity,contract,prompt,weights,manifest,native):
    from demoncat.runtime.vllm_ascend.worker_extension.rms18_tp2 import install_rms18_tp2_binding
    from demoncat.runtime.vllm_ascend.worker_extension.rope22_tp2 import install_rope22_tp2_binding
    from demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance import install_blue_acceptance_permit
    worker=actual_worker(wrapper)
    if worker.demoncat_worker_identity().get('worker')!=identity:return dict(status='NOT_TARGETED')
    if contract['mechanism']=='41.tail_column':
        from fault_injection.coverage_session import CoverageScope
        from fault_injection.deployments.postwrite import NativePostwriteLibrary
        from experiments.c2c3_postwrite_20261117.model_identity import model_source_identity
        from demoncat.runtime.vllm_ascend.worker_extension.serving41 import install_fault41_task_binding
        source=model_source_identity(worker.model_runner.model)
        scope=CoverageScope('41.tail_column',digest(prompt),digest(dict(worker=identity,source=manifest)),manifest,
            'natural_worker',weights['weights_sha256'],source['source_sha256'])
        result=install_fault41_task_binding(worker,transport=NativePostwriteLibrary(native['path'],native['sha256']),
            prompt_token_ids=prompt,scope=scope,first_live_validation=True,group_contract=contract)
    elif contract['mechanism']=='39.output_proxy':
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
    from demoncat.runtime.vllm_ascend.worker_extension.rms18_tp2 import commit_rms18_tp2_role
    from demoncat.runtime.vllm_ascend.worker_extension.rope22_tp2 import commit_rope22_tp2_role
    from demoncat.runtime.vllm_ascend.worker_extension.output39_tp2 import commit_output39_tp2_role
    from demoncat.runtime.vllm_ascend.worker_extension.kv41_tp2 import commit_kv41_tp2_role
    fn={'41.tail_column':commit_kv41_tp2_role,'39.output_proxy':commit_output39_tp2_role,'18.api_half':commit_rms18_tp2_role,'22.wrong_row':commit_rope22_tp2_role}[decision['group']['mechanism']]
    return fn(actual_worker(wrapper),decision)

def remove(wrapper,mechanism):
    from demoncat.runtime.vllm_ascend.worker_extension.serving_tensor_binding import remove_tensor_serving_binding
    from demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance import remove_blue_acceptance_permit
    worker=actual_worker(wrapper)
    if mechanism=='41.tail_column':
        from demoncat.runtime.vllm_ascend.worker_extension.serving41 import remove_fault41_serving_binding
        result=remove_fault41_serving_binding(worker)
    elif mechanism=='39.output_proxy':
        from demoncat.runtime.vllm_ascend.worker_extension.serving39_output import remove_fault39_output_serving_binding
        result=remove_fault39_output_serving_binding(worker)
    else:result=remove_tensor_serving_binding(worker,mechanism)
    remove_blue_acceptance_permit(worker)
    return dict(result,worker=worker.demoncat_worker_identity()['worker'])

def trace(wrapper,action):
    from experiments.worker_core_v11_20261118.quant_health_observer import HealthHashes
    from demoncat.runtime.vllm_ascend.worker_extension.serving_context import discover_logits_boundary
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    w=actual_worker(wrapper);attribute='_dcat_kv41_tp2_trace'
    if action=='install':
        require(not hasattr(w,attribute),'TRACE_ALREADY_ACTIVE')
        r=w.model_runner;b=discover_logits_boundary(r.model,actual_logits_processor_type=LogitsProcessor,actual_vocab_size=r.vllm_config.model_config.hf_config.vocab_size)
        require(b['processor'].use_all_gather is True,'ALL_RANK_LOGITS_REQUIRED')
        o=HealthHashes(r.model,b['processor']);o.install();setattr(w,attribute,o);return dict(installed=True,rank=w.rank)
    o=getattr(w,attribute)
    if action=='clear':o.rows.clear();return dict(cleared=True,rank=w.rank)
    if action=='read':
        require([r['kind'] for r in o.rows]==['hidden','logits','hidden','logits'],'COMPLETE_PREFILL_DECODE_TRACE')
        return dict(rank=w.rank,rows=list(o.rows))
    require(action=='remove','TRACE_ACTION');o.close();delattr(w,attribute);return dict(rank=w.rank,owned_hooks=0)

def verify41(states,requests):
    selected=states[0]['multi_rank']['selected_rank'];out={}
    require(set(states)=={0,1} and selected in (0,1),'CURRENT_RANK_ROLES_REQUIRED')
    for rank,state in states.items():
        rows=state['evidence']['actual_requests'];require([r['phase'] for r in rows]==['normal','normal_repeat','lossless','fault','off','health'],'SIX_REAL_REQUESTS')
        n,_,_,f,o,h=rows
        require(n['model_output_sha256']==o['model_output_sha256']==h['model_output_sha256'],'ALL_RANK_EXACT_HIDDEN_RECOVERY')
        require(f['new_fault_hits']==int(rank==selected),'ONE_NATIVE_INJECTOR')
        if rank==selected:require(f['effect']['changed_elements']>0 and f['effect']['restored'] and f['effect']['released'],'ACTUAL_NATIVE_EFFECT')
        else:require(f['effect']['unchanged'] and f['effect']['native_writes']==0,'GUARD_CACHE_CHANGED')
        traces=[next(t for t in req['trace'] if t['rank']==rank)['rows'] for req in requests]
        require(traces[0]==traces[4]==traces[5],'ALL_RANK_FULL_HIDDEN_LOGITS_OFF_HEALTH')
        out[rank]=dict(native_hits=f['new_fault_hits'],hidden_changed=traces[0][2]!=traces[3][2],logits_changed=traces[0][3]!=traces[3][3],off_health_exact=True)
    require(out[1-selected]['hidden_changed'],'CROSS_RANK_PROPAGATION_NOT_OBSERVED')
    require([r['request_id'] for r in states[0]['evidence']['actual_requests']]==[r['request_id'] for r in states[1]['evidence']['actual_requests']],'RANK_REQUEST_IDS_DIFFER')
    return out

def main():
    from vllm import LLM,SamplingParams
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--mechanism',choices=['41.tail_column'],default='41.tail_column');parser.add_argument('--recover-after-fault',action='store_true');parser.add_argument('--native-library');parser.add_argument('--native-sha256');options=parser.parse_args();mechanism=options.mechanism
    native=dict(path=options.native_library,sha256=options.native_sha256)
    if mechanism in ('39.output_proxy','41.tail_column'):require(all(native.values()),'EXPLICIT_NATIVE_BUILD_REQUIRED')
    fault_id,variant=mechanism.split('.',1);status='TP2_'+{'18':'RMS18','22':'ROPE22','39':'OUTPUT39','41':'KV41'}[fault_id]+'_TWO_SCOPES_READY_FOR_INDEPENDENT_REVIEW'
    out=Path('/evidence/probe');out.mkdir()
    report=dict(status='STARTED',mechanism=mechanism,explicit_group_recovery=options.recover_after_fault,scopes=[],qualification_granted=False,combination='A2/0.18/CANN8.5.1/Qwen3-0.6B BF16/TP2',
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
        params=SamplingParams(temperature=0,max_tokens=2,ignore_eos=True);seen=[]
        for index in range(2):
            scope=dict(index=index,requests=[],status='STARTED');report['scopes'].append(scope);save()
            scope['trace_install']=llm.collective_rpc(trace,args=('install',),timeout=120)
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
                llm.collective_rpc(trace,args=('clear',),timeout=120)
                require(not llm.llm_engine.has_unfinished_requests(),'IDLE_QUEUE_REQUIRED')
                require(llm.reset_prefix_cache(reset_running_requests=False) is True,'CACHE_RESET_REQUIRED')
                pending=list(llm.enqueue([dict(prompt_token_ids=prompt)],params,use_tqdm=False))
                mapping=read_enqueued_identity(llm,pending,seen_internal_request_ids=seen);seen.extend(pending)
                require(len(pending)==1,'ONE_REQUEST_REQUIRED');state=action('advance',pending[0])
                require(state['next_phase']==phase,'GROUP_PHASE_CHANGED')
                results=llm.wait_for_completion(use_tqdm=False)
                require(len(results)==1 and results[0].request_id==mapping['external_request_id'],'COMPLETED_REQUEST_MAPPING')
                scope['requests'].append(dict(phase=phase,mapping=mapping,output=output(results[0]),trace=llm.collective_rpc(trace,args=('read',),timeout=120)));save()
                return action('query')
            try:
                for name in ('prepare','register','check','enable'):action(name)
                for phase in ('normal','normal_repeat','lossless','fault','off','health'):
                    state=request(phase)
                    if phase=='fault' and options.recover_after_fault:action('recover')
                require(state['complete'],'ALL_RANK_COMPLETE');scope['checks']=verify41(state['rank_states'],scope['requests'])
                scope['final_states']=state['rank_states'];final=action('finish')
                require(all(not any(s[k] for k in ('owned_hooks','leases','inflight','replay_pending')) for s in final['rank_states'].values()),'ALL_RANK_RESOURCE_ZERO')
                action('disable');action('unregister');action('detach')
                scope['remove']=llm.collective_rpc(remove,args=(mechanism,),timeout=120)
                scope['trace_remove']=llm.collective_rpc(trace,args=('remove',),timeout=120)
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
