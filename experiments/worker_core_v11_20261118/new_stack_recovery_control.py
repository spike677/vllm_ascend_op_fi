"""No-fault exact-output controls across plain and RMS observation paths; private diagnostic only."""
import argparse,json,pathlib,traceback,uuid
from new_stack_model_smoke import current_worker

ATTRIBUTE='_dcat_newstack_recovery_control'

def observation_rpc(wrapper,action,owner,expected=None,profile=None):
    from types import SimpleNamespace as NS
    from copy import deepcopy
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    from demoncat.runtime.vllm_ascend.worker_extension.new_stack_request import NewStackTaskRequest
    from demoncat.runtime.vllm_ascend.worker_extension.rms18_request import _OneRMSReturn
    from demoncat.runtime.vllm_ascend.worker_extension.serving import complete_tensor
    worker=wrapper.worker if isinstance(wrapper,WorkerWrapperBase) else wrapper
    held=getattr(worker,ATTRIBUTE,None)
    if action=='install':
        assert held is None
        request_id,tokens=next(iter(expected.items()));runner=worker.model_runner
        rows=[];vectors={};errors=[]
        provider=NS(worker=worker,model=runner.model,binding=NS(prompt=tokens,execution_profile='v026_installed_c9c9432b',simulation=False),
            phase='normal' if profile=='baseline_observer' else 'off',requests=set(),target_module=None,target=None,
            injection_arm='baseline',_context=lambda:None,fail=lambda e:errors.append(repr(e)))
        def begin(row):provider.requests.add(row['request_id'])
        provider.begin_actual=begin;provider.end_actual=lambda row:rows.append(deepcopy(row))
        provider.admit=lambda *args:(_ for _ in ()).throw(AssertionError('NO_FAULT_ADMISSION_ALLOWED'))
        session=NewStackTaskRequest(provider,_OneRMSReturn);provider.session=session
        original=session.state_return
        def state_return(state):
            with session._diagnostics():
                for key,value in [('hidden',state.hidden_states),('logits',state.logits)]:
                    cpu,sha,finite=complete_tensor(value);assert finite
                    vectors[key]=cpu.clone()
            return original(state)
        session.state_return=state_return
        session.install();session.expect(request_id)
        setattr(worker,ATTRIBUTE,(owner,session,rows,vectors,errors,profile))
        return dict(status='READ_ONLY_CONTROL_INSTALLED',mode=profile,fault_calls=0)
    if action=='remove' and held is None:return dict(status='NOT_INSTALLED',owned_hooks=0,owned_wrappers=0,inflight=0)
    assert held is not None and held[0]==owner
    _,session,rows,vectors,errors,mode=held
    if action=='query':
        assert len(rows)==1 and not errors
        report=rows[0]['mechanism'];assert not report or not any(e.get('applied') for e in report['events'])
        history=getattr(worker,'_dcat_control_vectors',[])
        comparisons=[]
        for number,(previous_mode,previous) in enumerate(history):
            values={}
            for key,v in vectors.items():
                base=previous[key];assert v.shape==base.shape and v.dtype==base.dtype
                d=(v.double()-base.double()).abs()
                values[key]=dict(exact=bool(v.equal(base)),changed_elements=int((v!=base).sum()),max_abs=float(d.max()),elements=v.numel())
            comparisons.append(dict(previous=number,previous_mode=previous_mode,fields=values))
        history.append((mode,vectors));setattr(worker,'_dcat_control_vectors',history)
        return dict(mode=mode,row=rows[0],comparisons=comparisons,fault_calls=0,qualification_granted=False)
    assert action=='remove'
    session.close();r=session.readback();delattr(worker,ATTRIBUTE)
    return r


def main():
    a=argparse.Namespace(profile='v026_installed_c9c9432b')
    result=dict(status='STARTED',fault_calls=0,qualification_granted=False,scopes=[],source_profile=a.profile)
    llm=None;owner=None
    try:
        from vllm import LLM,SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.v026_text_observer import PROFILES
        import inspect,hashlib
        root=pathlib.Path(inspect.getsourcefile(LLM)).resolve().parents[1]
        rows=json.loads(PROFILES[a.profile].read_bytes())['files']
        result['host_scheduler_sources']=[]
        for name in ('entrypoints/llm.py','v1/engine/llm_engine.py','v1/engine/core_client.py','v1/engine/core.py'):
            raw=(root/name).read_bytes();sha=hashlib.sha256(raw).hexdigest()
            assert any(row['path']=='vllm/'+name and row['sha256']==sha for row in rows),'HOST_SCHEDULER_SOURCE_NOT_REVIEWED:'+name
            result['host_scheduler_sources'].append(dict(path='vllm/'+name,sha256=sha))
        llm=LLM(model='/model',dtype='bfloat16',tensor_parallel_size=2,distributed_executor_backend='mp',
            enforce_eager=True,enable_prefix_caching=False,async_scheduling=False,enable_chunked_prefill=True,
            max_num_batched_tokens=1024,max_model_len=1024,max_num_seqs=1,gpu_memory_utilization=.8,
            seed=42,disable_log_stats=True,limit_mm_per_prompt={'image':0,'video':0},quantization='ascend',
            speculative_config=None,kv_transfer_config=None,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        result['workers_before']=llm.collective_rpc(current_worker,timeout=60)
        tokens=llm.get_tokenizer().encode('The capital of France is',add_special_tokens=True)
        for number,mode in enumerate(('plain','baseline_observer','plain','baseline_observer','plain')):
            assert llm.reset_prefix_cache(reset_running_requests=False) is True
            llm.sleep(level=0,mode='wait')
            assert llm.llm_engine.is_sleeping() is True
            ids=list(llm.enqueue([dict(prompt_token_ids=tokens)],SamplingParams(temperature=0,max_tokens=1),use_tqdm=False))
            assert len(ids)==1
            states=llm.llm_engine.output_processor.request_states
            assert set(states)==set(ids) and states[ids[0]].request_id==ids[0]
            external=states[ids[0]].external_req_id
            owner=uuid.uuid4().hex
            scope=dict(mode=mode,owner=owner,internal_request_id=ids[0],external_request_id=external,prompt_tokens=tokens)
            result['scopes'].append(scope)
            scope['install']=llm.collective_rpc(observation_rpc,args=('install',owner,{ids[0]:tokens},mode),timeout=60)
            llm.wake_up(tags=['scheduling'])
            assert llm.llm_engine.is_sleeping() is False
            outputs=llm.wait_for_completion(use_tqdm=False)
            assert len(outputs)==1 and outputs[0].request_id==external
            scope['query']=llm.collective_rpc(observation_rpc,args=('query',owner),timeout=60)
            scope['remove']=llm.collective_rpc(observation_rpc,args=('remove',owner),timeout=60);owner=None
            scope['generated_tokens']=list(outputs[0].outputs[0].token_ids)
            assert len(scope['remove'])==2
            assert all(r['owned_hooks']==r['owned_wrappers']==r['inflight']==0 and not r['pending'] for r in scope['remove'])
            pathlib.Path('/evidence/recovery-control.json').write_text(json.dumps(result,indent=2)+'\n')
        result['workers_after']=llm.collective_rpc(current_worker,timeout=60)
        assert result['workers_before']==result['workers_after']
        assert result['scopes'][0]['generated_tokens']==result['scopes'][1]['generated_tokens']
        result['status']='FIVE_NO_FAULT_CONTROLS_RECORDED_NOT_QUALIFIED'
    except BaseException as e:
        result.update(status='FAILED',error=repr(e),traceback=traceback.format_exc())
        if owner is not None and llm is not None:
            try:result['failure_cleanup']=llm.collective_rpc(observation_rpc,args=('remove',owner),timeout=60)
            except BaseException as cleanup:result['failure_cleanup_error']=repr(cleanup)
    pathlib.Path('/evidence/recovery-control.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','error') if k in result}))
    return 0 if result['status'].startswith('FIVE_NO_FAULT') else 2
if __name__=='__main__':raise SystemExit(main())
