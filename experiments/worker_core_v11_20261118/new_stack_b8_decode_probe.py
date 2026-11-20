"""Private B8 prefill/first-decode mapping; no ordinary or Native fault qualification."""
import argparse,json,pathlib,traceback,uuid
from new_stack_model_smoke import current_worker

ATTRIBUTE='_dcat_newstack_b8_observer'
def observation_rpc(wrapper,action,owner,expected=None,profile=None):
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    from demoncat.runtime.vllm_ascend.worker_extension.v026_generation_observer import SourceBoundGenerationObserver
    worker=wrapper.worker if isinstance(wrapper,WorkerWrapperBase) else wrapper
    held=getattr(worker,ATTRIBUTE,None)
    if action=='install':
        assert held is None
        observer=SourceBoundGenerationObserver(worker.model_runner,expected,source_profile=profile)
        try:observer.install()
        except BaseException:
            observer.close();raise
        setattr(worker,ATTRIBUTE,(owner,observer))
        return observer.snapshot()
    if action=='remove' and held is None:
        return dict(status='NOT_INSTALLED',owned_hooks=0,owned_wrappers=0,inflight=0,qualification_granted=False)
    assert held is not None and held[0]==owner
    observer=held[1]
    if action=='remove':
        observer.close();result=observer.snapshot();delattr(worker,ATTRIBUTE);return result
    assert action=='query';return observer.snapshot()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['v026_installed_c9c9432b','preferred_a2_85a2bc5e'],required=True);a=parser.parse_args()
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
            max_num_batched_tokens=1024,max_model_len=1024,max_num_seqs=8,gpu_memory_utilization=.8,
            seed=42,disable_log_stats=True,limit_mm_per_prompt={'image':0,'video':0},quantization='ascend',
            speculative_config=None,kv_transfer_config=None,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        result['workers_before']=llm.collective_rpc(current_worker,timeout=60)
        tokens=llm.get_tokenizer().encode('Explain in one sentence why the sky appears blue.',add_special_tokens=True)
        for number in range(2):
            assert llm.reset_prefix_cache(reset_running_requests=False) is True
            llm.sleep(level=0,mode='wait')
            assert llm.llm_engine.is_sleeping() is True
            ids=list(llm.enqueue([dict(prompt_token_ids=tokens) for _ in range(8)],SamplingParams(temperature=0,max_tokens=2,ignore_eos=True),use_tqdm=False))
            assert len(ids)==8
            states=llm.llm_engine.output_processor.request_states
            assert set(states)==set(ids) and all(states[rid].request_id==rid for rid in ids)
            external={rid:states[rid].external_req_id for rid in ids}
            owner=uuid.uuid4().hex
            scope=dict(owner=owner,internal_request_ids=ids,external_request_id=external,prompt_tokens=tokens)
            result['scopes'].append(scope)
            scope['install']=llm.collective_rpc(observation_rpc,args=('install',owner,{rid:tokens for rid in ids},a.profile),timeout=60)
            llm.wake_up(tags=['scheduling'])
            assert llm.llm_engine.is_sleeping() is False
            outputs=llm.wait_for_completion(use_tqdm=False)
            assert len(outputs)==8 and {o.request_id for o in outputs}==set(external.values())
            scope['query']=llm.collective_rpc(observation_rpc,args=('query',owner),timeout=60)
            scope['remove']=llm.collective_rpc(observation_rpc,args=('remove',owner),timeout=60);owner=None
            scope['generated_tokens']=[list(o.outputs[0].token_ids) for o in outputs]
            assert len(scope['remove'])==2
            assert all(r['status']=='OBSERVED' and r['visual_execution_excluded'] and r['model_input_association_verified']
                and r['owned_hooks']==r['owned_wrappers']==r['inflight']==0 for r in scope['remove'])
        result['workers_after']=llm.collective_rpc(current_worker,timeout=60)
        assert result['workers_before']==result['workers_after']
        assert result['scopes'][0]['generated_tokens']==result['scopes'][1]['generated_tokens']
        result['status']='TWO_SAME_WORKER_B8_PREFILL_DECODE_OBSERVED_NOT_FAULT_QUALIFIED'
    except BaseException as e:
        result.update(status='FAILED',error=repr(e),traceback=traceback.format_exc())
        if owner is not None and llm is not None:
            try:result['failure_cleanup']=llm.collective_rpc(observation_rpc,args=('remove',owner),timeout=60)
            except BaseException as cleanup:result['failure_cleanup_error']=repr(cleanup)
    pathlib.Path('/evidence/b8-decode.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','error') if k in result}))
    return 0 if result['status'].startswith('TWO_SAME_WORKER') else 2
if __name__=='__main__':raise SystemExit(main())
