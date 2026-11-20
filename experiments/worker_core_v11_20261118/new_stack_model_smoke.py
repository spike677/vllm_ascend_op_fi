"""Standard vLLM smoke only; no fault registration or deployment qualification."""
import argparse,hashlib,inspect,json,math,os,pathlib,traceback

def current_worker(wrapper):
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    worker=wrapper.worker if isinstance(wrapper,WorkerWrapperBase) else wrapper
    runner=worker.model_runner
    functions={}
    for name in ('execute_model','sample_tokens','_preprocess','_model_forward'):
        value=getattr(runner,name,None)
        if value is None:functions[name]={'missing':True};continue
        value=inspect.unwrap(getattr(value,'__func__',value));path=pathlib.Path(inspect.getsourcefile(value));functions[name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    fields={name:dict(present=hasattr(runner,name),value=getattr(runner,name,None)) for name in ('pcp_size','dcp_size','use_async_scheduling')}
    return dict(identity=worker.demoncat_worker_identity(),runner_class=type(runner).__module__+'.'+type(runner).__name__,fields=fields,functions=functions)

def main():
    p=argparse.ArgumentParser();p.add_argument('--tp',type=int,choices=[1,2],required=True);p.add_argument('--qwen38-community',action='store_true');a=p.parse_args()
    r=dict(status='STARTED',tp=a.tp,fault_calls=0,qualification_granted=False,model='/model',requests=[])
    try:
        from vllm import LLM,SamplingParams
        extra=dict(limit_mm_per_prompt={'image':0,'video':0},quantization='ascend',speculative_config=None,kv_transfer_config=None) if a.qwen38_community else {}
        llm=LLM(model='/model',dtype='bfloat16',tensor_parallel_size=a.tp,distributed_executor_backend='mp',
            enforce_eager=True,enable_prefix_caching=False,async_scheduling=False,enable_chunked_prefill=True,
            max_num_batched_tokens=1024,max_model_len=1024,max_num_seqs=1,gpu_memory_utilization=.8 if a.qwen38_community else .25,
            seed=42,disable_log_stats=True,worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension',**extra)
        r['pure_text_multimodal_limits']=extra.get('limit_mm_per_prompt');r['community_artifact_not_yellow_identity']=a.qwen38_community
        r['workers_before']=llm.collective_rpc(current_worker,timeout=60)
        assert len(r['workers_before'])==a.tp
        tokens=llm.get_tokenizer().encode('Explain in one sentence why the sky appears blue.',add_special_tokens=True)
        for i in range(2):
            assert llm.reset_prefix_cache(reset_running_requests=False) is True
            outputs=llm.generate([dict(prompt_token_ids=tokens)],SamplingParams(temperature=0,max_tokens=8,logprobs=1),use_tqdm=False)
            assert len(outputs)==1 and len(outputs[0].outputs)==1
            o=outputs[0].outputs[0]
            finite=all(math.isfinite(float(v.logprob)) for x in (o.logprobs or []) for v in x.values())
            assert o.token_ids and finite
            r['requests'].append(dict(index=i,request_id=outputs[0].request_id,prompt_tokens=tokens,generated_tokens=list(o.token_ids),text=o.text,returned_logprobs_finite=finite))
        assert r['requests'][0]['generated_tokens']==r['requests'][1]['generated_tokens']
        r['workers_after']=llm.collective_rpc(current_worker,timeout=60)
        assert r['workers_after']==r['workers_before']
        r['status']='STANDARD_MODEL_REPEAT_SMOKE_PASSED_NOT_FAULT_QUALIFIED'
    except BaseException as e:r.update(status='FAILED',error=repr(e),traceback=traceback.format_exc())
    pathlib.Path('/evidence/model-smoke.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));return 0 if r['status'].startswith('STANDARD_MODEL_REPEAT_SMOKE_PASSED') else 2
if __name__=='__main__':raise SystemExit(main())
