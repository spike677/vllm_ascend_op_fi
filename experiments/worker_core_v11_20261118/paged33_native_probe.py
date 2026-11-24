"""Read-only native launch census; no fault binding, tensor export or grant."""
import hashlib,inspect,json,os
from pathlib import Path


def census(wrapper,action):
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    worker=wrapper.worker if isinstance(wrapper,WorkerWrapperBase) else wrapper
    import torch,torch_npu
    r=worker.model_runner
    if action=='start':
        from torch.utils._python_dispatch import TorchDispatchMode
        rows=[]
        def meta(x):
            if isinstance(x,torch.Tensor):
                return dict(shape=list(x.shape),dtype=str(x.dtype),stride=list(x.stride()),device=str(x.device))
            if isinstance(x,(tuple,list)):return [meta(t) for t in x]
            if isinstance(x,dict):return {k:meta(v) for k,v in x.items()}
            return x if x is None or type(x) in (str,int,float,bool) else type(x).__name__
        class Capture(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                kwargs=kwargs or {}
                out=func(*args,**kwargs)
                if any(t in str(func).lower() for t in ('attention','paged')):
                    rows.append(dict(op=str(func),args=meta(args),kwargs=meta(kwargs),output=meta(out),
                        request_ids=list(r.input_batch.req_ids)))
                return out
        mode=Capture();mode.__enter__()
        prof=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True,with_stack=False,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler('/evidence/native-profile-rank'+str(worker.rank)),
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level2,data_simplification=False))
        if os.environ.get('DCAT_EXTERNAL_MSPROF')=='1':
            class ExternalProfiler:
                def start(self):pass
                def stop(self):pass
            prof=ExternalProfiler()
        prof.start()
        real=torch_npu.npu_fused_infer_attention_score
        private=None
        if os.environ.get('DCAT_CAPTURE_LENGTHS'):
            from tools.paged33_private_capture import PrivateCapture
            private=PrivateCapture('/evidence/private-qkv-rank'+str(worker.rank),rank=worker.rank,
                lengths=json.loads(os.environ['DCAT_CAPTURE_LENGTHS']))
            worker._paged33_private_capture=private
        def attention(*args,**kwargs):
            out=real(*args,**kwargs)
            query=args[0] if args else kwargs.get('query')
            if private is not None:
                actual=dict(zip(('query','key','value'),args));actual.update(kwargs)
                private.observe(actual,out,list(r.input_batch.req_ids))
            elif os.environ.get('DCAT_CAPTURE_PRIVATE_QKV')=='1' and query is not None and query.shape[0]==1 and not getattr(worker,'_paged33_qkv_saved',False):
                q=query;keys=args[1] if len(args)>1 else kwargs['key'];values=args[2] if len(args)>2 else kwargs['value']
                key=keys if isinstance(keys,torch.Tensor) else keys[0]
                value=values if isinstance(values,torch.Tensor) else values[0]
                length=int(kwargs['actual_seq_lengths_kv'][0]);block=int(kwargs['block_size'])
                table=kwargs['block_table'][0,:((length+block-1)//block)].long()
                k=key.index_select(0,table).reshape(-1,kwargs['num_key_value_heads'],q.shape[-1])[:length]
                v=value.index_select(0,table).reshape(-1,kwargs['num_key_value_heads'],q.shape[-1])[:length]
                assert k.shape[0]==v.shape[0]==length, 'PRIVATE_QKV_LOGICAL_KEY_COUNT_MISMATCH'
                torch.save(dict(q=q.detach().cpu(),k=k.detach().cpu(),v=v.detach().cpu(),output=out[0].detach().cpu(),
                    metadata=dict(length=length,heads=kwargs['num_heads'],kv_heads=kwargs['num_key_value_heads'],scale=kwargs['scale'],request_ids=list(r.input_batch.req_ids))),
                    '/evidence/private-qkv-rank'+str(worker.rank)+'.pt')
                worker._paged33_qkv_saved=True
            rows.append(dict(op='DIRECT_npu_fused_infer_attention_score',args=meta(args),kwargs=meta(kwargs),output=meta(out),request_ids=list(r.input_batch.req_ids)))
            return out
        torch_npu.npu_fused_infer_attention_score=attention
        worker._paged33_function=(real,attention)
        worker._paged33_probe=(mode,prof,rows)
        sources=[]
        for name,module in r.model.named_modules():
            impl=getattr(module,'impl',None)
            if type(impl).__name__=='AscendAttentionBackendImpl':
                path=Path(inspect.getsourcefile(type(impl)))
                if not sources:
                    sources.append(dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                    Path('/evidence/attention_backend.py').write_bytes(path.read_bytes())
        return dict(rank=worker.rank,sources=sources)
    real,wrapped=worker._paged33_function
    assert torch_npu.npu_fused_infer_attention_score is wrapped
    torch_npu.npu_fused_infer_attention_score=real
    del worker._paged33_function
    mode,prof,rows=worker._paged33_probe
    torch.npu.synchronize();mode.__exit__(None,None,None);prof.stop()
    Path('/evidence/native-calls-rank'+str(worker.rank)+'.json').write_text(json.dumps(rows,indent=2))
    del worker._paged33_probe
    result=dict(rank=worker.rank,calls=len(rows),operators=sorted({r['op'] for r in rows}))
    if hasattr(worker,'_paged33_private_capture'):
        result['private_capture']=worker._paged33_private_capture.finish();del worker._paged33_private_capture
    return result


def main():
    from vllm import LLM,SamplingParams
    llm=LLM(model='/model',dtype='bfloat16',quantization='ascend',tensor_parallel_size=2,
        distributed_executor_backend='mp',enforce_eager=True,enable_prefix_caching=False,
        async_scheduling=False,enable_chunked_prefill=True,max_num_batched_tokens=4096,
        max_model_len=4096,max_num_seqs=1,gpu_memory_utilization=.8,seed=42,
        speculative_config=None,kv_transfer_config=None,trust_remote_code=True,
        limit_mm_per_prompt={'image':0,'video':0})
    tokens=llm.get_tokenizer().encode('Explain in one sentence why the sky appears blue.',add_special_tokens=True)
    tokens=(tokens*((2304+len(tokens)-1)//len(tokens)))[:2304]
    params=SamplingParams(temperature=0,max_tokens=4,min_tokens=4,ignore_eos=True)
    llm.generate([dict(prompt_token_ids=tokens)],params,use_tqdm=False)
    start=llm.collective_rpc(census,args=('start',),timeout=120)
    lengths=json.loads(os.environ.get('DCAT_CAPTURE_LENGTHS','[]'))
    if lengths:
        outputs=[]
        for length in lengths:
            prompt=(tokens*((length+len(tokens)-1)//len(tokens)))[:length-1]
            outputs.extend(llm.generate([dict(prompt_token_ids=prompt)],params,use_tqdm=False))
    else:outputs=llm.generate([dict(prompt_token_ids=tokens)],params,use_tqdm=False)
    end=llm.collective_rpc(census,args=('stop',),timeout=180)
    report=dict(status='NATIVE_PROFILE_CAPTURED_NOT_QUALIFICATION',start=start,end=end,
        token_ids=[o.outputs[0].token_ids for o in outputs],input_sha256=hashlib.sha256(bytes(str(tokens),'utf8')).hexdigest())
    Path('/evidence/native-launch.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
