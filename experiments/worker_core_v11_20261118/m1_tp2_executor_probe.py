"""Real TP2 executor and request observation only; no fault qualification."""
import hashlib,json,os,threading,traceback
from pathlib import Path


def find_logits_processor(model, base_type):
    processors=[module for module in model.modules() if isinstance(module,base_type)]
    if len(processors)!=1:raise RuntimeError('UNIQUE_ACTUAL_LOGITS_PROCESSOR_REQUIRED')
    return processors[0]


def worker_call(wrapper, action, owner):
    from experiments.c2c3_postwrite_20261117.natural41_probe import actual_worker
    from demoncat.runtime.vllm_ascend.worker_extension.multi_rank_prerequisites import tp2_eager_assessment
    from vllm.distributed import get_tp_group
    import torch
    worker=actual_worker(wrapper);runner=worker.model_runner;model=runner.model
    identity=worker.demoncat_worker_identity()
    assert identity['status']=='WORKER_IDENTIFIED',identity
    key='_dcat_m1_executor_observer'
    if action=='install':
        assert not hasattr(worker,key),'EXISTING_OBSERVER'
        report=tp2_eager_assessment(runner,get_tp_group(),worker.rank)
        assert report['status']=='matched',report
        state=dict(owner=owner,rows=[],handles=[],identity=identity['worker'],rpc_thread=threading.get_ident(),prerequisites=report)
        def capture(module,args,kwargs,output):
            def digest(tensor):
                if tensor is None:return None
                assert isinstance(tensor,torch.Tensor)
                data=tensor.detach().contiguous().cpu()
                return dict(shape=list(data.shape),dtype=str(data.dtype),
                    sha256=hashlib.sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest(),
                    finite=bool(torch.isfinite(data).all().item()))
            kind='hidden' if module is model else 'logits'
            state['rows'].append(dict(kind=kind,pid=os.getpid(),thread=threading.get_ident(),
                request_ids=list(runner.input_batch.req_ids),value=digest(output)))
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        processor=find_logits_processor(model,LogitsProcessor)
        state.update(processor_type=type(processor).__module__+'.'+type(processor).__name__,
                     use_all_gather=processor.use_all_gather)
        try:
            for module in (model,processor):
                state['handles'].append(module.register_forward_hook(capture,with_kwargs=True))
            setattr(worker,key,state)
        except BaseException:
            for handle in state['handles']:handle.remove()
            raise
    state=getattr(worker,key)
    assert state['owner']==owner and state['identity']==identity['worker'],'HELD_WORKER_CHANGED'
    result={k:v for k,v in state.items() if k!='handles'}
    result=dict(result,rank=worker.rank,pid=os.getpid(),rpc_thread=threading.get_ident(),owned_hooks=len(state['handles']))
    if action=='remove':
        for handle in state['handles']:handle.remove()
        assert all(h.id not in h.hooks_dict_ref() for h in state['handles']),'OWNED_HOOK_REMAINS'
        delattr(worker,key);result['owned_hooks']=0
    return result


def main():
    from vllm import LLM,SamplingParams
    from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter
    out=Path('/evidence/probe');out.mkdir()
    result=dict(status='STARTED',fault_injected=False,qualification_granted=False,scopes=[])
    def save():(out/'result.json').write_text(json.dumps(result,indent=2))
    try:
        llm=LLM(model='/model',dtype='bfloat16',load_format='safetensors',trust_remote_code=False,
            tensor_parallel_size=2,distributed_executor_backend='mp',enforce_eager=True,
            max_model_len=128,max_num_seqs=1,max_num_batched_tokens=128,num_gpu_blocks_override=64,
            enable_prefix_caching=False,enable_chunked_prefill=False,async_scheduling=False,
            gpu_memory_utilization=.35,seed=42,disable_log_stats=True,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        host=HostAdapter(llm,timeout=120)
        before=host.discover_workers();result['workers']=before;save()
        assert sorted(r['worker']['rank'] for r in before)==[0,1]
        assert len({r['worker']['pid'] for r in before})==2,'DISTINCT_WORKER_PROCESSES_REQUIRED'
        assert len({r['worker']['logical_device'] for r in before})==2,'DISTINCT_DEVICES_REQUIRED'
        tokens=llm.get_tokenizer().encode('The capital of France is',add_special_tokens=True)
        result['prompt_token_ids']=tokens
        for scope in range(2):
            owner='m1-executor-scope-'+str(scope)
            row=dict(owner=owner,requests=[]);result['scopes'].append(row);save()
            row['install']=llm.collective_rpc(worker_call,args=('install',owner),timeout=120)
            try:
                for _ in range(2):
                    outputs=llm.generate([dict(prompt_token_ids=tokens)],SamplingParams(temperature=0,max_tokens=1,ignore_eos=True),use_tqdm=False)
                    row['requests'].append([dict(id=r.request_id,tokens=list(r.outputs[0].token_ids),text=r.outputs[0].text) for r in outputs]);save()
                row['observations']=llm.collective_rpc(worker_call,args=('query',owner),timeout=120)
                for rank in row['observations']:
                    hidden=[r for r in rank['rows'] if r['kind']=='hidden']
                    logits=[r for r in rank['rows'] if r['kind']=='logits']
                    assert len(hidden)==len(logits)==2,'ALL_RANK_REAL_COMPLETION_REQUIRED'
                    assert hidden[0]['value']==hidden[1]['value'] and logits[0]['value']==logits[1]['value'],'REPEATED_BASELINE_CHANGED'
                    assert all(r['value'] is None or r['value']['finite'] for r in rank['rows'])
                request_maps=[[r['request_ids'] for r in rank['rows']] for rank in row['observations']]
                assert request_maps[0]==request_maps[1],'RANK_REQUEST_MAPPING_DIFFERS'
            finally:
                row['remove']=llm.collective_rpc(worker_call,args=('remove',owner),timeout=120);save()
            assert all(r['owned_hooks']==0 for r in row['remove'])
            assert host.discover_workers()==before,'WORKER_IDENTITY_CHANGED'
        result['status']='TP2_EXECUTOR_BASELINE_ONLY_PASSED_NOT_FAULT_ACCEPTANCE'
    except BaseException as e:
        result.update(status='FAILED',error=repr(e),traceback=traceback.format_exc())
    save();print(json.dumps(dict(status=result['status'],error=result.get('error'))),flush=True)
    return 0 if result['status'].startswith('TP2_EXECUTOR_BASELINE_ONLY_PASSED') else 2


if __name__=='__main__':raise SystemExit(main())
