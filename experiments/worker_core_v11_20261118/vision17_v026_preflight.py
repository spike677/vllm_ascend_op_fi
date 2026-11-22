"""Read-only TP2 visual preflight. No injection, binding or qualification."""
import json,time,traceback,hashlib
from pathlib import Path


def observe(wrapper,action):
    import inspect,torch
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    from torch.utils._python_dispatch import TorchDispatchMode,_disable_current_modes
    from fault_injection.sites.mla28 import fingerprint
    worker=wrapper.worker if isinstance(wrapper,WorkerWrapperBase) else wrapper
    runner=worker.model_runner;model=runner.model
    if action=='install':
        assert not hasattr(worker,'_vision17_preflight')
        visual=model.visual;proj=visual.patch_embed.proj
        state=dict(rank=worker.rank,events=[],outputs=[],handles=[],inside=False,
            weight=fingerprint(proj.weight),kernel=list(proj.kernel_size),stride=list(proj.stride),
            padding=list(proj.padding),use_data_parallel=model.use_data_parallel,
            sources={})
        for name,fn in [('patch',visual.patch_embed.forward),('projection',proj.forward),
                        ('encoder',visual.forward),('runner',runner._preprocess)]:
            p=Path(inspect.getsourcefile(fn));state['sources'][name]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                kwargs=kwargs or {};result=func(*args,**kwargs)
                if 'conv' in func._schema.name:
                    with _disable_current_modes():
                        values={a.name:a.default_value for a in func._schema.arguments};values.update(zip((a.name for a in func._schema.arguments),args));values.update(kwargs)
                        state['events'].append(dict(operator=func._schema.name,
                            padding=list(values.get('padding',[])),stride=list(values.get('stride',[])),
                            output=fingerprint(result) if isinstance(result,torch.Tensor) else None))
                return result
        mode=Mode()
        def enter(module,args,kwargs):
            assert not state['inside'];state['inside']=True
            state['events'].append(dict(encoder_enter=True,request_ids=list(runner.input_batch.req_ids),
                inputs=[fingerprint(x) for x in args if isinstance(x,torch.Tensor)]))
            mode.__enter__()
        def leave(module,args,value):
            mode.__exit__(None,None,None);state['inside']=False
            state['events'].append(dict(encoder_return=True,output=fingerprint(value)))
        state['handles']=[visual.register_forward_pre_hook(enter,with_kwargs=True),visual.register_forward_hook(leave)]
        assert 'execute_model' not in vars(runner)
        original=runner.execute_model
        def execute(*args,**kwargs):
            result=original(*args,**kwargs);current=runner.execute_model_state
            if current is not None:
                state['outputs'].append(dict(hidden=fingerprint(current.hidden_states),logits=fingerprint(current.logits),
                    finite=bool(torch.isfinite(current.hidden_states).all().item() and torch.isfinite(current.logits).all().item())))
            return result
        runner.execute_model=execute;state['execute']=execute;worker._vision17_preflight=state
        return dict(rank=worker.rank,kernel=state['kernel'],stride=state['stride'],padding=state['padding'],
            weight=state['weight'],use_data_parallel=state['use_data_parallel'],sources=state['sources'])
    state=worker._vision17_preflight
    if action=='collect':
        assert not state['inside']
        result=dict(rank=worker.rank,events=state['events'],outputs=state['outputs'],fault_calls=0)
        state['events']=[];state['outputs']=[];return result
    if action=='remove':
        assert not state['inside'] and runner.execute_model is state['execute']
        del runner.execute_model
        for h in state['handles']:h.remove()
        del worker._vision17_preflight
        return dict(rank=worker.rank,removed=True)
    raise ValueError(action)


def main():
    report=dict(status='STARTED',fault_calls=0,qualification_granted=False,requests=[])
    out=Path('/evidence/vision17_preflight.json');assert not out.exists()
    def save():out.write_text(json.dumps(report,indent=2)+'\n')
    try:
        from vllm import LLM,SamplingParams
        from PIL import Image
        llm=LLM(model='/model',dtype='bfloat16',quantization='ascend',tensor_parallel_size=2,
            distributed_executor_backend='mp',enforce_eager=True,enable_prefix_caching=False,
            async_scheduling=False,enable_chunked_prefill=True,max_num_batched_tokens=1024,
            max_model_len=1024,max_num_seqs=1,gpu_memory_utilization=.8,seed=42,
            speculative_config=None,kv_transfer_config=None,trust_remote_code=True,
            limit_mm_per_prompt={'image':1,'video':0},mm_processor_cache_gb=0,mm_encoder_tp_mode='weights',
            mm_processor_kwargs={'min_pixels':4096,'max_pixels':16384},
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        image=Image.new('RGB',(64,64));image.putdata([(x*4,y*4,((x+y)%64)*4) for y in range(64) for x in range(64)])
        prompt='<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image in one sentence.<|im_end|>\n<|im_start|>assistant\n'
        report['identity']=llm.collective_rpc(observe,args=('install',));save()
        params=SamplingParams(temperature=0,max_tokens=4,min_tokens=4,ignore_eos=True)
        for i in range(6):
            assert llm.reset_prefix_cache(reset_running_requests=False)
            # New MM identity on each request defeats scheduler/encoder reuse;
            # identical image bytes preserve the numerical input.
            start=time.monotonic()
            output=llm.generate([dict(prompt=prompt,multi_modal_data={'image':image},
                multi_modal_uuids={'image':'vision17-preflight-'+str(i)})],params,use_tqdm=False)
            rows=llm.collective_rpc(observe,args=('collect',))
            assert all(len(r['outputs'])==4 and all(x['finite'] for x in r['outputs']) for r in rows),'VISUAL_BASELINE_NONFINITE_OR_MISSING_STEPS'
            assert all(sum(e.get('encoder_enter',False) for e in r['events'])==1 for r in rows),'ENCODER_NOT_EXECUTED_ON_EACH_RANK'
            report['requests'].append(dict(seconds=time.monotonic()-start,ranks=rows,tokens=list(output[0].outputs[0].token_ids)));save()
            if i>=1 and [r['outputs'] for r in rows]==[r['outputs'] for r in report['requests'][-2]['ranks']]:
                report['baseline_stable']=True;break
        else:raise RuntimeError('VISUAL_BASELINE_NOT_STABLE_IN_SIX')
        report['cleanup']=llm.collective_rpc(observe,args=('remove',));report['status']='VISUAL_PREFLIGHT_PASSED_NOT_FAULT_ACCEPTANCE'
    except BaseException as e:report.update(status='FAILED',error=repr(e),traceback=traceback.format_exc())
    save();print(json.dumps(dict(status=report['status'],error=report.get('error'))),flush=True)
    return 0 if report['status'].startswith('VISUAL_PREFLIGHT_PASSED') else 2


if __name__=='__main__':raise SystemExit(main())
