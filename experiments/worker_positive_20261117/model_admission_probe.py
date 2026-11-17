"""Real pretrained inference admission; explicitly no fault acceptance claim."""
import argparse
import faulthandler
import hashlib
import inspect
import json
from pathlib import Path
import signal
import traceback


def attach_observation(worker):
    from demoncat.runtime.vllm_ascend.worker_extension.serving_context import discover_logits_boundary
    from demoncat.runtime.vllm_ascend.worker_extension.serving import complete_tensor
    from demoncat_fault39.provider import CtypesProvider
    from fault_injection.model_discovery import inspect_structure
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    model=worker.model_runner.model
    config=worker.model_runner.vllm_config.model_config.hf_config
    bound=discover_logits_boundary(model,actual_logits_processor_type=LogitsProcessor,
                                   actual_vocab_size=config.vocab_size)
    provider=CtypesProvider.attach_preloaded()
    before=provider.snapshot()
    if before['enabled'] or before['owner'] or before['inflight'] or before['leases']:
        raise RuntimeError('NATIVE_NOT_IDLE')
    rows=[]
    def observe(module,args,output):
        cpu,digest,finite=complete_tensor(output)
        rows.append(dict(shape=list(cpu.shape),sha256=digest,finite=finite,
                         top1=cpu.float().argmax(-1).tolist()))
    handle=bound['processor'].register_forward_hook(observe)
    worker._admission_owned=(handle,rows,provider,before)
    source=Path(inspect.getsourcefile(type(model)))
    return dict(worker=worker.demoncat_worker_identity(),actual_model_class=type(model).__module__+'.'+type(model).__name__,
                actual_model_type=config.model_type,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                boundary=bound['report'],structure=inspect_structure(model),native_before=before,
                loaded_from_pretrained=True,actual_fault_injection=False)


def detach_observation(worker):
    handle,rows,provider,before=worker._admission_owned
    handle.remove()
    after=provider.snapshot()
    del worker._admission_owned
    if after['enabled'] or after['owner'] or after['inflight'] or after['leases']:
        raise RuntimeError('NATIVE_STATE_CHANGED_IN_READONLY_ADMISSION')
    return dict(complete_logits=rows,native_idle=True,
                actual_matmul_runs=after['interposed_run_calls']-before['interposed_run_calls'],
                build_profile_sha256=after['build_profile_sha256'],actual_fault_injection=False)


def main():
    faulthandler.enable();faulthandler.register(signal.SIGUSR1,all_threads=True)
    parser=argparse.ArgumentParser();parser.add_argument('--memory',type=float,default=.85)
    args=parser.parse_args();path=Path('/evidence/result.json')
    result=dict(status='STARTING',acceptance='PRETRAINED_NPU_INFERENCE_ONLY',fault_injection=False)
    def save():path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    save()
    try:
        from vllm import LLM,SamplingParams
        llm=LLM(model='/model',dtype='bfloat16',tensor_parallel_size=1,enforce_eager=True,
                max_model_len=128,max_num_seqs=1,max_num_batched_tokens=128,enable_prefix_caching=False,
                enable_chunked_prefill=False,async_scheduling=False,gpu_memory_utilization=args.memory,
                worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension',
                disable_log_stats=True,trust_remote_code=False)
        result['loaded']=llm.collective_rpc(attach_observation,timeout=300);save()
        outputs=[]
        try:
            for _ in range(2):
                rows=llm.generate(['Explain in one sentence why the sky appears blue.'],
                                  SamplingParams(temperature=0,max_tokens=4,ignore_eos=True),use_tqdm=False)
                outputs.append([dict(request_id=r.request_id,prompt_token_ids=r.prompt_token_ids,
                                     tokens=list(r.outputs[0].token_ids),text=r.outputs[0].text) for r in rows])
        finally:result['observation']=llm.collective_rpc(detach_observation,timeout=300)
        result['requests']=outputs
        if outputs[0][0]['tokens']!=outputs[1][0]['tokens']:
            raise RuntimeError('REPEATED_NORMAL_GENERATION_CHANGED')
        if not all(r['finite'] for item in result['observation'] for r in item['complete_logits']):
            raise RuntimeError('NONFINITE_NORMAL_LOGITS')
        result['status']='PASS_PRETRAINED_NPU_INFERENCE_NOT_FAULT_ACCEPTANCE'
    except BaseException as error:
        result.update(status='FAILED_WITH_EVIDENCE',error=repr(error),traceback=traceback.format_exc())
    save()
    return 0 if result['status'].startswith('PASS_') else 2


if __name__=='__main__':raise SystemExit(main())
