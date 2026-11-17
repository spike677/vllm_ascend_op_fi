"""Natural TP1 inference admission; no Worker/runner method replacements.

This is a deployment probe, not a common Provider lifecycle acceptance.
The standard vLLM driver creates the temporary Worker. RPCs attach observers
and the existing native session between completed generate calls only.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import traceback

PROMPT='Explain in one sentence why the sky appears blue during the day.'


def actual_worker(worker):
    from vllm.v1.worker.worker_base import WorkerBase, WorkerWrapperBase
    if isinstance(worker, WorkerWrapperBase):worker=worker.worker
    if not isinstance(worker, WorkerBase):raise RuntimeError('ACTUAL_WORKER_REQUIRED')
    for obj in (worker,worker.model_runner):
        if any(name in vars(obj) for name in ('execute_model','_prepare_inputs')):
            raise RuntimeError('CORE_INSTANCE_OVERRIDE_REFUSED')
    return worker


def install(worker, tokens, arm, collect):
    import torch
    from fault_injection import attach
    from fault_injection.deployments.residual26 import Residual26Runtime
    from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_serial_eager
    from demoncat.runtime.vllm_ascend.worker_extension.call_binding import ForwardBinding
    worker=actual_worker(worker);runner=worker.model_runner;model=runner.model
    require_serial_eager(runner)
    build=json.loads(Path('/native/build.json').read_bytes())
    filename='libfault26_fia_writer_proxy_abi0.so'
    os.environ['FAULT26_FIA_WRITER_LOG_PATH']='/evidence/writer.jsonl'
    runtime=Residual26Runtime(model,'26.fia_destination',library='/native/'+filename,
        sha256=build['files'][filename],evidence_dir='/evidence/native',log_path='/evidence/writer.jsonl')
    binding=ForwardBinding(model)
    observed=[]
    def before(module,args,kwargs):
        from vllm.forward_context import get_forward_context
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import snapshot_fresh_prefill
        from demoncat.runtime.vllm_ascend.worker_extension.serving import complete_tensor
        ids=list(runner.input_batch.req_ids)
        if len(ids)!=1:raise RuntimeError('ONE_REQUEST_REQUIRED')
        actual=binding.bind(module,args,kwargs)
        snap=snapshot_fresh_prefill(runner,model,actual,get_forward_context(),
            expected_prompt_token_ids=tokens,expected_internal_request_id=ids[0])
        cpu,digest,_=complete_tensor(actual['input_ids'])
        if cpu.tolist()!=tokens:raise RuntimeError('DEVICE_INPUT_DIFFERS')
        snap['device_input_sha256']=digest
        observed.append(snap)
    guard=model.register_forward_pre_hook(before,prepend=True,with_kwargs=True)
    session=attach(model,'26.fia_destination',arm=arm,runtime=runtime,unsupported='passthrough')
    collector=None
    if collect:
        from capture import capture
        collector=capture('/evidence/capture')
        collector.start(model=model,rank_id=0)
    session.__enter__()
    worker._p4_probe=(runtime,session,guard,collector,observed)
    config=runner.vllm_config.model_config.hf_config
    return dict(status='DEPLOYMENT_PROBE_INSTALLED_NOT_COMMON_PROVIDER',
        actual_model_class=type(model).__module__+'.'+type(model).__name__,model_type=config.model_type,
        candidates=session.report()['candidates'],core_methods_replaced=False,
        native_sha256=build['files'][filename],model_forward_source=inspect.getsourcefile(model.forward))


def remove(worker):
    worker=actual_worker(worker)
    runtime,session,guard,collector,observed=worker._p4_probe
    if collector:collector.stop();collector.step()
    session.close();guard.remove()
    report=dict(coverage=session.report(),supported=runtime.supported,claims=runtime.claims,
        completed=runtime.completed,pending=runtime.pending(),error=runtime.error,scheduler=observed)
    Path('/evidence/provider.json').write_text(json.dumps(report,indent=2)+'\n')
    del worker._p4_probe
    return dict(supported=runtime.supported,claims=runtime.claims,completed=len(runtime.completed),
        pending=runtime.pending(),error=runtime.error,scheduler_snapshots=len(observed))


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['baseline','fault'],required=True)
    p.add_argument('--statistics',action='store_true');p.add_argument('--memory',type=float,default=.35)
    a=p.parse_args();report=dict(status='STARTED',arm=a.arm,scope='NATURAL_PRETRAINED_TP1_DEPLOYMENT_PROBE')
    try:
        from vllm import LLM,SamplingParams
        llm=LLM(model='/model',dtype='bfloat16',tensor_parallel_size=1,enforce_eager=True,
            max_model_len=128,max_num_seqs=1,max_num_batched_tokens=128,enable_prefix_caching=False,
            enable_chunked_prefill=False,async_scheduling=False,gpu_memory_utilization=a.memory,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension',
            disable_log_stats=True,seed=42)
        params=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True,logprobs=5)
        tokens=llm.get_tokenizer().encode(PROMPT,add_special_tokens=True)
        def generate():
            row,=llm.generate([dict(prompt_token_ids=tokens)],params,use_tqdm=False)
            out=row.outputs[0]
            return dict(tokens=list(out.token_ids),text=out.text,
                logprobs=[{str(k):v.logprob for k,v in r.items()} for r in out.logprobs])
        report['healthy']=generate()
        report['install'],=llm.collective_rpc(install,args=(tokens,a.arm,a.statistics))
        report['observed']=generate()
        report['native'],=llm.collective_rpc(remove)
        report['off']=generate()
        report.update(off_exact=report['healthy']==report['off'],changed=report['healthy']!=report['observed'])
        if not report['off_exact'] or report['native']['pending'] or report['native']['error']:
            raise RuntimeError('RECOVERY_FAILED')
        if a.arm=='baseline' and report['changed']:raise RuntimeError('BASELINE_NOT_LOSSLESS')
        report['status']='PASS_DEPLOYMENT_PROBE' if report['native']['supported'] else 'NO_ELIGIBLE_NATURAL_WRITER'
    except BaseException:
        report.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        Path('/evidence/result.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
