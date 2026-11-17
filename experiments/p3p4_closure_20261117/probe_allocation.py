"""Observe real pretrained inference allocations without arming any fault."""
import argparse
import json
from pathlib import Path
import traceback


def install(worker):
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    from trace_attention_allocation import AttentionAllocationTrace
    if isinstance(worker,WorkerWrapperBase):worker=worker.worker
    for obj in (worker,worker.model_runner):
        if any(x in vars(obj) for x in ('execute_model','_prepare_inputs')):raise RuntimeError('CORE_OVERRIDE')
    trace=AttentionAllocationTrace(worker.model_runner.model);trace.install()
    worker._p4_allocation_trace=trace
    return dict(actual_model=type(worker.model_runner.model).__module__+'.'+type(worker.model_runner.model).__name__,
        sources=list(trace.sources.values()),fault_enabled=False)


def remove(worker):
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    if isinstance(worker,WorkerWrapperBase):worker=worker.worker
    report=worker._p4_allocation_trace.close();del worker._p4_allocation_trace
    Path('/evidence/allocation.json').write_text(json.dumps(report,indent=2)+'\n')
    return dict(allocations=report['allocation_count'],copies=report['copy_count'],error=report['error'],
        linked=sum(x['same_storage'] for x in report['copies']),
        initially_nan=sum(x['after_empty']['nan']>0 for x in report['allocations']),
        prewrite_nan=sum(x['before_copy']['nan']>0 for x in report['copies']),
        all_full_finite_copies=all(x['full_logical_overwrite'] and x['finite_after_copy'] and x['copy_matches_rhs'] for x in report['copies']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--memory',type=float,default=.35);a=p.parse_args()
    report=dict(status='STARTED',fault_enabled=False,scope='NEW_RUN_ALLOCATION_PROVENANCE_NOT_HISTORICAL_POINTER_PROOF')
    try:
        from vllm import LLM,SamplingParams
        llm=LLM(model='/model',dtype='bfloat16',tensor_parallel_size=1,enforce_eager=True,
            max_model_len=128,max_num_seqs=1,max_num_batched_tokens=128,enable_prefix_caching=False,
            enable_chunked_prefill=False,async_scheduling=False,gpu_memory_utilization=a.memory,seed=42,
            disable_log_stats=True)
        params=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True,logprobs=5)
        def generate():
            r,=llm.generate(['Explain in one sentence why the sky appears blue during the day.'],params,use_tqdm=False)
            o=r.outputs[0]
            return dict(tokens=list(o.token_ids),text=o.text,
                logprobs=[{str(k):v.logprob for k,v in row.items()} for row in o.logprobs])
        report['before']=generate();report['setup'],=llm.collective_rpc(install)
        report['observed']=[generate(),generate()]
        report['trace'],=llm.collective_rpc(remove)
        report['after']=generate()
        report['inference_unchanged']=all(x==report['before'] for x in report['observed']+[report['after']])
        if not report['trace']['copies'] or not report['inference_unchanged']:raise RuntimeError('OBSERVATION_INCOMPLETE_OR_CHANGED')
        report['status']='PASS_NEW_RUN_OBSERVATION'
    except BaseException:
        report.update(status='FAILED',error=traceback.format_exc());raise
    finally:Path('/evidence/result.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
