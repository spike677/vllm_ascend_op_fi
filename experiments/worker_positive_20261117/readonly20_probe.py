"""Read-only current-call/controlled-PTA path probe from installed wheels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import traceback
import faulthandler


def start(worker, operator):
    from fault_injection.deployments.pta_resolver import inspect_loaded_pta_resolution
    Path('/evidence/pta_resolution.json').write_text(json.dumps(inspect_loaded_pta_resolution(),indent=2)+'\n')
    from fault_injection.deployments.matmul import CtypesMatmulRuntime
    from fault_injection.native import NativeSession
    from demoncat.runtime.vllm_ascend.worker_extension.providers.fault20 import Fault20Provider
    identity=worker.demoncat_worker_identity()
    if identity['status']!='WORKER_IDENTIFIED':raise RuntimeError('WORKER_NOT_READY')
    facts=Fault20Provider(worker,worker.model_runner.model,identity['worker'])._worker_facts()
    folder=Path('/evidence')/('observer-rank'+str(worker.rank))
    runtime=CtypesMatmulRuntime('20.buffer_substitute',facts,folder,operator=operator)
    session=NativeSession(worker.model_runner.model,'20.buffer_substitute',arm='observe',runtime=runtime)
    session.__enter__()
    worker._worker_positive_readonly20=(session,runtime,folder)
    return dict(status='READONLY_INSTALLED',build_profile=runtime.build_profile,pta=runtime.pta_gate_status,
                pta_resolution=getattr(runtime,'pta_resolver_status',None))


def finish(worker):
    session,runtime,folder=worker._worker_positive_readonly20
    session.close()
    report=session.report();states=runtime.snapshot()
    path=folder.with_suffix('.json')
    path.write_text(json.dumps(dict(report=report,native=states,pta=runtime.pta_gate_status),indent=2)+'\n')
    events=[row for row in report['events'] if row.get('native_facts')]
    if any(row['native_registered'] or row['inflight'] or row['leases'] for row in states.values()):
        raise RuntimeError('NATIVE_DID_NOT_DRAIN')
    del worker._worker_positive_readonly20
    return dict(native_events=len(events),per_forward=dict(Counter(row['forward'] for row in events)),
                modules=len({row['path'] for row in events}),forwards=len(report['forwards']),
                pta=runtime.pta_gate_status,pta_resolution=getattr(runtime,'pta_resolver_status',None),
                report=str(path),native_idle=True)


def main():
    faulthandler.enable()
    p=argparse.ArgumentParser();p.add_argument('--model',default='/model');p.add_argument('--tp',type=int,default=1)
    p.add_argument('--memory',type=float,default=.35)
    p.add_argument('--operator',choices=['aclnnMatmulWeightNz','aclnnQuantMatmulWeightNz'],default='aclnnMatmulWeightNz')
    a=p.parse_args()
    from vllm import LLM,SamplingParams
    record=dict(status='STARTED',operator=a.operator,actual_fault_injection=False)
    try:
        llm=LLM(model=a.model,dtype='bfloat16',tensor_parallel_size=a.tp,enforce_eager=True,
            max_model_len=128,max_num_seqs=1,max_num_batched_tokens=128,enable_prefix_caching=False,
            enable_chunked_prefill=False,async_scheduling=False,gpu_memory_utilization=a.memory,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension',disable_log_stats=True)
        def request():
            row,=llm.generate(['Explain in one sentence why the sky appears blue during the day.'],
                             SamplingParams(temperature=0,max_tokens=1,ignore_eos=True),use_tqdm=False)
            return list(row.outputs[0].token_ids)
        record['before']=request()
        record['start']=llm.collective_rpc(start,args=(a.operator,),timeout=300)
        try:record['observed']=[request(),request()]
        finally:record['finish']=llm.collective_rpc(finish,timeout=300)
        record['after']=request()
        if any(row!=record['before'] for row in record['observed']+[record['after']]):
            raise RuntimeError('READONLY_MODEL_OUTPUT_CHANGED')
        record['status']='READONLY_DISCOVERY_COMPLETED_NOT_FAULT_ACCEPTANCE'
    except BaseException as error:
        record.update(status='FAILED',error=repr(error),traceback=traceback.format_exc())
    Path('/evidence/result.json').write_text(json.dumps(record,indent=2)+'\n')
    return 0 if record['status'].startswith('READONLY_DISCOVERY') else 2


if __name__=='__main__':raise SystemExit(main())
