"""Standard vLLM common Worker B/F/OFF with actual enqueue/scheduler IDs."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import traceback

PROMPT='Explain in one sentence why the sky appears blue during the day.'
MECHANISM='26.fia_destination'


def rpc(llm,method,*args):
    rows=llm.collective_rpc(method,args=args,timeout=300)
    if len(rows)!=1:raise RuntimeError('TP1_REQUIRED')
    return rows[0]


def setup(worker,tokens,baseline,statistics):
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    if isinstance(worker,WorkerWrapperBase):worker=worker.worker
    from demoncat.runtime.vllm_ascend.worker_extension import serving26
    p=Path(inspect.getfile(serving26)).resolve()
    if not p.is_relative_to('/installed'):raise RuntimeError('INSTALLED_WHEEL_REQUIRED')
    build=json.loads(Path('/native/build.json').read_bytes())
    name='libfault26_fia_writer_proxy_abi0.so'
    result=serving26.install_fia26_binding(worker,prompt=tokens,library='/native/'+name,
        sha256=build['files'][name],evidence_dir='/evidence/serving',baseline_control=baseline,
        collect_statistics=statistics)
    result['installed_source_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    result['model_class']=type(worker.model_runner.model).__module__+'.'+type(worker.model_runner.model).__name__
    result['model_type']=worker.model_runner.vllm_config.model_config.hf_config.model_type
    return result


def unset(worker):
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    if isinstance(worker,WorkerWrapperBase):worker=worker.worker
    from demoncat.runtime.vllm_ascend.worker_extension.serving26 import remove_fia26_binding
    return remove_fia26_binding(worker)


def output(row):
    return dict(request_id=row.request_id,tokens=list(row.outputs[0].token_ids),text=row.outputs[0].text)


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline-control',action='store_true')
    p.add_argument('--statistics',action='store_true');p.add_argument('--memory',type=float,default=.35)
    a=p.parse_args();report=dict(status='STARTED',phases=[],baseline_control=a.baseline_control,
        deployment=dict(num_gpu_blocks_override=64,reason='Fixed B/F cache capacity; not inferred from fluctuating free memory'))
    llm=None;ticket=tenant=None;seen=[]
    def save():Path('/evidence/result.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        from vllm import LLM,SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core,read_enqueued_identity
        from demoncat.runtime.vllm_ascend.worker_extension.serving26 import PHASES
        llm=LLM(model='/model',dtype='bfloat16',tensor_parallel_size=1,enforce_eager=True,
            max_model_len=128,max_num_seqs=1,max_num_batched_tokens=128,num_gpu_blocks_override=64,enable_prefix_caching=False,
            enable_chunked_prefill=False,async_scheduling=False,gpu_memory_utilization=a.memory,seed=42,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension',disable_log_stats=True)
        require_host_driven_core(llm)
        tokens=llm.get_tokenizer().encode(PROMPT,add_special_tokens=True)
        prompt=dict(prompt_token_ids=tokens);params=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True)
        healthy,=llm.generate([prompt],params,use_tqdm=False);report['healthy_before']=output(healthy)
        report['setup']=rpc(llm,setup,tokens,a.baseline_control,a.statistics)
        identity=rpc(llm,'demoncat_worker_identity');report['identity']=identity
        attached=rpc(llm,'demoncat_attach',dict(tenant_id='p4-fia26',worker=identity['worker']))
        if attached['status']!='ATTACHED':raise RuntimeError(str(attached))
        tenant=attached['ticket']
        prepared=rpc(llm,'demoncat_fault_rpc','prepare',dict(tenant,mechanism=MECHANISM,
            implementation_version='1',owner='fia26-natural-prefill'))
        report['prepare']=prepared;save()
        if prepared['status']!='ACTION_COMPLETED':raise RuntimeError(str(prepared))
        ticket=prepared['ticket']
        for action in ('register','check','execute'):
            report[action]=rpc(llm,'demoncat_fault_rpc',action,ticket);save()
            if report[action]['status']!='ACTION_COMPLETED':raise RuntimeError(str(report[action]))
        for phase in PHASES:
            if llm.llm_engine.has_unfinished_requests():raise RuntimeError('UNRELATED_PENDING_REQUEST')
            if llm.reset_prefix_cache(reset_running_requests=False) is not True:raise RuntimeError('RESET_NOT_CONFIRMED')
            ids=llm.enqueue([prompt],params,use_tqdm=False)
            mapping=read_enqueued_identity(llm,ids,seen_internal_request_ids=seen);seen+=ids
            row=dict(phase=phase,mapping=mapping);report['phases'].append(row)
            row['advance']=rpc(llm,'demoncat_fault_rpc','advance',dict(ticket,settings=dict(request_id=ids[0])));save()
            if row['advance']['status']!='ACTION_COMPLETED':raise RuntimeError(str(row['advance']))
            responses=llm.wait_for_completion(use_tqdm=False)
            if len(responses)!=1:raise RuntimeError('ONE_COMPLETED_REQUEST_REQUIRED')
            row['output']=output(responses[0])
            if row['output']['request_id']!=mapping['external_request_id']:raise RuntimeError('DRIVER_MAPPING_CHANGED')
            row['collect']=rpc(llm,'demoncat_fault_rpc','advance',dict(ticket,settings=dict(collect=True)));save()
            if row['collect']['status']!='ACTION_COMPLETED':raise RuntimeError('EXPLICIT_COLLECTION_FAILED')
            row['query']=rpc(llm,'demoncat_fault_query',ticket);save()
            if row['query'].get('quarantined'):raise RuntimeError('SERVING_REPLAY_FAILED')
        report['finish']=rpc(llm,'demoncat_fault_rpc','finish',ticket)
        if report['finish']['status']!='ACTION_COMPLETED':raise RuntimeError('RECOVERY_NOT_VERIFIED')
        report['status']='PASS_COMMON_FIA26_BFOFF'
    except BaseException as error:
        report.update(status='FAILED_WITH_EVIDENCE',error=repr(error),traceback=traceback.format_exc());save()
        if llm is not None and seen:
            try:llm.llm_engine.abort_request(seen,internal=True)
            except BaseException as error:report['abort_error']=repr(error)
    finally:
        if llm is not None and ticket is not None:
            try:
                report['unregister']=rpc(llm,'demoncat_fault_rpc','unregister',ticket)
                if report['unregister']['status']!='ACTION_COMPLETED':raise RuntimeError('OWNER_NOT_RELEASED')
                report['detach']=rpc(llm,'demoncat_detach',tenant)
                report['unset']=rpc(llm,unset)
                healthy,=llm.generate([prompt],params,use_tqdm=False)
                report['healthy_after']=output(healthy)
                if any(report['healthy_before'][k]!=report['healthy_after'][k] for k in ('tokens','text')):
                    raise RuntimeError('POST_CLEAN_GENERATION_CHANGED')
            except BaseException as error:report.update(status='FAILED_WITH_EVIDENCE',cleanup_error=repr(error))
        save()
    return 0 if report['status']=='PASS_COMMON_FIA26_BFOFF' else 2


if __name__=='__main__':raise SystemExit(main())
