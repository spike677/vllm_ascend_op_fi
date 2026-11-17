"""Actual public enqueue -> common Worker -> B8 first-decode Q/scale evidence."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import traceback

PROMPTS=['Explain why the sky looks blue.','Name the capital city of France.',
    'Complete this sentence: Water freezes at','Write a Python function to add two integers.',
    'What is the result of seven multiplied by nine?','Describe a forest in spring.',
    'Translate good morning into Spanish.','Give one reason to read books.']
MECHANISM='quant.batch_shared_scale'


def rpc(llm,method,*args):
    result=llm.collective_rpc(method,args=args,timeout=300)
    if len(result)!=1:raise RuntimeError('TP1_REQUIRED')
    return result[0]


def setup(worker,prompts,baseline,statistics,capture_scope):
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    if isinstance(worker,WorkerWrapperBase):worker=worker.worker
    from demoncat.runtime.vllm_ascend.worker_extension import serving_quant
    source=Path(inspect.getfile(serving_quant)).resolve()
    if not source.is_relative_to('/installed'):raise RuntimeError('INSTALLED_WHEEL_REQUIRED')
    result=serving_quant.install_quant_binding(worker,prompts=prompts,root='/quant',
        build_sha256=hashlib.sha256(Path('/quant/runtime/build.json').read_bytes()).hexdigest(),
        evidence_dir='/evidence/serving',baseline_control=baseline,collect_statistics=statistics,capture_scope=capture_scope)
    result.update(model_class=type(worker.model_runner.model).__module__+'.'+type(worker.model_runner.model).__name__,
        model_type=worker.model_runner.vllm_config.model_config.hf_config.model_type,
        installed_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    return result


def unset(worker):
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    if isinstance(worker,WorkerWrapperBase):worker=worker.worker
    from demoncat.runtime.vllm_ascend.worker_extension.serving_quant import remove_quant_binding
    return remove_quant_binding(worker)


def output(row):return dict(request_id=row.request_id,tokens=list(row.outputs[0].token_ids),text=row.outputs[0].text)


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline-control',action='store_true')
    p.add_argument('--statistics',action='store_true');p.add_argument('--memory',type=float,default=.85)
    p.add_argument('--capture-scope',choices=['generation','first_decode'],default='generation');a=p.parse_args()
    result=dict(status='STARTED',baseline_control=a.baseline_control,phases=[],num_gpu_blocks_override=64,capture_scope=a.capture_scope)
    llm=None;ticket=tenant=None;seen=[]
    def save():Path('/evidence/result.json').write_text(json.dumps(result,indent=2)+'\n')
    try:
        from vllm import LLM,SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core,actual_driver_request_map
        from demoncat.runtime.vllm_ascend.worker_extension.serving_quant import PHASES
        llm=LLM(model='/model',dtype='bfloat16',tensor_parallel_size=1,enforce_eager=True,
            max_model_len=128,max_num_seqs=8,max_num_batched_tokens=256,num_gpu_blocks_override=64,
            enable_prefix_caching=False,enable_chunked_prefill=False,async_scheduling=False,
            gpu_memory_utilization=a.memory,seed=42,trust_remote_code=False,disable_log_stats=True,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        require_host_driven_core(llm)
        tokens=[llm.get_tokenizer().encode(x,add_special_tokens=True) for x in PROMPTS]
        prompts=[dict(prompt_token_ids=x) for x in tokens]
        params=SamplingParams(temperature=0,max_tokens=2,ignore_eos=True)
        result['healthy_before']=[output(x) for x in llm.generate(prompts,params,use_tqdm=False)]
        result['setup']=rpc(llm,setup,tokens,a.baseline_control,a.statistics,a.capture_scope)
        identity=rpc(llm,'demoncat_worker_identity');result['identity']=identity
        attach=rpc(llm,'demoncat_attach',dict(tenant_id='p4-quant',worker=identity['worker']))
        if attach['status']!='ATTACHED':raise RuntimeError(str(attach))
        tenant=attach['ticket']
        prepare=rpc(llm,'demoncat_fault_rpc','prepare',dict(tenant,mechanism=MECHANISM,implementation_version='1',owner='quant-first-decode'))
        result['prepare']=prepare;save()
        if prepare['status']!='ACTION_COMPLETED':raise RuntimeError(str(prepare))
        ticket=prepare['ticket']
        for action in ('register','check','execute'):
            result[action]=rpc(llm,'demoncat_fault_rpc',action,ticket);save()
            if result[action]['status']!='ACTION_COMPLETED':raise RuntimeError(str(result[action]))
        for phase in PHASES:
            if llm.llm_engine.has_unfinished_requests():raise RuntimeError('UNRELATED_REQUEST')
            if llm.reset_prefix_cache(reset_running_requests=False) is not True:raise RuntimeError('CACHE_RESET_UNCONFIRMED')
            ids=llm.enqueue(prompts,params,use_tqdm=False)
            mapping=actual_driver_request_map(llm.llm_engine.output_processor)
            if len(ids)!=8 or set(ids)!={x['internal_request_id'] for x in mapping} or any(x in seen for x in ids):
                raise RuntimeError('ACTUAL_ENQUEUE_MAPPING_FAILED')
            seen+=ids;row=dict(phase=phase,mapping=mapping,enqueued=ids);result['phases'].append(row)
            row['advance']=rpc(llm,'demoncat_fault_rpc','advance',dict(ticket,settings=dict(request_ids=ids)));save()
            if row['advance']['status']!='ACTION_COMPLETED':raise RuntimeError(str(row['advance']))
            if a.capture_scope=='first_decode':
                partial=llm.llm_engine.step()
                if any(x.finished for x in partial) or not llm.llm_engine.has_unfinished_requests():raise RuntimeError('EXPECTED_PREFILL_ONLY_ENGINE_STEP')
                row['prefill_step']=dict(public_engine_step=True,returned_partial_requests=len(partial),any_finished=False)
                row['capture_advance']=rpc(llm,'demoncat_fault_rpc','advance',dict(ticket,settings=dict(capture_decode=True)))
                if row['capture_advance']['status']!='ACTION_COMPLETED':raise RuntimeError(str(row['capture_advance']))
            responses=llm.wait_for_completion(use_tqdm=False);row['outputs']=[output(x) for x in responses]
            if len(responses)!=8 or {x.request_id for x in responses}!={x['external_request_id'] for x in mapping}:
                raise RuntimeError('COMPLETION_MAPPING_FAILED')
            row['query']=rpc(llm,'demoncat_fault_query',ticket);save()
            if row['query'].get('quarantined'):raise RuntimeError('REPLAY_QUARANTINED')
        result['finish']=rpc(llm,'demoncat_fault_rpc','finish',ticket)
        if result['finish']['status']!='ACTION_COMPLETED':raise RuntimeError('RECOVERY_NOT_VERIFIED')
        result['status']='PASS_COMMON_B8_QUANT_BFOFF'
    except BaseException as error:
        result.update(status='FAILED_WITH_EVIDENCE',error=repr(error),traceback=traceback.format_exc());save()
        if llm is not None and seen:
            try:llm.llm_engine.abort_request(seen,internal=True)
            except BaseException as error:result['abort_error']=repr(error)
    finally:
        if llm is not None and ticket is not None:
            try:
                result['unregister']=rpc(llm,'demoncat_fault_rpc','unregister',ticket)
                if result['unregister']['status']!='ACTION_COMPLETED':raise RuntimeError('OWNER_NOT_RELEASED')
                result['detach']=rpc(llm,'demoncat_detach',tenant);result['unset']=rpc(llm,unset)
                result['healthy_after']=[output(x) for x in llm.generate(prompts,params,use_tqdm=False)]
                if [x['tokens'] for x in result['healthy_after']]!=[x['tokens'] for x in result['healthy_before']]:
                    raise RuntimeError('POST_CLEAN_TOKENS_CHANGED')
            except BaseException as error:result.update(status='FAILED_WITH_EVIDENCE',cleanup_error=repr(error))
        save()
    return 0 if result['status']=='PASS_COMMON_B8_QUANT_BFOFF' else 2


if __name__=='__main__':raise SystemExit(main())
