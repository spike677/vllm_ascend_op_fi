"""Two independent ordinary KV41 scopes in one standard vLLM Worker.

No research-group route, injected scheduler, layer parameter or query collect.
Private control receipts stay under /evidence; no automatic statistics dump.
"""
import argparse
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import traceback
ROOT=Path('/source')
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'fault39/src'),str(ROOT)]
from experiments.c2c3_postwrite_20261117.natural41_probe import PROMPT,actual_worker,output
from experiments.c2c3_postwrite_20261117.natural41_coverage_probe import actual_vllm_weights
from experiments.c2c3_postwrite_20261117.model_identity import digest,model_source_identity


def setup(worker,identity,library,sha,build,tokens,weights,manifest,attempt):
    import torch
    import torch_npu
    from fault_injection.coverage_session import CoverageScope
    from fault_injection.deployments.postwrite import NativePostwriteLibrary
    from demoncat.runtime.vllm_ascend.worker_extension.serving41 import install_fault41_task_binding
    from demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance import install_blue_acceptance_permit
    from demoncat.runtime.vllm_ascend.worker_extension.deployment_authorization import configure_deployment_acceptance,authority_for
    from demoncat.runtime.vllm_ascend.worker_extension.providers.kv41_task import KV41TaskProvider
    worker=actual_worker(worker)
    if worker.demoncat_worker_identity()['worker']!=identity:raise RuntimeError('WORKER_CHANGED')
    src=Path(inspect.getfile(KV41TaskProvider)).resolve()
    if not src.is_relative_to(ROOT.resolve()):raise RuntimeError('ORDINARY_SOURCE_NOT_LOADED')
    if authority_for(worker) is None:
        configure_deployment_acceptance(worker,source='BLUE_COMBINATION',
            authorize_operator=lambda principal,operation:principal=='blue-harness')
    permit=install_blue_acceptance_permit(worker,mechanisms=['41.tail_column'],attempt=attempt,source_manifest_sha256=manifest)
    env=dict(torch=torch.__version__,torch_npu=torch_npu.__version__,worker=identity,
        hardware=torch.npu.get_device_name(0),native_sha256=sha,build_fingerprint=build)
    sources=model_source_identity(worker.model_runner.model)
    scope=CoverageScope('41.tail_column',digest(dict(prompt=tokens,max_tokens=2)),digest(env),manifest,
        'natural_worker',weights['weights_sha256'],sources['source_sha256'])
    bound=install_fault41_task_binding(worker,transport=NativePostwriteLibrary(library,sha),
        prompt_token_ids=tokens,scope=scope,first_live_validation=True)
    return dict(binding=bound,permit=permit,environment=env,model_source=sources,
        loaded_provider_path=str(src),loaded_provider_sha256=hashlib.sha256(src.read_bytes()).hexdigest())


def remove(worker):
    from demoncat.runtime.vllm_ascend.worker_extension.serving41 import remove_fault41_serving_binding
    from demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance import remove_blue_acceptance_permit
    worker=actual_worker(worker)
    result=remove_fault41_serving_binding(worker)
    remove_blue_acceptance_permit(worker)
    return result


def main():
    parser=argparse.ArgumentParser()
    for name in ('model','library','sha256','build-fingerprint','source-manifest-sha256','source-commit','attempt'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    out=Path('/evidence');result=dict(schema='demoncat.ordinary41.blue.v1',status='STARTED',
        source_commit=args.source_commit,source_manifest_sha256=args.source_manifest_sha256,
        native_sha256=args.sha256,build_fingerprint=args.build_fingerprint,scopes=[],
        public_qualification_granted=False,E5='NOT_ASSESSED',model_closed_claim=False)
    client=None
    def save(): (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    def rpc(fn,*a):
        rows=llm.collective_rpc(fn,args=a,timeout=300)
        if len(rows)!=1:raise RuntimeError('TP1_REQUIRED')
        return rows[0]
    try:
        from vllm import LLM,SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter,WorkerFaultClient
        from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core,read_enqueued_identity
        from demoncat.runtime.vllm_ascend.worker_extension.providers.kv41_task import PHASES
        result['weights']=actual_vllm_weights(args.model)
        llm=LLM(model=args.model,dtype='bfloat16',load_format='safetensors',trust_remote_code=False,
            tensor_parallel_size=1,enforce_eager=True,max_model_len=128,max_num_seqs=1,
            max_num_batched_tokens=128,num_gpu_blocks_override=64,enable_prefix_caching=False,
            enable_chunked_prefill=False,async_scheduling=False,gpu_memory_utilization=.35,
            seed=42,disable_log_stats=True,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        require_host_driven_core(llm)
        tokens=llm.get_tokenizer().encode(PROMPT,add_special_tokens=True)
        prompt=dict(prompt_token_ids=tokens);params=SamplingParams(temperature=0,max_tokens=2,ignore_eos=True)
        result['healthy_before']=output(llm.generate([prompt],params,use_tqdm=False)[0])
        def same_output(value):return all(value[k]==result['healthy_before'][k] for k in ('tokens','text'))
        for scope_index in range(2):
            scope=dict(index=scope_index,requests=[]);result['scopes'].append(scope)
            def bind(identity,config):
                scope['setup']=rpc(setup,identity,args.library,args.sha256,args.build_fingerprint,tokens,
                    result['weights'],args.source_manifest_sha256,args.attempt+'-scope'+str(scope_index))
                return {}
            client=WorkerFaultClient(HostAdapter(llm,prepare_binding=bind,timeout=300),FaultConfig(41,'tail_column'))
            def action(name,*values):
                receipt=getattr(client,name)(*values)
                (out/('scope'+str(scope_index)+'-host.json')).write_text(json.dumps(client.snapshot(),indent=2,allow_nan=False)+'\n')
                if receipt['status']!=('QUERY_OK' if name=='query' else 'DETACHED' if name=='detach' else 'ACTION_COMPLETED'):
                    raise RuntimeError('PUBLIC_ACTION_FAILED:'+name+':'+repr(receipt['response'])[:2000])
                return receipt['response'].get('provider_state')
            for name in ('prepare','register','check','enable'):action(name)
            scope['worker']=deepcopy(client.snapshot()['worker']);scope['owner']=client.snapshot()['owner'];save()
            seen=[]
            for phase in PHASES:
                if llm.llm_engine.has_unfinished_requests():raise RuntimeError('UNRELATED_REQUEST_ACTIVE')
                if llm.reset_prefix_cache(reset_running_requests=False) is not True:raise RuntimeError('FRESH_CACHE_UNCONFIRMED')
                ids=list(llm.enqueue([prompt],params,use_tqdm=False))
                mapping=read_enqueued_identity(llm,ids,seen_internal_request_ids=seen);seen+=ids
                if len(ids)!=1:raise RuntimeError('ONE_REQUEST_REQUIRED')
                state=action('advance',ids[0])
                if state['serving']['pending_request'][:2]!=[ids[0],phase]:raise RuntimeError('PHASE_MISMATCH')
                values=llm.wait_for_completion(use_tqdm=False)
                if len(values)!=1 or values[0].request_id!=mapping['external_request_id']:raise RuntimeError('REQUEST_MAPPING_CHANGED')
                value=output(values[0]);state=action('query')
                observed=state['evidence']['actual_requests']
                if not observed or observed[-1]['request_id']!=ids[0] or observed[-1]['phase']!=phase:
                    raise RuntimeError('ACTUAL_CALLBACK_COMPLETION_MISSING')
                scope['requests'].append(dict(phase=phase,mapping=mapping,output=value,worker=observed[-1]))
                if phase!='fault' and not same_output(value):raise RuntimeError('HEALTHY_TOKENS_OR_TEXT_CHANGED')
                save()
            scope['final_state']=state
            if not state['evidence']['matched_off_and_health_confirmed'] or state['owned_hooks'] or state['leases']:
                raise RuntimeError('RECOVERY_OR_RESOURCE_ZERO_FAILED')
            for name in ('finish','disable','unregister','detach'):action(name)
            scope['binding_removed']=rpc(remove)
            healthy=output(llm.generate([prompt],params,use_tqdm=False)[0])
            if not same_output(healthy) or client.host.route(client.config)!=scope['worker']:
                raise RuntimeError('POST_UNREGISTER_SAME_WORKER_HEALTH_FAILED')
            scope['healthy_after']=healthy;save()
        if result['scopes'][0]['owner']==result['scopes'][1]['owner'] or result['scopes'][0]['worker']!=result['scopes'][1]['worker']:
            raise RuntimeError('TWO_INDEPENDENT_OWNERS_SAME_WORKER_REQUIRED')
        if actual_vllm_weights(args.model)!=result['weights']:raise RuntimeError('MODEL_FILES_CHANGED')
        result['status']='TWO_ORDINARY_SCOPES_COMPLETE_PENDING_INDEPENDENT_REVIEW'
    except BaseException as error:
        result.update(status='FAILED',error=repr(error),traceback=traceback.format_exc())
        if client is not None:
            result['failed_host']=client.snapshot()
            # Do not repeat a failed F or turn container cleanup into recovery.
            try: result['recovery_request']=client.recover()
            except BaseException as cleanup:result['cleanup_error']=repr(cleanup)
    finally:save()
    print(json.dumps({k:result[k] for k in ('status','error') if k in result}))
    return 0 if result['status'].startswith('TWO_ORDINARY') else 2

if __name__=='__main__':raise SystemExit(main())
