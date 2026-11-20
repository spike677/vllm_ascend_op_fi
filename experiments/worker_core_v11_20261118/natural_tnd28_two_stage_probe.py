"""B3/B4/B5 research candidate, same standard Worker and two independent owners.

Real fault stages are explicitly identified. Public qualification remains closed.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'fault39/src'),str(ROOT/'experiments/c2c3_closure_20261117')]
from experiments.c2c3_closure_20261117 import natural_attention_worker_core_probe as common
from experiments.c2c3_closure_20261117 import review_tnd28_prepared_input as review

MECHANISM='28.fia.denominator_replay'
SUCCESS='TWO_SAME_WORKER_RESEARCH_SCOPES_RECORDED_PENDING_INDEPENDENT_REVIEW'


def save_record(path,value):
    # Preserve every field/value, avoid quadratic pretty-print indentation cost.
    # Each RPC still has its own original artifact and a complete host history.
    # The C encoder retains the identical JSON bytes while avoiding per-token
    # Python writes for large cumulative RPC histories. Replace atomically so
    # external interruption preserves the previous complete checkpoint.
    target=Path(path);temporary=target.with_name(target.name+'.partial')
    encoded=json.dumps(value,separators=(',',':'),allow_nan=False)+'\n'
    with temporary.open('w',encoding='utf-8',newline='\n') as stream:
        stream.write(encoded)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,target)



def need(ok,reason):
    if not ok:raise RuntimeError(reason)


def setup(worker,expected,tokens,provenance,settings):
    from experiments.worker_core_v11_20261118.tnd28_two_stage_research import TwoStageNativeResearch,install_research_binding
    worker=common.actual(worker);need(worker.demoncat_worker_identity()['worker']==expected,'ACTUAL_WORKER_CHANGED')
    native=TwoStageNativeResearch(worker,settings,mechanism=MECHANISM,observer_receipt=settings['observer'],
        observer_receipt_sha256=settings['observer_sha256'],evidence_dir=settings['calls'])
    install_research_binding(worker,prompt_token_ids=tokens,provenance=provenance,native=native)
    return dict(worker=expected,pid=os.getpid(),model_object_id=id(worker.model_runner.model),public_entry=False)


def remove(worker,expected):
    return common.remove(common.actual(worker),expected,MECHANISM)


def rpc(llm,callback,*args):
    rows=llm.collective_rpc(callback,args=args,timeout=300);need(type(rows) is list and len(rows)==1,'ONE_WORKER_REQUIRED')
    return rows[0]


def drive_scopes(llm,tokens,params,args,report,base_settings,provenance):
    from experiments.worker_core_v11_20261118.tnd28_two_stage_client import TwoStageResearchClient
    from experiments.worker_core_v11_20261118.tnd28_two_stage_research import TwoStageResearchProvider
    from demoncat.runtime.vllm_ascend.worker_extension.serving_context import read_enqueued_identity
    identities=llm.collective_rpc('demoncat_worker_identity');need(len(identities)==1,'ONE_WORKER_IDENTITY')
    worker=identities[0]['worker'];seen=[];report['scopes']=[];report['worker']=worker
    for index in range(2):
        folder=args.out/('scope%d'%index);folder.mkdir();settings=dict(base_settings,calls=str(folder/'native_calls'))
        scope=dict(index=index,status='STARTED',requests=[],setup=rpc(llm,setup,worker,tokens,provenance,settings))
        report['scopes'].append(scope);client=TwoStageResearchClient(llm,worker);pending=[];recorded=0
        def save():
            nonlocal recorded
            for entry in client.history[recorded:]:
                path=folder/('%03d-%s.json'%(recorded,entry['action']));save_record(path,entry);recorded+=1
            save_record(folder/'host_receipt.json',client.snapshot());save_record(args.out/'result.json',report)
        try:
            for action in ('prepare','register','check','enable'):state=client.call(action);save()
            need(state['research_request_plan']['phases']==list(TwoStageResearchProvider.phases),'RESEARCH_PLAN_DIFFERS')
            scope['request_plan']=state['research_request_plan']
            while state['serving']['next_phase']!='complete':
                phase=state['serving']['next_phase']
                need(len(scope['requests'])<len(TwoStageResearchProvider.phases),'RESEARCH_PLAN_OVERFLOW')
                need(not llm.llm_engine.has_unfinished_requests(),'UNRELATED_PENDING_REQUEST')
                need(llm.reset_prefix_cache(reset_running_requests=False) is True,'FRESH_PREFIX_CACHE_REQUIRED')
                pending=list(llm.enqueue([dict(prompt_token_ids=tokens)],params,use_tqdm=False));need(len(pending)==1,'ONE_REQUEST')
                mapping=read_enqueued_identity(llm,pending,seen_internal_request_ids=seen);seen.extend(pending)
                state=client.call('advance',pending[0]);save()
                output=llm.wait_for_completion(use_tqdm=False)
                need(len(output)==1 and output[0].request_id==mapping['external_request_id'],'ACTUAL_REQUEST_MAPPING')
                client.call('query');state=client.call('collect');save()
                current=state['evidence']['requests'][-1]
                need(current['phase']==phase and current['request_id']==pending[0],'ACTUAL_PHASE_MAPPING')
                scope['requests'].append(dict(phase=phase,mapping=mapping,output_tokens=list(output[0].outputs[0].token_ids)))
                pending=[];save()
            scope['complete']=client.call('finish');save()
            client.call('unregister');final=client.call('detach');save()
            need(not final['registered'] and not final['binding_reserved']
                and not any(final[k] for k in ('owned_hooks','owned_api_wrappers','inflight','leases')),'FINAL_RESOURCES_REMAIN')
            scope['removal']=rpc(llm,remove,worker);scope['resources_zero']=True;scope['status']='RESEARCH_SCOPE_COMPLETE_PENDING_REVIEW'
            need(llm.collective_rpc('demoncat_worker_identity')[0]['worker']==worker,'WORKER_CHANGED_BETWEEN_SCOPES')
            save()
        except BaseException as error:
            scope.update(status='REFUSED_OR_FAILED',error=repr(error))
            if pending:
                try:llm.llm_engine.abort_request(pending,internal=True)
                except BaseException as secondary:scope.setdefault('cleanup_errors',[]).append(repr(secondary))
            for action in ('disable','unregister','detach'):
                if not client.owned:break
                try:client.call(action)
                except BaseException as secondary:scope.setdefault('cleanup_errors',[]).append(repr(secondary))
            scope['ownership_retained']=client.owned;save();raise
    need(len({s['setup']['pid'] for s in report['scopes']})==1
        and len({s['setup']['model_object_id'] for s in report['scopes']})==1,'TWO_SCOPES_NOT_SAME_WORKER_MODEL')
    report['status']=SUCCESS


def bootstrap(args):
    build,files,_,_=review.bundle(args.build,args.build_sha256,args.vendor,args.source,args.source_manifest_sha256)
    need(common.sha(args.observer_receipt)==args.observer_receipt_sha256,'EXACT_OBSERVER_RECEIPT_REQUIRED')
    receipt=review.json_file(args.observer_receipt);library=receipt['observer_library']
    need(receipt['observer_compiled'] is True and receipt['actual_npu_calls']==0
        and receipt['observer_memory_selftest']['exit_code']==0 and common.sha(library['path'])==library['sha256'],
        'ACTUAL_COMPILED_RUNTIME_OBSERVER_REQUIRED')
    need(not any(k.startswith(('DCAT28_','DCAT33_','DCAT_F33_','FAULT28_','NES_FA_')) for k in os.environ),
        'CLEAN_NO_CONTROL_OR_REPLAY_ENVIRONMENT')
    if not args.child:
        need(not os.environ.get('LD_PRELOAD') and not os.environ.get('ASCEND_CUSTOM_OPP_PATH')
            and not any(k in sys.modules for k in ('torch','torch_npu','vllm')),'CLEAN_PREFRAMEWORK_PARENT')
        args.out.mkdir(parents=True,exist_ok=False);(args.out/'runtime').mkdir()
        env=dict(os.environ,LD_PRELOAD=library['path'],ASCEND_CUSTOM_OPP_PATH=str(args.vendor),
            DCAT_ATTN_OBSERVER_DIR=str(args.out/'runtime'),ACLNN_CACHE_LIMIT='0',TASK_QUEUE_ENABLE='0',
            PYTHONDONTWRITEBYTECODE='1',VLLM_ENABLE_V1_MULTIPROCESSING='0')
        os.execve(sys.executable,[sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:],'--child'],env)
    need(os.environ.get('LD_PRELOAD')==library['path'] and os.environ.get('ASCEND_CUSTOM_OPP_PATH')==str(args.vendor)
        and Path(os.environ['DCAT_ATTN_OBSERVER_DIR']).resolve()==(args.out/'runtime').resolve(), 'ACTUAL_STARTUP_ENVIRONMENT_CHANGED')
    return dict(vendor=str(args.vendor),artifacts=files,completion=build)


def run(args,bundle):
    report=dict(schema='demoncat.tnd28.two_stage_research_probe.v1',status='STARTED',mechanism=MECHANISM,
        probe_sha256=common.sha(__file__),build_sha256=args.build_sha256,source_manifest_sha256=args.source_manifest_sha256,
        C1=False,C2=False,C3=False,E5=False,public_fault_authorized=False,normal_stock_private_exact_proven=False,
        actual_cached_api_route_proven=False)
    try:
        from demoncat.runtime.vllm_ascend.worker_extension import paged33_loading
        active=paged33_loading.bootstrap(bundle);report['api_visibility_before_framework']=active.report()
        from vllm import LLM,SamplingParams
        from vllm.platforms import current_platform
        report['platform']=active.prepare_platform(current_platform)
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core
        cfg=review.json_file(args.model/'config.json')
        need(all(cfg.get(k)==v for k,v in dict(num_hidden_layers=28,num_attention_heads=16,num_key_value_heads=8,hidden_size=1024).items()),
            'SOURCE87_ACTUAL_MODEL_GEOMETRY_REQUIRED')
        weights={p.name:dict(sha256=common.sha(p),bytes=p.stat().st_size) for p in sorted(args.model.glob('*.safetensors'))}
        need(weights,'ACTUAL_MODEL_WEIGHTS_REQUIRED');report['model']=dict(config_sha256=common.sha(args.model/'config.json'),weights=weights)
        llm=LLM(model=str(args.model),dtype='float16',tensor_parallel_size=1,enforce_eager=True,max_model_len=2304,
            max_num_seqs=1,max_num_batched_tokens=2304,block_size=16,num_gpu_blocks_override=160,
            enable_prefix_caching=False,enable_chunked_prefill=False,async_scheduling=False,
            gpu_memory_utilization=args.memory,seed=42,disable_log_stats=True,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        require_host_driven_core(llm);report['api_visibility_after_model']=active.verify()
        words=llm.get_tokenizer().encode('Explain how attention reads a sequence of tokens. ',add_special_tokens=True)
        need(words,'ACTUAL_TOKENIZER_REQUIRED');tokens=(words*((2048+len(words)-1)//len(words)))[:2048]
        library=args.vendor/'op_api/lib/libcust_opapi.so'
        settings=dict(build=str(args.build),build_sha256=args.build_sha256,
            observer=str(args.observer_receipt),observer_sha256=args.observer_receipt_sha256,
            execute_borrow_library=str(library),execute_borrow_sha256=common.sha(library))
        provenance=dict(model=report['model'],standard_worker=True,probe_sha256=report['probe_sha256'],
            source_manifest_sha256=args.source_manifest_sha256,build_sha256=args.build_sha256,
            observer_only=False,research_only=True,qualification_granted=False)
        report['entry_protocol']='EXPLICIT_TWO_STAGE_RESEARCH_NOT_PUBLIC_RUNTIME'
        drive_scopes(llm,tokens,SamplingParams(temperature=0,max_tokens=1,ignore_eos=True),args,report,settings,provenance)
    except BaseException as error:
        report['status']='REFUSED_OR_FAILED'
        if 'error' in report:
            report.setdefault('post_protocol_errors',[]).append(dict(error=repr(error),traceback=traceback.format_exc()))
        else:report.update(error=repr(error),traceback=traceback.format_exc())
    save_record(args.out/'result.json',report)
    print(json.dumps({k:report[k] for k in ('status','error') if k in report}));return 0 if report['status']==SUCCESS else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','out','build','vendor','source','observer-receipt'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('build-sha256','source-manifest-sha256','observer-receipt-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--memory',type=float,default=.25);p.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args();need(args.out.resolve().is_relative_to('/evidence') and 0<args.memory<=.5,'OWNED_EVIDENCE_BOUNDED_MEMORY')
    return run(args,bootstrap(args))


if __name__=='__main__':raise SystemExit(main())
