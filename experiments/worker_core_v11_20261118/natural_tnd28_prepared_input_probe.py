"""One actual TND request, one compiled-input window, then unregistered health.

Trusted root deployment only. Uses the original public observe action and
original typed API. No replay/control environment, extra operator or F stage.
Unselected natural calls remain in the public inventory; no second census.
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
SUCCESS='ONE_NORMAL_INPUT_WINDOW_AND_UNREGISTERED_HEALTH_RECORDED_PENDING_REVIEW'


def need(ok,reason):
    if not ok:raise RuntimeError(reason)


def setup(worker,expected,tokens,provenance,settings):
    from demoncat.runtime.vllm_ascend.worker_extension.attention28_stock_census import StockTndPrefillCensus
    from demoncat.runtime.vllm_ascend.worker_extension.attention28_prepared_input import PreparedInputCapture
    from demoncat.runtime.vllm_ascend.worker_extension.serving_attention import install_attention_serving_binding
    worker=common.actual(worker)
    need(worker.demoncat_worker_identity().get('worker')==expected,'ACTUAL_WORKER_IDENTITY_CHANGED')
    capture=PreparedInputCapture(worker,build_completion=settings['build'],build_sha256=settings['build_sha256'],
        evidence_dir=settings['input_folder'])
    native=StockTndPrefillCensus(mechanism=MECHANISM,observer_receipt=settings['observer'],
        observer_receipt_sha256=settings['observer_sha256'],evidence_dir=settings['calls'],prepared_input_capture=capture)
    install_attention_serving_binding(worker,MECHANISM,prompt_token_ids=tokens,provenance=provenance,native=native)
    return dict(worker=expected,actual_pid=os.getpid(),worker_object_id=id(worker),model_object_id=id(worker.model_runner.model),
        build_sha256=settings['build_sha256'],one_current_object_only=True,qualification_granted=False)


def remove(worker,expected,settings):
    from demoncat.runtime.vllm_ascend.worker_extension.serving_attention import ATTRIBUTE
    from demoncat.runtime.vllm_ascend.worker_extension.attention28_prepared_input import PreparedInputCapture
    worker=common.actual(worker)
    need(worker.demoncat_worker_identity().get('worker')==expected,'ACTUAL_REMOVAL_WORKER_CHANGED')
    binding=getattr(worker,ATTRIBUTE,{}).get(MECHANISM)
    need(binding is not None,'OWNED_OBSERVATION_BINDING_MISSING')
    capture=binding.native.prepared_input_capture
    need(type(capture) is PreparedInputCapture and capture.worker is worker
        and capture.build_sha256==settings['build_sha256'] and str(capture.root)==settings['input_folder'],
        'FOREIGN_PREPARED_INPUT_BINDING')
    capture._check();need(capture.active is None and capture.refs is None,'PREPARED_INPUT_RESOURCES_REMAIN')
    result=common.remove(worker,expected,MECHANISM)
    return dict(removal=result,input_window=deepcopy(capture.receipt),first_error=capture.first_error)


def rpc(llm,callback,*args):
    values=llm.collective_rpc(callback,args=args,timeout=300)
    need(type(values) is list and len(values)==1,'ONE_STANDARD_WORKER_REQUIRED');return values[0]


def drive(llm,client,prompt,params,args,report,settings):
    from demoncat.runtime.vllm_ascend.worker_extension.serving_context import read_enqueued_identity
    folder=args.out/'worker_core_rpc';folder.mkdir();recorded=0;pending=[];seen=[]
    report['rpc']=[]
    def action(name,*values,expected='ACTION_COMPLETED'):
        nonlocal recorded
        reply=getattr(client,name)(*values)
        for item in reply['history'][recorded:]:
            path=folder/('%03d-%s.json'%(recorded,item['action']));common.save(path,item)
            report['rpc'].append(dict(path=str(path.relative_to(args.out)),sha256=common.sha(path),action=item['action']));recorded+=1
        common.save(args.out/'result.json',report)
        need(expected is None or reply['status']==expected,'PUBLIC_ACTION_FAILED:'+name+':'+reply['status'])
        return reply.get('response',{}).get('provider_state')
    def enqueue():
        need(not llm.llm_engine.has_unfinished_requests(),'UNRELATED_PENDING_REQUEST')
        need(llm.reset_prefix_cache(reset_running_requests=False) is True,'FRESH_CACHE_RESET_REQUIRED')
        ids=list(llm.enqueue([prompt],params,use_tqdm=False));need(len(ids)==1,'ONE_ACTUAL_REQUEST_REQUIRED')
        mapping=read_enqueued_identity(llm,ids,seen_internal_request_ids=seen);seen.extend(ids);return ids,mapping
    try:
        for name in ('prepare','register','check','enable'):action(name)
        pending,mapping=enqueue();state=action('advance',pending[0]);need(state['serving']['next_phase']=='observe','NORMAL_PHASE_ONLY')
        outputs=llm.wait_for_completion(use_tqdm=False)
        need(len(outputs)==1 and outputs[0].request_id==mapping['external_request_id'],'ACTUAL_REQUEST_MAPPING_CHANGED')
        action('query',expected='QUERY_OK');state=action('collect_replay')
        row=state['evidence']['requests'][0];calls=[c for f in row['forwards'] for c in f['calls']]
        need(row['request_id']==pending[0] and row['phase']=='observe' and len(calls)==28
            and all(c['original_calls']==1 and c['original_returned'] and c['inputs_unchanged'] for c in calls),
            'ONE_ACTUAL_REQUEST_FULL_NATURAL_DENOMINATOR_REQUIRED')
        need(sum('prepared_input_capture' in c.get('native_census',{}) for c in calls)==1,'ONE_ACTUAL_PREPARED_WINDOW_REQUIRED')
        report['request']=dict(mapping=mapping,worker=row,output_tokens=list(outputs[0].outputs[0].token_ids));pending=[]
        # The generic census finish requires two observations. This explicitly
        # ends the one-window diagnostic with disable, never pretends to finish
        # the provider's two-request qualification protocol.
        state=action('disable');report['cleanup']=state
        need(not any(state[k] for k in ('owned_hooks','owned_api_wrappers','inflight','leases','C1','C2','C3')),
            'PUBLIC_OWNED_RESOURCES_REMAIN')
        action('unregister');action('detach',expected='DETACHED')
        report['binding_removal']=rpc(llm,remove,client.snapshot()['worker'],settings)
        pending,mapping=enqueue();outputs=llm.wait_for_completion(use_tqdm=False)
        need(len(outputs)==1 and outputs[0].request_id==mapping['external_request_id'],'POST_UNREGISTER_REQUEST_MAPPING')
        report['health']=dict(mapping=mapping,output_tokens=list(outputs[0].outputs[0].token_ids),scope='POST_UNREGISTER_TOKENS_ONLY')
        need(report['health']['output_tokens']==report['request']['output_tokens'],'SAME_WORKER_TOKEN_HEALTH_CHANGED')
        pending=[];report['status']=SUCCESS
    except BaseException as error:
        report.update(status='REFUSED_OR_FAILED',error=repr(error),traceback=traceback.format_exc())
        if pending:
            try:llm.llm_engine.abort_request(pending,internal=True)
            except BaseException as secondary:report.setdefault('cleanup_errors',[]).append(repr(secondary))
    finally:
        snap=client.snapshot()
        if snap['ownership_retained']:
            # Uncertain prepare has no lease ticket; only public detach can
            # settle it. Do not send ticket-requiring disable/query first.
            actions=('disable','unregister','detach') if snap.get('ticket') else ('detach',)
            for name in actions:
                try:action(name,expected=None)
                except BaseException as error:report.setdefault('cleanup_errors',[]).append(repr(error))
        snap=client.snapshot()
        if not snap['ownership_retained'] and report.get('setup') and not report.get('binding_removal'):
            try:report['binding_removal']=rpc(llm,remove,snap['worker'],settings)
            except BaseException as error:report.setdefault('cleanup_errors',[]).append(repr(error))
        host=args.out/'host_receipt.json';common.save(host,snap)
        report['host_receipt']=dict(path=host.name,sha256=common.sha(host),bytes=host.stat().st_size)
        report['ownership_retained']=snap['ownership_retained'];common.save(args.out/'result.json',report)


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
    report=dict(schema='demoncat.tnd28.prepared_input_probe.v1',status='STARTED',mechanism=MECHANISM,
        probe_sha256=common.sha(__file__),build_sha256=args.build_sha256,source_manifest_sha256=args.source_manifest_sha256,
        C1=False,C2=False,C3=False,E5=False,fault_authorized=False,normal_stock_private_exact_proven=False,
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
        settings=dict(build=str(args.build),build_sha256=args.build_sha256,input_folder=str(args.out/'prepared_input'),
            observer=str(args.observer_receipt),observer_sha256=args.observer_receipt_sha256,calls=str(args.out/'native_calls'))
        provenance=dict(model=report['model'],standard_worker=True,probe_sha256=report['probe_sha256'],
            source_manifest_sha256=args.source_manifest_sha256,build_sha256=args.build_sha256,observer_only=True)
        def deploy(expected,config):
            need(config.mechanism==MECHANISM,'NORMAL_MECHANISM_CHANGED')
            report['setup']=rpc(llm,setup,expected,tokens,provenance,settings);return {}
        client=common.WorkerFaultClient(common.HostAdapter(llm,prepare_binding=deploy,timeout=300),common.FaultConfig(28,'fia.denominator_replay'))
        drive(llm,client,dict(prompt_token_ids=tokens),SamplingParams(temperature=0,max_tokens=1,ignore_eos=True),args,report,settings)
    except BaseException as error:
        report['status']='REFUSED_OR_FAILED'
        if 'error' in report:
            report.setdefault('post_protocol_errors',[]).append(dict(error=repr(error),traceback=traceback.format_exc()))
        else:report.update(error=repr(error),traceback=traceback.format_exc())
    common.save(args.out/'result.json',report)
    print(json.dumps({k:report[k] for k in ('status','error') if k in report}));return 0 if report['status']==SUCCESS else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','out','build','vendor','source','observer-receipt'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('build-sha256','source-manifest-sha256','observer-receipt-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--memory',type=float,default=.25);p.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args();need(args.out.resolve().is_relative_to('/evidence') and 0<args.memory<=.5,'OWNED_EVIDENCE_BOUNDED_MEMORY')
    return run(args,bootstrap(args))


if __name__=='__main__':raise SystemExit(main())
