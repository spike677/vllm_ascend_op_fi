"""Two natural stock TND requests with a pre-LLM read-only input journal.

Same-process standard Worker only. No private operator package, F, global model
wrapper, Slice decoder or qualification. Runtime launch scopes remain original.
"""
import argparse
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'fault39/src')]
from experiments.c2c3_closure_20261117 import natural_attention_worker_core_probe as common
from experiments.worker_core_v11_20261118 import prepare_natural_attention28_input_journal as preparation

CANONICAL='experiments.worker_core_v11_20261118.natural_attention28_input_journal_probe'
MECHANISM='28.fia.denominator_replay'
SUCCESS='NATURAL_STOCK_FIA_INPUT_JOURNAL_RECORDED_PENDING_INDEPENDENT_REVIEW'
_HELD=None
_CENSUS=None
_SOURCE=None
_CLEANUP_ATTEMPTS=0
_BINDING_INSTALLED=False
_BINDING_REMOVED=False


def need(ok,reason):
    if not ok:raise RuntimeError(reason)


def immutable(folder,name,value):
    folder.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    path=folder/name
    with path.open('xb') as stream:stream.write(raw)
    return dict(path=str(path),sha256=common.sha(path),bytes=len(raw))


def note_failure(report,error,stage):
    """Keep the original API/protocol failure when later evidence writes fail."""
    report['status']='REFUSED_OR_FAILED'
    if 'error' not in report:
        report.update(error=repr(error),traceback=traceback.format_exc(),error_stage=stage)
    else:
        report.setdefault('secondary_errors',[]).append(dict(stage=stage,error=repr(error),traceback=traceback.format_exc()))


def held_source(expected):
    # A serialized RPC cannot reconstruct an observer or create a new factory.
    # It must find the object already held by the canonical module in this PID.
    module=importlib.import_module(CANONICAL)
    need(Path(module.__file__).resolve()==Path(expected['path']).resolve()
        and common.sha(module.__file__)==expected['sha256'] and module._SOURCE==expected,
        'EXACT_HELD_PROBE_SOURCE_REQUIRED')
    from fault_injection.fia_input_journal import NativeFiaInputJournal, process_identity
    need(type(module._HELD) is NativeFiaInputJournal and module._HELD.startup is not None
        and module._HELD.startup['identity']==process_identity(),'SAME_PRE_LLM_PROCESS_AND_THREAD_REQUIRED')
    return module,module._HELD


def prepare_startup(args):
    global _HELD,_SOURCE
    from fault_injection.fia_input_journal import NativeFiaInputJournal
    need(_HELD is None,'ONE_PRESTART_OBJECT_REQUIRED')
    here=sys.modules[__name__];prior=sys.modules.get(CANONICAL)
    need(prior is None or prior is here,'CANONICAL_PROBE_MODULE_ALREADY_OWNED')
    sys.modules[CANONICAL]=here
    _SOURCE=dict(path=str(Path(__file__).resolve()),sha256=common.sha(__file__))
    _HELD=NativeFiaInputJournal(args.journal,args.journal_build_sha256)
    startup=_HELD.preflight()
    return dict(source=deepcopy(_SOURCE),native=startup,
        original=immutable(args.out/'input_journal','startup-before-llm.json',startup))


def bind_worker(worker,source):
    module,journal=held_source(source);worker=common.actual(worker)
    observed=worker.demoncat_worker_identity()
    need(observed.get('status')=='WORKER_IDENTIFIED' and not observed.get('quarantined'), 'ACTUAL_WORKER_IDENTITY_REQUIRED')
    model=worker.model_runner.model
    need(not model.training,'ACTUAL_INFERENCE_MODEL_REQUIRED')
    journal.bind_worker(pid=os.getpid(),process_start_identity=journal.startup['identity']['process_start_identity'],
        worker_object_id=id(worker),model_object_id=id(model))
    return dict(worker=observed['worker'],native=deepcopy(journal.bound_worker),source=source,
        actual_device=dict(type=getattr(worker.device,'type',None),index=getattr(worker.device,'index',None)),
        exact_model_object=True,qualified=False)


def verify_worker(worker,expected,source):
    module,journal=held_source(source);worker=common.actual(worker)
    need(worker.demoncat_worker_identity().get('worker')==expected and journal.bound_worker is not None
        and journal.bound_worker['worker_object_id']==id(worker)
        and journal.bound_worker['model_object_id']==id(worker.model_runner.model),'HELD_ACTUAL_WORKER_OR_MODEL_CHANGED')
    return module,journal,worker


def setup(worker,expected,tokens,provenance,observer,source):
    from demoncat.runtime.vllm_ascend.worker_extension.attention28_stock_census import StockTndPrefillCensus
    from demoncat.runtime.vllm_ascend.worker_extension.serving_attention import install_attention_serving_binding
    module,journal,worker=verify_worker(worker,expected,source)
    need(module._CENSUS is None,'ONE_ACTUAL_CENSUS_REQUIRED')
    journal.begin(max_calls=128,max_pairs=64,max_windows=64,max_bytes=16*1024*1024)
    native=StockTndPrefillCensus(mechanism=MECHANISM,observer_receipt=observer['path'],
        observer_receipt_sha256=observer['sha256'],evidence_dir=observer['calls'],input_journal=journal)
    module._CENSUS=native
    install_attention_serving_binding(worker,MECHANISM,prompt_token_ids=tokens,provenance=provenance,native=native)
    module._BINDING_INSTALLED=True
    return dict(worker=expected,source=source,observer_only=True,qualified=False,
        journal_build_sha256=journal.containers.manifest_sha256,limits=deepcopy(journal.limits),
        actual_worker_object_id=id(worker),actual_model_object_id=id(worker.model_runner.model))


def remove_owned_binding(worker,expected,source):
    from demoncat.runtime.vllm_ascend.worker_extension.serving_attention import ATTRIBUTE
    module,journal,worker=verify_worker(worker,expected,source)
    bindings=getattr(worker,ATTRIBUTE,None)
    need(bindings is None or type(bindings) is dict,'ACTUAL_BINDING_TABLE_CHANGED')
    binding=(bindings or {}).get(MECHANISM)
    if binding is None:
        need(not module._BINDING_INSTALLED or module._BINDING_REMOVED,
            'OWNED_BINDING_DISAPPEARED_WITHOUT_REMOVAL')
        return dict(status='SERVING_DEPLOYMENT_ALREADY_REMOVED' if module._BINDING_REMOVED else
            'SERVING_DEPLOYMENT_NOT_INSTALLED',source=source,qualified=False)
    need(binding.native is module._CENSUS and not module._BINDING_REMOVED,
        'FOREIGN_OR_REPLACED_OBSERVATION_BINDING')
    result=common.remove(worker,expected,MECHANISM)
    module._BINDING_REMOVED=True
    return result


def finish_journal(worker,expected,source,out):
    from demoncat.runtime.vllm_ascend.worker_extension.serving_attention import ATTRIBUTE
    module,journal,worker=verify_worker(worker,expected,source)
    bindings=getattr(worker,ATTRIBUTE,None)
    need((bindings is None or type(bindings) is dict) and MECHANISM not in (bindings or {})
        and (not module._BINDING_INSTALLED or module._BINDING_REMOVED),
        'ACTUAL_BINDING_MUST_BE_REMOVED_BEFORE_JOURNAL_RELEASE')
    module._CLEANUP_ATTEMPTS+=1
    need(module._CLEANUP_ATTEMPTS<=2,'BOUNDED_JOURNAL_CLEANUP_ATTEMPTS')
    folder=Path(out)/'input_journal';number=module._CLEANUP_ATTEMPTS
    result=dict(attempt=number,errors=[],qualified=False,global_wrappers_unloaded=False)
    if module._CENSUS is not None:
        try:result['census_stop']=module._CENSUS.stop()
        except BaseException as error:result['errors'].append(dict(stage='census_stop',error=repr(error)))
    try:journal.stop()
    except BaseException as error:result['errors'].append(dict(stage='journal_stop',error=repr(error)))
    try:
        snapshot=journal.snapshot()
        result['observation']=immutable(folder,'observation-%02d.json'%number,snapshot)
        result['state']=snapshot['state']
    except BaseException as error:result['errors'].append(dict(stage='snapshot',error=repr(error)))
    if not result['errors']:
        try:
            released=journal.release()
            result['released']=immutable(folder,'released-%02d.json'%number,released)
            result['released_state']=released['fia'];result['released_containers']=released['containers']
        except BaseException as error:result['errors'].append(dict(stage='release',error=repr(error)))
    # Even failed cleanup is an immutable original. It must not be overwritten
    # by the one allowed later retry or converted to an E5/C2 success.
    result['cleanup_original']=immutable(folder,'cleanup-%02d.json'%number,result)
    return result


def rpc_one(llm,callback,*args):
    values=llm.collective_rpc(callback,args=args,timeout=300)
    need(type(values) is list and len(values)==1,'ONE_STANDARD_WORKER_RPC_REQUIRED')
    return values[0]


def snapshot_failed_owner(worker,expected,source,out):
    """Read quiescent owned records, retaining scopes/pending/public ownership.

    This deliberately does not call census.stop, journal.stop/leave/release or
    _phase. Snapshot cannot authorize cleanup; no pending executor is inspected.
    """
    folder=Path(out)/'input_journal'
    result=dict(schema='demoncat.fia_input_journal.failed_owner_snapshot.v1',
        status='READ_ONLY_FAILURE_SNAPSHOT_REFUSED',source=source,expected_worker=expected,
        trigger='HOST_REPORTED_PUBLIC_OWNERSHIP_RETAINED',errors=[],
        cleanup_performed=False,ownership_release_confirmed=False,pending_reset=False,
        C1=False,C2=False,C3=False,E5=False,qualified=False)
    try:
        module,journal,worker=verify_worker(worker,expected,source)
        journal.containers.current()
        before=journal._states();result['state_before']=before
        journal._owners(before)
        need(before['fia']['active_calls']==0 and before['containers']['active_calls']==0
            and before['containers']['arrays']['active_captures']==0
            and before['containers']['arrays']['active_getters']==0,'FAILURE_SNAPSHOT_CALL_OR_GETTER_STILL_ACTIVE')
        data=journal.snapshot()
        after=journal._states();result['state_after']=after
        journal._owners(after)
        need(before==after and data['state']==before['fia']
            and data['containers']['state']==before['containers'],'FAILURE_SNAPSHOT_FULL_STATE_NOT_QUIESCENT')
        result['observation']=immutable(folder,'failure-observation-01.json',data)
        result['status']='READ_ONLY_PARTIAL_FAILURE_SNAPSHOT_RECORDED_OWNER_RETAINED'
    except BaseException as error:
        result['errors'].append(dict(stage='readonly_snapshot',error=repr(error)))
    # A refused read is itself an immutable original. No zero-state default is
    # invented when query/identity/ownership/quiescence cannot be verified.
    result['snapshot_original']=immutable(folder,'failure-snapshot-01.json',result)
    return result


def finish_failed_run(llm,client,expected,source,out,report):
    if report.get('journal_cleanup') and not report['journal_cleanup'].get('errors'):return
    # Unknown/pending public resources cannot be bypassed by the sidecar.
    if client is not None and client.snapshot()['ownership_retained']:
        report['failure_journal_cleanup_withheld']='PUBLIC_OWNERSHIP_RETAINED'
        if not report.get('failure_journal_snapshot') and not report.get('failure_journal_snapshot_error'):
            try:report['failure_journal_snapshot']=rpc_one(llm,snapshot_failed_owner,expected,source,str(out))
            except BaseException as error:report['failure_journal_snapshot_error']=repr(error)
        return
    try:
        report['failure_binding_removal']=rpc_one(llm,remove_owned_binding,expected,source)
        report['failure_journal_cleanup']=rpc_one(llm,finish_journal,expected,source,str(out))
    except BaseException as error:report['failure_journal_cleanup_error']=repr(error)


def drive(llm,client,*,prompt,params,out,report,source):
    """Original observe/observe_repeat protocol, with explicit final journal RPC.

    Finish/detach precedes journal release. Ordinary post-unregister health is
    then run after release, so the two captured requests are still the only
    observation requests. No interception of enqueue/execute_model is added.
    """
    from demoncat.runtime.vllm_ascend.worker_extension.serving_context import read_enqueued_identity
    folder=out/'worker_core_rpc';folder.mkdir(exist_ok=False)
    core=report.setdefault('worker_core',dict(receipts=[],qualification_granted=False))
    history_count=0;seen=[];pending=[]
    def save():common.save(out/'result.json',report)
    def action(name,*args,expected='ACTION_COMPLETED'):
        nonlocal history_count
        reply=getattr(client,name)(*args)
        for item in reply['history'][history_count:]:
            path=folder/('%03d-%s.json'%(len(core['receipts']),item['action']));common.save(path,item)
            core['receipts'].append(dict(action=item['action'],path=str(path.relative_to(out)),sha256=common.sha(path)))
        history_count=len(reply['history']);state=reply.get('response',{}).get('provider_state')
        if state is not None:report['provider_state']=deepcopy(state)
        core['last_scope']={key:deepcopy(value) for key,value in reply.items()
            if key not in ('history','response','capability_enumeration')};save()
        need(expected is None or reply['status']==expected,'PUBLIC_ATTENTION_ACTION_FAILED:'+name+':'+reply['status'])
        return reply,state
    def enqueue():
        need(not llm.llm_engine.has_unfinished_requests(),'UNRELATED_PENDING_REQUEST')
        need(llm.reset_prefix_cache(reset_running_requests=False) is True,'FRESH_REQUEST_CACHE_RESET_UNCONFIRMED')
        ids=list(llm.enqueue([prompt],params,use_tqdm=False));need(len(ids)==1,'ONE_ACTUAL_REQUEST_REQUIRED')
        mapping=read_enqueued_identity(llm,ids,seen_internal_request_ids=seen);seen.extend(ids)
        return ids,mapping
    try:
        for name in ('prepare','register','check','enable'):action(name)
        report['requests']=[]
        for phase in ('observe','observe_repeat'):
            pending,mapping=enqueue()
            request=dict(phase=phase,mapping=mapping,completed=False,stage='ENQUEUED')
            report['requests'].append(request);save()
            _,before=action('advance',pending[0]);request['stage']='PUBLIC_WINDOW_ARMED';save()
            need(before['serving']['next_phase']==phase,'ATTENTION_PHASE_CHANGED')
            outputs=llm.wait_for_completion(use_tqdm=False)
            need(len(outputs)==1 and outputs[0].request_id==mapping['external_request_id'],'ACTUAL_DRIVER_WORKER_REQUEST_MAPPING_CHANGED')
            _,queried=action('query',expected='QUERY_OK')
            need(queried['serving']['pending_request']==[pending[0],phase]
                and queried['evidence']['requests']==before['evidence']['requests'],'READ_ONLY_QUERY_ADVANCED_REQUEST')
            _,collected=action('collect_replay');row=collected['evidence']['requests'][-1]
            need(row['request_id']==pending[0] and row['phase']==phase,'ATTENTION_ACTUAL_PHASE_BINDING_CHANGED')
            request.update(worker=row,output_tokens=list(outputs[0].outputs[0].token_ids),
                completed=True,stage='COLLECTED');pending=[];save()
        _,state=action('finish')
        need(not any(state[k] for k in ('owned_hooks','owned_api_wrappers','inflight','leases','C1','C2','C3'))
            and state['native_loaded'] is True and state['native_state'].get('observer_only') is True
            and not state['native_state'].get('pending_drains') and not state['native_state'].get('pending_references'),
            'ZERO_F_DISCOVERY_CLEANUP_OR_QUALIFICATION_CHANGED')
        report['final_cleanup']=state
        action('disable');action('unregister');action('detach',expected='DETACHED')
        expected=client.snapshot()['worker']
        report['binding_removal']=rpc_one(llm,remove_owned_binding,expected,source)
        need(report['binding_removal']['status']=='SERVING_DEPLOYMENT_REMOVED','BINDING_REMOVAL_UNCONFIRMED')
        report['journal_cleanup']=rpc_one(llm,finish_journal,expected,source,str(out))
        need(not report['journal_cleanup']['errors'],'INPUT_JOURNAL_CLEANUP_INCOMPLETE')
        pending,mapping=enqueue();outputs=llm.wait_for_completion(use_tqdm=False)
        need(len(outputs)==1 and outputs[0].request_id==mapping['external_request_id'],'POST_RELEASE_REQUEST_MAPPING_CHANGED')
        tokens=list(outputs[0].outputs[0].token_ids)
        report['post_release_health']=dict(mapping=mapping,output_tokens=tokens,
            scope='SAME_WORKER_TOKENS_ONLY_AFTER_UNREGISTER_DETACH_AND_JOURNAL_RELEASE')
        need(tokens==report['healthy_before'] and all(x['output_tokens']==tokens for x in report['requests']),
            'SAME_WORKER_TOKEN_HEALTH_CHANGED');pending=[]
        counts=[sum(len(f['calls']) for f in row['worker']['forwards']) for row in report['requests']]
        need(counts==[28,28],'ACTUAL_2_BY_28_CALL_CENSUS_INCOMPLETE')
        calls=[call for row in report['requests'] for f in row['worker']['forwards'] for call in f['calls']]
        windows=[call.get('native_census',{}).get('input_journal',{}) for call in calls]
        need(all(x.get('window_entered') and x.get('window_left') and x.get('original_calls')==1 for x in windows)
            and len({x.get('call_id') for x in windows})==56,'ACTUAL_56_INPUT_JOURNAL_WINDOWS_INCOMPLETE')
        native=report['journal_cleanup']['state']
        need(native['accepted_windows']==native['prepare_returns']==native['execute_returns']==56
            and native['total_calls']==112 and native['retained_events']==native['total_events']==224
            and not any(native[k] for k in ('first_error','overflow','exceptions','foreign_thread_calls','getter_failures',
                'pending_executors','untracked_executors','failed_prepare_outputs_unknown')),
            'NATIVE_INPUT_JOURNAL_PARTIAL_OR_FAILED_RETAIN_ORIGINALS')
        report['actual_calls_per_request']=counts;report['status']=SUCCESS
    except BaseException as error:
        note_failure(report,error,'public_protocol')
        if pending:
            try:llm.llm_engine.abort_request(pending,internal=True)
            except BaseException as error:report['abort_error']=repr(error)
    finally:
        if client.snapshot()['ownership_retained']:
            for name in ('disable','unregister','detach'):
                try:action(name,expected=None)
                except BaseException as error:report.setdefault('cleanup_errors',[]).append(repr(error))
        host=out/'host_receipt.json';common.save(host,client.snapshot())
        report['host_receipt']=dict(path=host.name,sha256=common.sha(host),bytes=host.stat().st_size)
        core['ownership_retained']=client.snapshot()['ownership_retained'];save()


def bootstrap(args):
    bundle=preparation.validate_journal(args.journal,args.journal_build_sha256)
    observer=json.loads(args.observer_receipt.read_bytes())
    need(common.sha(args.observer_receipt)==args.observer_receipt_sha256,'EXACT_OBSERVER_RECEIPT_REQUIRED')
    library=observer['observer_library'];observer_path=library['path']
    need(common.sha(observer_path)==library['sha256']==bundle['profile']['originals']['observer']['file_sha256']
        and common.sha(bundle['profile']['originals']['observer']['path'])==library['sha256'],
        'ACTUAL_OBSERVER_DEPENDENCY_ALIAS_CHANGED')
    preload=observer_path+':'+str((args.journal/bundle['build']['library']).resolve())
    if not args.child:
        need(not os.environ.get('LD_PRELOAD') and not os.environ.get('ASCEND_CUSTOM_OPP_PATH')
            and not any(k.startswith(('NES_FA_','FAULT28_','DCAT33_','DCAT_F33_','DCAT_ATTN_')) for k in os.environ)
            and not any(k in sys.modules for k in ('torch','torch_npu','vllm')),'CLEAN_PREFRAMEWORK_PARENT_REQUIRED')
        args.out.mkdir(parents=True,exist_ok=False);(args.out/'runtime').mkdir()
        env=dict(os.environ,LD_PRELOAD=preload,DCAT_ATTN_OBSERVER_DIR=str(args.out/'runtime'),
            ACLNN_CACHE_LIMIT='0',TASK_QUEUE_ENABLE='0',PYTHONDONTWRITEBYTECODE='1',VLLM_ENABLE_V1_MULTIPROCESSING='0')
        os.execve(sys.executable,[sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:],'--child'],env)
    need(os.environ.get('LD_PRELOAD')==preload and Path(os.environ.get('DCAT_ATTN_OBSERVER_DIR','')).resolve()==(args.out/'runtime').resolve(),
        'ACTUAL_STARTUP_ENVIRONMENT_CHANGED')
    return bundle,observer


def run(args,bundle,observer):
    report=dict(status='STARTED',mechanism=MECHANISM,probe_sha256=common.sha(__file__),
        mode='TWO_NATURAL_TND_REQUESTS_READONLY_INPUT_JOURNAL',native_faults=0,
        C1=False,C2=False,C3=False,E5=False,qualification_granted=False,
        slice_pointer_lineage_proven=False,prepared_internal_tensor_lineage_proven=False,
        journal_build_sha256=args.journal_build_sha256,observer_receipt_sha256=args.observer_receipt_sha256)
    llm=client=None;source=expected=None
    try:
        report['pre_llm_startup']=prepare_startup(args);source=report['pre_llm_startup']['source']
        common.save(args.out/'result.json',report)
        # This import and constructor are strictly after preflight succeeded.
        from vllm import LLM,SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core
        weights={p.name:dict(sha256=common.sha(p),bytes=p.stat().st_size)
            for p in sorted({*args.model.glob('*.safetensors'),*args.model.glob('pytorch_model*.bin')})}
        config=json.loads((args.model/'config.json').read_bytes())
        need(weights and all(config.get(k)==v for k,v in dict(num_hidden_layers=28,num_attention_heads=16,
            num_key_value_heads=8,hidden_size=1024).items()),'ACTUAL_PRETRAINED_SOURCE87_MODEL_REQUIRED')
        report['model_identity']=dict(weights=weights,config_sha256=common.sha(args.model/'config.json'))
        llm=LLM(model=str(args.model),dtype='float16',tensor_parallel_size=1,enforce_eager=True,max_model_len=2304,
            max_num_seqs=1,max_num_batched_tokens=2304,block_size=16,num_gpu_blocks_override=160,
            enable_prefix_caching=False,enable_chunked_prefill=False,async_scheduling=False,
            gpu_memory_utilization=args.memory,seed=42,disable_log_stats=True,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        require_host_driven_core(llm)
        report['worker_binding']=rpc_one(llm,bind_worker,source);expected=report['worker_binding']['worker']
        words=llm.get_tokenizer().encode('Explain how attention reads a sequence of tokens. ',add_special_tokens=True)
        need(bool(words),'NONEMPTY_TOKENIZER_RESULT_REQUIRED')
        tokens=(words*((2048+len(words)-1)//len(words)))[:2048];prompt=dict(prompt_token_ids=tokens)
        params=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True)
        healthy,=llm.generate([prompt],params,use_tqdm=False);report['healthy_before']=list(healthy.outputs[0].token_ids)
        provenance=dict(model_identity=report['model_identity'],prompt_token_ids=tokens,standard_worker=True,
            constructed_replay=False,probe_sha256=report['probe_sha256'],observer_only=True,input_journal_only=True,
            observer_receipt_sha256=args.observer_receipt_sha256,journal_build_sha256=args.journal_build_sha256)
        ob=dict(path=str(args.observer_receipt),sha256=args.observer_receipt_sha256,calls=str(args.out/'native_calls'))
        def deploy(identity,config):
            need(config.mechanism==MECHANISM,'READ_ONLY_MECHANISM_SCOPE_CHANGED')
            report['setup']=rpc_one(llm,setup,identity,tokens,provenance,ob,source);return {}
        client=common.WorkerFaultClient(common.HostAdapter(llm,prepare_binding=deploy,timeout=300),common.FaultConfig(28,'fia.denominator_replay'))
        drive(llm,client,prompt=prompt,params=params,out=args.out,report=report,source=source)
    except BaseException as error:
        note_failure(report,error,'probe_run')
    finally:
        if llm is not None and expected is not None and source is not None:
            try:finish_failed_run(llm,client,expected,source,args.out,report)
            except BaseException as error:note_failure(report,error,'final_failure_diagnostic')
        try:common.save(args.out/'result.json',report)
        except BaseException as error:note_failure(report,error,'final_result_write')
    console={k:report[k] for k in ('status','error') if k in report}
    if report.get('secondary_errors'):
        console['secondary_errors']=[{key:row[key] for key in ('stage','error')} for row in report['secondary_errors']]
    print(json.dumps(console))
    return 0 if report['status']==SUCCESS else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','out','observer-receipt','journal'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('observer-receipt-sha256','journal-build-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--memory',type=float,default=.25);p.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    need(args.out.resolve().is_relative_to('/evidence') and 0<args.memory<=.5,'PRIVATE_EVIDENCE_AND_BOUNDED_MEMORY_REQUIRED')
    return run(args,*bootstrap(args))


if __name__=='__main__':raise SystemExit(main())
