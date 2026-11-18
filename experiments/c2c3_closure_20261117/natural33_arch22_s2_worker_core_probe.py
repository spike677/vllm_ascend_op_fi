"""Root-queued, single-site public 33 R/L/OFF/F/OFF after a fresh raw bridge.

The trusted bootstrap alone supplies immutable artifact paths. Public RPC has
only request IDs and lifecycle actions; it accepts no layer, callback or proof.
No complete-model C2/C3 or statistical localization is claimed by this probe.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import traceback

from attention33_contract import RUNNER,execution_contract,require,sha
from attention33_worker_probe import PROMPT,response,worker_object

PHASES=('reference','lossless','qualification_off','fault','recovery_off')
SUCCESS='PUBLIC_SINGLE_SITE_S2_FUNCTION_AND_OFF_RECORDED_PENDING_INDEPENDENT_REVIEW'


def save(path,data):
    with Path(path).open('w',encoding='utf-8') as stream:json.dump(data,stream,allow_nan=False,indent=2);stream.write('\n')


def record_setup_diagnostic(out,report):
    """Read only this attempt's Worker-written receipt; preserve RPC first error."""
    path=out/'deployment_setup.json'
    try:
        if path.is_file():
            report['deployment_setup']=dict(path=path.name,sha256=sha(path),receipt=json.loads(path.read_bytes()))
        save(out/'result.json',report)
    except Exception as error:report['deployment_diagnostic_error']=repr(error)


def remove(worker,expected,mechanism):
    from demoncat.runtime.vllm_ascend.worker_extension.serving_attention import remove_attention_serving_binding
    worker=worker_object(worker)
    require(worker.demoncat_worker_identity().get('worker')==expected,'S2_REMOVAL_WORKER_CHANGED')
    return remove_attention_serving_binding(worker,mechanism)


def setup(worker,expected,prompt,proof_args,native_args,execution):
    import inspect
    # Written by the Worker before crossing collective_rpc, which may retain
    # only an exception type. This is diagnostics, never a qualification token.
    path=Path(native_args['evidence_dir']).parent/'deployment_setup.json'
    diagnostic=dict(schema='demoncat.paged33.deployment_setup.v1',worker=expected,
        stage='installed_source',binding_installed=False,qualification_granted=False)
    try:
        from demoncat.runtime.vllm_ascend.worker_extension.paged33_s2_evidence import load_verified_normal_evidence
        from demoncat.runtime.vllm_ascend.worker_extension.paged33_arch22_s2_effects import Arch22S2EffectsNativeCall
        from demoncat.runtime.vllm_ascend.worker_extension.serving_attention import install_attention_serving_binding
        worker=worker_object(worker)
        require(Path(inspect.getfile(Arch22S2EffectsNativeCall)).resolve().is_relative_to('/installed'),
                'S2_INSTALLED_WHEEL_REQUIRED')
        diagnostic['stage']='worker_identity'
        observed=worker.demoncat_worker_identity()
        require(observed.get('status')=='WORKER_IDENTIFIED' and not observed.get('quarantined')
            and observed['worker']==expected,'S2_DEPLOYMENT_ACTUAL_WORKER_CHANGED')
        diagnostic['stage']='normal_raw_replay'
        normal=load_verified_normal_evidence(**proof_args,current_execution=execution)
        diagnostic['stage']='native_constructor'
        native=Arch22S2EffectsNativeCall(normal_evidence=normal,**native_args)
        diagnostic['stage']='serving_binding'
        install_attention_serving_binding(worker,'33.paged_decode',prompt_token_ids=prompt,native=native,
            provenance=dict(standard_worker=True,constructed_replay=False,normal_evidence=normal.data(),
                            normal_evidence_sha256=normal.fingerprint,execution_contract=execution))
        diagnostic.update(stage='complete',binding_installed=True,status='INERT_DEPLOYMENT_INSTALLED')
        result=dict(worker=observed['worker'],normal_evidence=normal.data(),
            normal_evidence_sha256=normal.fingerprint,native_build=native.fingerprint,qualification_granted=False)
        save(path,diagnostic)
        return result
    except BaseException as error:
        diagnostic.update(status='DEPLOYMENT_REFUSED',error_type=type(error).__name__,error=repr(error),
            reason=str(error),code=getattr(error,'code',None),traceback=traceback.format_exc())
        if getattr(error,'response',None) is not None:
            try:diagnostic['observed_failure']=deepcopy(error.response)
            except Exception as copy_error:diagnostic['observed_failure_copy_error']=repr(copy_error)
        try:save(path,diagnostic)
        except BaseException as write_error:diagnostic['diagnostic_write_error']=repr(write_error)
        # The file preserves the original failure if transport discards attrs.
        try:error.response=diagnostic
        except (AttributeError,TypeError):pass
        raise


def drive(llm,client,*,prompt,params,out,report):
    from demoncat.runtime.vllm_ascend.worker_extension.serving_context import read_enqueued_identity
    folder=out/'worker_core_rpc';folder.mkdir(exist_ok=False)
    core=report.setdefault('worker_core',dict(receipts=[]));report['requests']=[]
    history_count=0;seen=[];pending=[];failure=None
    def action(name,*args,expected='ACTION_COMPLETED'):
        nonlocal history_count
        reply=getattr(client,name)(*args)
        for item in reply['history'][history_count:]:
            path=folder/(f'{len(core["receipts"]):03d}-'+item['action']+'.json');save(path,item)
            core['receipts'].append(dict(action=item['action'],path=str(path.relative_to(out)),sha256=sha(path)))
        history_count=len(reply['history'])
        state=reply.get('response',{}).get('provider_state')
        if state is not None:report['provider_state']=deepcopy(state)
        save(out/'result.json',report)
        require(expected is None or reply['status']==expected,'S2_PUBLIC_ACTION_FAILED:'+name+':'+reply['status'])
        return state
    def request(phase):
        nonlocal pending
        require(not llm.llm_engine.has_unfinished_requests(),'S2_UNRELATED_PENDING_REQUEST')
        require(llm.reset_prefix_cache(reset_running_requests=False) is True,'S2_FRESH_REQUEST_RESET_REQUIRED')
        pending=list(llm.enqueue([prompt],params,use_tqdm=False))
        require(len(pending)==1,'S2_ONE_ACTUAL_REQUEST_REQUIRED')
        mapping=read_enqueued_identity(llm,pending,seen_internal_request_ids=seen);seen.extend(pending)
        original=dict(phase=phase,mapping=mapping,completed=False);report['requests'].append(original)
        before=action('advance',pending[0]);require(before['serving']['next_phase']==phase,'S2_ACTUAL_PHASE_CHANGED')
        values=llm.wait_for_completion(use_tqdm=False)
        require(len(values)==1 and values[0].request_id==mapping['external_request_id'],'S2_ACTUAL_REQUEST_MAPPING_CHANGED')
        original['response']=response(values[0]);save(out/'result.json',report)
        queried=action('query',expected='QUERY_OK')
        require(queried['serving']['pending_request']==[pending[0],phase]
            and queried['evidence']['actual_requests_sha256']==before['evidence']['actual_requests_sha256'],
            'S2_QUERY_ADVANCED_OR_CHANGED_REQUEST')
        collected=action('collect_replay');row=collected['evidence']['requests'][-1]
        require((row['phase'],row['request_id'])==(phase,pending[0]),'S2_ACTUAL_COLLECTED_REQUEST_CHANGED')
        original.update(worker=row,completed=True)
        pending=[];save(out/'result.json',report)
    try:
        for name in ('prepare','register','check','enable'):action(name)
        for phase in PHASES:request(phase)
    except BaseException as error:
        failure=repr(error);report.update(error=failure,traceback=traceback.format_exc())
        if pending:
            try:llm.llm_engine.abort_request(pending,internal=True);pending=[]
            except BaseException as abort_error:report['abort_error']=repr(abort_error)
    # A recovery request must leave the original except context. Natural
    # observers correctly reject calls started under an active old exception.
    if failure is not None and client.snapshot()['ticket'] is not None:
        try:
            state=action('query',expected='QUERY_OK')
            native=state['native_state'];resource=state['serving']['resource_state']
            if native.get('effect_attempted') and not native.get('off_confirmed') and resource['needs_off']:
                action('recover');request('recovery_off')
        except BaseException as error:report['recovery_error']=repr(error)
    try:
        if failure is not None and client.snapshot()['ticket'] is None:
            # An uncertain prepare may have reached the Worker without a
            # returned lease. Only same-tenant public detach can resolve it;
            # query/disable/unregister require a ticket we do not possess.
            action('detach',expected='DETACHED')
            require(not client.snapshot()['ownership_retained'],'S2_PUBLIC_OWNER_RETAINED')
            report['status']='FAILED_BEFORE_PROTOCOL_WITH_ORIGINAL_ERROR_AND_PUBLIC_DETACH'
            report['prepare_recovery']=dict(action='detach',lease_ticket_observed=False,
                ordinary_health_submitted=False,qualification_granted=False)
            return
        if failure is None:
            final=action('finish')
            require(final['C1'] is True and final['C2'] is False and final['C3'] is False
                and not any(final[key] for key in ('leases','inflight','owned_hooks','owned_api_wrappers'))
                and final['native_state']['retained_input_objects']==0
                and final['serving']['resource_state']['owned_input_references']==0,
                'S2_ACTUAL_FUNCTION_OR_FINAL_ZERO_RESOURCE_NOT_PROVEN')
            report['final_cleanup']=final
        # On a failed experiment these actions still require actual OFF, zero
        # resources and exact owner checks. Failure never upgrades to success.
        for name in ('disable','unregister'):action(name)
        action('detach',expected='DETACHED')
        require(not client.snapshot()['ownership_retained'],'S2_PUBLIC_OWNER_RETAINED')
        require(not llm.llm_engine.has_unfinished_requests(),'S2_PENDING_REQUEST_AFTER_UNREGISTER')
        require(llm.reset_prefix_cache(reset_running_requests=False) is True,'S2_HEALTH_RESET_UNCONFIRMED')
        pending=list(llm.enqueue([prompt],params,use_tqdm=False))
        mapping=read_enqueued_identity(llm,pending,seen_internal_request_ids=seen)
        values=llm.wait_for_completion(use_tqdm=False)
        require(len(values)==1 and values[0].request_id==mapping['external_request_id'],'S2_POST_UNREGISTER_MAPPING_CHANGED')
        report['post_unregister_health']=dict(mapping=mapping,response=response(values[0]),
            scope='TOKENS_TEXT_ONLY_NO_POST_UNREGISTER_HIDDEN_OR_LOGITS_OBSERVER')
        require(values[0].outputs and report['post_unregister_health']['response']['outputs']==report['warm_healthy']['outputs'],
                'S2_POST_UNREGISTER_HEALTH_DIFFERS')
        for row in report['requests']:
            if row['phase']!='fault' and row.get('completed'):
                require(row['response']['outputs']==report['warm_healthy']['outputs'],'S2_CONTROL_TOKEN_HEALTH_DIFFERS')
        pending=[]
        report['status']=SUCCESS if failure is None else 'FAILED_WITH_ORIGINAL_ERROR_AND_RECOVERED_OFF'
    except BaseException as error:
        report.setdefault('error',repr(error));report['cleanup_error']=repr(error)
        report['status']='FAILED_WITH_OWNER_OR_RECOVERY_EVIDENCE'
    finally:
        # No blind cleanup retry or replacement of a failed original receipt.
        path=out/'host_receipt.json';snapshot=client.snapshot();save(path,snapshot)
        report['host_receipt']=dict(path=path.name,sha256=sha(path),bytes=path.stat().st_size)
        core['ownership_retained']=snapshot['ownership_retained'];save(out/'result.json',report)


def normal_arguments(args):
    return dict(source=args.source,source_manifest_sha256=args.source_manifest_sha256,
        build_root=args.build_root,completion_sha256=args.completion_sha256,observer=args.observer,
        observer_receipt_sha256=args.observer_receipt_sha256,stock_run=args.stock_run,stock_sha256=args.stock_sha256,
        baseline_run=args.baseline_run,baseline_sha256=args.baseline_sha256,report=args.normal_report,
        report_sha256=args.normal_report_sha256)


def run(args,bundle):
    from demoncat.runtime.vllm_ascend.worker_extension.paged33_loading import bootstrap
    report=dict(schema='demoncat.paged33.public_s2_single.v1',status='STARTED',probe_sha256=sha(__file__),
        C1=False,C2=False,C3=False,E5='NOT_ASSESSED',source_manifest_sha256=args.source_manifest_sha256,
        normal_inputs={k:str(v) if isinstance(v,Path) else v for k,v in normal_arguments(args).items()},
        execution_contract=execution_contract(args.model,'/installed',__file__,.35,os.environ),
        scope='ONE_SOURCE_BOUND_FIRST_LAYER_TND_Q1_KV1025_PAGE16_FP16',all_model_eligible_qualified=False)
    save(args.out/'result.json',report)
    client=None;llm=None
    try:
        loader=bootstrap(bundle);report['library_bootstrap']=loader.report()
        from vllm import LLM,SamplingParams
        from vllm.platforms import current_platform
        from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
        from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter,WorkerFaultClient
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core
        report['library_coexistence']=loader.prepare_platform(current_platform)
        llm=LLM(model=str(args.model),gpu_memory_utilization=.35,**RUNNER);require_host_driven_core(llm)
        report['library_coexistence_after_standard_worker_initialization']=loader.verify()
        tokens=llm.get_tokenizer().encode(PROMPT*128,add_special_tokens=True)[:1024]
        require(len(tokens)==1024,'S2_ACTUAL_NORMAL_PROMPT_LENGTH_DIFFERS')
        prompt=dict(prompt_token_ids=tokens);params=SamplingParams(temperature=0,max_tokens=2,ignore_eos=True)
        warm,=llm.generate([prompt],params,use_tqdm=False);report['warm_healthy']=response(warm)
        report['prompt_token_ids']=tokens
        native=dict(completion=args.build_root/'evidence/completion.json',completion_sha256=args.completion_sha256,
            vendor=bundle['vendor'],schema=args.build_root/'evidence/arch22_schema/fia_abi_schema.json',
            schema_sha256=bundle['completion']['arch22_public_schema_sha256'],observer_receipt=args.observer,
            observer_receipt_sha256=args.observer_receipt_sha256,evidence_dir=args.out/'native')
        def deploy(expected,config):
            try:
                rows=llm.collective_rpc(setup,args=(expected,tokens,normal_arguments(args),native,report['execution_contract']),timeout=600)
            finally:record_setup_diagnostic(args.out,report)
            require(len(rows)==1 and rows[0]['worker']==expected,'S2_TP1_DEPLOYMENT_CHANGED')
            report['setup']=rows[0];return {}
        client=WorkerFaultClient(HostAdapter(llm,prepare_binding=deploy,timeout=600),FaultConfig(33,'paged_decode'))
        drive(llm,client,prompt=prompt,params=params,out=args.out,report=report)
        setup_receipt=report.get('deployment_setup',{}).get('receipt',{})
        if (report.get('setup') or setup_receipt.get('binding_installed') is True) and not client.snapshot()['ownership_retained']:
            report['library_coexistence_after_owned_cleanup']=loader.verify()
            rows=llm.collective_rpc(remove,args=(client.snapshot()['worker'],'33.paged_decode'),timeout=300)
            require(len(rows)==1 and rows[0]['status']=='SERVING_DEPLOYMENT_REMOVED','S2_DEPLOYMENT_REMOVAL_UNCONFIRMED')
            report['binding_removal']=rows[0]
    except BaseException as error:
        report.update(status='REFUSED_OR_FAILED')
        if 'error' in report:report['post_protocol_error']=dict(error=repr(error),traceback=traceback.format_exc())
        else:report.update(error=repr(error),traceback=traceback.format_exc())
    finally:save(args.out/'result.json',report)
    print(json.dumps({key:report[key] for key in ('status','error','C2','C3','E5') if key in report}))
    return 0 if report['status']==SUCCESS else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','build-root','observer','stock-run','baseline-run','normal-report','model','out'):
        parser.add_argument('--'+key,type=Path,required=True)
    for key in ('source-manifest-sha256','completion-sha256','observer-receipt-sha256','stock-sha256',
                'baseline-sha256','normal-report-sha256'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--child',action='store_true',help=argparse.SUPPRESS);args=parser.parse_args()
    require(args.out.resolve().is_relative_to('/evidence'),'S2_PRIVATE_EVIDENCE_DIRECTORY_REQUIRED')
    from demoncat.runtime.vllm_ascend.worker_extension.paged33_arch22_s2_normal import verify_bundle
    # Keep the exact same loader/vendor alias as the preceding normal pair.
    # The original build-root alias additionally exposes its patch receipt to
    # the independent raw reviewer; both mounts are read-only identical bytes.
    vendor=Path('/native/stage/vendors/dcat_f33_arch22_s2_transformer')
    bundle=verify_bundle(args.build_root/'evidence/completion.json',args.completion_sha256,vendor)
    require(sha(args.observer)==args.observer_receipt_sha256,'S2_OBSERVER_RECEIPT_CHANGED')
    observer=json.loads(args.observer.read_bytes());library=observer['observer_library']
    require(observer.get('observer_compiled') is True and observer.get('actual_npu_calls')==0
        and observer.get('observer_memory_selftest',{}).get('exit_code')==0 and sha(library['path'])==library['sha256'],
        'S2_FRESH_OBSERVER_MEMORY_SELFTEST_REQUIRED')
    if not args.child:
        require(not any(x in sys.modules for x in ('torch','torch_npu')),'S2_BOOTSTRAP_IMPORTED_FRAMEWORK')
        require(not os.environ.get('LD_PRELOAD') and not os.environ.get('ASCEND_CUSTOM_OPP_PATH')
            and not any(x.startswith(('DCAT33_TND_','DCAT_ATTN_')) for x in os.environ),'S2_CLEAN_NATIVE_ENV_REQUIRED')
        require(not args.out.exists(),'S2_FRESH_EVIDENCE_REQUIRED');args.out.mkdir(parents=True)
        runtime=args.out/'runtime';runtime.mkdir()
        environment=dict(os.environ,LD_PRELOAD=library['path'],DCAT_ATTN_OBSERVER_DIR=str(runtime),
            ASCEND_CUSTOM_OPP_PATH=str(vendor),ACLNN_CACHE_LIMIT='0',TASK_QUEUE_ENABLE='0',VLLM_ENABLE_V1_MULTIPROCESSING='0')
        os.execve(sys.executable,[sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:],'--child'],environment)
    return run(args,bundle)


if __name__=='__main__':raise SystemExit(main())
