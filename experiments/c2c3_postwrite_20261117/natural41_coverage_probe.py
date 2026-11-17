"""Root-queued actual Worker C1 -> all-live-KV C2 -> bounded C3 probe."""
import argparse
import json
from pathlib import Path
import traceback

from experiments.c2c3_postwrite_20261117.natural41_probe import (
    PROMPT, OWNER, ROOT, actual_worker, rpc, setup, expect, finish_request, output, record_cleanup_failure)
from experiments.c2c3_postwrite_20261117.model_identity import file_sha, digest, safetensors_identity


SOURCES = ('fault_injection/deployments/postwrite.py', 'fault_injection/deployments/postwrite_boundaries.py',
    'fault_injection/deployments/postwrite_evidence.py', 'fault_injection/deployments/postwrite_kv.py',
    'fault_injection/deployments/postwrite_kv_inventory.py', 'fault_injection/coverage.py',
    'fault_injection/coverage_session.py', 'fault_injection/runtime_discovery.py',
    'fault_injection/native_adapters/postwrite/postwrite.cpp', 'fault_injection/native_adapters/postwrite/postwrite.h',
    'src/demoncat/runtime/vllm_ascend/worker_extension/postwrite41.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/postwrite41_coverage.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/postwrite_collector.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/paged33_audit.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/batch_context.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/call_binding.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/serving_context.py',
    'experiments/c2c3_postwrite_20261117/natural41_probe.py',
    'experiments/c2c3_postwrite_20261117/natural41_coverage_probe.py',
    'experiments/c2c3_postwrite_20261117/model_identity.py')


def worker_model_source(worker):
    from experiments.c2c3_postwrite_20261117.model_identity import model_source_identity
    return model_source_identity(actual_worker(worker).model_runner.model)


def worker_group(worker, scope, max_seconds, expected_targets, capture_path):
    from fault_injection.coverage_session import CoverageScope
    from demoncat.runtime.vllm_ascend.worker_extension.postwrite41_coverage import KVGroupCoverage
    provider = getattr(actual_worker(worker), OWNER)
    if len(provider.candidates) != expected_targets:raise RuntimeError('ACTUAL_ATTENTION_UNIVERSE_COUNT_CHANGED')
    group = KVGroupCoverage(provider,CoverageScope(**scope),max_seconds=max_seconds,requests=2)
    group.raw_collector = None
    if capture_path is not None:
        import msprobe
        from msprobe.pytorch import PrecisionDebugger
        folder=Path(capture_path);folder.mkdir(parents=True,exist_ok=False)
        config=dict(task='statistics',level='mix',precision='high',rank=[],step=[0,1],
            dump_path=str(folder/'observation'),async_dump=False,
            statistics=dict(scope=[],list=[],data_mode=['all'],summary_mode='statistics',tensor_list=[]))
        (folder/'config.json').write_text(json.dumps(config,indent=2))
        package=Path(msprobe.__file__).parent
        identity=dict(package=str(package),files={path.relative_to(package).as_posix():file_sha(path)
                                                 for path in package.rglob('*.py')})
        (folder/'collector_identity.json').write_text(json.dumps(identity,indent=2))
        group.raw_collector=PrecisionDebugger(config_path=str(folder/'config.json'))
        # Register ownership even if a later request fails before capture start.
        provider.collectors.append(group.raw_collector)
    return dict(stage=group.stage,expected_targets=expected_targets,scope=scope)


def group_prepare(worker,trigger,baseline):
    return getattr(actual_worker(worker),OWNER).coverage.prepare(trigger,baseline_control=baseline)


def group_expect(worker,phase,request_id,prompt,captured):
    provider=getattr(actual_worker(worker),OWNER)
    collector=provider.coverage.raw_collector if captured else None
    if captured and collector is None:raise RuntimeError('ACTUAL_COLLECTOR_REQUIRED')
    provider.expect(phase,request_id,prompt,collector=collector)
    return dict(phase=phase,actual_request_id=request_id,captured=captured)


def group_request_finished(worker,captured):
    provider=getattr(actual_worker(worker),OWNER)
    report=provider.finish_request()
    if captured:provider.coverage.raw_collector.step()
    return report


def group_recover(worker):return getattr(actual_worker(worker),OWNER).coverage.begin_recovery()
def group_health_begin(worker,request_id,prompt):
    return getattr(actual_worker(worker),OWNER).coverage.begin_health(request_id,prompt)
def group_health_finish(worker):return getattr(actual_worker(worker),OWNER).coverage.finish_health()


def cleanup(worker):
    worker=actual_worker(worker);provider=getattr(worker,OWNER)
    if provider.coverage is not None:provider.coverage.close_health_observer()
    provider.close()
    result=dict(provider=provider.report(),coverage=provider.coverage.report() if provider.coverage else None)
    delattr(worker,OWNER)
    return result


def worker_report(worker):
    provider=getattr(actual_worker(worker),OWNER)
    return dict(provider=provider.report(),coverage=provider.coverage.report() if provider.coverage else None)


def actual_vllm_weights(path):
    identity=safetensors_identity(path)
    root=Path(path)
    expected=set(identity['manifest']['files'])
    actual={item.relative_to(root).as_posix() for item in root.glob('*.safetensors')}
    if actual!=expected:raise RuntimeError('VLLM_SAFETENSORS_FILE_SELECTION_AMBIGUOUS')
    identity['manifest']['loader_selection']="vllm.LLM(load_format='safetensors', trust_remote_code=False)"
    identity['weights_sha256']=digest(identity['manifest'])
    return identity


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--library',required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--build-fingerprint',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--arm',choices=('baseline','fault'),required=True)
    parser.add_argument('--statistics',action='store_true')
    parser.add_argument('--try-c3',action='store_true')
    parser.add_argument('--expected-targets',type=int,default=28)
    parser.add_argument('--max-seconds',type=int,default=600)
    parser.add_argument('--memory',type=float,default=.35)
    args=parser.parse_args()
    if not 0<args.expected_targets<=64 or not 0<args.max_seconds<=1500:parser.error('finite target/time bounds required')
    args.out.mkdir(parents=True,exist_ok=True)
    report=dict(schema='demoncat.postwrite41.natural_coverage_probe.v1',status='STARTED',arm=args.arm,
        mechanism='41.tail_column',evidence_kind='natural_worker',public_qualification_granted=False,
        native_out_of_bounds_claim=False,build_fingerprint=args.build_fingerprint,
        expected_target_count=args.expected_targets,c1=[],groups={})
    llm=None;installed=False;seen=[]
    def save():(args.out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    try:
        from vllm import LLM,SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core,read_enqueued_identity
        report['weights']=actual_vllm_weights(args.model)
        report['source_sha256']={name:file_sha(ROOT/name) for name in SOURCES}
        llm=LLM(model=str(args.model),dtype='bfloat16',load_format='safetensors',trust_remote_code=False,
            tensor_parallel_size=1,enforce_eager=True,max_model_len=128,max_num_seqs=1,max_num_batched_tokens=128,
            num_gpu_blocks_override=64,enable_prefix_caching=False,enable_chunked_prefill=False,async_scheduling=False,
            gpu_memory_utilization=args.memory,seed=42,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension',disable_log_stats=True)
        require_host_driven_core(llm)
        if actual_vllm_weights(args.model)!=report['weights']:raise RuntimeError('MODEL_ARTIFACTS_CHANGED_DURING_LOAD')
        tokens=llm.get_tokenizer().encode(PROMPT,add_special_tokens=True)
        prompt=dict(prompt_token_ids=tokens)
        params=SamplingParams(temperature=0,max_tokens=2,ignore_eos=True)
        healthy,=llm.generate([prompt],params,use_tqdm=False)
        report['healthy_before']=output(healthy)
        report['setup']=rpc(llm,setup,args.library,args.sha256);installed=True
        report['model_source']=rpc(llm,worker_model_source)
        def enqueue():
            if llm.llm_engine.has_unfinished_requests():raise RuntimeError('UNRELATED_PENDING_REQUEST')
            if llm.reset_prefix_cache(reset_running_requests=False) is not True:raise RuntimeError('RESET_NOT_CONFIRMED')
            ids=llm.enqueue([prompt],params,use_tqdm=False)
            mapping=read_enqueued_identity(llm,ids,seen_internal_request_ids=seen)
            seen.extend(ids)
            return ids[0],mapping
        def complete(mapping):
            response,=llm.wait_for_completion(use_tqdm=False)
            value=output(response)
            if value['request_id']!=mapping['external_request_id']:raise RuntimeError('DRIVER_REQUEST_MAPPING_CHANGED')
            return value
        for phase in ('observe','observe','lossless','fault','off'):
            request,mapping=enqueue();row=dict(phase=phase,mapping=mapping);report['c1'].append(row)
            row['expected']=rpc(llm,expect,phase,request,tokens,None);save()
            row['output']=complete(mapping);row['provider']=rpc(llm,finish_request);save()
        environment=dict(setup=report['setup'],build_fingerprint=args.build_fingerprint)
        # setup contains the real Worker/device/native library binding; all-model
        # weights and loaded Python source identities are independent of hits.
        scope=dict(mechanism='41.tail_column',workload_sha256=digest(dict(prompt=tokens,max_tokens=2,requests=2)),
            environment_sha256=digest(environment),implementation_sha256=digest(report['source_sha256']),
            evidence_kind='natural_worker',model_weights_sha256=report['weights']['weights_sha256'],
            model_source_sha256=report['model_source']['source_sha256'])
        report['group_setup']=rpc(llm,worker_group,scope,args.max_seconds,args.expected_targets,
                                  str(args.out/'capture') if args.statistics else None)
        def request_group(phase,rows,captured=False):
            request,mapping=enqueue();row=dict(phase=phase,mapping=mapping,captured=captured);rows.append(row)
            row['expected']=rpc(llm,group_expect,phase,request,tokens,captured);save()
            row['output']=complete(mapping);row['provider']=rpc(llm,group_request_finished,captured);save()
        report['discovery']=[]
        for _ in range(2):request_group('group_observe',report['discovery'])
        for label,trigger in [('c2','once')]+([('c3','every_match')] if args.try_c3 and args.arm=='fault' else []):
            record=dict(phases=[]);report['groups'][label]=record
            record['prepare']=rpc(llm,group_prepare,trigger,args.arm=='baseline')
            request_group('group_lossless',record['phases'])
            for _ in range(2):request_group('group_fault',record['phases'],captured=args.statistics and label=='c2')
            request_group('group_off',record['phases'])
            record['drained']=rpc(llm,group_recover);save()
            request,mapping=enqueue()
            record['health']=dict(mapping=mapping,observer=rpc(llm,group_health_begin,request,tokens));save()
            record['health']['output']=complete(mapping)
            if any(record['health']['output'][key]!=report['healthy_before'][key] for key in ('tokens','text')):
                raise RuntimeError('SAME_WORKER_NORMAL_HEALTH_GENERATION_CHANGED')
            record['coverage']=rpc(llm,group_health_finish);save()
            result=record['coverage']
            if not result['recovery']['complete'] or result['live_native_leases']:raise RuntimeError('GROUP_RECOVERY_NOT_CONFIRMED')
            if args.arm=='baseline':
                if (not result['all_targets_controlled_once'] or result['hits']!=0
                        or result['failures'] or any(row['native']['changed_elements'] for row in result['executions'])):
                    raise RuntimeError('ALL_TARGET_BASELINE_CONTROL_FAILED')
            elif not result['contract_satisfied']:raise RuntimeError('NATURAL_'+label.upper()+'_CONTRACT_NOT_SATISFIED')
        report['cleanup']=rpc(llm,cleanup);installed=False
        report.update(status=('NATURAL_KV_C2_BASELINE_CONTROL_RECOVERED' if args.arm=='baseline'
                              else 'NATURAL_KV_COVERAGE_RECORDED_PENDING_ROOT_REVIEW'),same_worker_health=True)
    except BaseException as error:
        report.update(status='FAILED_WITH_EVIDENCE',error=repr(error),traceback=traceback.format_exc())
        if llm is not None and installed:
            try:report['failure_snapshot']=rpc(llm,worker_report)
            except BaseException as snapshot_error:report['snapshot_error']=repr(snapshot_error)
        if llm is not None and seen:
            try:llm.llm_engine.abort_request(seen,internal=True)
            except BaseException as abort_error:report['abort_error']=repr(abort_error)
    finally:
        if installed:
            try:report['cleanup']=rpc(llm,cleanup)
            except BaseException as error:record_cleanup_failure(report,error)
        save()
    print(json.dumps({key:report[key] for key in ('status','arm','error','same_worker_health') if key in report}))
    return 0 if report['status'].startswith('NATURAL_KV_') and not report.get('cleanup_error') else 2


if __name__=='__main__':raise SystemExit(main())
