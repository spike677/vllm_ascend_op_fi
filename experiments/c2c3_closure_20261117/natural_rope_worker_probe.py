"""Root-queued actual fused RoPE C2/C3 requests; no fabricated model path."""
import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from experiments.c2c3_closure_20261117.natural_rms_worker_probe import actual,digest
from experiments.c2c3_closure_20261117.natural_tensor_census import PROMPT,sha_file


def setup(worker,prompt,weights,workload):
    import torch,torch_npu
    from fault_injection.rope_serving import ServingRoPEGroup
    worker=actual(worker)
    if hasattr(worker,'_natural_rope_group'):raise RuntimeError('FOREIGN_SERVING_ROPE_OWNER')
    environment=dict(torch=torch.__version__,torch_npu=torch_npu.__version__,device=torch.npu.get_device_name(0))
    session=ServingRoPEGroup(worker,prompt=prompt,weight_manifest=weights,
        workload_sha256=workload,environment_sha256=digest(environment))
    worker._natural_rope_group=session
    return dict(provenance=session.provenance,environment=environment)


def advance(worker,request_id,phase,arm):return actual(worker)._natural_rope_group.expect(request_id,phase,arm)
def collect(worker):return actual(worker)._natural_rope_group.collect()
def qualify(worker):return actual(worker)._natural_rope_group.qualify()
def finish_group(worker):return actual(worker)._natural_rope_group.finish_group()


def open_group(worker,trigger,forwards):
    session=actual(worker)._natural_rope_group
    # The original issued proof stays in its owning Worker process. It is
    # never reconstructed from a serialized report or copied through RPC.
    proof=session.provider.group.c2_proof() if trigger=='every_match' else None
    group=session.provider.open_group(trigger,forwards=forwards,c2_proof=proof)
    return dict(generation=group.generation,targets=len(group.group.leases),trigger=trigger)


def cleanup(worker,failed):
    worker=actual(worker);session=worker._natural_rope_group
    result=session.abort() if failed else session.close()
    del worker._natural_rope_group
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--memory',type=float,default=.35)
    args=parser.parse_args()
    if not .1<=args.memory<=.8:parser.error('Bounded Worker memory .1..8 required')
    args.out.mkdir(parents=True,exist_ok=False)
    report=dict(status='STARTED',mechanism='22.wrong_row.fused_positions',standard_vllm_worker=True,
        public_qualification_granted=False,raw_device_launch_payload_observed=False,requests=[])
    installed=completed=False;llm=None;seen=[]
    def save():
        (args.out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    try:
        from vllm import LLM,SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core,read_enqueued_identity
        weights={path.name:dict(sha256=sha_file(path),bytes=path.stat().st_size)
            for path in sorted(args.model.glob('*.safetensors'))}
        if not weights:raise RuntimeError('ACTUAL_PRETRAINED_WEIGHTS_REQUIRED')
        config_sha=sha_file(args.model/'config.json')
        report['model_identity']=dict(weights=weights,config_sha256=config_sha,route_configuration_modified=False)
        llm=LLM(model=str(args.model),dtype='bfloat16',tensor_parallel_size=1,enforce_eager=True,
            max_model_len=128,max_num_seqs=1,max_num_batched_tokens=128,num_gpu_blocks_override=64,
            enable_prefix_caching=False,enable_chunked_prefill=False,async_scheduling=False,
            gpu_memory_utilization=args.memory,seed=42,disable_log_stats=True,trust_remote_code=False,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        require_host_driven_core(llm)
        def rpc(function,*values):
            rows=llm.collective_rpc(function,args=values,timeout=300)
            if len(rows)!=1:raise RuntimeError('TP1_ROPE_SERVING_SCOPE_REQUIRED')
            return rows[0]
        tokens=llm.get_tokenizer().encode(PROMPT,add_special_tokens=True)
        prompt=dict(prompt_token_ids=tokens);params=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True)
        healthy,=llm.generate([prompt],params,use_tqdm=False)
        report['healthy_before']=list(healthy.outputs[0].token_ids)
        report['setup']=rpc(setup,tokens,weights,digest(dict(weights=weights,config_sha256=config_sha,tokens=tokens)))
        installed=True;save()
        def request(phase,arm=None):
            if llm.llm_engine.has_unfinished_requests():raise RuntimeError('UNRELATED_PENDING_REQUEST')
            if llm.reset_prefix_cache(reset_running_requests=False) is not True:
                raise RuntimeError('FRESH_PREFILL_CACHE_RESET_UNCONFIRMED')
            ids=llm.enqueue([prompt],params,use_tqdm=False)
            mapping=read_enqueued_identity(llm,ids,seen_internal_request_ids=seen);seen.extend(ids)
            rpc(advance,ids[0],phase,arm)
            results=llm.wait_for_completion(use_tqdm=False)
            if len(results)!=1 or results[0].request_id!=mapping['external_request_id']:
                raise RuntimeError('ACTUAL_DRIVER_WORKER_REQUEST_MAPPING_CHANGED')
            row=rpc(collect)
            report['requests'].append(dict(phase=phase,mapping=mapping,worker=row,
                output_tokens=list(results[0].outputs[0].token_ids)));save()
        for phase,arm in (('native_baseline',None),('baseline','baseline'),
                          ('baseline_repeat','baseline'),('lossless','baseline')):request(phase,arm)
        report['qualification']=rpc(qualify);save()
        report['c2_open']=rpc(open_group,'once',1)
        request('c2_fault','fault');request('c2_off');report['c2']=rpc(finish_group);save()
        if not report['c2']['contract_satisfied']:raise RuntimeError('STANDARD_WORKER_ROPE_C2_NOT_COMPLETE')
        report['c3_open']=rpc(open_group,'every_match',2)
        request('c3_fault_0','fault');request('c3_fault_1','fault');request('c3_off')
        report['c3']=rpc(finish_group);save()
        if not report['c3']['contract_satisfied']:raise RuntimeError('STANDARD_WORKER_ROPE_C3_NOT_COMPLETE')
        completed=True;report['status']='STANDARD_WORKER_ROPE_GROUP_CONTRACT_COMPLETE'
        report['statistics_status']='SEPARATE_RAW_BF_CAPTURE_REQUIRED'
    except BaseException as exc:
        report.update(status='REFUSED_OR_FAILED',error=repr(exc),traceback=traceback.format_exc())
        if llm is not None and seen:
            try:llm.llm_engine.abort_request(seen,internal=True)
            except BaseException as error:report['abort_error']=repr(error)
    finally:
        if installed:
            try:
                report['cleanup']=rpc(cleanup,not completed)
                if completed:
                    healthy,=llm.generate([prompt],params,use_tqdm=False)
                    report['healthy_after']=list(healthy.outputs[0].token_ids)
                    if report['healthy_after']!=report['healthy_before']:
                        raise RuntimeError('SAME_WORKER_GENERATION_NOT_RESTORED')
            except BaseException as error:report.update(status='REFUSED_OR_FAILED',cleanup_error=repr(error))
        report['probe_sha256']=sha_file(__file__);save()
    print(json.dumps({k:report[k] for k in ('status','error') if k in report}))
    return 0 if report['status']=='STANDARD_WORKER_ROPE_GROUP_CONTRACT_COMPLETE' else 2


if __name__=='__main__':raise SystemExit(main())
