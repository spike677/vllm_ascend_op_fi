"""Root-queued real multimodal Worker census: two healthy enqueues, zero F."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'fault39/src')]
from experiments.c2c3_closure_20261117.natural_tensor_worker_core_probe import sha_file,actual
from experiments.c2c3_closure_20261117.natural_vision_input import (
    IMAGE_SHA256,WEIGHTS_SHA256,WEIGHTS_BYTES,PROMPT,require_geometry)
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter,WorkerFaultClient
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import read_enqueued_identity,require_host_driven_core


def setup(worker,expected,prompt,image_sha,weight_sha):
    from demoncat.runtime.vllm_ascend.worker_extension.providers.vision17 import (
        install_vision17_binding,remove_vision17_binding,Vision17ObservationProvider)
    from demoncat.runtime.vllm_ascend.worker_extension.registry import create_worker_provider
    worker=actual(worker);identity=worker.demoncat_worker_identity()
    if identity.get('worker')!=expected or identity.get('quarantined') is not False:
        raise RuntimeError('ACTUAL_VISION17_WORKER_IDENTITY_CHANGED')
    result=install_vision17_binding(worker,prompt_token_ids=prompt,image_sha256=image_sha,
                                   weights_sha256=weight_sha,first_live_validation=True)
    # Read-only resolver check, before Runtime creates an owner. No patch/fallback.
    try:
        provider=create_worker_provider('17.spatial_padding','1',worker,worker.model_runner.model,expected)
        if type(provider) is not Vision17ObservationProvider:
            raise RuntimeError('PUBLIC_VISION17_OBSERVATION_ROUTE_NOT_INTEGRATED')
    except BaseException:
        remove_vision17_binding(worker);raise
    return result


def remove_binding(worker,expected):
    from demoncat.runtime.vllm_ascend.worker_extension.providers.vision17 import remove_vision17_binding
    worker=actual(worker)
    if worker.demoncat_worker_identity().get('worker')!=expected:
        raise RuntimeError('VISION17_CLEANUP_WORKER_CHANGED')
    return remove_vision17_binding(worker)


def driver_prompt(llm,request_id):
    """Read the actual already-enqueued processed MM prompt, never tokenize it again."""
    state=llm.llm_engine.output_processor.request_states[request_id]
    if state.request_id!=request_id:
        raise RuntimeError('ACTUAL_MM_DRIVER_STATE_ID_MISMATCH')
    tokens=getattr(state,'prompt_token_ids',None)
    if not isinstance(tokens,list) or not 2<=len(tokens)<=4096 or any(type(t) is not int for t in tokens):
        raise RuntimeError('ACTUAL_MM_PROCESSED_DRIVER_TOKEN_IDS_UNAVAILABLE')
    return list(tokens)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--memory',type=float,default=.60)
    args=parser.parse_args()
    if not .1<=args.memory<=.8:parser.error('Bounded memory .1..8 required')
    args.out.mkdir(parents=True,exist_ok=False)
    report=dict(status='STARTED',fault_injections=0,standard_worker_qualified=False,c1=False,c2=False,c3=False,e5=False,
                aclnn_launch_arguments=False,requests=[],rpc=[],mm_cache_reset_used=False)
    llm=client=None;bound=False;expected=None;pending=[];recorded=0
    def save():
        (args.out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    def rpc(function,*values):
        result=llm.collective_rpc(function,args=values,timeout=300)
        if len(result)!=1:raise RuntimeError('VISION17_TP1_WORKER_REQUIRED')
        return result[0]
    def action(name,*values,expected_status='ACTION_COMPLETED'):
        nonlocal recorded
        result=getattr(client,name)(*values)
        report['rpc'].extend(deepcopy(result['history'][recorded:]));recorded=len(result['history'])
        report['last_scope']={k:deepcopy(v) for k,v in result.items() if k not in ('history','capability_enumeration')}
        save()
        if result['status']!=expected_status:raise RuntimeError('PUBLIC_VISION17_ACTION_FAILED:'+name+':'+result['status'])
        return result
    try:
        weights=args.model/'model.safetensors'
        if weights.stat().st_size!=WEIGHTS_BYTES or sha_file(weights)!=WEIGHTS_SHA256 or sha_file(args.image)!=IMAGE_SHA256:
            raise RuntimeError('PINNED_PRETRAINED_WEIGHTS_AND_NATURAL_IMAGE_REQUIRED')
        geometry=require_geometry(json.loads((args.model/'config.json').read_bytes()))
        from PIL import Image
        from transformers import AutoProcessor
        from vllm import LLM,SamplingParams
        processor=AutoProcessor.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
        text=processor.apply_chat_template([dict(role='user',content=[dict(type='image'),dict(type='text',text=PROMPT)])],
                                           tokenize=False,add_generation_prompt=True)
        with Image.open(args.image) as opened:image=opened.convert('RGB')
        prompt=dict(prompt=text,multi_modal_data={'image':image})
        llm=LLM(model=str(args.model),dtype='bfloat16',tensor_parallel_size=1,enforce_eager=True,
            max_model_len=2048,max_num_seqs=1,max_num_batched_tokens=2048,num_gpu_blocks_override=128,
            enable_prefix_caching=False,enable_chunked_prefill=False,async_scheduling=False,
            gpu_memory_utilization=args.memory,limit_mm_per_prompt={'image':1},seed=42,disable_log_stats=True,
            trust_remote_code=False,worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        require_host_driven_core(llm)
        params=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True)
        report['input']=dict(image_sha256=IMAGE_SHA256,weights_sha256=WEIGHTS_SHA256,geometry=geometry,
            config_sha256=sha_file(args.model/'config.json'),prompt=PROMPT,processor_overrides={},raw_PIL_image=True)
        seen=[];frozen_prompt=None
        for phase in ('healthy_0','healthy_1'):
            if llm.llm_engine.has_unfinished_requests():raise RuntimeError('UNRELATED_PENDING_REQUEST')
            if llm.reset_prefix_cache(reset_running_requests=False) is not True:
                raise RuntimeError('PREFIX_RESET_UNCONFIRMED')
            pending=llm.enqueue([prompt],params,use_tqdm=False)
            mapping=read_enqueued_identity(llm,pending,seen_internal_request_ids=seen);seen.extend(pending)
            tokens=driver_prompt(llm,pending[0])
            if frozen_prompt is None:
                frozen_prompt=tokens
                def prepare(identity,config):
                    nonlocal bound,expected
                    expected=identity
                    report['setup']=rpc(setup,identity,tokens,IMAGE_SHA256,WEIGHTS_SHA256)
                    bound=True;return {}
                client=WorkerFaultClient(HostAdapter(llm,prepare_binding=prepare,timeout=300),FaultConfig(17,'spatial_padding'))
                for name in ('prepare','register','check','enable'):action(name)
            elif tokens!=frozen_prompt:raise RuntimeError('ACTUAL_PROCESSED_MM_PROMPT_CHANGED')
            action('advance',pending[0]);output=llm.wait_for_completion(use_tqdm=False)
            if len(output)!=1 or output[0].request_id!=mapping['external_request_id']:
                raise RuntimeError('ACTUAL_MM_OUTPUT_MAPPING_CHANGED')
            action('collect_replay')
            report['requests'].append(dict(phase=phase,mapping=mapping,prompt_token_ids=tokens,
                output_token_ids=list(output[0].outputs[0].token_ids),text=output[0].outputs[0].text))
            pending=[];save()
        state=action('finish')['response']['provider_state']
        report['observation']=state['observation'];action('unregister')
        action('detach',expected_status='DETACHED')
        report['unbind']=rpc(remove_binding,expected);bound=False
        report['status']='TWO_HEALTHY_MM_CENSUS_COMPLETE_NOT_FAULT_QUALIFICATION'
    except BaseException as error:
        report.update(status='REFUSED_OR_FAILED',error=repr(error),traceback=traceback.format_exc())
        if llm is not None and pending:
            try:llm.llm_engine.abort_request(pending,internal=True)
            except BaseException as cleanup:report['abort_error']=repr(cleanup)
        if client is not None:
            for name in ('recover','unregister','detach'):
                try:action(name,expected_status='DETACHED' if name=='detach' else 'ACTION_COMPLETED')
                except BaseException as cleanup:report.setdefault('cleanup_errors',[]).append(dict(action=name,error=repr(cleanup)))
        if bound and client is not None and not report.get('cleanup_errors'):
            try:report['unbind']=rpc(remove_binding,expected);bound=False
            except BaseException as cleanup:report['unbind_error']=repr(cleanup)
    finally:
        report['binding_still_owned']=bound;report['probe_sha256']=sha_file(__file__);save()
    print(json.dumps({k:report[k] for k in ('status','error','fault_injections','binding_still_owned') if k in report}))
    return 0 if report['status']=='TWO_HEALTHY_MM_CENSUS_COMPLETE_NOT_FAULT_QUALIFICATION' else 2


if __name__=='__main__':raise SystemExit(main())
