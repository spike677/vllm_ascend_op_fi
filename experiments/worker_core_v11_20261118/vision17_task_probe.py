"""Two independent four-request ordinary image tasks; Torch parameter scope only."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]

from experiments.c2c3_closure_20261117.natural_tensor_worker_core_probe import sha_file,actual
from experiments.c2c3_closure_20261117.natural_vision_input import (
    IMAGE_SHA256,WEIGHTS_SHA256,WEIGHTS_BYTES,PROMPT,require_geometry)
from experiments.c2c3_closure_20261117.vision17_worker_census_probe import driver_prompt
from experiments.worker_core_v11_20261118.vision17_cache_control import reset_idle_mm_cache
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter,WorkerFaultClient
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import read_enqueued_identity,require_host_driven_core
from demoncat.runtime.vllm_ascend.worker_extension.providers.vision17_task import PHASES


MODULES=('fault_injection.vision17_request','fault_injection.vision17_worker','fault_injection.vision17_context',
    'fault_injection.vision17_trace','fault_injection.vision17_padding','fault_injection.vision17_observation',
    'fault_injection.conv3d_natural','fault_injection._scoped_dispatch','fault_injection.tensor_callable')+tuple(
    'demoncat.runtime.vllm_ascend.worker_extension.'+name for name in
    ('providers.vision17_task','deployment_acceptance','ordinary_evidence','ordinary_admission','prerequisites','runtime','registry','host','config'))

def source_identity(expected):
    import hashlib
    manifest=ROOT/'source_manifest.json'
    if sha_file(manifest)!=expected:raise RuntimeError('FROZEN_SOURCE_MANIFEST_CHANGED')
    data=json.loads(manifest.read_bytes());entries={row['path']:row for row in data['files']}
    pins={}
    for name in MODULES:
        relative=('src/' if name.startswith('demoncat.') else '')+name.replace('.','/')+'.py'
        row=entries[relative];path=ROOT/relative
        if path.stat().st_size!=row['bytes'] or sha_file(path)!=row['sha256']:
            raise RuntimeError('FROZEN_SOURCE_CHANGED:'+relative)
        pins[name]=row['sha256']
    return dict(commit=data['base_commit'],manifest_sha256=expected,pins=pins)


def setup(worker,expected,prompt,image_sha,weight_sha,pins):
    import importlib
    for name,digest in pins.items():
        if sha_file(importlib.import_module(name).__file__)!=digest:
            raise RuntimeError("LOADED_TASK_SOURCE_CHANGED:"+name)
    from demoncat.runtime.vllm_ascend.worker_extension.providers.vision17_task import (
        install_vision17_task_binding,remove_vision17_task_binding,Vision17TaskProvider)
    from demoncat.runtime.vllm_ascend.worker_extension.registry import create_worker_provider
    import torch,torch_npu,vllm,vllm_ascend
    worker=actual(worker);identity=worker.demoncat_worker_identity()
    if identity.get('worker')!=expected or identity.get('quarantined') is not False:
        raise RuntimeError('ACTUAL_VISION17_WORKER_IDENTITY_CHANGED')
    environment={name:str(getattr(module,'__version__','unavailable'))
                 for name,module in (('torch',torch),('torch_npu',torch_npu),('vllm',vllm),('vllm_ascend',vllm_ascend))}
    environment['python']=sys.version
    environment_sha=hashlib.sha256(json.dumps(environment,sort_keys=True).encode()).hexdigest()
    result=install_vision17_task_binding(worker,prompt_token_ids=prompt,image_sha256=image_sha,
        weights_sha256=weight_sha,environment_sha256=environment_sha)
    from demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance import install_blue_acceptance_permit
    install_blue_acceptance_permit(worker,mechanisms=['17.spatial_padding'],attempt='ordinary17-unified-query',
        source_manifest_sha256=hashlib.sha256(json.dumps(pins,sort_keys=True).encode()).hexdigest())
    try:
        provider=create_worker_provider('17.spatial_padding','1',worker,worker.model_runner.model,expected)
        if type(provider) is not Vision17TaskProvider:raise RuntimeError('PUBLIC_VISION17_ORDINARY_ROUTE_NOT_INTEGRATED')
    except BaseException:
        remove_vision17_task_binding(worker)
        from demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance import remove_blue_acceptance_permit
        remove_blue_acceptance_permit(worker)
        raise
    return dict(result,environment=environment,environment_sha256=environment_sha)


def remove_binding(worker,expected):
    from demoncat.runtime.vllm_ascend.worker_extension.providers.vision17_task import remove_vision17_task_binding
    worker=actual(worker)
    if worker.demoncat_worker_identity().get('worker')!=expected:raise RuntimeError('VISION17_CLEANUP_WORKER_CHANGED')
    result=remove_vision17_task_binding(worker)
    from demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance import remove_blue_acceptance_permit
    remove_blue_acceptance_permit(worker)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--source-manifest-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--memory',type=float,default=.60)
    args=parser.parse_args()
    if not .1<=args.memory<=.8:parser.error('Bounded memory required')
    args.out.mkdir(parents=True,exist_ok=False)
    report=dict(status='STARTED',tasks=[],requests=[],rpc=[],mm_cache_resets=[],aclnn_launch_arguments=False,
        device_payload_observed=False,native_qualification=False,E5=False,multi_target_verified=False)
    llm=client=None;bound=False;expected=None;pending=[];recorded=0;seen=[];frozen_prompt=None
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
    def enqueue(phase):
        nonlocal pending,frozen_prompt,client,bound,expected
        if llm.llm_engine.has_unfinished_requests():raise RuntimeError('UNRELATED_PENDING_REQUEST')
        if llm.reset_prefix_cache(reset_running_requests=False) is not True:raise RuntimeError('PREFIX_RESET_UNCONFIRMED')
        # Actual phase is retained. The root-owned helper must explicitly admit
        # ordinary phases; no fake "healthy" phase is passed to bypass its guard.
        report['mm_cache_resets'].append(reset_idle_mm_cache(llm,phase,reset_encoder_cache=True));save()
        pending=llm.enqueue([prompt],params,use_tqdm=False)
        mapping=read_enqueued_identity(llm,pending,seen_internal_request_ids=seen);seen.extend(pending)
        tokens=driver_prompt(llm,pending[0])
        if frozen_prompt is None:
            frozen_prompt=tokens
            def prepare(identity,config):
                nonlocal bound,expected
                expected=identity;report['setup']=rpc(setup,identity,tokens,IMAGE_SHA256,WEIGHTS_SHA256,report['source']['pins'])
                bound=True;return {}
            client=WorkerFaultClient(HostAdapter(llm,prepare_binding=prepare,timeout=300),FaultConfig(17,'spatial_padding'))
            for name in ('prepare','register','check','enable'):action(name)
        elif tokens!=frozen_prompt:raise RuntimeError('ACTUAL_PROCESSED_MM_PROMPT_CHANGED')
        action('advance',pending[0]);output=llm.wait_for_completion(use_tqdm=False)
        if len(output)!=1 or output[0].request_id!=mapping['external_request_id']:
            raise RuntimeError('ACTUAL_MM_OUTPUT_MAPPING_CHANGED')
        action('query',expected_status='QUERY_OK')
        report['requests'].append(dict(phase=phase,mapping=mapping,prompt_token_ids=tokens,
            output_token_ids=list(output[0].outputs[0].token_ids),text=output[0].outputs[0].text))
        pending=[];save()
    try:
        weights=args.model/'model.safetensors'
        if weights.stat().st_size!=WEIGHTS_BYTES or sha_file(weights)!=WEIGHTS_SHA256 or sha_file(args.image)!=IMAGE_SHA256:
            raise RuntimeError('PINNED_PRETRAINED_WEIGHTS_AND_NATURAL_IMAGE_REQUIRED')
        report['source']=source_identity(args.source_manifest_sha256)
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
        require_host_driven_core(llm);params=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True)
        report['input']=dict(image_sha256=IMAGE_SHA256,weights_sha256=WEIGHTS_SHA256,geometry=geometry,
            config_sha256=sha_file(args.model/'config.json'),prompt=PROMPT,processor_overrides={},raw_PIL_image=True)
        for scope in range(2):
            first=len(report['requests'])
            for phase in PHASES:enqueue(phase)
            state=action('finish')['response']['provider_state']
            evidence=state['evidence']
            if not evidence['complete'] or not evidence['matched_off_and_health_confirmed']:
                raise RuntimeError('ORDINARY_TASK_RECOVERY_NOT_COMPLETE')
            baseline=report['requests'][first]
            report['tasks'].append(dict(scope=scope,evidence=evidence,
                tokens_changed=[r['output_token_ids']!=baseline['output_token_ids'] for r in report['requests'][first:]],
                cleanup=dict(owned_hooks=state['owned_hooks'],leases=state['leases'],inflight=state['inflight'])))
            action('disable');action('unregister');action('detach',expected_status='DETACHED')
            report['unbind']=rpc(remove_binding,expected);bound=False
            client=None;frozen_prompt=None;recorded=0
        report['status']='PUBLIC_ORDINARY_VISION17_TWO_TASKS_ACCEPTED_TORCH_PARAMETER_ONLY'
    except BaseException as error:
        report.update(status='REFUSED_OR_FAILED',error=repr(error),traceback=traceback.format_exc())
        if llm is not None and pending:
            try:llm.llm_engine.abort_request(pending,internal=True);pending=[]
            except BaseException as cleanup:report['abort_error']=repr(cleanup)
        if client is not None:
            try:
                state=action('recover')['response']['provider_state']
                for _ in range(2):
                    next_phase=state['serving']['next_phase']
                    if not state.get('recovery_validation_pending') or next_phase not in ('off','health'):break
                    enqueue(next_phase);state=action('query',expected_status='QUERY_OK')['response']['provider_state']
                action('unregister');action('detach',expected_status='DETACHED')
                report['unbind']=rpc(remove_binding,expected);bound=False
            except BaseException as cleanup:report['cleanup_error']=repr(cleanup)
    finally:
        report['binding_still_owned']=bound;report['probe_sha256']=sha_file(__file__);save()
    print(json.dumps({k:report[k] for k in ('status','error','fault_injections','binding_still_owned') if k in report}))
    return 0 if report['status']=='PUBLIC_ORDINARY_VISION17_TWO_TASKS_ACCEPTED_TORCH_PARAMETER_ONLY' else 2


if __name__=='__main__':raise SystemExit(main())
