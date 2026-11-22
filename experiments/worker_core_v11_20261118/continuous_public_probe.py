"""Public Runtime TP1/TP2 acceptance. Trusted bootstrap only installs inert bindings.

Inference remains llm.enqueue/wait_for_completion. No experiment injection RPC
is used. Raw receipts remain local; this script grants no deployment record.
"""
import argparse,hashlib,json,time,traceback
from pathlib import Path


def bootstrap(wrapper,action,mechanism,options=None):
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    worker=wrapper.worker if isinstance(wrapper,WorkerWrapperBase) else wrapper
    from demoncat.runtime.vllm_ascend.worker_extension.continuous_binding import (
        install_continuous_binding,remove_continuous_binding,observe_continuous_warmup)
    from demoncat.runtime.vllm_ascend.worker_extension.deployment_authorization import (
        authority_for,configure_deployment_acceptance,install_deployment_acceptance_permit,
        remove_deployment_acceptance_permit)
    if action=='bind':return install_continuous_binding(worker,mechanism=mechanism,**options)
    if action=='warmup':return observe_continuous_warmup(worker,mechanism,request_ids=options)
    if action=='permit':
        options=dict(options);source=options.pop('source')
        authority=authority_for(worker)
        if authority is None:
            authority=configure_deployment_acceptance(worker,source=source,
                authorize_operator=lambda principal,operation:principal=='rc3-acceptance-harness')
        assert authority.source==source,'AUTHORITY_SOURCE_CHANGED'
        return install_deployment_acceptance_permit(worker,authority=authority,
            operator='rc3-acceptance-harness',mechanisms=[mechanism],**options)
    if action=='diagnostic':
        from collections import Counter
        runtime=worker._demoncat_worker_fault_runtime;result=[]
        for lease in runtime._leases.values():
            provider=lease.provider;session=getattr(provider,'session',None);site=getattr(session,'site',None)
            events=getattr(site,'events',[])
            result.append(dict(error=str(getattr(provider,'error',None))[:500],
                session_error=str(getattr(session,'error',None))[:500],
                samples=getattr(session,'samples',None),events=len(events),
                eligible=sum(bool(e.get('eligible')) for e in events),
                applied=sum(bool(e.get('applied')) for e in events),
                consumed=sum(bool(e.get('consumed')) for e in events),
                restored=sum(bool(e.get('restored')) for e in events),
                changed_calls=sum(e.get('changed',0)>0 for e in events),
                partition_changed=sum(e.get('partition_reference',{}).get('partition_changed_elements',0) for e in events),
                selected_changed=sum(e.get('partition_reference',{}).get('selected_changed_elements',0) for e in events),
                reasons=dict(Counter(e.get('reason') for e in events if e.get('reason')))))
        return dict(rank=worker.rank,read_only=True,providers=result)
    if action=='cleanup':
        remove_deployment_acceptance_permit(worker,authority=authority_for(worker),operator='rc3-acceptance-harness')
        remove_continuous_binding(worker,mechanism)
        runtime=worker._demoncat_worker_fault_runtime
        return dict(rank=worker.rank,status='REMOVED',permit_remaining=hasattr(worker,'_demoncat_blue_acceptance_permit'),
                    runtime_leases=len(runtime._leases),runtime_tenants=len(runtime._tenants))
    if action=='failure_cleanup':
        from demoncat.runtime.vllm_ascend.worker_extension.continuous_binding import current_binding
        runtime=getattr(worker,'_demoncat_worker_fault_runtime',None)
        b=current_binding(worker,worker.model_runner.model,mechanism)
        if b is not None and b.warmup_request is not None:
            b.warmup_request.close();b.warmup_request=None
        leases=len(getattr(runtime,'_leases',{}));owners=len(getattr(runtime,'_tenants',{}))
        if leases or owners:return dict(rank=worker.rank,clean=False,leases=leases,owners=owners)
        if hasattr(worker,'_demoncat_blue_acceptance_permit'):
            remove_deployment_acceptance_permit(worker,authority=authority_for(worker),operator='rc3-acceptance-harness')
        if b is not None:remove_continuous_binding(worker,mechanism)
        return dict(rank=worker.rank,clean=True,leases=0,owners=0,permit_remaining=False)
    if action=='negative':
        from demoncat.runtime.vllm_ascend.worker_extension.ordinary_admission import ordinary_route_report
        return ordinary_route_report(mechanism,worker,worker.model_runner.model)
    raise ValueError('UNKNOWN_BOOTSTRAP_ACTION')


def main():
    p=argparse.ArgumentParser();p.add_argument('--mechanism',choices=('28','batch','33','39','39mr','18','22','8','17','41','26fia','20'),required=True)
    p.add_argument('--native-library');p.add_argument('--native-sha256')
    p.add_argument('--image');p.add_argument('--image-sha256')
    p.add_argument('--paged-length',type=int,choices=(768,2304),default=768)
    p.add_argument('--paged-fault-block',type=int,choices=(8,16,32),default=32)
    p.add_argument('--tp',type=int,choices=(1,2),default=2)
    p.add_argument('--reference-mode',choices=('immediate','forward'),default='immediate')
    p.add_argument('--model-family',choices=('qwen35','deepseek_v2'),default='qwen35')
    p.add_argument('--arithmetic-device',choices=('input','cpu'),default='input')
    p.add_argument('--model',default='/model');p.add_argument('--evidence',default='/evidence')
    p.add_argument('--source-manifest',default='/source/source_manifest.json')
    p.add_argument('--deployment-source',choices=('BLUE_COMBINATION','YELLOW_DEDICATED_INSTANCE'),default='BLUE_COMBINATION')
    a=p.parse_args()
    mechanism={'41':'41.tail_column','26fia':'26.fia_destination','20':'20.buffer_substitute','17':'17.spatial_padding','batch':'quant.batch_shared_scale','28':'28.fia.denominator_replay','33':'33.paged_decode','39':'39.output_proxy','39mr':'39.matmul_range','18':'18.api_half','22':'22.wrong_row','8':'8.restore_offset'}[a.mechanism]
    from fault_injection.rc3_preview_profile import TEST_CANDIDATES
    if a.deployment_source=='YELLOW_DEDICATED_INSTANCE' and mechanism not in TEST_CANDIDATES:
        raise RuntimeError('RC3_PREVIEW_OUTSIDE_DELIVERY_SCOPE:'+mechanism)
    batch=8 if a.mechanism=='batch' else 1
    report=dict(status='STARTED',mechanism=mechanism,scopes=[],warmup=[],qualification_granted=False,
        acceptance_scope='PUBLIC_WORKER_RUNTIME',arithmetic_device=a.arithmetic_device,reference_mode=a.reference_mode)
    evidence=Path(a.evidence);evidence.mkdir(parents=True,exist_ok=True)
    out=evidence/'continuous_public.json'
    if out.exists():raise RuntimeError('EVIDENCE_MUST_NOT_BE_OVERWRITTEN')
    def save():out.write_text(json.dumps(report,indent=2)+'\n')
    client=None
    try:
        from vllm import LLM,SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
        from demoncat.runtime.vllm_ascend.worker_extension.multi_rank import MultiRankTaskClient
        from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter,WorkerFaultClient
        from demoncat.runtime.vllm_ascend.worker_extension.continuous_binding import digest
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import actual_driver_request_map
        extra=dict(limit_mm_per_prompt={'image':0,'video':0}) if a.model_family=='qwen35' else {}
        if a.mechanism=='17':
            extra=dict(limit_mm_per_prompt={'image':1,'video':0},mm_processor_cache_gb=0,mm_encoder_tp_mode='weights',
                mm_processor_kwargs={'min_pixels':4096,'max_pixels':16384})
        llm=LLM(model=a.model,dtype='bfloat16',quantization='ascend',tensor_parallel_size=a.tp,
            distributed_executor_backend='mp',enforce_eager=True,enable_prefix_caching=False,
            async_scheduling=False,enable_chunked_prefill=True,max_num_batched_tokens=4096 if a.mechanism=='33' and a.paged_length>1024 else 1024,
            max_model_len=4096 if a.mechanism=='33' and a.paged_length>1024 else 1024,max_num_seqs=batch,gpu_memory_utilization=.8,seed=42,
            speculative_config=None,kv_transfer_config=None,trust_remote_code=True,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension',**extra)
        seeds=['Explain in one sentence why the sky appears blue.','Explain in one sentence why the grass appears green.',
            'Summarize the purpose of a computer in one sentence.','What makes a stone sink in water?',
            'Explain why snow appears white in daylight.','Give one example of a simple machine.',
            'What is the sum of four and five?','Name one planet in the solar system.']
        tokens=[llm.get_tokenizer().encode(s,add_special_tokens=True) for s in seeds[:batch]]
        if a.mechanism=='33':
            tokens=[(tokens[0]*((a.paged_length+len(tokens[0])-1)//len(tokens[0])))[:a.paged_length]]
            report['paged_deployment']=dict(prompt_tokens=a.paged_length,control_block=128,fault_block=a.paged_fault_block,accumulation='FP32')
        image_prompt=None;serial=0;primed=None
        if a.mechanism=='17':
            from PIL import Image
            from fault_injection.vision_input import load_fixed_image
            if a.image:
                image,identity=load_fixed_image(a.image,a.image_sha256)
            else:
                if a.deployment_source=='YELLOW_DEDICATED_INSTANCE':raise RuntimeError('YELLOW_FIXED_IMAGE_REQUIRED')
                image=Image.new('RGB',(64,64));image.putdata([(x*4,y*4,((x+y)%64)*4) for y in range(64) for x in range(64)])
                identity=dict(source='BLUE_GENERATED_FIXTURE',size=[64,64],pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest())
            report['visual_input']=identity
            image_prompt='<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image in one sentence.<|im_end|>\n<|im_start|>assistant\n'
            tokens=[]
        params=SamplingParams(temperature=0,max_tokens=4,min_tokens=4,ignore_eos=True)
        source_sha=hashlib.sha256(Path(a.source_manifest).read_bytes()).hexdigest()
        report['source_manifest_sha256']=source_sha
        report['request_seconds_limit']=120
        report['warmup_seconds_limit']=600
        report['warmup_max_requests']=6
        def rpc(action,options=None):
            print(json.dumps(dict(progress='rpc_begin',action=action)),flush=True)
            result=llm.collective_rpc(bootstrap,args=(action,mechanism,options),timeout=120)
            print(json.dumps(dict(progress='rpc_end',action=action)),flush=True)
            return result
        def enqueue():
            nonlocal serial
            assert llm.reset_prefix_cache(reset_running_requests=False)
            llm.sleep(level=0,mode='wait')
            inputs=[dict(prompt_token_ids=t) for t in tokens]
            if image_prompt is not None:
                serial+=1;inputs=[dict(prompt=image_prompt,multi_modal_data={'image':image},multi_modal_uuids={'image':'vision17-public-'+str(serial)})]
            ids=list(llm.enqueue(inputs,params,use_tqdm=False))
            if image_prompt is not None:
                state=llm.llm_engine.output_processor.request_states[ids[0]]
                actual=list(state.prompt_token_ids);assert state.request_id==ids[0]
                if not tokens:tokens.append(actual)
                else:assert tokens==[actual],'ACTUAL_IMAGE_PROMPT_CHANGED'
            mapping=actual_driver_request_map(llm.llm_engine.output_processor)
            assert len(mapping)==len(ids)==batch and {r['internal_request_id'] for r in mapping}==set(ids)
            return ids,mapping
        def run(mapping, *, warmup=False):
            start=time.monotonic();llm.wake_up(tags=['scheduling'])
            outputs=llm.wait_for_completion(use_tqdm=False)
            assert {o.request_id for o in outputs}=={r['external_request_id'] for r in mapping}
            elapsed=time.monotonic()-start
            limit=600 if warmup else 120
            assert elapsed<=limit,('CONTINUOUS_WARMUP_TIME_BUDGET_EXCEEDED' if warmup else 'CONTINUOUS_REQUEST_EXPERIMENT_TIME_BUDGET_EXCEEDED')
            return dict(seconds=elapsed,tokens=[list(o.outputs[0].token_ids) for o in outputs])
        if image_prompt is not None:primed=enqueue()
        for scope_index in range(2):
            number,variant=mechanism.split('.',1);config=FaultConfig(number if number=='quant' else int(number),variant)
            client=MultiRankTaskClient(llm,config,prepare_binding=lambda *_:{},continuous=True,world_size=a.tp,
                batch_size=batch,timeout=120)
            contract=client.contract()
            members=sorted([r['worker'] for r in llm.collective_rpc('demoncat_worker_identity')],key=lambda r:r['rank'])
            contract.update(members=members,members_sha256=digest(members))
            rpc('bind',dict(prompts=tokens,group_contract=contract,arithmetic_device=a.arithmetic_device,reference_mode=a.reference_mode,
                **(dict(paged_partition=[128,a.paged_fault_block]) if a.mechanism=='33' else {}),
                **(dict(native_library=[a.native_library,a.native_sha256]) if a.mechanism in ('41','26fia','20') else {})))
            negative=rpc('negative');assert all(not r['can_prepare'] for r in negative),'MISSING_PERMIT_ADMITTED'
            for attempt in range(6):
                if primed is not None:ids,mapping=primed;primed=None
                else:ids,mapping=enqueue()
                rpc('warmup',ids);run(mapping,warmup=True)
                warm=rpc('warmup');report['warmup'].append(dict(scope=scope_index,attempt=attempt,ranks=warm));save()
                if all(r['warmup_valid'] for r in warm):break
            else:raise RuntimeError('BASELINE_NOT_STABLE_IN_SIX')
            negative_creation=[]
            for member in members:
                host=HostAdapter(llm,select_worker=lambda config,rows,m=member:m,
                    prepare_binding=lambda *_:{},timeout=120)
                denied=WorkerFaultClient(host,config)
                creation=denied.prepare()
                assert creation['status'] in ('PROVIDER_UNAVAILABLE','REJECTED'),creation
                assert denied.snapshot()['ticket'] is None,creation
                detached=denied.detach();assert detached['status']=='DETACHED',detached
                negative_creation.append(dict(rank=member['rank'],prepare=creation,detach=detached,
                    snapshot=denied.snapshot()))
            rpc('permit',dict(attempt='continuous-public-'+str(scope_index),source_manifest_sha256=source_sha,
                              source=a.deployment_source))
            scope=dict(owner=contract['owner'],requests=[],negative_missing_permit=negative,
                       negative_missing_permit_creation=negative_creation)
            report['scopes'].append(scope);save()
            for action in ('prepare','register','check','enable'):
                result=getattr(client,action)();scope[action]=result;save()
                assert result['status']=='ACTION_COMPLETED',(action,result)
            from fault_injection.rc3_preview_profile import next_request_phase
            state=client.query();plan=state['response']['provider_state']['request_plan'];completed=[]
            while True:
                phase=next_request_phase(state,plan,completed)
                if phase=='complete':break
                ids,mapping=enqueue()
                result=client.advance_batch(ids) if batch>1 else client.advance(ids[0])
                assert result['status']=='ACTION_COMPLETED',result
                timing=run(mapping)
                if phase=='fault':scope['fault_diagnostic']=rpc('diagnostic');save()
                result=client.query()
                if result['status']!='QUERY_OK':scope['failure_diagnostic']=rpc('diagnostic');save()
                scope['requests'].append(dict(phase=phase,mapping=mapping,result=result,**timing));save()
                assert result['status']=='QUERY_OK',result
                completed.append(phase);state=result
            extra=client.advance_batch(['extra'+str(i) for i in range(batch)]) if batch>1 else client.advance('extra')
            assert extra['status']=='GROUP_PHASES_EXHAUSTED';scope['negative_extra_advance']=extra['status']
            for action in ('finish','unregister','detach'):
                result=getattr(client,action)();scope[action]=result;save()
                assert result['status']==('DETACHED' if action=='detach' else 'ACTION_COMPLETED'),result
            scope['snapshot']=client.snapshot();scope['cleanup']=rpc('cleanup')
            scope['fault_time_ratio']=scope['requests'][1]['seconds']/scope['requests'][0]['seconds'];save()
        report['status']='PUBLIC_TWO_SCOPES_COMPLETED_PENDING_INDEPENDENT_REVIEW'
    except BaseException as e:
        report.update(status='FAILED',error=repr(e),traceback=traceback.format_exc())
        if client is not None:
            report['failure_snapshot']=client.snapshot()
            try:report['recovery']=client.recover()
            except Exception as cleanup:report['recovery_error']=repr(cleanup)
        try:report['failure_cleanup']=rpc('failure_cleanup')
        except Exception as cleanup:report['failure_cleanup_error']=repr(cleanup)
    save();print(json.dumps(dict(status=report['status'],error=report.get('error'))),flush=True)
    return 0 if report['status'].startswith('PUBLIC_TWO') else 2


if __name__=='__main__':raise SystemExit(main())
