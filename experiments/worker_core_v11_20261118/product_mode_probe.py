"""Acceptance of the product facade. Not the main-project service entry point."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback


def trusted_bootstrap(wrapper, action, options):
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    worker = wrapper.worker if isinstance(wrapper, WorkerWrapperBase) else wrapper
    from demoncat.runtime.vllm_ascend.worker_extension.deployment_authorization import (
        authority_for, configure_deployment_acceptance, install_deployment_acceptance_permit,
        remove_deployment_acceptance_permit)
    authority = authority_for(worker)
    if action == 'configure':
        if authority is None:
            authority = configure_deployment_acceptance(worker, source='BLUE_COMBINATION',
                authorize_operator=lambda principal, operation: principal == 'product-acceptance')
        from demoncat.runtime.vllm_ascend.worker_extension.product_config import configure_product_deployments
        configure_product_deployments(worker, authority=authority, operator='product-acceptance', **options)
        return dict(status='TRUSTED_PROFILES_CONFIGURED')
    if action == 'permit':
        return install_deployment_acceptance_permit(worker, authority=authority, operator='product-acceptance', **options)
    if action == 'remove_permit':
        remove_deployment_acceptance_permit(worker, authority=authority, operator='product-acceptance')
        runtime = worker._demoncat_worker_fault_runtime
        return dict(status='PERMIT_REMOVED', leases=len(runtime._leases), owners=len(runtime._tenants))
    raise ValueError('UNKNOWN_BOOTSTRAP')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mechanism', choices=('18', '39', '22', '28', '17'), nargs='+', required=True)
    parser.add_argument('--model', default='/model')
    parser.add_argument('--evidence', default='/evidence')
    args = parser.parse_args()
    target = Path(args.evidence) / 'product_mode.json'
    if target.exists():
        raise RuntimeError('EVIDENCE_ALREADY_EXISTS')
    report = dict(status='STARTED', scopes=[], product_qualified=False)
    def save(): target.write_text(json.dumps(report, indent=2) + '\n')
    try:
        from vllm import LLM, SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.product_host import ProductClient
        from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
        mechanisms = {'18':'18.api_half', '39':'39.output_proxy', '22':'22.wrong_row', '28':'28.fia.denominator_replay', '17':'17.spatial_padding'}
        if '17' in args.mechanism and len(args.mechanism)!=1:raise ValueError('SEPARATE_VISUAL_INSTANCE_REQUIRED')
        mechanism = mechanisms[args.mechanism[0]]
        number, variant = mechanism.split('.', 1)
        config = FaultConfig(int(number), variant)
        report['mechanism'] = mechanism
        report['source_manifest_sha256'] = hashlib.sha256(Path('/source/source_manifest.json').read_bytes()).hexdigest()
        visual = args.mechanism == ['17']
        extra = dict(limit_mm_per_prompt={'image':1,'video':0},mm_processor_cache_gb=0,mm_encoder_tp_mode='weights',mm_processor_kwargs={'min_pixels':4096,'max_pixels':16384}) if visual else dict(limit_mm_per_prompt={'image':0,'video':0})
        llm = LLM(model=args.model, dtype='bfloat16', quantization='ascend', tensor_parallel_size=2,
            distributed_executor_backend='mp', enforce_eager=True, enable_prefix_caching=False,
            async_scheduling=False, enable_chunked_prefill=True, max_num_batched_tokens=1024,
            max_model_len=1024, max_num_seqs=1, gpu_memory_utilization=.8, seed=42,
            speculative_config=None, kv_transfer_config=None, trust_remote_code=True,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension', **extra)
        tokenizer = llm.get_tokenizer()
        canary = tokenizer.encode('Explain briefly why the sky appears blue.', add_special_tokens=True)
        serial = 0
        def image_input(text, image):
            nonlocal serial
            serial += 1
            return dict(prompt='<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>'+text+'<|im_end|>\n<|im_start|>assistant\n',
                        multi_modal_data={'image':image}, multi_modal_uuids={'image':'product-'+str(serial)})
        images = []
        if visual:
            from PIL import Image
            for shift in (0,23,47):
                image=Image.new('RGB',(64,64))
                image.putdata([((x*4+shift)%256,(y*4+shift)%256,((x+y)*3+shift)%256) for y in range(64) for x in range(64)])
                images.append(image)
            report['image_pixel_sha256']=[hashlib.sha256(image.tobytes()).hexdigest() for image in images]
        def enqueue(value, max_tokens=4):
            llm.sleep(level=0, mode='wait')
            return list(llm.enqueue([value], SamplingParams(temperature=0, max_tokens=max_tokens, min_tokens=max_tokens, ignore_eos=True),use_tqdm=False))
        def complete():
            start=time.monotonic();llm.wake_up(tags=['scheduling']);outputs=llm.wait_for_completion(use_tqdm=False)
            return dict(seconds=time.monotonic()-start,tokens=[list(o.outputs[0].token_ids) for o in outputs])
        primed=None
        if visual:
            primed=enqueue(image_input('Describe this image briefly.',images[0]))
            canary=list(llm.llm_engine.output_processor.request_states[primed[0]].prompt_token_ids)
            images[0].save(Path(args.evidence)/'canary.png')
        deployment_paths={}
        for key in args.mechanism:
            number,variant=mechanisms[key].split('.',1)
            deployment=dict(schema='demoncat.product.deployment.v1', profile=mechanisms[key],
                fault=dict(fault_id=int(number),variant=variant,implementation_version='1'),model=args.model,tp=2,
                mode='PRODUCT_ACCEPTANCE_ONLY',source='BLUE_COMBINATION',source_manifest='/source/source_manifest.json',
                source_manifest_sha256=report['source_manifest_sha256'],native_library=None,native_sha256=None,
                canary=dict(prompt='Describe this image briefly.' if visual else 'Explain briefly why the sky appears blue.',
                    image=str(Path(args.evidence)/'canary.png') if visual else None,max_tokens=4),evidence_directory=args.evidence)
            path=Path(args.evidence)/('deployment-'+key+'.json');path.write_text(json.dumps(deployment,indent=2)+'\n')
            deployment_paths[key]=str(path)
        llm.collective_rpc(trusted_bootstrap, args=('configure', dict(paths=list(deployment_paths.values()),
            canary_tokens={mechanisms[key]:canary for key in args.mechanism})))
        def inference(value,before,*,max_tokens=4):
            ids=enqueue(value,max_tokens);before(ids);return ids,complete()
        def canary_run(spec, before):
            nonlocal primed
            if primed is not None:
                ids=primed;primed=None;before(ids);return complete()
            value=image_input(spec['prompt'],Image.open(spec['image']).convert('RGB')) if visual else dict(prompt_token_ids=tokenizer.encode(spec['prompt'],add_special_tokens=True))
            return inference(value,before,max_tokens=spec['max_tokens'])
        users = ['Name a city with a river.', 'Explain how a bicycle moves.', 'What is seven plus six?']
        for key in args.mechanism:
            mechanism=mechanisms[key]
            number,variant=mechanism.split('.',1);config=FaultConfig(int(number),variant)
            target=Path(args.evidence)/('product_mode_'+key+'.json')
            if target.exists():raise RuntimeError('EVIDENCE_ALREADY_EXISTS')
            report=dict(status='STARTED',mechanism=mechanism,scopes=[],product_qualified=False,
                        source_manifest_sha256=hashlib.sha256(Path('/source/source_manifest.json').read_bytes()).hexdigest(),
                        product_entry='ProductClient.from_deployment',
                        between_request_cache_reset=False,
                        deployment_config_sha256=hashlib.sha256(Path(deployment_paths[key]).read_bytes()).hexdigest())
            if visual:report['image_pixel_sha256']=[hashlib.sha256(image.tobytes()).hexdigest() for image in images]
            for scope in range(2):
                llm.collective_rpc(trusted_bootstrap, args=('permit', dict(mechanisms=[mechanism],
                    attempt='product-'+str(scope), source_manifest_sha256=report['source_manifest_sha256'])))
                client = ProductClient.from_deployment(llm, deployment_paths[key], operator='product-acceptance',
                    run_canary=canary_run, drain=lambda: None)
                row = dict(scope=scope, requests=[]); report['scopes'].append(row); save()
                row['attach'] = client.attach()
                row['enable'] = client.enable({'mode':'next','count':3} if scope==1 else {'mode':'all'}); save()
                for index, text in enumerate(users + ['This request exceeds the validated generation budget.']):
                    value=image_input(text,images[1+index%2]) if visual else dict(prompt_token_ids=tokenizer.encode(text,add_special_tokens=True))
                    ids, timing = inference(value,
                        lambda ids: client.before_request(ids[0]), max_tokens=5 if index==3 else 4)
                    result = client.after_request(ids[0])
                    for rank in result['ranks']:
                        if index == 3:
                            assert rank['status']=='NOT_INJECTED' and rank['coverage']['applied']==0 and rank['reasons']
                        else:
                            c=rank['coverage'];assert c['eligible']>0 and c['eligible']==c['applied']==c['consumed']==c['restored'] and c['failed']==0
                    row['requests'].append(dict(result=result, **timing)); save()
                if scope == 1:
                    # Isolate next-N expiry from the separate generation-range negative.
                    value=image_input('Name one visible color.',images[1]) if visual else dict(prompt_token_ids=tokenizer.encode('Name one musical instrument.',add_special_tokens=True))
                    ids,timing=inference(value,lambda ids:client.before_request(ids[0]))
                    result=client.after_request(ids[0])
                    assert all(r['status']=='NOT_INJECTED' and r['coverage']['applied']==0
                        and r['reasons']==['SCOPE_REQUEST_LIMIT_REACHED'] for r in result['ranks'])
                    row['requests'].append(dict(result=result,**timing));save()
                row['disable'] = client.disable(); save()
                row['detach'] = client.detach()
                row['cleanup'] = llm.collective_rpc(trusted_bootstrap, args=('remove_permit', {})); save()
                assert all(x['leases']==x['owners']==0 for x in row['cleanup'])
            report['status']='PRODUCT_TWO_SCOPES_COMPLETED_PENDING_REVIEW'
            save()
    except BaseException as error:
        report.update(status='FAILED', error=repr(error), traceback=traceback.format_exc())
    save()
    print(json.dumps(dict(status=report['status'], error=report.get('error'))), flush=True)
    return 0 if report['status'].startswith('PRODUCT_TWO') else 2


if __name__ == '__main__': raise SystemExit(main())
