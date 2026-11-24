"""Isolated localhost HTTP acceptance, never a production bootstrap endpoint."""
import argparse,base64,hashlib,io,json,os,subprocess,sys,time,traceback,urllib.request
from pathlib import Path
from demoncat.runtime.vllm_ascend.controller import DemoncatWorkerExtension


class AcceptanceExtension(DemoncatWorkerExtension):
    def demoncat_acceptance_bootstrap(self,action,options):
        from product_mode_probe import trusted_bootstrap
        if action=='capture_start':
            runner=self.model_runner
            assert 'execute_model' not in vars(runner)
            original=runner.execute_model;self._http_canary=[]
            def execute(scheduler,*args,**kwargs):
                for r in scheduler.scheduled_new_reqs:self._http_canary.append(dict(id=r.req_id,tokens=r.prompt_token_ids))
                return original(scheduler,*args,**kwargs)
            runner.execute_model=execute;self._http_capture=execute
            return dict(status='CAPTURE_ARMED')
        if action=='capture_stop':
            assert self.model_runner.execute_model is self._http_capture
            del self.model_runner.execute_model
            assert 1<=len(self._http_canary)<=8
            return self._http_canary
        return trusted_bootstrap(self,action,options)


class HTTPExecutor:
    def __init__(self,base):self.base=base
    def post(self,path,payload,timeout=900):
        raw=json.dumps(payload).encode()
        req=urllib.request.Request(self.base+path,data=raw,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=timeout) as response:return json.load(response)
    def collective_rpc(self,method,*,args=(),kwargs=None,timeout=120):
        reply=self.post('/collective_rpc',dict(method=method,args=args,kwargs=kwargs or {},timeout=timeout),timeout=timeout+30)
        return reply['results'] if isinstance(reply,dict) and 'results' in reply else reply


def main():
    p=argparse.ArgumentParser();p.add_argument('--mechanism',nargs='+',choices=['15','18','39','22','28','17','batch','39mr','33','41','26fia','20'],required=True)
    p.add_argument('--negative-regression',action='store_true')
    p.add_argument('--batch-canary-negative',action='store_true')
    p.add_argument('--effect-probe',action='store_true')
    p.add_argument('--external-example',action='store_true')
    p.add_argument('--external-sampling',action='store_true')
    p.add_argument('--external-vision',action='store_true')
    p.add_argument('--variance18',action='store_true')
    p.add_argument('--status-diagnostics',action='store_true')
    p.add_argument('--image')
    p.add_argument('--native-library');p.add_argument('--native-sha256')
    a=p.parse_args();visual=a.mechanism==['17'];batch=a.mechanism==['batch'];paged=a.mechanism==['33'];root=Path('/evidence')
    if a.external_vision and not visual:p.error('External vision uses the isolated image scheduler: --mechanism 17')
    if a.variance18 and a.mechanism!=['18']:p.error('Variance proxy requires --mechanism 18')
    if sum((a.external_example,a.external_sampling,a.external_vision,a.variance18))>1:p.error('Choose one adapter variant')
    if a.batch_canary_negative and not batch:p.error('Nonserial canary negative requires Batch')
    if ('batch' in a.mechanism or '33' in a.mechanism) and len(a.mechanism)!=1:p.error('Batch/33 use their own scheduler profile')
    report=dict(status='STARTED',entry='VLLM_HTTP_API',host_per_request_hooks=False,scopes=[])
    target=root/'product_http.json'
    def save():target.write_text(json.dumps(report,indent=2)+'\n')
    server=None
    try:
        cmd=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model','/model','--served-model-name','acceptance',
            '--host','127.0.0.1','--port','18937','--dtype','bfloat16','--quantization','ascend',
            '--tensor-parallel-size','2','--distributed-executor-backend','mp','--enforce-eager',
            '--no-enable-prefix-caching','--no-async-scheduling','--enable-chunked-prefill',
            '--max-num-batched-tokens','8192' if batch else '4096' if paged else '1024','--max-model-len','4096' if paged else '1024','--max-num-seqs','8' if batch else '1',
            '--gpu-memory-utilization','0.8','--seed','42','--trust-remote-code',
            '--worker-extension-cls','product_http_probe.AcceptanceExtension',
            '--limit-mm-per-prompt',json.dumps(dict(image=1 if visual else 0,video=0))]
        if visual:cmd += ['--mm-processor-cache-gb','0','--mm-encoder-tp-mode','weights',
            '--mm-processor-kwargs',json.dumps(dict(min_pixels=4096,max_pixels=16384))]
        server=subprocess.Popen(cmd,env=dict(os.environ,VLLM_WORKER_MULTIPROC_METHOD='spawn'),stdout=(root/'http-server.log').open('w'),stderr=subprocess.STDOUT)
        ex=HTTPExecutor('http://127.0.0.1:18937')
        for _ in range(360):
            if server.poll() is not None:raise RuntimeError('HTTP_SERVER_START_FAILED')
            try:
                with urllib.request.urlopen(ex.base+'/health',timeout=2):break
            except Exception:time.sleep(1)
        else:raise RuntimeError('HTTP_SERVER_READY_TIMEOUT')
        from demoncat.runtime.vllm_ascend.worker_extension.product_host import ProductClient
        if a.status_diagnostics:
            from demoncat.runtime.vllm_ascend.worker_extension.product_diagnostics import DiagnosticProductClient as ProductClient
        mechanisms={'15':'15.alias_detach','18':'18.api_half','39':'39.output_proxy','22':'22.wrong_row','28':'28.fia.denominator_replay','17':'17.spatial_padding','batch':'quant.batch_shared_scale','39mr':'39.matmul_range','33':'33.paged_decode','41':'41.tail_column','26fia':'26.fia_destination','20':'20.buffer_substitute'}
        if a.external_example:mechanisms['18']='external.gain_three_quarters'
        if a.external_sampling:mechanisms['18']='external.exponential_perturb'
        if a.external_vision:mechanisms['17']='external.conv3d_output_gain'
        if a.variance18:mechanisms['18']='18.variance_over4'
        images=[]
        if visual:
            from PIL import Image
            for shift in (0,23,47):
                image=Image.new('RGB',(64,64));image.putdata([((x*4+shift)%256,(y*4+shift)%256,((x+y)*3+shift)%256) for y in range(64) for x in range(64)])
                if shift==0:
                    from fault_injection.vision_input import deterministic_image,load_fixed_image
                    image,visual_identity=load_fixed_image(a.image) if a.image else deterministic_image()
                buf=io.BytesIO();image.save(buf,format='PNG');images.append('data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode())
                if shift==0:image.save(root/'canary.png')
        def inference(text,*,image=0,max_tokens=4,temperature=0):
            common=dict(model='acceptance',temperature=temperature,seed=42,max_tokens=max_tokens,min_tokens=max_tokens,ignore_eos=True)
            if visual:
                payload=dict(common,messages=[dict(role='user',content=[dict(type='image_url',image_url={'url':images[image]}),dict(type='text',text=text)])])
                endpoint='/v1/chat/completions'
            else:payload=dict(common,prompt=text);endpoint='/v1/completions'
            start=time.monotonic();reply=ex.post(endpoint,payload)
            return dict(seconds=time.monotonic()-start,api_id=reply.get('id'),choices=reply['choices'])
        canary_text='Describe this image briefly.' if visual else 'Explain briefly why the sky appears blue.'
        if batch:canary_text=['Explain briefly this subject: '+t for t in ['rain','rivers','clouds','wind','bicycles','forests','mountains','ocean']]
        if paged:
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained('/model',local_files_only=True,trust_remote_code=True)
            def long_prompt(seed):
                ids=tokenizer.encode((seed+' ')*3000,add_special_tokens=False)[:2304]
                text=tokenizer.decode(ids,skip_special_tokens=False)
                assert len(tokenizer.encode(text,add_special_tokens=False))==2304
                return text
            canary_text=long_prompt('Explain rivers and mountains.')
        ex.collective_rpc('demoncat_acceptance_bootstrap',args=('capture_start',{}))
        if batch:
            for prompt in canary_text:inference(prompt)
        else:inference(canary_text)
        captures=ex.collective_rpc('demoncat_acceptance_bootstrap',args=('capture_stop',{}))
        assert len(captures)==2 and [r['tokens'] for r in captures[0]]==[r['tokens'] for r in captures[1]]
        canary=[r['tokens'] for r in captures[0]] if batch else captures[0][0]['tokens'];paths={}
        source_sha=hashlib.sha256(Path('/source/source_manifest.json').read_bytes()).hexdigest()
        for key in a.mechanism:
            number,variant=mechanisms[key].split('.',1)
            config=dict(schema='demoncat.product.deployment.v1',profile=mechanisms[key],
                fault=dict(fault_id=number if number in ("quant","external") else int(number),variant=variant,implementation_version='1'),model='/model',tp=2,
                mode='PRODUCT_ACCEPTANCE_ONLY',source='BLUE_COMBINATION',source_manifest='/source/source_manifest.json',
                source_manifest_sha256=source_sha,native_library=a.native_library,native_sha256=a.native_sha256,
                canary=dict(prompt=canary_text,image=a.image if visual else None,max_tokens=4),evidence_directory='/evidence')
            path=root/('deployment-'+key+'.json');path.write_text(json.dumps(config));paths[key]=str(path)
        if a.external_example or a.external_sampling or a.external_vision:
            ex.collective_rpc('demoncat_acceptance_bootstrap',args=('external_configure',dict(canary_tokens=canary,sampling=a.external_sampling,vision=a.external_vision)))
        else:ex.collective_rpc('demoncat_acceptance_bootstrap',args=('configure',dict(paths=list(paths.values()),canary_tokens={mechanisms[k]:canary for k in a.mechanism})))
        for key in a.mechanism:
            mechanism=mechanisms[key];target=root/('product_http_'+key+'.json')
            report=dict(status='STARTED',entry='VLLM_HTTP_API',host_per_request_hooks=False,mechanism=mechanism,scopes=[],source_manifest_sha256=source_sha)
            if visual:
                report['visual_input']=visual_identity
                report['image_pixel_sha256']=[hashlib.sha256(Image.open(io.BytesIO(base64.b64decode(data.split(',',1)[1]))).convert('RGB').tobytes()).hexdigest() for data in images]
            if a.negative_regression:
                from demoncat.runtime.vllm_ascend.worker_extension.product_errors import ProductContractRefused,product_control_reply
                negative = dict(mechanism=mechanism, expected_exit=4)
                target_negative = root/'product_control_negative.json'
                unknown=ex.collective_rpc('demoncat_product_binding',args=('capabilities',dict(operator='product-acceptance',profile='unknown-profile')))
                assert all(r['status']=='PRODUCT_BINDING_REFUSED' and r['reason']=='PRODUCT_PROFILE_UNKNOWN' for r in unknown)
                negative['unknown_profile']=unknown
                unknown_client=ProductClient(ex,__import__('demoncat.runtime.vllm_ascend.worker_extension.config',fromlist=['FaultConfig']).FaultConfig(18,'api_half'),profile='unknown-profile',operator='product-acceptance',run_canary=lambda:None,drain=lambda:None)
                negative['unknown_profile_cli']=product_control_reply(unknown_client.capabilities)
                assert negative['unknown_profile_cli']['exit_code']==4 and 'PRODUCT_PROFILE_UNKNOWN' in negative['unknown_profile_cli']['refusal_codes']
                # capabilities census acquired no Runtime attachment or lease.
                assert not unknown_client.bound
                from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
                negative['cli_99']=product_control_reply(lambda:FaultConfig(99,'unknown'))
                assert negative['cli_99']['exit_code']==4
                negative['healthy_after_unknown']=inference(canary_text)
                rejected=ProductClient.from_deployment(ex,paths[key],operator='product-acceptance',
                    run_canary=lambda spec:inference(spec['prompt']),drain=lambda:None,timeout=120)
                negative['attach']=rejected.attach()
                negative['missing_permit_cli']=product_control_reply(rejected.enable)
                assert negative['missing_permit_cli']['exit_code']==4 and negative['missing_permit_cli']['error_type']=='ProductContractRefused'
                assert 'ORDINARY_ROUTE_NOT_ACCEPTED_FOR_DEPLOYMENT' in negative['missing_permit_cli']['refusal_codes']
                negative['missing_permit']=negative['missing_permit_cli']['reason']
                assert not rejected.bound and rejected.recovery['resources_zero']
                negative['rollback']=rejected.cleanup_evidence
                negative['disable_after_rollback']=rejected.disable()
                negative['detach_after_rollback']=rejected.detach()
                negative['healthy_after_rejection']=inference(canary_text)
                negative['resources']=ex.collective_rpc('demoncat_acceptance_bootstrap',args=('audit',{}))
                assert all(r['leases']==r['owners']==r['bindings']==r['wrappers']==0 for r in negative['resources'])
                target_negative.write_text(json.dumps(negative,indent=2)+'\n')
            if a.batch_canary_negative:
                from demoncat.runtime.vllm_ascend.worker_extension.product_errors import product_control_reply
                negative=dict(mechanism=mechanism,expected_exit=4)
                rejected=ProductClient.from_deployment(ex,paths[key],operator='product-acceptance',run_canary=lambda spec:inference(spec['prompt']),drain=lambda:None,timeout=120)
                negative['attach']=rejected.attach()
                negative['nonserial_canary']=product_control_reply(rejected.enable)
                assert negative['nonserial_canary']['exit_code']==4 and 'PRODUCT_BATCH_CANARY_REQUIRES_SERIAL_B1' in negative['nonserial_canary']['refusal_codes']
                negative['rollback']=rejected.cleanup_evidence
                assert not rejected.bound and rejected.recovery['resources_zero']
                negative['healthy_inference_after_refusal']=inference(canary_text[0])
                with urllib.request.urlopen(ex.base+'/health',timeout=10) as response:negative['health_http']=response.status
                negative['resources']=ex.collective_rpc('demoncat_acceptance_bootstrap',args=('audit',{}))
                assert negative['health_http']==200 and all(r['leases']==r['owners']==r['bindings']==r['wrappers']==0 for r in negative['resources'])
                (root/'product_batch_canary_negative.json').write_text(json.dumps(negative,indent=2)+'\n')
                rejected.run_canary=lambda:[inference(p) for p in canary_text]
            for index in range(2):
                ex.collective_rpc('demoncat_acceptance_bootstrap',args=('permit',dict(mechanisms=[mechanism],attempt='http-'+str(index),source_manifest_sha256=source_sha)))
                if a.external_example or a.external_sampling or a.external_vision:
                    from demoncat.runtime.vllm_ascend.worker_extension.external_provider import ExternalFaultConfig
                    client=ProductClient(ex,ExternalFaultConfig('external',mechanism.split('.',1)[1]),profile=mechanism,operator='product-acceptance',run_canary=lambda:inference(canary_text),drain=lambda:None,timeout=120)
                else:
                    client=ProductClient.from_deployment(ex,paths[key],operator='product-acceptance',
                        run_canary=lambda spec:[inference(p) for p in spec['prompt']] if batch else inference(spec['prompt']),drain=lambda:None,timeout=120)
                if (a.negative_regression or a.batch_canary_negative) and index==0:client=rejected
                scope=dict(scope=index,requests=[]);report['scopes'].append(scope)
                scope['attach']=negative['attach'] if (a.negative_regression or a.batch_canary_negative) and index==0 else client.attach();scope['workers']=client.members;scope['capabilities']=client.capabilities();scope['enable']=client.enable();save()
                if a.negative_regression and index==0:
                    scope['duplicate_enable']=product_control_reply(client.enable)
                    assert scope['duplicate_enable']['exit_code']==4 and client.enabled and client.bound
                if batch:
                    from concurrent.futures import ThreadPoolExecutor
                    from threading import Barrier
                    seen=0
                    for count in (2,4,8):
                        barrier=Barrier(count)
                        def concurrent_call(n):
                            barrier.wait()
                            return inference('Explain briefly '+['weather','cities','plants','rivers','physics','music','bikes','oceans'][n]+' using simple words.')
                        with ThreadPoolExecutor(max_workers=count) as pool:timings=list(pool.map(concurrent_call,range(count)))
                        status=client.status();new=status['requests'][seen:];assert len(new)==count
                        for result in new:
                            scope['requests'].append(dict(result=result,scenario='concurrent',submitted_concurrency=count,api_group_timings=timings,expected_injection=True))
                        seen+=count;save()
                    for text,max_tokens in [('One standalone request should remain unmodified.',4),('This request exceeds the generation scope.',5)]:
                        timing=inference(text,max_tokens=max_tokens);status=client.status();assert len(status['requests'])==seen+1
                        scope['requests'].append(dict(result=status['requests'][-1],scenario='single' if max_tokens==4 else 'range_negative',expected_injection=False,**timing));seen+=1;save()
                for n,text in ([] if batch else enumerate([long_prompt(t) for t in ['Rivers and cities.','Bicycles and gears.','Numbers and arithmetic.','Outside validated generation.']] if paged else (['Name a city with a river.','Explain how a bicycle moves.','What is seven plus six?','This request exceeds the validated generation budget.']+(['Greedy sampling should not enter exponential noise.'] if a.external_sampling else [])))):
                    timing=inference(text,image=1+n%2,max_tokens=5 if n==3 else 4,temperature=0.7 if a.external_sampling and n<4 else 0)
                    status=client.status();result=status['requests'][-1]
                    assert len(status['requests'])==n+1
                    for r in result['ranks']:
                        c=r['coverage']
                        if n>=3:assert r['status']=='NOT_INJECTED' and c['applied']==0 and r['reasons']
                        else:assert c['eligible']>0 and c['eligible']==c['applied']==c['consumed']==c['restored'] and c['failed']==0
                    scope['requests'].append(dict(result=result,**timing));save()
                scope['disable']=client.disable();scope['detach']=client.detach()
                if a.negative_regression and index==0:
                    scope['repeat_disable']=product_control_reply(client.disable)
                    assert scope['repeat_disable']['exit_code'] in (0,4) and client.recovery['restored'] and not client.enabled
                scope['cleanup']=ex.collective_rpc('demoncat_acceptance_bootstrap',args=('remove_permit',{}));save()
                assert all(r['leases']==r['owners']==0 for r in scope['cleanup'])
                scope['negative_controls']=ex.collective_rpc('demoncat_acceptance_bootstrap',args=('negative',dict(mechanism=mechanism)));save()
            report['status']='HTTP_PRODUCT_TWO_SCOPES_PENDING_REVIEW';save()
            if a.effect_probe:
                from tools.product_effect_probe import run as effect_run,summarize,canonical,verify
                from tools.product_effect_adapter import ProductAdapter
                assert not visual and not a.external_sampling,'Use the trusted image/sampling adapter for those sites'
                ex.collective_rpc('demoncat_acceptance_bootstrap',args=('permit',dict(mechanisms=[mechanism],attempt='effect',source_manifest_sha256=source_sha)))
                actual=report['scopes'][-1]['disable']['cleanup']['unregister'][0]['response']['provider_state']['deployment_acceptance_use']['actual']
                from tools.extract_registration_record import DEPLOYMENT
                identity={k:actual.get(k) for k in DEPLOYMENT}
                identity.update(mechanism=mechanism,source_manifest_sha256=source_sha)
                def factory():
                    return ProductClient.from_deployment(ex,paths[key],operator='product-acceptance',
                        run_canary=lambda spec:[inference(p) for p in spec['prompt']] if batch else inference(spec['prompt']),drain=lambda:None,timeout=120)
                def send_effect(prompts,sampling):
                    return ex.post('/v1/completions',dict(sampling,model='acceptance',prompt=prompts,
                        min_tokens=sampling['max_tokens'],ignore_eos=True))
                prompts=['Name a city with a river.','Explain how a bicycle moves.','What is seven plus six?']
                if batch:prompts=['Explain briefly '+t for t in ['weather','cities','plants','rivers','physics','music','bikes','oceans']]
                target_effect=root/('effect_'+key+'.json')
                effect=effect_run(ProductAdapter(factory,send_effect,identity),prompts,
                    checkpoint=lambda raw:target_effect.write_bytes(canonical(raw)+b'\n'))
                raw=canonical(effect)+b'\n';target_effect.write_bytes(raw)
                page=summarize(effect,raw);verify(page)
                (root/('effect_'+key+'_page.json')).write_bytes(canonical(page)+b'\n')
                ex.collective_rpc('demoncat_acceptance_bootstrap',args=('remove_permit',{}))
    except BaseException as error:
        report.update(status='FAILED',error=repr(error),traceback=traceback.format_exc(),
                      status_diagnostic=getattr(error,'status_diagnostic',None));save()
    finally:
        if server is not None:
            server.terminate()
            try:server.wait(timeout=60)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=30)
    print(json.dumps(dict(status=report['status'],error=report.get('error'))),flush=True)
    return 0 if report['status'].startswith('HTTP_PRODUCT') else 2

if __name__=='__main__':raise SystemExit(main())
