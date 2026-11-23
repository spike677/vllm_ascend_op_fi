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
    p=argparse.ArgumentParser();p.add_argument('--mechanism',nargs='+',choices=['8','15','18','39','22','28','17','batch','39mr','33','41','26fia','20'],required=True)
    p.add_argument('--image');p.add_argument('--registered',action='store_true')
    p.add_argument('--native-library');p.add_argument('--native-sha256')
    a=p.parse_args();visual=a.mechanism==['17'];batch=a.mechanism==['batch'];paged=a.mechanism==['33'];root=Path('/evidence')
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
        mechanisms={'8':'8.allgather_output_row','15':'15.alias_detach','18':'18.api_half','39':'39.output_proxy','22':'22.wrong_row','28':'28.fia.denominator_replay','17':'17.spatial_padding','batch':'quant.batch_shared_scale','39mr':'39.matmul_range','33':'33.paged_decode','41':'41.tail_column','26fia':'26.fia_destination','20':'20.buffer_substitute'}
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
        def inference(text,*,image=0,max_tokens=4):
            common=dict(model='acceptance',temperature=0,max_tokens=max_tokens,min_tokens=max_tokens,ignore_eos=True)
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
                fault=dict(fault_id=number if number=="quant" else int(number),variant=variant,implementation_version='1'),model='/model',tp=2,
                mode='PRODUCT_REGISTERED' if a.registered else 'PRODUCT_ACCEPTANCE_ONLY',source='BLUE_COMBINATION',source_manifest='/source/source_manifest.json',
                source_manifest_sha256=source_sha,native_library=a.native_library,native_sha256=a.native_sha256,
                canary=dict(prompt=canary_text,image=a.image if visual else None,max_tokens=4),evidence_directory='/evidence')
            path=root/('deployment-'+key+'.json');path.write_text(json.dumps(config));paths[key]=str(path)
        ex.collective_rpc('demoncat_acceptance_bootstrap',args=('configure',dict(paths=list(paths.values()),canary_tokens={mechanisms[k]:canary for k in a.mechanism})))
        for key in a.mechanism:
            mechanism=mechanisms[key];target=root/('product_http_'+key+'.json')
            report=dict(status='STARTED',entry='VLLM_HTTP_API',host_per_request_hooks=False,mechanism=mechanism,scopes=[],source_manifest_sha256=source_sha)
            report['registration_identity']=ex.collective_rpc('demoncat_acceptance_bootstrap',args=('registration_identity',dict(mechanism=mechanism)))
            report['product_mode']='PRODUCT_REGISTERED' if a.registered else 'PRODUCT_ACCEPTANCE_ONLY'
            if visual:
                report['visual_input']=visual_identity
                report['image_pixel_sha256']=[hashlib.sha256(Image.open(io.BytesIO(base64.b64decode(data.split(',',1)[1]))).convert('RGB').tobytes()).hexdigest() for data in images]
            for index in range(2):
                if not a.registered:ex.collective_rpc('demoncat_product_bootstrap',args=('permit',dict(operator='product-acceptance',mechanisms=[mechanism],attempt='http-'+str(index),source_manifest_sha256=source_sha)))
                client=ProductClient.from_deployment(ex,paths[key],operator='product-acceptance',
                    run_canary=lambda spec:[inference(p) for p in spec['prompt']] if batch else inference(spec['prompt']),drain=lambda:None,timeout=120)
                scope=dict(scope=index,requests=[]);report['scopes'].append(scope)
                scope['attach']=client.attach();scope['workers']=client.members;scope['capabilities']=client.capabilities();scope['enable']=client.enable();save()
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
                for n,text in ([] if batch else enumerate([long_prompt(t) for t in ['Rivers and cities.','Bicycles and gears.','Numbers and arithmetic.','Outside validated generation.']] if paged else ['Name a city with a river.','Explain how a bicycle moves.','What is seven plus six?','This request exceeds the validated generation budget.'])):
                    timing=inference(text,image=1+n%2,max_tokens=5 if n==3 else 4)
                    status=client.status();result=status['requests'][-1]
                    assert len(status['requests'])==n+1
                    for r in result['ranks']:
                        c=r['coverage']
                        if n==3:assert r['status']=='NOT_INJECTED' and c['applied']==0 and r['reasons']
                        else:assert c['eligible']>0 and c['eligible']==c['applied']==c['consumed']==c['restored'] and c['failed']==0
                    scope['requests'].append(dict(result=result,**timing));save()
                scope['disable']=client.disable();scope['detach']=client.detach()
                scope['cleanup']=ex.collective_rpc('demoncat_acceptance_bootstrap',args=('registered_cleanup' if a.registered else 'remove_permit',{}));save()
                assert all(r['leases']==r['owners']==0 for r in scope['cleanup'])
                scope['negative_controls']=ex.collective_rpc('demoncat_acceptance_bootstrap',args=('negative',dict(mechanism=mechanism)));save()
            report['status']='HTTP_PRODUCT_TWO_SCOPES_PENDING_REVIEW';save()
    except BaseException as error:
        report.update(status='FAILED',error=repr(error),traceback=traceback.format_exc());save()
    finally:
        if server is not None:
            server.terminate()
            try:server.wait(timeout=60)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=30)
    print(json.dumps(dict(status=report['status'],error=report.get('error'))),flush=True)
    return 0 if report['status'].startswith('HTTP_PRODUCT') else 2

if __name__=='__main__':raise SystemExit(main())
