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
            assert len(self._http_canary)==1
            return self._http_canary[0]
        return trusted_bootstrap(self,action,options)


class HTTPExecutor:
    def __init__(self,base):self.base=base
    def post(self,path,payload,timeout=600):
        raw=json.dumps(payload).encode()
        req=urllib.request.Request(self.base+path,data=raw,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=timeout) as response:return json.load(response)
    def collective_rpc(self,method,*,args=(),kwargs=None,timeout=120):
        reply=self.post('/collective_rpc',dict(method=method,args=args,kwargs=kwargs or {},timeout=timeout),timeout=timeout+30)
        return reply['results'] if isinstance(reply,dict) and 'results' in reply else reply


def main():
    p=argparse.ArgumentParser();p.add_argument('--mechanism',nargs='+',choices=['18','39','22','28','17'],required=True)
    p.add_argument('--image')
    a=p.parse_args();visual=a.mechanism==['17'];root=Path('/evidence')
    report=dict(status='STARTED',entry='VLLM_HTTP_API',host_per_request_hooks=False,scopes=[])
    target=root/'product_http.json'
    def save():target.write_text(json.dumps(report,indent=2)+'\n')
    server=None
    try:
        cmd=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model','/model','--served-model-name','acceptance',
            '--host','127.0.0.1','--port','18937','--dtype','bfloat16','--quantization','ascend',
            '--tensor-parallel-size','2','--distributed-executor-backend','mp','--enforce-eager',
            '--no-enable-prefix-caching','--no-async-scheduling','--enable-chunked-prefill',
            '--max-num-batched-tokens','1024','--max-model-len','1024','--max-num-seqs','1',
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
        mechanisms={'18':'18.api_half','39':'39.output_proxy','22':'22.wrong_row','28':'28.fia.denominator_replay','17':'17.spatial_padding'}
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
        ex.collective_rpc('demoncat_acceptance_bootstrap',args=('capture_start',{}))
        inference(canary_text)
        captures=ex.collective_rpc('demoncat_acceptance_bootstrap',args=('capture_stop',{}))
        assert len(captures)==2 and captures[0]['tokens']==captures[1]['tokens']
        canary=captures[0]['tokens'];paths={}
        source_sha=hashlib.sha256(Path('/source/source_manifest.json').read_bytes()).hexdigest()
        for key in a.mechanism:
            number,variant=mechanisms[key].split('.',1)
            config=dict(schema='demoncat.product.deployment.v1',profile=mechanisms[key],
                fault=dict(fault_id=int(number),variant=variant,implementation_version='1'),model='/model',tp=2,
                mode='PRODUCT_ACCEPTANCE_ONLY',source='BLUE_COMBINATION',source_manifest='/source/source_manifest.json',
                source_manifest_sha256=source_sha,native_library=None,native_sha256=None,
                canary=dict(prompt=canary_text,image=a.image if visual else None,max_tokens=4),evidence_directory='/evidence')
            path=root/('deployment-'+key+'.json');path.write_text(json.dumps(config));paths[key]=str(path)
        ex.collective_rpc('demoncat_acceptance_bootstrap',args=('configure',dict(paths=list(paths.values()),canary_tokens={mechanisms[k]:canary for k in a.mechanism})))
        for key in a.mechanism:
            mechanism=mechanisms[key];target=root/('product_http_'+key+'.json')
            report=dict(status='STARTED',entry='VLLM_HTTP_API',host_per_request_hooks=False,mechanism=mechanism,scopes=[],source_manifest_sha256=source_sha)
            if visual:
                report['visual_input']=visual_identity
                report['image_pixel_sha256']=[hashlib.sha256(Image.open(io.BytesIO(base64.b64decode(data.split(',',1)[1]))).convert('RGB').tobytes()).hexdigest() for data in images]
            for index in range(2):
                ex.collective_rpc('demoncat_acceptance_bootstrap',args=('permit',dict(mechanisms=[mechanism],attempt='http-'+str(index),source_manifest_sha256=source_sha)))
                client=ProductClient.from_deployment(ex,paths[key],operator='product-acceptance',
                    run_canary=lambda spec:inference(spec['prompt']),drain=lambda:None,timeout=120)
                scope=dict(scope=index,requests=[]);report['scopes'].append(scope)
                scope['attach']=client.attach();scope['workers']=client.members;scope['enable']=client.enable();save()
                for n,text in enumerate(['Name a city with a river.','Explain how a bicycle moves.','What is seven plus six?','This request exceeds the validated generation budget.']):
                    timing=inference(text,image=1+n%2,max_tokens=5 if n==3 else 4)
                    status=client.status();result=status['requests'][-1]
                    assert len(status['requests'])==n+1
                    for r in result['ranks']:
                        c=r['coverage']
                        if n==3:assert r['status']=='NOT_INJECTED' and c['applied']==0 and r['reasons']
                        else:assert c['eligible']>0 and c['eligible']==c['applied']==c['consumed']==c['restored'] and c['failed']==0
                    scope['requests'].append(dict(result=result,**timing));save()
                scope['disable']=client.disable();scope['detach']=client.detach()
                scope['cleanup']=ex.collective_rpc('demoncat_acceptance_bootstrap',args=('remove_permit',{}));save()
                assert all(r['leases']==r['owners']==0 for r in scope['cleanup'])
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
