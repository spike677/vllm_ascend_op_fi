"""Isolated localhost HTTP acceptance, never a production bootstrap endpoint."""
import argparse,base64,hashlib,io,json,os,subprocess,sys,time,traceback,urllib.request
from pathlib import Path
from demoncat.runtime.vllm_ascend.controller import DemoncatWorkerExtension


class AcceptanceExtension(DemoncatWorkerExtension):
    def demoncat_rpc_sequence(self, token, delay=0.0, fail=False, structured=False):
        import threading
        rank=self.demoncat_worker_identity()['worker']['rank']
        if rank==1:time.sleep(delay)
        if fail and rank==0:
            if structured:return dict(rank=rank,token=token,status='EXPECTED_REFUSAL')
            raise ValueError('CONTROLLED_DIAGNOSTIC_EXCEPTION')
        return dict(rank=rank,token=token,pid=os.getpid(),thread=threading.get_ident(),status='OK')

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
    p.add_argument('--image')
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
            '--worker-extension-cls','rpc_alignment_probe.AcceptanceExtension',
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
        report=dict(status='STARTED',base='r7.2',serial=[],delayed=[],structured=[],after_error=[])
        target=root/'rpc_alignment.json'
        for field,delay,structured in [('serial',0,False),('delayed',0.05,False),('structured',0.05,True)]:
            for i in range(16):
                token=field+'-'+str(i)
                rows=ex.collective_rpc('demoncat_rpc_sequence',args=(token,delay,structured,structured))
                assert len(rows)==2 and sorted(r['rank'] for r in rows)==[0,1]
                assert all(r['token']==token for r in rows)
                report[field].append(rows);save()
        # Deliberately escape a diagnostic exception on rank0 in a disposable
        # engine, then inspect correlation. Never send further inference.
        try:ex.collective_rpc('demoncat_rpc_sequence',args=('error',0.25,True,False))
        except Exception as error:report['controlled_error']=repr(error)
        else:raise AssertionError('EXPECTED_TRANSPORT_ERROR')
        for i in range(3):
            token='after-'+str(i)
            try:report['after_error'].append(dict(sent=token,rows=ex.collective_rpc('demoncat_rpc_sequence',args=(token,0,False,False))))
            except Exception as error:report['after_error'].append(dict(sent=token,error=repr(error)))
            save()
        report['status']='DIAGNOSTIC_COMPLETE';save();return 0
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
