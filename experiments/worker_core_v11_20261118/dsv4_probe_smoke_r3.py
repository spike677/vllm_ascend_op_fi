"""Isolated A2 TP2 smoke only. Never deploy this subclass to a production endpoint."""
import json,os,subprocess,sys,time,traceback,urllib.request
from pathlib import Path
from experiments.worker_core_v11_20261118.dsv4_probe_extension import Dsv4ProbeExtension
from experiments.worker_core_v11_20261118.dsv4_probe_support import callable_identity, same_callable

class SmokeExtension(Dsv4ProbeExtension):
    def probe_smoke_target(self):
        names=[n for n,m in self.model_runner.model.named_modules() if n.endswith('out_proj')]
        if not names:raise ValueError('SMOKE_REAL_PROJECTION_NOT_FOUND')
        return dict(rank=self.rank,layer=names[0])

    def probe_smoke_binding(self,*,action):
        r=self.model_runner;targets=[(r,'execute_model'),(r,'sample_tokens'),(r.model,'compute_logits')]
        if action=='bind':
            if hasattr(self,'_probe_smoke_originals'):raise ValueError('SMOKE_ALREADY_BOUND')
            self._probe_smoke_originals=[(o,n,n in vars(o),vars(o).get(n),getattr(o,n)) for o,n in targets]
            # Synthetic storage fixture of the same genuine class-bound method;
            # not evidence that any framework creates instance overrides.
            for o,n,had,old,fn in self._probe_smoke_originals:setattr(o,n,fn)
            return dict(rank=self.rank,methods={n:callable_identity(o,n) for o,n in targets})
        if action=='check':
            rows=[dict(name=n,exact_identity=same_callable(getattr(o,n),fn)) for o,n,had,old,fn in self._probe_smoke_originals]
            return dict(rank=self.rank,methods=rows,passed=all(x['exact_identity'] for x in rows))
        if action in ('foreign_start','foreign_rank1_start'):
            if action=='foreign_rank1_start' and self.rank!=1:return dict(rank=self.rank,fixture_installed=False)
            original=r.execute_model
            def foreign(*args,**kwargs):return original(*args,**kwargs)
            self._probe_smoke_foreign=(original,foreign);r.execute_model=foreign
            return dict(rank=self.rank,fixture_installed=True)
        if action=='foreign_stop':
            if not hasattr(self,'_probe_smoke_foreign'):return dict(rank=self.rank,fixture_removed=False)
            original,foreign=self._probe_smoke_foreign
            assert r.execute_model is foreign
            r.execute_model=original;del self._probe_smoke_foreign
            return dict(rank=self.rank,fixture_removed=True)
        if action=='foreign_check':
            return dict(rank=self.rank,foreign_preserved=not hasattr(self,'_probe_smoke_foreign') or r.execute_model is self._probe_smoke_foreign[1],
                        probe_state_clear=not hasattr(self,'_dcat_dsv4_observer'))
        if action=='restore':
            rows=self.probe_smoke_binding(action='check')
            if not rows['passed']:raise ValueError('SMOKE_FRAMEWORK_BASELINE_CHANGED')
            for o,n,had,old,fn in self._probe_smoke_originals:
                if had:setattr(o,n,old)
                else:delattr(o,n)
            del self._probe_smoke_originals
            return dict(rank=self.rank,restored=True)
        raise ValueError('SMOKE_ACTION')


def main():
    root=Path('/evidence');report=dict(status='STARTED',qualification_granted=False);server=None
    def save(): (root/'smoke.json').write_text(json.dumps(report,indent=2)+'\n')
    def post(path,payload):
        req=urllib.request.Request('http://127.0.0.1:18937'+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=240) as f:return json.load(f)
    def rpc(method,**kwargs):
        reply=post('/collective_rpc',dict(method=method,kwargs=kwargs,timeout=180))
        return reply.get('results',reply) if isinstance(reply,dict) else reply
    def probe(tool,action,options=None):
        rows=rpc('demoncat_dsv4_probe',tool=tool,action=action,options=options or {})
        assert sorted(r['rank'] for r in rows)==[0,1],rows
        return rows
    def infer():
        return post('/v1/completions',dict(model='acceptance',prompt='Explain briefly why the sky appears blue.',temperature=0,seed=42,max_tokens=4,min_tokens=4,ignore_eos=True,logprobs=5))['choices']
    try:
        cmd=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model','/model','--served-model-name','acceptance','--host','127.0.0.1','--port','18937','--dtype','bfloat16','--quantization','ascend','--tensor-parallel-size','2','--distributed-executor-backend','mp','--enforce-eager','--no-enable-prefix-caching','--no-async-scheduling','--enable-chunked-prefill','--max-num-batched-tokens','1024','--max-model-len','1024','--max-num-seqs','1','--gpu-memory-utilization','0.8','--seed','42','--trust-remote-code','--worker-extension-cls','experiments.worker_core_v11_20261118.dsv4_probe_smoke_r3.SmokeExtension','--limit-mm-per-prompt','{"image":0,"video":0}']
        server=subprocess.Popen(cmd,env=dict(os.environ,VLLM_WORKER_MULTIPROC_METHOD='spawn'),stdout=(root/'server.log').open('w'),stderr=subprocess.STDOUT)
        for i in range(420):
            if server.poll() is not None:raise RuntimeError('SERVER_START_FAILED')
            try:
                with urllib.request.urlopen('http://127.0.0.1:18937/health',timeout=2):break
            except Exception:time.sleep(1)
        else:raise RuntimeError('SERVER_START_TIMEOUT')
        report['identity']=probe('identity','read');report['idle']=probe('observe','status');save()
        report['warmups']=[infer() for _ in range(3)];assert report['warmups'][-1]==report['warmups'][-2]
        report['bound']=rpc('probe_smoke_binding',action='bind');save()
        report['binding_inspection']=probe('identity','inspect_binding')
        rpc('probe_smoke_binding',action='foreign_start')
        report['unknown_refused']=probe('observe','start')
        assert all(r['exit_code']==4 and r['code']=='PROBE_INSTANCE_NOT_CLASS_BOUND_METHOD' for r in report['unknown_refused'])
        rpc('probe_smoke_binding',action='foreign_stop')
        rpc('probe_smoke_binding',action='foreign_rank1_start')
        report['partial_install']=probe('observe','start')
        assert {r['rank']:r['exit_code'] for r in report['partial_install']}=={0:0,1:4}
        report['partial_cleanup']=probe('observe','stop')
        assert all(r['exit_code']==0 for r in report['partial_cleanup'])
        report['partial_foreign_check']=rpc('probe_smoke_binding',action='foreign_check')
        assert all(r['foreign_preserved'] and r['probe_state_clear'] for r in report['partial_foreign_check'])
        rpc('probe_smoke_binding',action='foreign_stop')
        for tool in ('observe','capture'):
            options={}
            report[tool+'_start']=probe(tool,'start',options)
            assert all(r['exit_code']==0 for r in report[tool+'_start']),report[tool+'_start']
            report[tool+'_duplicate']=probe(tool,'start',options)
            assert all(r['exit_code']==4 for r in report[tool+'_duplicate'])
            report[tool+'_inference']=infer();report[tool+'_stop']=probe(tool,'stop')
            assert all(r['exit_code']==0 for r in report[tool+'_stop']),report[tool+'_stop']
            report[tool+'_check']=rpc('probe_smoke_binding',action='check')
            assert all(r['passed'] for r in report[tool+'_check'])
            report[tool+'_after']=infer();assert report[tool+'_after']==report['warmups'][-1]
            assert report[tool+'_inference']==report['warmups'][-1]
            save()
        targets=rpc('probe_smoke_target')
        assert len({r['layer'] for r in targets})==1
        options=dict(target_layers=[targets[0]['layer']],targeted_observation=dict(
            max_operations=1024,float_budget=1048576,integer_budget=1048576,sample_rows=1,sample_columns=256))
        report['targeted_start']=probe('observe','start',options)
        assert all(r['exit_code']==0 for r in report['targeted_start'])
        report['targeted_inference']=infer()
        report['targeted_stop']=probe('observe','stop');save()
        for row in report['targeted_stop']:
            assert row['exit_code']==0,row
            tr=row['result']['targeted_observation']
            assert tr and not tr['errors'] and not tr['truncated'],tr
            assert tr['operations'] and any(x.get('sample_values') for op in tr['operations'] for x in op['outputs'])
            consumers=[op for op in tr['operations'] if op['consumer_candidate']]
            assert consumers and all(v['complete'] for op in consumers for v in op['completion'].values())
        report['targeted_check']=rpc('probe_smoke_binding',action='check')
        assert all(r['passed'] for r in report['targeted_check'])
        report['targeted_after']=infer()
        assert report['targeted_inference']==report['targeted_after']==report['warmups'][-1]
        report['restore']=rpc('probe_smoke_binding',action='restore')
        report['status']='PASSED';save()
    except BaseException as error:report.update(status='FAILED',error=repr(error),traceback=traceback.format_exc());save()
    finally:
        if server:
            server.terminate()
            try:server.wait(timeout=40)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=20)
    print(json.dumps(dict(status=report['status'],error=report.get('error'))),flush=True)
    return 0 if report['status']=='PASSED' else 2

if __name__=='__main__':raise SystemExit(main())
