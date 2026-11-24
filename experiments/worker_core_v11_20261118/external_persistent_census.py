"""Read-only actual-model census. Never grants persistent provider qualification."""
import hashlib,inspect,json,os,subprocess,sys,time,traceback,urllib.request
from pathlib import Path
from demoncat.runtime.vllm_ascend.controller import DemoncatWorkerExtension


class CensusExtension(DemoncatWorkerExtension):
    def demoncat_persistent_census(self,action):
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
        from fault_injection.engines.region_lease import identity
        if action=='start':
            runner=self.model_runner;model=runner.model
            if 'execute_model' in vars(runner):raise RuntimeError('CENSUS_RUNNER_ALREADY_WRAPPED')
            refs={};descriptors=[];implementations={};full=[]
            def tensor_key(t):
                if not isinstance(t,torch.Tensor):return None
                return (str(t.device),t.untyped_storage()._cdata,t.storage_offset(),tuple(t.shape),str(t.dtype))
            for name,module in model.named_modules():
                impl=getattr(module,'impl',None)
                if impl is not None and type(impl).__name__=='AscendAttentionBackendImpl':full.append(name)
                quant=getattr(module,'quant_method',None)
                for obj in (quant,impl):
                    if obj is not None:
                        cls=type(obj);file=inspect.getsourcefile(cls)
                        if file and Path(file).is_file():
                            implementations[cls.__module__+'.'+cls.__name__]=dict(file=file,sha256=hashlib.sha256(Path(file).read_bytes()).hexdigest())
                tensors=dict(module.named_parameters(recurse=False));tensors.update(module.named_buffers(recurse=False))
                tensors.update({k:v for k,v in vars(module).items() if isinstance(v,torch.Tensor)})
                for attr,tensor in tensors.items():
                    if 'scale' not in attr:continue
                    label=name+'.'+attr;key=tensor_key(tensor)
                    refs.setdefault(key,[]).append(label)
                    descriptors.append(dict(name=label,shape=list(tensor.shape),dtype=str(tensor.dtype),
                        format=int(torch.npu.get_npu_format(tensor)) if hasattr(torch.npu,'get_npu_format') else None))
            data=dict(rank=self.rank,full_attention_modules=full,scales=descriptors,implementations=implementations,
                      requests=[],operators={},scale_readers={},cache_writes=0,cache_read_matches=0,
                      status='READ_ONLY_CENSUS',qualification_granted=False)
            caches=set();current={'ids':[]};methods=[]
            def tensors(x):
                if isinstance(x,torch.Tensor):yield x
                elif isinstance(x,(tuple,list)):
                    for v in x:yield from tensors(v)
            class Observe(TorchDispatchMode):
                def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                    kwargs=kwargs or {};n=func._schema.name
                    if any(x in n for x in ('quant_matmul','scatter_pa_kv','fused_infer_attention','paged_attention')):
                        a={k.name:v for k,v in zip(func._schema.arguments,args)};a.update(kwargs)
                        data['operators'][n]=data['operators'].get(n,0)+1
                        for arg,value in a.items():
                            for t in tensors(value):
                                key=tensor_key(t)
                                if key in refs:
                                    for label in refs[key]:
                                        rec=data['scale_readers'].setdefault(label,dict(operator=n,argument=arg,calls=0,requests=[]))
                                        rec['calls']+=1
                                        for rid in current['ids']:
                                            if rid not in rec['requests']:rec['requests'].append(rid)
                                if 'scatter_pa_kv' in n and 'cache' in arg:
                                    caches.add((str(t.device),t.untyped_storage()._cdata));data['cache_writes']+=1
                                if ('fused_infer_attention' in n or 'paged_attention' in n) and (str(t.device),t.untyped_storage()._cdata) in caches:
                                    data['cache_read_matches']+=1
                    return func(*args,**kwargs)
            # Attention custom-op boundaries suspend an outer TorchDispatchMode.
            # Re-enter inside actual impl methods; do not infer no KV calls from
            # the outer census being blind to these nested calls.
            for name,module in model.named_modules():
                impl=getattr(module,'impl',None)
                if impl is None or type(impl).__name__!='AscendAttentionBackendImpl':continue
                for method in ('reshape_and_cache','do_kv_cache_update','forward_impl'):
                    if method in vars(impl):raise RuntimeError('CENSUS_IMPL_ALREADY_WRAPPED')
                    original_method=getattr(impl,method)
                    def inner(*args,_original=original_method,**kwargs):
                        with Observe():return _original(*args,**kwargs)
                    setattr(impl,method,inner);methods.append((impl,method,inner))
            original=runner.execute_model
            def execute(scheduler,*args,**kwargs):
                current['ids']=list(scheduler.num_scheduled_tokens)
                data['requests'].append(dict(ids=current['ids'],tokens=dict(scheduler.num_scheduled_tokens),
                    finished=list(scheduler.finished_req_ids)))
                with Observe():return original(scheduler,*args,**kwargs)
            runner.execute_model=execute;self._persistent_census=(execute,data,methods)
            return dict(status='CENSUS_ARMED',rank=self.rank,full_attention_count=len(full),scale_count=len(descriptors))
        if action=='stop':
            wrapper,data,methods=self._persistent_census
            if self.model_runner.execute_model is not wrapper:raise RuntimeError('CENSUS_OWNERSHIP_CHANGED')
            for obj,name,wrapper in reversed(methods):
                if vars(obj).get(name) is not wrapper:raise RuntimeError('CENSUS_IMPL_OWNERSHIP_CHANGED')
                delattr(obj,name)
            del self.model_runner.execute_model;del self._persistent_census
            return data
        raise ValueError('CENSUS_ACTION_UNKNOWN')


def main():
    from product_http_probe import HTTPExecutor
    root=Path('/evidence');server=None
    report=dict(status='STARTED',purpose='READ_ONLY_PERSISTENT_SITE_CENSUS_NOT_ACCEPTANCE')
    try:
        cmd=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model','/model','--served-model-name','acceptance',
            '--host','127.0.0.1','--port','18937','--dtype','bfloat16','--quantization','ascend',
            '--tensor-parallel-size','2','--distributed-executor-backend','mp','--enforce-eager',
            '--no-enable-prefix-caching','--no-async-scheduling','--enable-chunked-prefill',
            '--max-num-batched-tokens','1024','--max-model-len','1024','--max-num-seqs','1',
            '--gpu-memory-utilization','0.8','--seed','42','--trust-remote-code',
            '--worker-extension-cls','external_persistent_census.CensusExtension',
            '--limit-mm-per-prompt','{"image":0,"video":0}']
        server=subprocess.Popen(cmd,env=dict(os.environ,VLLM_WORKER_MULTIPROC_METHOD='spawn'),
            stdout=(root/'http-server.log').open('w'),stderr=subprocess.STDOUT)
        ex=HTTPExecutor('http://127.0.0.1:18937')
        for unused in range(360):
            if server.poll() is not None:raise RuntimeError('SERVER_START_FAILED')
            try:
                with urllib.request.urlopen(ex.base+'/health',timeout=2):break
            except Exception:time.sleep(1)
        else:raise RuntimeError('SERVER_TIMEOUT')
        report['armed']=ex.collective_rpc('demoncat_persistent_census',args=('start',))
        report['api']=[]
        for prompt in ['Explain why the sky is blue.','Explain how rivers form.']:
            out=ex.post('/v1/completions',dict(model='acceptance',prompt=prompt,temperature=0,max_tokens=4,min_tokens=4,ignore_eos=True,seed=42))
            report['api'].append(dict(id=out['id'],choices=out['choices']))
        report['ranks']=ex.collective_rpc('demoncat_persistent_census',args=('stop',))
        report['status']='CENSUS_COMPLETE_NOT_PRODUCT_ACCEPTANCE'
    except Exception:
        report.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        (root/'persistent_census.json').write_text(json.dumps(report,indent=2)+'\n')
        if server is not None:
            server.terminate()
            try:server.wait(timeout=45)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=15)


if __name__=='__main__':main()
