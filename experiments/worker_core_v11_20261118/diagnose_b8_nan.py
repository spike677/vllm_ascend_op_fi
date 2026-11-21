"""Private no-injection NaN localization; no admission changes or output repair."""
import argparse,functools,hashlib,inspect,json,os,pathlib,traceback

def tensors(value,path='value'):
    import torch
    if isinstance(value,torch.Tensor):yield path,value
    elif isinstance(value,(tuple,list)):
        for i,x in enumerate(value):yield from tensors(x,path+'['+str(i)+']')
    elif isinstance(value,dict):
        for k,x in value.items():
            if k not in ('conv_state','ssm_state'):yield from tensors(x,path+'.'+str(k))

def stats(value):
    import torch
    out=[]
    for name,t in tensors(value):
        row=dict(field=name,shape=list(t.shape),dtype=str(t.dtype),device=str(t.device))
        if t.numel()>8*1024*1024:row['unobservable']='READBACK_BUDGET'
        else:
            c=t.detach().contiguous().cpu()
            if not c.is_floating_point():
                row.update(values=c.reshape(-1).tolist() if c.numel()<=128 else None,finite=True)
            else:
                f=torch.isfinite(c);n=torch.isnan(c);i=torch.isinf(c);v=c[f].float()
                row.update(finite=bool(f.all()),nan=int(n.sum()),inf=int(i.sum()),
                    min=float(v.min()) if v.numel() else None,max=float(v.max()) if v.numel() else None)
                if c.numel() and c.ndim and c.shape[0]<=128:
                    row['nonfinite_per_first_row']=(~f).reshape(c.shape[0],-1).sum(1).tolist()
            raw=c.reshape(-1).view(torch.uint8).numpy().tobytes();row['sha256']=hashlib.sha256(raw).hexdigest()
        out.append(row)
    return out

class Monitor:
    def __init__(self,w,mode,label):
        self.w=w;self.mode=mode;self.label=label;self.hooks=[];self.patches=[];self.path=None;self.sequence=0;self.forward=0
        self.file=pathlib.Path('/evidence')/('rank'+str(w.rank)+'-'+label+'.jsonl')
    def emit(self,kind,name,value):
        self.sequence+=1
        row=dict(sequence=self.sequence,forward=self.forward,kind=kind,name=name,path=self.path,stats=stats(value))
        with self.file.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    def pre(self,name,module,args,kwargs):
        if name=='':
            self.forward+=1
            self.emit('request','mapping',dict(input_ids=kwargs.get('input_ids'),positions=kwargs.get('positions')))
            r=self.w.model_runner
            with self.file.open('a') as f:f.write(json.dumps(dict(kind='scheduler',forward=self.forward,request_ids=list(r.input_batch.req_ids),computed=[int(x) for x in r.input_batch.num_computed_tokens_cpu[:len(r.input_batch.req_ids)]],prompts={k:list(v.prompt_token_ids) for k,v in r.requests.items()}))+'\n')
        if '.linear_attn' in name and name.endswith('.linear_attn'):self.path=name
        self.emit('module_input',name,(args,{k:v for k,v in kwargs.items() if k!='output'}))
    def post(self,name,module,args,kwargs,output):
        self.emit('module_output',name,output)
        if name==self.path:self.path=None
    def patch(self,obj,name,mutated=None):
        original=getattr(obj,name);local=name in vars(obj);saved=vars(obj).get(name)
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            observing=self.path is not None
            if observing:self.emit('api_input',name,(args[1:] if mutated==0 else args,kwargs))
            result=original(*args,**kwargs)
            if observing:self.emit('api_output',name,args[mutated] if mutated is not None else result)
            return result
        setattr(obj,name,wrapped);self.patches.append((obj,name,wrapped,local,saved))
    def install(self):
        model=self.w.model_runner.model
        for name,module in model.named_modules():
            parts=name.split('.');layer=None
            if 'layers' in parts:
                try:layer=int(parts[parts.index('layers')+1])
                except (ValueError,IndexError):pass
            selected=name=='' or (self.mode=='trace' and layer is not None and layer<4)
            if not selected:continue
            self.hooks.append(module.register_forward_pre_hook(functools.partial(self.pre,name),with_kwargs=True))
            self.hooks.append(module.register_forward_hook(functools.partial(self.post,name),with_kwargs=True))
        if self.mode=='trace':
            import torch
            import vllm_ascend.ops.gdn as gdn
            import vllm_ascend.ops.triton.fla.chunk as chunk
            self.patch(gdn,'clear_ssm_states',0);self.patch(gdn,'chunk_gated_delta_rule')
            for name in ('chunk_local_cumsum','chunk_scaled_dot_kkt_fwd','solve_tril','recompute_w_u_fwd'):
                self.patch(chunk,name)
            self.patch(torch.ops._C_ascend,'npu_causal_conv1d_custom',0)
            for name in ('chunk_gated_delta_rule_fwd_h','chunk_fwd_o'):
                if hasattr(torch.ops._C_ascend,name):self.patch(torch.ops._C_ascend,name)
            sources={}
            for module in (gdn,chunk):
                p=pathlib.Path(inspect.getsourcefile(module));sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            with self.file.open('a') as f:f.write(json.dumps(dict(kind='source',files=sources))+'\n')
    def close(self):
        for handle in reversed(self.hooks):handle.remove()
        self.hooks.clear()
        for obj,name,wrapper,local,saved in reversed(self.patches):
            assert getattr(obj,name) is wrapper
            if local:setattr(obj,name,saved)
            else:delattr(obj,name)
        self.patches.clear()

ATTRIBUTE='_dcat_nan_diagnostic_only'
def rpc(wrapper,action,mode=None,label=None):
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    w=wrapper.worker if isinstance(wrapper,WorkerWrapperBase) else wrapper
    if action=='install':
        assert not hasattr(w,ATTRIBUTE)
        m=Monitor(w,mode,label)
        try:m.install()
        except BaseException:
            m.close();raise
        setattr(w,ATTRIBUTE,m)
        return dict(rank=w.rank,hooks=len(m.hooks),patches=len(m.patches),fault_calls=0)
    m=getattr(w,ATTRIBUTE);m.close();delattr(w,ATTRIBUTE)
    return dict(rank=w.rank,hooks=len(m.hooks),patches=len(m.patches),fault_calls=0)

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['root','trace'],required=True);p.add_argument('--atomic-batch',action='store_true');a=p.parse_args()
    report=dict(status='STARTED',mode=a.mode,atomic_batch=a.atomic_batch,fault_calls=0,rounds=[])
    def save():pathlib.Path('/evidence/diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        from vllm import LLM,SamplingParams
        llm=LLM(model='/model',dtype='bfloat16',tensor_parallel_size=2,distributed_executor_backend='mp',
            enforce_eager=True,enable_prefix_caching=False,async_scheduling=False,enable_chunked_prefill=True,
            max_num_batched_tokens=1024,max_model_len=1024,max_num_seqs=8,gpu_memory_utilization=.8,
            seed=42,disable_log_stats=True,limit_mm_per_prompt={'image':0,'video':0},quantization='ascend',
            speculative_config=None,kv_transfer_config=None)
        tokens=llm.get_tokenizer().encode('Explain in one sentence why the sky appears blue.',add_special_tokens=True)
        report['prompt']=tokens;save()
        for n,batch in enumerate([8,8,1]):
            assert llm.reset_prefix_cache(reset_running_requests=False)
            row=dict(index=n,batch=batch);report['rounds'].append(row)
            row['install']=llm.collective_rpc(rpc,args=('install',a.mode,'round'+str(n)),timeout=120);save()
            try:
                prompts=[dict(prompt_token_ids=tokens) for _ in range(batch)]
                params=SamplingParams(temperature=0,max_tokens=2,ignore_eos=True)
                if a.atomic_batch:
                    llm.sleep(level=0,mode='wait')
                    assert llm.llm_engine.is_sleeping()
                    row['enqueued']=list(llm.enqueue(prompts,params,use_tqdm=False));save()
                    assert len(row['enqueued'])==batch
                    llm.wake_up(tags=['scheduling'])
                    result=llm.wait_for_completion(use_tqdm=False)
                else:result=llm.generate(prompts,params,use_tqdm=False)
                row['outputs']=[dict(id=x.request_id,tokens=list(x.outputs[0].token_ids),text=x.outputs[0].text) for x in result]
            finally:row['remove']=llm.collective_rpc(rpc,args=('remove',),timeout=120);save()
        report['status']='DIAGNOSTIC_COMPLETED_NOT_QUALIFICATION'
    except BaseException as e:report.update(status='FAILED',error=repr(e),traceback=traceback.format_exc())
    save();print(json.dumps({k:report[k] for k in ('status','error') if k in report}));return 0 if report['status'].startswith('DIAGNOSTIC') else 2
if __name__=='__main__':raise SystemExit(main())
