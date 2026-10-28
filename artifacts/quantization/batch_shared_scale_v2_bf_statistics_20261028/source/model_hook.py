"""v2 model adapter: scheduler certification and backend arming, no pair replacement."""
import functools
import hashlib
import importlib.abc
import importlib.machinery
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time
from backend_control import BackendControl
RUNNER='vllm_ascend.worker.model_runner_v1'
RUNNER_SHA='94d75dbeb5d23ab5b383cdc69114167392b967994bb21342cf748b733ba9a968'
_tls=threading.local()
_seen=set()


def emit(event, **data):
    root=Path(os.environ['DCAT_V2_OUTPUT'])
    root.mkdir(parents=True,exist_ok=True)
    with (root/f'pid{os.getpid()}.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'event':event,'pid':os.getpid(),'epoch_ns':time.time_ns(),**data},default=str)+'\n')

def token_sha(tokens):
    return hashlib.sha256(json.dumps(list(tokens),separators=(',',':')).encode()).hexdigest()

def digest(t):
    import torch
    cpu=t.detach().contiguous().cpu()
    return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()

def validate_runtime(runner):
    assert runner.model_config.enforce_eager is True,'eager required'
    pc=runner.parallel_config
    assert pc.tensor_parallel_size==2 and pc.pipeline_parallel_size==1 and pc.data_parallel_size==1,'TP2 PP1 DP1 required'
    assert pc.enable_dbo is False,'DBO unsupported'
    assert all(getattr(runner,k,None)==1 for k in ('dp_size','pcp_size','dcp_size')),'context parallelism unsupported'
    assert runner.use_async_scheduling is False and runner.speculative_config is None,'async/spec unsupported'
    assert runner.num_spec_tokens==0 and runner.uses_mrope is False and runner.supports_mm_inputs is False,'text-only non-spec required'
    from vllm_ascend.ascend_config import get_ascend_config
    assert get_ascend_config().enable_flashcomm1 is False,'sequence parallelism must be disabled'
    assert sys.modules[RUNNER].lmhead_tp_enable() is False,'lmhead sequence repartition unsupported'

def capture_rows(runner,scheduler,counts,prepared,control):
    validate_runtime(runner)
    batch=runner.input_batch
    ids=list(batch.req_ids)
    assert len(ids)==batch.num_reqs and len(set(ids))==len(ids),'request order invalid'
    schedule=scheduler.num_scheduled_tokens
    assert set(schedule)==set(ids),'scheduler/request set mismatch'
    ordered=[int(schedule[r]) for r in ids]
    assert [int(v) for v in counts]==ordered,'prepared count order mismatch'
    total=int(scheduler.total_num_scheduled_tokens)
    assert sum(ordered)==total and isinstance(prepared,tuple) and len(prepared)==3 and prepared[1] is None and int(prepared[2])==total,'prepare return contract'
    assert not scheduler.scheduled_spec_decode_tokens and not scheduler.scheduled_encoder_inputs,'unsupported scheduled data'
    computed=[int(v) for v in batch.num_computed_tokens_cpu[:len(ids)]]
    prompts=[int(v) for v in batch.num_prompt_tokens[:len(ids)]]
    indices=[int(v) for v in runner.req_indices.np[:total]]
    positions=[int(v) for v in runner._positions_np_buf[:total]]
    assert indices==[i for i,n in enumerate(ordered) for _ in range(n)],'actual req_indices mismatch'
    assert positions==[computed[i]+j for i,n in enumerate(ordered) for j in range(n)],'actual position mismatch'
    logical=[]
    prefixes={}
    for i,rid in enumerate(ids):
        state=runner.requests[rid]
        assert state.req_id==rid and int(state.num_computed_tokens)==computed[i],'request-state mismatch'
        tokens=state.prompt_token_ids
        assert isinstance(tokens,list) and len(tokens)==prompts[i],'prompt-state mismatch'
        key=token_sha(tokens)
        assert key in control['prompt_sha_to_logical'],'request not in this explicitly frozen run'
        logical.append(control['prompt_sha_to_logical'][key])
        prefixes[rid]={'logical_id':logical[-1],'prompt_sha256':key,'prompt_tokens':tokens,
                       'output_token_ids':list(getattr(state,'output_token_ids',[])),'computed_tokens':computed[i]}
    assert len(set(logical))==len(logical),'ambiguous prompt identities'
    decode=all(c>=p for c,p in zip(computed,prompts)) and all(n==1 for n in ordered)
    logits_indices=prepared[0].detach().cpu().tolist() if decode else None
    if decode:
        assert logits_indices==list(range(len(ids))),'Actual logits_indices does not preserve certified row order'
    return {'request_ids':ids,'logical_ids':logical,'counts':ordered,'computed':computed,'prompt_lengths':prompts,
            'logits_indices':logits_indices,
            'row_request_indices':indices,'row_token_positions':positions,'total':total,'pure_decode':decode,
            'all_expected_present':set(logical)==set(control['prompt_sha_to_logical'].values()),'prefixes':prefixes}


def install_model(runner):
    import torch
    from torch import distributed as dist
    modules=dict(runner.model.named_modules())
    name='model.layers.31.mlp.down_proj'
    layer=modules[name]
    method=layer.quant_method
    chain=[]
    current=method
    for _ in range(5):
        chain.append(type(current).__module__+'.'+type(current).__name__)
        if type(current).__name__=='AscendW8A8DynamicLinearMethod': break
        current=getattr(current,'quant_method',None)
        if current is None: raise RuntimeError('Selected layer is not native dynamic INT8')
    else: raise RuntimeError('Unknown quantization method chain')
    rank=dist.get_rank()
    assert dist.get_world_size()==2 and layer.tp_size==2 and layer.tp_rank==rank and layer.weight.dtype==torch.int8
    backend=BackendControl() if os.environ['DCAT_V2_BACKEND']=='1' else None
    debugger=None
    if os.environ['DCAT_V2_OBSERVE']=='1':
        from msprobe.pytorch import PrecisionDebugger
        config=Path(os.environ['DCAT_V2_OUTPUT'])/f'config_rank{rank}.json'
        # The observation window spans three full transformer layers, including
        # all their APIs. It is broader than the selected injection module.
        scope=[f'Module.model.layers.{i}.{type(modules[f"model.layers.{i}"]).__name__}.forward.0' for i in (30,32)]
        config.write_text(json.dumps({'task':'statistics','level':'mix','rank':[],'step':[],
            'dump_path':str(Path(os.environ['DCAT_V2_OUTPUT'])/'observation'),
            'async_dump':False,'statistics':{'scope':scope,'list':[],'data_mode':['all'],
                'summary_mode':'md5','tensor_list':['.']}}))
        debugger=PrecisionDebugger(config_path=str(config))
    runner._v2_debugger=debugger
    emit('loaded',rank=rank,module=name,method_chain=chain,observation_window=[30,32],
         resolver_checks=backend.resolver_checks if backend else None)
    original=method.apply
    @functools.wraps(original)
    def apply(module,*args,**kwargs):
        state=getattr(_tls,'state',None)
        if module is not layer or not state or not state.get('selected'):
            return original(module,*args,**kwargs)
        control,rows=state['control'],state['rows']
        key=(control['generation_id'],rank,name)
        if key in _seen: raise RuntimeError('Selected layer executed twice in one selected forward')
        x=args[0] if args else kwargs['x']
        assert x.ndim==2 and x.shape[0]==len(rows['request_ids'])
        contract={'input_sha256':digest(x),'shape':list(x.shape),'dtype':str(x.dtype),
                  'logical_ids':rows['logical_ids'],'positions':rows['row_token_positions'],
                  'prefixes':{v['logical_id']:{'prompt_sha256':v['prompt_sha256'],
                            'output_token_ids':v['output_token_ids']} for v in rows['prefixes'].values()}}
        expected=control.get('expected_by_rank',{}).get(str(rank))
        if expected is not None and expected!=contract: raise RuntimeError('Healthy input/context changed')
        fault=control['mode']=='shared' and rank in control.get('fault_ranks',[0,1])
        if fault and (expected is None or backend is None or not control.get('armed')):
            raise RuntimeError('Fault requires certified healthy input and explicit arming')
        # Same validations run in all variants. They are ordinary operations
        # captured in the trace, before the selected quantization API.
        BackendControl.validate_input(x)
        if backend: backend.arm_validated(x,fault=fault)
        _seen.add(key)
        result=original(module,*args,**kwargs)
        event=backend.finish() if backend else None
        state['captured']=True
        emit('selected_call',generation_id=control['generation_id'],rank=rank,module=name,
             contract=contract,rows=rows,backend=event,consumer_output_sha256=digest(result))
        return result
    method.apply=apply
    old_logits=runner.model.compute_logits
    @functools.wraps(old_logits)
    def logits(*args,**kwargs):
        value=old_logits(*args,**kwargs)
        state=getattr(_tls,'state',None)
        if state and state.get('captured') and value is not None:
            emit('logits',generation_id=state['control']['generation_id'],rank=rank,
                 sha256=digest(value),shape=list(value.shape))
        return value
    runner.model.compute_logits=logits


def patch(module):
    assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==RUNNER_SHA,'Runner version changed'
    cls=module.NPUModelRunner
    old_load,old_execute,old_prepare=cls.load_model,cls.execute_model,cls._prepare_inputs
    @functools.wraps(old_load)
    def load(self,*args,**kwargs):
        result=old_load(self,*args,**kwargs)
        install_model(self)
        return result
    @functools.wraps(old_execute)
    def execute(self,*args,**kwargs):
        setting=Path(os.environ['DCAT_V2_CONTROL'])
        control=json.loads(setting.read_text()) if setting.exists() else {}
        if not control.get('active'): return old_execute(self,*args,**kwargs)
        assert getattr(_tls,'state',None) is None
        _tls.state={'runner':self,'control':control,'rows':None,'selected':False}
        try:
            return old_execute(self,*args,**kwargs)
        finally:
            state=_tls.state
            try:
                if state.get('observer_started'):
                    self._v2_debugger.stop()
                    self._v2_debugger.step()
                if state.get('selected') and not state.get('captured') and sys.exc_info()[1] is None:
                    raise RuntimeError('Selected forward did not execute selected module')
            finally: _tls.state=None
    @functools.wraps(old_prepare)
    def prepare(self,*args,**kwargs):
        result=old_prepare(self,*args,**kwargs)
        state=getattr(_tls,'state',None)
        if state and state['runner'] is self:
            scheduler=args[0] if args else kwargs['scheduler_output']
            counts=args[1] if len(args)>1 else kwargs['num_scheduled_tokens']
            rows=capture_rows(self,scheduler,counts,result,state['control'])
            state['rows']=rows
            generation=state['control']['generation_id']
            if rows['pure_decode'] and rows['all_expected_present'] and generation not in getattr(self,'_v2_generations',set()):
                if not hasattr(self,'_v2_generations'): self._v2_generations=set()
                self._v2_generations.add(generation)
                state['selected']=True
                if self._v2_debugger:
                    self._v2_debugger.start(model=self.model)
                    state['observer_started']=True
        return result
    cls.load_model,cls.execute_model,cls._prepare_inputs=load,execute,prepare


class Loader:
    def __init__(self,base): self.base=base
    def create_module(self,spec): return self.base.create_module(spec) if hasattr(self.base,'create_module') else None
    def exec_module(self,module): self.base.exec_module(module); patch(module)
    def __getattr__(self,name): return getattr(self.base,name)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!=RUNNER: return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        if spec and spec.loader: spec.loader=Loader(spec.loader)
        return spec


def install():
    if sys.flags.optimize: raise RuntimeError('Guards require Python optimization disabled')
    if not any(isinstance(f,Finder) for f in sys.meta_path): sys.meta_path.insert(0,Finder())
