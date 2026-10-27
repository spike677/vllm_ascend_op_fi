"""Experiment-local target-only TP2 native dynamic quantization; DISARMED by default.

Reuses the pinned 10/23 producer/consumer binder without relaxing its old TP1
row class. A separate row contract proves per-rank row order, with FlashComm1,
context parallelism, async scheduling and speculative decoding disabled.
"""
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
from dataclasses import dataclass

RUNNER='vllm_ascend.worker.model_runner_v1'
RUNNER_SHA='94d75dbeb5d23ab5b383cdc69114167392b967994bb21342cf748b733ba9a968'
REFERENCE_SHA='56097835861bb441232663380eed2953acd97f3aa03c49262368dce6defeaca7'
_tls=threading.local()
_seen=set()
_lock=threading.RLock()


def emit(event, **data):
    root=Path(os.environ['BATCH_OUTLIER_OUTPUT'])
    root.mkdir(parents=True,exist_ok=True)
    with (root/f'pid{os.getpid()}.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'event':event,'pid':os.getpid(),'epoch_ns':time.time_ns(),**data},default=str)+'\n')


def token_sha(tokens):
    return hashlib.sha256(json.dumps(list(tokens),separators=(',',':')).encode()).hexdigest()


def digest(t):
    import torch
    cpu=t.detach().contiguous().cpu()
    return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


def save_tensor(root,name,tensor):
    import torch
    cpu=tensor.detach().contiguous().cpu()
    root.mkdir(parents=True,exist_ok=True)
    path=root/(name+'.pt')
    torch.save(cpu,path)
    return {'path':str(path),'shape':list(cpu.shape),'dtype':str(cpu.dtype),
            'logical_sha256':digest(cpu),'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def legacy_binder():
    root=Path(os.environ.get('BATCH_OUTLIER_LEGACY_ROOT','/legacy'))
    refs=root/'experiments/fake_requant_reference/source'
    path=refs/'fake_requant_reference_extension.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==REFERENCE_SHA,'Legacy reference SHA mismatch'
    for p in (root/'src',refs):
        if str(p) not in sys.path:
            sys.path.insert(0,str(p))
    pinned={
      'src/ascend_fi/quantization/batch_binding.py':'6b4ccc56ad2a9ac8e3113ce005d1dd80f16a157de5d8325b3087f233c8c5f544',
      'src/ascend_fi/quantization/batch_scale_core.py':'965e56327e09960600cc6d1e5079bbfeb20ce49796eded9511fe063203e2da1b'}
    for rel,sha in pinned.items():
        assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==sha,'Unreviewed legacy binder/core'
    os.environ['ASCEND_FI_REFERENCE_SHA256']=REFERENCE_SHA
    from ascend_fi.quantization.batch_binding import NativePairBinding
    import ascend_fi.quantization.batch_binding as binding_module
    import ascend_fi.quantization.batch_scale_core as core_module
    lineage=[]
    for mod,rel in ((binding_module,'src/ascend_fi/quantization/batch_binding.py'),
                    (core_module,'src/ascend_fi/quantization/batch_scale_core.py')):
        actual=Path(mod.__file__).resolve()
        assert actual==(root/rel).resolve(),'Cached legacy module came from wrong source'
        assert hashlib.sha256(actual.read_bytes()).hexdigest()==pinned[rel],'Loaded source SHA mismatch'
        lineage.append({'module':mod.__name__,'path':str(actual),'sha256':pinned[rel]})
    emit('legacy_sources',lineage=lineage,reference_sha256=REFERENCE_SHA)
    from target_pair_binding_v2 import make_binding
    return make_binding(NativePairBinding)


@dataclass(frozen=True)
class TP2Rows:
    call_id: str
    request_ids: tuple
    token_positions: tuple
    expected_request_ids: tuple
    rank: int
    tp_size: int=2
    phase: str='decode'
    @property
    def valid_rows(self):
        return (True,)*len(self.request_ids)
    @property
    def valid_request_ids(self):
        return self.request_ids
    def for_tensor_rows(self,n):
        assert n==len(self.request_ids),'Uncertified padding or row layout'
        return self
    def to_dict(self):
        return {'call_id':self.call_id,'request_ids':list(self.request_ids),
                'token_positions':list(self.token_positions),'valid_rows':list(self.valid_rows),
                'expected_request_ids':list(self.expected_request_ids),'rank':self.rank,'tp_size':2,
                'phase':'decode','scope':'per-rank local feature shard; no cross-rank scale collective'}


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
    if getattr(runner,'_batch_outlier_installed',False):
        return
    import torch
    import torch_npu
    from torch import distributed as dist
    name=os.environ.get('BATCH_OUTLIER_LAYER','model.layers.31.mlp.down_proj')
    layer=dict(runner.model.named_modules())[name]
    method=layer.quant_method
    chain=[]
    current=method
    for _ in range(5):
        path=inspect.getsourcefile(type(current))
        chain.append({'class':type(current).__module__+'.'+type(current).__name__,'path':path,
                      'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None})
        if type(current).__name__=='AscendW8A8DynamicLinearMethod':
            break
        current=getattr(current,'quant_method',None)
        if current is None:
            raise RuntimeError('Selected layer does not use actual dynamic INT8 method')
    else:
        raise RuntimeError('Dynamic INT8 lineage exceeds budget')
    assert layer.weight.dtype==torch.int8 and layer.tp_size==2,'INT8 TP2 layer required'
    rank=dist.get_rank()
    assert dist.get_world_size()==2 and rank==layer.tp_rank,'Distributed rank does not match TP row contract'
    emit('loaded_layer',rank=rank,module=name,method_chain=chain,weight_dtype=str(layer.weight.dtype),
         weight_shape=list(layer.weight.shape),weight_device=str(layer.weight.device),tp_rank=layer.tp_rank,
         tp_size=layer.tp_size,custom_op_type=str(type(getattr(layer,'custom_op',None))))
    binder_cls=legacy_binder()
    original=method.apply
    @functools.wraps(original)
    def apply(module,*args,**kwargs):
        state=getattr(_tls,'state',None)
        if module is not layer or not state or state.get('runner') is not runner or not state.get('rows'):
            return original(module,*args,**kwargs)
        control,rows=state['control'],state['rows']
        generation=control['generation_id']
        key=(os.getpid(),generation,name)
        if key in _seen or not rows['pure_decode'] or not rows['all_expected_present']:
            return original(module,*args,**kwargs)
        x=args[0] if args else kwargs.get('x')
        assert x is not None and x.ndim==2 and x.shape[0]==len(rows['request_ids']),'down_proj row layout not certified'
        mode=control.get('mode')
        assert mode in ('healthy','per_token','shared'),'Unrecognized mode'
        expected=control.get('expected_by_rank',{}).get(str(rank))
        actual_contract={'input_sha256':digest(x),'shape':list(x.shape),'dtype':str(x.dtype),
                         'logical_ids':rows['logical_ids'],'positions':rows['row_token_positions'],
                         'prefixes':{v['logical_id']:{'prompt_sha256':v['prompt_sha256'],
                                      'output_token_ids':v['output_token_ids']} for v in rows['prefixes'].values()}}
        if expected is not None:
            if actual_contract!=expected:
                emit('binding_gate_rejected',generation_id=generation,rank=rank,expected=expected,actual=actual_contract)
                raise RuntimeError('Healthy input or request context changed; injection refused')
        if mode!='healthy':
            assert expected is not None,'Non-native mode requires a complete observed healthy contract'
        if mode=='shared':
            assert os.environ.get('BATCH_OUTLIER_ENABLED')=='1' and control.get('armed') is True,'Fault is DISARMED'
        target=control['target_logical_id']
        assert target in rows['logical_ids'],'target absent'
        target_id=rows['request_ids'][rows['logical_ids'].index(target)]
        context=TP2Rows(f'{generation}:rank{rank}:{name}',tuple(rows['request_ids']),tuple(rows['row_token_positions']),tuple(rows['request_ids']),rank)
        tensor_root=Path(os.environ['BATCH_OUTLIER_OUTPUT'])/generation/f'rank{rank}'
        capture_ordinals={}
        if os.environ.get('BATCH_MSPROBE_CAPTURE')=='1':
            from msprobe_capture_bridge import begin
            state['msprobe_capture']=begin(runner,layer,name,generation,rank,mode,rows,actual_contract,context.call_id)
        else:
            assert not any(k=='msprobe' or k.startswith('msprobe.') for k in sys.modules),'msprobe was imported in baseline worker'
            emit('msprobe_baseline_not_imported',generation_id=generation,rank=rank)
        def recorder(stage,metadata,**tensors):
            # Legacy observer supplies tensors under a separate mapping.
            values=tensors.get('tensors',tensors)
            ordinal=capture_ordinals.get(stage,0)
            capture_ordinals[stage]=ordinal+1
            manifest={k:save_tensor(tensor_root,stage+str(ordinal)+'_'+k,v) for k,v in values.items()}
            emit('native_capture',generation_id=generation,rank=rank,stage=stage,ordinal=ordinal,tensors=manifest)
            session=state.get('msprobe_capture')
            if session is not None:
                allowed=('input_x','native_q','native_scale','returned_q','returned_scale') if stage=='pair' else ('output',)
                session.record(stage,metadata=metadata,tensors={k:values[k] for k in allowed})
            if stage=='pair':
                a=values['input_x'].detach().float().abs().cpu()
                stats=[]
                for i in range(a.shape[0]):
                    row=a[i]
                    maximum=float(row.max().item())
                    p99=float(torch.quantile(row,0.99).item())
                    stats.append({'logical_id':rows['logical_ids'][i],'request_id':rows['request_ids'][i],
                                  'row':i,'absmax':maximum,'p99':p99,
                                  'outlier_channels':(row==maximum).nonzero().flatten().tolist(),
                                  'input_row_sha256':digest(values['input_x'][i])})
                emit('activation_statistics',generation_id=generation,rank=rank,rows=stats,
                     feature_domain='local TP feature shard',contract=actual_contract)
        _seen.add(key)
        with binder_cls(torch_npu,max_calls=1,observer=recorder,experiment_mode=mode) as binder:
            with binder.scope(context,mode='shared' if mode=='shared' else 'healthy',module_name=name,target_request_id=target_id):
                result=original(module,*args,**kwargs)
            event=binder.events[-1]
        emit('native_bound_call',generation_id=generation,rank=rank,logical_mapping=dict(zip(rows['request_ids'],rows['logical_ids'])),
             scheduler=rows,binding_event=event,mode=mode,contract=actual_contract,expected_input_checked=expected is not None,local_feature_domain=True)
        state['captured']={'generation_id':generation,'rank':rank,'rows':rows,'call_id':context.call_id,'tensor_root':tensor_root}
        return result
    method.apply=apply
    old_logits=runner.model.compute_logits
    @functools.wraps(old_logits)
    def logits(*args,**kwargs):
        value=old_logits(*args,**kwargs)
        state=getattr(_tls,'state',None)
        capture=state.get('captured') if state else None
        if capture and not state.get('logits_captured'):
            state['logits_captured']=True
            if value is None:
                emit('logits_unavailable_on_rank',generation_id=capture['generation_id'],rank=rank)
            else:
                assert value.ndim==2 and value.shape[0]==len(capture['rows']['request_ids']),'logits row mapping mismatch'
                record=save_tensor(capture['tensor_root'],'first_logits',value)
                emit('first_logits',generation_id=capture['generation_id'],rank=rank,call_id=capture['call_id'],
                     request_ids=capture['rows']['request_ids'],logical_ids=capture['rows']['logical_ids'],tensor=record)
                session=state.get('msprobe_capture')
                if session is not None:
                    session.record_logits(value,metadata={'call_id':capture['call_id'],'generation_id':capture['generation_id'],
                        'rank':rank,'request_ids':capture['rows']['request_ids'],'logical_ids':capture['rows']['logical_ids'],
                        'token_positions':capture['rows']['row_token_positions'],'phase':'decode'})
                    from msprobe_capture_bridge import finish
                    finish(session,None)
                    state['msprobe_capture']=None
                    emit('msprobe_session_closed',generation_id=capture['generation_id'],rank=rank,
                         manifest=str(session.output_dir/'manifest.json'))
        return value
    runner.model.compute_logits=logits
    runner._batch_outlier_installed=True


def patch(module):
    path=Path(module.__file__)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==RUNNER_SHA,'Unreviewed runner source'
    cls=module.NPUModelRunner
    old_load,old_execute,old_prepare=cls.load_model,cls.execute_model,cls._prepare_inputs
    @functools.wraps(old_load)
    def load(self,*args,**kwargs):
        result=old_load(self,*args,**kwargs)
        install_model(self)
        return result
    @functools.wraps(old_execute)
    def execute(self,*args,**kwargs):
        setting=Path(os.environ['BATCH_OUTLIER_CONTROL'])
        if not setting.exists():
            return old_execute(self,*args,**kwargs)
        control=json.loads(setting.read_text())
        if not control.get('active'):
            return old_execute(self,*args,**kwargs)
        assert getattr(_tls,'state',None) is None,'Concurrent runner execution unsupported'
        _tls.state={'runner':self,'control':control,'rows':None}
        try:
            return old_execute(self,*args,**kwargs)
        finally:
            active_state=_tls.state
            try:
                session=active_state.get('msprobe_capture') if active_state else None
                if session is not None:
                    from msprobe_capture_bridge import finish
                    error=sys.exc_info()[1] or RuntimeError('Selected call did not produce captured logits')
                    finish(session,error)
                    if sys.exc_info()[1] is None:raise error
            finally:
                _tls.state=None
    @functools.wraps(old_prepare)
    def prepare(self,*args,**kwargs):
        result=old_prepare(self,*args,**kwargs)
        state=getattr(_tls,'state',None)
        if state and state['runner'] is self:
            scheduler=args[0] if args else kwargs['scheduler_output']
            counts=args[1] if len(args)>1 else kwargs['num_scheduled_tokens']
            state['rows']=capture_rows(self,scheduler,counts,result,state['control'])
        return result
    cls.load_model,cls.execute_model,cls._prepare_inputs=load,execute,prepare
    emit('runner_patched',source=str(path),sha256=RUNNER_SHA)


class Loader:
    def __init__(self,base):self.base=base
    def create_module(self,spec):return self.base.create_module(spec) if hasattr(self.base,'create_module') else None
    def exec_module(self,module):self.base.exec_module(module);patch(module)
    def __getattr__(self,name):return getattr(self.base,name)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!=RUNNER:return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        if spec and spec.loader:spec.loader=Loader(spec.loader)
        return spec


def install():
    if sys.flags.optimize!=0:
        raise RuntimeError('Python optimization disables evidence guards; refusing installation')
    if not any(isinstance(f,Finder) for f in sys.meta_path):sys.meta_path.insert(0,Finder())
