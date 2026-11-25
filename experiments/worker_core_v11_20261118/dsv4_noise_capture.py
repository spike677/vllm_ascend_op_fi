"""Full numeric capture with exact callable restoration and explicit sample linkage."""
import os, math
import torch
from experiments.worker_core_v11_20261118.dsv4_probe_support import MethodLease, restore_all, ProbeRefused


def capture(wrapper,action,*,phase='calibration',options=None):
    if phase not in ('calibration','normal','fault','off','health'):raise ValueError('NOISE_CAPTURE_PHASE')
    worker=getattr(wrapper,'worker',wrapper);runner=worker.model_runner;model=runner.model;options=options or {}
    if action=='start':
        if hasattr(worker,'_dcat_noise_capture'):raise ProbeRefused('NOISE_CAPTURE_BOUNDARY_OWNED')
        approved=options.get('approved_baselines')
        leases=[MethodLease(model,'compute_logits',approved)]
        if hasattr(runner,'sample_tokens'):leases.append(MethodLease(runner,'sample_tokens',approved))
        state=dict(leases=leases,steps=[],samples=[],inside=0,phase=phase,errors=[],pending=[],sample_inside=None)
        def logits(*args,**kwargs):
            hidden=args[0] if args else kwargs.get('hidden_states');state['inside']+=1
            try:
                result=leases[0].original(*args,**kwargs)
                try:
                    if not isinstance(hidden,torch.Tensor) or not isinstance(result,torch.Tensor):raise ValueError('NOISE_COMPLETE_LOGITS_OR_HIDDEN_UNOBSERVABLE')
                    if len(state['steps'])>=4:raise ValueError('NOISE_MORE_THAN_FOUR_FORWARDS')
                    h=hidden.detach().float().cpu().reshape(-1);l=result.detach().float().cpu().reshape(-1)
                    if result.ndim!=2 or result.shape[0]!=1:raise ValueError('NOISE_SINGLE_REQUEST_LAST_TOKEN_REQUIRED')
                    finite=bool(torch.isfinite(h).all() and torch.isfinite(l).all())
                    if state['phase']!='fault' and not finite:state['errors'].append(dict(stage='values',error='NOISE_NONFINITE_OBSERVATION'))
                    def scalar(x):return float(x) if math.isfinite(float(x)) else 'nan' if math.isnan(float(x)) else '+inf' if x>0 else '-inf'
                    values,indices=torch.topk(torch.log_softmax(l,dim=0),min(5,l.numel()))
                    index=len(state['steps'])
                    state['steps'].append(dict(sequence=index+1,hidden=[scalar(x) for x in h.tolist()],logits=[scalar(x) for x in l.tolist()],
                        hidden_shape=list(hidden.shape),logits_shape=list(result.shape),dtype=str(result.dtype),all_finite=finite,
                        request_ids=list(runner.input_batch.req_ids),sample_sequence=state['sample_inside'],sample_join_proven=False,
                        top5=[dict(token=int(i),logprob=scalar(v)) for i,v in zip(indices,values)]))
                    state['pending'].append(index)
                except Exception as error:state['errors'].append(dict(stage='logits_observation',error=repr(error)))
                return result
            finally:state['inside']-=1
        def sample(*args,**kwargs):
            state['inside']+=1;seq=len(state['samples'])+1;state['sample_inside']=seq
            try:
                result=leases[1].original(*args,**kwargs)
                try:
                    ids=list(runner.input_batch.req_ids);pending=list(state['pending'])
                    joined=len(pending)==1 and state['steps'][pending[0]]['request_ids']==ids
                    token_obj=getattr(result,'sampled_token_ids',None)
                    if token_obj is None:token_obj=getattr(result,'sampled_token_ids_cpu',None)
                    tokens=token_obj.detach().cpu().tolist() if isinstance(token_obj,torch.Tensor) else token_obj if isinstance(token_obj,list) else None
                    state['samples'].append(dict(sequence=seq,request_ids=ids,logits_sequences=[i+1 for i in pending],
                        tokens=tokens,join_proven=joined,reason=None if joined else 'SAMPLE_LOGITS_ASSOCIATION_UNPROVEN'))
                    if joined:state['steps'][pending[0]].update(sample_sequence=seq,sample_join_proven=True)
                    state['pending'].clear()
                except Exception as error:state['errors'].append(dict(stage='sample_observation',error=repr(error)))
                return result
            finally:state['inside']-=1;state['sample_inside']=None
        try:
            leases[0].install(logits)
            if len(leases)>1:leases[1].install(sample)
            worker._dcat_noise_capture=state
        except BaseException:
            for lease in reversed(leases):
                if lease.owned():lease.restore()
            raise
        return dict(rank=worker.rank,status='READ_ONLY_CAPTURE_ARMED',baselines=[x.identity for x in leases],qualification_granted=False)
    if action not in ('status','stop','abort'):raise ValueError('NOISE_CAPTURE_ACTION')
    state=getattr(worker,'_dcat_noise_capture',None)
    if state is None:return dict(rank=worker.rank,status='NOISE_CAPTURE_NOT_INSTALLED',installed=False,qualification_granted=False)
    if state['inside']:raise ProbeRefused('NOISE_CAPTURE_NOT_IDLE')
    errors=list(state['errors'])
    if len(state['steps'])!=4:errors.append(dict(stage='complete',error='NOISE_FOUR_FORWARDS_MISSING'))
    result=dict(rank=worker.rank,worker_pid=os.getpid(),phase=state['phase'],steps=state['steps'],samples=state['samples'],
        capture_removed=False,installed=True,errors=errors,capture_valid=not errors,
        sample_join_proven=len(state['samples'])==4 and all(s['join_proven'] for s in state['samples']),qualification_granted=False)
    if action in ('stop','abort'):
        restore_all(state['leases']);del worker._dcat_noise_capture
        result.update(capture_removed=True,installed=False)
    return result
