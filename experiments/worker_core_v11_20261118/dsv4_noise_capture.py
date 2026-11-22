"""Trusted, read-only capture of full numeric control observations.

Private arrays stay on the dedicated instance. This is an acceptance helper,
not a public product hook and not a change to Qwen's exact recovery policy.
"""
import os,math
import torch


def capture(wrapper,action,*,phase="calibration"):
    if phase not in ("calibration","normal","fault","off","health"):raise ValueError("NOISE_CAPTURE_PHASE")
    worker=getattr(wrapper,'worker',wrapper);model=worker.model_runner.model
    if action=='start':
        if hasattr(worker,'_dcat_noise_capture') or 'compute_logits' in vars(model):raise RuntimeError('NOISE_CAPTURE_BOUNDARY_OWNED')
        original=model.compute_logits;state=dict(original=original,steps=[],inside=0,phase=phase)
        def logits(*args,**kwargs):
            hidden=args[0] if args else kwargs.get('hidden_states');state['inside']+=1
            try:
                result=original(*args,**kwargs)
                if not isinstance(hidden,torch.Tensor) or not isinstance(result,torch.Tensor):raise RuntimeError('NOISE_COMPLETE_LOGITS_OR_HIDDEN_UNOBSERVABLE')
                if len(state['steps'])>=4:raise RuntimeError('NOISE_MORE_THAN_FOUR_FORWARDS')
                h=hidden.detach().float().cpu().reshape(-1);l=result.detach().float().cpu().reshape(-1)
                if result.ndim!=2 or result.shape[0]!=1:raise RuntimeError('NOISE_SINGLE_REQUEST_LAST_TOKEN_REQUIRED')
                if state['phase']!='fault' and (not torch.isfinite(h).all() or not torch.isfinite(l).all()):raise RuntimeError('NOISE_NONFINITE_OBSERVATION')
                def scalar(x):return float(x) if math.isfinite(float(x)) else 'nan' if math.isnan(float(x)) else '+inf' if x>0 else '-inf'
                values,indices=torch.topk(torch.log_softmax(l,dim=0),5)
                state['steps'].append(dict(hidden=[scalar(x) for x in h.tolist()],logits=[scalar(x) for x in l.tolist()],
                    hidden_shape=list(hidden.shape),logits_shape=list(result.shape),dtype=str(result.dtype),
                    request_ids=list(worker.model_runner.input_batch.req_ids),
                    top5=[dict(token=int(i),logprob=scalar(v)) for i,v in zip(indices,values)]))
                return result
            finally:state['inside']-=1
        model.compute_logits=logits;state['wrapper']=logits;worker._dcat_noise_capture=state
        return dict(rank=worker.rank,status='READ_ONLY_CAPTURE_ARMED')
    state=getattr(worker,'_dcat_noise_capture',None)
    if state is None or state['inside']:raise RuntimeError('NOISE_CAPTURE_NOT_IDLE')
    if action not in ('stop','abort'):raise ValueError('NOISE_CAPTURE_ACTION')
    if model.compute_logits is not state['wrapper']:raise RuntimeError('NOISE_CAPTURE_OWNER_CHANGED')
    del model.compute_logits;del worker._dcat_noise_capture
    if action=='stop' and len(state['steps'])!=4:raise RuntimeError('NOISE_FOUR_FORWARDS_MISSING')
    return dict(rank=worker.rank,worker_pid=os.getpid(),phase=state['phase'],steps=state['steps'],capture_removed=True,
        qualification_granted=False)
