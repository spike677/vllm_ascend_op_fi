"""Trusted isolated-instance observe callable; launches no Worker or inference.

Deployment calls this through its own authorized collective_rpc adapter. Never
register as a public untrusted arbitrary-code endpoint.
"""
from pathlib import Path
import hashlib,importlib
from fault_injection.sites.dsv4_observer import Dsv4ObserveMode,empty_report,export_report


def source_fingerprints(paths):
    pkg=importlib.import_module('vllm_ascend');root=Path(pkg.__file__).resolve().parent
    result=[]
    for name in paths:
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts or rel.parts[0]!='vllm_ascend':raise ValueError('SOURCE_PATH_OUTSIDE_PACKAGE')
        p=root.joinpath(*rel.parts[1:]).resolve()
        if not p.is_relative_to(root):raise ValueError('SOURCE_PATH_OUTSIDE_PACKAGE')
        result.append(dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None))
    return result


def observe(wrapper,action,options=None):
    worker=getattr(wrapper,'worker',wrapper);runner=worker.model_runner;options=options or {}
    if action=='source':return dict(rank=worker.rank,files=source_fingerprints(options['paths']))
    if action=='start':
        if hasattr(worker,'_dcat_dsv4_observer') or 'execute_model' in vars(runner):raise RuntimeError('OBSERVE_BOUNDARY_ALREADY_OWNED')
        state=dict(report=empty_report(),original=runner.execute_model,inside=0,steps=0,unknown_phase=0)
        def execute(scheduler,*args,**kwargs):
            ids=list(scheduler.num_scheduled_tokens);new={r.req_id for r in scheduler.scheduled_new_reqs}
            phase=[]
            for rid in ids:
                if rid in new:phase.append('prefill');continue
                req=getattr(runner,'requests',{}).get(rid)
                computed=getattr(req,'num_computed_tokens',None);tokens=getattr(req,'prompt_token_ids',None)
                phase.append('unknown' if computed is None or tokens is None else 'decode' if computed>=len(tokens) else 'prefill')
            label=phase[0] if phase and len(set(phase))==1 else 'mixed_or_unknown'
            state['steps']+=1;state['inside']+=1;state['unknown_phase']+=int(label in ('unknown','mixed_or_unknown'))
            try:
                with Dsv4ObserveMode(state['report'],requests=ids,phase=label):return state['original'](scheduler,*args,**kwargs)
            finally:state['inside']-=1
        state['wrapper']=execute;runner.execute_model=execute;worker._dcat_dsv4_observer=state
        return dict(rank=worker.rank,status='READ_ONLY_OBSERVER_INSTALLED',qualification_granted=False)
    state=getattr(worker,'_dcat_dsv4_observer',None)
    if state is None:raise RuntimeError('OBSERVE_NOT_INSTALLED')
    if state['inside']:raise RuntimeError('OBSERVE_REQUEST_INFLIGHT')
    result=export_report(state['report'],worker.rank);result.update(steps=state['steps'],unknown_phase_steps=state['unknown_phase'])
    if action=='stop':
        if runner.execute_model is not state['wrapper']:raise RuntimeError('OBSERVE_OWNER_CHANGED')
        del runner.execute_model;del worker._dcat_dsv4_observer
        result['observer_removed']=True
    elif action!='status':raise ValueError('UNKNOWN_OBSERVE_ACTION')
    return result
