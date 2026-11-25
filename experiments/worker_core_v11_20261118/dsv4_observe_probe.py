"""Trusted read-only observe RPC. Tools own only their explicit method leases."""
from pathlib import Path
import hashlib, importlib
from fault_injection.sites.dsv4_observer import empty_report, export_report
from experiments.worker_core_v11_20261118.dsv4_probe_support import MethodLease, restore_all, deployment_identity, ProbeRefused
from experiments.worker_core_v11_20261118.dsv4_probe_journal import CallJournal


def source_fingerprints(paths):
    pkg=importlib.import_module('vllm_ascend');root=Path(pkg.__file__).resolve().parent
    result=[]
    for name in paths:
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts or not rel.parts or rel.parts[0]!='vllm_ascend':raise ValueError('SOURCE_PATH_OUTSIDE_PACKAGE')
        p=root.joinpath(*rel.parts[1:]).resolve()
        if not p.is_relative_to(root):raise ValueError('SOURCE_PATH_OUTSIDE_PACKAGE')
        result.append(dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None))
    return result


def observe(wrapper,action,options=None):
    worker=getattr(wrapper,'worker',wrapper);runner=worker.model_runner;options=options or {}
    if action=='identity':return deployment_identity(worker)
    if action=='source':return dict(rank=worker.rank,files=source_fingerprints(options['paths']))
    if action=='start':
        if hasattr(worker,'_dcat_dsv4_observer'):raise ProbeRefused('OBSERVE_BOUNDARY_ALREADY_OWNED')
        max_calls=options.get('max_calls',4096);integer_budget=options.get('integer_budget',262144)
        if type(max_calls) is not int or not 1<=max_calls<=16384 or type(integer_budget) is not int or not 0<=integer_budget<=4194304:
            raise ProbeRefused('PROBE_CAPTURE_BUDGET_OUT_OF_RANGE')
        lease=MethodLease(runner,'execute_model',options.get('approved_baselines'))
        identity=deployment_identity(worker)
        state=dict(report=empty_report(),lease=lease,inside=0,steps=0,unknown_phase=0,calls=[],dispatches=[],
                   truncated=False,errors=[],max_calls=max_calls,integer_budget_remaining=integer_budget,identity=identity)
        modules={id(m):n for n,m in runner.model.named_modules()} if hasattr(runner.model,'named_modules') else {}
        def execute(*args,**kwargs):
            scheduler=args[0] if args else kwargs.get('scheduler_output',kwargs.get('scheduler'))
            state['inside']+=1;state['steps']+=1
            try:
                ids=list(scheduler.num_scheduled_tokens);new={r.req_id for r in scheduler.scheduled_new_reqs}
                phases=[]
                for rid in ids:
                    req=getattr(runner,'requests',{}).get(rid)
                    computed=getattr(req,'num_computed_tokens',None);tokens=getattr(req,'prompt_token_ids',None)
                    phases.append('prefill' if rid in new else 'unknown' if computed is None or tokens is None else 'decode' if computed>=len(tokens) else 'prefill')
                label=phases[0] if phases and len(set(phases))==1 else 'mixed_or_unknown'
                state['unknown_phase']+=int(label in ('unknown','mixed_or_unknown'))
                mode=CallJournal(state,runner,scheduler,worker.rank,identity['groups'],modules,requests=ids,phase=label)
            except Exception as error:
                state['errors'].append(dict(stage='scheduler_observation',error=repr(error)));mode=None
            try:
                if mode is None:return lease.original(*args,**kwargs)
                with mode:return lease.original(*args,**kwargs)
            finally:state['inside']-=1
        try:
            lease.install(execute);worker._dcat_dsv4_observer=state
        except BaseException:
            if lease.owned():lease.restore()
            raise
        return dict(rank=worker.rank,status='READ_ONLY_OBSERVER_INSTALLED',baseline=lease.identity,qualification_granted=False)
    if action not in ('status','stop'):raise ValueError('UNKNOWN_OBSERVE_ACTION')
    state=getattr(worker,'_dcat_dsv4_observer',None)
    if state is None:return dict(rank=worker.rank,status='OBSERVE_NOT_INSTALLED',installed=False,qualification_granted=False)
    if state['inside']:raise ProbeRefused('OBSERVE_REQUEST_INFLIGHT')
    result=export_report(state['report'],worker.rank)
    result.update(steps=state['steps'],unknown_phase_steps=state['unknown_phase'],calls=state['calls'],
                  truncated=state['truncated'],observation_errors=state['errors'],identity=state['identity'],
                  installed=True,combine_output_row_join='COMBINE_OUTPUT_ROW_JOIN_UNPROVEN')
    if action=='stop':
        restore_all([state['lease']]);del worker._dcat_dsv4_observer
        result.update(observer_removed=True,installed=False)
    return result
