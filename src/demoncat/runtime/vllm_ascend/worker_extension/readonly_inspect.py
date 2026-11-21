"""Bounded structural RPC, deliberately independent of injection prerequisites.

No model forward, tensor readback, native load, attachment or permit mutation.
Missing rank responses are evidence gaps, not inferred identities or success.
"""
from .deployment_acceptance import observe_deployment


def inspect_worker(worker,*,offset=0,limit=128):
    if type(offset) is not int or offset<0 or type(limit) is not int or not 1<=limit<=128:
        raise ValueError('BOUNDED_INSPECTION_PAGE_REQUIRED')
    runner=getattr(worker,'model_runner',None);model=getattr(runner,'model',None)
    rank=getattr(worker,'rank',None)
    if type(rank) is not int or rank<0:return dict(status='INSPECTION_UNAVAILABLE',rank=None,reason='RANK_MISSING')
    if model is None:return dict(status='INSPECTION_UNAVAILABLE',rank=rank,reason='MODEL_NOT_READY')
    rows=[];next_offset=None
    for index,(name,module) in enumerate(model.named_modules()):
        if index<offset:continue
        if len(rows)==limit:next_offset=index;break
        rows.append(dict(path=name,module_type=type(module).__module__+'.'+type(module).__qualname__))
    if runner.model is not model:raise RuntimeError('INSPECTION_MODEL_CHANGED')
    return dict(status='READ_ONLY_INSPECTION',rank=rank,read_only=True,
        actual=observe_deployment(worker,model),modules=rows,next_offset=next_offset,
        limits=dict(offset=offset,limit=limit),qualification_granted=False)


def aggregate_inspection(expected_ranks,responses):
    expected=list(expected_ranks)
    if not expected or len(set(expected))!=len(expected) or any(type(r) is not int or r<0 for r in expected):
        raise ValueError('EXPLICIT_RANK_SET_REQUIRED')
    ranks={};errors=[]
    for row in responses:
        rank=row.get('rank') if isinstance(row,dict) else None
        if type(rank) is not int or rank not in expected:errors.append('UNIDENTIFIED_OR_UNEXPECTED_RESPONSE');continue
        if rank in ranks:errors.append('DUPLICATE_RANK:'+str(rank));continue
        ranks[rank]=row
    missing=[r for r in expected if r not in ranks]
    failed=[r for r,row in ranks.items() if row.get('status')!='READ_ONLY_INSPECTION']
    return dict(status='PARTIAL' if missing or failed or errors else 'COMPLETE',
        ranks=ranks,missing_ranks=missing,failed_ranks=failed,errors=errors,qualification_granted=False)


def inspect_group(executor,expected_ranks,*,offset=0,limit=128,timeout=30):
    """A failed transport cannot invent partial results it did not deliver."""
    try:
        rows=executor.collective_rpc('demoncat_inspect_readonly',args=(offset,limit),timeout=timeout)
    except Exception as error:
        result=aggregate_inspection(expected_ranks,[])
        result['transport_error_type']=type(error).__name__
        return result
    return aggregate_inspection(expected_ranks,rows)
