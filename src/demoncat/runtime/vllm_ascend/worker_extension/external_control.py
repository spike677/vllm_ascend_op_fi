"""Trusted control/probe replies; never wrap model forward or device kernels.

vLLM 0.26's multiprocess queue has no message correlation after a rank exception.
Contain control exceptions on each Worker and validate every rank before use.
This cannot repair a queue previously desynchronized by unrelated code.
"""
from .serving_context import ContextRefused


def control_reply(worker, request_token, operation):
    # This is trusted Python bootstrap code, never a user code loader. Even bad
    # probe input is serialized instead of escaping the executor's Worker loop.
    rank=None
    try:
        rank=worker.demoncat_worker_identity()['worker']['rank']
        if type(rank) is not int or rank < 0:raise ValueError('CONTROL_RANK_REQUIRED')
        if type(request_token) is not str or not 1<=len(request_token)<=128:
            raise ValueError('CONTROL_REQUEST_TOKEN_REQUIRED')
        if not callable(operation):raise ValueError('TRUSTED_CONTROL_CALLBACK_REQUIRED')
        result=operation()
        # Ensure serialization errors do not occur after leaving this boundary.
        import json
        json.dumps(result,allow_nan=False)
        return dict(schema='demoncat.external_control.v1',rank=rank,request_token=request_token,
                    status='CONTROL_OK',exit_code=0,result=result)
    except Exception as error:
        expected=isinstance(error,(ContextRefused,ValueError,PermissionError))
        return dict(schema='demoncat.external_control.v1',rank=rank,request_token=request_token
                    if type(request_token) is str and len(request_token)<=128 else None,
                    status='CONTROL_REFUSED' if expected else 'CONTROL_ERROR',exit_code=4 if expected else 5,
                    error_type=type(error).__name__,reason=str(error))


def validate_control_replies(rows, *, request_token, world_size):
    if (type(world_size) is not int or world_size<1 or type(rows) not in (list,tuple)
        or len(rows)!=world_size or any(type(r) is not dict or type(r.get('rank')) is not int
            or r.get('schema')!='demoncat.external_control.v1' or r.get('request_token')!=request_token for r in rows)
        or sorted(r['rank'] for r in rows)!=list(range(world_size))):
        raise RuntimeError('CONTROL_RPC_CORRELATION_UNPROVEN_STOP_AND_RESTART_EXECUTOR')
    rows=sorted(rows,key=lambda r:r['rank'])
    for row in rows:
        if row.get('status') not in ('CONTROL_OK','CONTROL_REFUSED','CONTROL_ERROR'):
            raise RuntimeError('CONTROL_RPC_REPLY_SCHEMA_INVALID')
        if row.get('exit_code')!={'CONTROL_OK':0,'CONTROL_REFUSED':4,'CONTROL_ERROR':5}[row['status']]:
            raise RuntimeError('CONTROL_RPC_REPLY_SCHEMA_INVALID')
    return rows
