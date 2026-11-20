"""TP2 source-bound fused RoPE: one actual rank writes, others observe original."""
from .rms18_tp2 import TP2Rms18Provider,TP2Rms18Request,_install_tp2_tensor_binding
from .rope22_task import Rope22TaskProvider,Rope22TaskRequest
from .serving_context import require

def install_rope22_tp2_binding(worker,**kwargs):
    return _install_tp2_tensor_binding(worker,'22.wrong_row',**kwargs)

class TP2Rope22Request(TP2Rms18Request,Rope22TaskRequest):
    pass

class TP2Rope22Provider(TP2Rms18Provider,Rope22TaskProvider):
    def make_request(self):return TP2Rope22Request(self)

def commit_rope22_tp2_role(worker,decision):
    runtime=getattr(worker,'_demoncat_worker_fault_runtime',None)
    lease=getattr(runtime,'_leases',{}).get('22.wrong_row')
    require(lease is not None and type(lease.provider) is TP2Rope22Provider,'TP2_ROPE_RUNTIME_LEASE_REQUIRED')
    return lease.provider.commit_role(decision)
