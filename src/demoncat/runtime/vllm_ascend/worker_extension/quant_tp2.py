"""TP2 ordinary batch quantization; one Native writer, one read-only F guard.

Explicit current group/blue permit only. Calibration remains per-rank and real;
normal inputs and guard Q/scale must match exactly, not just final logits.
"""
from copy import deepcopy
from dataclasses import dataclass
import os,threading
from .serving_context import require
from .serving_quant_projection import QuantProjectionServingSession,MODES
from .rms18_tp2 import validate_tp2_group,digest
from .multi_rank_prerequisites import tp2_eager_assessment

@dataclass(frozen=True)
class TP2QuantBinding:
    make_session: object

class TP2QuantSession(QuantProjectionServingSession):
    parallel_sizes=(2,)
    def __init__(self,*args,group_context,**kwargs):
        self.group_context=group_context
        self.expected_rank=group_context.identity['rank']
        super().__init__(*args,**kwargs)
        require(self.rpc_identity==group_context.identity,'TP2_QUANT_CURRENT_IDENTITY_REQUIRED')
        require(self.processor.use_all_gather is True,'TP2_QUANT_ALL_RANK_LOGITS_REQUIRED')

    def _require_runner(self):
        c=self.group_context
        report=tp2_eager_assessment(self.runner,c.tp_group,self.expected_rank,expected_max_seqs=8)
        require(not report['unmet'],'TP2_QUANT_PREREQUISITES:'+str(report['unmet']))

    def _snapshot_decode(self,actual,context):
        from .batch_context import _snapshot_first_decode_rows
        self._require_runner()
        rows=_snapshot_first_decode_rows(self.runner,self.model,actual,context,self.expected,expected_max_seqs=8)
        require(rows['request_count']==8,'TP2_EIGHT_ACTUAL_DECODE_ROWS_REQUIRED')
        return rows

    def _fault_injector(self):
        c=self.group_context
        require(c.decision is not None,'TP2_QUANT_COMMITTED_ROLE_REQUIRED')
        return c.decision['selected_rank']==self.expected_rank

    def _native_mode(self,phase):
        if phase=='fault' and not self._fault_injector():return None
        return MODES.get(phase)

    def group_proof(self):
        c=self.group_context;n=self.rows[0] if self.rows else None
        rows=None if n is None else n['decode']['decode']
        return dict(owner=c.contract['owner'],group_generation=1,worker=deepcopy(self.rpc_identity),
            members_sha256=c.contract['members_sha256'],actual_normal_request_observed=n is not None,
            eligible_target=self.target is not None,normal_request_id=None if rows is None else rows['request_ids'],
            request_fingerprint=None if rows is None else digest(self.row_signature(rows)),
            selected_rank=None if c.decision is None else c.decision['selected_rank'],
            role=None if c.decision is None else ('injector' if c.decision['selected_rank']==self.expected_rank else 'guard'))

    def commit_role(self,decision):
        self._identity();c=self.group_context;proof=self.group_proof()
        require(c.decision is None and self.phase_name()=='fault' and self.expected is None
            and [r['phase'] for r in self.rows]==['normal','reference','lossless','calibration_off'],
            'TP2_QUANT_ROLE_ONCE_AFTER_CURRENT_CALIBRATION')
        require(decision.get('group')==c.contract and decision.get('selected_rank') in (0,1)
            and decision.get('normal_request_id')==proof['normal_request_id']
            and decision.get('request_fingerprint')==proof['request_fingerprint']
            and {str(k):v for k,v in decision.get('roles',{}).items()}=={str(i):'injector' if i==decision['selected_rank'] else 'guard' for i in (0,1)},
            'TP2_QUANT_ROLE_NOT_BOUND_TO_CURRENT_NORMAL')
        c.decision=deepcopy(decision)
        return self.group_proof()

    def _requires_off(self):
        return self.group_context.decision is not None

    def report(self):
        row=super().report()
        row.update(scope='ONE_ORDINARY_W8A8_PROJECTION_TP2_B8_FIRST_DECODE',multi_rank=self.group_proof(),
            guard_boundary_unchanged=(self.group_context.decision is not None and not self._fault_injector()
                and any(x['phase']=='fault' and x['projection']['hashes']==self.healthy for x in self.rows)))
        return row


def install_quant_tp2_binding(worker,*,root,build_sha256,group_contract):
    from fault_injection.providers import register_runtime
    from .serving_quant import MECHANISM
    c=validate_tp2_group(worker,MECHANISM,group_contract,expected_max_seqs=8)
    require(not hasattr(worker,'_dcat_quant_binding') and not hasattr(worker,'_demoncat_shared_scale_inventory'),
            'EXCLUSIVE_QUANT_DEPLOYMENT_BINDING_REQUIRED')
    require(os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING')=='0' and os.environ.get('TASK_QUEUE_ENABLE')=='0',
            'HOST_DRIVEN_SYNCHRONOUS_QUANT_REQUIRED')
    for item in (worker,worker.model_runner):
        require(not any(x in vars(item) for x in ('execute_model','_prepare_inputs')),'CORE_OVERRIDE_REFUSED')
    model=worker.model_runner.model
    def make_session(actual_worker,actual_model,*,owner,generation,worker_identity):
        require(actual_worker is worker and actual_model is model and worker.model_runner.model is model
            and owner==c.contract['owner'] and worker_identity==c.identity,'TP2_QUANT_GROUP_OWNER_REQUIRED')
        return TP2QuantSession(worker,model,root=root,build_sha256=build_sha256,owner=owner,
            generation=generation,worker_identity=worker_identity,group_context=c)
    def factory(actual_model):
        require(actual_model is model,'FOREIGN_MODEL')
        return TP2QuantBinding(make_session)
    register_runtime(MECHANISM,factory);worker._dcat_quant_binding=factory
    worker._dcat_quant_ordinary_model=model
    return dict(status='TP2_BOUND_NOT_QUALIFIED',rank=c.identity['rank'],native_owner_installed=False)


def commit_quant_tp2_role(worker,decision):
    runtime=getattr(worker,'_demoncat_worker_fault_runtime',None)
    lease=getattr(runtime,'_leases',{}).get('quant.batch_shared_scale')
    require(lease is not None and type(lease.provider.session) is TP2QuantSession,'TP2_QUANT_CURRENT_LEASE_REQUIRED')
    return lease.provider.session.commit_role(decision)
