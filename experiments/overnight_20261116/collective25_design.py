"""CPU-only Fault25 communication/rounding design; no provider registration.

This demonstrates an association-order mechanism and validates scatter layout.
It never executes HCCL/NCCL, does not synthesize a supposedly measured native
delta, and is not an implementation of a collectives interposer.
"""
from dataclasses import dataclass
import math
import struct


@dataclass(frozen=True)
class GroupPlan:
    members: tuple[int, ...]
    sequence: int
    receive_count: int
    operation: str = 'sum'
    dtype: str = 'float32'
    evidence_kind: str = 'CPU_DESIGN_ONLY'

    def __post_init__(self):
        if (len(self.members)<2 or any(type(x) is not int or x<0 for x in self.members)
                or len(set(self.members))!=len(self.members)):
            raise ValueError('Distinct real group-member identifiers are required')
        if type(self.sequence) is not int or self.sequence<0:
            raise ValueError('Actual per-group collective sequence required')
        if type(self.receive_count) is not int or self.receive_count<1:
            raise ValueError('Positive receive count required')
        if self.operation!='sum' or self.dtype!='float32' or self.evidence_kind!='CPU_DESIGN_ONLY':
            raise ValueError('This CPU design supports FP32 SUM only and cannot claim native proof')

    @property
    def send_count(self):
        return len(self.members)*self.receive_count

    def shard(self,global_rank):
        local=self.members.index(global_rank)
        return local*self.receive_count,(local+1)*self.receive_count


def fp32(value):
    try:return struct.unpack('<f',struct.pack('<f',value))[0]
    except OverflowError:return math.copysign(math.inf,value)


def simulate(plan, inputs_by_global_rank, contribution_order):
    """Each contribution participates exactly once; only addition order changes."""
    if set(inputs_by_global_rank)!=set(plan.members):raise ValueError('Group/input ranks differ')
    if len(contribution_order)!=len(plan.members) or set(contribution_order)!=set(plan.members):
        raise ValueError('Dropping or duplicating a contribution changes the mechanism')
    rows=[]
    for rank in plan.members:
        values=inputs_by_global_rank[rank]
        if len(values)!=plan.send_count or any(not math.isfinite(v) for v in values):
            raise ValueError('Invalid input count/nonfinite healthy input')
        rows.append(values)
    reduced=[]
    for col in range(plan.send_count):
        acc=fp32(inputs_by_global_rank[contribution_order[0]][col])
        for rank in contribution_order[1:]:
            acc=fp32(acc+fp32(inputs_by_global_rank[rank][col]))
        reduced.append(acc)
    return {rank:reduced[slice(*plan.shard(rank))] for rank in plan.members}


def require_measured_delta(sample, expected_identity):
    """Admission design only: no arbitrary noise or CPU delta can be called native."""
    if sample.get('source')!='OBSERVED_NATIVE_SAME_INPUT_PAIR':
        raise ValueError('A real same-input Native B/F sample is required')
    if sample.get('identity')!=expected_identity:
        raise ValueError('Group/rank/input/sequence/layout identity differs')
    required=('process_group','members','sequence','operator','input_sha256','layout_sha256',
              'loaded_build_sha256','soc','dtype','reduce_op','send_count','receive_count')
    if any(k not in expected_identity for k in required):
        raise ValueError('Incomplete actual communication identity')
    if sample.get('native_execution_verified') is not True or sample.get('input_equal_verified') is not True:
        raise ValueError('Native execution or exact input equality unproven')
    raise NotImplementedError('Native replay provider is not implemented; this design cannot arm it')
