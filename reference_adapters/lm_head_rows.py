"""Library12 persistent row construction, built on the existing v2 leases.

This module supplies the region adapter, not public qualification. A trusted
model binding must prove the actual weight reader and sampler before admission.
Unknown derived weights and tied embeddings are refused, never inferred absent.
"""
from dataclasses import dataclass
import torch
from fault_injection.engines.region_lease import identity,exact
from demoncat.runtime.vllm_ascend.worker_extension.external_persistent_regions import PersistentRegion, PersistentRegions
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require

# Deployment-owned constants. There is deliberately no request/CLI token field.
TOKEN_IDS=(10,11,12)
VARIANTS=('row12_lm_head_collapse','row12_lm_head_swap')


def is_weight_reader(name,args,weight):
    if name not in ('aten::mm','aten::linear','npu::npu_linear','vllm::unquantized_gemm') or len(args)<2:
        return False
    w=args[1]
    if not isinstance(w,torch.Tensor) or w.dtype!=weight.dtype or w.storage_offset()!=weight.storage_offset():
        return False
    expected=weight.t() if name=='aten::mm' else weight
    return (w.untyped_storage()._cdata==weight.untyped_storage()._cdata
            and w.shape==expected.shape and w.stride()==expected.stride())


@dataclass(frozen=True)
class RowAdmission:
    """Filled only by a source-reviewed, actual-call census on the Worker."""
    vocabulary_size: int
    vocabulary_start: int
    vocabulary_end: int
    padded_start: int
    padded_end: int
    derived_copies: object
    tied_embedding: bool
    source_sha256: str
    actual_weight_consumer: str


def validate(weight, admission):
    require(type(admission) is RowAdmission,'ROW12_TYPED_ADMISSION_REQUIRED')
    a=admission
    require(type(a.derived_copies) is tuple and a.derived_copies==(),
            'ROW12_DERIVED_COPIES_UNSUPPORTED_OR_UNPROVEN')
    require(a.tied_embedding is False,'ROW12_TIED_EMBEDDING_UNSUPPORTED')
    require(isinstance(a.source_sha256,str) and len(a.source_sha256)==64
        and all(c in '0123456789abcdef' for c in a.source_sha256)
        and isinstance(a.actual_weight_consumer,str) and a.actual_weight_consumer,
        'ROW12_REVIEWED_ACTUAL_WEIGHT_READER_REQUIRED')
    require(isinstance(weight,torch.Tensor) and weight.ndim==2
        and weight.dtype==torch.bfloat16 and weight.is_contiguous(),
        'ROW12_BF16_ND_WEIGHT_REQUIRED')
    fields=(a.vocabulary_size,a.vocabulary_start,a.vocabulary_end,a.padded_start,a.padded_end)
    require(all(type(v) is int for v in fields) and a.vocabulary_size>max(TOKEN_IDS)
        and 0<=a.padded_start<=a.vocabulary_start<=a.vocabulary_end<=a.vocabulary_size
        and a.vocabulary_end<=a.padded_end
        and a.padded_end-a.padded_start==weight.shape[0],
        'ROW12_VOCAB_SHARD_MAPPING_INVALID')
    owns=[a.vocabulary_start<=t<a.vocabulary_end for t in TOKEN_IDS]
    # Two-row swapping across ranks requires an explicitly reviewed exchange;
    # the first fixed variant avoids inventing communication at a hit point.
    require(not any(owns) or all(owns),'ROW12_FIXED_GROUP_SPLIT_ACROSS_RANKS')
    return all(owns)


class LmHeadRows:
    """One contiguous owned interval, so view version counters cannot conflict."""
    def __init__(self,weight,admission,*,variant,leases,boundary,observe_weight):
        require(variant in VARIANTS,'ROW12_UNKNOWN_NAMED_VARIANT')
        require(callable(observe_weight),'ROW12_LIVE_OBJECT_OBSERVER_REQUIRED')
        self.owns_rows=validate(weight,admission)
        self.weight=weight;self.admission=admission;self.variant=variant
        self.observe_weight=observe_weight;self.parent_identity=identity(weight)
        self.regions=PersistentRegions(leases,boundary=boundary,maximum_bytes=1024*1024)
        self.region=None;self.started=False;self.closed=False;self.guard_reads=0
        if self.owns_rows:
            self.start=TOKEN_IDS[0]-admission.padded_start
            self.region=weight[self.start:self.start+len(TOKEN_IDS)]
            self.original=self.region.detach().clone()
        else:
            # Strong reference and digest of this rank's own shard are a guard,
            # not a write lease or fabricated eligible call.
            from fault_injection.sites.mla28 import fingerprint
            self.guard_sha=fingerprint(weight)

    def current(self):
        weight=self.observe_weight()
        require(weight is self.weight and identity(weight)==self.parent_identity,
                'ROW12_WEIGHT_ALLOCATION_CHANGED_NO_WRITEBACK')
        return identity(weight[self.start:self.start+len(TOKEN_IDS)]) if self.owns_rows else identity(weight)

    def enable(self):
        require(not self.started and not self.closed,'ROW12_NEW_SCOPE_REQUIRED')
        self.regions._boundary();self.current()
        if self.owns_rows:
            r=self.region
            self.regions.prepare([PersistentRegion('token_rows',r,'persistent_weight',identity(r),
                self.current,(self.admission.actual_weight_consumer,),())])
            value=self.original.detach().clone()
            if self.variant=='row12_lm_head_collapse':value[:]=self.original[0]
            else:
                value[0]=self.original[1];value[1]=self.original[0]
            self.regions.write('token_rows',value)
        self.started=True

    def consumed(self,actual_weight,*,consumer,request_id):
        require(self.started and not self.closed,'ROW12_NOT_ENABLED')
        self.current()
        require(actual_weight is self.weight and consumer==self.admission.actual_weight_consumer
                and isinstance(request_id,str) and request_id,'ROW12_ACTUAL_READER_MISMATCH')
        if self.owns_rows:
            entry=self.regions.entries['token_rows']
            expected=self.regions.leases._get(entry['token'])['expected']
            require(exact(self.region,expected),'ROW12_READER_BYTES_CHANGED')
            self.regions.consumed('token_rows',consumer,request_generation=request_id)
        else:
            self._guard();self.guard_reads+=1

    def _guard(self):
        from fault_injection.sites.mla28 import fingerprint
        require(fingerprint(self.weight)==self.guard_sha,'ROW12_NONOWNER_WEIGHT_CHANGED')

    def close(self):
        self.regions._boundary();self.current()
        if self.owns_rows:
            for name,entry in tuple(self.regions.entries.items()):
                if not entry['events']:self.regions.release_unwritten(name)
                elif entry['readers']:self.regions.restore(name)
                else:self.regions.rollback_unconsumed_weight(name)
        else:self._guard()
        self.closed=True

    def snapshot(self):
        return dict(self.regions.snapshot(),variant=self.variant,token_ids=list(TOKEN_IDS),
            owns_rows=self.owns_rows,nonowner_guard_reads=self.guard_reads,
            source_sha256=self.admission.source_sha256,
            semantics='ORIGINAL_DESCRIBED_WEIGHT_VECTOR_CONSTRUCTION',
            sampler_consumption_proven=False,qualification_granted=False)
