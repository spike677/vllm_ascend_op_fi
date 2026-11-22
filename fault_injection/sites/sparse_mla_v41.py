"""CPU-reviewable lowering of actual SMLA TND/PA_BBND call arguments.

Not a deployment grant or installed hook. Packed/quantized cache paths are
rejected until their decoding and consumer binding are independently reviewed.
"""
from dataclasses import dataclass
import torch
from ..engines.attention import recompute_attention_row


def require(ok, reason):
    if not ok:raise ValueError(reason)


def ints(value, name):
    require(isinstance(value,torch.Tensor) and value.dtype in (torch.int32,torch.int64), name)
    return value.detach().cpu().tolist()


@dataclass
class SparseMlaRow:
    query: object
    keys: object
    sinks: object
    scale: float
    request_id: str
    query_row: int
    selected: tuple

    def recompute(self, *, block_size, replay_blocks=()):
        return recompute_attention_row(self.query,self.keys,self.keys,scale=self.scale,
            block_size=block_size,replay_blocks=replay_blocks,sinks=self.sinks)


def from_operator_call(q, kwargs, *, row, request_ids):
    """Use native call coordinates including SWA rebase and compressed residual.

request_ids MUST come from the runner's current TND order, not a config or
normal-phase census. The runtime adapter still has to prove that join.
"""
    a=kwargs
    require(a.get('layout_q')=='TND' and a.get('layout_kv')=='PA_BBND'
        and a.get('topk_value_mode')==1,'SMLA_LAYOUT_NOT_REVIEWED')
    require(isinstance(q,torch.Tensor) and q.ndim==3 and q.shape[-1]==512
        and q.is_floating_point() and type(row)==int and 0<=row<q.shape[0], 'SMLA_ACTUAL_QUERY_REQUIRED')
    starts=ints(a.get('cu_seqlens_q'),'SMLA_QUERY_BOUNDARIES_REQUIRED')
    require(len(starts)==len(request_ids)+1 and starts[0]==0 and starts[-1]==q.shape[0]
        and all(x<y for x,y in zip(starts,starts[1:]))
        and len(set(request_ids))==len(request_ids) and all(type(r) is str and r for r in request_ids),
        'SMLA_CURRENT_REQUEST_ORDER_REQUIRED')
    batch=next(i for i in range(len(request_ids)) if starts[i]<=row<starts[i+1])
    local=row-starts[batch];nq=starts[batch+1]-starts[batch]
    sinks=a.get('sinks')
    require(isinstance(sinks,torch.Tensor) and sinks.shape==(q.shape[1],), 'SMLA_CALLER_SINKS_REQUIRED')
    selected=[];values=[]
    def gather(kind, visible, lower=0):
        cache=a.get(kind+'_kv');table=a.get(kind+'_block_table')
        require(isinstance(cache,torch.Tensor) and cache.ndim==4 and cache.shape[2:]==(1,512)
            and cache.is_floating_point() and cache.device==q.device and cache.dtype==q.dtype
            and cache.is_contiguous(), 'SMLA_PACKED_OR_UNKNOWN_CACHE_REFUSED')
        tables=ints(table,'SMLA_REAL_BLOCK_TABLE_REQUIRED')
        require(len(tables)==len(request_ids),'SMLA_BLOCK_REQUEST_COUNT')
        sequence=ints(a.get('seqused_'+kind+'_kv'),'SMLA_REAL_USED_LENGTH_REQUIRED')
        require(len(sequence)==len(request_ids) and 0<=visible<=sequence[batch], 'SMLA_VISIBLE_LENGTH_INVALID')
        indices=a.get(kind+'_sparse_indices')
        if indices is None:
            ids=list(range(lower,visible))
        else:
            shape=tuple(indices.shape)
            require(len(shape)==3 and shape[:2]==(q.shape[0],1), 'SMLA_SPARSE_INDEX_LAYOUT')
            ids=ints(indices,'SMLA_INTEGER_INDICES_REQUIRED')[row][0]
            if kind=='ori':
                lengths=ints(a.get('ori_topk_length'),'SMLA_ORI_TOPK_LENGTH_REQUIRED')
                length=lengths[row][0] if isinstance(lengths[row],list) else lengths[row]
                require(type(length)==int and 0<=length<=len(ids),'SMLA_TOPK_LENGTH_INVALID')
                ids=ids[:length]
            ids=[i for i in ids if lower<=i<visible]
            require(len(ids)==len(set(ids)), 'SMLA_DUPLICATE_VISIBLE_KEYS_UNREVIEWED')
        block=cache.shape[1]
        for i in ids:
            require(i//block<len(tables[batch]),'SMLA_LOGICAL_PAGE_MISSING')
            page=tables[batch][i//block]
            require(0<=page<cache.shape[0], 'SMLA_PHYSICAL_PAGE_OUT_OF_RANGE')
            selected.append((kind,i,page,i%block));values.append(cache[page,i%block])
    lengths=ints(a.get('seqused_ori_kv'),'SMLA_ORI_LENGTH_REQUIRED')
    require(len(lengths)==len(request_ids),'SMLA_ORI_REQUEST_COUNT')
    position=lengths[batch]-nq+local
    require(position>=0,'SMLA_QUERY_OUTSIDE_ORI_TIME_AXIS')
    mode=a.get('ori_mask_mode')
    require(mode in (3,4),'SMLA_NONCAUSAL_ORI_MODE_UNREVIEWED')
    if mode==4:
        left=a.get('ori_win_left');right=a.get('ori_win_right')
        require(type(left)==int and left>=0 and right==0,'SMLA_WINDOW_UNREVIEWED')
        lower=max(0,position-left)
    else:lower=0
    gather('ori',position+1,lower)
    if a.get('cmp_kv') is not None:
        ratio=a.get('cmp_ratio')
        require(ratio in (1,2) and a.get('cmp_mask_mode')==3,'SMLA_COMPRESSION_MODE_UNREVIEWED')
        counts=ints(a.get('seqused_cmp_kv'),'SMLA_COMPRESSED_LENGTH_REQUIRED')
        residual=ints(a.get('cmp_residual_kv'),'SMLA_COMPRESSED_RESIDUAL_REQUIRED')
        require(len(counts)==len(residual)==len(request_ids) and 0<=residual[batch]<ratio,
            'SMLA_COMPRESSED_COORDINATES_INVALID')
        original_length=counts[batch]*ratio+residual[batch]
        visible=(original_length-nq+local+1)//ratio
        gather('cmp',visible)
    else:require(a.get('cmp_ratio')==0 and a.get('cmp_sparse_indices') is None,'SMLA_MISSING_COMPRESSED_CACHE')
    require(bool(values),'SMLA_NO_VALID_KEYS')
    return SparseMlaRow(q[row],torch.stack(values),sinks,a['softmax_scale'],request_ids[batch],row,tuple(selected))
