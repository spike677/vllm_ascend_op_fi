"""Closed D2 typed-call adapters for reviewed 0.32 source, CPU-exercisable.

No public factory or hook registration. Live construction requires D1 review
which is intentionally unavailable. These methods consume actual tensors, not
metadata booleans. Caller/consumer attachment and packed cache decoding remain
unqualified until the observed source, schema and request joins are reviewed.
"""
import torch
from .mla28 import fingerprint
from .sparse_mla_v41 import from_operator_call
from ..engines.matmul_range import recompute_cell
from ..engines.region_lease import RegionLeases


def create_live_adapter(*args, **kwargs):
    raise RuntimeError('DSV4_D2_GATE_CLOSED_D1_REVIEW_REQUIRED')


def require(ok, reason):
    if not ok: raise ValueError(reason)


class D2CallAdapter:
    """Typed call lowering; live factory stays closed until source-bound D1 review."""
    def __init__(self, leases, *, fault=True):
        self.fault=fault;self.events=[];self.pending=[];self.leases=leases

    def tensors(self,*values):
        for value in values:
            require(isinstance(value,torch.Tensor),'D2_REAL_TENSOR_REQUIRED')

    def watch(self,target,value,mechanism):
        self.tensors(target,value)
        require(target.shape==value.shape and target.dtype==value.dtype,'D2_RESULT_LAYOUT_CHANGED')
        e=dict(mechanism=mechanism,applied=False,changed=0,changed_elements=0,consumed=False,restored=False)
        self.events.append(e)
        if self.fault:
            token=self.leases.acquire(target);n=int((target!=value).sum())
            e.update(applied=True,changed=int(n>0),changed_elements=n,before_sha256=fingerprint(target))
            self.leases.write(token,value);self.pending.append((target,token,e))
        return e

    def consume(self,original,*values):
        # The supplied actual consumer gets these exact tensor objects. D1 must
        # establish this join at the live site; fixtures exercise it explicitly.
        self.tensors(*values)
        for region,_,_ in self.pending:
            require(any(v.untyped_storage()._cdata==region.untyped_storage()._cdata
                and v.is_contiguous() and v.storage_offset()<=region.storage_offset()
                and region.storage_offset()+region.numel()<=v.storage_offset()+v.numel()
                for v in values),'D2_CONSUMER_REGION_NOT_COVERED')
        result=original(*values)
        for region,token,e in self.pending:
            self.leases.consumed(token);self.leases.restore(token)
            e.update(consumed=True,restored=True,restored_sha256=fingerprint(region))
            require(e['before_sha256']==e['restored_sha256'],'D2_BYTES_NOT_RESTORED')
        self.pending.clear();return result

    def invoke(self,mechanism,operator,original,args,kwargs,context):
        """Boundary wrapper ready for D1-reviewed hook attachment (currently closed).

        Context is produced by a trusted actual-call join, never public config.
        Quantized weights must already be in verified logical layout; packed
        paths refuse. The consumer wrapper calls consume after the true read.
        """
        if mechanism=='22':
            require(operator=='_C_ascend::inplace_partial_rotary_mul' and len(args)==3,
                    'D2_ROTARY_CALL_SCHEMA')
            require(kwargs.get('rotary_mode')=='interleave','D2_ROTARY_MODE')
            return self.rotary_call(original,*args,positions=context['positions'],
                cos_table=context['cos_table'],sin_table=context['sin_table'],partial_slice=kwargs['partial_slice'])
        supported={'18':('npu::npu_rms_norm','_C_ascend::npu_rms_norm_cast'),
            '28':('_C_ascend::npu_sparse_attn_sharedkv','cann_ops_transformer::sparse_flash_mla'),
            '39op':('MatMulV2','QuantMatmulWeightNz'), '39mr':('MatMulV2','QuantMatmulWeightNz')}
        require(operator in supported.get(mechanism,()),'D2_ACTUAL_OPERATOR_UNSUPPORTED')
        result=original(*args,**kwargs)
        output=result[0] if isinstance(result,(tuple,list)) else result
        if mechanism=='18':
            self.norm_output(result if operator.endswith('rms_norm_cast') else output)
        elif mechanism=='28':
            self.sparse_output(output,args[0],kwargs,row=context['row'],request_ids=context['request_ids'],
                block_size=context['block_size'],replay_blocks=context['replay_blocks'])
        elif mechanism=='39op':self.projection_output(output,mechanism=mechanism)
        elif operator=='QuantMatmulWeightNz':
            self.quant_projection_output(output,context['q'],context['logical_weight'],context['weight_scale'],context['token_scale'],bias=context['bias'])
        else:
            self.projection_output(output,mechanism=mechanism,left=context['left'],right=context['right'],scale=context['scale'],bias=context['bias'])
        return result

    def norm_output(self,result):
        # npu_rms_norm_cast returns BF16 hidden AND independent FP32 route.
        pair=isinstance(result,(tuple,list))
        values=list(result) if pair else [result];self.tensors(*values)
        require((len(values)==2 and values[0].dtype==torch.bfloat16 and values[1].dtype==torch.float32
            and values[0].shape==values[1].shape) if pair else values[0].dtype==torch.bfloat16,
            'D2_RMS_CAST_PAIR_LAYOUT')
        for v in values:
            require(v.ndim==2 and v.is_contiguous(),'D2_NORM_TND_LAYOUT')
            self.watch(v[-1:,:1],v[-1:,:1]*.5,'18')
        return result

    def rotary_call(self,original,q,cos,sin,*,positions,cos_table,sin_table,partial_slice):
        self.tensors(q,cos,sin,positions,cos_table,sin_table)
        start,end=partial_slice
        require(q.ndim==4 and type(start)==type(end)==int and 0<=start<end==q.shape[-1]
            and (end-start)%2==0,'D2_PARTIAL_ROTARY_TAIL_REQUIRED')
        require(positions.dtype==torch.int64 and positions.ndim==1 and positions.numel()==q.shape[0]
            and cos_table.shape==sin_table.shape and cos_table.shape[0]>1,'D2_REAL_ROTARY_POSITIONS')
        wanted=[t.index_select(0,positions).reshape_as(x) for t,x in ((cos_table,cos),(sin_table,sin))]
        require(torch.equal(cos,wanted[0]) and torch.equal(sin,wanted[1]),'D2_ROTARY_LOOKUP_JOIN')
        shifted=(positions+1)%cos_table.shape[0]
        for t,x in ((cos_table,cos),(sin_table,sin)):
            self.watch(x,t.index_select(0,shifted).reshape_as(x),'22')
        # Original in-place operator writes q; restore only the temporary input
        # table rows after its device-consumption boundary, never restore q.
        return self.consume(lambda c,s:original(q,c,s,rotary_mode='interleave',partial_slice=[start,end]),cos,sin)

    def projection_output(self,out,*,mechanism,left=None,right=None,scale=1.,bias=0.):
        self.tensors(out)
        require(out.dtype==torch.bfloat16 and out.ndim==2 and out.is_contiguous(),'D2_BF16_ND_PROJECTION')
        cell=out[-1:,:1]
        if mechanism=='39op':value=torch.full_like(cell,torch.inf)
        else:
            require(mechanism=='39mr','D2_PROJECTION_MECHANISM')
            self.tensors(left,right)
            full,faulty,proof=recompute_cell(left,right,scale=scale,bias=bias)
            observed=float(cell.item());reference=float(full)
            require(abs(observed-reference)<=max(abs(observed),abs(reference))*2**-7+1e-6,
                    'D2_MATMUL_CONTROL_MISMATCH')
            value=faulty.to(out.dtype).reshape_as(cell)
        e=self.watch(cell,value,mechanism)
        if mechanism=='39mr':e.update(full_range_control_passed=True,range_reference=proof)
        return out

    def quant_projection_output(self,out,q,weight,weight_scale,token_scale,*,bias=None):
        # Only already-decoded logical KxN weights. NZ/W4 packed formats are
        # refused, rather than guessing a reshape or borrowing Qwen layout.
        self.tensors(out,q,weight,weight_scale,token_scale)
        require(q.dtype==weight.dtype==torch.int8 and q.ndim==weight.ndim==2
            and weight.shape==(q.shape[1],out.shape[1]) and q.shape[0]==out.shape[0]
            and token_scale.shape==(q.shape[0],) and weight_scale.numel()==out.shape[1]
            and bias is None,'D2_QMM_LOGICAL_LAYOUT_AND_SCALES_REQUIRED')
        return self.projection_output(out,mechanism='39mr',left=q[-1],right=weight[:,0],
            scale=float(token_scale[-1])*float(weight_scale.reshape(-1)[0]))

    def sparse_output(self,result,q,kwargs,*,row,request_ids,block_size=128,replay_blocks=(0,)):
        self.tensors(result,q)
        require(result.shape==q.shape and result.dtype==q.dtype and result.is_contiguous(),'D2_SPARSE_OUTPUT_LAYOUT')
        actual=from_operator_call(q,kwargs,row=row,request_ids=request_ids)
        normal=actual.recompute(block_size=block_size)
        require(torch.allclose(result[row],normal,rtol=2**-7 if result.dtype==torch.bfloat16 else 2e-5,atol=1e-6),
            'D2_SPARSE_CONTROL_MISMATCH')
        faulty=actual.recompute(block_size=block_size,replay_blocks=replay_blocks)
        e=self.watch(result[row:row+1],faulty.unsqueeze(0),'28')
        e.update(request_id=actual.request_id,keys=actual.selected,sink_replayed=False)
        return result

    def close(self):
        require(not self.pending and self.leases.count()==0,'D2_OUTSTANDING_CONSUMER_OR_LEASE')


class D2FixtureCalls(D2CallAdapter):
    def __init__(self,owner,*,fault=True):
        self.completions=[]
        super().__init__(RegionLeases(owner,1,stream=lambda:'cpu',complete=self.completions.append),fault=fault)
    def tensors(self,*values):
        super().tensors(*values)
        require(all(v.device.type=='cpu' for v in values),'D2_CPU_FIXTURE_ONLY')
