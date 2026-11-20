"""Private fused-mRoPE call association for a trusted source-bound adapter.
No registration, rank selection, global patch or deployment permission here.
The adapter obtains the module from structural discovery and invokes these
methods only at its real index / fused API / Attention consumer boundaries.
"""
import hashlib
from .core import UnsupportedCall
from .mrope_index22 import call_mrope_index22


def need(ok,message):
    if not ok:raise UnsupportedCall(message)


def tensor_sha(tensor):
    import torch
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class FusedMrope22Boundary:
    def __init__(self,module,positions,expected_positions,*,arm,admit,simulation=False):
        self.module=module;self.positions=positions;self.expected=expected_positions
        self.arm=arm;self.admit=admit;self.simulation=simulation
        self.indexed=None;self.outputs=None;self.record={};self.stage='created'
    def index(self,original,cache,positions):
        need(self.stage=='created' and cache is self.module.rotary_emb.cos_sin_cache
             and positions is self.positions,'FUSED22_ACTUAL_CACHE_POSITION_OR_ORDER')
        self.indexed,self.record=call_mrope_index22(original,cache,positions,
            request_positions=self.positions,expected_positions=self.expected,arm=self.arm,
            admit=self.admit,simulation=self.simulation)
        self.stage='indexed';return self.indexed
    def fused(self,original,actual_qkv,**kw):
        import torch
        m=self.module;rope=m.rotary_emb
        need(self.stage=='indexed' and kw.get('qkv') is actual_qkv,'FUSED22_ACTUAL_PROJECTION_OR_ORDER')
        expected=dict(num_q_heads=m.num_heads,num_kv_heads=m.num_kv_heads,head_size=m.head_dim,
            eps=m.config.rms_norm_eps,mrope_section=rope.mrope_section,is_interleaved=rope.mrope_interleaved,
            rope_dim=rope.rotary_dim,has_gate=m.attn_output_gate)
        need(set(kw)==set(expected)|{'qkv','q_weight','k_weight','cos_sin'}
             and all(kw[k]==v for k,v in expected.items()),'FUSED22_ACTUAL_FUSED_GEOMETRY')
        need(rope.rotary_dim*4==m.head_dim and rope.mrope_interleaved is True
             and list(rope.mrope_section)==[11,11,10],'FUSED22_REVIEWED_PARTIAL_MROPE_REQUIRED')
        need(torch.equal(kw['q_weight'].detach(),1.0+m.q_norm.weight.detach())
             and torch.equal(kw['k_weight'].detach(),1.0+m.k_norm.weight.detach()),'FUSED22_GEMMA_WEIGHTS_CHANGED')
        c=kw['cos_sin'];qkv=actual_qkv;n=self.positions.shape[1]
        need(c.device==qkv.device and c.dtype==qkv.dtype and qkv.ndim==2 and qkv.shape[0]==n,
             'FUSED22_CONSUMER_DEVICE_DTYPE_OR_TOKENS')
        # The reviewed source allows device/dtype conversion between index and
        # fused API. Require full converted values, not an assumed object alias.
        expected_cache=self.indexed.detach().to(device=c.device,dtype=c.dtype)
        need(c.shape==expected_cache.shape and torch.equal(c,expected_cache),'FUSED22_INDEX_RESULT_NOT_CONSUMED')
        before=tensor_sha(qkv);outputs=original(**kw)
        need(isinstance(outputs,(tuple,list)) and len(outputs)==4,'FUSED22_FOUR_NATIVE_OUTPUTS_REQUIRED')
        q,k,v,gate=outputs;qh=m.num_heads*m.head_dim;kh=m.num_kv_heads*m.head_dim
        need(qkv.shape[1]==qh*(2 if m.attn_output_gate else 1)+2*kh,'FUSED22_PACKED_QKV_WIDTH')
        for value,width in ((q,qh),(k,kh),(v,kh),(gate,qh if m.attn_output_gate else 0)):
            need(tuple(value.shape)==(n,width) and value.dtype==qkv.dtype and value.device==qkv.device
                 and bool(torch.isfinite(value).all()),'FUSED22_ACTUAL_OUTPUT_GEOMETRY_OR_FINITE')
        start=qh*(2 if m.attn_output_gate else 1)
        need(torch.equal(v,qkv[:,start+kh:start+2*kh]) and tensor_sha(qkv)==before,'FUSED22_V_OR_QKV_CHANGED')
        if m.attn_output_gate:
            need(torch.equal(gate,qkv[:,:2*qh].reshape(n,m.num_heads,2*m.head_dim)[:,:,m.head_dim:].reshape(n,qh)),
                 'FUSED22_GATE_CHANGED')
        self.outputs=outputs;self.stage='fused'
        self.record.update(fused_original_calls=1,fused_q_sha256=tensor_sha(q),fused_k_sha256=tensor_sha(k),
            fused_q_non_target_sha256=tensor_sha(q[:-1]),fused_k_non_target_sha256=tensor_sha(k[:-1]),
            fused_q_last_sha256=tensor_sha(q[-1:]),fused_k_last_sha256=tensor_sha(k[-1:]),
            qkv_sha256=before,index_result_consumed=True,qkv_unchanged=True,value_and_gate_preserved=True,
            fused_qk_effect_verified=False,qualification_granted=False)
        return outputs
    def consume(self,original,q,k,v):
        need(self.stage=='fused' and all(a is b for a,b in zip((q,k,v),self.outputs[:3])),
             'FUSED22_ACTUAL_ATTENTION_INPUT_ASSOCIATION')
        result=original(q,k,v);self.stage='consumed';self.record['attention_consumer_calls']=1;return result
    def finish(self):
        need(self.stage=='consumed','FUSED22_INCOMPLETE_NATIVE_CONSUMER_CHAIN')
        self.stage='finished';self.outputs=None;self.indexed=None
        return dict(self.record,provided_boundary_chain_complete=True,source_bound_observation_verified=False,qualification_granted=False,
                    effect_comparison='Requires separate same-input B/F Q/K hashes; no inferred change from hit')
