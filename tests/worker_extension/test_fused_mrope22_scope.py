"""CPU real dispatcher + index + module hooks, not a fused NPU kernel."""
import sys,unittest
from types import SimpleNamespace as N
from unittest.mock import patch
import torch
from demoncat.runtime.vllm_ascend.worker_extension.fused_mrope22_scope import FusedMrope22CallScope

LIB=torch.library.Library('dcat_fixture_fused22','DEF')
LIB.define('compute(Tensor qkv, Tensor q_weight, Tensor k_weight, Tensor cos_sin, int num_q_heads, int num_kv_heads, int head_size, float eps, int[] mrope_section, bool is_interleaved, int rope_dim, bool has_gate) -> (Tensor, Tensor, Tensor, Tensor)')
def implementation(qkv,q_weight,k_weight,cos_sin,num_q_heads,num_kv_heads,head_size,eps,mrope_section,is_interleaved,rope_dim,has_gate):
    qh=num_q_heads*head_size;kh=num_kv_heads*head_size;start=qh*2
    return (qkv[:,:qh].clone()+cos_sin[0].sum(-1,keepdim=True),
        qkv[:,start:start+kh].clone()+cos_sin[1].sum(-1,keepdim=True),
        qkv[:,start+kh:start+2*kh].clone(),
        qkv[:,:2*qh].reshape(-1,num_q_heads,2*head_size)[:,:,head_size:].reshape(-1,qh).clone())
LIB.impl('compute',implementation,'CPU')
OP=torch.ops.dcat_fixture_fused22.compute.default
class Projection(torch.nn.Module):
    def forward(self,x):return x,None
class Consumer(torch.nn.Module):
    def forward(self,q,k,v):return q+k.repeat(1,2)+v.repeat(1,2)
class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__();self.qkv_proj=Projection();self.attn=Consumer()
        self.config=N(model_type='qwen3_5_text',rms_norm_eps=1e-6)
        self.num_heads=2;self.num_kv_heads=1;self.head_dim=256;self.attn_output_gate=True
        self.q_norm=N(weight=torch.zeros(256,dtype=torch.float16));self.k_norm=N(weight=torch.zeros(256,dtype=torch.float16))
        self.rotary_emb=N(cos_sin_cache=torch.arange(8*64,dtype=torch.float16).reshape(8,64)/512,
            rotary_dim=64,mrope_interleaved=True,mrope_section=[11,11,10])
        self.call_extra=False
    def forward(self,positions,hidden_states,output=None):
        qkv,_=self.qkv_proj(hidden_states)
        c=self.rotary_emb.cos_sin_cache[positions]
        q,k,v,gate=OP(qkv=qkv,q_weight=1+self.q_norm.weight,k_weight=1+self.k_norm.weight,cos_sin=c,
            num_q_heads=2,num_kv_heads=1,head_size=256,eps=1e-6,mrope_section=[11,11,10],is_interleaved=True,rope_dim=64,has_gate=True)
        if self.call_extra:self.qkv_proj(hidden_states)
        return self.attn(q,k,v)
Attention.__name__='AscendQwen3NextAttention';Attention.__module__='vllm_ascend.patch.worker.patch_qwen3_5'
class ScopeTests(unittest.TestCase):
    def scope(self,arm='fault',admit=lambda:True):
        m=Attention();positions=torch.tensor([[0,1,2]]*3);x=torch.arange(3*1536,dtype=torch.float16).reshape(3,1536)/8192
        self.enterContext(patch('demoncat.runtime.vllm_ascend.worker_extension.fused_mrope22_scope._native',return_value=(OP,implementation)))
        self.enterContext(patch('demoncat.runtime.vllm_ascend.worker_extension.fused_mrope22_scope._verify_function',return_value={'CPU_FIXTURE_NOT_SOURCE_REVIEW':True}))
        s=FusedMrope22CallScope(m,positions,positions.tolist(),arm=arm,admit=admit,simulation=True).install()
        self.addCleanup(s.close);return s,m,positions,x
    def test_actual_dispatch_chain_b_f_off_and_owned_cleanup(self):
        outputs=[]
        for arm in ('baseline','fault','off'):
            s,m,p,x=self.scope(arm);outputs.append(m(p,x));r=s.snapshot()['result']
            self.assertTrue(r['registered_implementation_observed']);self.assertTrue(r['actual_projection_observed'])
            self.assertFalse(r['qualification_granted']);self.assertFalse(r['ordinary_runtime_integrated'])
            s.close();self.assertEqual(s.snapshot()['owned_wrappers'],0);self.assertEqual(len(m.qkv_proj._forward_hooks),0)
        self.assertTrue(torch.equal(outputs[0],outputs[2]));self.assertFalse(torch.equal(outputs[0],outputs[1]));self.assertIsNone(sys.getprofile())
    def test_changed_request_object_before_any_dispatch(self):
        s,m,p,x=self.scope()
        with self.assertRaisesRegex(RuntimeError,'REQUEST_POSITIONS'):m(p.clone(),x)
        self.assertFalse(s.used)
    def test_one_call_only_and_off_restored_without_wrapper(self):
        s,m,p,x=self.scope('baseline');expected=m(p,x)
        with self.assertRaisesRegex(RuntimeError,'SINGLE_ACTUAL_CALL'):m(p,x)
        s.close();self.assertTrue(torch.equal(expected,m(p,x)))
    def test_live_owner_refusal_preserves_positions(self):
        s,m,p,x=self.scope(admit=lambda:False);before=p.clone()
        with self.assertRaisesRegex(RuntimeError,'ADMISSION'):m(p,x)
        self.assertTrue(torch.equal(p,before));s.close();self.assertIsNone(sys.getprofile())
    def test_foreign_wrapper_not_removed(self):
        s,m,p,x=self.scope();own=m.forward;m.forward=lambda *a:None
        with self.assertRaisesRegex(RuntimeError,'FOREIGN_WRAPPER'):s.close()
        m.forward=own;s.close()
    def test_existing_profiler_is_not_overwritten(self):
        s,m,p,x=self.scope();profiler=lambda *args:None
        sys.setprofile(profiler)
        try:
            with self.assertRaisesRegex(RuntimeError,'EXISTING_PROFILER'):m(p,x)
            self.assertIs(sys.getprofile(),profiler)
        finally:sys.setprofile(None)
        s.close()
    def test_registered_implementation_must_actually_execute(self):
        s,m,p,x=self.scope()
        # Installation retains its source-reviewed function; modifying this
        # descriptive attribute cannot change that actual identity.
        original=s.op
        def foreign(*args):return None
        s.close()
        with patch('demoncat.runtime.vllm_ascend.worker_extension.fused_mrope22_scope._native',return_value=(original,foreign)):
            other=FusedMrope22CallScope(m,p,p.tolist(),arm='baseline',admit=lambda:True,simulation=True).install()
        try:
            with self.assertRaisesRegex(RuntimeError,'IMPLEMENTATION_NOT_OBSERVED'):m(p,x)
        finally:other.close()
        self.assertIsNone(sys.getprofile())
    def test_extra_projection_rejected(self):
        s,m,p,x=self.scope();m.call_extra=True
        with self.assertRaisesRegex(RuntimeError,'PROJECTION_CALL_ORDER'):m(p,x)
        s.close();self.assertIsNone(sys.getprofile())
if __name__=='__main__':unittest.main()
