"""CPU boundary association tests; the stand-in is not an NPU fused kernel."""
from types import SimpleNamespace as N
import unittest
import torch
from fault_injection.fused_mrope22_boundary import FusedMrope22Boundary as B
from fault_injection.core import UnsupportedCall

class FusedBoundaryTests(unittest.TestCase):
    def fixture(self,arm='baseline',gate=True):
        cache=torch.arange(8*64,dtype=torch.float16).reshape(8,64)/512
        pos=torch.tensor([[0,1,2]]*3);m=N(num_heads=2,num_kv_heads=1,head_dim=256,attn_output_gate=gate,
            config=N(rms_norm_eps=1e-6),q_norm=N(weight=torch.zeros(256,dtype=torch.float16)),
            k_norm=N(weight=torch.zeros(256,dtype=torch.float16)),
            rotary_emb=N(cos_sin_cache=cache,rotary_dim=64,mrope_section=[11,11,10],mrope_interleaved=True))
        qh=512;kh=256;width=qh*(2 if gate else 1)+2*kh
        qkv=torch.arange(3*width,dtype=torch.float16).reshape(3,width)/8192
        b=B(m,pos,pos.tolist(),arm=arm,admit=lambda:True,simulation=True)
        c=b.index(torch.ops.aten.index.Tensor,cache,pos)
        kw=dict(qkv=qkv,q_weight=1+m.q_norm.weight,k_weight=1+m.k_norm.weight,cos_sin=c,
            num_q_heads=2,num_kv_heads=1,head_size=256,eps=1e-6,mrope_section=[11,11,10],is_interleaved=True,rope_dim=64,has_gate=gate)
        def original(**p):
            # Deliberately simple CPU stand-in, tests association/ownership only.
            x=p['qkv'];start=qh*(2 if gate else 1)
            q=x[:,:qh].clone()+p['cos_sin'][0].sum(-1,keepdim=True)
            k=x[:,start:start+kh].clone()+p['cos_sin'][1].sum(-1,keepdim=True)
            v=x[:,start+kh:start+2*kh].clone()
            g=x[:,:2*qh].reshape(3,2,512)[:,:,256:].reshape(3,qh).clone() if gate else x.new_empty((3,0))
            return q,k,v,g
        return b,qkv,kw,original
    def finish(self,arm,gate=True):
        b,x,k,f=self.fixture(arm,gate);before=x.clone();q,v,w,g=b.fused(f,x,**k)
        b.consume(lambda *a:a[0],q,v,w);r=b.finish();self.assertTrue(torch.equal(x,before));return r
    def test_b_f_off_real_index_effect_and_no_qualification(self):
        b,f,o=[self.finish(a) for a in ('baseline','fault','off')]
        self.assertEqual(b['fused_q_sha256'],o['fused_q_sha256']);self.assertNotEqual(b['fused_q_sha256'],f['fused_q_sha256'])
        self.assertNotEqual(b['fused_k_sha256'],f['fused_k_sha256']);self.assertEqual(b['qkv_sha256'],f['qkv_sha256'])
        self.assertTrue(f['provided_boundary_chain_complete']);self.assertFalse(f['qualification_granted']);self.assertFalse(f['fused_qk_effect_verified'])
    def test_no_gate_supported(self):self.assertTrue(self.finish('fault',False)['value_and_gate_preserved'])
    def test_wrong_index_consumer(self):
        b,x,k,f=self.fixture('fault');k['cos_sin']=torch.zeros_like(k['cos_sin'])
        with self.assertRaisesRegex(UnsupportedCall,'INDEX_RESULT_NOT_CONSUMED'):b.fused(f,x,**k)
    def test_changed_norm_weights_refused(self):
        b,x,k,f=self.fixture();k['q_weight']=k['q_weight']+1
        with self.assertRaisesRegex(UnsupportedCall,'WEIGHTS_CHANGED'):b.fused(f,x,**k)
    def test_actual_projection_identity(self):
        b,x,k,f=self.fixture()
        with self.assertRaisesRegex(UnsupportedCall,'ACTUAL_PROJECTION'):b.fused(f,x.clone(),**k)
    def test_wrong_attention_objects(self):
        b,x,k,f=self.fixture();q,h,v,g=b.fused(f,x,**k)
        with self.assertRaisesRegex(UnsupportedCall,'INPUT_ASSOCIATION'):b.consume(lambda *a:None,q.clone(),h,v)
    def test_missing_consumer_and_second_native_call_refused(self):
        b,x,k,f=self.fixture();b.fused(f,x,**k)
        with self.assertRaisesRegex(UnsupportedCall,'INCOMPLETE'):b.finish()
        with self.assertRaisesRegex(UnsupportedCall,'ORDER'):b.fused(f,x,**k)
    def test_corrupt_value_output_refused(self):
        b,x,k,f=self.fixture()
        def bad(**kw):
            q,h,v,g=f(**kw);return q,h,v+1,g
        with self.assertRaisesRegex(UnsupportedCall,'V_OR_QKV_CHANGED'):b.fused(bad,x,**k)
if __name__=='__main__':unittest.main()
