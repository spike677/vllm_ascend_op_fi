"""CPU dispatcher fixture for the new-stack ordinary fused22 window; no NPU claim."""
import unittest
from types import SimpleNamespace as N
from unittest.mock import patch
import torch
from test_fused_mrope22_scope import Attention,OP,implementation
from demoncat.runtime.vllm_ascend.worker_extension.new_stack_rope22 import _OneFusedMrope,_Fused22Mixin

class WindowTests(unittest.TestCase):
    def fixture(self):
        self.enterContext(patch('demoncat.runtime.vllm_ascend.worker_extension.fused_mrope22_scope._native',return_value=(OP,implementation)))
        self.enterContext(patch('demoncat.runtime.vllm_ascend.worker_extension.fused_mrope22_scope._verify_function',return_value={'cpu_fixture':True}))
        model=torch.nn.Module();model.attention=Attention()
        # Real runner passes a non-contiguous prefix of a larger 3-axis buffer.
        pos=torch.arange(8).repeat(3,1)[:,:3]
        x=torch.arange(3*1536,dtype=torch.float16).reshape(3,1536)/8192
        p=N(model=model,target=None,target_module=None,normal=None,injection_arm='baseline',
            binding=N(prompt=[1,2,3],execution_profile='v026_installed_c9c9432b',simulation=True),
            session=N(root_arguments={'positions':pos}),phase='normal',fault_started=False,fault_applied=False,_context=lambda:None)
        p.admit=lambda e,a:_Fused22Mixin.admit(p,e,a)
        return model,pos,x,p
    def baseline(self,p,m,pos,x):
        w=_OneFusedMrope(p);p.session.window=w;w.__enter__()
        try:
            y=m.attention(pos,x);r=w.report()
        finally:w.close()
        e=r['events'][0];p.normal={'mechanism':r};p.target_module=m.attention
        p.target=dict(path=e['path'],ordinal=0,actual=e['actual'],operator=e['operator'],native_schema=e['schema'])
        return y,r
    def test_actual_first_candidate_baseline_fault_and_unwrapped_off(self):
        m,pos,x,p=self.fixture();b,br=self.baseline(p,m,pos,x)
        self.assertEqual(br['events'][0]['path'],'attention');self.assertFalse(br['events'][0]['applied'])
        p.phase='fault';p.injection_arm='fault';p.fault_started=True
        w=_OneFusedMrope(p);p.session.window=w;w.__enter__()
        try:f=m.attention(pos,x);r=w.report()
        finally:w.close()
        e=r['events'][0];self.assertTrue(p.fault_applied);self.assertTrue(e['output_changed'])
        self.assertTrue(e['effect']['non_target_qk_unchanged']);self.assertFalse(torch.equal(b,f))
        self.assertTrue(torch.equal(b,m.attention(pos,x)));self.assertEqual(w.owned_hook_count,0)
        self.assertNotIn('forward',vars(m.attention));self.assertNotIn('forward',vars(m.attention.attn))
    def test_guard_rank_remains_read_only(self):
        m,pos,x,p=self.fixture();b,_=self.baseline(p,m,pos,x)
        p.phase='fault';p.injection_arm='baseline';p.fault_started=True
        w=_OneFusedMrope(p);p.session.window=w;w.__enter__()
        try:f=m.attention(pos,x);r=w.report()
        finally:w.close()
        self.assertTrue(torch.equal(b,f));self.assertFalse(r['events'][0]['output_changed']);self.assertFalse(p.fault_applied)
    def test_changed_projection_refused_before_fault_admission(self):
        m,pos,x,p=self.fixture();self.baseline(p,m,pos,x)
        p.phase='fault';p.injection_arm='fault';p.fault_started=True
        w=_OneFusedMrope(p);p.session.window=w;w.__enter__()
        try:
            with self.assertRaisesRegex(RuntimeError,'CURRENT_CALL_OR_SOURCE_CHANGED'):m.attention(pos,x+1)
        finally:w.close()
        self.assertFalse(p.fault_applied);self.assertEqual(w.owned_hook_count,0)
    def test_replaced_target_or_missing_structure_refused(self):
        m,pos,x,p=self.fixture();self.baseline(p,m,pos,x);m.attention=Attention()
        with self.assertRaisesRegex(RuntimeError,'HELD_TARGET'):_OneFusedMrope(p).__enter__()
        del m.attention
        with self.assertRaisesRegex(RuntimeError,'NO_REVIEWED'):_OneFusedMrope(p).__enter__()

if __name__=='__main__':unittest.main()
