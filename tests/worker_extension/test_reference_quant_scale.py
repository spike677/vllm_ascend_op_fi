import types
import unittest
import numpy as np
import torch
from reference_adapters.quant_scale import requantize, SaturationHalf, SqueezeDouble
from fault_injection.engines.region_lease import RegionLeases

LIB=torch.library.Library('dcat_ref_quant','DEF')
LIB.define('dynamic(Tensor x) -> (Tensor, Tensor)')
LIB.define('qmm(Tensor x, Tensor weight, Tensor pertoken_scale) -> Tensor')
CAPTURE=[]
def dynamic(x):
    s=x.float().abs().amax(-1)/127
    q=torch.round(x.float()/s[:,None]).clamp(-127,127).to(torch.int8)
    CAPTURE.append((q,s,q.clone(),s.clone()))
    return q,s
LIB.impl('dynamic',dynamic,'CPU')
LIB.impl('qmm',lambda x,weight,pertoken_scale:(x.float()*pertoken_scale[:,None])@weight.float(),'CPU')

class AscendW8A8DynamicLinearMethod:
    def apply(self,layer,x):
        q,s=torch.ops.dcat_ref_quant.dynamic(x)
        return torch.ops.dcat_ref_quant.qmm(q,layer.weight,s)

class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.eye(3,dtype=torch.bfloat16)
        self.quant_method=types.SimpleNamespace(quant_method=AscendW8A8DynamicLinearMethod())
    def forward(self,x):return self.quant_method.quant_method.apply(self,x)

class QuantTests(unittest.TestCase):
    def test_fixed_scales_and_independent_rounding(self):
        x=torch.tensor([[.1,-1.5,4.],[0.,0.,0.]],dtype=torch.bfloat16);s=torch.tensor([.03125,0.])
        for factor in (.25,.5,2.,4.):
            q,scale,stats=requantize(x,s,factor)
            a=x.float().numpy();b=s.numpy()*factor
            expected=np.clip(np.rint(a/np.where(b==0,1,b)[:,None]),-127,127).astype(np.int8)
            np.testing.assert_array_equal(q.numpy(),expected)
            np.testing.assert_array_equal(scale.numpy(),b)
        with self.assertRaises(RuntimeError):requantize(x,s,.3)

    def test_two_scopes_real_qmm_read_and_byte_restore(self):
        model=torch.nn.Sequential(Linear());x=torch.tensor([[.2,32,127]],dtype=torch.bfloat16)
        baseline=model(x)
        for generation,cls in enumerate((SaturationHalf,SqueezeDouble),1):
            leases=RegionLeases('owner',generation,stream=lambda:0,complete=lambda s:None)
            site=cls(model,types.SimpleNamespace(input_batch=types.SimpleNamespace(req_ids=['q'])),fault=True,region_leases=leases)
            site.mode.quant_name='dcat_ref_quant::dynamic';site.mode.qmm_name='dcat_ref_quant::qmm'
            site.install();fault=model(x);site.finish_forward();site.close()
            self.assertFalse(torch.equal(fault,baseline))
            self.assertTrue(torch.equal(model(x),baseline))
            self.assertEqual(leases.count(),0)
            self.assertEqual(len(site.events),1)
            e=site.events[0]
            self.assertTrue(e['applied'] and e['consumed'] and e['restored'])
            self.assertEqual(e['before_sha256'],e['restored_sha256'])
            self.assertEqual(e['after_sha256'],e['consumer_sha256'])
        for q,s,bq,bs in CAPTURE:
            self.assertTrue(torch.equal(q,bq) and torch.equal(s,bs))
