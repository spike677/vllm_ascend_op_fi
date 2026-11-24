import types
import unittest
import torch
from fault_injection.engines.region_lease import RegionLeases
from reference_adapters.vision_precision import VisionPrecisionSite, ConvInputFp16RoundTrip
from reference_adapters.swiglu import GateFp16Round

LIB=torch.library.Library('dcat_ref_act','DEF')
LIB.define('swiglu(Tensor x) -> Tensor')
def swiglu(x):
    gate,up=x.float().chunk(2,-1)
    return (torch.nn.functional.silu(gate)*up).to(x.dtype)
LIB.impl('swiglu',swiglu,'CPU')

class Act(torch.nn.Module):
    def forward(self,x):return torch.ops.dcat_ref_act.swiglu(x)

class Mlp(torch.nn.Module):
    def __init__(self):
        super().__init__();self.gate_up_proj=torch.nn.Identity();self.act_fn=Act();self.down_proj=torch.nn.Identity()
    def forward(self,x):return self.act_fn(x)+1

class Visual(torch.nn.Module):
    def __init__(self):
        super().__init__();self.patch_embed=torch.nn.Module()
        self.patch_embed.proj=torch.nn.Conv3d(1,2,1,bias=False).to(torch.bfloat16)
        self.patch_embed.proj.weight.data.fill_(.3)
    def forward(self,x):return self.patch_embed.proj(x).flatten(1)+1

class PrecisionTests(unittest.TestCase):
    def leases(self,n):return RegionLeases('owner',n,stream=lambda:0,complete=lambda s:None)
    def runner(self):return types.SimpleNamespace(input_batch=types.SimpleNamespace(req_ids=['r']))
    def test_real_conv_readonly_and_restore(self):
        for cls in (VisionPrecisionSite,ConvInputFp16RoundTrip):
            model=torch.nn.Module();model.visual=Visual()
            x=torch.arange(1,9,dtype=torch.bfloat16).reshape(1,1,2,2,2)
            baseline=model.visual(x)
            for generation in (1,2):
                leases=self.leases(generation);site=cls(model,self.runner(),fault=True,region_leases=leases)
                site.install();model.visual(x);site.finish_forward();site.close()
                e=site.events[0]
                self.assertTrue(e['applied'] and e['consumed'] and e['restored'])
                self.assertTrue(e['readonly_inputs_weights_unchanged'])
                self.assertEqual(e['before_sha256'],e['restored_sha256'])
                self.assertTrue(torch.equal(model.visual(x),baseline))
                self.assertEqual(leases.count(),0)
                if cls is ConvInputFp16RoundTrip:
                    self.assertEqual(e['changed_elements'],0) # BF16 values are exactly FP16-representable here.
                    self.assertTrue(e['encoder_consumer_operator'])
    def test_actual_swiglu_then_downstream_consumer(self):
        model=Mlp();x=torch.linspace(-5,5,256,dtype=torch.bfloat16).reshape(2,128)
        baseline=model(x)
        for generation in (1,2):
            site=GateFp16Round(model,self.runner(),fault=True,region_leases=self.leases(generation))
            site.install();model(x);site.finish_forward();site.close()
            e=site.events[0]
            self.assertTrue(e['consumed'] and e['restored'])
            self.assertEqual(e['native_operator'],'dcat_ref_act::swiglu')
            self.assertTrue(torch.equal(model(x),baseline))
