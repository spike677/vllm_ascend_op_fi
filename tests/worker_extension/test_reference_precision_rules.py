import math
import types
import unittest
import torch
from reference_adapters.precision_rules import fp8_like,truncate_bf16
from reference_adapters.control_guard import preflight,ReferenceRefused,snapshot
from reference_adapters.projection_round import ProjectionMantissa5,ProjectionMantissa3
from fault_injection.engines.region_lease import RegionLeases


class PrecisionRulesTests(unittest.TestCase):
    def test_fp8_independent_scalar_reference(self):
        # All finite BF16 bit patterns, not merely a few convenient inputs.
        x=torch.arange(-32768,32768,dtype=torch.int32).short().view(torch.bfloat16)
        for variant,fraction,minimum,maximum in [('e4m3',3,-6,448.),('e5m2',2,-14,57344.)]:
            result=fp8_like(x,variant).float().tolist()
            for value,actual in zip(x.float().tolist(),result):
                if not math.isfinite(value):
                    self.assertTrue(math.isnan(actual) if math.isnan(value) else value==actual)
                    continue
                magnitude=min(abs(value),maximum)
                exponent=math.floor(math.log2(max(magnitude,2.**minimum)))
                q=2.**(exponent-fraction)
                expected=math.copysign(min(round(magnitude/q)*q,maximum),value)
                self.assertEqual(actual,expected,(variant,value))
    def test_rounding_ties_underflow_saturation_and_sign(self):
        x=torch.tensor([1.0625,1.1875,2.**-10,3*2.**-10,500.,-500.,-0.])
        y=fp8_like(x,'e4m3')
        self.assertEqual(y.tolist(),[1.,1.25,0.,2.**-8,448.,-448.,-0.])
        self.assertTrue(torch.signbit(y[-1]).item())
    def test_mantissa_bits_all_bf16_patterns(self):
        bits=torch.arange(-32768,32768,dtype=torch.int32).short()
        x=bits.view(torch.bfloat16)
        for n in (3,5):
            y=truncate_bf16(x,n).view(torch.int16)
            finite=torch.isfinite(x)
            expected=bits.int()//(1<<(7-n))*(1<<(7-n))
            self.assertTrue(torch.equal(y[finite].int(),expected[finite]))
            self.assertTrue(torch.equal(y[~finite],bits[~finite]))
    def test_invalid_precision_refuses(self):
        for x,n in [(torch.ones(2),3),(torch.ones(2,dtype=torch.bfloat16),4)]:
            with self.assertRaises(ValueError):truncate_bf16(x,n)
    def test_projection_preflight_missing_is_control_refusal(self):
        model=torch.nn.Module();before=snapshot(model)
        for factory in (ProjectionMantissa3,ProjectionMantissa5):
            with self.assertRaises(ReferenceRefused):preflight(factory,model,None)
        self.assertEqual(before,snapshot(model))


class Down(torch.nn.Module):
    def forward(self,x):return x.clone(),None
class Mlp(torch.nn.Module):
    def __init__(self):
        super().__init__();self.gate_up_proj=torch.nn.Identity()
        self.down_proj=Down();self.act_fn=torch.nn.Identity()
    def forward(self,x):return self.down_proj(x)[0]


class ProjectionBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib=torch.library.Library('_C_ascend','FRAGMENT')
        cls.lib.define('npu_add_rms_norm_bias(Tensor x) -> Tensor')
        cls.lib.impl('npu_add_rms_norm_bias',lambda x:x*2,'CPU')
    def test_real_projection_identity_consumer_and_byte_restore(self):
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.mlp=Mlp()
            def forward(self,x):
                self.output=self.mlp(x)
                return torch.ops._C_ascend.npu_add_rms_norm_bias(self.output)
        for factory in (ProjectionMantissa3,ProjectionMantissa5):
            model=Model();runner=types.SimpleNamespace(input_batch=types.SimpleNamespace(req_ids=['test']))
            leases=RegionLeases('owner',1,stream=lambda:0,complete=lambda s:None)
            site=factory(model,runner,fault=True,region_leases=leases);site.install()
            x=torch.tensor([[1.0078125,1.0390625,-2.0625]],dtype=torch.bfloat16)
            expected=truncate_bf16(x,site.fraction_bits)*2
            y=model(x);site.finish_forward();site.close()
            self.assertTrue(torch.equal(y,expected));self.assertTrue(torch.equal(model.output,x))
            self.assertEqual(leases.count(),0)
            e=site.events[0];self.assertGreater(e['changed_elements'],0)
            self.assertTrue(e['consumed'] and e['restored'])
            self.assertEqual(e['consumer_operator'],'_C_ascend::npu_add_rms_norm_bias')
