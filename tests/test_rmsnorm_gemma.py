"""CPU Gemma API lineage, not CANN or real Qwen acceptance."""
import unittest,torch
from fault_injection.rmsnorm_return import RMSNormReturnSession,GEMMA_SCHEMA,native_rms_schema
LIB=torch.library.Library('_C_ascend','FRAGMENT')
LIB.define(GEMMA_SCHEMA.split('::',1)[1])
def impl(x,gamma,epsilon=1e-6):
    rstd=torch.rsqrt(x.float().square().mean(-1,keepdim=True)+epsilon)
    return (x.float()*rstd*(1+gamma.float())).to(x.dtype),rstd
LIB.impl('npu_gemma_rms_norm',impl,'CPU')
class Gemma(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.linspace(-.5,.5,8));self.last_rstd=None
    def forward(self,x):
        y,self.last_rstd=torch.ops._C_ascend.npu_gemma_rms_norm(x,self.weight,1e-6);return y
class Model(torch.nn.Module):
    def __init__(self):super().__init__();self.norm=Gemma()
    def forward(self,x):return self.norm(x)*3
class GemmaReturnTests(unittest.TestCase):
    def test_native_math_retained_output_half_and_off_exact(self):
        model=Model().eval();x=torch.arange(24.).reshape(3,8)
        with torch.inference_mode():
            expected=model(x);normal_aux=model.norm.last_rstd.clone()
            with RMSNormReturnSession(model,arm='fault',simulation=True) as session:actual=model(x)
            self.assertTrue(torch.equal(actual,expected*.5));self.assertTrue(torch.equal(normal_aux,model.norm.last_rstd))
            self.assertTrue(torch.equal(model(x),expected));self.assertEqual(len(session.events),1)
            self.assertEqual(session.events[0]['native_operator'],'_C_ascend.npu_gemma_rms_norm.default')
            self.assertTrue(session.events[0]['consumers']);self.assertFalse(session.events[0]['native_reduce_error_claimed'])
    def test_gemma_schema_change_refused(self):
        schema=torch._C.parse_schema(GEMMA_SCHEMA.replace('epsilon=9.9999999999999995e-07','epsilon=0.01'))
        self.assertFalse(native_rms_schema(schema))
if __name__=='__main__':unittest.main()
