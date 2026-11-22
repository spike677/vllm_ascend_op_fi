import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from fault_injection.sites.native_fia26 import ResidualRead
from fault_injection.sites.mla28 import fingerprint
import test_native_regions


class Op:
    def __init__(self,fn,mutable=False):
        self.fn=fn;self._schema=NS(name='npu::npu_add_rms_norm',arguments=[
            NS(name=n,default_value=None,alias_info=NS(is_write=True) if mutable and n=='residual' else None)
            for n in ('x','residual')])
    def __call__(self,*args,**kw):return self.fn(*args,**kw)


class ResidualTests(unittest.TestCase):
    def test_actual_residual_copy_consumer_fresh_output_and_original_recovery(self):
        native=test_native_regions.NativeRegionsTests().manager();source=torch.full((64,),3,dtype=torch.bfloat16)
        residual=torch.ones(2,128,dtype=torch.bfloat16);original=residual.clone();x=residual+1
        site=NS(native=native,fault=True);e=dict(source=source,source_sha256=fingerprint(source))
        class Event:
            def record(self,s):pass
            def synchronize(self):pass
        with patch.object(torch,'npu',NS(Event=Event),create=True):
            mode=ResidualRead(site,e,residual)
            out=mode.__torch_dispatch__(Op(lambda x,residual:(x+residual,None,x+residual)),(),(x,residual),{})
        self.assertEqual(out[0][-1,0].item(),5);self.assertEqual(out[0][0,0].item(),3)
        self.assertTrue(torch.equal(residual,original));self.assertTrue(e['consumed'] and e['restored'])
        self.assertEqual(native.count(),0);self.assertEqual(mode.calls,1)

    def test_mutating_or_changed_source_refuses_before_write(self):
        native=test_native_regions.NativeRegionsTests().manager();source=torch.ones(64,dtype=torch.bfloat16)
        residual=torch.zeros(1,128,dtype=torch.bfloat16);site=NS(native=native,fault=True)
        e=dict(source=source,source_sha256=fingerprint(source))
        with self.assertRaisesRegex(RuntimeError,'INPLACE_CONSUMER'):
            ResidualRead(site,e,residual).__torch_dispatch__(Op(lambda *a:None,True),(),(residual+1,residual),{})
        source.add_(1)
        with self.assertRaisesRegex(RuntimeError,'SOURCE_CHANGED'):
            ResidualRead(site,e,residual).__torch_dispatch__(Op(lambda *a:None),(),(residual+1,residual),{})
        self.assertEqual(native.count(),0);self.assertEqual(residual.sum().item(),0)
