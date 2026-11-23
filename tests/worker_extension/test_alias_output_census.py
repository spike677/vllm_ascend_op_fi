import hashlib,unittest
from pathlib import Path
import torch
from tools.alias_output_census import AliasOutputCensus

class AscendAttentionBackendImpl:
    def forward(self,query,output=None):
        output.copy_(query.view_as(output)+1);return output
class Attention(torch.nn.Module):
    def __init__(self):super().__init__();self.impl=AscendAttentionBackendImpl()
    def forward(self,x):
        output=torch.empty(x.shape,dtype=x.dtype,device=x.device)
        view=output.view(x.shape[0],2,2);self.impl.forward(x,output=view)
        return view.view_as(x)
class Model(torch.nn.Module):
    def __init__(self):super().__init__();self.attn=Attention();self.o_proj=torch.nn.Identity()
    def forward(self,x):return self.o_proj(self.attn(x))
class GatedModel(Model):
    def forward(self,x):return self.o_proj(self.attn(x)*torch.sigmoid(x))

class OutputCensusTests(unittest.TestCase):
    def test_natural_allocation_write_consumer_no_numeric_change(self):
        model=Model();x=torch.arange(8,dtype=torch.float32).reshape(2,4)
        # named root Attention parent is root; use actual module path with a dot.
        wrapper=torch.nn.Sequential(model)
        empty=torch.empty;c=AliasOutputCensus(wrapper,format_reader=lambda t:2,complete=lambda:None,
            source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        c.install()
        try:
            self.assertTrue(torch.equal(wrapper(x),x+1));self.assertEqual(len(c.events),1)
            e=c.events[0];self.assertTrue(e['device_consumer_completed']);self.assertTrue(e['consumer_input_same_storage'])
            self.assertFalse(e['base_is_view_object']);self.assertEqual(e['view_after_sha256'],e['consumer_sha256'])
        finally:c.close()
        self.assertIs(torch.empty,empty);self.assertFalse(c.methods);self.assertFalse(c.hooks)
        self.assertTrue(torch.equal(wrapper(x),x+1))

    def test_source_mismatch_refuses_before_install(self):
        c=AliasOutputCensus(torch.nn.Sequential(Model()),format_reader=lambda t:2,complete=lambda:None,source_sha='0'*64)
        with self.assertRaisesRegex(RuntimeError,'SOURCE_UNREVIEWED'):c.install()
        self.assertFalse(c.methods);self.assertFalse(c.hooks)

    def test_real_intermediate_gate_consumer_is_distinct_from_projection(self):
        model=torch.nn.Sequential(GatedModel());x=torch.arange(8,dtype=torch.float32).reshape(2,4)
        expected=model(x);mul=torch.Tensor.__mul__
        c=AliasOutputCensus(model,format_reader=lambda t:2,complete=lambda:None,source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        c.install()
        try:
            self.assertTrue(torch.equal(expected,model(x)))
            e=c.events[0];self.assertTrue(e['gate_device_read_completed']);self.assertFalse(e['consumer_input_same_storage'])
            self.assertEqual(e['gate_result_sha256'],e['consumer_sha256'])
        finally:c.close()
        self.assertIs(torch.Tensor.__mul__,mul)

if __name__=='__main__':unittest.main()
