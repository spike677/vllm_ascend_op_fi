import types
import unittest
import torch
from reference_adapters.hidden_rows import MlpContributionLost, DecoderPaddingRow
from fault_injection.engines.region_lease import RegionLeases

LIB=torch.library.Library('dcat_ref_hidden','DEF')
LIB.define('metadata(Tensor x, Tensor residual) -> Tensor')
LIB.impl('metadata',lambda x,residual:residual,'CPU')


class Mlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = torch.nn.Identity()
        self.act_fn = torch.nn.Identity()
        self.down_proj = torch.nn.Identity()
    def forward(self, x):
        return x * 2


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = Mlp()
        self.input_layernorm = torch.nn.Identity()
        self.post_attention_layernorm = torch.nn.Identity()
    def forward(self, x):
        return self.mlp(x) + 1


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Decoder(), Decoder()])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x + 1


class HiddenTests(unittest.TestCase):
    def test_shape_only_custom_op_is_not_a_consumer(self):
        class WithResidual(torch.nn.Module):
            def __init__(self):super().__init__();self.mlp=Mlp()
            def forward(self,x):
                out=self.mlp(x)
                residual=torch.ops.dcat_ref_hidden.metadata(out,x)
                return out+residual
        model=WithResidual();x=torch.ones(2,4,dtype=torch.bfloat16)
        leases=RegionLeases('owner',1,stream=lambda:0,complete=lambda s:None)
        site=MlpContributionLost(model,types.SimpleNamespace(input_batch=types.SimpleNamespace(req_ids=['q'])),fault=True,region_leases=leases)
        site.mode.metadata_name='dcat_ref_hidden::metadata'
        site.install();result=model(x);site.finish_forward();site.close()
        self.assertTrue(torch.equal(result[-1],x[-1]))
        self.assertEqual(site.events[0]['consumer_operator'],'aten::add')
        self.assertEqual(leases.count(),0)

    def test_real_reader_two_generations_and_clean_recovery(self):
        for cls in (MlpContributionLost, DecoderPaddingRow):
            model = Model()
            x = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
            baseline = model(x)
            for generation in (1, 2):
                leases = RegionLeases('owner', generation, stream=lambda:0, complete=lambda s:None)
                runner = types.SimpleNamespace(input_batch=types.SimpleNamespace(req_ids=['q']))
                site = cls(model, runner, fault=True, region_leases=leases)
                site.install()
                result = model(x)
                site.finish_forward(); site.close()
                self.assertTrue(torch.equal(result[:-1], baseline[:-1]))
                self.assertFalse(torch.equal(result[-1], baseline[-1]))
                self.assertTrue(torch.equal(model(x), baseline))
                self.assertEqual(leases.count(), 0)
                self.assertEqual(len(site.events), 2)
                for event in site.events:
                    self.assertTrue(event['consumed'] and event['restored'])
                    self.assertEqual(event['before_sha256'], event['restored_sha256'])
                    self.assertEqual(event['changed'], 1)
                    self.assertEqual(event['changed_elements'], 4)
