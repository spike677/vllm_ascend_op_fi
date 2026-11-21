import unittest
import sys
from types import SimpleNamespace
from unittest.mock import patch
import torch
from fault_injection.engines.region_lease import RegionLeases
from fault_injection.sites.projection39 import ProjectionConsumerMode,Projection39Site,contains_element


class ProjectionTests(unittest.TestCase):
    def test_alias_schema_contiguous_copy_is_consumption_but_noop_is_not(self):
        class Contiguous:
            _schema=torch.ops.aten.contiguous.default._schema
            def __call__(self,x):return x.contiguous()
        x=torch.arange(24,dtype=torch.bfloat16).reshape(4,6);before=x.clone()
        leases=RegionLeases('cpu',1,stream=lambda:0,complete=lambda s:None);mode=ProjectionConsumerMode(leases)
        event={};mode.watch(x[-1:,:1].reshape(1),event)
        same=mode.__torch_dispatch__(Contiguous(),(),(x,),{})
        self.assertEqual(leases.count(),1);self.assertTrue(torch.isinf(same[-1,0]))
        result=mode.__torch_dispatch__(Contiguous(),(),(x[:,:3],),{})
        self.assertEqual(leases.count(),0);self.assertTrue(torch.isinf(result[-1,0]))
        self.assertTrue(torch.equal(x,before));self.assertEqual(event['consumer_operator'],'aten::contiguous')
    def test_pure_text_sites_exclude_the_forbidden_visual_branch(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.text=torch.nn.Linear(4,4,bias=False).to(torch.bfloat16)
                self.visual=torch.nn.Linear(4,4,bias=False).to(torch.bfloat16)
            def forward(self,x):return self.text(x).sin()
        model=Model();runner=SimpleNamespace(input_batch=SimpleNamespace(req_ids=['r']))
        leases=RegionLeases('cpu',1,stream=lambda:0,complete=lambda s:None)
        with patch.dict(sys.modules,{'vllm.model_executor.layers.linear':SimpleNamespace(LinearBase=torch.nn.Linear),
                                    'torch_npu':SimpleNamespace(get_npu_format=lambda t:2)}):
            site=Projection39Site(model,runner,fault=False,region_leases=leases)
            self.assertEqual(site.install(),['text'])
            model(torch.ones(2,4,dtype=torch.bfloat16));site.close()
            self.assertEqual(len(site.events),1);self.assertEqual(site.excluded[0]['site'],'visual')
    def test_view_is_not_consumer_and_copy_reads_fault_before_restore(self):
        x=torch.arange(12,dtype=torch.bfloat16).reshape(3,4);before=x.clone();events=[]
        leases=RegionLeases('cpu',1,stream=lambda:0,complete=lambda s:events.append(s));mode=ProjectionConsumerMode(leases)
        e={};mode.watch(x[-1:,:1].reshape(1),e)
        with mode:
            view=x.view(3,4)
            self.assertEqual(leases.count(),1)
            destination=torch.empty_like(view);destination.copy_(view)
        self.assertTrue(torch.isinf(destination[-1,0]));self.assertTrue(torch.equal(x,before))
        self.assertTrue(e['consumed'] and e['restored']);self.assertEqual(leases.count(),0)
        self.assertEqual(e['consumer_operator'],'aten::copy_')

    def test_strided_view_does_not_claim_to_read_a_gap(self):
        x=torch.arange(24,dtype=torch.bfloat16).reshape(4,6)
        self.assertTrue(contains_element(x[:,::2],x[-1:,:1].reshape(1)))
        self.assertFalse(contains_element(x[:,1::2],x[-1:,:1].reshape(1)))
        self.assertTrue(contains_element(x.t(),x[-1:,:1].reshape(1)))

    def test_mutating_consumer_refused_before_legitimate_write(self):
        x=torch.ones(3,dtype=torch.bfloat16)
        leases=RegionLeases('cpu',1,stream=lambda:0,complete=lambda s:None);mode=ProjectionConsumerMode(leases)
        mode.watch(x[:1],{})
        with mode,self.assertRaisesRegex(RuntimeError,'WRITES_PROTECTED'):x.add_(2)
        self.assertEqual(leases.count(),1);self.assertTrue(torch.equal(x[1:],torch.ones(2,dtype=torch.bfloat16)))
