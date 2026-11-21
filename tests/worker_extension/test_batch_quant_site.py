import hashlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from fault_injection.sites.batch_quant import BatchQuantSite


class AscendW8A8DynamicLinearMethod:
    def apply(self,layer,x):
        import torch_npu
        q,s=torch_npu.npu_dynamic_quant(x)
        return torch_npu.npu_quant_matmul(q,pertoken_scale=s)


class Event:
    def record(self,stream): pass
    def synchronize(self): pass


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant_method=types.SimpleNamespace(quant_method=AscendW8A8DynamicLinearMethod())
    def forward(self,x):return self.quant_method.quant_method.apply(self,x)


class BatchSiteTests(unittest.TestCase):
    def setUp(self):
        self.buffers=[]
        def quant(x,**kwargs):
            s=x.abs().amax(-1)/127;q=torch.round(x/s[:,None]).to(torch.int8)
            self.buffers.append((q,s,q.clone(),s.clone()))
            return q,s
        def qmm(q,*,pertoken_scale):return q.float()*pertoken_scale[:,None]
        self.api=types.SimpleNamespace(npu_dynamic_quant=quant,npu_quant_matmul=qmm)
        self.enterContext(patch.dict(sys.modules,torch_npu=self.api))
        self.enterContext(patch.object(torch,'npu',types.SimpleNamespace(Event=Event,current_stream=lambda:7),create=True))
        sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.enterContext(patch('fault_injection.sites.batch_quant.SOURCE_SHA',sha))
        self.model=Model();self.runner=types.SimpleNamespace(input_batch=types.SimpleNamespace(req_ids=['a','b']))

    def test_all_calls_original_consumer_and_byte_restore(self):
        x=torch.tensor([[1.,2.,3.],[4.,20.,100.]])
        site=BatchQuantSite(self.model,self.runner,fault=True);site.actual_rows=lambda:2
        site.install()
        try:
            for _ in range(4):self.model(x)
            self.assertEqual(len(site.events),4)
            self.assertTrue(all(e['applied'] and e['consumed'] and e['restored'] and e['reference_exact'] for e in site.events))
            self.assertTrue(all(e['changed']>0 for e in site.events))
            for q,s,bq,bs in self.buffers:
                self.assertTrue(torch.equal(q,bq));self.assertTrue(torch.equal(s,bs))
        finally:self.assertEqual(site.close(),dict(leases=0,hooks=0,inflight=0))

    def test_consumer_failure_still_restores_owned_buffers(self):
        def fail(*args,**kwargs):raise RuntimeError('device consumer failed')
        self.api.npu_quant_matmul=fail
        site=BatchQuantSite(self.model,self.runner,fault=True);site.actual_rows=lambda:2;site.install()
        try:
            with self.assertRaisesRegex(RuntimeError,'device consumer failed'):
                self.model(torch.tensor([[1.,2.],[20.,200.]]))
            self.assertTrue(site.events[0]['failed']);self.assertTrue(site.events[0]['restored'])
            self.assertFalse(site.events[0]['consumed'])
        finally:site.close()
