import unittest
from types import SimpleNamespace as NS
import torch
from fault_injection.sites.mla_rope22 import MLARope22Site,MLAConsumerMode
from fault_injection.engines.region_lease import RegionLeases

class MLARopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib=torch.library.Library('npu','FRAGMENT')
        cls.lib.define('npu_interleave_rope(Tensor x, Tensor cos, Tensor sin) -> Tensor')
        cls.lib.define('npu_kv_rmsnorm_rope_cache(Tensor x, Tensor weight, Tensor cos, Tensor sin) -> (Tensor,Tensor,Tensor,Tensor)')
        cls.lib.impl('npu_interleave_rope',lambda x,cos,sin:x*cos+sin,'CPU')
        cls.lib.impl('npu_kv_rmsnorm_rope_cache',lambda x,w,cos,sin:(x*cos+sin,x.clone(),x*cos+sin,x.clone()),'CPU')
    @classmethod
    def tearDownClass(cls):cls.lib._destroy()
    def test_real_lookup_pair_consumed_then_byte_restored(self):
        tables=torch.arange(32,dtype=torch.bfloat16).reshape(8,4)
        positions=torch.tensor([0,2,3]);cos=tables[positions,None,None,:].clone();sin=(tables+1)[positions,None,None,:].clone()
        saved=(cos.clone(),sin.clone());part=NS(input_positions=positions,cos=cos,sin=sin)
        meta=NS(prefill=part,num_actual_tokens=3,num_prefills=1,num_decodes=0)
        site=MLARope22Site(None,NS(input_batch=NS(req_ids=['request'])),fault=True,
            region_leases=RegionLeases('owner',1,stream=lambda:0,complete=lambda s:None))
        site.rotary=NS(_cos_cache=tables,_sin_cache=tables+1);site.forward_calls=[{}]
        def original(*args):
            q=torch.ops.npu.npu_interleave_rope(torch.ones_like(cos),cos,sin)
            k=torch.ops.npu.npu_kv_rmsnorm_rope_cache(torch.ones_like(cos),tables,cos,sin)
            return q,k
        q,k=site.call('mla','mla_preprocess_prefill',original,(None,None,None,meta))
        self.assertTrue(torch.equal(q,2*tables[positions+1,None,None,:]+1))
        self.assertTrue(torch.equal(cos,saved[0]) and torch.equal(sin,saved[1]))
        self.assertEqual(site.pending,0);self.assertEqual(site.events[0]['consumer_calls'],['q','k'])
        self.assertEqual(site.events[0]['rotary_rows_changed'],3)
    def test_foreign_cos_buffer_is_not_consumer_proof(self):
        x=torch.ones(2,1,1,4);mode=MLAConsumerMode(x,x,True)
        with self.assertRaisesRegex(RuntimeError,'INPUT_CHANGED'),mode:
            torch.ops.npu.npu_interleave_rope(x,x.clone(),x)
