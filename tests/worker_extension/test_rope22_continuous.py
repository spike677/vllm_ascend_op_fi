import unittest
from types import SimpleNamespace as NS
import torch
from fault_injection.engines.region_lease import RegionLeases
from fault_injection.sites.rope22 import Rope22Site


class RopeTests(unittest.TestCase):
    def test_original_consumer_reads_wrong_rows_then_transient_is_restored(self):
        cache=torch.arange(64,dtype=torch.bfloat16).reshape(8,8);before=cache.clone()
        positions=torch.tensor([[0,1],[0,1],[0,1]]);saved=positions.clone()
        cos=cache[positions].clone();original=cos.clone();calls=[]
        class Fused:
            _schema=torch._C.parse_schema('vllm::triton_split_qkv_rmsnorm_mrope(Tensor cos_sin) -> (Tensor, Tensor)')
            def __call__(self,cos_sin):calls.append(cos_sin);return cos_sin.clone(),cos_sin.clone()
        site=Rope22Site(None,NS(input_batch=NS(req_ids=['actual'])),fault=True,
            region_leases=RegionLeases('owner',1,stream=lambda:0,complete=lambda s:None))
        site.parents=[('actual.attn',NS(rotary_emb=NS(cos_sin_cache=cache)),positions)];site.forward_calls=[{}]
        result=site.mode.__torch_dispatch__(Fused(),(),(cos,),{})
        self.assertEqual(len(calls),1);self.assertIs(calls[0],cos)
        self.assertTrue(torch.equal(result[0],cache[positions+1]))
        self.assertTrue(torch.equal(cos,original) and torch.equal(cache,before) and torch.equal(positions,saved))
        self.assertEqual(site.pending,0);self.assertEqual(site.events[0]['rotary_rows_changed'],6)
        self.assertTrue(site.events[0]['consumed'] and site.events[0]['restored'])

    def test_mismatched_lookup_refused_before_original_or_write(self):
        cache=torch.ones(8,8,dtype=torch.bfloat16);pos=torch.zeros(3,1,dtype=torch.long)
        class Fused:
            _schema=torch._C.parse_schema('vllm::triton_split_qkv_rmsnorm_mrope(Tensor cos_sin) -> Tensor')
            def __call__(self,cos_sin):raise AssertionError('must not execute')
        site=Rope22Site(None,NS(input_batch=NS(req_ids=['actual'])),fault=True,
            region_leases=RegionLeases('owner',1,stream=lambda:0,complete=lambda s:None))
        site.parents=[('attn',NS(rotary_emb=NS(cos_sin_cache=cache)),pos)];site.forward_calls=[{}]
        with self.assertRaisesRegex(RuntimeError,'LOOKUP_NOT_PROVEN'):
            site.mode.__torch_dispatch__(Fused(),(),(torch.zeros(3,1,8,dtype=torch.bfloat16),),{})
        self.assertEqual(site.pending,0)
