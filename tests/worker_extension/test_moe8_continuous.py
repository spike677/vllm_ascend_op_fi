import sys,unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from fault_injection.sites.moe8 import remap_groups,MoE8Site
from fault_injection.engines.region_lease import RegionLeases


class MoETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library=torch.library.Library('npu','FRAGMENT')
        cls.library.define('npu_moe_token_unpermute(Tensor permuted_tokens, Tensor sorted_indices, Tensor? probs=None, bool padded_mode=False, int[]? restore_shape=None) -> Tensor')
        def cpu(permuted_tokens,sorted_indices,probs=None,padded_mode=False,restore_shape=None):
            return (permuted_tokens[sorted_indices.long()].float().reshape(*probs.shape,-1)*probs.unsqueeze(-1)).sum(1).to(permuted_tokens.dtype)
        cls.library.impl('npu_moe_token_unpermute',cpu,'CPU')
    @classmethod
    def tearDownClass(cls):cls.library._destroy()

    def test_actual_producer_join_native_consumption_and_input_restore(self):
        h=torch.arange(24,dtype=torch.bfloat16).reshape(6,4)
        i=torch.tensor([4,0,5,1,3,2],dtype=torch.int32);p=torch.tensor([[.2,.8],[.4,.6],[.1,.9]])
        saved=(i.clone(),p.clone());metadata=NS(expanded_row_idx=i,topk_weights=p,restore_shape=(3,4))
        site=MoE8Site(None,NS(input_batch=NS(req_ids=['actual-request'])),fault=True,
            region_leases=RegionLeases('owner',1,stream=lambda:0,complete=lambda s:None))
        dispatcher=object();site.parents=[('moe',NS(top_k=2))];site.forward_calls=[{}]
        site.pipelines[id(dispatcher)]=dict(dispatcher=dispatcher,metadata=metadata,mlp_output=h,
            site='moe',request_ids=['actual-request'],topk_ids_sha256='route-observation')
        normal=torch.ops.npu.npu_moe_token_unpermute(h,i,p)
        def original(d,x,m,b):return torch.ops.npu.npu_moe_token_unpermute(x,m.expanded_row_idx.abs(),m.topk_weights)
        with patch.dict(sys.modules,{'vllm.forward_context':NS(get_forward_context=lambda:NS(attn_metadata=NS(num_actual_tokens=3)))}):
            result=site.combine(original,dispatcher,h,metadata,None)
        self.assertTrue(torch.equal(result,normal[[1,2,2]]))
        self.assertTrue(torch.equal(i,saved[0]) and torch.equal(p,saved[1]))
        e=site.events[0];self.assertTrue(e['consumed'] and e['restored'] and e['actual_producer_join'])
        self.assertEqual(e['row_request_ids'],['actual-request']*3);self.assertEqual(e['native_calls'],1)
        self.assertEqual(e['index_abs_calls'],1)
        self.assertFalse(site.pipelines);self.assertEqual(site.pending,0)

    def test_decode_one_group_applies_identity_and_invalid_permutation_refuses(self):
        i=torch.tensor([1,0],dtype=torch.int32);p=torch.tensor([[.1,.9]])
        ni,np,g=remap_groups(i,p);self.assertTrue(torch.equal(i,ni) and torch.equal(p,np));self.assertEqual(g.tolist(),[0])
        with self.assertRaisesRegex(RuntimeError,'PERMUTATION'):remap_groups(torch.tensor([0,0]),p)

    def test_foreign_request_pipeline_refused_before_consumer(self):
        site=MoE8Site(None,NS(input_batch=NS(req_ids=['new'])),fault=True,
            region_leases=RegionLeases('owner',1,stream=lambda:0,complete=lambda s:None))
        site.parents=[('moe',NS(top_k=2))]
        with self.assertRaisesRegex(RuntimeError,'JOIN_UNPROVEN'):
            site.combine(lambda *a: self.fail('consumer must not run'),object(),None,None,None)
        self.assertEqual(site.pending,0)
