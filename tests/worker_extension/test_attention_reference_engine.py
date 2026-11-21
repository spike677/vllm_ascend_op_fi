import unittest
import torch
from fault_injection.engines.attention import recompute_attention_row


class AttentionEngineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def reference(self,q,k,v,ids):
        group=q.shape[0]//k.shape[1]
        k=k[ids].double().repeat_interleave(group,dim=1)
        v=v[ids].double().repeat_interleave(group,dim=1)
        return torch.einsum('hk,khd->hd',torch.einsum('hd,khd->hk',q.double(),k).mul(.25).softmax(-1),v)

    def test_gqa_absorbed_mla_sparse_and_tail(self):
        for hq,hkv,dk,dv in ((4,2,8,8),(4,1,12,5)):
            q=torch.randn(hq,dk); k=torch.randn(19,hkv,dk); v=torch.randn(19,hkv,dv)
            originals=[x.clone() for x in (q,k,v)]
            for ids in (list(range(19)),[0,2,5,8,13,17]):
                expected=self.reference(q,k,v,ids)
                for block in (1,4,32):
                    actual=recompute_attention_row(q,k,v,scale=.25,block_size=block,key_indices=ids)
                    torch.testing.assert_close(actual.double(),expected,rtol=2e-6,atol=2e-7)
            for a,b in zip((q,k,v),originals): self.assertTrue(torch.equal(a,b))

    def test_denominator_only_replay_has_known_large_effect(self):
        q=torch.zeros(2,4); k=torch.zeros(256,1,4); v=torch.ones(256,1,3)
        normal=recompute_attention_row(q,k,v,scale=.5,block_size=128)
        fault=recompute_attention_row(q,k,v,scale=.5,block_size=128,replay_blocks=(1,))
        self.assertTrue(torch.equal(normal,torch.ones_like(normal)))
        torch.testing.assert_close(fault,torch.full_like(fault,2/3))

    def test_rescale_of_replayed_denominator(self):
        q=torch.ones(1,1); k=torch.tensor([0.,1.,20.,21.,30.]).view(5,1,1)
        v=torch.arange(5.).view(5,1,1)
        result=recompute_attention_row(q,k,v,scale=1.,block_size=2,replay_blocks=(0,))
        weights=torch.exp(k[:,0,0].double()-30)
        expected=(weights*v[:,0,0]).sum()/(weights.sum()+weights[:2].sum())
        torch.testing.assert_close(result.double().squeeze(),expected,rtol=1e-6,atol=1e-7)

    def test_rejects_duplicate_or_invalid_keys_and_replay(self):
        q=torch.ones(1,2); k=torch.ones(3,1,2); v=torch.ones(3,1,4)
        for ids in ([],[0,0],[-1],[3]):
            with self.assertRaises(ValueError):
                recompute_attention_row(q,k,v,scale=1.,block_size=2,key_indices=ids)
        for replay in ((2,),(-1,),(0,0)):
            with self.assertRaises(ValueError):
                recompute_attention_row(q,k,v,scale=1.,block_size=2,replay_blocks=replay)

if __name__=='__main__': unittest.main()
