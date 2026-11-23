import unittest
import torch
from tools.alias_chain_census import AliasChainCensus


class CensusTests(unittest.TestCase):
    def test_real_view_write_read_preserves_values(self):
        base = torch.zeros(4,4)
        census = AliasChainCensus(lambda t: 2)
        with census:
            view = base[:2]
            view.copy_(torch.ones_like(view))
            out = base + 1
        self.assertTrue(torch.equal(out, torch.tensor([[2.]*4]*2+[[1.]*4]*2)))
        r = census.report()
        self.assertGreater(r['counts']['base_read_after_view_write'],0)
        self.assertTrue(all(s['base_object_same'] and s['shared_storage'] for s in r['samples']))
        self.assertFalse(r['injection_enabled'])

    def test_readonly_view_not_a_chain(self):
        base=torch.ones(4,4);c=AliasChainCensus()
        with c:
            view=base[:2];out=view+1
        self.assertEqual(c.counts['view_writes'],0)
        self.assertEqual(c.report()['samples'],[])

    def test_detached_storage_not_alias(self):
        base=torch.zeros(4,4);c=AliasChainCensus()
        with c:
            view=base[:2].clone();view.fill_(7);out=base+1
        self.assertEqual(c.counts['base_read_after_view_write'],0)
        self.assertTrue(torch.equal(out,torch.ones(4,4)))

    def test_schema_out_and_bounded_samples(self):
        base=torch.zeros(4,4);c=AliasChainCensus(lambda t:None,max_samples=1)
        with c:
            for _ in range(3):
                view=base[:2];torch.add(view,1,out=view);out=base+1
        self.assertEqual(len(c.samples),1)
        self.assertTrue(c.report()['sampling_truncated'])
        self.assertIsNone(c.samples[0]['base']['format'])

    def test_other_view_reads_same_region(self):
        base=torch.zeros(4,4);c=AliasChainCensus()
        with c:
            view=base[:2];view.fill_(7);consumer=base.view(2,8)+1
        self.assertGreater(c.counts['base_read_after_view_write'],0)
        self.assertFalse(c.samples[0]['base_object_same'])

    def test_disjoint_view_does_not_consume_write(self):
        base=torch.zeros(4,4);c=AliasChainCensus()
        with c:
            base[:2].fill_(7);consumer=base[2:]+1
        self.assertEqual(c.counts['base_read_after_view_write'],0)


if __name__=='__main__':unittest.main()
