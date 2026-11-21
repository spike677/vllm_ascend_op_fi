import unittest
from fault_injection.sites.ep_context import validate_ep_boundary


class EPTests(unittest.TestCase):
    def data(self,n=8):
        return dict(members=range(n),owner='scope',generation=2,request_ids=['r'],
            rank_calls={i:[] for i in range(n)},
            rank_placement={i:[dict(physical_slot=0,logical_expert=100+i)] for i in range(n)})
    def call(self,rank,route=0):
        return dict(owner='scope',generation=2,request_id='r',token_position=3,
            invocation=4,route_ordinal=route,physical_slot=0,logical_expert=100+rank)
    def test_dynamic_fault_route_and_idle_ranks_are_not_forced(self):
        d=self.data();d['rank_calls'][5]=[self.call(5)]
        r=validate_ep_boundary(**d);self.assertTrue(r['passed']);self.assertEqual(len(r['no_call_ranks']),7)
        d['rank_calls'][5]=[];d['rank_calls'][1]=[self.call(1)]
        self.assertTrue(validate_ep_boundary(**d)['passed'])
    def test_stale_missing_placement_and_duplicate_dispatch_rejected(self):
        d=self.data();d['rank_calls'][0]=[self.call(0)];d['rank_calls'][1]=[self.call(1)]
        self.assertFalse(validate_ep_boundary(**d)['passed'])
        d['rank_calls'][1]=[];d['rank_calls'][0][0]['generation']=1
        self.assertFalse(validate_ep_boundary(**d)['passed'])
        d=self.data();d['rank_calls'][0]=[self.call(1)]
        self.assertFalse(validate_ep_boundary(**d)['passed'])
    def test_one_rank_uses_same_path(self):
        d=self.data(1);d['rank_calls'][0]=[self.call(0)]
        self.assertTrue(validate_ep_boundary(**d)['passed'])
