import copy
import unittest
from unittest.mock import patch
from tools.review_disable_drain import check


class DrainReviewTests(unittest.TestCase):
    def fixture(self):
        scopes=[]
        for scope in range(2):
            completed=[]
            for request_id in (f'{scope}-a',f'{scope}-b'):
                ranks=[dict(rank=r,request_id=request_id,completed=True,
                    coverage=dict(eligible=4,applied=4,consumed=4,restored=4,failed=0),
                    resources=dict(leases=0,hooks=0,inflight=0)) for r in (0,1)]
                completed.append(dict(request_id=request_id,ranks=ranks))
            scopes.append(dict(scope=scope,drain_stress=dict(submitted=2,pending_at_disable=2,
                http_results=[{},{}],status_failures=[],completed_requests=completed)))
        return dict(scopes=scopes)
    @patch('tools.review_disable_drain.review',return_value={})
    def test_completed_cohort_is_required(self,_):
        data=self.fixture();self.assertEqual(check(data)['status'],'TWO_OWNER_DRAIN_REVIEW_PASSED')
        data['scopes'][1]['drain_stress']['completed_requests'][1]['ranks'][1]['completed']=False
        with self.assertRaisesRegex(ValueError,'ASSOCIATION'):check(data)
    @patch('tools.review_disable_drain.review',return_value={})
    def test_old_rank_reply_and_unreleased_lease_are_rejected(self,_):
        for field,value in [('request_id','old-request'),('resources',dict(leases=1,hooks=0,inflight=0))]:
            data=self.fixture();data['scopes'][0]['drain_stress']['completed_requests'][0]['ranks'][1][field]=value
            with self.assertRaises(ValueError):check(data)
    @patch('tools.review_disable_drain.review',return_value={})
    def test_sequential_only_run_is_not_stress_evidence(self,_):
        data=self.fixture();data['scopes'][0]['drain_stress']['pending_at_disable']=0
        with self.assertRaisesRegex(ValueError,'OUTSTANDING'):check(data)


if __name__=='__main__':unittest.main()
