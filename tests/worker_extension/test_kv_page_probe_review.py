import copy,unittest
from tools.review_kv_page_probe import review


def receipt():
    events=[];requests=[]
    for rid in ('a','b'):
        for i in range(4):
            requests.append(dict(ids=[rid]))
            events.append(dict(sequence=len(events)+1,action='before_execute',request_id=rid,
                               request_generation=rid+'-generation',pages=[[0,4]],full_attention_layers=list(range(16))))
        events.append(dict(events[-1],sequence=len(events)+1,action='before_free'))
    return dict(status='CENSUS_COMPLETE_NOT_PRODUCT_ACCEPTANCE',ranks=[dict(rank=r,
        page_barrier_events=copy.deepcopy(events),requests=copy.deepcopy(requests),active_page_tickets=0,
        page_injection_executed=False) for r in range(2)])


class ReviewTests(unittest.TestCase):
    def test_actual_alignment_is_only_readonly_proof(self):
        r=review(receipt());self.assertTrue(r['passed']);self.assertFalse(r['qualification_granted'])
    def test_missing_plugin_events_do_not_pass(self):
        d=receipt();d['ranks'][0]['page_barrier_events']=[];self.assertFalse(review(d)['passed'])
    def test_lagged_rank_and_unmatched_free_refuse(self):
        d=receipt();d['ranks'][1]['page_barrier_events'][-1]['request_generation']='old'
        r=review(d);self.assertIn('1:UNMATCHED_FREE',r['issues']);self.assertIn('1:RANK_NOTICES_DIFFER',r['issues'])
    def test_worker_request_mismatch_refuses(self):
        d=receipt();d['ranks'][0]['requests'][0]['ids']=['wrong'];self.assertFalse(review(d)['passed'])


if __name__=='__main__':unittest.main()
