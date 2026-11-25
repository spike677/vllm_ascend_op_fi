import copy,unittest
from reference_adapters.review_refusal import review


def fixture():
    before=dict(api=[dict(text='ok')],ranks=[dict(rank=r,removed=True,
        steps=[dict(hidden='a'*64,logits='b'*64) for _ in range(4)]) for r in range(2)])
    checks=[]
    for kind,code in [('missing_site','REFERENCE_NO_REAL_BF16_WEIGHT'),
                      ('identity_drift','EXTERNAL_DEPLOYMENT_IDENTITY_MISMATCH'),
                      ('range','PRODUCT_ONLY_ALL_REQUESTS_SUPPORTED'),
                      ('missing_site_repeat','REFERENCE_NO_REAL_BF16_WEIGHT')]:
        checks.append(dict(kind=kind,reply=dict(exit_code=4,refusal_codes=[code],refusals=[dict(rank=r) for r in range(2)]),
            after=copy.deepcopy(before),canary_exact=True,engine_alive=True,client_status=dict(enabled=False),
            audit=[dict(rank=r,leases=0,owners=0,bindings=0,wrappers=0) for r in range(2)]))
    return dict(status='REFERENCE_CONTROL_REFUSAL_A2_PASSED',before=before,checks=checks)


class RefusalReviewTests(unittest.TestCase):
    def test_good(self):self.assertTrue(review(fixture())['passed'])
    def test_exact_hash_not_boolean_claim(self):
        d=fixture();d['checks'][1]['after']['ranks'][1]['steps'][0]['hidden']='c'*64
        self.assertFalse(review(d)['passed'])
    def test_leak_is_failure(self):
        d=fixture();d['checks'][0]['audit'][1]['leases']=1
        self.assertFalse(review(d)['passed'])
    def test_rank_replay_or_missing_step_is_failure(self):
        for key in ('rank','steps'):
            d=fixture();d['before']['ranks'][1][key]=0 if key=='rank' else []
            self.assertFalse(review(d)['passed'])
