import unittest
from copy import deepcopy
from test_continuous28_review import fixture as attention_fixture
from tools.review_continuous_batch_site import review


def fixture():
    report=attention_fixture()
    for scope in report['scopes']:
        for req in scope['requests']:
            req['finite']=True
            ids=[req['phase']+'0',req['phase']+'1']
            req['mapping']=[dict(internal_request_id=i) for i in ids]
            for r in req['ranks']:
                r['uncovered']=[]
                for row in r['requests']:row['request_ids']=ids
                for event in r['events']:
                    event.update(request_ids=ids,reference_exact=True,all_real_rows=True,failed=False,qmm_calls=1,
                                 before_sha256=['a'*64,'b'*64],restored_sha256=['a'*64,'b'*64])
                r['coverage']=dict(observed=4,eligible=4,attempted=4,applied=4,changed=4,consumed=4,restored=4,failed=0)
    report['warmup']=[deepcopy(report['scopes'][0]['requests'][0]) for _ in range(2)]
    return report


class ReviewTests(unittest.TestCase):
    def test_positive_does_not_grant_public_qualification(self):
        self.assertFalse(review(fixture())['qualification_granted'])

    def test_uncovered_reference_restore_mapping_and_warmup_fail_closed(self):
        for field in ('reference_exact','all_real_rows','consumed'):
            r=fixture();r['scopes'][0]['requests'][1]['ranks'][0]['events'][0][field]=False
            with self.assertRaises(ValueError):review(r)
        r=fixture();r['scopes'][0]['requests'][1]['ranks'][0]['uncovered']=['missed']
        with self.assertRaises(ValueError):review(r)
        r=fixture();r['scopes'][0]['requests'][1]['ranks'][0]['events'][0]['restored_sha256'][1]='c'*64
        with self.assertRaises(ValueError):review(r)
        r=fixture();r['warmup'][-1]['finite']=False
        with self.assertRaises(ValueError):review(r)
