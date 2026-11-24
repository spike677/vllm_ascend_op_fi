import copy
import unittest
from unittest.mock import patch
from tools.review_alias15_nonfinite import review


class SafetyReviewTests(unittest.TestCase):
    def setUp(self):
        self.data={'scopes':[]}; self.rows=[]; self.seeds=[]
        for scope in range(2):
            requests=[]
            for n in range(3):
                rid=f'{scope}-{n}'
                ranks=[]
                for rank in (0,1):
                    ranks.append(dict(rank=rank,coverage={},completed=True,
                        alias_detach_effect={'calls':[dict(site='a',initial_all_finite=False,
                                                          after_sha256='base',consumer_sha256='consumer')]}))
                    self.seeds.append(dict(rank=rank,request_ids=[rid],reused=True,
                                           actual_base_written_by_stress=False,actual_nonfinite=8))
                    for kind in ('projection_input','model_hidden','model_logits','attention_base_return'):
                        self.rows.append(dict(rank=rank,request_ids=[rid],kind=kind,nonfinite=8,elements=8,
                            site='a' if kind in ('projection_input','attention_base_return') else 'model',
                            sha256='consumer' if kind=='projection_input' else 'base'))
                requests.append(dict(result=dict(request_id=rid,ranks=ranks)))
            self.data['scopes'].append(dict(scope=scope,requests=requests))

    @patch('tools.review_alias15_nonfinite.mechanism_review',return_value={'passed':True})
    def test_scoped_result_still_grants_no_qualification(self, base):
        result=review(self.data,self.rows,self.seeds)
        base.assert_called_once_with(self.data)
        self.assertEqual(len(result['requests']),12)
        self.assertFalse(result['qualification_granted'])
        self.assertFalse(result['natural_nonfinite_origin_proven'])

    @patch('tools.review_alias15_nonfinite.mechanism_review',return_value={})
    def test_direct_base_write_or_failed_reuse_not_accepted(self, base):
        for delta in (dict(actual_base_written_by_stress=True),dict(reused=False)):
            seeds=copy.deepcopy(self.seeds);seeds[0].update(delta)
            with self.assertRaisesRegex(ValueError,'REUSE_NOT_PROVEN'):
                review(self.data,self.rows,seeds)

    @patch('tools.review_alias15_nonfinite.mechanism_review',return_value={})
    def test_unrelated_request_nan_does_not_prove_consumer(self, base):
        self.rows[0]['request_ids']=['wrong-request']
        with self.assertRaisesRegex(ValueError,'CONSUMER_OR_HIDDEN|OBSERVATION_MISSING'):
            review(self.data,self.rows,self.seeds)

    @patch('tools.review_alias15_nonfinite.mechanism_review',return_value={})
    def test_finite_only_logits_cannot_pass(self, base):
        for r in self.rows:
            if r['kind']=='model_logits':r['nonfinite']=0
        with self.assertRaisesRegex(ValueError,'LOGITS_NOT_OBSERVED'):
            review(self.data,self.rows,self.seeds)
