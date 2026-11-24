import unittest
from copy import deepcopy
from tools.review_alias15_finiteness import link_calls


class FiniteReviewTests(unittest.TestCase):
    def fixture(self):
        call=dict(site='a',after_sha256='base',consumer_sha256='consumer',initial_all_finite=False)
        rows=[dict(rank=0,request_ids=['q'],kind=kind,site=site,sha256=sha,nonfinite=1,elements=8)
              for kind,site,sha in [('attention_base_return','a','base'),
                                    ('projection_input','a','consumer'),('model_hidden','model','hidden')]]
        return [call],rows
    def test_links_nonfinite_through_consumer_and_hidden(self):
        calls,rows=self.fixture();result=link_calls(calls,rows,0,'q')
        self.assertEqual(result['projection_nonfinite_calls'],1)
        self.assertEqual(result['hidden_max_nonfinite'],1)
    def test_other_request_cannot_supply_missing_evidence(self):
        calls,rows=self.fixture();rows[0]['request_ids']=['other']
        with self.assertRaisesRegex(ValueError,'MISSING'):link_calls(calls,rows,0,'q')
    def test_sha_mismatch_rejected(self):
        calls,rows=self.fixture();rows[1]['sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'IDENTITY'):link_calls(calls,rows,0,'q')
    def test_extra_call_rejected(self):
        calls,rows=self.fixture();rows.append(deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError,'UNASSOCIATED'):link_calls(calls,rows,0,'q')
