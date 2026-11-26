import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/worker_core_v11_20261118'))
from batch_baseline_control import request_order

class RequestJoinTests(unittest.TestCase):
    def sample(self):
        rows=[dict(request_id='cmpl-r-'+str(i)+'-abc',prompt_sha256=str(i)*64,position=i,prompt_length=7,computed=0,scheduled=7) for i in range(8)]
        return dict(id='cmpl-r',choices=[dict(index=i) for i in reversed(range(8))]),[
          dict(rank=r,methods_restored=True,truncated=False,source='ACTUAL_RUNNER_PREPROCESS_AND_LOGITS',steps=[dict(rows=copy.deepcopy(rows))]) for r in range(2)]
    def test_actual_ids_join_api_order(self):
        reply,trace=self.sample();self.assertEqual([r['index'] for r in request_order(reply,trace)],list(range(8)))
    def test_unrelated_reply_not_joined_by_prompt(self):
        reply,trace=self.sample();reply['id']='cmpl-other'
        with self.assertRaisesRegex(ValueError,'JOIN_UNPROVEN'):request_order(reply,trace)
    def test_rank_ids_must_agree(self):
        reply,trace=self.sample();trace[1]['steps'][0]['rows'][0]['request_id']='cmpl-r-0-other'
        with self.assertRaisesRegex(ValueError,'RANK_REQUEST_JOIN'):request_order(reply,trace)
