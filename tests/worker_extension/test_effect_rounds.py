import copy
import unittest
from types import SimpleNamespace
from test_product_effect_probe import Adapter,response
from tools.product_effect_probe import run,summarize,canonical
from tools.effect_rounds import same_composition


def trace(split=False):
    rows=[dict(request_id='random-'+str(i),position=i,prompt_sha256=str(i)*64,
               prompt_length=12,computed=0,scheduled=12) for i in range(2)]
    steps=[dict(rows=rows)] if not split else [dict(rows=[rows[0]]),dict(rows=[dict(rows[1],position=0)])]
    return [dict(rank=r,steps=steps,truncated=False,methods_restored=True,
                 source='ACTUAL_RUNNER_PREPROCESS_AND_LOGITS') for r in range(2)]


class BatchAdapter(Adapter):
    def __init__(self, always_split=False, exact=True):
        super().__init__(); self.calls=0;self.always_split=always_split;self.exact=exact
    def identity(self):
        return dict(super().identity(),mechanism='quant.batch_shared_scale',max_num_seqs=8)
    def infer_observed(self,prompts,sampling):
        self.calls+=1
        split=self.calls==3 or (self.always_split and self.calls>=3)
        return dict(output=response(.1 if self.active else (.2 if split or (not self.exact and self.calls==4) else 0)),trace=trace(split))


class RoundTests(unittest.TestCase):
    def test_actual_request_ids_preserved_but_not_comparison_keys(self):
        a=trace();b=copy.deepcopy(a)
        for r in b:
            for s in r['steps']:
                for row in s['rows']:row['request_id']+='new'
        self.assertTrue(same_composition(a,b,2))
        self.assertFalse(same_composition(a,trace(True),2))

    def test_mismatch_retry_retains_all_replies_and_stops_at_match(self):
        raw=run(BatchAdapter(),['a','b'])
        self.assertEqual(len(raw['scopes'][0]['attempts']),4)
        self.assertEqual([a['composition_matches_baseline'] for a in raw['scopes'][0]['attempts'] if a['phase']=='off'],[False,True])
        summarize(raw,canonical(raw))

    def test_same_composition_numerical_mismatch_is_not_retried(self):
        adapter=BatchAdapter(exact=False)
        raw=run(adapter,['a','b'])
        with self.assertRaisesRegex(ValueError,'OFF_API_NOT_EXACT'):summarize(raw,canonical(raw))
        self.assertEqual(adapter.calls,7)

    def test_exhaustion_detaches_and_preserves_attempts(self):
        snapshots=[]
        with self.assertRaisesRegex(ValueError,'RETRY_EXHAUSTED'):
            run(BatchAdapter(always_split=True),['a','b'],max_off_attempts=2,
                checkpoint=lambda raw:snapshots.append(copy.deepcopy(raw)))
        self.assertEqual(snapshots[-1]['control_sequence'][-1]['action'],'detach')
        self.assertEqual(len(snapshots[-1]['scopes'][0]['attempts']),4)

    def test_missing_or_truncated_trace_fails_closed(self):
        for field,value in [('truncated',True),('methods_restored',False),('steps',[])]:
            t=trace();t[0][field]=value
            with self.assertRaises(ValueError):same_composition(trace(),t,2)

    def test_tampered_selected_output_is_rejected(self):
        raw=run(BatchAdapter(),['a','b']);raw['scopes'][0]['outputs']['off']=response(.3)
        with self.assertRaisesRegex(ValueError,'SELECTED_REPLY'):summarize(raw,canonical(raw))

    def test_observer_restores_exact_existing_callable(self):
        from experiments.worker_core_v11_20261118.effect_schedule_probe import RoundObserver
        class Model:
            def compute_logits(self,x):return x
        class Runner:
            def execute_model(self,scheduler):return self._preprocess()
            def _preprocess(self):return 'unchanged'
        r=Runner();r.model=Model()
        r.input_batch=SimpleNamespace(req_ids=['r'],num_computed_tokens_cpu=[0])
        r.requests={'r':SimpleNamespace(prompt_token_ids=[1,2])}
        old=r.execute_model;r.execute_model=old
        observer=RoundObserver(SimpleNamespace(model_runner=r))
        self.assertEqual(r.execute_model(SimpleNamespace(num_scheduled_tokens={'r':2})),'unchanged')
        evidence=observer.stop()
        self.assertIs(vars(r)['execute_model'],old)
        self.assertNotIn('_preprocess',vars(r));self.assertNotIn('compute_logits',vars(r.model))
        self.assertEqual(evidence['steps'][0]['rows'][0]['scheduled'],2)
