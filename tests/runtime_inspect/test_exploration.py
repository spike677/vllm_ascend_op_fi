import copy
import unittest
from ascend_fi.inspect.effect_stats import EffectStats, validate_stats
from ascend_fi.inspect.exploration import make_experiment, ROLE_GROUPS, window_gate, contextual_findings
from ascend_fi.evaluation.distortion.mechanism import validate_mechanism
from ascend_fi.evaluation.distortion.contracts import FaultConfigurationRecord
from evaluation.test_distortion import fault_row, response, config


def event(i, layer=0):
    return {'module':f'model.layers.{layer}.mlp.down_proj','layer':layer,'rank':0,
            'mode':'REFERENCE_REQUANT','scale_provider':'tensor_absmax','scale_multiplier':.25,
            'hit_index':i,'changed_elements':8,'total_elements':16,'clip_count':2,
            'mean_abs_error':1.,'max_abs_error':2.,'scale_min':.1,'scale_max':.1,'scale_mean':.1}


class ExplorationTests(unittest.TestCase):
    def test_bounded_history_exact_counts(self):
        s=EffectStats()
        for i in range(1,10001):s.append(event(i,i%2))
        raw=s.snapshot()
        self.assertEqual(raw['event_count'],10000)
        self.assertEqual(sum(m['changed_elements'] for m in raw['per_module'].values()),80000)
        self.assertEqual(len(list(s)),32)
        self.assertTrue(raw['contiguous_hit_indices'])
        s.clear();self.assertEqual(s.snapshot()['event_count'],0)

    def test_duplicate_index_detected(self):
        s=EffectStats();s.append(event(1));s.append(event(1))
        self.assertFalse(s.snapshot()['contiguous_hit_indices'])

    def fixture(self):
        r=fault_row(response('fixture'))
        w=r['fault_transaction']['measurement_query']['results'][0]
        s=EffectStats();s.append(event(1,0));s.append(event(2,1))
        w.update(effect_summary=s.snapshot(),hit_count=2,tensor_effect_history=list(s),
                 per_module={n:{'hit_count':1} for n in config().actual_module_names})
        cfg=config().to_dict();cfg['hit_policy']={'max_hits':2,'budget_semantics':'AT_MOST_NO_PADDING'}
        cfg['frozen_arm_config']['max_hits']=2
        for worker in r['fault_transaction']['measurement_query']['results']:
            worker['armed_config']['max_hits']=2
        for worker in r['fault_transaction']['arm']['results']:worker['armed_config']['max_hits']=2
        t=r['fault_transaction']['transaction_query'];t['results']=copy.deepcopy(r['fault_transaction']['measurement_query']['results'])
        return r,FaultConfigurationRecord.from_dict(cfg)

    def test_mechanism_accepts_explicit_aggregate(self):
        r,c=self.fixture()
        self.assertEqual(validate_mechanism(r,c,True)['status'],'PASS')

    def test_mechanism_rejects_missing_module(self):
        r,c=self.fixture();w=r['fault_transaction']['measurement_query']['results'][0]
        w['effect_summary']['per_module'].pop(c.actual_module_names[0])
        self.assertEqual(validate_mechanism(r,c,True)['status'],'INVALID')

    def test_mechanism_rejects_aggregate_miscount(self):
        r,c=self.fixture();w=r['fault_transaction']['measurement_query']['results'][0]
        w['effect_summary']['event_count']=3
        self.assertEqual(validate_mechanism(r,c,True)['status'],'INVALID')

    def test_mechanism_rejects_wrong_strength(self):
        r,c=self.fixture();w=r['fault_transaction']['measurement_query']['results'][0]
        w['effect_summary']['per_module'][c.actual_module_names[0]]['scale_multiplier']=20
        self.assertEqual(validate_mechanism(r,c,True)['status'],'INVALID')

    def test_no_numerical_change_not_valid(self):
        r,c=self.fixture()
        for m in r['fault_transaction']['measurement_query']['results'][0]['effect_summary']['per_module'].values():m['changed_elements']=0
        self.assertEqual(validate_mechanism(r,c,True)['status'],'INVALID')

    def test_window_requires_request_isolation(self):
        r,c=self.fixture()
        self.assertEqual(window_gate(r,c)['status'],'WINDOW_INCOMPLETE')

    def test_budget_not_negative_or_unbounded(self):
        rows=[{'module_name':f'model.layers.{i}.mlp.down_proj','logical_role':'MLP_DOWN','layer_index':i} for i in range(64)]
        original={'mode':'REFERENCE_REQUANT','max_hits':4096,'scale_multiplier':.25}
        e=make_experiment('A1',original,rows,ROLE_GROUPS[0],.25,1024,69)
        self.assertEqual(e['arm']['max_hits'],141952)
        self.assertEqual(original['max_hits'],4096)

    def test_missing_layer_is_not_discovery(self):
        with self.assertRaises(ValueError):make_experiment('x',{},[],ROLE_GROUPS[0],2,1024,69)

    def test_table_separator_is_not_confirmed_spam(self):
        row={'probe_id':'x','fault':{'full_text':'|---|---------------------|'},'effect':{'detectors':[
            {'detector':'punctuation','triggered':True,'fault_value':{'longest_run_character_span':[5,26]},'phenotype':'PUNCTUATION_SPAM'}]}}
        self.assertFalse(contextual_findings([row])[0]['confirmed_pattern'])

    def test_length_alone_not_confirmed(self):
        row={'probe_id':'x','fault':{'full_text':'more text'},'effect':{'detectors':[
            {'detector':'length','triggered':True,'fault_value':{},'phenotype':'RUNAWAY'}]}}
        self.assertFalse(contextual_findings([row])[0]['confirmed_pattern'])
