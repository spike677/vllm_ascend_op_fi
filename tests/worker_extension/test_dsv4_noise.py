import unittest,copy,json
from fault_injection.dsv4_noise import TARGET,calibrate,review

def combo():return dict(TARGET,model_config_sha256='a'*64,quant_model_description_sha256='b'*64,runtime_source_sha256='c'*64,safetensors_index_sha256={'weights.safetensors.index.json':'d'*64},max_model_len=4096,max_num_batched_tokens=4096,max_num_seqs=1)
def run(process,sample,delta=0):
    step=dict(hidden=[1+delta,2],logits=[3+delta,4],top5=[dict(token=i,logprob=-i+delta) for i in range(5)])
    return dict(process_identity=process,sample_id=str(sample),prompt_tokens=2500,prompt_sha256='e'*64,single_chunk=True,tokens=[1,2,3,4],ranks={str(i):[copy.deepcopy(step) for _ in range(4)] for i in range(8)},region_bytes_restored=True,resources=dict(leases=0,hooks=0,inflight=0))
def data():return [run(p,i,i*.1) for p in ['p1','p2'] for i in range(5)]
def phases():return {p:run('accept',i,.7 if p=='fault' else .5 if p in ('off','health') else 0) for i,p in enumerate(['normal','fault','off','health'])}

class NoiseTests(unittest.TestCase):
    def test_fixed_rule_and_recovery(self):
        r=review(combo(),data(),phases());self.assertEqual(r['calibration']['safety_factor'],1.5)
        self.assertAlmostEqual(r['calibration']['bounds']['0'][0]['hidden_max_abs'],.6)
        self.assertTrue(r['off_health_passed']);self.assertTrue(r['model_effect_visible']);self.assertFalse(r['qualification_granted'])
        self.assertLess(len(json.dumps(r).encode()),60000)
    def test_fault_inside_noise_is_site_only(self):
        p=phases();p['fault']=run('accept','fault',.2)
        self.assertFalse(review(combo(),data(),p)['model_effect_visible'])
    def test_qwen_cannot_use_noise(self):
        c=combo();c['model']='Qwen3.8'
        with self.assertRaisesRegex(ValueError,'COMBINATION'):calibrate(c,data())
    def test_calibration_requires_two_processes_real_samples(self):
        d=data();d[-1]=copy.deepcopy(d[-2])
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):calibrate(combo(),d)
        with self.assertRaisesRegex(ValueError,'K5'):calibrate(combo(),data()[:9])
    def test_tokens_unstable_or_missing_rank_refuse(self):
        d=data();d[-1]['tokens'][0]=99
        with self.assertRaisesRegex(ValueError,'TOKENS_UNSTABLE'):calibrate(combo(),d)
        d=data();d[-1]['ranks'].pop('7')
        with self.assertRaisesRegex(ValueError,'ALL_RANKS'):calibrate(combo(),d)
    def test_storage_recovery_never_relaxed(self):
        p=phases();p['fault']['region_bytes_restored']=False
        self.assertFalse(review(combo(),data(),p)['off_health_passed'])
    def test_off_token_or_noise_failure(self):
        p=phases();p['off']['tokens'][0]=99
        self.assertFalse(review(combo(),data(),p)['off_health_passed'])
        p=phases();p['health']=run('accept','health',.9)
        self.assertFalse(review(combo(),data(),p)['off_health_passed'])
    def test_nonfinite_uncalibrated_prompt_or_changed_shape(self):
        d=data();d[0]['ranks']['0'][0]['hidden'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'FINITE'):calibrate(combo(),d)
        p=phases();p['off']['prompt_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'UNCALIBRATED'):review(combo(),data(),p)
        p=phases();p['off']['ranks']['0'][0]['hidden'].append(2)
        with self.assertRaisesRegex(ValueError,'SHAPE'):review(combo(),data(),p)
    def test_new_normal_does_not_expand_bounds(self):
        p=phases();p['normal']=run('accept','normal',3)
        with self.assertRaisesRegex(ValueError,'NORMAL_OUTSIDE'):review(combo(),data(),p)

    def test_fault_nonfinite_is_explicit_and_recovery_remains_finite(self):
        p=phases();p['fault']['ranks']['0'][0]['logits'][0]='+inf'
        result=review(combo(),data(),p)
        self.assertTrue(result['model_effect_visible']);self.assertTrue(result['off_health_passed'])
        self.assertIsNone(result['comparisons']['fault']['max_difference']['logits_max_abs'])
        json.dumps(result,allow_nan=False)
        p['off']['ranks']['0'][0]['logits'][0]='nan'
        with self.assertRaisesRegex(ValueError,'FINITE'):review(combo(),data(),p)
