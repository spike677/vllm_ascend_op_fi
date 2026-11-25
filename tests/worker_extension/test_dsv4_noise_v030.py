import copy,json,tempfile,unittest
from pathlib import Path
from fault_injection import dsv4_noise as old, dsv4_noise_v030 as new
from tools.extract_dsv4_v030_record import pages
from tools.extract_registration_record import canonical,sha
from test_dsv4_noise import combo,data,phases

def actual():
    c=combo();c.update(new.TARGET,vllm='0.30.0+empty',ep=False,
                      ep_source='parallel_config.enable_expert_parallel');return c

class V030NoiseTests(unittest.TestCase):
    def test_same_numerical_rule_separate_identity(self):
        a=old.review(combo(),data(),phases());b=new.review(actual(),data(),phases())
        self.assertEqual(a['comparisons'],b['comparisons'])
        self.assertEqual(a['calibration']['bounds'],b['calibration']['bounds'])
        self.assertNotEqual(a['schema'],b['schema']);self.assertFalse(b['qualification_granted'])
        self.assertFalse(b['calibration']['combination']['ep'])
        with self.assertRaisesRegex(ValueError,'COMBINATION'):old.calibrate(actual(),data())
        with self.assertRaisesRegex(ValueError,'COMBINATION'):new.calibrate(combo(),data())

    def test_runtime_ep_is_never_assumed(self):
        for key,value in [('ep',None),('ep',1),('ep_source',''),('vllm','0.32.0')]:
            c=actual();c[key]=value
            with self.assertRaises(ValueError):new.calibrate(c,data())
        c=actual();c['ep']=True
        self.assertTrue(new.calibrate(c,data())['combination']['ep'])

    def test_finite_samples_and_storage_remain_strict(self):
        d=data();d[0]['ranks']['7'][0]['hidden'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'FINITE'):new.calibrate(actual(),d)
        with self.assertRaisesRegex(ValueError,'K5'):new.calibrate(actual(),data()[:9])
        p=phases();p['off']['region_bytes_restored']=False
        self.assertFalse(new.review(actual(),data(),p)['off_health_passed'])

    def test_bounded_deterministic_text_keeps_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'noise.json';path.write_text(json.dumps(new.calibrate(actual(),data())))
            a=pages('noise',path);self.assertEqual(a,pages('noise',path));self.assertEqual(len(a),8)
            for r in a:
                self.assertLessEqual(len(canonical(r))+1,4096)
                self.assertIsNone(r['payload']['identity_error'])
                self.assertEqual(r['payload']['combination']['vllm'],'0.30.0+empty')
                x=copy.deepcopy(r);digest=x.pop('extract_sha256');self.assertEqual(sha(canonical(x)),digest)
            path.write_text(json.dumps(old.calibrate(combo(),data())))
            with self.assertRaisesRegex(ValueError,'SCHEMA'):pages('noise',path)
