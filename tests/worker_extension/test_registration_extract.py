import json,tempfile,unittest
from pathlib import Path
from copy import deepcopy
from test_continuous_public_review import fixture
from tools.review_continuous_public import review
from tools.extract_registration_record import extract,verify,seal,canonical,sha,DEPLOYMENT


def filled():
    d=fixture();d.update(release_tag='demoncatpfi-rc3-preview-20261122-r5',source_commit='d4ae552df2b4fd82641bb39f234d2071dcccaed5')
    actual=dict(zip(DEPLOYMENT,['A3','Ascend910_9382','0.26.0','0.26.0rc1','2.10.0+cpu','2.10.0.post4',['9.1.0'],'sha256:'+'a'*64,'b'*64,'c'*64,{'quant_model_weights.safetensors.index.json':'d'*64},1024,1024,1,2,1,1,True,'torch.bfloat16',True]))
    for s in d['scopes']:
        s['negative_missing_permit_creation']=[dict(rank=r,prepare=dict(status='REJECTED'),snapshot=dict(ticket=None,ownership_retained=False),detach=dict(status='DETACHED')) for r in range(2)]
        s['cleanup']=[dict(rank=r,permit_remaining=False,runtime_leases=0,runtime_tenants=0) for r in range(2)]
        for host in s['requests']:
            for rank,state in host['result']['response']['provider_state']['rank_states'].items():
                state['deployment_acceptance_use'].update(registration_source='YELLOW_DEDICATED_INSTANCE',permit=dict(source_manifest_sha256='eff3ec54d292deec219555f32fc928278c30c23541e3ef794ef7f89f152a2fff'),actual=dict(actual,hardware_logical_device=int(rank)))
    return d

class RegistrationTests(unittest.TestCase):
    def run_extract(self,d,mutate=None):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);raw=canonical(d);summary=review(d,require_negative_creation=True);summary['source_receipt_sha256']=sha(raw)
            if mutate:mutate(summary)
            (p/'receipt.json').write_bytes(raw);(p/'summary.json').write_bytes(canonical(summary))
            return extract(p/'summary.json',p/'receipt.json')
    def test_complete_tp2_under_2k_and_reproducible(self):
        r=self.run_extract(filled());self.assertTrue(verify(r)['consistent']);self.assertLessEqual(len(canonical(r))+1,2048)
        self.assertEqual(r,self.run_extract(filled()))
    def test_missing_field_stays_null(self):
        d=filled()
        for s in d['scopes']:
            for h in s['requests']:
                for state in h['result']['response']['provider_state']['rank_states'].values():state['deployment_acceptance_use']['actual'].pop('image_digest')
        r=self.run_extract(d);self.assertIsNone(r['d'][7]);self.assertIn('IDENTITY_MISSING',verify(r)['registration_issues'])
    def test_summary_cannot_claim_different_coverage(self):
        r=self.run_extract(filled(),lambda s:s.update(passed=False));self.assertFalse(r['review'][2]);self.assertFalse(verify(r)['consistent'])
    def test_invalid_raw_local_review_not_hidden(self):
        d=filled()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);s=review(d,require_negative_creation=True);d['scopes'][1]['negative_extra_advance']='OK'
            (p/'r').write_bytes(canonical(d));(p/'s').write_bytes(canonical(s));r=extract(p/'s',p/'r')
            self.assertFalse(r['review'][0]);self.assertIn('EXTRA_ADVANCE',r['review'][1]);self.assertFalse(verify(r)['consistent'])
    def test_tampering_and_hard_limit(self):
        r=self.run_extract(filled());r['s'][0][1][0][1][3]+=1
        with self.assertRaisesRegex(ValueError,'HASH'):verify(r)
        self.assertFalse(verify(seal(r))['consistent'])
        with self.assertRaisesRegex(ValueError,'4096'):seal(dict(text='x'*5000))
    def test_conflicting_rank_deployment_never_collapsed(self):
        d=filled();d['scopes'][1]['requests'][-1]['result']['response']['provider_state']['rank_states']['1']['deployment_acceptance_use']['actual']['cann']=['8.5.1']
        r=self.run_extract(d);self.assertIn('deployment:cann',r['conflicts']);self.assertIsNone(r['d'][6])

class DeterministicImageTests(unittest.TestCase):
    def test_product_canary_has_same_default_without_file(self):
        from demoncat.runtime.vllm_ascend.worker_extension.product_config import product_canary_image
        from fault_injection.vision_input import deterministic_image
        a,identity=product_canary_image({'image':None});b,other=deterministic_image()
        self.assertEqual(a.tobytes(),b.tobytes());self.assertEqual(identity,other)
    def test_frozen_reviewer_exact_r5_bytes(self):
        import tarfile
        root=Path(__file__).resolve().parents[2]
        with tarfile.open(root/'releases/demoncatpfi-rc3-preview-20261122-r5/demoncatpfi-rc3-preview-20261122-r5.tar.gz') as t:
            original=t.extractfile('tools/review_continuous_public.py').read()
        self.assertEqual((root/'tools/_r5/review_continuous_public.py').read_bytes(),original)
        self.assertEqual(sha(original),'deb3c5c0d600f7bcae7d0997cfd883591ece87231ed8128e486edb5d823ffadc')
    def test_exact_pixels_shared_fixture(self):
        from fault_injection.vision_input import deterministic_image
        a,identity=deterministic_image();b,other=deterministic_image()
        self.assertEqual(identity,other);self.assertEqual(a.tobytes(),b.tobytes())
        self.assertEqual(a.getpixel((63,63)),(252,252,248));self.assertEqual(identity['pixel_sha256'],sha(a.tobytes()))


class DeepSeekExtractTests(unittest.TestCase):
    def test_calibration_per_rank_pages_preserve_all_bounds(self):
        from test_dsv4_noise import combo,data
        from fault_injection.dsv4_noise import calibrate
        from tools.extract_dsv4_record import pages
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'noise.json';r=calibrate(combo(),data());p.write_bytes(canonical(r));out=pages('noise',p)
        self.assertEqual(len(out),8)
        for i,page in enumerate(out):
            self.assertLessEqual(len(canonical(page))+1,4096);self.assertEqual(page['payload']['rank'],i)
            self.assertTrue(page['payload']['finite']);self.assertTrue(page['payload']['token_stable'])
            self.assertEqual(len(page['payload']['bounds']),4)
    def test_observation_ineligible_and_missing_source_preserved(self):
        from tools.extract_dsv4_record import pages
        d=dict(schema='demoncat.dsv4_0731.observe.v1',rank=0,read_only=True,injected_calls=0,mechanisms={'8':dict(observed=0,eligible=0,sites=[],unavailable_reason='NOT_OBSERVED')})
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'observe.json';p.write_bytes(canonical(d));out=pages('observe',p)
            self.assertEqual(out[0]['payload']['unavailable_reason'],'NOT_OBSERVED');self.assertIsNone(out[0]['payload']['observer_removed'])
            p.write_bytes(canonical(dict(rank=0,files=[dict(path='vllm_ascend/a.py',sha256=None)])))
            self.assertIsNone(pages('observe',p)[0]['payload']['source_file']['sha256'])
