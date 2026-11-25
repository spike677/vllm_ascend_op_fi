import hashlib,io,json,tarfile,tempfile,unittest
from pathlib import Path
from tools.build_lm_head_delivery import build
from test_reference_lm_head_review import fixture

class LmHeadDeliveryTests(unittest.TestCase):
    def test_existing_delivery_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'sentinel').write_bytes(b'frozen')
            with self.assertRaisesRegex(ValueError,'IMMUTABLE'):build(p,'absent.tar.gz',p)
            self.assertEqual((p/'sentinel').read_bytes(),b'frozen')

    def test_incomplete_or_unrestored_window_is_not_deliverable(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for status,restored in [('RUNNING',False),('CASE_FAILED',True),('FINISHED',False)]:
                (p/'controller.json').write_text(json.dumps(dict(status=status,isolation_removed=restored)))
                with self.assertRaisesRegex(ValueError,'A2_WINDOW_NOT_FINISHED'):
                    build(p,'absent.tar.gz',p/'out')
                self.assertFalse((p/'out').exists())

    def test_old_owner_review_error_cannot_hide_real_recovery_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);m=json.dumps(dict(base_commit='a'*40,files=[])).encode();arc=p/'source.tar'
            with tarfile.open(arc,'w') as t:
                i=tarfile.TarInfo('source_manifest.json');i.size=len(m);t.addfile(i,io.BytesIO(m))
            cases=[]
            for variant in ('collapse','swap'):
                name='attempt-row12-lm-head-'+variant;folder=p/name;folder.mkdir()
                cases.append(dict(id=name,exit_code=1,review=dict(source_manifest_sha256=hashlib.sha256(m).hexdigest())))
                d=fixture();d['scopes'][0]['disable']['recovery']['bytes_restored']=False
                raw=json.dumps(d).encode();(folder/'persistent_product.json').write_bytes(raw)
                (folder/'independent-review.json').write_text(json.dumps(dict(passed=False,issues=['OWNER_MISMATCH'],receipt_sha256=hashlib.sha256(raw).hexdigest())))
            (p/'controller.json').write_text(json.dumps(dict(status='CASE_FAILED',isolation_removed=True,cases=cases)))
            with self.assertRaisesRegex(ValueError,'A2_INDEPENDENT_REVIEW_FAILED'):
                build(p,arc,p/'out')
            self.assertFalse((p/'out').exists())
