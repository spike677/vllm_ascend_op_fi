import copy,json,subprocess,unittest
from pathlib import Path
from tools.generate_native_build_metadata import generate

class MetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[2];folder=root/'docs/worker_core_v1/r7_native_a2_20261123'
        cls.receipt=(folder/'build.json').read_bytes();cls.context=json.loads((folder/'build_context.json').read_bytes())
        read=lambda p:subprocess.check_output(['git','show','82c5b7f8:'+p],cwd=root)
        cls.kw=dict(commit='82c5b7f81b0155cec958aac9535a20209beb75d4',release='demoncatpfi-rc3-preview-20261123-r6-5-g82c5b7f8',
            recipe=read('tools/build_native_regions.py'),sources={n:read('fault_injection/native_adapters/postwrite/'+n) for n in json.loads(cls.receipt)['sources']})
    def test_exact_audited_sidecar(self):
        result=generate(self.receipt,self.context,**self.kw)
        expected=json.loads((Path(__file__).resolve().parents[2]/'docs/worker_core_v1/native_build_metadata_d9aa_20261126/dcat-native-build.json').read_bytes())
        self.assertEqual(result,expected);self.assertEqual(result['cann'],'9.1.0')
    def test_receipt_tampering_rejected(self):
        with self.assertRaisesRegex(ValueError,'RECEIPT_SHA'):generate(self.receipt+b' ',self.context,**self.kw)
    def test_source_tampering_rejected(self):
        kw=copy.deepcopy(self.kw);kw['sources']['postwrite.cpp']+=b' '
        with self.assertRaisesRegex(ValueError,'SOURCE_FILES'):generate(self.receipt,self.context,**kw)
    def test_recipe_tampering_rejected(self):
        kw=dict(self.kw,recipe=self.kw['recipe']+b' ')
        with self.assertRaisesRegex(ValueError,'RECIPE_SHA'):generate(self.receipt,self.context,**kw)
    def test_native_registration_fixture_needs_matching_metadata(self):
        import hashlib
        from demoncat.runtime.vllm_ascend.worker_extension.product_registration import validate_identity
        root=Path(__file__).resolve().parents[2]/'docs/worker_core_v1'
        # CPU-only fixture; this does not assert that yellow has installed the sidecar.
        a=json.loads((root/'r8_1_expected_ranks_draft_20261126.json').read_bytes())['records'][0]['expected_ranks'][0]
        raw=(root/'native_build_metadata_d9aa_20261126/dcat-native-build.json').read_bytes();bm=json.loads(raw)
        a['native_identity']=dict(applicable=True,sha256=bm['library_sha256'],declared_sha256=bm['library_sha256'],build_metadata=None,build_metadata_sha256=None)
        with self.assertRaisesRegex(ValueError,'NATIVE_BUILD_REGISTRATION_PENDING'):validate_identity(a)
        a['native_identity'].update(build_metadata=bm,build_metadata_sha256=hashlib.sha256(raw).hexdigest())
        validate_identity(a)
        bm['cann']='9.2.1'
        with self.assertRaisesRegex(ValueError,'NATIVE_BUILD_REGISTRATION_PENDING'):validate_identity(a)
