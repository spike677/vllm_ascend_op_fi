"""CPU loader doubles retain exact file/ABI failures and structured values."""
from pathlib import Path
from unittest.mock import patch
import ctypes,json,os,tempfile,unittest
from fault_injection.deployments.postwrite import NativePostwriteLibrary,PostwriteRefused
from fault_injection import quant_group_native as q
from demoncat.runtime.vllm_ascend.worker_extension.runtime import _provider_error
import test_quant_group_binding as fixtures

class Function:
    def __init__(self,value=0):self.value=value
    def __call__(self,*args):return self.value
class PostwriteDouble:
    def __init__(self,abi):
        for name in ('abi','open','apply','restore','release','live'):
            setattr(self,'dcat_pw_'+name,Function(abi if name=='abi' else 0))

class NativePrerequisiteTests(unittest.TestCase):
    def diagnostic(self,error,field,expected,actual):
        report=_provider_error(error)['prerequisites']
        self.assertEqual(report['primary']['layer'],'deployment_identity')
        row=next(x for x in report['checks'] if x['field'].endswith(field))
        self.assertEqual(row['expected'],expected);self.assertEqual(row['actual'],actual)
        self.assertFalse(report['execution_authorized'])
    def test_postwrite_sha_before_cdll_and_abi_actual_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'fixture.so';p.write_bytes(b'not executable CPU DSO fixture')
            digest=q._sha(p)
            with patch('ctypes.CDLL') as load:
                with self.assertRaises(PostwriteRefused) as caught:NativePostwriteLibrary(p,'a'*64)
                load.assert_not_called();self.diagnostic(caught.exception,'library_sha256','a'*64,digest)
            with patch('ctypes.CDLL',return_value=PostwriteDouble(7)):
                with self.assertRaises(PostwriteRefused) as caught:NativePostwriteLibrary(p,digest)
                self.diagnostic(caught.exception,'abi',1,7)
    def test_postwrite_missing_file_and_missing_symbols_are_specific(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'fixture.so'
            with self.assertRaisesRegex(PostwriteRefused,'LIBRARY_FILE_REQUIRED') as caught:NativePostwriteLibrary(p,'a'*64)
            self.assertIn('prerequisites',_provider_error(caught.exception))
            p.write_bytes(b'CPU fixture')
            with patch('ctypes.CDLL',return_value=object()):
                with self.assertRaisesRegex(PostwriteRefused,'SYMBOLS_MISSING') as caught:NativePostwriteLibrary(p,q._sha(p))
            self.assertIn('dcat_pw_abi',caught.exception.prerequisites['primary']['actual']['missing'])
    def test_quant_aggregates_source_and_environment_mismatches_before_load(self):
        fixture=fixtures.QuantGroupProfileTests()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);approved,real=fixture.bundle(root,prior_source=True)
            with patch.object(q.isolation,'mapped_native_modes',return_value=()),patch.object(q.C,'CDLL') as load,patch.dict(os.environ,{'DCAT_REAL_OPAPI':'','DCAT_DQ_GROUP_V1_ALLOWED':'0','VLLM_SERVER_DEV_MODE':'0'}):
                with self.assertRaises(q.UnsupportedCall) as caught:q.QuantGroupNative.load(root,approved_build_sha256=approved)
            load.assert_not_called()
            report=_provider_error(caught.exception)['prerequisites']
            self.assertIn('SOURCE_DOES_NOT_MATCH',report['primary']['code'])
            self.assertTrue(any(x['field'].endswith('DCAT_DQ_GROUP_V1_ALLOWED') for x in report['unmet']))
            self.assertTrue(any(x['field'].endswith('real_opapi_absolute_file') for x in report['unmet']))
    def test_quant_missing_sdk_path_cannot_match_two_none_values(self):
        fixture=fixtures.QuantGroupProfileTests()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,_=fixture.bundle(root)
            p=root/'build.json';d=json.loads(p.read_bytes());d['sdk_identity'].pop('real_opapi');d['sdk_identity'].pop('real_opapi_sha256')
            p.write_text(json.dumps(d))
            with patch.dict(os.environ,{'DCAT_REAL_OPAPI':'','DCAT_DQ_GROUP_V1_ALLOWED':'1','VLLM_SERVER_DEV_MODE':'1'}):
                with self.assertRaises(q.UnsupportedCall):q.inspect_bundle(root,q._sha(p))
    def test_host_binding_failure_retains_structured_report(self):
        from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter,WorkerFaultClient
        from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
        from test_core_host import ExistingExecutor
        from fault_injection.prerequisites import require_identity
        client=WorkerFaultClient(HostAdapter(ExistingExecutor([])),FaultConfig(39,'output_proxy'))
        try:require_identity(False,'postwrite','abi',1,7,'POSTWRITE_ABI_MISMATCH',PostwriteRefused)
        except PostwriteRefused as error:
            result=client._failure('prepare_binding',error,uncertain=True)
        self.assertEqual(result['response']['prerequisites']['primary']['actual'],7)
        self.assertEqual(result['response']['provider_code'],'POSTWRITE_ABI_MISMATCH')

    def test_quant_actual_ctypes_dso_sdk_mismatch_values(self):
        wrong=dict(fixtures.SDK_A3_SIZES);wrong['view']+=8
        with self.assertRaisesRegex(q.UnsupportedCall,'LAYOUT_DIFFERS') as caught:
            q.QuantGroupNative(fixtures.FakeLibrary(),dict(sdk_abi_sizes=wrong,sdk_metadata_abi_sizes=fixtures.METADATA_MOCK_SIZES),_token=q._VERIFIED_LOAD)
        primary=_provider_error(caught.exception)['prerequisites']['primary']
        self.assertEqual(primary['expected']['sdk'],wrong)
        self.assertEqual(primary['actual'],fixtures.SDK_A3_SIZES)

if __name__=='__main__':unittest.main()
