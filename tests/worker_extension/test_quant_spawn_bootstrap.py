import hashlib,json,os,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from experiments.worker_core_v11_20261118 import quant_spawn_bootstrap as b

class SpawnVendorTests(unittest.TestCase):
    def run_fixture(self, fail=False, foreign=False):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();stock=root/'stock';private=root/'quant/runtime/vendor';vendor=stock/'vendor'
            for d in (private/'op_api/lib',vendor/'op_api/lib'):d.mkdir(parents=True)
            (stock/'platform.py').write_bytes(b'platform');(vendor/'op_api/lib/libcust_opapi.so').write_bytes(b'stock')
            (private/'op_api/lib/libcust_opapi.so').write_bytes(b'private')
            manifest=root/'quant/build.json';manifest.write_text(json.dumps({'library':'runtime/vendor/op_api/lib/libcust_opapi.so'}))
            calls=[]
            def initialize():
                calls.append('original');os.environ['ASCEND_CUSTOM_OPP_PATH']=str(vendor)+os.pathsep+os.environ['ASCEND_CUSTOM_OPP_PATH']
            fake=NS(current_platform=NS(import_kernels=initialize))
            original=str(private)+os.pathsep+str(vendor)+((os.pathsep+'foreign') if foreign else '')
            pins={'platform.py':b._sha(stock/'platform.py'),'vendor/op_api/lib/libcust_opapi.so':b._sha(vendor/'op_api/lib/libcust_opapi.so')}
            with patch.object(b,'STOCK_ROOT',stock),patch.object(b,'STOCK_VENDOR',vendor),patch.object(b,'STOCK_FILES',pins),patch.dict(sys.modules,{'vllm.platforms':fake}),patch.dict(os.environ,{'ASCEND_CUSTOM_OPP_PATH':original}):
                if foreign:
                    with self.assertRaisesRegex(RuntimeError,'UNKNOWN_VENDOR'): 
                        with b.worker_spawn_vendor_order(root/'quant',b._sha(manifest)):self.fail('unreviewed path accepted')
                    return
                caught=False
                try:
                    with b.worker_spawn_vendor_order(root/'quant',b._sha(manifest)) as record:
                        self.assertEqual(os.environ['ASCEND_CUSTOM_OPP_PATH'],str(private)+os.pathsep+str(vendor))
                        if fail:raise ValueError('constructor failed')
                except ValueError:caught=True
                self.assertEqual(caught,fail);self.assertEqual(calls,['original'])
                self.assertEqual(os.environ['ASCEND_CUSTOM_OPP_PATH'],str(vendor)+os.pathsep+original)
                self.assertTrue(record['restored']);self.assertFalse(record['worker_class_patched'])
    def test_normal(self):self.run_fixture()
    def test_constructor_failure_restores(self):self.run_fixture(fail=True)
    def test_foreign_vendor_refused(self):self.run_fixture(foreign=True)

if __name__=='__main__':unittest.main()
