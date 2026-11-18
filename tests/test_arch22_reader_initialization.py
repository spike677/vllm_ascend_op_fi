"""Cold-process CPU readers; no framework or Native observer is imported."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT=ROOT/'experiments/c2c3_closure_20261117'


class ColdReaderContracts(unittest.TestCase):
    def test_cold_numeric_readers_work_then_dynamic_load_and_torch_remain_refused(self):
        code=r'''
import hashlib,json,struct,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import review_arch22_normal_bridge as review
import arch22_normal_bridge_evidence as evidence
assert 'numpy' not in sys.modules
identity=review.initialize_cpu_byte_reader()
sys.addaudithook(evidence.prohibit_runtime)
helpers=evidence.helpers({'root':Path(sys.argv[2])})
run=Path(sys.argv[3]);left=run/'left.bin';right=run/'right.bin'
left.write_bytes(struct.pack('<4e',0.0,-0.0,1.5,-2.0))
right.write_bytes(struct.pack('<4e',-0.0,0.0,2.0,-2.0))
def row(path):
 return dict(path='/evidence/run/'+path.name,dtype='torch.float16',shape=[4],
             finite=True,bytes=8,sha256=evidence.sha(path))
values=[helpers.bridge.raw_tensor(run,row(p),[4]) for p in (left,right)]
snapshot=dict(metadata=dict(dtype='torch.float16',shape=[4],stride=[1],storage_offset=0,
    storage_nbytes=8,object_id=1,storage_identity=1,device='npu:0'),
    bytes=8,sha256=evidence.sha(left),finite=True,path='/evidence/run/left.bin')
assert helpers.boundary.raw_snapshot(run,snapshot)['sha256']==evidence.sha(left)
difference=helpers.differences.raw_difference(*values)
assert difference['changed_elements']==3 and difference['numeric_changed_elements']==1
assert difference['max_abs_difference']==.5 and difference['exact'] is False
bad=run/'bad.bin';bad.write_bytes(struct.pack('<H',0x7f80))
try:
 helpers.bridge.raw_tensor(run,dict(path='/evidence/run/bad.bin',dtype='torch.bfloat16',
     shape=[1],finite=True,bytes=2,sha256=evidence.sha(bad)),[1])
except RuntimeError as error:assert 'NONFINITE' in str(error)
else:raise AssertionError('nonfinite bytes accepted')
try:
 import ctypes
 ctypes.CDLL('cpu-audit-must-not-load-library.invalid')
except RuntimeError as error:assert 'READ_ONLY_CPU_ACTION_REFUSED:ctypes.dlopen' in str(error)
else:raise AssertionError('dynamic loading was not rejected')
try:__import__('torch')
except RuntimeError as error:assert 'DEVICE_FRAMEWORK_IMPORT_REFUSED:torch' in str(error)
else:raise AssertionError('framework import was not rejected')
assert not any(name.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend') for name in sys.modules)
print(json.dumps(identity))
'''
        with tempfile.TemporaryDirectory() as folder:
            result=subprocess.run([sys.executable,'-B','-c',code,str(EXPERIMENT),str(ROOT),folder],
                                  capture_output=True,text=True,timeout=45)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        identity=json.loads(result.stdout)
        self.assertEqual(identity['actual_npu_calls'],0)
        self.assertFalse(identity['device_frameworks_imported'])
        self.assertTrue(identity['initialized_before_no_dynamic_load_audit'])
        self.assertTrue(identity['version'])
        self.assertEqual(len(identity['module_sha256']),64)
        self.assertEqual(len(identity['core_sha256']),64)

    def test_preloaded_device_framework_refuses_before_numpy_initialization(self):
        code=r'''
import sys
sys.path.insert(0,sys.argv[1])
import review_arch22_normal_bridge as review
sys.modules['torch']=object() # Explicit sentinel, never imports torch.
try:review.initialize_cpu_byte_reader()
except ValueError as error:assert 'DEVICE_FRAMEWORK_ALREADY_IMPORTED' in str(error)
else:raise AssertionError('ambient framework accepted')
assert 'numpy' not in sys.modules
print('refused-before-numpy')
'''
        result=subprocess.run([sys.executable,'-B','-c',code,str(EXPERIMENT)],
                              capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(result.stdout.strip(),'refused-before-numpy')


if __name__=='__main__':unittest.main()
