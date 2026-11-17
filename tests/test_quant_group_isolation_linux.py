"""Linux CPU-only mapped-ELF contracts; tiny marker fixtures are not CANN DSOs."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


CHILD = r'''
import ctypes as C,hashlib,json,os,sys
from pathlib import Path
from types import SimpleNamespace as NS
from fault_injection import quant_native_isolation as isolation
from fault_injection.quantization import PinnedQuantRuntime,_LEASE
from fault_injection.quant_group_native import QuantGroupNative
from fault_injection.core import UnsupportedCall
mode,path=sys.argv[1],Path(sys.argv[2])
calls=[]
old=PinnedQuantRuntime.__new__(PinnedQuantRuntime)
old.backend=NS(arm_validated=lambda *a,**k:calls.append(True));old.poisoned=False
library=C.CDLL(str(path),mode=C.RTLD_LOCAL)
if mode=='deleted':
    path.unlink()
    try:isolation.require_mode('legacy')
    except UnsupportedCall as error:assert 'MAPPING_DELETED' in str(error)
    else:raise AssertionError('deleted mapping was accepted')
else:
    expected='legacy' if mode=='old_first' else 'group'
    rows=isolation.mapped_native_modes()
    actual=[row for row in rows if row['mode']==expected]
    assert len(actual)==1 and actual[0]['path']==str(path.resolve())
    assert actual[0]['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual[0]['inode']==path.stat().st_ino and actual[0]['symbols']
    try:
        if mode=='old_first':QuantGroupNative.load('/missing',approved_build_sha256='a'*64)
        elif mode=='group_first':PinnedQuantRuntime.from_existing_backend('/missing',context_provider=lambda c:{},approved_build_sha256='a'*64)
        else:old.arm(NS(input=object()),fault=True)
    except UnsupportedCall as error:assert 'CANNOT_COLOAD' in str(error),str(error)
    else:raise AssertionError('opposite managed mode was admitted')
    assert not calls
assert not _LEASE.locked()
print(json.dumps({'case':mode,'status':'PASS_REAL_ELF_MAPPING_CPU_ONLY','actual_npu_calls':0}))
'''


@unittest.skipUnless(sys.platform == 'linux' and shutil.which('g++'), 'Actual ELF mapping requires Linux host g++')
class QuantMappedElfTests(unittest.TestCase):
    def test_symbol_owner_and_managed_modes_without_basename_assumptions(self):
        with tempfile.TemporaryDirectory(prefix='quant-mode-cpu-') as temporary:
            root = Path(temporary)
            code = {'legacy': 'extern "C" int dcat_dq_arm(){return 0;} extern "C" int dcat_dq_finish(){return 0;}\n',
                    'group': 'extern "C" int dcat_dqg_abi_v1(){return 0;} extern "C" int dcat_dqg_state_v1(){return 0;}\n'}
            libraries = {}
            for mode, value in code.items():
                source, library = root / (mode + '.cpp'), root / ('arbitrary-' + mode + '.data')
                source.write_text(value)
                subprocess.run(['g++', '-shared', '-fPIC', '-Wall', '-Wextra', '-Werror', str(source), '-o', str(library)],
                               check=True, capture_output=True, text=True, timeout=60)
                libraries[mode] = library
            for case, mode in (('old_first', 'legacy'), ('group_first', 'group'), ('cached_old_object', 'group'), ('deleted', 'group')):
                with self.subTest(case=case):
                    result = subprocess.run([sys.executable, '-c', CHILD, case, str(libraries[mode])],
                                            capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(json.loads(result.stdout)['status'], 'PASS_REAL_ELF_MAPPING_CPU_ONLY')


if __name__ == '__main__':
    unittest.main()
