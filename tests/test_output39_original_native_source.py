"""Old mutation bodies stay fixed; additive diagnostic tested on CPU stubs."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from experiments.worker_core_v11_20261118.output39_refusal_diagnostic import OriginalInfo
import ctypes


NATIVE = Path(__file__).resolve().parents[1]/'fault_injection/native_adapters/postwrite'
FROZEN = {
    'dcat_pw_open':'aa413569aca4fd897dad3f4de49dd495d95ff7d1eb9ac4bfa90eb5579eb9abc8',
    'dcat_pw_apply':'e4e7711b4367e02852ebd79abe14b7c35c8d8d85230dc00d2a713288b5225b76',
    'dcat_pw_restore':'f4f049d7c185ed9ecaffe93dcee0277652fc4368cc7f30a0fc115720931a56c3',
    'dcat_pw_release':'6738457833024689e4633bf5843eb4ba9baeca459bd6f8e40e817feb7a02d602',
    'dcat_pw_live':'8dbaef24f2bb34cd0187d6c9e9ec0420387a153533f01ef2b94a20b46030c485',
    'dcat_pw_read_span':'1d66f093dd5e33758de23687e8f0580c546e29b448e2d24c475fd2323591db96',
}


class OriginalSourceTests(unittest.TestCase):
    def test_frozen_old_mutation_transfer_and_recovery_bodies(self):
        source = (NATIVE/'postwrite.cpp').read_text()
        for name, expected in FROZEN.items():
            with self.subTest(name=name):
                begin = source.index('extern "C" int '+name+'(')
                end = source.index('\n}', begin)+2
                self.assertEqual(hashlib.sha256(source[begin:end].encode()).hexdigest(), expected)

    def test_additive_information_layout_is_fixed(self):
        self.assertEqual(ctypes.sizeof(OriginalInfo), 64)
        self.assertEqual(OriginalInfo.span_bytes.offset, 48)
        self.assertEqual(OriginalInfo.device.offset, 56)

    @unittest.skipUnless(shutil.which('g++') or shutil.which('clang++'), 'Host C++ compiler unavailable; parent runs CPU stub harness')
    def test_actual_native_implementation_on_explicit_host_only_acl_stubs(self):
        with tempfile.TemporaryDirectory(prefix='dcat39-original-cpu-') as directory:
            work = Path(directory)
            (work/'acl').mkdir()
            (work/'acl/acl.h').write_text('''#pragma once
#include <stddef.h>
#include <stdint.h>
typedef int aclError;
enum aclrtMemcpyKind { ACL_MEMCPY_HOST_TO_DEVICE=1, ACL_MEMCPY_DEVICE_TO_HOST=2 };
#define ACL_SUCCESS 0
extern "C" {
aclError aclrtGetDevice(int32_t*);
aclError aclrtSynchronizeDevice();
aclError aclrtMemcpy(void*, size_t, const void*, size_t, aclrtMemcpyKind);
}
''')
            compiler = shutil.which('g++') or shutil.which('clang++')
            output = work/'test-original.exe'
            subprocess.run([compiler, '-std=c++17', '-pthread', '-Wall', '-Wextra', '-Werror',
                '-I'+str(work), str(NATIVE/'test_original_snapshot.cpp'), '-o', str(output)],
                check=True, capture_output=True, text=True, timeout=60)
            result = subprocess.run([str(output)], check=True, capture_output=True, text=True, timeout=20)
            self.assertIn('PASS_CPU_LEASE_ORIGINAL_DIAGNOSTIC', result.stdout)


if __name__ == '__main__':
    unittest.main()
