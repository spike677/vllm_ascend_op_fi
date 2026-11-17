import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from engineering_closure.native_observer import (PinnedNativeLibrary, SymbolBinding,
    preload_environment, symbol_entry_status)


class LoaderTests(unittest.TestCase):
    def test_load_evidence_alone_never_confirms_an_entry(self):
        binding = SymbolBinding('op', 4, '/test', 'a' * 64)
        self.assertEqual(symbol_entry_status(binding, [vars(binding)]), 'NOT_VISIBLE')
        self.assertEqual(symbol_entry_status(binding, [dict(vars(binding), actual_function_entry=True)]), 'OBSERVED')

    @unittest.skipUnless(sys.platform == 'linux', 'Linux ELF mapping required')
    def test_existing_libc_symbol_matches_loaded_image(self):
        paths = {line.split(maxsplit=5)[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                 if line.endswith('/libc.so.6')}
        path, = map(Path, paths)
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        library = PinnedNativeLibrary(path, sha256)
        binding = library.bind('getpid')
        self.assertEqual(binding.library_path, str(path.resolve()))
        self.assertEqual(library.handle.getpid(), os.getpid())
        with self.assertRaisesRegex(RuntimeError, 'NATIVE_LIBRARY_IDENTITY_CHANGED'):
            PinnedNativeLibrary(path, '0' * 64)

    @unittest.skipUnless(sys.platform == 'linux' and shutil.which('cc'), 'Linux ELF toolchain required')
    def test_real_ld_preload_and_call_counter(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / 'original.c'
            interposer = root / 'hook.c'
            client = root / 'client.c'
            original.write_text('int dcat_demo(int x) { return x + 2; }\n')
            interposer.write_text('static int n; int dcat_demo(int x) { ++n; return x + 5; }\n'
                                  'int dcat_demo_calls(void) { return n; }\n')
            client.write_text('extern int dcat_demo(int); int run(void) { return dcat_demo(1); }\n')
            for tag, source in (('real', original), ('hook', interposer), ('client', client)):
                extra = ['-L' + folder, '-lreal', '-Wl,-rpath,$ORIGIN'] if tag == 'client' else []
                subprocess.run(['cc', '-shared', '-fPIC', str(source), '-o', str(root / ('lib' + tag + '.so')),
                                *extra], check=True, capture_output=True)
            hook = root / 'libhook.so'
            sha256 = hashlib.sha256(hook.read_bytes()).hexdigest()
            code = '''
import ctypes, sys
from engineering_closure.native_observer import PinnedNativeLibrary, symbol_entry_status
hook, expected, client = sys.argv[1:]
loaded = PinnedNativeLibrary(hook, expected)
binding = loaded.bind('dcat_demo')
assert loaded.handle.dcat_demo_calls() == 0
assert symbol_entry_status(binding, []) == 'NOT_VISIBLE'
assert ctypes.CDLL(client).run() == 6
assert loaded.handle.dcat_demo_calls() == 1
assert symbol_entry_status(binding, [dict(vars(binding), actual_function_entry=True)]) == 'OBSERVED'
'''
            environment = preload_environment(hook, sha256)
            subprocess.run([sys.executable, '-c', code, str(hook), sha256, str(root / 'libclient.so')],
                           env=environment, check=True, capture_output=True)
