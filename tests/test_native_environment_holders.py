"""Real ctypes ABI holders; construction must not execute any cached operator."""
import ctypes as C
import hashlib
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from fault_injection.deployments import writer26_group as writer
from fault_injection.deployments.postwrite import NativePostwriteLibrary, PostwriteRefused


class NativeHolderTests(unittest.TestCase):
    def getter(self):
        @C.CFUNCTYPE(C.c_int, C.c_uint32, C.POINTER(C.c_uint64))
        def value(slot, output):
            self.fail('holder construction invoked an address getter')
        return value

    def test_26_requires_fresh_profile_retains_loaded_handles_and_both_original_custom_slots(self):
        native = writer.NativeWriter26Group.__new__(writer.NativeWriter26Group)
        native.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        custom = native.root/'libcust_opapi.so'
        custom.write_bytes(b'CPU-only fixture')
        native.config = NS(thread=threading.get_native_id())
        native.build = dict(files={'libcust_opapi.so': hashlib.sha256(custom.read_bytes()).hexdigest()})
        with self.assertRaisesRegex(RuntimeError, 'FRESH_WRITER26_CACHED_ADDRESS_BUILD_REQUIRED'):
            native.environment_bindings()
        native.build['bound_symbol_profile'] = 'demoncat.writer26.cached_addresses.v1'
        native.lib = NS(dcat26_group_cached_address_v1=self.getter())
        native.sha256 = 'a'*64
        custom_library = NS(dcat26_custom_cached_address_v1=self.getter())
        for flag in ('RTLD_NOW', 'RTLD_LOCAL', 'RTLD_NOLOAD'):
            self.enterContext(patch.object(writer.os, flag, 1, create=True))
        with patch.object(writer.C, 'CDLL', return_value=custom_library) as load:
            group, proxy = native.environment_bindings()
        self.assertIs(group.library, native.lib)
        self.assertIs(proxy.library, custom_library)
        self.assertEqual(len(group.symbols), 14)
        self.assertEqual(proxy.symbols, ((0, 'aclnnAddRmsNormBias'), (1, 'aclnnAddRmsNormBiasGetWorkspaceSize')))
        self.assertEqual(load.call_args.kwargs['mode'], 1)
        self.assertEqual((C.sizeof(C.c_uint32), C.sizeof(C.c_uint64)), (4, 8))

    def test_39_reports_actual_linked_relocations_not_a_custom_operator_cache(self):
        library = NativePostwriteLibrary.__new__(NativePostwriteLibrary)
        library.lib, library.path, library.sha256 = NS(), '/native/libdcat_postwrite.so', 'a'*64
        with self.assertRaisesRegex(PostwriteRefused, 'FRESH_POSTWRITE_LINKED_ADDRESS_BUILD_REQUIRED'):
            library.environment_bindings()
        library.lib.dcat_pw_linked_address_v1 = self.getter()
        holder, = library.environment_bindings()
        self.assertIs(holder.library, library.lib)
        self.assertEqual(holder.kind, 'LINKED_FUNCTION_RELOCATION')
        self.assertEqual(holder.symbols, ((0, 'aclrtGetDevice'), (1, 'aclrtSynchronizeDevice'), (2, 'aclrtMemcpy')))


if __name__ == '__main__':
    unittest.main()
