"""Host-array readback contracts, including closed gates on stale/leaking DSOs."""
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fault_injection.metadata_arrays import (LoadedMetadataArrays, MetadataArraySession,
    LoadedWriter26MetadataArrays, MetadataArrayStateV1, _DlInfo, SCHEMA)
from metadata_fixture import metadata_report


class CounterContractTests(unittest.TestCase):
    def setUp(self):
        self.value = metadata_report(0, 'b' * 64)
        self.session = MetadataArraySession(lambda: copy.deepcopy(self.value), 'b' * 64)

    def test_actual_workload_activity_and_unused_proxy(self):
        self.assertEqual(ctypes.sizeof(MetadataArrayStateV1), 120)
        self.session.read()
        with self.assertRaisesRegex(RuntimeError, 'NO_ACTUAL_WORKLOAD'):
            self.session.read(require_activity=True)
        self.value = metadata_report(2, 'b' * 64)
        result = self.session.read(require_activity=True)
        self.assertEqual(result['images'][0]['state']['acquired_arrays'], 6)
        self.assertEqual(result['images'][1]['state']['acquired_arrays'], 0)

    def test_each_leak_error_or_active_capture_rejects(self):
        for field in ('active_captures', 'active_getters', 'getter_failures', 'getter_exceptions',
                      'capture_exceptions', 'live_arrays', 'ownership_errors', 'acquired_arrays'):
            with self.subTest(field=field):
                self.value = metadata_report(0, 'b' * 64)
                self.value['images'][0]['state'][field] = 1
                with self.assertRaisesRegex(RuntimeError, 'UNRELEASED_ARRAYS'):
                    self.session.read()

    def test_missing_reader_or_counter_report(self):
        with self.assertRaisesRegex(RuntimeError, 'READER_REQUIRED'):
            MetadataArraySession(None, 'b' * 64)
        self.value['images'] = []
        with self.assertRaisesRegex(RuntimeError, 'EMPTY_LOADED'):
            self.session.read()

    def test_old_writer_bundle_rejects_before_loading_any_symbols(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(os, 'RTLD_NOLOAD', 4, create=True), \
                patch('fault_injection.metadata_arrays.ctypes.CDLL') as loader:
            path = Path(folder) / 'build.json'
            path.write_text(json.dumps(dict(schema='demoncat.writer26.group_deployment.v2', files={})))
            with self.assertRaisesRegex(RuntimeError, 'OLD_WRITER_OR_TRACE_DEPENDENCIES_REQUIRE_REBUILD'):
                LoadedWriter26MetadataArrays(bundle=folder, manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            loader.assert_not_called()

    def test_no_reset_counter_instance_or_image_change(self):
        for change in ('reset', 'instance', 'sha256', 'proxy_set'):
            with self.subTest(change=change):
                self.setUp()
                self.value = metadata_report(2, 'b' * 64)
                self.session.read()
                if change == 'reset':
                    self.value = metadata_report(1, 'b' * 64)
                elif change == 'proxy_set':
                    self.value['images'].pop()
                elif change == 'instance':
                    self.value['images'][0]['state']['instance'] = 99
                else:
                    self.value['images'][0]['sha256'] = 'c' * 64
                with self.assertRaisesRegex(RuntimeError, 'CHANGED_OR_COUNTER_RESET|PROXY_SET_CHANGED'):
                    self.session.read()

    def test_shared_instance_or_bad_abi_cannot_satisfy_counter_proof(self):
        self.value['images'][1]['state']['instance'] = 1
        with self.assertRaisesRegex(RuntimeError, 'SHARED_OR_MISSING'):
            self.session.read()
        self.value = metadata_report(1, 'b' * 64)
        self.value['images'][0]['state']['size'] = 8
        with self.assertRaisesRegex(RuntimeError, 'ABI_MISMATCH'):
            self.session.read()


class LoadedImageTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.path = self.root / 'libopapi.so'
        self.path.write_bytes(b'CPU-mock-not-an-ELF')
        self.core = self.root / 'libfault39_native_v1.so'
        self.core.write_bytes(b'CPU-mock-core-not-an-ELF')
        self.profile = dict(schema='fault39.cann.build_profile.v1', adapter='matmul-cache-abi0-v1',
            proxy_layout='main_only', metadata_array_ownership=dict(schema=SCHEMA, state_version=1,
                state_symbol='fault39_metadata_arrays_v1', proxy_images=['libopapi.so'],
                reset_supported=False, runtime_quiescent_readback_required=True))
        self.profile['profile_sha256'] = hashlib.sha256(json.dumps(self.profile, sort_keys=True,
                                                                 separators=(',', ':')).encode()).hexdigest()
        profile_path = self.root / 'version_profile.json'
        profile_path.write_text(json.dumps(self.profile))
        self.manifest = dict(schema='fault39.native.build_manifest.v1', profile_sha256=self.profile['profile_sha256'],
            files={'version_profile.json': hashlib.sha256(profile_path.read_bytes()).hexdigest(),
                   'libopapi.so': hashlib.sha256(self.path.read_bytes()).hexdigest(),
                   self.core.name: hashlib.sha256(self.core.read_bytes()).hexdigest()})
        (self.root / 'build_manifest.json').write_text(json.dumps(self.manifest))
        self.maps = {'libopapi.so': (str(self.path),), self.core.name: (str(self.core),)}
        self.enterContext(patch('fault_injection.metadata_arrays._mapped_proxies', side_effect=lambda *a: copy.deepcopy(self.maps)))
        self.enterContext(patch.object(os, 'RTLD_NOLOAD', 4, create=True))
        self.enterContext(patch.object(os, 'RTLD_NOW', 2, create=True))
        self.origin = str(self.path).encode()
        self.core_origin = str(self.core).encode()
        self.counter = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(MetadataArrayStateV1))(self.read_counter)
        self.anchor = ctypes.CFUNCTYPE(ctypes.c_int)(lambda: 0)
        def locate(address, info):
            ctypes.cast(info, ctypes.POINTER(_DlInfo)).contents.dli_fname = (
                self.core_origin if address == ctypes.cast(self.anchor, ctypes.c_void_p).value else self.origin)
            return 1
        self.locate = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_DlInfo))(locate)
        class Library:
            pass
        self.library = Library()
        self.library.dladdr, self.library.fault39_metadata_arrays_v1 = self.locate, self.counter
        self.loader = self.enterContext(patch('fault_injection.metadata_arrays.ctypes.CDLL', return_value=self.library))

    @staticmethod
    def read_counter(pointer):
        pointer.contents.instance = 17
        return 0

    def attach(self):
        return LoadedMetadataArrays(kind='39', build_profile=self.profile['profile_sha256'], anchor=self.anchor)

    def test_exact_loaded_origin_no_late_dlopen_and_stable_files(self):
        reader = self.attach()
        value = reader.read()
        self.assertEqual(value['images'][0]['state']['instance'], 17)
        self.assertEqual(self.loader.call_args_list[1].kwargs, {'mode': os.RTLD_NOLOAD | os.RTLD_NOW})
        self.path.write_bytes(b'changed-ELF')
        with self.assertRaisesRegex(RuntimeError, 'BUILD_FILES_CHANGED'):
            reader.read()

    def test_wrong_symbol_origin_or_missing_symbol_rejects(self):
        self.origin = str(self.root / 'different.so').encode()
        (self.root / 'different.so').write_bytes(b'wrong origin')
        with self.assertRaisesRegex(RuntimeError, 'WRONG_IMAGE'):
            self.attach()
        del self.library.fault39_metadata_arrays_v1
        with self.assertRaisesRegex(RuntimeError, 'COUNTER_SYMBOL_MISSING'):
            self.attach()

    def test_wrong_artifact_bytes_reject_before_attachment(self):
        self.path.write_bytes(b'stale-different-proxy')
        with self.assertRaisesRegex(RuntimeError, 'PROXY_BUILD_HASH_MISMATCH'):
            self.attach()
        self.assertEqual(self.loader.call_count, 1)  # The process handle only.

    @unittest.skipIf(os.name == 'nt', 'Actual inode alias contract runs on Linux; Windows sandbox disallows hardlinks')
    def test_manifest_inode_verified_alias_is_allowed_unknown_copy_is_not(self):
        alias = self.root / 'sdk' / 'libopapi.so'
        alias.parent.mkdir()
        os.link(self.path, alias)
        self.maps['libopapi.so'] = (str(alias),)
        self.origin = str(alias).encode()
        value = self.attach().read()
        self.assertEqual(value['images'][0]['mapped_paths'], [str(alias)])
        self.assertEqual(value['images'][0]['path'], str(self.path))
        copied = self.root / 'foreign' / 'libopapi.so'
        copied.parent.mkdir()
        copied.write_bytes(self.path.read_bytes())
        self.maps['libopapi.so'] = (str(alias), str(copied))
        with self.assertRaisesRegex(RuntimeError, 'UNKNOWN_PROXY_ALIAS_OR_DUPLICATE'):
            self.attach()


if __name__ == '__main__':
    unittest.main()
