"""Actual ctypes address getters, synthetic executable maps; never device proof."""
from copy import deepcopy
import ctypes as C
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from fault_injection import environment as env


class BoundEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.getter_calls, self.code, self.target = [], 0, 0x5200
        self.name = 'cpu_cached_getter'
        @C.CFUNCTYPE(C.c_int, C.c_uint32, C.POINTER(C.c_uint64))
        def getter(slot, output):
            self.getter_calls.append(slot)
            output[0] = self.target if self.code == 0 else 0
            return self.code
        self.getter = getter
        self.getter_address = C.cast(getter, C.c_void_p).value
        self.library = NS(cpu_cached_getter=getter)
        self.holder = env.NativeSymbolHolder(self.library, getter, self.name, '/native/libcust_opapi.so',
            'a'*64, ((0, 'cpu_actual_continuation'),))
        self.environment = dict(pid=os.getpid(), maps_status='OBSERVED', machine='CPU_FIXTURE', python='fixture',
            packages={}, torch_cxx11_abi=False, driver=dict(installation=None, running_module_versions={}),
            loaded_libraries=[self.image('/native/libcust_opapi.so', 'a', self.getter_address-0x100, 11),
                              self.image('/vendor/libcust_opapi.so', 'b', 0x5000, 22),
                              self.image('/vendor/libtorch_npu.so', 'c', 0x8000, 33)])
        for row in self.environment['loaded_libraries'][:2]:
            row.update(status='AMBIGUOUS_MAPPING', file_identity_status='MAPPED_FILE_VERIFIED')
        self.enterContext(patch.object(env, 'inspect_environment', side_effect=lambda **kwargs: deepcopy(self.environment)))
        self.enterContext(patch.object(env, '_address_origin', side_effect=self.origin))
        self.enterContext(patch.object(env.os, 'major', return_value=8, create=True))
        self.enterContext(patch.object(env.os, 'minor', return_value=2, create=True))
        self.enterContext(patch.object(env.os, 'stat', side_effect=lambda path: NS(**self.stat(path))))
        self.enterContext(patch.object(Path, 'read_text', side_effect=lambda *args, **kwargs: self.maps()))

    def stat(self, path):
        row = next(row for row in self.environment['loaded_libraries'] if row['path'] == str(path))
        return dict(st_dev=8, st_ino=int(row['mapping']['inode']), st_size=4096, st_mode=0o100444,
                    st_mtime_ns=1, st_ctime_ns=1)

    def image(self, path, sha, start, inode):
        return dict(path=path, name=path.rsplit('/', 1)[-1], sha256=sha*64, elf_build_id=None,
            status='MAPPED_FILE_VERIFIED', mapping=dict(device='08:02', inode=str(inode), deleted=False,
                executable_segments=[dict(address=f'{start:x}-{start+0x1000:x}', file_offset='00001000')]),
            path_identity_current=dict(device=8, major=8, minor=2, inode=inode, bytes=4096,
                                       mode=0o100444, mtime_ns=1, ctime_ns=1))

    def maps(self):
        return '\n'.join(f"{part['address']} r-xp {part['file_offset']} 08:02 {row['mapping']['inode']} {row['path']}"
            for row in self.environment['loaded_libraries'] for part in row['mapping']['executable_segments'])

    def origin(self, address):
        getter = address == self.getter_address
        return dict(path='/native/libcust_opapi.so' if getter else '/vendor/libcust_opapi.so',
            base=self.getter_address-0x100 if getter else 0x5000, address=address, symbol_address=address,
            symbol=self.name if getter else 'cpu_actual_continuation')

    def inspect(self, holders=None):
        return env.inspect_bound_environment((self.holder,) if holders is None else holders,
                                             required_libraries=('libcust_opapi.so',))

    def test_complete_collision_set_plus_original_held_address_has_distinct_explicit_profile(self):
        self.assertIsNone(env.environment_fingerprint(self.environment, ('libcust_opapi.so',)))
        result, fingerprint = self.inspect()
        self.assertEqual(len(fingerprint), 64)
        self.assertEqual(self.getter_calls, [0, 0])
        audit = result['bound_symbol_evidence']
        self.assertEqual(audit['status'], 'EXACT_MAPPED_FUNCTION_BINDINGS_OBSERVED')
        self.assertEqual(len(result['loaded_libraries']), 3)
        self.assertEqual(audit['collision_groups'][0]['complete_paths'],
                         ['/native/libcust_opapi.so', '/vendor/libcust_opapi.so'])
        self.assertFalse(audit['qualification_granted'] or audit['serializable_authority'])
        self.assertIsNone(env.environment_fingerprint(result, ('libcust_opapi.so',)))

    def test_unselected_verified_image_remains_in_fingerprint(self):
        _, before = self.inspect()
        self.environment['loaded_libraries'][2]['sha256'] = 'd'*64
        _, after = self.inspect()
        self.assertNotEqual(before, after)

    def test_unverified_extra_image_prevents_even_reading_the_native_getter(self):
        self.environment['loaded_libraries'][2]['status'] = 'MAPPED_FILE_IDENTITY_CHANGED'
        result, fingerprint = self.inspect()
        self.assertIsNone(fingerprint)
        self.assertFalse(self.getter_calls)
        self.assertIn('COMPLETE_VERIFIED', result['bound_symbol_evidence']['refusal'])

    def test_unresolved_cache_is_not_resolved_or_replaced(self):
        self.code = 3
        result, fingerprint = self.inspect()
        self.assertIsNone(fingerprint)
        self.assertEqual(self.getter_calls, [0])
        self.assertIn('NATIVE_CACHED_SYMBOL_UNRESOLVED', result['bound_symbol_evidence']['refusal'])

    def test_getter_manifest_mismatch_refuses_before_call(self):
        result, fingerprint = self.inspect((replace(self.holder, sha256='f'*64),))
        self.assertIsNone(fingerprint)
        self.assertFalse(self.getter_calls)
        self.assertIn('BOUND_GETTER_MANIFEST_IMAGE_CHANGED', result['bound_symbol_evidence']['refusal'])

    def test_unknown_cached_address_owner_is_not_path_based_acceptance(self):
        self.target = 0x999999
        result, fingerprint = self.inspect()
        self.assertIsNone(fingerprint)
        self.assertIn('UNIQUE_VERIFIED_EXECUTABLE', result['bound_symbol_evidence']['refusal'])
        self.assertEqual(result['bound_symbol_evidence']['readback_before_failure']['current_binding']['cached_address'],
                         self.target)

    def test_current_lookup_or_json_cannot_replace_a_live_holder(self):
        for holders in ((dict(path=self.holder.path, address=self.target),),
                        (replace(self.holder, getter=lambda *args: 0),)):
            with self.subTest(holders=type(holders[0])):
                _, fingerprint = self.inspect(holders)
                self.assertIsNone(fingerprint)
        self.assertFalse(self.getter_calls)

    def test_another_same_name_collision_without_actual_holder_stays_refused(self):
        for index in range(2):
            row = self.image('/extra'+str(index)+'/libopapi.so', 'e', 0x10000+index*0x2000, 40+index)
            row.update(status='AMBIGUOUS_MAPPING', file_identity_status='MAPPED_FILE_VERIFIED')
            self.environment['loaded_libraries'].append(row)
        result, fingerprint = self.inspect()
        self.assertIsNone(fingerprint)
        self.assertIn('DUPLICATE_IMAGE_WITHOUT_ACTUAL_BOUND_SYMBOL', result['bound_symbol_evidence']['refusal'])

    def test_mapping_or_path_change_after_read_cannot_keep_fingerprint(self):
        with patch.object(Path, 'read_text', return_value=''):
            result, fingerprint = self.inspect()
        self.assertIsNone(fingerprint)
        self.assertIn('MAPPING_SET_CHANGED', result['bound_symbol_evidence']['refusal'])
        with patch.object(env.os, 'stat', side_effect=FileNotFoundError('CPU path replaced')):
            _, fingerprint = self.inspect()
        self.assertIsNone(fingerprint)

    def test_native_pointer_must_be_the_same_exact_symbol_on_both_reads(self):
        original = self.origin
        calls = []
        def origin(address):
            value = original(address)
            if address == self.target:
                calls.append(address)
                if len(calls) == 2:
                    value['base'] += 1
            return value
        with patch.object(env, '_address_origin', side_effect=origin):
            result, fingerprint = self.inspect()
        self.assertIsNone(fingerprint)
        self.assertIn('NATIVE_BOUND_SYMBOL_CHANGED', result['bound_symbol_evidence']['refusal'])

    def test_loader_symlink_spelling_requires_exact_same_mapped_file_identity(self):
        original = self.origin
        def origin(address):
            value = original(address)
            if address == self.getter_address:
                value['path'] = '/loader-alias/libcust_opapi.so'
            return value
        def stat(path):
            canonical = '/native/libcust_opapi.so' if str(path).startswith('/loader-alias/') else path
            return NS(**self.stat(canonical))
        with patch.object(env, '_address_origin', side_effect=origin), patch.object(env.os, 'stat', side_effect=stat):
            result, fingerprint = self.inspect()
        self.assertIsNotNone(fingerprint)
        getter = result['bound_symbol_evidence']['bindings'][0]['getter']
        self.assertEqual(getter['path'], '/loader-alias/libcust_opapi.so')
        self.assertEqual(getter['mapped_path'], '/native/libcust_opapi.so')

    def test_same_inode_under_different_path_device_is_not_a_valid_symbol_alias(self):
        def stat(path):
            value = self.stat(path)
            if path == '/native/libcust_opapi.so':
                value['st_dev'] += 1
            return NS(**value)
        with patch.object(env.os, 'stat', side_effect=stat):
            result, fingerprint = self.inspect()
        self.assertIsNone(fingerprint)
        self.assertFalse(self.getter_calls)
        self.assertIn('BOUND_SYMBOL_PATH_DOES_NOT_NAME_MAPPED_FILE', result['bound_symbol_evidence']['refusal'])


if __name__ == '__main__':
    unittest.main()
