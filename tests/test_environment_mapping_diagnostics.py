"""File-only CPU fixtures: diagnostics must not repair unproven loaded identity."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fault_injection import environment as env


class MappingDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.library = self.root/'libascendcl.so'
        self.library.write_bytes(b'\x7fELF'+b'CPU fixture, not a native library')
        self.info = self.library.stat()
        # Windows has no os.major/minor; these only label the CPU fixture.
        self.enterContext(patch.object(env.os, 'major', return_value=8, create=True))
        self.enterContext(patch.object(env.os, 'minor', return_value=2, create=True))
        self.offsets = [('1000-2000', '00001000')]

    def row(self, *, inode=None):
        return env._loaded_library((str(self.library), '08:02',
            str(self.info.st_ino if inode is None else inode), False), self.offsets)

    def test_exact_mapping_retains_maps_and_three_file_stamps(self):
        row = self.row()
        self.assertEqual(row['status'], 'MAPPED_FILE_VERIFIED')
        self.assertEqual(row['sha256'], hashlib.sha256(self.library.read_bytes()).hexdigest())
        self.assertEqual(row['mapping'], dict(device='08:02', inode=str(self.info.st_ino), deleted=False,
            executable_segments=[dict(address='1000-2000', file_offset='00001000')]))
        self.assertEqual(row['path_identity_before'], row['path_identity_after'])
        self.assertEqual(row['path_identity_before'], row['path_identity_current'])
        self.assertIsNone(row['mapping_backing'])

    def test_overlay_style_inode_mismatch_never_hashes_or_accepts_path(self):
        with (patch.object(env, '_digest', side_effect=AssertionError('unbound pathname hash')),
              patch.object(env, '_mapping_backing_diagnostic', return_value=[dict(authority_granted=False)]) as backing):
            row = self.row(inode=self.info.st_ino+1)
        self.assertEqual(row['status'], 'MAPPED_FILE_IDENTITY_CHANGED')
        self.assertEqual(row['identity_failure'], 'EXECUTABLE_MAP_DOES_NOT_MATCH_OPEN_PATH')
        self.assertEqual(row['path_identity_before']['inode'], self.info.st_ino)
        self.assertEqual(int(row['mapping']['inode']), self.info.st_ino+1)
        self.assertIsNone(row['sha256'])
        backing.assert_called_once_with(self.offsets)

    def test_map_files_permission_failure_is_retained_without_fallback(self):
        with patch.object(env.os, 'readlink', return_value='/mapped/libascendcl.so'), patch.object(
                Path, 'open', side_effect=PermissionError(1, 'CPU fixture missing capability')):
            row, = env._mapping_backing_diagnostic(self.offsets)
        self.assertEqual(row['target'], '/mapped/libascendcl.so')
        self.assertEqual(row['open_error'], dict(type='PermissionError', errno=1))
        self.assertIsNone(row['file_identity'])
        self.assertFalse(row['authority_granted'])

    def test_map_files_readback_is_diagnostic_even_if_it_can_open(self):
        with patch.object(env.os, 'readlink', return_value=str(self.library)), patch.object(
                Path, 'open', side_effect=lambda *args: open(self.library, 'rb')):
            row, = env._mapping_backing_diagnostic(self.offsets)
        self.assertEqual(row['file_identity']['inode'], self.info.st_ino)
        self.assertFalse(row['authority_granted'])
        self.assertNotIn('sha256', row)

    def test_duplicate_basename_keeps_both_mappings_and_underlying_failure(self):
        maps = ('1000-2000 r-xp 00001000 08:02 11 /native/libcust_opapi.so\n'
                '3000-4000 r-xp 00001000 00:ab 22 /image/libcust_opapi.so\n')
        def library(key, offsets):
            path, device, inode, deleted = key
            good = path.startswith('/native')
            return dict(path=path, name='libcust_opapi.so',
                status='MAPPED_FILE_VERIFIED' if good else 'MAPPED_FILE_IDENTITY_CHANGED',
                sha256='a'*64 if good else None, elf_build_id=None,
                mapping=dict(device=device, inode=inode, deleted=deleted, executable_segments=offsets))
        with patch.object(Path, 'read_text', return_value=maps), patch.object(env, '_loaded_library', library):
            rows, status = env._loaded_libraries(())
        self.assertEqual(status, 'OBSERVED')
        self.assertEqual(len(rows), 2)
        self.assertEqual({row['file_identity_status'] for row in rows},
                         {'MAPPED_FILE_VERIFIED', 'MAPPED_FILE_IDENTITY_CHANGED'})
        self.assertTrue(all(row['status'] == 'AMBIGUOUS_MAPPING' for row in rows))
        self.assertTrue(all(len(row['same_basename_mappings']) == 2 for row in rows))
        self.assertIsNone(env.environment_fingerprint(dict(maps_status=status, loaded_libraries=rows),
                                                     ('libcust_opapi.so',)))

    def test_verified_required_library_does_not_hide_unverified_extra_mapping(self):
        row = self.row()
        extra = deepcopy(row)
        extra.update(name='libtorch_npu.so', status='MAPPED_FILE_IDENTITY_CHANGED', sha256=None)
        actual = dict(maps_status='OBSERVED', loaded_libraries=[row, extra])
        self.assertIsNone(env.environment_fingerprint(actual, ('libascendcl.so',)))


if __name__ == '__main__':
    unittest.main()
