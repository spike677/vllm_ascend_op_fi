"""Mapped-owner fixture contracts; actual container behavior needs root's probe."""
import ctypes as C
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from fault_injection import quant_native_isolation as isolation
from fault_injection.core import UnsupportedCall


class Function:
    def __init__(self, function):
        self.function = function
    def __call__(self, *args):
        return self.function(*args)


class QuantMarkerOwnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / 'arbitrary.data'
        self.path.write_bytes(b'\x7fELFfixture marker content')
        self.stat = self.path.stat()
        self.row = dict(path=str(self.path), start=1000, end=2000, inode=self.stat.st_ino,
                        major=7, minor=11, deleted=False)
        self.info = NS(name=str(self.path).encode(), base=900)
        self.major = patch.object(isolation.os, 'major', return_value=7, create=True)
        self.minor = patch.object(isolation.os, 'minor', return_value=11, create=True)
        self.major.start()
        self.minor.start()

    def tearDown(self):
        self.minor.stop()
        self.major.stop()
        self.temporary.cleanup()

    def test_only_actual_marker_owner_needs_exact_mapping_identity(self):
        unrelated = dict(path='/unrelated/system/overlay', start=1, end=999, inode=123,
                         major=253, minor=2, deleted=False)
        result = isolation._marker_owner(self.info, 1001, [unrelated, self.row])
        self.assertEqual(result['path'], str(self.path.resolve()))
        self.assertEqual(result['inode'], self.stat.st_ino)

    def test_quant_owner_overlay_device_or_inode_mismatch_is_still_rejected(self):
        for change in (dict(major=253, minor=2), dict(inode=self.stat.st_ino + 1)):
            with self.subTest(change=change), self.assertRaisesRegex(UnsupportedCall, 'MAPPED_FILE_IDENTITY_CHANGED'):
                isolation._marker_owner(self.info, 1001, [dict(self.row, **change)])

    def test_deleted_actual_marker_owner_is_rejected(self):
        with self.assertRaisesRegex(UnsupportedCall, 'MAPPING_DELETED'):
            isolation._marker_owner(self.info, 1001, [dict(self.row, deleted=True)])

    def test_wrong_address_or_dladdr_path_is_rejected(self):
        with self.assertRaisesRegex(UnsupportedCall, 'NOT_IN_VERIFIED'):
            isolation._marker_owner(self.info, 3000, [self.row])
        other = self.path.with_name('other.data')
        other.write_bytes(self.path.read_bytes())
        with self.assertRaisesRegex(UnsupportedCall, 'FILE_OWNER_DIFFER'):
            isolation._marker_owner(NS(name=str(other).encode(), base=900), 1001, [self.row])

    def test_non_elf_marker_owner_is_rejected(self):
        self.path.write_bytes(b'not an ELF library')
        with self.assertRaisesRegex(UnsupportedCall, 'MAPPED_FILE_IDENTITY_CHANGED'):
            isolation._marker_owner(self.info, 1001, [self.row])

    def test_unrelated_overlay_mappings_are_not_statted_before_marker_discovery(self):
        # Both paths intentionally do not exist. NOLOAD finds no quant marker;
        # unrelated overlay/deleted identity is not reclassified as a quant DSO.
        lines = ('1-3 r-xp 0000 fd:02 50993568 /unrelated/python\n'
                 '4-6 r-xp 0000 fd:02 50993569 /unrelated/lib.data (deleted)\n')
        closed = []
        runtime = NS(dlopen=Function(lambda *args: 7), dlsym=Function(lambda *args: 0),
                     dladdr=Function(lambda *args: 0), dlclose=Function(lambda value: closed.append(value) or 0))
        with patch.object(isolation.sys, 'platform', 'linux'), patch.object(Path, 'read_text', return_value=lines), \
                patch.object(isolation.os, 'RTLD_NOLOAD', 4, create=True), \
                patch.object(isolation.os, 'RTLD_NOW', 2, create=True), \
                patch.object(isolation.os, 'RTLD_LOCAL', 0, create=True), patch.object(isolation.C, 'CDLL', return_value=runtime):
            self.assertEqual(isolation.mapped_native_modes(), ())
        self.assertEqual(closed, [7, 7])


if __name__ == '__main__':
    unittest.main()
