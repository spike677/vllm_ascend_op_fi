"""Negative checks for read-only source/artifact identity auditing."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from experiments.c2c3_closure_20261117.audit_cpu_builds_124 import check_files, normalized_symbols


class CpuBuildAuditTests(unittest.TestCase):
    def test_original_digest_and_size_are_both_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'artifact'; path.write_bytes(b'original')
            entry = dict(path='artifact', sha256=hashlib.sha256(b'original').hexdigest(), bytes=8)
            self.assertFalse(check_files(folder, [entry])['mismatches'])
            path.write_bytes(b'replaced')
            self.assertEqual(len(check_files(folder, [entry])['mismatches']), 1)
            entry['sha256'] = hashlib.sha256(b'replaced').hexdigest(); entry['bytes'] = 9
            self.assertEqual(len(check_files(folder, [entry])['mismatches']), 1)

    def test_parent_and_duplicate_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'UNSAFE_OR_DUPLICATE'):
                check_files(folder, [dict(path='../outside', sha256='0' * 64)])
            path = Path(folder) / 'artifact'; path.write_bytes(b'x')
            entry = dict(path='artifact', sha256=hashlib.sha256(b'x').hexdigest())
            with self.assertRaisesRegex(ValueError, 'UNSAFE_OR_DUPLICATE'):
                check_files(folder, [entry, entry])

    def test_nm_tool_format_does_not_hide_symbol_or_address_change(self):
        sdk = '  U malloc@GLIBC_2.17\n00000001 T dcat26_group_state_v2\n'
        host = '00000001 T dcat26_group_state_v2\n U malloc\n'
        self.assertEqual(normalized_symbols(sdk), normalized_symbols(host))
        self.assertNotEqual(normalized_symbols(sdk), normalized_symbols(host.replace('00000001', '00000002')))
        self.assertNotEqual(normalized_symbols(sdk), normalized_symbols(host.replace('state_v2', 'state_v1')))

    def test_ambiguous_symbol_versions_refused(self):
        with self.assertRaisesRegex(ValueError, 'AMBIGUOUS_EXPORTED_SYMBOL'):
            normalized_symbols(' U f@V1\n U f@V2\n')


if __name__ == '__main__':
    unittest.main()
