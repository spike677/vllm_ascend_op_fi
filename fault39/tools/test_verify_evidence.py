import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from verify_evidence import verify_manifest


class EvidenceVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'root'
        self.root.mkdir()
        self.file = self.root / 'evidence.json'
        self.file.write_bytes(b'{"a":1}\n')
        self.entry = {'path': self.file.name, 'bytes': self.file.stat().st_size,
                      'sha256': hashlib.sha256(self.file.read_bytes()).hexdigest()}

    def test_exact_file_passes_and_same_size_corruption_fails(self):
        manifest = {'files': [self.entry]}
        result = verify_manifest(manifest, self.root)
        self.assertEqual((result['status'], result['count'], result['verified_count']), ('PASS', 1, 1))
        self.file.write_bytes(b'{"a":2}\n')
        result = verify_manifest(manifest, self.root)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['checked_count'], 1)
        self.assertEqual(result['failures'][0]['code'], 'CHECKSUM_MISMATCH')

    def test_unsafe_paths_and_duplicates_are_rejected(self):
        for path in ('../secret.json', 'x/../evidence.json', '/absolute.json',
                     'C:/absolute.json', 'C:relative.json', '\\\\server\\share\\file', './evidence.json'):
            with self.subTest(path=path):
                result = verify_manifest({'files': [dict(self.entry, path=path)]}, self.root)
                self.assertEqual(result['failures'][0]['code'], 'UNSAFE_RELATIVE_PATH')
                self.assertEqual(result['checked_count'], 0)
        result = verify_manifest({'files': [self.entry, self.entry]}, self.root)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['failures'][0]['code'], 'DUPLICATE_PATH')

    def test_resolved_escape_is_rejected_before_external_file_read(self):
        outside = Path(self.temp.name) / 'secret.json'
        outside.write_bytes(b'private')
        original_resolve = Path.resolve
        # Model resolve() following a symlink/junction without requiring OS
        # symlink-creation privileges on the Windows test host.
        def resolve(path, *args, **kwargs):
            if path == self.file:
                return original_resolve(outside, *args, **kwargs)
            return original_resolve(path, *args, **kwargs)
        with patch.object(Path, 'resolve', resolve), patch.object(Path, 'open', side_effect=AssertionError('Must not read escaped file')):
            result = verify_manifest({'files': [self.entry]}, self.root)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['checked_count'], 0)
        self.assertEqual(result['failures'][0]['code'], 'RESOLVED_PATH_ESCAPE')


if __name__ == '__main__':
    unittest.main()
