"""Actual controller log storage, including exclusive-create safety."""
import sys
from pathlib import Path
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controller import open_case_log


class LogStorageTests(unittest.TestCase):
    def test_link_preserves_original_written_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);e=root/'evidence';r=root/'raw';e.mkdir();r.mkdir()
            handle,link=open_case_log(e,r,'case')
            with handle:handle.write(b'first\nsecond\r\n')
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.read_bytes(),b'first\nsecond\r\n')
            self.assertEqual(link.resolve(),r/'case.log')

    def test_existing_raw_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);e=root/'evidence';r=root/'raw';e.mkdir();r.mkdir()
            (r/'case.log').write_bytes(b'existing')
            with self.assertRaises(FileExistsError):open_case_log(e,r,'case')
            self.assertEqual((r/'case.log').read_bytes(),b'existing')


if __name__=='__main__':unittest.main()
