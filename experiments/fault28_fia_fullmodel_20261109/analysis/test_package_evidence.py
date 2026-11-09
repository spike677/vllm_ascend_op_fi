from pathlib import Path
import tempfile
import unittest
from package_evidence import Export


class ExportTests(unittest.TestCase):
    def test_full_tensor_suffix_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'private.pt';source.write_bytes(b'private tensor')
            export=Export(root/'out')
            with self.assertRaises(AssertionError):export.copy(source,Path('tensor.pt'))
            self.assertFalse((root/'out/tensor.pt').exists())

    def test_destination_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'dump.json';source.write_bytes(b'{"Mean":1.25}\n')
            export=Export(root/'out')
            with self.assertRaises(AssertionError):export.copy(source,Path('../escape.json'))
            self.assertFalse((root/'escape.json').exists())

    def test_original_statistical_bytes_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'dump.json';data=b'{"Mean":1.2345678901234567,"Norm":null}\r\n'
            source.write_bytes(data);export=Export(root/'out')
            export.copy(source,Path('B/observation/step0/rank0/dump.json'))
            self.assertEqual((root/'out/B/observation/step0/rank0/dump.json').read_bytes(),data)
            self.assertTrue(export.mapping[0]['original_bytes_preserved'])

    def test_symlink_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source.json';source.write_text('{}')
            link=root/'linked.json';link.symlink_to(source)
            export=Export(root/'out')
            with self.assertRaises(AssertionError):export.copy(link,Path('linked.json'))


if __name__=='__main__':unittest.main()
