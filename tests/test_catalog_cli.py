import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from fault_injection.__main__ import main


class CatalogCliTests(unittest.TestCase):
    def test_catalog_has_no_torch_or_model_side_effects(self):
        root = Path(__file__).resolve().parents[1]
        code = ("import sys,json;from fault_injection.capabilities import list_mechanisms;"
                "s=list_mechanisms();assert 'torch' not in sys.modules;"
                "assert 'torch_npu' not in sys.modules;print(len(s))")
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=root,
                                text=True, capture_output=True, check=True)
        self.assertEqual(result.stdout.strip(), '17')

    def test_export_is_json_and_cannot_replace_existing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'catalog.json'
            self.assertEqual(main(['catalog', '--output', str(p)]), 0)
            data = json.loads(p.read_text(encoding='utf-8'))
            self.assertEqual(len(data['mechanisms']), 17)
            original = p.read_bytes()
            with self.assertRaises(FileExistsError):
                main(['catalog', '--output', str(p)])
            self.assertEqual(p.read_bytes(), original)

    def test_export_has_no_unimplemented_execution_command(self):
        with self.assertRaises(SystemExit): main(['inject'])


if __name__ == '__main__':
    unittest.main()
