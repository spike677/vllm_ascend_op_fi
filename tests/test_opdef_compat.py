import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fault_injection.native_adapters.attention28.fia import opdef_compat as compat


class OpdefCompatTests(unittest.TestCase):
    def test_only_unsupported_registrations_removed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            originals = self.populate(root)
            with patch.object(compat.subprocess, 'check_output', side_effect=self.reader(originals)):
                receipt = compat.adapt(root)
            self.assertEqual(3, len(receipt['files']))
            for path, before in originals.items():
                after = (root / path).read_bytes()
                self.assertIn(b'AddConfig("ascend910b", target);', after)
                self.assertIn(b'AddConfig("ascend310p", other);', after)
                self.assertIn(b'tiling_and_kernel_unchanged();', after)
                self.assertNotIn(b'"mc62"', after)
                self.assertNotIn(b'"ascend950"', after)

    def test_changed_original_refuses_without_partial_write(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            originals = self.populate(root)
            last = list(originals)[-1]
            (root / last).write_bytes(originals[last] + b'// foreign modification\n')
            before = {path: (root/path).read_bytes() for path in originals}
            with patch.object(compat.subprocess, 'check_output', side_effect=self.reader(originals)):
                with self.assertRaisesRegex(RuntimeError, 'PRISTINE_DEFINITION'):
                    compat.adapt(root)
            self.assertEqual(before, {path: (root/path).read_bytes() for path in originals})

    @staticmethod
    def populate(root):
        originals = {}
        for op, removed in compat.TARGETS.items():
            name = f'attention/{op}/op_host/{op}_def.cpp'
            raw = (b'tiling_and_kernel_unchanged();\n'
                   b'    this->AICore().AddConfig("ascend910b", target);\n'
                   b'    this->AICore().AddConfig("ascend310p", other);\n')
            raw += b''.join(b'    this->AICore().AddConfig("'+s.encode()+b'", new_soc);\n' for s in removed)
            path = root/name
            path.parent.mkdir(parents=True)
            path.write_bytes(raw)
            originals[name] = raw
        return originals

    @staticmethod
    def reader(originals):
        def read(argv, **kwargs):
            if argv[-1] == 'HEAD':
                return compat.COMMIT + '\n'
            return originals[argv[-1].split(':', 1)[1]]
        return read
