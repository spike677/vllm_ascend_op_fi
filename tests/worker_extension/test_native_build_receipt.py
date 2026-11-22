import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tools.build_native_regions import build


class BuildTests(unittest.TestCase):
    def test_failed_compile_keeps_failed_receipt_and_existing_output_is_refused(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';sdk=root/'sdk';output=root/'output'
            source.mkdir();(sdk/'include/acl').mkdir(parents=True)
            (sdk/'include/acl/acl.h').write_text('fixture')
            for name in ('postwrite.cpp','postwrite.h','test_bound_addresses.cpp',
                         'test_original_snapshot.cpp','test_stream_regions.cpp'):
                (source/name).write_text('fixture')
            def run(args,**kw):
                if '--version' in args:return SimpleNamespace(stdout='fixture compiler',stderr='',returncode=0,check_returncode=lambda:None)
                raise RuntimeError('compile failure')
            with patch('tools.build_native_regions.subprocess.run',run),self.assertRaisesRegex(RuntimeError,'compile failure'):
                build(source,sdk,output)
            receipt=json.loads((output/'receipt.json').read_text())
            self.assertEqual(receipt['status'],'FAILED');self.assertEqual(receipt['npu_calls'],0)
            self.assertFalse(receipt['qualification_granted']);self.assertEqual(len(receipt['sources']),5)
            with self.assertRaisesRegex(ValueError,'FRESH_OUTPUT'):build(source,sdk,output)
            with self.assertRaisesRegex(ValueError,'FRESH_OUTPUT'):build(source,sdk,source/'inside')
