import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tools.build_reference_adapter_delivery import build

class DeliveryTests(unittest.TestCase):
    def test_unreviewed_variant_cannot_enter_supplement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);docs=root/'docs/worker_core_v1';docs.mkdir(parents=True)
            (docs/'reference_adapter_a2_20261124.json').write_text(json.dumps({'variants':{}}))
            out=root/'delivery'
            with patch('tools.build_reference_adapter_delivery.ROOT',root):
                with self.assertRaisesRegex(ValueError,'A2_REVIEW_NOT_PASSED'):build('quant',out)
            self.assertFalse(out.exists())
    def test_existing_supplement_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);sentinel=path/'frozen';sentinel.write_bytes(b'keep')
            with self.assertRaisesRegex(ValueError,'IMMUTABLE'):build('quant',path)
            self.assertEqual(sentinel.read_bytes(),b'keep')
