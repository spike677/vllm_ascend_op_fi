"""Collector composition: only reviewed wrapper code and exact closure owner."""
import hashlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from fault_injection.module_faults import ModuleFaultSession, discover_modules, _MSPROBE_MODULE


FIXTURE = '''def wrap_forward_with_hook_safety(module):
    original_forward = module.forward
    def wrapped_forward(*args, **kwargs):
        if module is None:
            raise RuntimeError('unreachable')
        return original_forward(*args, **kwargs)
    module.forward = wrapped_forward
'''


class CollectorWrapper(unittest.TestCase):
    def test_reviewed_collector_is_kept_and_real_original_executes(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'module_processor.py';source.write_text(FIXTURE)
            collector=types.ModuleType(_MSPROBE_MODULE);collector.__file__=str(source)
            exec(compile(FIXTURE,str(source),'exec'),collector.__dict__)
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            with patch.dict(sys.modules,{_MSPROBE_MODULE:collector}),patch(
                'fault_injection.module_faults._REVIEWED_MSPROBE_FORWARD_HASHES',{digest}):
                for mechanism,module,inputs in (
                    ('18.api_half',torch.nn.RMSNorm(8),torch.arange(8.)),
                    ('17.spatial_padding',torch.nn.Conv3d(1,1,(1,16,16),stride=(1,16,16)),
                     torch.arange(256.).reshape(1,1,1,16,16))):
                    module.eval();healthy=module(inputs)
                    collector.wrap_forward_with_hook_safety(module)
                    wrapped=module.forward
                    self.assertEqual(len(discover_modules(module,mechanism)),1)
                    with ModuleFaultSession(module,mechanism) as session:
                        actual=module(inputs)
                        self.assertEqual(session.report()['events'][0]['module_forward_identity'],id(wrapped))
                    self.assertIs(module.forward,wrapped)
                    self.assertTrue(torch.equal(module(inputs),healthy))
                    self.assertFalse(torch.equal(actual,healthy))

    def test_same_name_unknown_source_and_custom_forward_not_unwrapped(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'module_processor.py';source.write_text(FIXTURE)
            collector=types.ModuleType(_MSPROBE_MODULE);collector.__file__=str(source)
            exec(compile(FIXTURE,str(source),'exec'),collector.__dict__)
            module=torch.nn.RMSNorm(8).eval()
            with patch.dict(sys.modules,{_MSPROBE_MODULE:collector}):
                collector.wrap_forward_with_hook_safety(module)
                self.assertFalse(discover_modules(module,'18.api_half'))
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            with patch.dict(sys.modules,{_MSPROBE_MODULE:collector}),patch(
                'fault_injection.module_faults._REVIEWED_MSPROBE_FORWARD_HASHES',{digest}):
                other=torch.nn.RMSNorm(8).eval()
                other.forward=module.forward
                self.assertFalse(discover_modules(other,'18.api_half'))
                custom=torch.nn.RMSNorm(8).eval()
                custom.forward=types.MethodType(lambda self,x:x,custom)
                collector.wrap_forward_with_hook_safety(custom)
                self.assertFalse(discover_modules(custom,'18.api_half'))


if __name__=='__main__':unittest.main()
