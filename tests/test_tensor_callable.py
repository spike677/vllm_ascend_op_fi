"""Exact live CustomOp dispatch identities and narrowly reviewed collector wrapper."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from fault_injection.core import UnsupportedCall
from fault_injection import module_faults
from fault_injection.tensor_callable import TensorCallableGuard


class Dispatch(torch.nn.Module):
    def __init__(self):
        super().__init__(); self._forward_method = self.actual
    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)
    def actual(self, x, gain=1):
        return x * gain
    def alternative(self, x, gain=2):
        return x * gain


class TensorCallableTests(unittest.TestCase):
    def test_actual_extra_method_is_bound_even_when_generic_dispatch_is_unused(self):
        class Direct(torch.nn.Module):
            def forward(self, x):
                return self.forward_impl(x)
            def forward_impl(self, x):
                return x * 2
        module = Direct().eval()
        module._forward_method = lambda x: x  # Not used by this overridden forward.
        guard = TensorCallableGuard(module, extra_methods=('forward_impl',), bind_optional_dispatch=False)
        self.assertEqual(guard.names, ('forward', 'forward_impl'))
        module.forward_impl = types.MethodType(Dispatch.alternative, module)
        with self.assertRaisesRegex(UnsupportedCall, 'DISPATCH_CALLABLE_CHANGED'):
            guard.verify()

    def test_forward_and_actual_dispatch_both_bound(self):
        module = Dispatch().eval(); guard = TensorCallableGuard(module)
        self.assertEqual([row['name'] for row in guard.report()['methods']], ['forward', '_forward_method'])
        self.assertTrue(guard.verify())
        module._forward_method = module.alternative
        with self.assertRaisesRegex(UnsupportedCall, 'DISPATCH_CALLABLE_CHANGED'):
            guard.verify()

    def test_same_source_foreign_callable_and_foreign_instance_are_refused(self):
        module = Dispatch().eval(); guard = TensorCallableGuard(module)
        original = module._forward_method.__func__
        clone = types.FunctionType(original.__code__, original.__globals__, original.__name__, original.__defaults__)
        clone.__module__ = original.__module__; clone.__qualname__ = original.__qualname__
        module._forward_method = types.MethodType(clone, module)
        with self.assertRaisesRegex(UnsupportedCall, 'DISPATCH_CALLABLE_CHANGED'):
            guard.verify()
        module._forward_method = Dispatch().actual
        with self.assertRaisesRegex(UnsupportedCall, 'ACTUAL_BOUND'):
            TensorCallableGuard(module)

    def test_default_change_is_not_hidden_by_same_function_identity(self):
        module = Dispatch().eval()
        original = module.actual.__func__
        clone = types.FunctionType(original.__code__, original.__globals__, original.__name__, (1,))
        clone.__module__ = original.__module__; clone.__qualname__ = original.__qualname__
        module._forward_method = types.MethodType(clone, module)
        guard = TensorCallableGuard(module)
        clone.__defaults__ = (3,)
        with self.assertRaisesRegex(UnsupportedCall, 'SOURCE_OR_DEFAULTS_CHANGED'):
            guard.verify()

    def test_only_exact_reviewed_collector_wrapper_can_change_forward_object(self):
        module = Dispatch().eval(); other = Dispatch().eval()
        guard = TensorCallableGuard(module)
        original = module.forward
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)/'module_processor.py'
            source.write_text('def wrap_forward_with_hook_safety(module):\n'
                '    original_forward = module.forward\n'
                '    def wrapped_forward(*args, **kwargs):\n'
                '        if module.training:\n'
                '            raise RuntimeError("inference only")\n'
                '        return original_forward(*args, **kwargs)\n'
                '    module.forward = wrapped_forward\n')
            name = module_faults._MSPROBE_MODULE
            spec = importlib.util.spec_from_file_location(name, source)
            collector = importlib.util.module_from_spec(spec); spec.loader.exec_module(collector)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            with patch.dict(sys.modules, {name: collector}), patch.object(module_faults,
                    '_REVIEWED_MSPROBE_FORWARD_HASHES', {digest}):
                collector.wrap_forward_with_hook_safety(module)
                self.assertTrue(guard.verify())
                collector.wrap_forward_with_hook_safety(other)
                module.forward = other.forward
                with self.assertRaisesRegex(UnsupportedCall, 'ACTUAL_BOUND'):
                    guard.verify()
            module.forward = original
        self.assertTrue(guard.verify())


if __name__ == '__main__':
    unittest.main()
