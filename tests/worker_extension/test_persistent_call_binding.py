"""Pinned-source CPU fixture for starting another session after collector stop."""
import hashlib
import inspect
from pathlib import Path
import sys
import tempfile
from types import FunctionType, ModuleType
import unittest
from unittest.mock import patch

from demoncat.runtime.vllm_ascend.worker_extension import call_binding as binding
from test_call_binding import CpuForward


SOURCE = '''def wrap_forward_with_hook_safety(module):
    original_forward = module.forward
    def wrapped_forward(*args, **kwargs):
        module._cpu_calls = getattr(module, '_cpu_calls', 0) + 1
        return original_forward(*args, **kwargs)
    module.forward = wrapped_forward
'''


class PersistentBindingTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.path = self.folder / 'module_processor.py'
        self.path.write_text(SOURCE)
        self.collector = ModuleType(binding.COLLECTOR_MODULE)
        self.collector.__file__ = str(self.path)
        exec(compile(self.path.read_bytes(), str(self.path), 'exec'), self.collector.__dict__)
        self.enterContext(patch.dict(sys.modules, {binding.COLLECTOR_MODULE: self.collector}))
        self.enterContext(patch.object(binding, 'REVIEWED_COLLECTOR_SHA256', hashlib.sha256(self.path.read_bytes()).hexdigest()))
        self.module = CpuForward()

    def test_second_session_uses_original_signature_and_preserves_collector(self):
        first = binding.ForwardBinding(self.module)
        self.collector.wrap_forward_with_hook_safety(self.module)
        self.collector.wrap_forward_with_hook_safety(self.module)
        wrapper = self.module.forward
        self.assertEqual(str(inspect.signature(wrapper)), '(*args, **kwargs)')
        second = binding.ForwardBinding(self.module)
        self.assertTrue(binding.same_callable(first.original, second.original))
        self.assertEqual(first.signature, second.signature)
        self.assertEqual(second.bind(self.module, ('tokens', 'positions'), {}),
                         dict(input_ids='tokens', positions='positions'))
        self.assertIs(self.module.forward, wrapper)
        self.assertEqual(self.module.executions, 0)
        self.module.forward('tokens', 'positions')
        self.assertEqual(self.module.executions, 1)
        self.assertEqual(self.module._cpu_calls, 2)

    def test_unknown_source_and_replaced_factory_do_not_gain_trust(self):
        self.collector.wrap_forward_with_hook_safety(self.module)
        self.path.write_text(self.path.read_text() + '# changed\n')
        with self.assertRaisesRegex(RuntimeError, 'UNREVIEWED_PERSISTENT_FORWARD_WRAPPER_SOURCE'):
            binding.ForwardBinding(self.module)
        self.assertEqual(self.module.executions, 0)

    def test_foreign_module_and_original_closure_are_rejected(self):
        other = CpuForward()
        self.collector.wrap_forward_with_hook_safety(other)
        self.module.forward = other.forward
        with self.assertRaisesRegex(RuntimeError, 'FORWARD_WRAPPER_OWNER_CHANGED'):
            binding.ForwardBinding(self.module)
        self.module = CpuForward()
        self.collector.wrap_forward_with_hook_safety(self.module)
        cells = dict(zip(self.module.forward.__code__.co_freevars, self.module.forward.__closure__))
        cells['original_forward'].cell_contents = CpuForward().forward
        with self.assertRaisesRegex(RuntimeError, 'PERSISTENT_FORWARD_ORIGINAL_OWNER_CHANGED'):
            binding.ForwardBinding(self.module)

    def test_cyclic_prior_wrapper_is_bounded(self):
        self.collector.wrap_forward_with_hook_safety(self.module)
        cells = dict(zip(self.module.forward.__code__.co_freevars, self.module.forward.__closure__))
        cells['original_forward'].cell_contents = self.module.forward
        with self.assertRaisesRegex(RuntimeError, 'FORWARD_WRAPPER_DEPTH_EXCEEDED'):
            binding.ForwardBinding(self.module)

    def test_arbitrary_wrapped_attribute_under_collector_is_not_trusted(self):
        original = self.module.forward
        def arbitrary(*args, **kwargs):
            return original(*args, **kwargs)
        arbitrary.__wrapped__ = original
        self.module.forward = arbitrary
        self.collector.wrap_forward_with_hook_safety(self.module)
        with self.assertRaisesRegex(RuntimeError, 'PERSISTENT_FORWARD_ORIGINAL_BOUND_METHOD_REQUIRED'):
            binding.ForwardBinding(self.module)

    def test_copied_factory_or_wrapper_code_cannot_use_foreign_globals(self):
        factory = self.collector.wrap_forward_with_hook_safety
        for copied in ('factory', 'wrapper'):
            with self.subTest(copied=copied):
                module = CpuForward()
                factory(module)
                if copied == 'factory':
                    self.collector.wrap_forward_with_hook_safety = FunctionType(factory.__code__, dict(factory.__globals__))
                else:
                    wrapper = module.forward
                    module.forward = FunctionType(wrapper.__code__, dict(wrapper.__globals__), closure=wrapper.__closure__)
                try:
                    with self.assertRaisesRegex(RuntimeError, 'PERSISTENT_FORWARD_COLLECTOR_GLOBALS_CHANGED'):
                        binding.ForwardBinding(module)
                finally:
                    self.collector.wrap_forward_with_hook_safety = factory


if __name__ == '__main__':
    unittest.main()
