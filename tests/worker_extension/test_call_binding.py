"""CPU fixtures for the inspected collector wrapper contract, not msprobe/NPU acceptance.

The fixture deliberately has the observed variadic signature and the two closure
cells ``module`` / ``original_forward``. It is installed in a fake module table;
these tests do not claim to import or validate an installed msprobe package.
"""
import functools
import inspect
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from demoncat.runtime.vllm_ascend.worker_extension.call_binding import ForwardBinding
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import ContextRefused


COLLECTOR_MODULE = 'msprobe.pytorch.dump.module_dump.module_processor'


def wrap_forward_with_hook_safety(module, original_forward):
    # Deliberately no functools.wraps: match the signature-loss failure mode.
    def wrapped_forward(*args, **kwargs):
        module._cpu_wrapper_calls = getattr(module, '_cpu_wrapper_calls', 0) + 1
        return original_forward(*args, **kwargs)
    return wrapped_forward


def collector_fixture(factory=wrap_forward_with_hook_safety):
    module = ModuleType(COLLECTOR_MODULE)
    module.wrap_forward_with_hook_safety = factory
    return module


class CpuForward:
    def __init__(self):
        self.executions = 0

    def forward(self, input_ids, positions, *, optional=None):
        self.executions += 1
        return input_ids, positions, optional


class ForwardBindingTests(unittest.TestCase):
    def setUp(self):
        self.collector = collector_fixture()
        self.enterContext(patch.dict(sys.modules, {COLLECTOR_MODULE: self.collector}))
        self.module = CpuForward()
        self.binding = ForwardBinding(self.module)
        self.left, self.right = object(), object()

    def bind(self):
        return self.binding.bind(self.module, (self.left,), {'positions': self.right})

    def wrap(self, module=None, original=None):
        module = self.module if module is None else module
        original = module.forward if original is None else original
        module.forward = wrap_forward_with_hook_safety(module, original)
        return module.forward

    def test_unwrapped_original_does_not_require_collector_or_execute_forward(self):
        with patch.dict(sys.modules, {COLLECTOR_MODULE: None}):
            arguments = self.bind()
        self.assertIs(arguments['input_ids'], self.left)
        self.assertIs(arguments['positions'], self.right)
        self.assertNotIn('optional', arguments)
        self.assertEqual(self.module.executions, 0)

    def test_fresh_bound_method_is_the_same_original_callable(self):
        self.assertIsNot(self.module.forward, self.binding.original)
        self.assertEqual(set(self.bind()), {'input_ids', 'positions'})

    def test_variadic_wrapper_recovers_original_parameter_names_without_execution(self):
        wrapped = self.wrap()
        self.assertEqual(str(inspect.signature(wrapped)), '(*args, **kwargs)')
        arguments = self.bind()
        self.assertIs(arguments['input_ids'], self.left)
        self.assertIs(arguments['positions'], self.right)
        self.assertEqual(self.module.executions, 0)
        self.assertFalse(hasattr(self.module, '_cpu_wrapper_calls'))

    def test_kwargs_preserve_identity_and_original_keyword_only_signature(self):
        self.wrap()
        optional = object()
        actual = self.binding.bind(self.module, (), {
            'input_ids': self.left, 'positions': self.right, 'optional': optional})
        self.assertIs(actual['optional'], optional)
        for args, kwargs in (((self.left,), {}), ((self.left, self.right, optional), {}),
                             ((self.left, self.right), {'unexpected': 1})):
            with self.subTest(args=args, kwargs=kwargs), self.assertRaises(TypeError):
                self.binding.bind(self.module, args, kwargs)

    def test_nested_owned_wrappers_and_restored_original_are_accepted(self):
        for _ in range(4):
            self.wrap()
        self.assertEqual(set(self.bind()), {'input_ids', 'positions'})
        self.module.forward = self.binding.original
        self.assertEqual(set(self.bind()), {'input_ids', 'positions'})

    def test_other_module_cannot_reuse_binding_even_with_same_class(self):
        with self.assertRaisesRegex(ContextRefused, 'FOREIGN_FORWARD_BINDING_MODULE'):
            self.binding.bind(CpuForward(), (self.left, self.right), {})

    def test_wrapper_closure_module_must_be_actual_owner(self):
        self.module.forward = wrap_forward_with_hook_safety(CpuForward(), self.binding.original)
        with self.assertRaisesRegex(ContextRefused, 'FORWARD_WRAPPER_OWNER_CHANGED'):
            self.bind()

    def test_wrapper_with_another_modules_original_is_rejected(self):
        self.wrap(original=CpuForward().forward)
        with self.assertRaisesRegex(ContextRefused, 'UNREVIEWED_FORWARD_WRAPPER'):
            self.bind()

    def test_unreviewed_wrapper_cannot_gain_trust_through_wrapped_attribute(self):
        original = self.binding.original
        @functools.wraps(original)
        def arbitrary_wrapper(*args, **kwargs):
            return original(*args, **kwargs)
        self.module.forward = arbitrary_wrapper
        self.assertEqual(inspect.signature(arbitrary_wrapper), self.binding.signature)
        with self.assertRaisesRegex(ContextRefused, 'UNREVIEWED_FORWARD_WRAPPER_CODE'):
            self.bind()

    def test_expected_wrapper_is_rejected_when_collector_factory_is_absent_or_not_function(self):
        self.wrap()
        for factory in (None, object()):
            with self.subTest(factory=factory):
                self.collector.wrap_forward_with_hook_safety = factory
                with self.assertRaisesRegex(ContextRefused, 'UNREVIEWED_FORWARD_WRAPPER'):
                    self.bind()

    def test_same_named_code_from_a_different_factory_is_rejected(self):
        def other_factory(module, original_forward):
            def wrapped_forward(*args, **kwargs):
                if module is None:
                    raise RuntimeError('unexpected')
                return original_forward(*args, **kwargs)
            return wrapped_forward
        self.module.forward = other_factory(self.module, self.binding.original)
        with self.assertRaisesRegex(ContextRefused, 'UNREVIEWED_FORWARD_WRAPPER_CODE'):
            self.bind()

    def test_extra_closure_cell_is_refused_even_for_matching_factory_code(self):
        def extra_factory(module, original_forward):
            extra = object()
            def wrapped_forward(*args, **kwargs):
                if extra is module:
                    raise RuntimeError('unexpected')
                return original_forward(*args, **kwargs)
            return wrapped_forward
        self.collector.wrap_forward_with_hook_safety = extra_factory
        self.module.forward = extra_factory(self.module, self.binding.original)
        with self.assertRaisesRegex(ContextRefused, 'FORWARD_WRAPPER_OWNER_CHANGED'):
            self.bind()

    def test_ambiguous_factory_nested_code_is_refused(self):
        def ambiguous_factory(module, original_forward):
            if module is None:
                def wrapped_forward(*args, **kwargs):
                    return original_forward(*args, **kwargs)
            else:
                def wrapped_forward(*args, **kwargs):
                    module._cpu_wrapper_calls = 0
                    return original_forward(*args, **kwargs)
            return wrapped_forward
        self.collector.wrap_forward_with_hook_safety = ambiguous_factory
        self.module.forward = ambiguous_factory(self.module, self.binding.original)
        with self.assertRaisesRegex(ContextRefused, 'UNREVIEWED_FORWARD_WRAPPER_CODE'):
            self.bind()

    def test_missing_closure_owner_cell_fails_closed_without_forward_execution(self):
        wrapped = self.wrap()
        cells = dict(zip(wrapped.__code__.co_freevars, wrapped.__closure__))
        del cells['module'].cell_contents
        with self.assertRaises(ValueError):
            self.bind()
        self.assertEqual(self.module.executions, 0)

    def test_cyclic_wrapper_cannot_escape_bounded_unwrapping(self):
        wrapped = self.wrap()
        cells = dict(zip(wrapped.__code__.co_freevars, wrapped.__closure__))
        cells['original_forward'].cell_contents = wrapped
        with self.assertRaisesRegex(RuntimeError, 'FORWARD_WRAPPER_DEPTH_EXCEEDED'):
            self.bind()
        self.assertEqual(self.module.executions, 0)

    def test_deep_wrapper_chain_is_bounded(self):
        for _ in range(31):
            self.wrap()
        self.assertEqual(set(self.bind()), {'input_ids', 'positions'})
        self.wrap()
        with self.assertRaisesRegex(RuntimeError, 'FORWARD_WRAPPER_DEPTH_EXCEEDED'):
            self.bind()


if __name__ == '__main__':
    unittest.main(verbosity=2)
