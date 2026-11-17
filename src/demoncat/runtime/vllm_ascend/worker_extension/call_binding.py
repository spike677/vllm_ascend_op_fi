"""Bind actual hook arguments through the inspected msprobe forward wrapper.

The collector's variadic wrapper loses inspect.signature information. Retain
the original bound callable and require its exact identity underneath only the
known collector closure; no arbitrary __wrapped__ chain is trusted.
"""
import inspect
import sys
import types

from .serving_context import require


def same_callable(left, right):
    return (getattr(left, '__func__', left) is getattr(right, '__func__', right)
            and getattr(left, '__self__', None) is getattr(right, '__self__', None))


class ForwardBinding:
    def __init__(self, module):
        self.module = module
        self.original = module.forward
        self.signature = inspect.signature(self.original)

    def bind(self, module, args, kwargs):
        require(module is self.module, 'FOREIGN_FORWARD_BINDING_MODULE')
        actual = module.forward
        for _ in range(32):
            if same_callable(actual, self.original):
                return self.signature.bind(*args, **kwargs).arguments
            collector = sys.modules.get('msprobe.pytorch.dump.module_dump.module_processor')
            factory = getattr(collector, 'wrap_forward_with_hook_safety', None)
            require(isinstance(factory, types.FunctionType) and isinstance(actual, types.FunctionType),
                    'UNREVIEWED_FORWARD_WRAPPER')
            codes = [value for value in factory.__code__.co_consts
                     if isinstance(value, types.CodeType) and value.co_name == 'wrapped_forward']
            require(len(codes) == 1 and actual.__code__ is codes[0], 'UNREVIEWED_FORWARD_WRAPPER_CODE')
            cells = dict(zip(actual.__code__.co_freevars, actual.__closure__ or ()))
            require(set(cells) == {'module', 'original_forward'}
                    and cells['module'].cell_contents is module, 'FORWARD_WRAPPER_OWNER_CHANGED')
            actual = cells['original_forward'].cell_contents
        raise RuntimeError('FORWARD_WRAPPER_DEPTH_EXCEEDED')
