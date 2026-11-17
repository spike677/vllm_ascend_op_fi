"""Bind actual hook arguments through the inspected msprobe forward wrapper.

The collector's variadic wrapper loses inspect.signature information. Retain
the original bound callable and require its exact identity underneath only the
known collector closure; no arbitrary __wrapped__ chain is trusted.
"""
import inspect
import hashlib
from pathlib import Path
import sys
import types

from .serving_context import require


COLLECTOR_MODULE = 'msprobe.pytorch.dump.module_dump.module_processor'
REVIEWED_COLLECTOR_SHA256 = '4876e89f7f726cd552fc1bf0e7b7745a898461deec2cb2abf33b02deba7329d7'


def original_before_collector(module):
    """Recover a prior session's original through only the pinned collector.

Collector stop may retain its instance wrapper. Freezing that variadic wrapper
as the next session's original would lose the actual semantic argument names.
No arbitrary wrapper or __wrapped__ attribute grants access to an original.
"""
    actual = module.forward
    reviewed = False
    for _ in range(32):
        collector = sys.modules.get(COLLECTOR_MODULE)
        factory = getattr(collector, 'wrap_forward_with_hook_safety', None)
        if not isinstance(factory, types.FunctionType) or not isinstance(actual, types.FunctionType):
            require(not reviewed or (inspect.ismethod(actual) and actual.__self__ is module),
                    'PERSISTENT_FORWARD_ORIGINAL_BOUND_METHOD_REQUIRED')
            return actual
        codes = [value for value in factory.__code__.co_consts
                 if isinstance(value, types.CodeType) and value.co_name == 'wrapped_forward']
        if len(codes) != 1 or actual.__code__ is not codes[0]:
            require(not reviewed, 'PERSISTENT_FORWARD_ORIGINAL_BOUND_METHOD_REQUIRED')
            return actual
        if not reviewed:
            source = Path(getattr(collector, '__file__', ''))
            require(source.is_file() and Path(factory.__code__.co_filename).resolve() == source.resolve(),
                    'PERSISTENT_FORWARD_WRAPPER_SOURCE_UNPROVEN')
            raw = source.read_bytes()
            require(hashlib.sha256(raw).hexdigest() == REVIEWED_COLLECTOR_SHA256,
                    'UNREVIEWED_PERSISTENT_FORWARD_WRAPPER_SOURCE')
            compiled = compile(raw, factory.__code__.co_filename, 'exec', dont_inherit=True)
            factories = [code for code in compiled.co_consts
                         if isinstance(code, types.CodeType) and code.co_name == 'wrap_forward_with_hook_safety']
            require(len(factories) == 1 and factory.__code__ == factories[0],
                    'PERSISTENT_FORWARD_FACTORY_CODE_CHANGED')
            reviewed = True
        cells = dict(zip(actual.__code__.co_freevars, actual.__closure__ or ()))
        require(set(cells) == {'module', 'original_forward'} and cells['module'].cell_contents is module,
                'FORWARD_WRAPPER_OWNER_CHANGED')
        actual = cells['original_forward'].cell_contents
        require(getattr(actual, '__self__', module) is module, 'PERSISTENT_FORWARD_ORIGINAL_OWNER_CHANGED')
    raise RuntimeError('FORWARD_WRAPPER_DEPTH_EXCEEDED')


def same_callable(left, right):
    return (getattr(left, '__func__', left) is getattr(right, '__func__', right)
            and getattr(left, '__self__', None) is getattr(right, '__self__', None))


class ForwardBinding:
    def __init__(self, module):
        self.module = module
        self.original = original_before_collector(module)
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
