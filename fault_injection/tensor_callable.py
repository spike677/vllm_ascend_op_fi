"""Bind the actual module dispatch callable, not just CustomOp.forward text.

Only the separately reviewed msprobe module wrapper may be traversed. Generic
``__wrapped__``/decorator chains, free functions and foreign bound instances are
not accepted. A new implementation needs its own observed qualification.
"""
import hashlib
import json
import marshal
from pathlib import Path
import sys
import types

from .core import UnsupportedCall
from .module_faults import _original_standard_forward


def _literal(value):
    if value is None or type(value) in (bool, int, float, str):
        return value
    if type(value) is tuple:
        return [_literal(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _literal(item) for key, item in sorted(value.items())}
    raise UnsupportedCall('MUTABLE_OR_UNREVIEWED_CALLABLE_DEFAULT')


def _method(module, name):
    method = _original_standard_forward(module) if name == 'forward' else getattr(module, name, None)
    function = getattr(method, '__func__', None)
    if (not isinstance(function, types.FunctionType) or getattr(method, '__self__', None) is not module):
        raise UnsupportedCall('ACTUAL_BOUND_TENSOR_CALLABLE_REQUIRED:' + name)
    if function.__closure__:
        raise UnsupportedCall('UNREVIEWED_TENSOR_CALLABLE_CLOSURE:' + name)
    namespace = sys.modules.get(function.__module__)
    if namespace is None or function.__globals__ is not vars(namespace):
        raise UnsupportedCall('TENSOR_CALLABLE_GLOBAL_OWNER_CHANGED:' + name)
    source = Path(function.__code__.co_filename).resolve()
    if source.suffix != '.py' or not source.is_file():
        raise UnsupportedCall('TENSOR_CALLABLE_SOURCE_FILE_REQUIRED:' + name)
    row = dict(name=name, module=function.__module__, qualname=function.__qualname__,
        source_file=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(marshal.dumps(function.__code__)).hexdigest(),
        first_line=function.__code__.co_firstlineno,
        defaults=_literal(function.__defaults__), kwdefaults=_literal(function.__kwdefaults__))
    return function, row


class TensorCallableGuard:
    """Live object/function/code and exact source guard for one model module."""
    def __init__(self, module, *, extra_methods=(), bind_optional_dispatch=True):
        self.module = module
        forward, _ = _method(module, 'forward')
        self.names = ('forward', '_forward_method') if (
            '_forward_method' in forward.__code__.co_names or
            bind_optional_dispatch and hasattr(module, '_forward_method')) else ('forward',)
        self.names += tuple(extra_methods)
        if len(set(self.names)) != len(self.names):
            raise UnsupportedCall('DUPLICATE_TENSOR_CALLABLE_BOUNDARY')
        self._functions = {}
        self._codes = {}
        self.sources = []
        for name in self.names:
            function, source = _method(module, name)
            self._functions[name] = function
            self._codes[name] = function.__code__
            self.sources.append(source)
        self.source_sha256 = self._digest(self.sources)
        self.callable_identity = ':'.join(name+'='+str(id(self._functions[name])) for name in self.names)

    @staticmethod
    def _digest(sources):
        return hashlib.sha256(json.dumps(sources, sort_keys=True, allow_nan=False,
                                         separators=(',', ':')).encode()).hexdigest()

    def verify(self):
        current = []
        for name in self.names:
            function, source = _method(self.module, name)
            if function is not self._functions[name] or function.__code__ is not self._codes[name]:
                raise UnsupportedCall('TENSOR_ACTUAL_DISPATCH_CALLABLE_CHANGED:' + name)
            current.append(source)
        if self._digest(current) != self.source_sha256:
            raise UnsupportedCall('TENSOR_ACTUAL_DISPATCH_SOURCE_OR_DEFAULTS_CHANGED')
        return True

    def report(self):
        return dict(module_object_id=id(self.module), callable_identity=self.callable_identity,
                    source_sha256=self.source_sha256, methods=self.sources,
                    wrapper_policy='Exact reviewed msprobe wrapper only; no generic unwrap')
