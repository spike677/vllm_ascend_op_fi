"""Preserve reviewed msprobe custom-op wrappers under TorchDispatchMode.

The collector replaces an OpOverloadPacket with a function. PyTorch's dispatcher
still resolves namespace.packet.default when a mode is active. Supply only the
original packet's overload attributes on that same collector function. Neither
the collector callable nor its underlying operator implementation is replaced.
"""
import hashlib
import inspect
from pathlib import Path
import sys
from types import CodeType, FunctionType

from .serving_context import require

REGISTRY_MODULE = 'msprobe.core.dump.api_dump.api_registry'
APPROVED_REGISTRY_SHA256 = '1b6d5cc168034e994d3cb425474a318c8058366b28afe6d8014835ce4f98edeb'


def nested(code, path):
    for name in path:
        values = [value for value in code.co_consts if isinstance(value, CodeType) and value.co_name == name]
        require(len(values) == 1, 'COLLECTOR_NESTED_CODE_NOT_UNIQUE')
        code = values[0]
    return code


class CollectorPacketAttributes:
    def __init__(self):
        self.owned = []
        self.observed = []

    def install(self):
        import torch
        from torch._ops import OpOverloadPacket, _OpNamespace
        require(not self.owned, 'COLLECTOR_PACKET_ATTRIBUTES_ALREADY_OWNED')
        module = sys.modules.get(REGISTRY_MODULE)
        if module is None:
            return
        path = Path(module.__file__)
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == APPROVED_REGISTRY_SHA256,
                'UNREVIEWED_COLLECTOR_CUSTOM_OP_REGISTRY')
        factory = module.ApiRegistry.register_custom_api
        compiled = compile(raw, factory.__code__.co_filename, 'exec', dont_inherit=True)
        factories = ((factory, ('ApiRegistry', 'register_custom_api'), ('wrap_api_func', 'api_function')),
                     (module.ApiWrapper.wrap_api_func, ('ApiWrapper', 'wrap_api_func'), ('api_function',)))
        expected_wrappers = []
        for actual, factory_path, wrapper_path in factories:
            require(inspect.isfunction(actual) and Path(actual.__code__.co_filename).resolve() == path.resolve(),
                    'COLLECTOR_FACTORY_SOURCE_MISMATCH')
            require(actual.__code__ == nested(compiled, factory_path), 'COLLECTOR_FACTORY_CODE_CHANGED')
            expected_wrappers.append((nested(actual.__code__, wrapper_path), '.'.join(factory_path)))
        try:
            for namespace_name, namespace in tuple(vars(torch.ops).items()):
                if not isinstance(namespace, _OpNamespace):
                    continue
                for name, wrapper in tuple(vars(namespace).items()):
                    if not isinstance(wrapper, FunctionType) or wrapper.__code__.co_filename != factory.__code__.co_filename:
                        continue
                    matched = [label for code, label in expected_wrappers if wrapper.__code__ is code]
                    require(len(matched) == 1, 'COLLECTOR_CUSTOM_OP_WRAPPER_CODE_CHANGED:'
                            + namespace_name + '.' + name + ':' + wrapper.__code__.co_qualname)
                    closure = inspect.getclosurevars(wrapper).nonlocals
                    packet = closure.get('api_func')
                    if not isinstance(packet, OpOverloadPacket):
                        continue
                    require(closure.get('api_name') in (name, namespace_name + '.' + name)
                            and str(packet) == namespace_name + '.' + name,
                            'COLLECTOR_CUSTOM_OP_PACKET_OWNERSHIP_CHANGED:' + namespace_name + '.' + name)
                    for overload in packet.overloads():
                        value = getattr(packet, overload)
                        require(not hasattr(wrapper, overload) or getattr(wrapper, overload) is value,
                                'COLLECTOR_OVERLOAD_ATTRIBUTE_ALREADY_OWNED')
                        if not hasattr(wrapper, overload):
                            setattr(wrapper, overload, value)
                            self.owned.append((namespace, name, wrapper, overload, value))
                    self.observed.append(dict(namespace=namespace_name, operator=name,
                        overloads=packet.overloads(), registry_sha256=APPROVED_REGISTRY_SHA256,
                        factory=matched[0], public_api_name=closure.get('api_name'),
                        collector_callable_preserved=True, operator_callable_preserved=True))
        except BaseException:
            self.restore()
            raise

    def restore(self):
        for namespace, name, wrapper, overload, value in reversed(self.owned):
            require(getattr(namespace, name) is wrapper and getattr(wrapper, overload, None) is value,
                    'COLLECTOR_OVERLOAD_ATTRIBUTE_OWNER_CHANGED')
            delattr(wrapper, overload)
        self.owned.clear()

    def report(self):
        return dict(owned_attributes=len(self.owned), observations=self.observed,
                    source_bytes_changed=False, collector_call_replaced=False)
