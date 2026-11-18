"""Preserve dispatcher metadata on exact msprobe-generated custom wrappers.

This edits no package source, callable or statistic. It mirrors the collector's
normal API_ATTR_LIST handling for its custom-API wrappers in a disposable Worker.
Direct OpOverload invocation is not a new collector boundary.
"""
import hashlib
import inspect
from pathlib import Path
import sys
import types

REGISTRY = 'core/dump/api_dump/api_registry.py'
REGISTRY_SHA = '1b6d5cc168034e994d3cb425474a318c8058366b28afe6d8014835ce4f98edeb'


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def codes(code):
    yield code
    for value in code.co_consts:
        if isinstance(value, types.CodeType):
            yield from codes(value)


class CustomPacketMetadata:
    def __init__(self, collector, package, files):
        import torch
        self.collector, self.rows = collector, []
        path = Path(package)/REGISTRY
        raw = path.read_bytes()
        require(files.get(REGISTRY) == REGISTRY_SHA
                and hashlib.sha256(raw).hexdigest() == REGISTRY_SHA,
                'UNREVIEWED_CUSTOM_REGISTRY_SOURCE')
        expected = [code for code in codes(compile(raw, str(path), 'exec', dont_inherit=True))
                    if code.co_qualname == 'ApiRegistry.register_custom_api.<locals>.wrap_api_func.<locals>.api_function']
        require(len(expected) == 1, 'EXACT_CUSTOM_WRAPPER_CODE_REQUIRED')
        expected_code = expected[0]
        keys = collector._custom_op_auto_registered
        require(type(keys) is set, 'CUSTOM_REGISTRATION_SET_REQUIRED')
        pending = []
        for namespace, name in sorted(keys):
            module = getattr(torch.ops, namespace)
            wrapper = getattr(module, name)
            require(type(wrapper) is types.FunctionType
                    and wrapper.__code__ == expected_code
                    and Path(wrapper.__code__.co_filename).resolve() == path.resolve(),
                    'UNKNOWN_CUSTOM_API_WRAPPER')
            registry = sys.modules.get(wrapper.__globals__.get('__name__'))
            require(registry is not None and wrapper.__globals__ is vars(registry)
                    and Path(registry.__file__).resolve() == path.resolve(),
                    'CUSTOM_REGISTRY_GLOBALS_CHANGED')
            closure = inspect.getclosurevars(wrapper).nonlocals
            packet = closure.get('api_func')
            require(type(packet) is torch._ops.OpOverloadPacket
                    and str(packet) == namespace+'.'+name
                    and closure.get('api_name') == name and closure.get('prefix') == namespace,
                    'ORIGINAL_CUSTOM_PACKET_NOT_BOUND')
            default = getattr(packet, 'default', None)
            require(type(default) is torch._ops.OpOverload and default.overloadpacket is packet,
                    'EXACT_DEFAULT_OVERLOAD_REQUIRED')
            require(not hasattr(wrapper, 'default') or wrapper.default is default,
                    'FOREIGN_DEFAULT_ATTRIBUTE_MUST_NOT_BE_REPLACED')
            pending.append((namespace, name, module, wrapper, packet, default, not hasattr(wrapper, 'default')))
        # Complete validation precedes mutation; no foreign attributes overwritten.
        self.keys = frozenset(keys)
        for namespace, name, module, wrapper, packet, default, added in pending:
            if added:
                wrapper.default = default
            self.rows.append(dict(namespace=namespace, name=name, module=module,
                                  wrapper=wrapper, packet=packet, default=default, added=added))
        self.verify()

    def verify(self):
        require(frozenset(self.collector._custom_op_auto_registered) == self.keys,
                'CUSTOM_API_SET_CHANGED_AFTER_CAPTURE_SETUP')
        for row in self.rows:
            require(getattr(row['module'], row['name']) is row['wrapper']
                    and row['wrapper'].default is row['default']
                    and row['default'].overloadpacket is row['packet'],
                    'CUSTOM_PACKET_METADATA_OWNER_CHANGED')

    def report(self):
        self.verify()
        return dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    registry_sha256=REGISTRY_SHA, package_bytes_edited=False,
                    generated_wrapper_metadata_changed=True, callable_replaced=False,
                    dispatcher_default_is_original=True, raw_statistics_rewritten=False,
                    direct_overload_capture_added=False, restored_on_process_exit_only=True,
                    operators=[dict(namespace=r['namespace'], name=r['name'],
                                    attribute='default', added=r['added'], schema=str(r['default']._schema))
                               for r in self.rows])
