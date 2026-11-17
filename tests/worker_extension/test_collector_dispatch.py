"""Real CPU dispatcher calls; no collector statistics or NPU qualification."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch
from uuid import uuid4

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from demoncat.runtime.vllm_ascend.worker_extension import collector_dispatch as adapter


REGISTRY = '''class ApiRegistry:
    @staticmethod
    def register_custom_api(module, api_name, api_prefix, hook_build_func, api_template):
        def wrap_api_func(api_name, api_func, prefix, hook_build_func, api_template):
            def api_function(*args, **kwargs):
                return api_template(api_name, api_func, prefix, hook_build_func)(*args, **kwargs)
            api_function.__name__ = api_name
            return api_function
        setattr(module, api_name, wrap_api_func(api_name, getattr(module, api_name),
                api_prefix, hook_build_func, api_template))
'''


class CountingMode(TorchDispatchMode):
    def __init__(self):
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.calls.append(str(func))
        return func(*args, **(kwargs or {}))


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'api_registry.py'
        path.write_text(REGISTRY, encoding='utf8')
        raw = path.read_bytes()
        module = ModuleType(adapter.REGISTRY_MODULE)
        module.__file__ = str(path)
        exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
        name = 'collector_contract_' + uuid4().hex
        library = torch.library.Library(name, 'DEF')
        library.define('identity(Tensor x) -> Tensor')
        counts = {'native': 0, 'collector': 0}

        def original(x):
            counts['native'] += 1
            return x + 1

        library.impl('identity', original, 'CPU')
        namespace = getattr(torch.ops, name)
        packet = namespace.identity

        def template(api_name, api_func, prefix, hook):
            def collect(*args, **kwargs):
                counts['collector'] += 1
                return api_func(*args, **kwargs)
            return collect

        module.ApiRegistry.register_custom_api(namespace, 'identity', name, None, template)
        wrapper = namespace.identity
        with patch.dict(sys.modules, {adapter.REGISTRY_MODULE: module}), patch.object(
                adapter, 'APPROVED_REGISTRY_SHA256', hashlib.sha256(raw).hexdigest()):
            try:
                yield module, namespace, packet, wrapper, counts, path, raw
            finally:
                namespace.identity = packet
                library._destroy()


class CollectorDispatcherTests(unittest.TestCase):
    def test_preserves_actual_collector_and_operator_calls_under_dispatch(self):
        with fixture() as (_, namespace, packet, wrapper, counts, path, raw):
            x = torch.tensor([1.0, 2.0])
            with self.assertRaisesRegex(AttributeError, 'default'), CountingMode():
                namespace.identity(x)
            counts.update(native=0, collector=0)
            owned = adapter.CollectorPacketAttributes()
            owned.install()
            self.assertIs(namespace.identity, wrapper)
            self.assertIs(wrapper.default, packet.default)
            with CountingMode() as mode:
                result = namespace.identity(x)
            self.assertEqual(result.tolist(), [2.0, 3.0])
            self.assertEqual(counts, {'native': 1, 'collector': 1})
            self.assertEqual(mode.calls, [str(packet.default)])
            self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(owned.report()['owned_attributes'], 1)
            owned.restore()
            self.assertIs(namespace.identity, wrapper)
            self.assertFalse(hasattr(wrapper, 'default'))
            self.assertEqual(owned.report()['owned_attributes'], 0)

    def test_changed_source_or_factory_cannot_get_overload_attributes(self):
        for kind in ('source', 'factory'):
            with self.subTest(kind=kind), fixture() as (module, _, _, wrapper, _, path, _):
                if kind == 'source':
                    path.write_text(REGISTRY + '\n# unreviewed\n', encoding='utf8')
                else:
                    module.ApiRegistry.register_custom_api = lambda *args: None
                owned = adapter.CollectorPacketAttributes()
                with self.assertRaises(RuntimeError):
                    owned.install()
                self.assertFalse(hasattr(wrapper, 'default'))
                self.assertEqual(owned.report()['owned_attributes'], 0)

    def test_foreign_overload_is_preserved_and_refused(self):
        with fixture() as (_, _, _, wrapper, _, _, _):
            foreign = object()
            wrapper.default = foreign
            owned = adapter.CollectorPacketAttributes()
            with self.assertRaisesRegex(RuntimeError, 'ATTRIBUTE_ALREADY_OWNED'):
                owned.install()
            self.assertIs(wrapper.default, foreign)
            self.assertEqual(owned.report()['owned_attributes'], 0)

    def test_changed_owner_during_restore_is_not_deleted(self):
        with fixture() as (_, _, packet, wrapper, _, _, _):
            owned = adapter.CollectorPacketAttributes()
            owned.install()
            foreign = object()
            wrapper.default = foreign
            with self.assertRaisesRegex(RuntimeError, 'ATTRIBUTE_OWNER_CHANGED'):
                owned.restore()
            self.assertIs(wrapper.default, foreign)
            wrapper.default = packet.default
            owned.restore()
            self.assertFalse(hasattr(wrapper, 'default'))


if __name__ == '__main__':
    unittest.main()
