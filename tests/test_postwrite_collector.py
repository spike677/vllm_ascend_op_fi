"""CPU-only ownership contracts. Fixture hashes never authorize production collectors."""
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from demoncat.runtime.vllm_ascend.worker_extension import postwrite_collector as cleanup
from experiments.c2c3_postwrite_20261117.natural41_probe import record_cleanup_failure

REGISTRY = '''class ApiWrapper:
    def wrap_api_func(self, api_name, api_func, prefix, hook_build_func, api_template):
        api_instance = api_template(api_name, api_func, prefix, hook_build_func)
        def api_function(*args, **kwargs):
            if prefix == 'passthrough':
                return api_func(*args, **kwargs)
            return api_instance(*args, **kwargs), api_name
        return api_function
class ApiRegistry:
    def __init__(self):
        self.ori_api_attr = {}
        self.wrapped_api_attr = {}
        self.inner_used_ori_attr = {}
        self.inner_used_wrapped_attr = {}
    @staticmethod
    def register_custom_api(module, api_name, api_prefix, hook_build_func, api_template):
        def wrap_api_func(api_name, api_func, prefix, hook_build_func, api_template):
            def api_function(*args, **kwargs):
                return api_template(api_name, api_func, prefix, hook_build_func)(*args, **kwargs)
            return api_function
        setattr(module, api_name, wrap_api_func(api_name, getattr(module, api_name), api_prefix, hook_build_func, api_template))
'''
TEMPLATE = '''class ApiTemplate:
    def __init__(self, api_name, api_func, prefix, hook_build_func):
        self.api_name, self.api_func = api_name, api_func
    def __call__(self, *args, **kwargs):
        return self.api_func(*args, **kwargs)
api_register = None
'''


@contextmanager
def collector_fixture():
    with tempfile.TemporaryDirectory() as tmp:
        mods = []
        for name, source, filename in ((cleanup.REGISTRY_MODULE, REGISTRY, 'registry.py'),
                                       (cleanup.TEMPLATE_MODULE, TEMPLATE, 'template.py')):
            path = Path(tmp) / filename
            path.write_text(source)
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            mods.append(module)
        registry, template = mods
        template.api_register = registry.ApiRegistry()
        with patch.dict(sys.modules, {cleanup.REGISTRY_MODULE:registry, cleanup.TEMPLATE_MODULE:template}), \
             patch.object(cleanup, 'REGISTRY_SHA256', hashlib.sha256((Path(tmp)/'registry.py').read_bytes()).hexdigest()), \
             patch.object(cleanup, 'TEMPLATE_SHA256', hashlib.sha256((Path(tmp)/'template.py').read_bytes()).hexdigest()):
            yield registry, template


class CollectorOwnershipTests(unittest.TestCase):
    def test_direct_descriptors_restore_together_and_retry_is_idempotent(self):
        original = lambda:'original'
        installed = lambda:'owned'
        api = NS(call=installed)
        class Device:pass
        old, new = staticmethod(original), staticmethod(installed)
        Device.write = new
        patches = [(api, 'call', original, installed), (Device, 'write', old, new)]
        cleanup.restore_owned_api_patches(patches)
        self.assertIs(api.call, original)
        self.assertIs(vars(Device)['write'], old)
        cleanup.restore_owned_api_patches(patches)

    def test_standard_wrapper_kept_but_owned_links_and_registry_original_removed(self):
        with collector_fixture() as (registry, template):
            original = lambda:'original'
            installed = lambda:'owned'
            api = NS(call=installed)
            wrapped = registry.ApiWrapper().wrap_api_func('call', installed, 'NPU', None, template.ApiTemplate)
            api.call = wrapped
            register = template.api_register
            register.ori_api_attr = {'pytorch.NPU':{'call':[([api], installed)]}}
            register.wrapped_api_attr = {'pytorch.NPU':{'call':[([api], wrapped)]}}
            report = cleanup.restore_owned_api_patches([(api, 'call', original, installed)])
            self.assertIs(api.call, wrapped)
            self.assertEqual(api.call(), ('original', 'call'))
            self.assertIs(register.ori_api_attr['pytorch.NPU']['call'][0][1], original)
            self.assertTrue(report['collector_callables_preserved'])
            cleanup.restore_owned_api_patches([(api, 'call', original, installed)])

    def test_custom_nested_wrapper_and_service_restore_reference_are_preserved(self):
        with collector_fixture() as (registry, template):
            original = lambda:'original'
            installed = lambda:'owned'
            api = NS(call=installed)
            registry.ApiRegistry.register_custom_api(api, 'call', 'NPU', None, template.ApiTemplate)
            first = api.call
            registry.ApiRegistry.register_custom_api(api, 'call', 'NPU', None, template.ApiTemplate)
            outer = api.call
            mapping = {str(api) + '.call':installed}
            collector = NS(service=NS(api_register=template.api_register, ori_customer_func=mapping))
            cleanup.restore_owned_api_patches([(api, 'call', original, installed)], collectors=[collector])
            self.assertIs(api.call, outer)
            self.assertEqual(api.call(), 'original')
            self.assertEqual(first(), 'original')
            self.assertIs(next(iter(mapping.values())), original)

    def test_foreign_api_refuses_before_restoring_any_other_patch(self):
        original, installed, foreign = lambda:1, lambda:2, lambda:3
        api = NS(first=foreign, second=installed)
        with self.assertRaises(RuntimeError):
            cleanup.restore_owned_api_patches([(api,'first',original,installed),(api,'second',original,installed)])
        self.assertIs(api.first, foreign)
        self.assertIs(api.second, installed)

    def test_changed_api_instance_refuses_without_partial_mutation(self):
        with collector_fixture() as (registry, template):
            original, installed, foreign = lambda:1, lambda:2, lambda:3
            wrapped = registry.ApiWrapper().wrap_api_func('call', installed, 'NPU', None, template.ApiTemplate)
            cells = dict(zip(wrapped.__code__.co_freevars, wrapped.__closure__))
            cells['api_instance'].cell_contents.api_func = foreign
            api = NS(call=wrapped, other=installed)
            with self.assertRaisesRegex(RuntimeError, 'TEMPLATE_TARGET_CHANGED'):
                cleanup.restore_owned_api_patches([(api,'call',original,installed),(api,'other',original,installed)])
            self.assertIs(api.other, installed)
            self.assertIs(cells['api_func'].cell_contents, installed)

    def test_changed_collector_source_or_original_is_not_trusted(self):
        with collector_fixture() as (registry, template):
            original, installed, foreign = lambda:1, lambda:2, lambda:3
            api = NS(call=registry.ApiWrapper().wrap_api_func('call', foreign, 'NPU', None, template.ApiTemplate))
            with self.assertRaises(RuntimeError):
                cleanup.restore_owned_api_patches([(api,'call',original,installed)])
            with patch.object(cleanup, 'REGISTRY_SHA256', '0'*64):
                with self.assertRaisesRegex(RuntimeError, 'UNREVIEWED_COLLECTOR_SOURCE'):
                    cleanup.restore_owned_api_patches([(api,'call',original,installed)])

    def test_cleanup_failure_does_not_overwrite_primary_evidence(self):
        report = dict(status='FAILED_WITH_EVIDENCE', error='primary', traceback='original trace')
        record_cleanup_failure(report, RuntimeError('cleanup'))
        self.assertEqual(report['status'], 'FAILED_WITH_EVIDENCE')
        self.assertEqual(report['error'], 'primary')
        self.assertEqual(report['traceback'], 'original trace')
        self.assertEqual(report['cleanup_status'], 'FAILED')


if __name__ == '__main__':unittest.main()
