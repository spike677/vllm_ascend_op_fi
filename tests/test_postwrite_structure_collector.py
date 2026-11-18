"""Source-pinned collector coexistence; no generic decorator exemption."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from demoncat.runtime.vllm_ascend.worker_extension import call_binding
from fault_injection.deployments.postwrite_structure import structural_module_inventory
from tests import test_postwrite_coverage as contracts


SOURCE = '''def wrap_forward_with_hook_safety(module):
    original_forward = module.forward
    def wrapped_forward(*args, **kwargs):
        assert module is not None
        return original_forward(*args, **kwargs)
    module.forward = wrapped_forward
'''


@contextmanager
def collector():
    with tempfile.TemporaryDirectory() as folder:
        source=Path(folder)/'module_processor.py'; source.write_bytes(SOURCE.encode())
        module=types.ModuleType(call_binding.COLLECTOR_MODULE); module.__file__=str(source)
        exec(compile(SOURCE,str(source),'exec',dont_inherit=True),vars(module))
        with patch.dict(sys.modules,{call_binding.COLLECTOR_MODULE:module}), patch.object(
                call_binding,'REVIEWED_COLLECTOR_SHA256',hashlib.sha256(source.read_bytes()).hexdigest()):
            yield module,source


class PostwriteStructuralCollectorTests(unittest.TestCase):
    def prepare(self):
        return contracts.ProjectionCoverageTests().listed(contracts.ListedLayers())

    def test_persistent_collector_during_all_controls_and_off_keeps_identity(self):
        provider,x,group=self.prepare()
        with collector() as (module,_):
            @contextmanager
            def window(model):
                module.wrap_forward_with_hook_safety(model.layers)
                yield
            report=group.run('once',x,baseline_control=True,model_observation=window)
            self.assertTrue(report['all_targets_controlled_once'])
            self.assertTrue(report['recovery']['complete'])
            self.assertEqual(report['hits'],0)
            self.assertEqual(structural_module_inventory(provider.model,group.actual_module_calls),group.structural_inventory)
            self.assertEqual(provider.live_lease_count,0)

    def test_generic_wrapped_attribute_does_not_grant_structural_qualification(self):
        provider,_,group=self.prepare()
        module=provider.model.layers; original=module.forward
        def foreign(*args,**kwargs):return original(*args,**kwargs)
        foreign.__wrapped__=original;module.forward=foreign
        self.assertFalse(structural_module_inventory(provider.model,group.actual_module_calls)[0]['structural_noncall'])

    def test_source_changed_after_registration_refuses(self):
        provider,_,group=self.prepare()
        with collector() as (module,source):
            module.wrap_forward_with_hook_safety(provider.model.layers)
            source.write_bytes(source.read_bytes()+b'\n# changed\n')
            with self.assertRaisesRegex(Exception,'UNREVIEWED_PERSISTENT_FORWARD_WRAPPER_SOURCE'):
                structural_module_inventory(provider.model,group.actual_module_calls)

    def test_foreign_module_closure_refuses(self):
        provider,_,group=self.prepare()
        with collector() as (module,_):
            other=contracts.ListedLayers().layers;module.wrap_forward_with_hook_safety(other)
            provider.model.layers.forward=other.forward
            with self.assertRaisesRegex(Exception,'FORWARD_WRAPPER_OWNER_CHANGED'):
                structural_module_inventory(provider.model,group.actual_module_calls)

    def test_explicit_instance_original_is_still_not_a_reviewed_wrapper(self):
        provider,_,group=self.prepare();module=provider.model.layers
        module.forward=module.forward
        self.assertFalse(structural_module_inventory(provider.model,group.actual_module_calls)[0]['structural_noncall'])


if __name__=='__main__':unittest.main()
