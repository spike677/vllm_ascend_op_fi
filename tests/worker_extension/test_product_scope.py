"""Current public choices cannot inherit historical or group qualification."""
import json
import importlib
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fault_injection.catalog import list_faults
from fault_injection.delivery_scope import describe_delivery
from demoncat.runtime.vllm_ascend.worker_extension import registry


class ProductScopeTests(unittest.TestCase):
    def test_pure_catalog_and_fresh_status_do_not_claim_live_admission(self):
        root = Path(__file__).resolve().parents[2]
        code = ('import sys,json;from fault_injection.catalog import list_faults;'
                'rows=list_faults();assert len(rows)==17;'
                'assert "torch" not in sys.modules;assert "torch_npu" not in sys.modules;'
                'print(json.dumps(rows))')
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=root,
                                text=True, capture_output=True, check=True)
        rows = json.loads(result.stdout)
        self.assertEqual({r['name'] for r in rows}, {r['name'] for r in list_faults()})
        for row in rows:
            scope = row['current_delivery']
            self.assertFalse(scope['product_closed'])
            self.assertEqual(scope['main_project_status'], 'INNER_FRAMEWORK_INTEGRATION_PENDING')
            self.assertEqual(scope['live_deployment_status'], 'NOT_EVALUATED')
        changed = describe_delivery('39.output_proxy')
        changed['deployment_requirements'].clear()
        self.assertTrue(describe_delivery('39.output_proxy')['deployment_requirements'])

    def test_fault25_refuses_before_provider_import_for_every_version(self):
        with patch('importlib.import_module') as load:
            for version in ('1', 'unknown'):
                with self.assertRaisesRegex(RuntimeError, 'FAULT25_UNSUPPORTED_NO_SUBSTITUTE'):
                    registry.create_worker_provider('25.reduce_scatter', version, None, None, {})
            load.assert_not_called()

    def test_tnd_fault_is_unfinished_and_legal_variants_are_historical(self):
        self.assertEqual(describe_delivery('28.fia.denominator_replay')['state'],
                         'TND_FAULT_IMPLEMENTATION_INCOMPLETE')
        for name in ('28.fa.legal_tiling', '28.fia.legal_tiling'):
            self.assertFalse(describe_delivery(name)['selected_for_current_delivery'])

    def test_output39_public_route_rejects_group_before_native_or_owner(self):
        module = importlib.import_module('demoncat.runtime.vllm_ascend.worker_extension.providers.fault39_output')
        Output39ServingBinding = module.Output39ServingBinding
        for stage, leases in (('c1', 1), ('c1', 2), ('c2_once', 1), ('c2_c3', 1)):
            # Only inert constructor data; no native binding or tensor storage.
            values = {field.name: None for field in fields(Output39ServingBinding)}
            values.update(stage=stage, max_live_leases=leases, runtime_only=False)
            binding = Output39ServingBinding(**values)
            worker = SimpleNamespace(**{module._ATTRIBUTE: binding})
            with patch.object(module, 'Fault39OutputServingProvider') as factory, \
                    patch.object(module, 'read_native_binding') as readback:
                if stage == 'c1' and leases == 1:
                    registry.create_worker_provider('39.output_proxy', '1', worker, 'model', {})
                    factory.assert_called_once_with(worker, 'model', {})
                else:
                    with self.assertRaisesRegex(RuntimeError, 'SINGLE_REQUEST_SINGLE_TARGET_REQUIRED'):
                        registry.create_worker_provider('39.output_proxy', '1', worker, 'model', {})
                    factory.assert_not_called()
                readback.assert_not_called()

    def test_paged33_group_is_not_implicitly_selected(self):
        module = importlib.import_module('demoncat.runtime.vllm_ascend.worker_extension.providers.paged33_s2')
        from demoncat.runtime.vllm_ascend.worker_extension.paged33_group_native import GroupEffectsNativeCall
        native = object.__new__(GroupEffectsNativeCall)
        worker = SimpleNamespace(**{module.ATTRIBUTE: {'33.paged_decode': SimpleNamespace(native=native)}})
        with patch.object(module, 'resolve_paged_decode_provider') as resolve:
            with self.assertRaisesRegex(RuntimeError, 'PAGED33_CURRENT_DELIVERY_SINGLE_SITE_REQUIRED'):
                registry.create_worker_provider('33.paged_decode', '1', worker, 'model', {})
            resolve.assert_not_called()


if __name__ == '__main__': unittest.main()
