"""Optional device-free integration with the actual pinned msprobe package.

Run in root's CPU-only production image with the reviewed package mounted.
Unavailable package is a local skip, never evidence that this integration ran.
An available but incompatible collector fails rather than being skipped.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import test_core_tensor_serving as core
from experiments.worker_core_v11_20261118.core_tensor_statistics_capture import (
    PublicTensorStatisticsSidecar, RUNTIME_FILES, sha)
from experiments.worker_core_v11_20261118.review_core_tensor_statistics import review


@unittest.skipUnless(importlib.util.find_spec('msprobe'), 'Pinned msprobe package is not installed locally')
class ActualMsprobeCpuTests(unittest.TestCase):
    def test_real_pinned_collector_three_steps_on_public_cpu_protocol(self):
        library = torch.library.Library('_C_ascend', 'FRAGMENT')
        library.define('dcat_metadata_probe(Tensor x) -> Tensor')
        calls = []
        def implementation(x):
            calls.append('original')
            return x.clone() + 2
        library.impl('dcat_metadata_probe', implementation, 'CPU')
        packet = torch.ops._C_ascend.dcat_metadata_probe
        dispatches = []
        class Observer(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func is packet.default:
                    dispatches.append(func)
                return func(*args, **(kwargs or {}))
        case = core.CoreTensorServingTests();case.setUp()
        self.addCleanup(case.tearDown)
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        root = Path(__file__).resolve().parents[1]
        pins = {str(root/name): sha(root/name) for name in RUNTIME_FILES}
        case.start()
        sidecar = PublicTensorStatisticsSidecar(case.provider.session, folder, source_pins=pins)
        self.addCleanup(sidecar.close)
        wrapper = torch.ops._C_ascend.dcat_metadata_probe
        self.assertIsNot(wrapper, packet)
        self.assertIs(wrapper.default, packet.default)
        with Observer():
            value = wrapper(torch.tensor([1.0, 3.0]))
        self.assertEqual(value.tolist(), [3.0, 5.0])
        self.assertEqual(calls, ['original'])
        self.assertEqual(dispatches, [packet.default])
        # Reproduce the observed failure without changing either callable.
        del wrapper.default
        try:
            with self.assertRaisesRegex(AttributeError, 'default'), Observer():
                wrapper(torch.tensor([1.0]))
        finally:
            wrapper.default = packet.default
        sidecar.packet_metadata.verify()
        wrapper.default = object()
        try:
            with self.assertRaisesRegex(RuntimeError, 'METADATA_OWNER_CHANGED'):
                sidecar.packet_metadata.verify()
        finally:
            wrapper.default = packet.default
        for index in range(10):
            case.request('real-collector-cpu-'+str(index))
        case.ok(case.client.finish());case.ok(case.client.unregister())
        sidecar.close();case.assert_clean()
        summary, _ = review(folder/'statistics')
        self.assertTrue(summary['baseline_controls_pass'], json.dumps(summary, default=str)[:2000])
        self.assertTrue(summary['observable_difference_supported'], json.dumps(summary, default=str)[:2000])
        self.assertFalse(summary['e5'])
        self.assertTrue(all(row['completed'] and row['owned_hook_restore_exact'] for row in sidecar.rows))
        identity = json.loads((folder/'statistics/collector_identity.json').read_text())
        self.assertTrue(identity['custom_packet_metadata_compatibility']['dispatcher_default_is_original'])
        self.assertFalse(identity['custom_packet_metadata_compatibility']['callable_replaced'])


if __name__ == '__main__':
    unittest.main()
