"""Collector-window contracts; do not pretend fake collectors are msprobe proof."""
from contextlib import contextmanager
import unittest
from unittest.mock import patch
import torch

import test_natural_tensor_group as fixtures
from test_rmsnorm_natural import Network
from fault_injection.rmsnorm_return import RMSNormReturnSession
from demoncat.runtime.vllm_ascend.worker_extension.providers import tensor_natural


class GuardWindow:
    def __init__(self):
        self.active = False
        self.rows = []
        self.pauses = 0
    @contextmanager
    def __call__(self, model, phase):
        enabled = phase != 'statistics_off'
        self.rows.append(phase)
        self.active = enabled
        try:
            yield
        finally:
            self.active = False
    @contextmanager
    def diagnostics(self):
        before = self.active
        if before:
            self.pauses += 1
        self.active = False
        try:
            yield
        finally:
            self.active = before


class StatisticsWindowTests(unittest.TestCase):
    def test_snapshots_excluded_real_model_returns_included_and_group_unchanged(self):
        model = Network().eval(); inputs = (torch.arange(16.).reshape(2, 8),)
        provider = fixtures.NaturalTensorGroupTests().provider(model, '18.api_half', inputs)
        provider.qualify(); provider.run()
        group = provider.group; before = group.report()
        capture = GuardWindow()
        real_capture, real_cpu = tensor_natural._capture, RMSNormReturnSession.cpu
        snapshots = []
        def guarded_capture(value):
            self.assertFalse(capture.active)
            snapshots.append('root')
            return real_capture(value)
        def guarded_cpu(session, value):
            self.assertFalse(capture.active)
            snapshots.append('local')
            return real_cpu(session, value)
        real_returns = []
        handle = model.left.register_forward_hook(lambda *args: real_returns.append(capture.active))
        try:
            with patch.object(tensor_natural, '_capture', guarded_capture), \
                 patch.object(RMSNormReturnSession, 'cpu', guarded_cpu):
                result = provider.collect_statistics_pair(capture)
        finally:
            handle.remove()
        self.assertEqual(capture.rows, ['statistics_baseline', 'statistics_baseline_repeat',
                                        'statistics_fault', 'statistics_off'])
        self.assertEqual(real_returns, [True, True, True, False])
        self.assertGreater(capture.pauses, 3)
        self.assertIn('root', snapshots); self.assertIn('local', snapshots)
        self.assertTrue(result['baseline_exact'] and result['off_exact'])
        self.assertIs(provider.group, group)
        self.assertEqual(before, group.report())


if __name__ == '__main__':
    unittest.main()
