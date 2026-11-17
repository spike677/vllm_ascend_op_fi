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
    def test_only_observer_metadata_is_paused_real_model_stride_remains_captured(self):
        from fault_injection.tensor_observer import NaturalTensorObserver, metadata
        class Model(torch.nn.Module):
            def forward(self, x):
                x.stride()  # A genuine model API must remain inside capture.
                return x + 1
        model = Model().eval(); capture = GuardWindow(); stride_states = []
        original = torch.Tensor.stride
        def record(tensor, *args):
            stride_states.append(capture.active)
            return original(tensor, *args)
        with patch.object(torch.Tensor, 'stride', record):
            with NaturalTensorObserver(model, diagnostics_context=capture.diagnostics):
                with capture(model, 'statistics_baseline'):
                    model(torch.ones(2))
            # The observer's context-local guard must be released on close.
            with capture(model, 'statistics_baseline'):
                metadata(torch.ones(2))
        self.assertIn(False, stride_states)
        self.assertEqual(sum(stride_states), 2)

    def test_snapshots_excluded_real_model_returns_included_and_group_unchanged(self):
        model = Network().eval(); inputs = (torch.arange(16.).reshape(2, 8),)
        provider = fixtures.NaturalTensorGroupTests().provider(model, '18.api_half', inputs)
        provider.qualify(); provider.run()
        group = provider.group; before = group.report()
        capture = GuardWindow()
        real_capture, real_cpu = tensor_natural._capture, RMSNormReturnSession.cpu
        real_stride = torch.Tensor.stride
        snapshots = []
        def guarded_capture(value):
            self.assertFalse(capture.active)
            snapshots.append('root')
            return real_capture(value)
        def guarded_cpu(session, value):
            self.assertFalse(capture.active)
            snapshots.append('local')
            return real_cpu(session, value)
        def guarded_stride(tensor, *args):
            self.assertFalse(capture.active, 'Observer metadata leaked into the raw model capture')
            snapshots.append('metadata_stride')
            return real_stride(tensor, *args)
        real_returns = []
        handle = model.left.register_forward_hook(lambda *args: real_returns.append(capture.active))
        try:
            with patch.object(tensor_natural, '_capture', guarded_capture), \
                 patch.object(RMSNormReturnSession, 'cpu', guarded_cpu), \
                 patch.object(torch.Tensor, 'stride', guarded_stride):
                result = provider.collect_statistics_pair(capture)
        finally:
            handle.remove()
        self.assertEqual(capture.rows, ['statistics_baseline', 'statistics_baseline_repeat',
                                        'statistics_fault', 'statistics_off'])
        self.assertEqual(real_returns, [True, True, True, False])
        self.assertGreater(capture.pauses, 3)
        self.assertIn('root', snapshots); self.assertIn('local', snapshots)
        self.assertIn('metadata_stride', snapshots)
        self.assertTrue(result['baseline_exact'] and result['off_exact'])
        self.assertIs(provider.group, group)
        self.assertEqual(before, group.report())


if __name__ == '__main__':
    unittest.main()
