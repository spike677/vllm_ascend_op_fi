"""Collector-window contracts; do not pretend fake collectors are msprobe proof."""
from contextlib import contextmanager
import threading
import unittest
from unittest.mock import patch
import torch

import test_natural_tensor_group as fixtures
from test_rmsnorm_natural import Network, CustomFusedNorm
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
    def test_native_rms_return_lineage_metadata_is_paused_without_hiding_native_call(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.norm = CustomFusedNorm()
            def forward(self, x):
                normalized, residual = self.norm(x, torch.ones_like(x))
                return normalized * 3 + residual
        model = Model().eval(); capture = GuardWindow(); states = []
        original = torch.Tensor.stride
        def stride(tensor, *args):
            states.append(capture.active)
            self.assertFalse(capture.active, 'Native lineage diagnostics leaked into raw statistics')
            return original(tensor, *args)
        from torch.utils._python_dispatch import TorchDispatchMode
        native_states = []
        class NativeReader(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if str(func) == '_C_ascend.npu_add_rms_norm_bias.default':
                    native_states.append(capture.active)
                return func(*args, **(kwargs or {}))
        with patch.object(torch.Tensor, 'stride', stride), NativeReader(), \
             RMSNormReturnSession(model, simulation=True, diagnostics_context=capture.diagnostics):
            with capture(model, 'statistics_fault'):
                model(torch.arange(24.).reshape(3, 8))
        self.assertTrue(states)
        self.assertEqual(native_states, [True])

    def test_public_control_window_can_span_rpcs_without_reset_during_nested_pauses(self):
        from experiments.c2c3_closure_20261117.tensor_statistics_capture import TensorStatisticsCapture
        events = []
        class FakeCollector:
            def start(self, **kwargs):
                events.append('start')
            def stop(self):
                events.append('stop')
            def step(self):
                events.append('step')
        capture = object.__new__(TensorStatisticsCapture)
        capture.thread = threading.get_ident(); capture.active = False; capture.depth = 0
        capture.model = None; capture.rows = []; capture.collector = FakeCollector()
        capture.synchronize = lambda: None
        capture.save = lambda: None
        capture.begin(object(), 'statistics_baseline')
        with capture.diagnostics():
            with capture.diagnostics():
                self.assertEqual(events, ['start', 'stop'])
        self.assertEqual(events, ['start', 'stop', 'start'])
        capture.end(completed=True)
        self.assertEqual(events, ['start', 'stop', 'start', 'stop', 'step'])
        self.assertTrue(capture.rows[0]['completed'])
        self.assertEqual(capture.rows[0]['pauses'], 1)
        self.assertFalse(capture.active)
        self.assertEqual(capture.depth, 0)
        with self.assertRaisesRegex(RuntimeError, 'NOT_READY_TO_CLOSE'):
            capture.end(completed=True)

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
