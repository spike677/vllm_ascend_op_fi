import unittest
from unittest.mock import patch

import torch

from fault_injection.deployments import postwrite
from experiments.engineering_20261117.model_capture import statistics_model_call
from tests.test_postwrite_providers import CpuTransport, Projection


class Collector:
    def __init__(self, *, fail_start=False):
        self.active = False
        self.events = []
        self.fail_start = fail_start
    def start(self, **kwargs):
        self.active = True
        self.events.append('start')
        if self.fail_start:raise RuntimeError('START_FAILURE')
    def stop(self):
        self.events.append('stop')
        self.active = False
    def synchronize(self):self.events.append('synchronize')


class CaptureWindowTests(unittest.TestCase):
    def test_bf_capture_excludes_checks_and_recovery_but_includes_real_forward(self):
        for phase in ('baseline', 'fault'):
            with self.subTest(phase=phase):
                model, collector = Projection(), Collector()
                provider = postwrite.PostwriteProvider(model, CpuTransport(), '39.output_proxy')
                inputs = torch.ones(1, 4, dtype=torch.bfloat16)
                provider.qualify(inputs)
                provider.run('baseline', inputs)
                forward_seen = []
                h = model.register_forward_pre_hook(lambda *unused: forward_seen.append(collector.active))
                equal = postwrite._equal
                recover = provider._recover_owned_entries
                recovery_calls = []
                def checked_equal(*args):
                    self.assertFalse(collector.active)
                    return equal(*args)
                def checked_recover():
                    self.assertFalse(collector.active)
                    recovery_calls.append(True)
                    return recover()
                try:
                    with patch.object(postwrite, '_equal', checked_equal), patch.object(provider, '_recover_owned_entries', checked_recover):
                        provider.run(phase, inputs, model_observation=lambda actual:
                            statistics_model_call(collector, actual, collector.synchronize))
                finally:h.remove()
                self.assertEqual(forward_seen, [True])
                self.assertEqual(recovery_calls, [True])
                self.assertEqual(collector.events, ['start', 'synchronize', 'stop'])
                self.assertFalse(collector.active)

    def test_window_stops_after_forward_start_or_sync_exception(self):
        for failure in ('model', 'start', 'sync'):
            with self.subTest(failure=failure):
                c = Collector(fail_start=failure == 'start')
                def synchronize():
                    c.synchronize()
                    if failure == 'sync':raise RuntimeError('SYNC_FAILURE')
                with self.assertRaises(RuntimeError):
                    with statistics_model_call(c, object(), synchronize):
                        if failure == 'model':raise RuntimeError('MODEL_FAILURE')
                self.assertEqual(c.events, ['start', 'synchronize', 'stop'])
                self.assertFalse(c.active)


if __name__ == '__main__':unittest.main()
