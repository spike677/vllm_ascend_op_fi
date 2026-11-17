"""CPU worker adapter orchestration contracts; no simulated Worker qualification."""
from types import SimpleNamespace as NS
import sys
import threading
import unittest
from unittest.mock import patch

import torch

from demoncat.runtime.vllm_ascend.worker_extension import postwrite41 as worker_adapter
from fault_injection.deployments.postwrite_kv import live_value_tail_plan
from tests.test_postwrite_natural_kv import fixture, CpuSpanTransport


def bound_provider():
    f = fixture()
    module = NS(impl=f['impl'], layer_name='actual-layer', kv_cache=[(f['impl'].key_cache, f['impl'].value_cache)])
    context = NS(virtual_engine=0, attn_metadata={'actual-layer':f['metadata']},
                 no_compile_layers={'actual-layer':module})
    p = worker_adapter.Postwrite41WorkerProvider.__new__(worker_adapter.Postwrite41WorkerProvider)
    p.stack = [('actual.attention', module)]
    p.rows, p.errors, p.independent_calls = [], [], []
    p.writers = {id(module):f['writer']}
    p.current, p.forward_serial, p.tail_elements = f['scheduler'], 1, 2
    p.target = worker_adapter._target_key('actual.attention', module, context)
    p.phase, p.owner, p.generation = 'fault', 34, 1
    p.lease, p.transport, p.pending_run = None, CpuSpanTransport(), dict(new_fault_hits=0)
    fake_module = NS(get_forward_context=lambda:context)
    return p, f, module, fake_module


class WorkerAdapterOrchestrationTests(unittest.TestCase):
    def consume(self, p, f, module, original):
        with patch.dict(sys.modules, {'vllm.forward_context':module}):
            # Only the device acceptance is replaced by a prevalidated CPU
            # fixture here. Neither this test nor its receipts is NPU evidence.
            with patch.object(worker_adapter, 'live_value_tail_plan', return_value=live_value_tail_plan(**f)):
                return p._consume('npu_fused_infer_attention_score', original, (), f['actual'])

    def test_inventory_is_recorded_before_native_admission_and_actual_call(self):
        p, f, _, module = bound_provider()
        before = f['impl'].value_cache.clone()
        calls = []
        def original(**kwargs):
            self.assertEqual(len(p.independent_calls), 1)
            self.assertTrue(p.lease.live)
            calls.append(kwargs)
            return f['impl'].value_cache[1, 1].clone()
        result = self.consume(p, f, module, original)
        self.assertEqual(len(calls), 1)
        self.assertEqual(p.pending_run['new_fault_hits'], 1)
        self.assertIsNone(p.lease)
        self.assertTrue(p.rows[0]['intervention']['released'])
        self.assertTrue(torch.equal(f['impl'].value_cache, before))
        self.assertFalse(torch.equal(result, before[1, 1]))

    def test_off_has_denominator_and_original_but_no_native_lease(self):
        p, f, _, module = bound_provider(); p.phase = 'off'
        calls = []
        def original(**kwargs):
            self.assertIsNone(p.lease)
            calls.append(kwargs)
            return 'real return'
        self.assertEqual(self.consume(p, f, module, original), 'real return')
        self.assertEqual(len(p.independent_calls), 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(p.pending_run['new_fault_hits'], 0)

    def test_unselected_target_is_observed_and_passes_through(self):
        p, f, _, module = bound_provider(); p.target = ('different target',)
        calls = []
        self.consume(p, f, module, lambda **kw:calls.append(kw))
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(p.independent_calls), 1)
        self.assertNotIn('intervention', p.rows[0])

    def test_bad_external_writer_prevents_release_and_close(self):
        p, f, _, module = bound_provider()
        def overwritten(**kw):f['impl'].value_cache[1, 1, 0, 0] += 1
        with self.assertRaisesRegex(RuntimeError, 'EXTERNAL_WRITE'):
            self.consume(p, f, module, overwritten)
        self.assertIsNotNone(p.lease)
        self.assertTrue(p.lease.live)
        with self.assertRaisesRegex(RuntimeError, 'EXTERNAL_WRITE'):p.recover()

    def test_phase_requires_stable_model_lossless_and_once(self):
        p = worker_adapter.Postwrite41WorkerProvider.__new__(worker_adapter.Postwrite41WorkerProvider)
        p.model = torch.nn.Linear(2, 2)
        p.model_identity = p._identity()
        p.installed, p.qualified, p.lossless, p.fault_attempted = True, True, False, False
        p.expected = p.current = p.lease = None
        with self.assertRaisesRegex(RuntimeError, 'LOSSLESS_ONCE'):p.expect('fault', 'req', [1])
        p.lossless, p.generation = True, 1
        p.expect('fault', 'req', [1])
        p.expected = None
        with self.assertRaisesRegex(RuntimeError, 'LOSSLESS_ONCE'):p.expect('fault', 'req2', [1])
        with torch.no_grad():p.model.weight.add_(1)
        with self.assertRaisesRegex(RuntimeError, 'MODEL_BINDING_CHANGED'):p.expect('off', 'req2', [1])


if __name__ == '__main__':unittest.main()
