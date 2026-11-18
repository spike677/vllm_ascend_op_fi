"""Public consumer unwind with explicit failed Native receipts; no NPU proof."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import weakref

import torch

from fault_injection.deployments.writer26_experiment import Writer26Experiment, held_storage
from fault_injection.deployments.writer26_group import packet
from demoncat.runtime.vllm_ascend.worker_extension.serving import complete_tensor
import test_writer26_group_experiment as fixture
from test_writer26_inventory import CpuReader, CpuWriterModel


class FailedReceiptNative(fixture.CpuV2):
    def __init__(self, reader):
        super().__init__(reader)
        self.inject_failure = self.drain_fails = self.stopped = self.drained = False
        self.last_error = 0
        self.restore_calls, self.callback_results = [], []

    def state(self):
        value = super().state()
        value['core'].update(error=self.last_error, stopped=int(self.stopped))
        return value

    def execute(self, call, mode, target, allocation):
        value = super().execute(call, mode, target, allocation)
        self.residual_weak = weakref.ref(self.last)
        return value

    def producer(self, x, residual):
        finished = self.finished
        def reject(pointer):
            # Match the old shim's error before completion validation, while
            # still returning the actual public AddRms result normally.
            pointer.contents.error = -2
            self.receipts[pointer.contents.token] = packet(pointer.contents)
            status = finished(pointer)
            self.callback_results.append(status)
            self.last_error = -10 if status else 0
            self.receipts[pointer.contents.token]['error'] = self.last_error
            return 0  # The real trace does not throw when its callback refuses.
        if self.inject_failure and self.transaction is not None:
            self.finished = reject
        try:
            return super().producer(x, residual)
        finally:
            self.finished = finished

    def receipt(self, token):
        return dict(self.receipts[token])

    def stop(self):
        self.stopped = True

    def drain(self):
        assert self.stopped
        if self.drain_fails:
            raise RuntimeError('EXPLICIT_CPU_DRAIN_FAILED')
        self.drained = True

    def restore(self, token):
        assert self.stopped and self.drained and self.transaction['call'].token == token
        residual = self.residual_weak()
        assert residual is not None  # Native fixture owns no public Tensor reference.
        with torch.inference_mode():
            residual.copy_(self.transaction['residual_before'])
        self.restore_calls.append(token)


class PendingConsumerTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.reader = CpuReader()
        self.native = FailedReceiptNative(self.reader)
        self.model = CpuWriterModel(self.native).eval()
        self.experiment = Writer26Experiment(self.model, self.native, self.reader, worker_identity=dict(rank=0),
            folder=self.folder / 'experiment', stage='effects', max_targets=8)
        def allocation(call, row):
            row['allocation'] = dict(residual_block=[call.residual, call.output],
                output_block=[call.output, call.output + call.output_bytes], shift=32)
        self.enterContext(patch.object(torch.ops._C_ascend, 'npu_add_rms_norm_bias', self.native.producer, create=True))
        self.enterContext(patch.object(self.experiment, '_allocation', allocation))
        self.enterContext(patch.dict(sys.modules, {'torch_npu': NS(get_npu_format=lambda tensor: 2)}))
        self.experiment.install()
        self.addCleanup(self.cleanup)
        for index in range(2):
            self.request(index)

    def request(self, index):
        with torch.inference_mode():
            self.experiment.begin('request-' + str(index), 'e' * 64)
            output = self.model(torch.ones(4, 16, dtype=torch.bfloat16))
            _, digest, finite = complete_tensor(output)
            self.experiment.end(digest, model_output_finite=finite)

    def failure(self):
        self.native.inject_failure = True
        with self.assertRaisesRegex(RuntimeError, 'ACTUAL_CONSUMER_DID_NOT_COMPLETE_NATIVE_LEASE'):
            self.request(2)
        self.assertEqual(self.native.callback_results, [1])
        token, outcome = next(iter(self.experiment.pending.items()))
        self.assertNotIn(outcome['native_call']['residual'], self.experiment.retained)
        self.assertIsNotNone(self.native.residual_weak())
        self.experiment.fail('COMPLETE_FRESH_PREFILL_HIDDEN_STATES_REQUIRED')
        return token, outcome

    def cleanup(self):
        if self.experiment.installed:
            self.native.drain_fails = False
            if self.experiment.active:
                self.experiment.fail('EXPLICIT_CPU_FIXTURE_CLEANUP')
            self.experiment.close()

    def test_failed_consumer_keeps_original_storage_after_inventory_advances(self):
        token, outcome = self.failure()
        owned = self.experiment.pending_lifetimes[token]
        call = outcome['native_call']
        self.assertTrue(held_storage(owned['residual'], call['residual'], call['residual_bytes']))
        self.assertTrue(held_storage(owned['consumer'], call['residual'], call['residual_bytes']))
        self.assertTrue(held_storage(owned['writer'], call['output'], call['output_bytes']))
        self.experiment.close()
        self.assertEqual(self.native.restore_calls, [token])
        self.assertFalse(self.experiment.pending_lifetimes or self.experiment.output_refs or self.experiment.retained)
        self.assertFalse(self.experiment.installed or self.native.state()['core']['pending'])
        self.assertFalse(self.experiment.report()['c1'] or self.experiment.report()['c2'])

    def test_first_failure_and_both_native_receipts_survive_root_unwind(self):
        token, outcome = self.failure()
        report = json.loads((self.experiment.folder / 'writer_inventory.json').read_text())
        self.assertEqual(report['first_failure'], 'EXACT_NATIVE_WRITER_CONSUMER_RECEIPT_REQUIRED')
        self.assertEqual(outcome['observed_consumer_receipt']['error'], -2)
        self.assertEqual(report['last_native_receipt']['error'], -10)
        self.assertEqual(report['pending_outcomes'][0]['native_call']['token'], token)
        self.assertTrue(any(row['reason'] == 'COMPLETE_FRESH_PREFILL_HIDDEN_STATES_REQUIRED'
                            for row in report['failure_events']))
        failed = json.loads((self.experiment.folder / 'failed_request.json').read_text())
        self.assertEqual(failed['failure'], report['first_failure'])
        self.assertEqual(failed['pending_outcomes'][0]['observed_consumer_receipt']['error'], -2)
        detached = self.experiment.report()
        detached['pending_outcomes'][0]['observed_consumer_receipt']['error'] = 0
        self.assertEqual(outcome['observed_consumer_receipt']['error'], -2)

    def test_failed_drain_keeps_public_owners_and_retry_restores_once(self):
        token, _ = self.failure()
        owned = self.experiment.pending_lifetimes[token]
        self.native.drain_fails = True
        with self.assertRaisesRegex(RuntimeError, 'EXPLICIT_CPU_DRAIN_FAILED'):
            self.experiment.close()
        self.assertIs(self.experiment.pending_lifetimes[token], owned)
        self.assertTrue(self.experiment.installed and self.experiment.handles)
        self.assertEqual(self.native.restore_calls, [])
        self.native.drain_fails = False
        self.experiment.close()
        self.assertEqual(self.native.restore_calls, [token])

    def test_rebound_public_tensor_refuses_off_even_with_old_storage_held(self):
        token, outcome = self.failure()
        owned = self.experiment.pending_lifetimes[token]
        tensor, storage = owned['residual']['tensor'], owned['residual']['storage']
        with torch.inference_mode():
            tensor.set_(torch.empty_like(tensor))
        try:
            with self.assertRaisesRegex(RuntimeError, 'PENDING_OFF_POINTER_LIVENESS_UNVERIFIED_DO_NOT_WRITE'):
                self.experiment.close()
            self.assertEqual(self.native.restore_calls, [])
            self.assertTrue(self.experiment.installed and self.experiment.handles)
        finally:
            # Explicit fixture repair; production never reconstructs a tensor
            # from a bare address to authorize recovery.
            with torch.inference_mode():
                tensor.set_(storage, 0, owned['residual']['shape'], owned['residual']['stride'])
        self.assertEqual(outcome['native_call']['residual'], storage.data_ptr())


if __name__ == '__main__':
    unittest.main()
