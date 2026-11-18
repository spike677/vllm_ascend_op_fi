"""Actual CPU Torch API returns and Native-v2 mocks; never NPU acceptance."""
import ctypes as C
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import weakref

import torch

from fault_injection.deployments.writer26_experiment import Writer26Experiment
from fault_injection.deployments.writer26_group import Call
from demoncat.runtime.vllm_ascend.worker_extension.serving import complete_tensor
from demoncat.runtime.vllm_ascend.worker_extension.serving26_group import Serving26GroupExperiment
from test_writer26_inventory import CpuReader, Producer
from test_writer26_group_experiment import CpuV2
from test_collector_dispatch import fixture as collector_fixture


_LIB = torch.library.Library('npu', 'FRAGMENT')
_LIB.define('npu_fused_infer_attention_score(Tensor query, Tensor key, Tensor value) -> (Tensor, Tensor)')
_RUNTIME = None


def cpu_fia(query, key, value):
    result = _RUNTIME.writer(query)
    if _RUNTIME.two_native_calls:
        result = _RUNTIME.writer(query)
    if _RUNTIME.return_clone:
        result = result.clone()
    return result, torch.ones(1)


_LIB.impl('npu_fused_infer_attention_score', cpu_fia, 'CPU')
_PACKET = torch.ops.npu.npu_fused_infer_attention_score


class PublicFiaV2(CpuV2):
    mechanism, fia = '26.fia_destination', True

    def __init__(self, reader):
        super().__init__(reader)
        self.config.mechanism = 2
        self.output_ids, self.consumer_alive = [], []
        self.output_ref = None
        self.return_clone = self.two_native_calls = self.wrong_exit_token = False
        self.wrong_shape = False

    def writer(self, x):
        self.writer_calls += 1
        output = x.clone()
        self.output_ref = weakref.ref(output)
        self.output_ids.append(id(output))
        self.reader.raw.all_runs += 1
        self.reader.raw.non_target_runs += 1
        event = self.reader.append(3, output=output, residual=self.last, producer=self.sequence,
                                   dtype=27, format=47)
        if self.wrong_shape:
            event.shape[0] += 1
        call = Call(version=2, size=C.sizeof(Call), mechanism=2, dimensions=output.ndim,
            device=0, dtype=27, format=47, token=event.call, thread=threading.get_native_id(),
            independent_call=event.call, producer_sequence=self.sequence, output=output.data_ptr(),
            output_bytes=output.untyped_storage().nbytes(), residual=self.last.data_ptr(),
            residual_bytes=self.last.untyped_storage().nbytes(),
            shape=(C.c_int64 * 4)(*output.shape, *([0] * (4 - output.ndim))))
        if self.admit is not None:
            self.admit(C.pointer(call))
        if self.transaction is not None and self.transaction['mode'] == 'fault':
            self.last.view(-1)[-16:] += 1
        # Mirror the real trace: before has original formals, GWS/enter have
        # effective formals, and exit has only call/thread/executor/status.
        self.reader.append(4, output=output, call=event.call, executor=456, dtype=27, format=47)
        self.reader.append(5, output=output, residual=self.last, producer=self.sequence,
            call=event.call, executor=456, registration=1, dtype=27, format=47)
        self.reader.append(6, call=event.call + int(self.wrong_exit_token), executor=456)
        return output

    def producer(self, x, residual):
        if self.output_ref is not None:
            self.consumer_alive.append(self.output_ref() is not None)
            self.output_ref = None
        return super().producer(x, residual)


class CopiedAttention(torch.nn.Module):
    """Source-equivalent ownership shape: API output -> view -> distinct buffer."""
    def forward(self, x):
        import torch_npu
        query = x.view(13, 9, 64)
        returned, _ = torch_npu.npu_fused_infer_attention_score(query=query, key=query, value=query)
        output = torch.empty_like(query)
        output[:] = returned.view(13, 9, 64)[:]
        return output.view(13, 576)


class PublicFiaModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.producer = Producer()
        self.first, self.second = CopiedAttention(), CopiedAttention()
        self.consumer_view = self.consumer_clone = False

    def consumer_residual(self, residual):
        if self.consumer_clone:
            return residual.clone()
        return residual.view_as(residual) if self.consumer_view else residual

    def forward(self, x):
        residual = self.producer(x, torch.zeros_like(x))
        x = self.first(x)
        residual = self.producer(x, self.consumer_residual(residual))
        x = self.second(x)
        return self.producer(x, self.consumer_residual(residual))


class WriterPublicReturnTests(unittest.TestCase):
    def setUp(self):
        global _RUNTIME
        self.reader = CpuReader()
        self.native = _RUNTIME = PublicFiaV2(self.reader)
        self.model = PublicFiaModel().eval()
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.npu = NS(npu_fused_infer_attention_score=_PACKET, get_npu_format=lambda tensor: 47)
        self.enterContext(patch.dict(sys.modules, {'torch_npu': self.npu}))
        self.enterContext(patch.object(torch.ops._C_ascend, 'npu_add_rms_norm_bias',
                                      self.native.producer, create=True))
        self.experiment = Writer26Experiment(self.model, self.native, self.reader,
            worker_identity=dict(rank=0), folder=self.folder / 'writer', stage='effects', max_targets=8)
        self.addCleanup(self.cleanup)

    def cleanup(self):
        # Tests restore any deliberately foreign wrapper before this cleanup.
        if self.experiment.current is not None:
            self.experiment.fail('CPU_TEST_ABORT')
        self.experiment.close()

    def forward(self, index):
        with torch.inference_mode():
            self.experiment.begin('cpu-request-' + str(index), 'e' * 64)
            result = self.model(torch.ones(13, 576, dtype=torch.bfloat16))
            _, digest, finite = complete_tensor(result)
            self.experiment.end(digest, model_output_finite=finite)
        return result

    def run_all(self):
        self.experiment.install()
        for index in range(40):
            if self.experiment.phase == 'complete':
                break
            self.forward(index)
        self.assertEqual(self.experiment.phase, 'complete')
        self.experiment.close()

    def test_real_api_output_stays_live_through_consumer_after_attention_copies_it(self):
        self.run_all()
        self.assertEqual(len(self.experiment.targets), 2)
        self.assertEqual(len(self.native.executes), 10)
        rows = self.experiment.rows[0]['callbacks']
        for index, row in enumerate(rows):
            retained = row['writer_output_retained']
            self.assertEqual(retained['evidence'], 'ACTUAL_PUBLIC_FIA_RETURN')
            self.assertEqual(retained['tensor_object_identity'], self.native.output_ids[index])
            self.assertEqual(retained['shape'], [13, 9, 64])
            self.assertEqual(retained['format'], 47)
            self.assertTrue(row['public_consumer_lifetime'])
        self.assertTrue(all(self.native.consumer_alive[:-2]))  # Last health call has no observer.
        self.assertTrue(self.experiment.recovered)
        self.assertFalse(self.experiment.report()['c2'])
        self.assertIs(self.npu.npu_fused_infer_attention_score, _PACKET)
        self.assertFalse(self.experiment.public_output.owned or self.experiment.output_refs)

    def test_identical_clone_is_not_the_actual_native_return_formal(self):
        self.native.return_clone = True
        self.run_all()
        self.assertFalse(self.experiment.targets or self.native.executes)
        self.assertTrue(all('PUBLIC_WRITER_OUTPUT_NOT_RETAINED' in row['reasons']
                            for row in self.experiment.candidate_inventory))
        self.assertTrue(any('FULL_NATIVE_FORMAL_REQUIRED' in error
            for call in self.experiment.public_output.calls for error in call['errors']))

    def test_wrong_native_exit_token_cannot_bind_one_api_return(self):
        self.native.wrong_exit_token = True
        self.run_all()
        self.assertFalse(self.experiment.targets or self.native.executes)
        self.assertTrue(any('COMPLETED_NATIVE_CALL' in error
            for call in self.experiment.public_output.calls for error in call['errors']))

    def test_multiple_native_calls_inside_one_api_return_remain_unverified(self):
        self.native.two_native_calls = True
        self.run_all()
        self.assertEqual(len(self.experiment.candidate_inventory), 4)
        self.assertFalse(self.experiment.targets or self.native.executes)
        self.assertTrue(any('ONE_NATIVE_FIA_CALL' in error
            for call in self.experiment.public_output.calls for error in call['errors']))

    def test_wrong_original_native_shape_cannot_bind_same_storage(self):
        self.native.wrong_shape = True
        self.run_all()
        self.assertFalse(self.experiment.targets or self.native.executes)
        self.assertTrue(any('COMPLETED_NATIVE_CALL' in error
            for call in self.experiment.public_output.calls for error in call['errors']))

    def test_wrong_public_format_cannot_bind_same_shape_and_storage(self):
        self.npu.get_npu_format = lambda tensor: 2
        self.run_all()
        self.assertFalse(self.experiment.targets or self.native.executes)
        self.assertTrue(any('NATIVE_FORMAT_CHANGED' in error
            for call in self.experiment.public_output.calls for error in call['errors']))

    def test_actual_consumer_view_has_same_retained_storage_identity(self):
        self.model.consumer_view = True
        self.run_all()
        self.assertEqual(len(self.experiment.targets), 2)
        for row in self.experiment.rows[0]['callbacks']:
            binding = row['public_consumer_binding']
            self.assertNotEqual(binding['producer_tensor_object_identity'], binding['consumer_tensor_object_identity'])
            self.assertTrue(binding['same_residual_storage'] and binding['writer_retained_through_actual_consumer'])

    def test_identical_consumer_clone_is_not_the_retained_residual(self):
        self.model.consumer_clone = True
        self.run_all()
        self.assertFalse(self.experiment.targets or self.native.executes)
        self.assertTrue(all('PUBLIC_CONSUMER_LIFETIME_NOT_PROVEN' in row['reasons']
                            for row in self.experiment.candidate_inventory))

    def test_previous_public_binding_cannot_admit_a_later_unobserved_api_call(self):
        self.experiment.install()
        self.forward(0)
        self.forward(1)
        with torch.inference_mode():
            self.experiment.begin('cpu-changed-api', 'e' * 64)
            owned = self.npu.npu_fused_infer_attention_score
            self.npu.npu_fused_infer_attention_score = _PACKET
            try:
                output = self.model(torch.ones(13, 576, dtype=torch.bfloat16))
                _, digest, finite = complete_tensor(output)
                with self.assertRaisesRegex(RuntimeError, 'LIFETIME_OR_EFFECT_INCOMPLETE'):
                    self.experiment.end(digest, model_output_finite=finite)
                self.assertFalse(self.native.executes or self.experiment.pending)
                self.assertIn('PUBLIC_FIA_CALL_SCOPE_CHANGED', self.experiment.failure)
            finally:
                self.npu.npu_fused_infer_attention_score = owned

    def test_foreign_outer_wrapper_is_not_unwrapped_by_name_or_wrapped_attribute(self):
        def foreign(*args, **kwargs):
            return _PACKET(*args, **kwargs)
        foreign.__wrapped__ = _PACKET
        self.npu.npu_fused_infer_attention_score = foreign
        self.run_all()
        self.assertFalse(self.experiment.targets or self.native.executes)
        self.assertIs(self.npu.npu_fused_infer_attention_score, foreign)
        self.assertTrue(self.experiment.public_output.refusals)

    def test_reviewed_collector_calls_and_alias_survive_each_scoped_observer(self):
        with collector_fixture() as (module, _, _, _, _, _, _):
            calls = []
            def template(api_name, api_func, prefix, hook):
                def collect(*args, **kwargs):
                    calls.append(api_name)
                    return api_func(*args, **kwargs)
                return collect
            module.ApiRegistry.register_custom_api(self.npu, 'npu_fused_infer_attention_score',
                                                    'npu', None, template)
            original = self.npu.npu_fused_infer_attention_score
            self.run_all()
            self.assertEqual(len(self.experiment.targets), 2)
            self.assertEqual(len(calls), self.native.writer_calls)
            self.assertIs(self.npu.npu_fused_infer_attention_score, original)
            self.assertTrue(self.experiment.rows[0]['callbacks'][0]['writer_output_retained']['collector_wrappers'])

    def test_foreign_alias_at_cleanup_is_preserved_with_live_owner_until_retry(self):
        self.experiment.install()
        self.experiment.begin('cpu-request', 'e' * 64)
        owned = self.npu.npu_fused_infer_attention_score
        foreign = object()
        self.npu.npu_fused_infer_attention_score = foreign
        self.experiment.fail('CPU_ABORT_BEFORE_FORWARD')
        try:
            with self.assertRaisesRegex(RuntimeError, 'FIA_PUBLIC_ALIAS_OWNER_CHANGED'):
                self.experiment.close()
            self.assertIs(self.npu.npu_fused_infer_attention_score, foreign)
            self.assertTrue(self.experiment.installed)
            self.assertEqual(len(self.experiment.public_output.owned), 1)
        finally:
            self.npu.npu_fused_infer_attention_score = owned
        self.experiment.close()
        self.assertFalse(self.experiment.installed or self.experiment.public_output.owned)
        self.assertIs(self.npu.npu_fused_infer_attention_score, _PACKET)

    def restore_failure(self, after):
        class FailingNamespace(NS):
            def __setattr__(self, name, value):
                fail = (name == 'npu_fused_infer_attention_score' and value is _PACKET
                        and getattr(self, 'fail_restore', False))
                if fail and not after:
                    raise RuntimeError('RESTORE_BEFORE_SET')
                super().__setattr__(name, value)
                if fail:
                    raise RuntimeError('RESTORE_AFTER_SET')
        self.npu = FailingNamespace(npu_fused_infer_attention_score=_PACKET, get_npu_format=lambda tensor: 47)
        sys.modules['torch_npu'] = self.npu
        self.experiment.install()
        self.experiment.begin('cpu-cleanup-retry', 'e' * 64)
        self.experiment.fail('CPU_ABORT')
        self.npu.fail_restore = True
        with self.assertRaisesRegex(RuntimeError, 'RESTORE_'):
            self.experiment.close()
        self.assertTrue(self.experiment.installed)
        self.assertEqual(len(self.experiment.public_output.owned), 0 if after else 1)
        self.npu.fail_restore = False
        self.experiment.close()
        self.assertIs(self.npu.npu_fused_infer_attention_score, _PACKET)
        self.assertFalse(self.experiment.installed or self.experiment.public_output.owned)

    def test_restore_failure_before_mutation_retains_alias_owner_for_retry(self):
        self.restore_failure(False)

    def test_restore_failure_after_mutation_records_actual_state_before_retry(self):
        self.restore_failure(True)

    def test_hook_cleanup_failure_keeps_producer_owner_for_same_thread_retry(self):
        self.experiment.install()
        self.experiment.begin('cpu-hook-cleanup-retry', 'e' * 64)
        self.experiment.fail('CPU_ABORT')
        handle = self.experiment.handles[-1]
        with patch.object(handle, 'remove', side_effect=RuntimeError('HOOK_REMOVE_FAILED')):
            with self.assertRaisesRegex(RuntimeError, 'HOOK_REMOVE_FAILED'):
                self.experiment.close()
        self.assertTrue(self.experiment.installed and self.experiment.handles)
        self.assertIs(torch.ops._C_ascend.npu_add_rms_norm_bias, self.experiment.producer_observer)
        self.assertIs(self.npu.npu_fused_infer_attention_score, _PACKET)
        self.experiment.close()
        self.assertFalse(self.experiment.installed or self.experiment.handles)

    def test_failed_request_releases_own_alias_before_collector_stop(self):
        self.experiment.install()
        self.experiment.begin('cpu-collector-failure', 'e' * 64)
        owned = self.npu.npu_fused_infer_attention_score
        self.experiment.fail('CPU_ABORT')
        session = Serving26GroupExperiment.__new__(Serving26GroupExperiment)
        session.backend, session.needs_collect = self.experiment, True
        session._same_worker = lambda: None
        session.collector_active, session.capture_step_executed = True, False
        stopped = []
        def stop():
            self.assertIs(self.npu.npu_fused_infer_attention_score, _PACKET)
            self.assertFalse(self.experiment.public_output.owned)
            stopped.append(True)
        session.collector = NS(stop=stop)
        foreign = object()
        self.npu.npu_fused_infer_attention_score = foreign
        try:
            with self.assertRaisesRegex(RuntimeError, 'FIA_PUBLIC_ALIAS_OWNER_CHANGED'):
                session.collect()
            self.assertFalse(stopped)
            self.assertTrue(session.collector_active and self.experiment.installed)
            self.assertIs(self.npu.npu_fused_infer_attention_score, foreign)
        finally:
            self.npu.npu_fused_infer_attention_score = owned
        session.collect()
        self.assertEqual(stopped, [True])
        self.assertFalse(session.collector_active or self.experiment.public_output.owned)


if __name__ == '__main__':
    unittest.main()
