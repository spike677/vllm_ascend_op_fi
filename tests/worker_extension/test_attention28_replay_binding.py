"""Actual CPU call objects through the existing Attention wrapper; no Native.

The scheduler and API implementation are explicit CPU seams. Geometry matches
the observed source71 TND call, including its strided V view; these contracts
grant neither a stock/private normal result nor prepared-executor admission.
"""
from copy import copy, deepcopy
import os
import pickle
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from demoncat.runtime.vllm_ascend.worker_extension import attention28_replay_binding as B
from demoncat.runtime.vllm_ascend.worker_extension import attention_execution as E
from demoncat.runtime.vllm_ascend.worker_extension.serving_attention import (
    install_attention_serving_binding, remove_attention_serving_binding)


class Attention(torch.nn.Module):
    def __init__(self, api):
        super().__init__()
        self.api, self.layer_name = api, 'actual'
        base = torch.zeros(2048, 4096, dtype=torch.float16)
        self.register_buffer('q', base[:, :2048].view(2048, 16, 128))
        self.register_buffer('k', base[:, 2048:3072].view(2048, 8, 128))
        self.register_buffer('v', base[:, 3072:].view(2048, 8, 128))
        self.register_buffer('mask', torch.ones(2048, 2048, dtype=torch.int8).triu(1))

    def forward(self, value):
        return self.api.npu_fused_infer_attention_score(self.q, self.k, self.v,
            atten_mask=self.mask, block_table=None, input_layout='TND', block_size=128,
            num_heads=16, num_key_value_heads=8, scale=128**-.5, sparse_mode=3,
            actual_seq_lengths=[2048], actual_seq_lengths_kv=[2048])[0]


class Logits(torch.nn.Module):
    def forward(self, hidden):return hidden.mean((-1, -2))


class Model(torch.nn.Module):
    def __init__(self, api):
        super().__init__();self.attn = Attention(api);self.logits = Logits()

    def forward(self, input_ids, positions):return self.attn(input_ids)


class CpuObserver:
    mechanism = B.MECHANISM
    observer_only = True

    def state(self):
        return dict(native_loaded=True, scope_active=False, scope_owned=False, environment_owned=0)

    def observe(self, original, args, actual, row, metadata, once):
        return self.callback(row, once), dict(simulation=True, fault_authorized=False)

    def stop(self):return self.state()


class ReplayInputContracts(unittest.TestCase):
    def setUp(self):
        old_threads = torch.get_num_threads();torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        self.enterContext(patch.object(B, 'identity', side_effect=lambda: dict(pid=os.getpid(),
            native_tid=threading.get_native_id(), python_thread_ident=threading.get_ident(),
            process_start_ticks='EXPLICIT_CPU_TEST_PROCESS')))
        self.calls = 0
        self.during_original = lambda: None
        def original(*args, **kwargs):
            self.calls += 1;self.during_original()
            return args[0], torch.empty(0)
        defaults = dict(pre_tokens=2147483647, next_tokens=2147483647, inner_precise=0, softmax_lse_flag=False)
        original.default = NS(_schema=NS(arguments=[NS(name=k, default_value=v, has_default_value=lambda: True)
            for k, v in defaults.items()]))
        self.original = original
        self.api = NS(npu_fused_infer_attention_score=original)
        self.model = Model(self.api).eval()
        self.worker = NS(rank=0, device=NS(type='cpu', index=0), model_runner=NS(model=self.model, with_prefill=True))
        metadata = NS(num_prefills=1, num_decodes=0, num_actual_tokens=2048, causal=True,
            seq_lens_list=[2048], actual_seq_lengths_q=[2048], attn_mask=self.model.attn.mask)
        self.context = NS(model_instance=self.model, no_compile_layers={'actual': self.model.attn},
            attn_metadata={'actual': metadata})
        self.driver = CpuObserver()
        setup = dict(api_module=self.api, candidates=[('attn', self.model.attn)], processor=self.model.logits,
            context=lambda: self.context, snapshot=lambda request_id, actual, context: dict(
                status='READ_ONLY_SCHEDULER_SNAPSHOT_CORROBORATED_NOT_NPU_ACCEPTANCE',
                internal_request_id=request_id, num_requests=1, num_computed_tokens=0,
                num_prompt_tokens=2048, num_scheduled_tokens=2048, block_size=16))
        self.deployment = install_attention_serving_binding(self.worker, B.MECHANISM, prompt_token_ids=[1],
            provenance={'cpu_only': True}, native=self.driver, _simulation=True, _test_context=setup)
        self.deployment.leases.add(('cpu-owner', 1))
        self.session = E.AttentionExecutionAdapter(self.deployment, lambda: None)
        self.session.install()
        self.binding = B.ReplayInputBinding(self.session)
        self.addCleanup(self.clean)

    def clean(self):
        self.worker.model_runner.model = self.model
        self.deployment.leases.clear();self.deployment.leases.add(('cpu-owner', 1))
        if self.binding.active is not None:self.binding.close_public(self.binding.active)
        # Only this fixture disposes deliberately corrupted CPU ledger entries.
        # Production cleanup never clears an unknown live-call owner.
        self.session.live_api_calls.clear()
        self.session.close();self.deployment.leases.clear()
        remove_attention_serving_binding(self.worker, B.MECHANISM)
        self.assertIsNone(E._ACTIVE)

    def run_call(self, callback):
        self.driver.callback = callback
        self.session.expect('actual-request', 'observe')
        with torch.inference_mode():
            hidden = self.model(torch.ones(2048, dtype=torch.int32), torch.arange(2048))
            self.model.logits(hidden)
        self.assertEqual(self.calls, 1)
        self.assertEqual(self.session.live_api_calls, [])

    def test_actual_live_frame_and_strided_views_are_held_before_and_inside_original(self):
        def callback(row, once):
            plan = self.binding.begin(row)
            self.assertIs(plan.actual['value'], self.model.attn.v)
            self.assertEqual(plan.before['inputs']['value']['snapshot']['metadata']['stride'], [4096, 128, 1])
            self.assertEqual(plan.before['inputs']['value']['snapshot']['metadata']['storage_offset'], 3072)
            self.binding.check_current(plan)
            self.during_original = lambda: self.binding.check_current(plan)
            result = once()
            with self.assertRaisesRegex(RuntimeError, 'TWO_STAGE_ADMISSION_UNBOUND'):
                self.binding.authorize_fault(plan)
            self.binding.close_public(plan)
            self.assertIsNone(plan.actual)
            return result
        self.run_call(callback)
        snapshot = self.binding.snapshot()
        self.assertEqual(snapshot['closed_count'], 1)
        self.assertFalse(snapshot['native_abort_performed'] or snapshot['fault_authorized'])

    def test_saved_report_copy_pickle_and_foreign_plan_cannot_be_authority(self):
        def callback(row, once):
            plan = self.binding.begin(row)
            for clone in (copy, deepcopy, pickle.dumps):
                with self.assertRaises(TypeError):clone(plan)
            snapshot = self.binding.snapshot();snapshot['input']['inputs']['value']['snapshot']['sha256'] = 'changed'
            self.binding.check_current(plan)
            with self.assertRaisesRegex(RuntimeError, 'EXACT_LIVE_PLAN_REQUIRED'):
                self.binding.check_current(snapshot)
            with self.assertRaisesRegex(RuntimeError, 'PLAN_ALREADY_HELD'):self.binding.begin(row)
            self.binding.close_public(plan)
            with self.assertRaisesRegex(RuntimeError, 'CALL_ALREADY_PLANNED'):self.binding.begin(row)
            return once()
        self.run_call(callback)

    def test_same_value_clone_of_actual_argument_is_not_same_live_object(self):
        def callback(row, once):
            plan = self.binding.begin(row)
            frame = self.session.live_api_calls[-1]
            self.session.live_api_calls[-1] = (frame[0], frame[1], (frame[2][0].clone(), *frame[2][1:]), frame[3])
            with self.assertRaisesRegex(RuntimeError, 'CURRENT_CALL_IDENTITY_CHANGED'):self.binding.check_current(plan)
            self.session.live_api_calls[-1] = frame
            self.binding.close_public(plan)
            return once()
        self.run_call(callback)

    def test_inference_tensor_write_invalidates_plan_and_error_is_sticky(self):
        def callback(row, once):
            plan = self.binding.begin(row)
            # Actual tensor values, not a cached hash or optional version counter.
            self.model.attn.v[0, 0, 0] = 1
            with self.assertRaisesRegex(RuntimeError, 'CURRENT_INPUT_CHANGED'):self.binding.check_current(plan)
            self.model.attn.v[0, 0, 0] = 0
            with self.assertRaisesRegex(RuntimeError, 'EXACT_LIVE_PLAN_REQUIRED'):self.binding.check_current(plan)
            self.assertIn('CURRENT_INPUT_CHANGED', self.binding.first_error)
            self.binding.close_public(plan)
            return once()
        self.run_call(callback)

    def test_actual_scalar_mutation_is_rejected(self):
        def callback(row, once):
            plan = self.binding.begin(row)
            kwargs = self.session.live_api_calls[-1][3]
            original = kwargs['actual_seq_lengths'];kwargs['actual_seq_lengths'] = [2047]
            with self.assertRaisesRegex(RuntimeError, 'CURRENT_INPUT_CHANGED'):self.binding.check_current(plan)
            kwargs['actual_seq_lengths'] = original
            self.binding.close_public(plan)
            return once()
        self.run_call(callback)

    def test_same_value_mask_clone_is_rejected(self):
        def callback(row, once):
            plan = self.binding.begin(row);kwargs = self.session.live_api_calls[-1][3]
            original = kwargs['atten_mask'];kwargs['atten_mask'] = original.clone()
            with self.assertRaisesRegex(RuntimeError, 'ACTUAL_INPUT_OBJECT_CHANGED'):self.binding.check_current(plan)
            kwargs['atten_mask'] = original;self.binding.close_public(plan)
            return once()
        self.run_call(callback)

    def test_live_ledger_corruption_is_retained_and_cannot_clean_as_zero(self):
        def callback(row, once):
            result = once();self.session.live_api_calls.clear();return result
        self.driver.callback = callback;self.session.expect('actual-request', 'observe')
        with torch.inference_mode():
            hidden = self.model(torch.ones(2048, dtype=torch.int32), torch.arange(2048))
            self.model.logits(hidden)
        self.assertEqual(self.calls, 1)
        self.assertIn('LIVE_API_CALL_OWNER_CHANGED', self.session.last_error)
        self.assertEqual(self.session.state()['inflight'], 1)
        with self.assertRaisesRegex(RuntimeError, 'INFLIGHT_OR_FOREIGN_CLOSE'):self.session.close()

    def test_changed_request_phase_or_metadata_does_not_release_native_authority(self):
        def callback(row, once):
            plan = self.binding.begin(row)
            self.session.current['phase'] = 'fault'
            with self.assertRaisesRegex(RuntimeError, 'CURRENT_CALL_IDENTITY_CHANGED'):self.binding.check_current(plan)
            self.session.current['phase'] = 'observe'
            self.binding.close_public(plan)
            return once()
        self.run_call(callback)

    def test_foreign_thread_and_owner_cannot_release_public_references(self):
        def callback(row, once):
            plan = self.binding.begin(row);errors = []
            def other():
                try:self.binding.close_public(plan)
                except RuntimeError as error:errors.append(str(error))
            thread = threading.Thread(target=other);thread.start();thread.join()
            self.assertIn('OWNER_THREAD_CHANGED', errors[0]);self.assertIs(self.binding.active, plan)
            self.deployment.leases.add(('foreign-owner', 1))
            with self.assertRaisesRegex(RuntimeError, 'RESERVATION_CHANGED'):self.binding.close_public(plan)
            self.deployment.leases.remove(('foreign-owner', 1));self.binding.close_public(plan)
            return once()
        self.run_call(callback)

    def test_retained_plan_is_expired_after_original_window_but_local_refs_can_release(self):
        held = []
        def callback(row, once):held.append(self.binding.begin(row));return once()
        self.run_call(callback)
        with self.assertRaisesRegex(RuntimeError, 'CURRENT_NATURAL_CALL_REQUIRED'):self.binding.check_current(held[0])
        self.binding.close_public(held[0])
        self.assertFalse(self.binding.snapshot()['native_prepared_bound'])

    def test_partial_capture_failure_keeps_local_owner_without_native_claim(self):
        def callback(row, once):
            with patch.object(B, 'capture_inputs', side_effect=MemoryError('CPU snapshot failure')):
                with self.assertRaises(MemoryError):self.binding.begin(row)
            self.assertIsNotNone(self.binding.active)
            self.assertFalse(self.binding.snapshot()['native_abort_performed'])
            self.binding.close_public(self.binding.active)
            return once()
        self.run_call(callback)


if __name__ == '__main__':unittest.main()
