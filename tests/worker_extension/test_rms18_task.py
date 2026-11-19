"""Real CPU ops and public Host/Runtime; patched Worker metadata is no NPU proof."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

from demoncat.runtime.vllm_ascend.worker_extension import identity, serving_tensor_binding as bindings
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
from test_extension import _extended_worker
from test_core_host import ExistingExecutor
import test_serving_session_lifecycle as life
from test_rmsnorm_natural import CustomFusedNorm


class RmsModel(life.CpuModel):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.RMSNorm(4, dtype=torch.bfloat16)
        self.later = torch.nn.RMSNorm(4, dtype=torch.bfloat16)
        self.forward_calls = 0
        self.skip_consumer = False
        self.first_output = self.later_output = None

    def forward(self, input_ids, positions, inputs_embeds=None, intermediate_tensors=None):
        self.forward_calls += 1
        hidden = super().forward(input_ids, positions, inputs_embeds, intermediate_tensors)
        self.first_output = self.first(hidden)
        consumed = hidden if self.skip_consumer else self.first_output + 1
        self.later_output = self.later(consumed)
        return self.later_output


class LastRmsModel(RmsModel):
    def forward(self, input_ids, positions, inputs_embeds=None, intermediate_tensors=None):
        self.forward_calls += 1
        hidden = life.CpuModel.forward(self, input_ids, positions, inputs_embeds, intermediate_tensors)
        return self.first(hidden)


class FusedFour(CustomFusedNorm):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))


class FusedModel(RmsModel):
    def __init__(self):
        super().__init__()
        self.first = FusedFour()

    def forward(self, input_ids, positions, inputs_embeds=None, intermediate_tensors=None):
        self.forward_calls += 1
        hidden = life.CpuModel.forward(self, input_ids, positions, inputs_embeds, intermediate_tensors)
        self.first_output, self.residual = self.first(hidden, torch.ones_like(hidden))
        return self.first_output + self.residual


class Rms18TaskTests(unittest.TestCase):
    snapshot = life.ServingSessionLifecycleTests.snapshot
    root = life.ServingSessionLifecycleTests.root
    forward = life.ServingSessionLifecycleTests.forward

    def setUp(self):
        life.ServingSessionLifecycleTests.setUp(self)
        self.worker = _extended_worker(rank=0)
        self.worker.load_model()
        self.runner = self.worker.model_runner
        self.model = RmsModel().eval()
        self.runner.model = self.model
        self.runner.vllm_config = NS(model_config=NS(hf_config=NS(vocab_size=3)))
        self.enterContext(patch.object(identity, '_current_device', return_value=0))
        self.enterContext(patch.object(bindings, 'require_serial_eager'))
        self.executor = ExistingExecutor([self.worker])
        def bind(actual, config):
            self.assertEqual(actual, self.worker.demoncat_worker_identity()['worker'])
            bindings.install_rms18_task_binding(self.worker, prompt_token_ids=[1, 2, 3, 4],
                weight_manifest={'cpu_fixture': True}, workload_sha256='a'*64, environment_sha256='b'*64,
                _simulation=True, _synchronize=lambda: None)
            return {}
        self.client = WorkerFaultClient(HostAdapter(self.executor, prepare_binding=bind), FaultConfig(18, 'api_half'))
        self.provider = None
        self.addCleanup(self.dispose)

    def dispose(self):
        if self.provider and self.provider.session:
            self.provider.session.request_error = self.provider.session.request_error or 'CPU_FIXTURE_TEARDOWN_ONLY'
            self.provider.session.close()
        self.assertEqual(_get_current_dispatch_mode_stack(), [])

    def ok(self, reply):
        self.assertEqual(reply['status'], 'ACTION_COMPLETED', repr(reply.get('response'))[:2000])
        return reply['response']['provider_state']

    def start(self):
        self.ok(self.client.prepare())
        self.provider = self.worker._demoncat_worker_fault_runtime._leases['18.api_half'].provider
        for action in ('register', 'check', 'enable'): self.ok(getattr(self.client, action)())
        self.assertEqual(self.model.forward_calls, 0)

    def enqueue(self, name):
        self.actual_request_id = name
        calls = self.model.forward_calls
        self.ok(self.client.advance(name))
        self.assertEqual(self.model.forward_calls, calls)
        with torch.inference_mode(): return self.forward(self.provider.session)

    def finish(self):
        state = self.ok(self.client.finish())
        self.ok(self.client.unregister())
        self.assertEqual(self.client.detach()['status'], 'DETACHED')
        return state

    def test_plan_on_prepare_query_short_and_exhausted(self):
        from fault_injection.request_plan import request_plan
        created=self.ok(self.client.prepare())
        self.assertEqual(created['request_plan'],request_plan('18.api_half'))
        self.provider=self.worker._demoncat_worker_fault_runtime._leases['18.api_half'].provider
        for action in ('register','check','enable'):self.ok(getattr(self.client,action)())
        self.enqueue('short-normal')
        short=self.client.query()['response']['provider_state']
        self.assertFalse(short['complete']);self.assertEqual(short['next_phase'],'fault')
        self.assertEqual(short['request_plan'],created['request_plan'])
        for phase in ('fault','off','health'):self.enqueue(phase)
        done=self.client.query()['response']['provider_state']
        self.assertTrue(done['complete']);before=self.model.forward_calls
        extra=self.client.advance('extra-request')
        self.assertNotEqual(extra['status'],'ACTION_COMPLETED')
        self.assertIn('PHASES_EXHAUSTED',str(extra.get('response')))
        self.assertEqual(before,self.model.forward_calls)

    def test_short_plan_abort_cleans_without_fault_or_extra_forward(self):
        self.start(); self.enqueue('normal-only')
        state = self.client.query()['response']['provider_state']
        self.assertFalse(state['complete'])
        calls = self.model.forward_calls
        self.ok(self.client.abort())
        self.ok(self.client.unregister())
        self.assertEqual(self.client.detach()['status'], 'DETACHED')
        self.assertEqual(self.model.forward_calls, calls)
        self.assertFalse(self.provider.fault_started)
        state = self.provider.session.readback()
        self.assertEqual((state['owned_hooks'], state['leases'], state['inflight']), (0, 0, 0))

    def test_four_actual_requests_one_return_no_group_census_or_files(self):
        from demoncat.runtime.vllm_ascend.worker_extension.providers.tensor_natural import NaturalTensorGroupProvider
        original = {}
        hooks = [self.model.first.register_forward_hook(lambda m, a, v: original.update(first=v)),
                 self.model.later.register_forward_hook(lambda m, a, v: original.update(later=v))]
        self.addCleanup(lambda: [hook.remove() for hook in hooks])
        with patch.object(NaturalTensorGroupProvider, '__init__', side_effect=AssertionError('no group')):
            self.start()
            normal = self.enqueue('normal')
            self.assertIs(self.model.first_output, original['first'])
            self.assertIs(self.model.later_output, original['later'])
            self.assertEqual(self.provider.target['path'], 'first')
            fault = self.enqueue('fault')
            self.assertIsNot(self.model.first_output, original['first'])
            self.assertIs(self.model.later_output, original['later'])
            off = self.enqueue('off'); health = self.enqueue('health')
            state = self.finish()
        rows = state['serving']['actual_requests']
        self.assertEqual([row['phase'] for row in rows], ['normal', 'fault', 'off', 'health'])
        self.assertEqual([len((r['mechanism'] or {}).get('events', [])) for r in rows], [1, 1, 0, 0])
        self.assertTrue(torch.equal(normal, off) and torch.equal(normal, health))
        self.assertFalse(torch.equal(normal, fault))
        self.assertEqual(self.model.forward_calls, 4)
        self.assertTrue(state['serving']['recovered'])
        self.assertEqual(state['last_fault_effect']['matched_effects'], 1)
        self.assertFalse(any(state[k] for k in ('C1', 'C2', 'C3', 'E5', 'qualification_granted')))
        self.assertEqual((state['owned_hooks'], state['leases'], state['inflight']), (0, 0, 0))
        self.assertFalse(self.evidence_root.exists())

    def test_pending_off_effect_is_unverified_and_query_executes_nothing(self):
        self.start(); self.enqueue('normal'); self.enqueue('fault')
        before = self.model.forward_calls
        state = self.ok(self.client.advance('off'))
        self.assertEqual(state['current_request_phase'], 'off')
        self.assertEqual(state['current_request_effect']['status'], 'EFFECT_NOT_VERIFIED')
        self.assertIsNone(state['current_request_effect']['scope'])
        self.assertEqual(state['last_fault_effect']['matched_effects'], 1)
        self.assertEqual(self.client.query()['status'], 'QUERY_OK')
        self.assertEqual(self.model.forward_calls, before)
        self.actual_request_id = 'off'
        with torch.inference_mode(): self.forward(self.provider.session)
        self.enqueue('health'); self.finish()
        self.assertTrue(self.provider.attempted)

    def test_first_unconsumed_return_refuses_without_scanning_later_layers(self):
        self.model.skip_consumer = True
        self.start()
        with self.assertRaisesRegex(RuntimeError, 'NO_ACTUAL_CONSUMED_RETURN'): self.enqueue('normal')
        self.assertIsNone(self.provider.target)
        self.assertFalse(self.provider.fault_started)
        self.assertEqual(self.model.forward_calls, 1)
        self.assertEqual(len(self.provider.session.current['mechanism']['events']), 1)

    def test_actual_scheduler_id_mismatch_refuses_before_model(self):
        self.start(); self.ok(self.client.advance('expected'))
        self.actual_request_id = 'different'
        with torch.inference_mode(), self.assertRaisesRegex(RuntimeError, 'SCHEDULER_ID_MISMATCH'):
            self.root(self.provider.session)
        self.assertEqual(self.model.forward_calls, 0)
        self.assertFalse(self.provider.fault_started)

    def test_changed_target_weight_refuses_before_fault_write(self):
        self.start(); self.enqueue('normal')
        with torch.no_grad(): self.model.first.weight.add_(1)
        with self.assertRaisesRegex(RuntimeError, 'CURRENT_RETURN_INPUT_WEIGHT_OR_SOURCE_CHANGED'):
            self.enqueue('fault')
        self.assertTrue(self.provider.fault_started)
        self.assertFalse(self.provider.fault_applied)

    def test_failed_fault_requires_actual_off_health_and_never_rearms(self):
        self.start(); self.enqueue('normal')
        self.model.processor.raise_forward = True
        with self.assertRaisesRegex(RuntimeError, 'PROCESSOR_FORWARD_FAILED'): self.enqueue('fault')
        self.assertTrue(self.provider.fault_applied)
        self.model.processor.raise_forward = False
        self.assertNotEqual(self.client.finish()['status'], 'ACTION_COMPLETED')
        self.ok(self.client.recover())
        self.enqueue('recover-off'); self.enqueue('recover-health')
        state = self.finish()
        self.assertTrue(state['serving']['recovered'])
        self.assertTrue(state['serving']['first_error'])
        self.assertTrue(self.provider.attempted and self.provider.fault_started)
        self.assertEqual(self.model.forward_calls, 4)

    def test_no_silent_reenable_or_second_fault(self):
        self.start()
        for phase in ('normal', 'fault', 'off', 'health'): self.enqueue(phase)
        self.ok(self.client.finish())
        reply = self.client.enable()
        self.assertNotEqual(reply['status'], 'ACTION_COMPLETED')
        self.assertTrue(self.provider.attempted)

    def test_outside_host_read_is_not_a_consumer_real_logits_is(self):
        self.model = LastRmsModel().eval(); self.runner.model = self.model
        self.start(); self.actual_request_id = 'normal'
        self.ok(self.client.advance('normal'))
        with torch.inference_mode():
            hidden = self.root(self.provider.session)
            window = self.provider.session.window
            torch.mm(hidden, torch.ones(4, 4, dtype=torch.bfloat16))
            self.assertEqual(window.events[0]['consumers'], [])
            self.provider.session.processor(self.model.head, hidden[-1:].contiguous())
        row = self.provider.rows[0]
        self.assertEqual(len(row['mechanism']['events'][0]['consumers']), 1)
        self.assertTrue(row['mechanism']['events'][0]['consumers'][0]['post_root'])
        for phase in ('fault', 'off', 'health'): self.enqueue(phase)
        self.finish()

    def test_changed_module_object_refuses_before_actual_fault_forward(self):
        self.start(); self.enqueue('normal')
        self.model.first = torch.nn.RMSNorm(4, dtype=torch.bfloat16).eval()
        before = self.model.forward_calls
        with self.assertRaisesRegex(RuntimeError, 'HELD_TARGET_MODULE_CHANGED'): self.enqueue('fault')
        self.assertEqual(self.model.forward_calls, before)
        self.assertFalse(self.provider.fault_started)

    def test_changed_positions_refuse_before_actual_forward(self):
        self.start(); self.ok(self.client.advance('normal')); self.actual_request_id = 'normal'
        with torch.inference_mode(), self.assertRaisesRegex(RuntimeError, 'ACTUAL_DEVICE_INPUT_BYTES'):
            self.model(input_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int32), positions=torch.tensor([0, 1, 2, 4]))
        self.assertEqual(self.model.forward_calls, 0)

    def test_foreign_mode_blocks_cleanup_and_is_not_popped(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        class Foreign(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None): return func(*args, **(kwargs or {}))
        self.start(); self.actual_request_id = 'normal'; self.ok(self.client.advance('normal'))
        with torch.inference_mode(): self.root(self.provider.session)
        self.provider.session.request_error = 'CPU_ABORT_BEFORE_LOGITS'
        with Foreign() as mode:
            with self.assertRaisesRegex(RuntimeError, 'Foreign dispatcher'):
                self.provider.session.close_window()
            self.assertIs(_get_current_dispatch_mode_stack()[-1], mode)
        self.provider.session.close_window()

    def test_later_root_hook_exception_cannot_be_recorded_as_returned(self):
        self.start(); self.enqueue('normal')
        def failure(module, args, output): raise RuntimeError('CPU_FOREIGN_ROOT_EXCEPTION')
        hook = self.model.register_forward_hook(failure)
        try:
            with self.assertRaisesRegex(RuntimeError, 'CPU_FOREIGN_ROOT_EXCEPTION'): self.enqueue('fault')
        finally: hook.remove()
        self.assertEqual(len(self.provider.rows), 1)
        self.assertIsNotNone(self.provider.session.request_error)
        self.ok(self.client.recover()); self.enqueue('off'); self.enqueue('health'); self.finish()

    def test_actual_cpu_native_schema_one_call_auxiliary_identity_preserved(self):
        self.model = FusedModel().eval(); self.runner.model = self.model
        self.start(); self.enqueue('normal')
        self.assertIs(self.model.residual, self.model.first.actual_auxiliary)
        self.enqueue('fault')
        event = self.provider.rows[-1]['mechanism']['events'][0]
        self.assertEqual(event['native_calls'], 1)
        self.assertEqual(event['native_operator'], '_C_ascend.npu_add_rms_norm_bias.default')
        self.assertIs(self.model.residual, self.model.first.actual_auxiliary)
        self.assertTrue(event['auxiliary_return_storage_unchanged'])
        self.enqueue('off'); self.enqueue('health'); self.finish()

    def test_zero_effect_missing_target_does_not_pass(self):
        self.start(); self.enqueue('normal')
        self.model.skip_consumer = True
        with self.assertRaisesRegex(RuntimeError, 'SELECTED_RETURN_EFFECT_AND_CONSUMER_REQUIRED'):
            self.enqueue('fault')
        self.assertEqual(len(self.provider.rows), 1)
        self.model.skip_consumer = False
        self.ok(self.client.recover()); self.enqueue('off'); self.enqueue('health'); self.finish()


if __name__ == '__main__': unittest.main()
