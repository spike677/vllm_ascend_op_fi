"""Real Host/Runtime/owned hooks with existing RMS/MoE CPU implementations."""
from contextlib import ExitStack
from copy import copy, deepcopy
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

import test_serving_tensor_group as rms
import test_serving_moe_group as moe
from test_extension import _extended_worker
from test_core_host import ExistingExecutor
from demoncat.runtime.vllm_ascend.worker_extension import identity, serving_tensor, serving_tensor_binding as binding
from demoncat.runtime.vllm_ascend.worker_extension.providers import tensor_serving
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient


class Counted:
    def __init__(self):
        super().__init__()
        self.forward_calls = self.logits_calls = 0
        self.fail_forward = self.fail_logits = False
    def forward(self, input_ids, positions):
        self.forward_calls += 1
        if self.fail_forward:
            raise RuntimeError('CPU actual root failed')
        return super().forward(input_ids, positions)
    def logits(self, hidden):
        self.logits_calls += 1
        if self.fail_logits:
            raise RuntimeError('CPU actual logits failed')
        return super().logits(hidden)


class RMSModel(Counted, rms.WorkerModel):
    pass


class MoEModel(Counted, moe.Model):
    pass


class CoreTensorServingTests(unittest.TestCase):
    mechanism = '18.api_half'
    model_type = RMSModel

    def setUp(self):
        self.patches = ExitStack()
        fake_logits = ModuleType('vllm.model_executor.layers.logits_processor')
        fake_logits.LogitsProcessor = rms.Processor
        fake_context = ModuleType('vllm.forward_context')
        fake_context.get_forward_context = lambda: SimpleNamespace(cpu_contract=True)
        self.patches.enter_context(patch.dict(sys.modules, {fake_logits.__name__: fake_logits,
                                                          fake_context.__name__: fake_context}))
        for module in (serving_tensor, binding):
            self.patches.enter_context(patch.object(module, 'require_serial_eager', lambda runner: None))
        self.patches.enter_context(patch.object(serving_tensor, 'discover_logits_boundary',
            lambda model, **kwargs: dict(processor=model.processor, report=dict(cpu_contract=True))))
        self.scheduler_request = None
        def snapshot(runner, model, actual, context, **kwargs):
            self.assertTrue(context.cpu_contract)
            self.assertEqual(actual['input_ids'].tolist(), kwargs['expected_prompt_token_ids'])
            if self.scheduler_request != kwargs['expected_internal_request_id']:
                raise RuntimeError('CPU_ACTUAL_REQUEST_ID_MISMATCH')
            self.assertNotIn(self.scheduler_request, kwargs['seen_internal_request_ids'])
            return dict(cpu_contract=True, request_id=self.scheduler_request)
        for module in (serving_tensor, tensor_serving):
            self.patches.enter_context(patch.object(module, 'snapshot_fresh_prefill', snapshot))
        self.worker = _extended_worker(rank=7)
        self.worker.load_model()
        self.model = self.model_type().eval()
        self.worker.model_runner.model = self.model
        self.worker.model_runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(vocab_size=4)))
        self.patches.enter_context(patch.object(identity, '_current_device', return_value=0))
        self.patches.enter_context(patch.object(torch.distributed, 'is_initialized', return_value=True))
        self.patches.enter_context(patch.object(torch.distributed, 'get_rank', return_value=7))
        self.executor = ExistingExecutor([self.worker])
        fault, variant = self.mechanism.split('.', 1)
        self.config = FaultConfig(int(fault), variant)
        self.client = WorkerFaultClient(HostAdapter(self.executor, prepare_binding=self.bind), self.config)
        self.provider = None

    def bind(self, actual, config):
        self.assertEqual(actual, self.worker.demoncat_worker_identity()['worker'])
        self.assertEqual(config.mechanism, self.mechanism)
        self.bound = binding.install_tensor_serving_binding(self.worker, self.mechanism,
            prompt_token_ids=[1, 2, 3], weight_manifest={'cpu_fixture': True}, workload_sha256='a'*64,
            environment_sha256='b'*64, first_live_validation=True, _simulation=True,
            _synchronize=lambda: None, _trace_factory=moe.template if self.mechanism.startswith('8.') else None)
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in self.model.modules()))
        return {}

    def tearDown(self):
        try:
            runtime = getattr(self.worker, '_demoncat_worker_fault_runtime', None)
            lease = runtime._leases.get(self.mechanism) if runtime else None
            if lease:
                p = lease.provider
                p._owned_abort()
                group = getattr(getattr(p.session, 'provider', None), 'group', None)
                if group is not None and not group.recovery_report()['complete']:
                    # Failure teardown only: no successful Recovery claim.
                    group.disable(timeout_seconds=0)
        finally:
            self.patches.close()
            self.assertEqual(_get_current_dispatch_mode_stack(), [])
            self.assertIsNone(sys.getprofile())

    def ok(self, value):
        self.assertEqual(value['status'], 'ACTION_COMPLETED', repr(value.get('response'))[:2000])
        return value['response']['provider_state']

    def start(self):
        self.ok(self.client.prepare())
        self.provider = self.worker._demoncat_worker_fault_runtime._leases[self.mechanism].provider
        for action in ('register', 'check', 'enable'):
            self.ok(getattr(self.client, action)())

    def actual(self, request_id, *, logits=True):
        self.scheduler_request = request_id
        with torch.inference_mode():
            hidden = self.model(input_ids=torch.tensor([1, 2, 3]), positions=torch.arange(3))
            return self.model.logits(hidden) if logits else hidden

    def request(self, request_id, *, collect=True):
        self.ok(self.client.advance(request_id))
        result = self.actual(request_id)
        if collect:
            self.ok(self.client.collect_replay())
        return result

    def first_four(self):
        self.start()
        for index in range(4):
            self.request('actual-' + str(index))

    def recovery(self):
        self.ok(self.client.recover())
        while self.provider.recovering:
            self.request('recovery-' + self.provider.recovering)
        self.ok(self.client.finish())
        self.ok(self.client.unregister())

    def assert_clean(self):
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))
        self.assertFalse(getattr(self.worker, binding._ATTRIBUTE)[self.mechanism].leases)
        self.assertFalse(self.worker._demoncat_worker_fault_runtime._leases)

    def test_public_protocol_actual_groups_original_proof_and_zero_resource_health(self):
        self.start()
        with patch.object(self.worker, 'demoncat_worker_identity', side_effect=AssertionError('recursive RPC')):
            for index in range(10):
                self.request('actual-' + str(index))
                queried = self.client.query()
                self.assertEqual(queried['status'], 'QUERY_OK')
                state = queried['response']['provider_state']
                effect = state['current_request_effect']
                if self.mechanism == '18.api_half' and index in (4, 6, 7):
                    self.assertEqual(effect['status'], 'MATCHED_SAME_RETURN_VALUES_OBSERVED')
                    self.assertEqual(effect['scope']['request_id'], 'actual-' + str(index))
                    self.assertEqual(effect['matched_effects'], 2)
                    self.assertTrue(all(item['actual_consumer_observed'] for item in effect['positions']))
                    self.assertTrue(all(item['values']['changed_elements'] > 0 for item in effect['positions']))
                else:
                    self.assertEqual(effect['status'], 'EFFECT_NOT_VERIFIED')
                self.assertFalse(effect['recovery_proven'])
                self.assertFalse(effect['qualification_granted'])
            state = self.ok(self.client.finish())
        c2, c3 = state['coverage']['C2'], state['coverage']['C3']
        self.assertEqual((c2['hits'], c3['hits']), (2, 4))
        self.assertEqual((c2['observed_eligible_calls'], c3['observed_eligible_calls']), (2, 4))
        issued, = c2['live_c2_proof']['issued']
        consumed = c3['live_c2_proof']['consumed']
        self.assertEqual(issued['object_binding'], consumed['object_binding'])
        self.assertEqual(consumed['object_binding']['python_object_id'], id(self.provider.proof))
        self.assertEqual(consumed['proof']['owner'], state['owner'])
        self.assertEqual(consumed['proof']['scope'], consumed['consumer']['scope'])
        self.assertEqual(consumed['proof']['target_ids'], consumed['consumer']['target_ids'])
        self.assertFalse(c2['real_worker_evidence'])
        self.assertFalse(state['public_qualification_granted'])
        self.assertTrue(state['coverage']['same_worker_healthy']['cleanup_before_request'])
        self.assertEqual((self.model.forward_calls, self.model.logits_calls), (10, 10))
        self.ok(self.client.clean())
        self.assertTrue(self.provider.binding_reserved)
        self.assertEqual(self.provider.binding.leases, {self.provider._ticket})
        with self.assertRaisesRegex(RuntimeError, 'OWNER_CLEAN_REQUIRED'):
            binding.remove_tensor_serving_binding(self.worker, self.mechanism)
        self.ok(self.client.unregister())
        self.assertEqual(self.client.detach()['status'], 'DETACHED')
        self.assert_clean()
        binding.remove_tensor_serving_binding(self.worker, self.mechanism)
        self.actual('unmanaged-after-cleanup')
        self.assertEqual(self.model.forward_calls, 11)

    def test_query_is_detached_read_only_and_collection_is_explicit(self):
        self.start()
        self.request('real-first', collect=False)
        before = (self.provider.index, self.provider.pending, len(self.provider.session.phases))
        for _ in range(2):
            row = self.client.query()
            self.assertEqual(row['status'], 'QUERY_OK')
            row['response']['provider_state']['serving']['last_request']['calls'].clear()
            self.assertEqual((self.provider.index, self.provider.pending, len(self.provider.session.phases)), before)
        self.assertTrue(self.provider.session.provider.rows[-1]['calls'])
        self.ok(self.client.collect_replay())
        self.assertEqual(self.provider.index, 1)
        self.assertIsNone(self.provider.pending)
        self.assertEqual(self.model.forward_calls, 1)
        self.ok(self.client.unregister())
        self.assert_clean()

    def test_default_missing_binding_refuses_before_runtime_owner(self):
        client = WorkerFaultClient(HostAdapter(self.executor), self.config)
        row = client.prepare()
        self.assertEqual(row['status'], 'PROVIDER_UNAVAILABLE', row)
        self.assertEqual(row['response']['provider_code'], 'REVIEWED_TENSOR_SERVING_BINDING_REQUIRED')
        self.assertFalse(self.worker._demoncat_worker_fault_runtime._leases)
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in self.model.modules()))
        self.assertEqual(client.detach()['status'], 'DETACHED')

    def test_prepare_settings_failure_releases_zero_resource_lease(self):
        def bad_bind(actual, config):
            self.bind(actual, config)
            return {'prepare': {'not_reviewed': True}}
        self.client = WorkerFaultClient(HostAdapter(self.executor, prepare_binding=bad_bind), self.config)
        row = self.client.prepare()
        self.assertEqual(row['status'], 'PROVIDER_ACTION_FAILED', row)
        self.assertTrue(row['ownership_retained'])
        self.ok(self.client.unregister())
        self.assert_clean()

    def test_scheduled_id_does_not_run_model_and_incomplete_collection_keeps_owner(self):
        self.start()
        self.ok(self.client.advance('queued-not-run'))
        self.assertEqual(self.model.forward_calls, 0)
        self.assertEqual(_get_current_dispatch_mode_stack(), [])
        self.assertIsNone(sys.getprofile())
        row = self.client.collect_replay()
        self.assertEqual(row['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(row['ownership_retained'])
        self.assertIsNone(self.provider.session.provider.group)
        self.ok(self.client.unregister())
        self.assert_clean()

    def test_foreign_or_missing_binding_reservation_cannot_be_cleaned(self):
        self.start()
        for replacement in (set(), {('foreign', 77)}):
            self.provider.binding.leases.clear()
            self.provider.binding.leases.update(replacement)
            row = self.client.clean()
            self.assertEqual(row['status'], 'PROVIDER_ACTION_FAILED')
            self.assertTrue(row['ownership_retained'])
            self.assertEqual(self.provider.binding.leases, replacement)
            self.provider.binding.leases.clear()
            self.provider.binding.leases.add(self.provider._ticket)
        self.ok(self.client.unregister())
        self.assert_clean()

    def test_foreign_hook_and_transient_removal_keep_owner_for_retry(self):
        self.start()
        handle = self.provider.session.handles[0]
        hooks, own = handle.hooks_dict_ref(), self.provider.session._callbacks[handle.id]
        foreign = lambda *args: None
        hooks[handle.id] = foreign
        row = self.client.unregister()
        self.assertEqual(row['status'], 'PROVIDER_ACTION_FAILED')
        self.assertIs(hooks[handle.id], foreign)
        hooks[handle.id] = own
        with patch.object(handle, 'remove', side_effect=RuntimeError('CPU transient removal')):
            self.assertEqual(self.client.unregister()['status'], 'PROVIDER_ACTION_FAILED')
        self.ok(self.client.unregister())
        self.assert_clean()

    def test_actual_fault_failure_cannot_release_until_real_off_and_health(self):
        self.first_four()
        self.ok(self.client.advance('failed-fault'))
        self.model.fail_logits = True
        with self.assertRaisesRegex(RuntimeError, 'actual logits failed'):
            self.actual('failed-fault')
        self.model.fail_logits = False
        row = self.client.unregister()
        self.assertEqual(row['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(row['ownership_retained'])
        self.assertFalse(self.provider.session.provider.group.recovery_report()['complete'])
        self.recovery()
        self.assert_clean()
        self.assertFalse(self.provider.complete)
        self.assertTrue(self.provider.recovered)
        self.assertNotIn('C3', self.provider.reports)

    def test_actual_busy_root_refuses_cleanup_without_popping_owned_hooks(self):
        self.start()
        self.ok(self.client.advance('busy-root'))
        found = []
        def busy(module, args):
            found.append(self.client.unregister())
        handle = self.model.first.register_forward_pre_hook(busy)
        try:
            self.actual('busy-root')
        finally:
            handle.remove()
        self.assertEqual(found[0]['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(found[0]['ownership_retained'])
        self.ok(self.client.unregister())
        self.assert_clean()

    def test_wrong_actual_request_refuses_before_any_group_or_effect(self):
        self.start()
        self.ok(self.client.advance('expected'))
        with self.assertRaisesRegex(RuntimeError, 'CPU_ACTUAL_REQUEST_ID_MISMATCH'):
            self.actual('different')
        self.assertIsNone(self.provider.session.provider.group)
        self.ok(self.client.unregister())
        self.assert_clean()

    def test_copied_live_proof_cannot_start_c3_or_first_fault(self):
        self.first_four()
        self.request('c2-fault')
        self.request('c2-off', collect=False)
        original = self.provider.session.provider.group.c2_proof
        with patch.object(self.provider.session.provider.group, 'c2_proof', side_effect=lambda: copy(original())):
            row = self.client.collect_replay()
        self.assertEqual(row['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(row['ownership_retained'])
        self.assertEqual(self.model.forward_calls, 6)
        self.assertEqual(self.provider.session.provider.group.group.trigger, 'once')
        self.ok(self.client.recover())
        self.request('recovery-health')
        self.ok(self.client.finish())
        self.ok(self.client.unregister())
        self.assert_clean()

    def test_lost_collection_reply_retains_owner_without_replaying_model(self):
        self.start()
        self.request('real-first', collect=False)
        def lose(method, args, replies):
            if method == 'demoncat_fault_rpc' and args[0] == 'advance' and args[1].get('settings') == {'collect': True}:
                self.executor.after = None
                raise TimeoutError('CPU reply lost after collection')
            return replies
        self.executor.after = lose
        row = self.client.collect_replay()
        self.assertTrue(row['ownership_retained'])
        self.assertEqual(self.provider.index, 1)
        self.assertEqual(self.model.forward_calls, 1)
        self.ok(self.client.recover())
        self.ok(self.client.unregister())
        self.assert_clean()

    def test_close_failure_after_root_hook_removal_is_retryable(self):
        self.start()
        self.ok(self.client.advance('partial-close'))
        hidden = self.actual('partial-close', logits=False)
        observer = self.provider.session.observer
        with patch.object(observer, 'close', side_effect=RuntimeError('CPU observer close failed')):
            with self.assertRaisesRegex(RuntimeError, 'observer close failed'):
                self.model.logits(hidden)
        self.assertIsNone(self.provider.session.root_post_handle)
        self.assertIs(self.provider.session.observer, observer)
        self.assertTrue(self.provider.binding_reserved)
        self.ok(self.client.unregister())
        self.assert_clean()

    def test_current_device_drift_keeps_original_owner(self):
        self.start()
        with patch.object(identity, '_current_device', return_value=1):
            row = self.client.unregister()
            self.assertTrue(row['ownership_retained'])
            self.assertTrue(self.provider.binding_reserved)
            self.assertTrue(self.provider.session.handles)
        self.ok(self.client.unregister())
        self.assert_clean()

    def test_foreign_root_prehook_cannot_rewrite_scheduled_input(self):
        self.start()
        self.ok(self.client.advance('foreign-pre'))
        called = []
        handle = self.model.register_forward_pre_hook(lambda *args: called.append(True))
        try:
            with self.assertRaisesRegex(RuntimeError, 'TENSOR_ROOT_PRE_HOOK_ORDER_UNREVIEWED'):
                self.actual('foreign-pre')
            self.assertEqual(self.model.forward_calls, 0)
            self.assertEqual(called, [])
            self.assertIsNone(self.provider.session.adapter)
        finally:
            handle.remove()
        self.ok(self.client.unregister())
        self.assert_clean()


class CoreMoEServingTests(CoreTensorServingTests):
    mechanism = '8.restore_offset'
    model_type = MoEModel


if __name__ == '__main__':
    unittest.main()
