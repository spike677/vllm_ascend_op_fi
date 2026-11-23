"""Real public Host/Runtime and CPU model hooks; never Native qualification."""
from copy import deepcopy
from dataclasses import replace
import sys
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from demoncat.runtime.vllm_ascend.worker_extension import identity, serving41, postwrite41, batch_context
from demoncat.runtime.vllm_ascend.worker_extension import postwrite_collector
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
from fault_injection.coverage_session import CoverageScope
from fault_injection.deployments.postwrite_kv import live_value_tail_plan
from test_extension import _extended_worker
from test_core_host import ExistingExecutor
from kv41_fixture import CpuSpanTransport, CpuWorkerModel, DeviceOperator, provider_factory


class CoreKV41Tests(unittest.TestCase):
    def setUp(self):
        from research_native_fixture import install
        install(self)  # explicit research factory; public ordinary admission stays closed
        self.assertIsNone(postwrite41._ACTIVE)
        self.worker = _extended_worker(rank=0)
        self.worker.load_model()
        self.runner = self.worker.model_runner
        self.runner.with_prefill = False
        self.api = NS()
        self.model = CpuWorkerModel(self.api, self.runner).eval()
        self.runner.model = self.model
        self.transport = CpuSpanTransport()
        self.prompt = [1, 2, 3, 4, 5]
        self.context = NS(virtual_engine=0,
            attn_metadata={module.layer_name: module.fixture['metadata'] for module in self.model.attentions},
            no_compile_layers={module.layer_name: module for module in self.model.attentions})
        self.current_request = None
        self.enterContext(patch.dict(sys.modules, {'vllm.forward_context': NS(get_forward_context=lambda: self.context)}))
        self.enterContext(patch.object(identity, '_current_device', return_value=0))
        self.enterContext(patch.object(torch.distributed, 'is_initialized', return_value=True))
        self.enterContext(patch.object(torch.distributed, 'get_rank', return_value=0))
        self.enterContext(patch.object(serving41, 'require_serial_eager', lambda *a, **kw: None))
        self.enterContext(patch.object(postwrite41, 'live_value_tail_plan',
            side_effect=lambda **kw: live_value_tail_plan(**kw, allow_cpu_test=True)))
        def snapshot(runner, model, actual, context, expected, **kwargs):
            self.assertIs(runner, self.runner)
            self.assertIs(model, self.model)
            self.assertIs(context, self.context)
            self.assertEqual(list(expected), [self.current_request])
            self.assertEqual(expected[self.current_request]['prompt'], self.prompt)
            self.assertEqual(actual['input_ids'].tolist(), [7])
            self.assertEqual(actual['positions'].tolist(), [5])
            return dict(self.model.attentions[0].fixture['scheduler'], request_ids=[self.current_request],
                        prefixes=[dict(prompt=self.prompt, output=[7])], evidence_kind='cpu_contract')
        self.enterContext(patch.object(batch_context, 'snapshot_first_decode_rows', snapshot))
        self.executor = ExistingExecutor([self.worker])
        self.binding_calls = []
        def bind(actual_identity, config):
            self.binding_calls.append((deepcopy(actual_identity), config.as_dict()))
            self.assertEqual(actual_identity, self.worker.demoncat_worker_identity()['worker'])
            serving41.install_fault41_serving_binding(self.worker, transport=self.transport,
                prompt_token_ids=self.prompt, scope=CoverageScope('41.tail_column', 'a' * 64, 'b' * 64,
                    'c' * 64, 'cpu_contract', 'e' * 64, 'f' * 64), first_live_validation=True,
                _simulation=True, _provider_factory=provider_factory, _native_readback=lambda transport: transport.readback())
            return {}
        self.client = WorkerFaultClient(HostAdapter(self.executor, prepare_binding=bind), FaultConfig(41, 'tail_column'))
        self.addCleanup(self.dispose)

    def dispose(self):
        # Test disposal never constitutes an asserted recovery or qualification.
        runtime = getattr(self.worker, '_demoncat_worker_fault_runtime', None)
        lease = runtime._leases.get('41.tail_column') if runtime else None
        if lease is not None:
            provider = lease.provider
            p, group = provider.p, provider.group
            if group is not None:
                group.health_current = None
                group.close_health_observer()
            if p is not None:
                self.transport.fail_restore = False
                p.current = None
                p.close()
        self.assertIsNone(postwrite41._ACTIVE)

    def ok(self, result):
        self.assertEqual(result['status'], 'ACTION_COMPLETED', repr(result)[:1200])
        return result['response']['provider_state']

    def start(self):
        for action in ('prepare', 'register', 'check', 'enable'):
            self.ok(getattr(self.client, action)())
        self.provider = self.worker._demoncat_worker_fault_runtime._leases['41.tail_column'].provider

    def actual_request(self, request_id):
        self.current_request = request_id
        with torch.inference_mode():
            self.runner.with_prefill = True
            self.model(input_ids=torch.tensor(self.prompt), positions=torch.arange(5))
            self.runner.with_prefill = False
            return self.model(input_ids=torch.tensor([7]), positions=torch.tensor([5]))

    def enqueue(self, index):
        request_id = 'actual-enqueued-' + str(index)
        before = self.model.forward_calls
        self.ok(self.client.advance(request_id))
        self.assertEqual(self.model.forward_calls, before)
        return self.actual_request(request_id)

    def through(self, count):
        for index in range(count):
            self.enqueue(index)

    def test_public_protocol_uses_original_group_proof_and_complete_actual_inventory(self):
        self.start()
        with patch.object(self.worker, 'demoncat_worker_identity', side_effect=AssertionError('recursive identity RPC')):
            self.through(17)
            state = self.ok(self.client.finish())
        self.assertEqual((state['coverage']['C2']['planned_targets'], state['coverage']['C2']['hits']), (2, 2))
        self.assertEqual((state['coverage']['C3']['planned_targets'], state['coverage']['C3']['hits']), (2, 4))
        self.assertEqual(state['coverage']['C2']['observed_eligible_calls'], 4)
        self.assertEqual(state['coverage']['C3']['observed_eligible_calls'], 4)
        issued = state['coverage']['C2']['live_c2_proof']['issued'][0]
        consumed = state['coverage']['C3']['live_c2_proof']['consumed']
        self.assertEqual(issued['object_binding'], consumed['object_binding'])
        self.assertEqual(issued['object_binding']['python_object_id'], id(self.provider.proof))
        self.assertEqual(issued['proof_sha256'], consumed['proof_sha256'])
        self.assertEqual(consumed['proof']['owner'], state['owner'])
        self.assertLess(consumed['proof']['generation'], consumed['consumer']['generation'])
        self.assertEqual(consumed['proof']['scope'], consumed['consumer']['scope'])
        self.assertEqual(consumed['proof']['target_ids'], consumed['consumer']['target_ids'])
        self.assertEqual(consumed['proof']['scope']['evidence_kind'], 'cpu_contract')
        self.assertFalse(state['coverage']['C2']['real_worker_evidence'])
        self.assertFalse(state['public_qualification_granted'])
        self.assertEqual(len(state['serving']['actual_requests']), 17)
        self.assertEqual((self.model.forward_calls, self.model.decode_calls), (34, 17))
        for row in self.provider.p.history:
            inventory = row['independent_inventory']
            self.assertEqual(len(inventory['calls']), 2)
            self.assertEqual(len(inventory['writers']), 2)
            self.assertTrue(all(call['original_calls'] == 1 for call in inventory['calls']))
        self.assertEqual((state['owned_hooks'], state['owned_api_state']['owned_links_present'], state['leases'],
                          state['coverage_held_leases']), (0, 0, 0, 0))
        self.ok(self.client.disable())
        self.ok(self.client.unregister())
        self.assertEqual(self.client.detach()['status'], 'DETACHED')
        healthy = self.actual_request('ordinary-after-unregister')
        self.assertTrue(torch.equal(healthy, self.provider.p.outputs[0]))
        self.assertEqual((self.model.forward_calls, self.model.decode_calls), (36, 18))
        self.assertFalse(getattr(self.worker, serving41._ATTRIBUTE).leases)
        self.assertEqual(set(self.binding_calls[0][1]), {'fault_id', 'variant', 'implementation_version'})

    def test_advance_without_forward_and_query_never_manufacture_request_or_effect(self):
        self.start()
        self.ok(self.client.advance('enqueued-not-run'))
        before = (self.model.forward_calls, len(self.transport.opened), self.transport.restore_attempts)
        result = self.client.query()
        self.assertEqual(result['status'], 'QUERY_OK', result)
        self.assertEqual(before, (self.model.forward_calls, len(self.transport.opened), self.transport.restore_attempts))
        self.assertEqual(result['response']['provider_state']['serving']['actual_requests'], [])
        failed = self.client.finish()
        self.assertEqual(failed['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(failed['ownership_retained'])
        self.ok(self.client.unregister())

    def test_foreign_owner_generation_and_other_worker_cannot_drive_this_binding(self):
        self.start()
        ticket = self.client.snapshot()['ticket']
        for changed, reason in ((dict(owner='foreign'), 'OWNER_MISMATCH'),
                                (dict(generation=ticket['generation'] + 1), 'STALE_GENERATION')):
            response = self.worker.demoncat_fault_rpc('advance', dict(ticket, **changed,
                settings={'request_id': 'foreign-request'}))
            self.assertEqual(response['reason'], reason)
        other = _extended_worker(rank=4)
        other.load_model()
        self.executor.workers.insert(0, other)
        self.enqueue(0)
        self.assertFalse(other._demoncat_worker_fault_runtime._leases)
        self.ok(self.client.unregister())

    def test_forged_serializable_or_copied_c2_does_not_enable_c3(self):
        self.start()
        self.through(12)
        self.ok(self.client.advance('c3-lossless-queued'))  # Completes actual C2 health and keeps the issued object.
        self.provider.group.c2_live_proof = replace(self.provider.proof)
        self.actual_request('c3-lossless-queued')
        # The original group checks proof authority when actual lossless
        # completion opens C3, before any C3 fault request may be scheduled.
        refused = self.client.advance('must-not-schedule-c3-fault')
        self.assertEqual(refused['status'], 'PROVIDER_ACTION_FAILED', refused)
        self.assertTrue(refused['ownership_retained'])
        self.assertIsNotNone(self.provider.error)
        self.assertNotIn('C3', self.provider.reports)
        self.assertEqual(self.provider.index, 12)
        self.ok(self.client.recover())
        self.ok(self.client.unregister())

    def test_late_api_callback_from_previous_installation_refuses_without_native_write(self):
        self.start()
        old = self.api.npu_fused_infer_attention_score
        self.through(12)
        self.ok(self.client.advance('new-generation-lossless'))
        calls = len(self.transport.opened)
        with self.assertRaisesRegex(RuntimeError, 'LATE_OR_FOREIGN_CALLBACK'):
            old(**self.model.attentions[0].fixture['actual'])
        self.assertEqual(len(self.transport.opened), calls)
        self.ok(self.client.recover())
        self.ok(self.client.unregister())

    def test_removed_owner_callback_cannot_reenter_after_unregister(self):
        self.start()
        old = self.api.npu_fused_infer_attention_score
        self.ok(self.client.unregister())
        before = self.model.consumer_calls
        with self.assertRaisesRegex(RuntimeError, 'LATE_OR_FOREIGN_CALLBACK'):
            old(**self.model.attentions[0].fixture['actual'])
        self.assertEqual(self.model.consumer_calls, before)
        self.assertFalse(self.transport.opened)

    def test_query_uses_ownership_walker_read_only_without_native_or_host_mutation(self):
        self.start()
        api = self.api.npu_fused_infer_attention_score
        writer = DeviceOperator.__dict__['reshape_and_cache']
        with patch.object(postwrite_collector._Edit, 'set', side_effect=AssertionError('query mutated API')):
            result = self.client.query()
        self.assertEqual(result['status'], 'QUERY_OK', result)
        self.assertIs(self.api.npu_fused_infer_attention_score, api)
        self.assertIs(DeviceOperator.__dict__['reshape_and_cache'], writer)
        self.assertFalse(self.transport.opened)
        self.assertEqual(self.transport.restore_attempts, 0)
        self.assertEqual(self.model.forward_calls, 0)
        native = result['response']['provider_state']
        self.assertEqual(native['native_registration'], 'NOT_APPLICABLE_DIRECT_ABI_LEASES')
        self.assertTrue(native['owned_api_state']['read_only'])
        self.assertGreater(native['owned_api_state']['owned_links_present'], 0)
        self.ok(self.client.unregister())

    def cleanup_failure(self, mode):
        self.start()
        p = self.provider.p
        handle = p.handles[-1]
        remove = handle.remove
        def fail():
            if mode == 'after':
                remove()
            if mode != 'silent':
                raise RuntimeError('EXPLICIT_CPU_HOOK_REMOVE_' + mode)
        with patch.object(handle, 'remove', fail):
            refused = self.client.unregister()
        self.assertEqual(refused['status'], 'PROVIDER_ACTION_FAILED', refused)
        self.assertTrue(refused['ownership_retained'])
        self.assertIn('41.tail_column', self.worker._demoncat_worker_fault_runtime._leases)
        self.assertTrue(getattr(self.worker, serving41._ATTRIBUTE).leases)
        self.assertEqual(handle in p.handles, mode != 'after')
        self.ok(self.client.unregister())
        self.assertFalse(p.handles or p.patches or p.installed)

    def test_hook_remove_before_failure_retains_owner_and_retry_cleans(self):
        self.cleanup_failure('before')

    def test_hook_remove_after_failure_uses_actual_readback_on_retry(self):
        self.cleanup_failure('after')

    def test_hook_silent_remove_failure_is_not_a_cleanup_receipt(self):
        self.cleanup_failure('silent')

    def test_foreign_hook_is_preserved_and_owner_remains_until_repaired(self):
        self.start()
        p = self.provider.p
        handle = p.handles[-1]
        hooks, owned = handle.hooks_dict_ref(), p.hook_owners[handle.id]
        foreign = lambda *args: None
        hooks[handle.id] = foreign
        try:
            refused = self.client.unregister()
            self.assertEqual(refused['status'], 'PROVIDER_ACTION_FAILED')
            self.assertTrue(refused['ownership_retained'])
            self.assertIs(hooks[handle.id], foreign)
        finally:
            hooks[handle.id] = owned
        self.ok(self.client.unregister())

    def test_clean_keeps_binding_ownership_until_exact_unregister(self):
        self.start()
        self.ok(self.client.clean())
        binding = self.provider.binding
        owned = set(binding.leases)
        binding.leases.clear()
        binding.leases.add(('foreign-native-owner', 99))
        try:
            refused = self.client.unregister()
            self.assertEqual(refused['status'], 'PROVIDER_ACTION_FAILED')
            self.assertTrue(refused['ownership_retained'])
            self.assertEqual(binding.leases, {('foreign-native-owner', 99)})
        finally:
            binding.leases.clear()
            binding.leases.update(owned)
        self.ok(self.client.unregister())

    def test_cleanup_during_actual_forward_retains_owner_and_stops_further_native_calls(self):
        self.start()
        self.through(3)
        self.ok(self.client.advance('actual-inflight-cleanup'))
        receipts = []
        def during(module, args):
            receipts.append(self.client.unregister())
        hook = self.model.attentions[0].register_forward_pre_hook(during)
        opened = len(self.transport.opened)
        try:
            with self.assertRaisesRegex(RuntimeError, 'FAILED_CALLBACK_REQUIRES_RECOVERY'):
                self.actual_request('actual-inflight-cleanup')
            self.assertEqual(len(receipts), 1)
            self.assertEqual(receipts[0]['status'], 'PROVIDER_ACTION_FAILED')
            self.assertTrue(receipts[0]['ownership_retained'])
            self.assertEqual(len(self.transport.opened), opened)
            self.assertFalse(self.provider.root_active)
        finally:
            hook.remove()
        self.ok(self.client.recover())
        self.ok(self.client.advance('actual-off-after-inflight-refusal'))
        self.actual_request('actual-off-after-inflight-refusal')
        self.ok(self.client.finish())
        self.ok(self.client.unregister())

    def test_foreign_thread_root_post_cannot_forge_unwind_of_actual_forward(self):
        self.start()
        self.ok(self.client.advance('actual-foreign-thread-callback'))
        root_post = next(reversed(self.model._forward_hooks.values()))
        failures = []
        def foreign():
            try:
                root_post(self.model, (), {}, None)
            except BaseException as error:
                failures.append(str(error))
        def during(module, args):
            self.assertTrue(self.provider.root_active)
            thread = threading.Thread(target=foreign)
            thread.start()
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
            self.assertTrue(self.provider.root_active)
            self.assertFalse(self.provider.root_unwound)
        hook = self.model.attentions[0].register_forward_pre_hook(during)
        try:
            with self.assertRaisesRegex(RuntimeError, 'FAILED_CALLBACK_REQUIRES_RECOVERY'):
                self.actual_request('actual-foreign-thread-callback')
            self.assertEqual(failures, ['KV41_HELD_WORKER_CONTEXT_CHANGED'])
            self.assertFalse(self.provider.root_active)
            self.assertTrue(self.provider.root_unwound)
            self.assertFalse(self.transport.opened)
        finally:
            hook.remove()
        self.ok(self.client.recover())
        self.ok(self.client.unregister())

    def test_foreign_api_query_and_cleanup_never_overwrite_it(self):
        self.start()
        owned = self.api.npu_fused_infer_attention_score
        foreign = lambda **kwargs: None
        self.api.npu_fused_infer_attention_score = foreign
        try:
            self.assertEqual(self.client.query()['status'], 'NATIVE_STATE_UNAVAILABLE')
            refused = self.client.unregister()
            self.assertEqual(refused['status'], 'PROVIDER_ACTION_FAILED')
            self.assertIs(self.api.npu_fused_infer_attention_score, foreign)
            self.assertTrue(self.provider.p.handles and self.provider.p.patches)
        finally:
            self.api.npu_fused_infer_attention_score = owned
        self.ok(self.client.unregister())

    def test_uncertain_native_readback_does_not_release_runtime_owner_or_binding(self):
        self.start()
        self.transport.fail_readback = True
        self.assertEqual(self.client.query()['status'], 'NATIVE_STATE_UNAVAILABLE')
        refused = self.client.unregister()
        self.assertEqual(refused['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(refused['ownership_retained'])
        with self.assertRaisesRegex(RuntimeError, 'WORKER_RPC_OWNER_STILL_RETAINED'):
            serving41.remove_fault41_serving_binding(self.worker)
        self.transport.fail_readback = False
        self.ok(self.client.unregister())
        serving41.remove_fault41_serving_binding(self.worker)

    def test_group_cleanup_failure_keeps_all_reservations_then_real_health_finishes_recovery(self):
        self.start()
        self.through(11)  # C2 OFF has actually returned; action must consume it.
        handle = self.provider.p.handles[-1]
        with patch.object(handle, 'remove', side_effect=RuntimeError('EXPLICIT_CPU_GROUP_HOOK_FAILURE')):
            refused = self.client.advance('health-must-not-be-scheduled-yet')
        self.assertEqual(refused['status'], 'PROVIDER_ACTION_FAILED', refused)
        self.assertEqual(self.provider.group.stage, 'recovery_pending')
        self.assertEqual(len(self.provider.group.session.recovery_report()['held_leases']), 2)
        calls = self.model.forward_calls
        state = self.ok(self.client.recover())
        self.assertEqual(self.model.forward_calls, calls)
        self.assertEqual(state['serving']['next_phase'], 'recovery_health')
        self.ok(self.client.advance('actual-recovery-health'))
        self.actual_request('actual-recovery-health')
        state = self.ok(self.client.finish())
        self.assertTrue(state['coverage']['failed_attempt_recovery']['recovery']['complete'])
        self.assertFalse(state['serving']['complete'])
        self.assertEqual(state['coverage_held_leases'], 0)
        self.ok(self.client.unregister())

    def test_health_hook_after_removal_failure_is_retryable_without_forging_health(self):
        self.start()
        self.through(12)
        group = self.provider.group
        handle = group.health_handles[-1]
        remove = handle.remove
        def after():
            remove()
            raise RuntimeError('EXPLICIT_CPU_HEALTH_AFTER_REMOVE')
        with patch.object(handle, 'remove', after):
            refused = self.client.advance('c3-not-yet-authorized')
        self.assertEqual(refused['status'], 'PROVIDER_ACTION_FAILED', refused)
        self.assertNotIn(handle, group.health_handles)
        self.assertEqual(len(group.session.recovery_report()['held_leases']), 2)
        self.ok(self.client.recover())
        self.ok(self.client.advance('new-actual-health'))
        self.actual_request('new-actual-health')
        self.ok(self.client.finish())
        self.ok(self.client.unregister())

    def fail_native_c1(self):
        self.start()
        self.through(3)
        self.ok(self.client.advance('actual-failed-native-c1'))
        self.transport.fail_restore = True
        with self.assertRaisesRegex(RuntimeError, 'EXPLICIT_CPU_NATIVE_RESTORE_FAILED'):
            self.actual_request('actual-failed-native-c1')
        self.assertIsNotNone(self.provider.p.lease)
        return self.provider.p.lease

    def test_native_restore_failure_keeps_span_then_explicit_off_and_post_unregister_health(self):
        lease = self.fail_native_c1()
        refused = self.client.unregister()
        self.assertEqual(refused['status'], 'PROVIDER_ACTION_FAILED')
        self.assertIs(self.provider.p.lease, lease)
        self.assertTrue(lease.live)
        first = self.provider.error
        self.transport.fail_restore = False
        state = self.ok(self.client.recover())
        self.assertEqual(state['serving']['next_phase'], 'recovery_c1_off')
        self.assertTrue(any(row['reason'] == first for row in state['serving']['failures']))
        self.ok(self.client.advance('real-off-after-native-recovery'))
        self.actual_request('real-off-after-native-recovery')
        state = self.ok(self.client.finish())
        self.assertTrue(state['serving']['recovered'])
        self.assertFalse(state['serving']['complete'])
        self.assertNotIn('C2', state['coverage'])
        self.ok(self.client.unregister())
        healthy = self.actual_request('normal-after-unregister')
        self.assertTrue(torch.equal(healthy, self.provider.p.outputs[0]))
        self.assertTrue(all(lease.closed for lease in self.transport.opened))

    def test_changed_pending_cache_storage_refuses_readback_and_restore_write(self):
        lease = self.fail_native_c1()
        cache = lease.plan.cache
        original_storage, shape, stride = cache.untyped_storage(), tuple(cache.shape), tuple(cache.stride())
        offset = cache.storage_offset()
        self.transport.fail_restore = False
        with torch.inference_mode():
            cache.set_(torch.empty_like(cache))
        try:
            self.assertEqual(self.client.query()['status'], 'NATIVE_STATE_UNAVAILABLE')
            refused = self.client.recover()
            self.assertEqual(refused['status'], 'PROVIDER_ACTION_FAILED')
            self.assertTrue(refused['ownership_retained'])
            self.assertTrue(lease.live)
        finally:
            # Explicit CPU fixture repair only; production never rebuilds an
            # authorization from a saved pointer or reconstructed tensor.
            with torch.inference_mode():
                cache.set_(original_storage, offset, shape, stride)
        self.ok(self.client.recover())
        self.ok(self.client.advance('current-storage-off'))
        self.actual_request('current-storage-off')
        self.ok(self.client.finish())
        self.ok(self.client.unregister())

    def test_once_exhausted_passthrough_still_checks_full_key_prefix_and_keeps_denominator(self):
        self.start()
        self.through(9)  # Both actual targets have fired once.
        attention = self.model.attentions[0]
        original = attention.initial_key.clone()
        attention.initial_key.reshape(-1)[0] += 1
        opened = len(self.transport.opened)
        self.ok(self.client.advance('once-exhausted-with-changed-prefix'))
        try:
            with self.assertRaisesRegex(RuntimeError, 'ONCE_EXHAUSTED_OR_CONTROL_PREFIX_MISMATCH'):
                self.actual_request('once-exhausted-with-changed-prefix')
            self.assertEqual(len(self.transport.opened), opened)
            self.assertEqual(len(self.provider.group.identities), 2)
            self.assertEqual(len(self.provider.group.session.recovery_report()['held_leases']), 2)
            self.assertNotIn('C2', self.provider.reports)
        finally:
            attention.initial_key.copy_(original)
        self.ok(self.client.recover())
        self.ok(self.client.advance('actual-full-prefix-off'))
        self.actual_request('actual-full-prefix-off')
        self.ok(self.client.advance('actual-full-prefix-health'))
        self.actual_request('actual-full-prefix-health')
        state = self.ok(self.client.finish())
        self.assertFalse(state['coverage']['failed_attempt_recovery']['contract_satisfied'])
        self.assertEqual(state['coverage']['failed_attempt_recovery']['planned_targets'], 2)
        self.assertTrue(state['coverage']['failed_attempt_recovery']['recovery']['complete'])
        self.assertTrue(state['serving']['failures'])
        self.ok(self.client.unregister())

    def test_actual_health_output_mismatch_retains_group_until_new_natural_health(self):
        self.start()
        self.through(11)
        self.model.output_shift = 65536  # Survives BF16 rounding of this fixture.
        returned = self.enqueue(11)
        self.assertFalse(torch.equal(returned, self.provider.p.outputs[0]))
        refused = self.client.advance('c3-not-authorized-by-bad-health')
        self.assertEqual(refused['status'], 'PROVIDER_ACTION_FAILED', repr(refused)[:1200])
        self.assertIn('HEALTH_OUTPUT_MISMATCH', repr(refused))
        self.assertTrue(refused['ownership_retained'])
        self.assertEqual(len(self.provider.group.session.recovery_report()['held_leases']), 2)
        self.assertNotIn('C2', self.provider.reports)
        self.assertIsNone(self.provider.proof)
        self.model.output_shift = 0
        before = self.model.forward_calls
        self.ok(self.client.recover())
        self.assertEqual(self.model.forward_calls, before)
        self.ok(self.client.advance('actual-new-uninstalled-health'))
        self.actual_request('actual-new-uninstalled-health')
        state = self.ok(self.client.finish())
        self.assertTrue(state['coverage']['failed_attempt_recovery']['recovery']['complete'])
        self.assertEqual(state['coverage_held_leases'], 0)
        self.assertFalse(state['serving']['complete'])
        self.assertNotIn('C2', state['coverage'])
        self.assertTrue(state['serving']['failures'])
        self.ok(self.client.unregister())

    def test_changed_native_handle_owner_or_generation_is_never_restored(self):
        lease = self.fail_native_c1()
        handle = lease.native
        self.transport.fail_restore = False
        for attribute in ('owner', 'generation'):
            original = getattr(handle, attribute)
            setattr(handle, attribute, original + 1)
            try:
                self.assertEqual(self.client.query()['status'], 'NATIVE_STATE_UNAVAILABLE')
                before = lease.plan.cache.clone()
                refused = self.client.recover()
                self.assertEqual(refused['status'], 'PROVIDER_ACTION_FAILED')
                self.assertTrue(refused['ownership_retained'])
                self.assertTrue(torch.equal(before, lease.plan.cache))
                self.assertIs(lease.native, handle)
            finally:
                setattr(handle, attribute, original)
        self.ok(self.client.recover())
        self.ok(self.client.advance('restored-original-handle-off'))
        self.actual_request('restored-original-handle-off')
        self.ok(self.client.finish())
        self.ok(self.client.unregister())

    def test_later_root_output_hook_cannot_qualify_intermediate_return(self):
        self.start()
        hook = self.model.register_forward_hook(lambda module, args, output: output + 1)
        self.ok(self.client.advance('late-root-output'))
        try:
            with self.assertRaisesRegex(RuntimeError, 'FINAL_ROOT_RETURN_OBSERVER_REQUIRED'):
                self.actual_request('late-root-output')
            self.assertFalse(self.transport.opened)
            self.assertIsNotNone(self.provider.error)
        finally:
            hook.remove()
        self.ok(self.client.unregister())

    def test_later_root_input_hook_cannot_qualify_arguments_that_the_model_did_not_receive(self):
        self.start()
        hook = self.model.register_forward_pre_hook(lambda module, args, kwargs:
            (args, dict(kwargs, input_ids=kwargs['input_ids'] + 1)), with_kwargs=True)
        self.ok(self.client.advance('late-root-input'))
        try:
            with self.assertRaisesRegex(RuntimeError, 'FINAL_ROOT_INPUT_OBSERVER_REQUIRED'):
                self.actual_request('late-root-input')
            self.assertEqual(self.model.forward_calls, 0)
            self.assertFalse(self.transport.opened)
        finally:
            hook.remove()
        self.ok(self.client.unregister())

    def test_public_settings_cannot_supply_layer_pointer_or_serialized_proof(self):
        self.start()
        ticket = self.client.snapshot()['ticket']
        result = self.worker.demoncat_fault_rpc('advance', dict(ticket,
            settings=dict(request_id='forged', layer_name='actual.0', address=123, c2_proof={})))
        self.assertEqual(result['status'], 'PROVIDER_ACTION_FAILED')
        self.assertEqual(result['provider_code'], 'UNSUPPORTED_PROVIDER_SETTINGS')
        self.assertEqual(self.model.forward_calls, 0)
        self.assertFalse(self.transport.opened)
        self.ok(self.client.unregister())

    def test_unbound_factory_refuses_inertly_and_never_creates_a_native_owner(self):
        client = WorkerFaultClient(HostAdapter(self.executor), FaultConfig(41, 'tail_column'))
        result = client.prepare()
        self.assertEqual(result['status'], 'PROVIDER_UNAVAILABLE', result)
        self.assertFalse(self.worker._demoncat_worker_fault_runtime._leases)
        self.assertFalse(self.transport.opened)
        self.assertEqual(self.model.forward_calls, 0)


if __name__ == '__main__':
    unittest.main()
