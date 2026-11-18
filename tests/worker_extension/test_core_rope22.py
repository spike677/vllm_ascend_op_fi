"""Real Host/Runtime/torch hook contracts; CPU evidence grants no qualification."""
import copy
from contextlib import ExitStack
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from rope22_fixture import Processor, WorkerModel, template, JIT
from demoncat.runtime.vllm_ascend.worker_extension import identity, serving22, serving_tensor
from demoncat.runtime.vllm_ascend.worker_extension.providers import fault22
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
from test_extension import _extended_worker
from test_core_host import ExistingExecutor


class CountedModel(WorkerModel):
    def __init__(self):
        super().__init__()
        self.forward_calls = self.logits_calls = 0
        self.fail_forward = self.fail_logits = False

    def forward(self, input_ids, positions):
        self.forward_calls += 1
        if self.fail_forward:
            raise RuntimeError('real root failure')
        return super().forward(input_ids, positions)

    def logits(self, hidden):
        self.logits_calls += 1
        if self.fail_logits:
            raise RuntimeError('real logits failure')
        return super().logits(hidden)


class CoreRoPE22Tests(unittest.TestCase):
    def setUp(self):
        JIT.skip_launch = JIT.replace_position = False
        self.patches = ExitStack()
        fake_logits = ModuleType('vllm.model_executor.layers.logits_processor')
        fake_logits.LogitsProcessor = Processor
        fake_context = ModuleType('vllm.forward_context')
        fake_context.get_forward_context = lambda: SimpleNamespace(cpu_contract=True)
        self.patches.enter_context(patch.dict(sys.modules, {fake_logits.__name__: fake_logits,
                                                           fake_context.__name__: fake_context}))
        self.patches.enter_context(patch.object(serving_tensor, 'require_serial_eager', lambda runner: None))
        self.patches.enter_context(patch.object(serving_tensor, 'discover_logits_boundary',
            lambda model, **kwargs: dict(processor=model.processor, report=dict(cpu_contract=True))))

        def snapshot(runner, model, actual, context, **kwargs):
            self.assertTrue(context.cpu_contract)
            self.assertEqual(actual['input_ids'].tolist(), kwargs['expected_prompt_token_ids'])
            self.assertNotIn(kwargs['expected_internal_request_id'], kwargs['seen_internal_request_ids'])
            return dict(cpu_contract=True, request_id=kwargs['expected_internal_request_id'])

        self.patches.enter_context(patch.object(serving_tensor, 'snapshot_fresh_prefill', snapshot))
        self.worker = _extended_worker(rank=7)
        self.worker.load_model()
        self.model = CountedModel().eval()
        self.worker.model_runner.model = self.model
        self.worker.model_runner.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(vocab_size=4)))
        self.patches.enter_context(patch.object(identity, '_current_device', return_value=0))
        self.patches.enter_context(patch.object(torch.distributed, 'is_initialized', return_value=True))
        self.patches.enter_context(patch.object(torch.distributed, 'get_rank', return_value=7))
        self.patches.enter_context(patch.object(serving22, 'require_serial_eager', lambda runner: None))
        self.patches.enter_context(patch.object(fault22, 'snapshot_fresh_prefill',
                                                serving_tensor.snapshot_fresh_prefill))
        self.executor = ExistingExecutor([self.worker])
        self.binding_calls = []

        def bind(actual_identity, config):
            self.binding_calls.append((copy.deepcopy(actual_identity), config.as_dict()))
            self.assertEqual(actual_identity, self.worker.demoncat_worker_identity()['worker'])
            serving22.install_fault22_serving_binding(self.worker, prompt_token_ids=[1, 2, 3],
                weight_manifest={'cpu_fixture': True}, workload_sha256='a'*64,
                environment_sha256='b'*64, first_live_validation=True,
                _simulation=True, _synchronize=lambda: None, _trace_factory=template)
            return {}

        self.client = WorkerFaultClient(HostAdapter(self.executor, prepare_binding=bind),
                                       FaultConfig(22, 'wrong_row'))

    def tearDown(self):
        runtime = getattr(self.worker, '_demoncat_worker_fault_runtime', None)
        lease = runtime._leases.get('22.wrong_row') if runtime else None
        if lease:
            provider = lease.provider
            try:
                provider._owned_abort()
                group = getattr(getattr(provider.session, 'provider', None), 'group', None)
                if group is not None and not group.recovery_report()['complete']:
                    # Test failure teardown only, never a passing Recovery claim.
                    group.disable(timeout_seconds=0)
            finally:
                self.patches.close()
        else:
            self.patches.close()

    def ok(self, row):
        self.assertEqual(row['status'], 'ACTION_COMPLETED', row['response'])
        return row['response']

    def start(self):
        for action in ('prepare', 'register', 'check', 'enable'):
            self.ok(getattr(self.client, action)())
        self.provider = self.worker._demoncat_worker_fault_runtime._leases['22.wrong_row'].provider

    def enqueue(self, index):
        self.ok(self.client.advance('actual-enqueued-' + str(index)))
        # This is the serving executor's actual model call, never an RPC replay.
        hidden = self.model(input_ids=torch.tensor([1, 2, 3]), positions=torch.arange(3))
        return self.model.logits(hidden)

    def first_four(self):
        self.start()
        for index in range(4):
            self.enqueue(index)

    def assert_clean(self):
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks
                            for module in self.model.modules()))
        self.assertFalse(getattr(self.worker, serving22._ATTRIBUTE).leases)

    def test_public_host_real_group_original_proof_full_denominator_and_clean_health(self):
        self.start()
        # Runtime's identity read still runs, but provider recursion is forbidden.
        with patch.object(self.worker, 'demoncat_worker_identity', side_effect=AssertionError('recursive RPC')):
            for index in range(10):
                self.enqueue(index)
                queried = self.client.query()
                self.assertEqual(queried['status'], 'QUERY_OK', queried['response'])
                state = queried['response']['provider_state']
                self.assertFalse(state['enabled'])
                self.assertEqual((state['inflight'], state['leases']), (0, 0))
            state = self.ok(self.client.finish())['provider_state']
        coverage = state['coverage']
        self.assertEqual(state['selected_variant'], '22.wrong_row.fused_positions')
        self.assertEqual((coverage['C2']['hits'], coverage['C3']['hits']), (3, 6))
        self.assertEqual((coverage['C2']['observed_eligible_calls'], coverage['C3']['observed_eligible_calls']), (3, 6))
        issued = coverage['C2']['live_c2_proof']['issued'][0]
        used = coverage['C3']['live_c2_proof']['consumed']
        self.assertEqual(issued['proof_sha256'], used['proof_sha256'])
        self.assertEqual(issued['object_binding'], used['object_binding'])
        self.assertEqual(used['proof']['owner'], state['owner'])
        self.assertLess(used['proof']['generation'], used['consumer']['generation'])
        self.assertEqual(used['proof']['scope'], used['consumer']['scope'])
        self.assertEqual(used['proof']['target_ids'], used['consumer']['target_ids'])
        self.assertEqual(used['object_binding']['python_object_id'], id(self.provider.proof))
        self.assertFalse(coverage['C2']['real_worker_evidence'])
        self.assertEqual(used['proof']['scope']['evidence_kind'], 'cpu_contract')
        self.assertFalse(state['public_qualification_granted'])
        self.assertEqual(state['E5'], 'NOT_ASSESSED_SEPARATE_RAW_STATISTICS_REQUIRED')
        self.assertTrue(coverage['same_worker_healthy']['cleanup_before_request'])
        self.assertEqual((self.model.forward_calls, self.model.logits_calls), (10, 10))
        phases = self.provider.session.provider.rows
        self.assertEqual(len(phases), 9)
        for row in phases:
            self.assertEqual(len(row['independent']['rope_trace']['readers']), 3)
            self.assertTrue(row['independent']['rope_trace']['complete'])
            self.assertEqual(len(row['independent']['rope_trace']['root_forwards']), 1)
            self.assertTrue(all(item['original_calls'] == 1
                for item in row['independent']['rope_trace']['outer_operators']))
        self.ok(self.client.disable()); self.ok(self.client.unregister())
        self.assertEqual(self.client.detach()['status'], 'DETACHED')
        self.assert_clean()
        # No owner: ordinary serving continues without any managed request ID.
        hidden = self.model(input_ids=torch.tensor([1, 2, 3]), positions=torch.arange(3))
        self.model.logits(hidden)
        self.assertEqual(self.model.forward_calls, 11)
        self.assertEqual(set(self.binding_calls[0][1]), {'fault_id', 'variant', 'implementation_version'})
        self.assertEqual(self.binding_calls[0][0]['rank'], 7)

    def test_advance_without_actual_enqueue_never_runs_model_or_qualifies(self):
        self.start()
        self.ok(self.client.advance('queued-not-run'))
        self.assertEqual(self.model.forward_calls, 0)
        failed = self.client.finish()
        self.assertEqual(failed['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(failed['ownership_retained'])
        self.assertEqual(set(self.provider.reports), {'deployment'})
        self.ok(self.client.unregister())
        self.assert_clean()

    def test_actual_positional_root_arguments_enter_owned_hooks_once(self):
        self.start()
        self.ok(self.client.advance('positional-root'))
        hidden = self.model(torch.tensor([1, 2, 3]), torch.arange(3))
        self.model.logits(hidden)
        row, = self.provider.session.provider.rows
        self.assertTrue(row['independent']['rope_trace']['complete'])
        self.assertEqual(len(row['independent']['rope_trace']['root_forwards']), 1)
        self.assertEqual(len(row['independent']['rope_trace']['readers']), 3)
        self.assertEqual(self.model.forward_calls, 1)
        self.ok(self.client.unregister())
        self.assert_clean()

    def test_final_health_is_inflight_until_actual_logits_returns(self):
        self.start()
        for index in range(9):
            self.enqueue(index)
        self.ok(self.client.advance('actual-final-health'))
        hidden = self.model(input_ids=torch.tensor([1, 2, 3]), positions=torch.arange(3))
        owner, generation = self.provider._ticket
        self.assertEqual(self.provider.query(owner=owner, generation=generation)['inflight'], 1)
        self.model.logits(hidden)
        self.assertEqual(self.provider.query(owner=owner, generation=generation)['inflight'], 0)
        self.ok(self.client.finish()); self.ok(self.client.unregister())
        self.assert_clean()

    def test_foreign_owner_and_other_worker_cannot_advance_held_group(self):
        self.start()
        ticket = self.client.snapshot()['ticket']
        forged = dict(ticket, owner='foreign', settings={'request_id': 'foreign'})
        self.assertEqual(self.worker.demoncat_fault_rpc('advance', forged)['reason'], 'OWNER_MISMATCH')
        other = _extended_worker(rank=2); other.load_model()
        self.executor.workers.insert(0, other)
        self.enqueue(0)
        self.assertFalse(other._demoncat_worker_fault_runtime._leases)
        self.assertEqual(self.model.forward_calls, 1)
        self.ok(self.client.unregister())
        self.assert_clean()

    def test_partial_hook_cleanup_and_silent_removal_failure_keep_owner_for_retry(self):
        for silent in (False, True):
            with self.subTest(silent=silent):
                if silent:
                    self.tearDown(); self.setUp()
                self.start()
                handle = self.provider.session.handles[0]
                with patch.object(handle, 'remove', **({'return_value': None} if silent
                        else {'side_effect': OSError('remove failed')})):
                    row = self.client.unregister()
                    self.assertEqual(row['status'], 'PROVIDER_ACTION_FAILED')
                    self.assertTrue(row['ownership_retained'])
                    self.assertTrue(self.provider.binding.leases)
                self.ok(self.client.unregister())
                self.assert_clean()

    def test_failed_c2_forward_requires_real_off_and_never_retries_fault(self):
        self.first_four()
        self.ok(self.client.advance('fault-root-fails'))
        self.model.fail_forward = True
        with self.assertRaisesRegex(RuntimeError, 'real root failure'):
            self.model(input_ids=torch.tensor([1, 2, 3]), positions=torch.arange(3))
        self.model.fail_forward = False
        failed = self.client.query()
        self.assertTrue(failed['ownership_retained'])
        self.assertEqual(self.client.advance('no-retry')['response']['reason'], 'RECOVERY_REQUIRED')
        self.ok(self.client.recover())
        self.enqueue('actual-recovery-off')
        recovered = self.ok(self.client.finish())['provider_state']
        self.assertTrue(recovered['serving']['recovered'])
        self.assertNotIn('C2', recovered['coverage'])
        self.assertFalse(recovered['coverage']['failed_attempt_recovery']['contract_satisfied'])
        self.assertTrue(recovered['coverage']['failed_attempt_recovery']['recovery']['complete'])
        self.assertIsNone(self.provider.proof)
        self.ok(self.client.unregister())
        self.assert_clean()

    def test_missing_real_rope_site_rejects_fused_qualification_without_lookup_fallback(self):
        self.start()
        self.model.model.body.skip_last = True
        with self.assertRaisesRegex(RuntimeError, 'INDEPENDENT_ROPE_CENSUS_INCOMPLETE'):
            self.enqueue(0)
        self.assertIsNone(self.provider.session.provider.candidates)
        self.assertEqual(self.client.query()['response']['provider_state']['selected_variant'], None)
        self.ok(self.client.unregister())
        self.assert_clean()

    def test_legacy_lookup_binding_cannot_satisfy_public_fused_deployment(self):
        self.worker._dcat_tensor_module_bindings = {'22.wrong_row': object()}
        client = WorkerFaultClient(HostAdapter(self.executor), FaultConfig(22, 'wrong_row'))
        row = client.prepare()
        self.assertEqual(row['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(row['ownership_retained'])
        self.assertEqual(self.model.forward_calls, 0)
        self.ok(client.unregister())

    def test_copied_proof_from_completed_c2_never_admits_public_c3(self):
        from fault_injection.coverage_session import CoverageSession
        self.first_four()
        self.enqueue(4); self.enqueue(5)
        original = CoverageSession.c2_proof
        with patch.object(CoverageSession, 'c2_proof', lambda group: copy.copy(original(group))):
            row = self.client.advance('c3-must-refuse-copy')
        self.assertEqual(row['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(row['ownership_retained'])
        self.assertEqual(self.model.forward_calls, 6)
        self.assertNotIn('C3', self.provider.reports)
        self.ok(self.client.unregister())
        self.assert_clean()

    def test_lost_advance_reply_keeps_owner_and_only_explicit_off_can_recover(self):
        self.first_four()
        self.enqueue(4); self.enqueue(5)

        def lose_reply(method, args, replies):
            if method == 'demoncat_fault_rpc' and args[0] == 'advance':
                self.executor.after = None
                raise TimeoutError('reply lost after C3 scheduling')
            return replies

        self.executor.after = lose_reply
        row = self.client.advance('lost-c3-request')
        self.assertEqual(row['status'], 'RPC_UNCERTAIN')
        self.assertTrue(row['ownership_retained'])
        before = len(self.executor.calls)
        self.assertEqual(self.client.advance('retry-disallowed')['status'], 'HOST_REJECTED')
        self.assertEqual(before, len(self.executor.calls))
        self.assertEqual(self.model.forward_calls, 6)
        self.assertEqual(self.client.unregister()['status'], 'PROVIDER_ACTION_FAILED')
        self.ok(self.client.recover())
        self.enqueue('off-after-uncertain')
        state = self.ok(self.client.finish())['provider_state']
        self.assertTrue(state['serving']['recovered'])
        self.assertNotIn('C3', state['coverage'])
        self.ok(self.client.unregister())
        self.assert_clean()

    def test_actual_processor_exception_retains_owner_and_requires_natural_off(self):
        self.first_four()
        self.ok(self.client.advance('fault-logits-fails'))
        hidden = self.model(input_ids=torch.tensor([1, 2, 3]), positions=torch.arange(3))
        with patch.object(self.model.processor, 'forward', side_effect=RuntimeError('processor failed')):
            with self.assertRaisesRegex(RuntimeError, 'processor failed'):
                self.model.logits(hidden)
        self.assertTrue(self.client.query()['ownership_retained'])
        self.ok(self.client.recover())
        self.enqueue('off-after-logits-failure')
        state = self.ok(self.client.finish())['provider_state']
        self.assertTrue(state['serving']['recovered'])
        self.assertNotIn('C2', state['coverage'])
        self.ok(self.client.unregister())
        self.assert_clean()

    def _foreign_return_failure(self, logits):
        module = self.model.processor if logits else self.model

        def fail(*args):
            raise RuntimeError('foreign return callback failed')

        handle = module.register_forward_hook(fail)
        try:
            self.start()
            with self.assertRaisesRegex(RuntimeError, 'foreign return callback failed'):
                self.enqueue(0)
            self.assertTrue(self.provider.session.request_error)
            self.assertEqual(self.provider.session.seen, [])
            self.assertTrue(self.client.query()['ownership_retained'])
            self.ok(self.client.unregister())
            self.assertIn(handle.id, module._forward_hooks)
        finally:
            handle.remove()
        self.assert_clean()

    def test_foreign_root_return_failure_cannot_be_marked_as_a_completed_request(self):
        self._foreign_return_failure(logits=False)

    def test_foreign_logits_return_failure_cannot_be_marked_as_a_completed_request(self):
        self._foreign_return_failure(logits=True)

    def test_real_forward_rechecks_held_device_and_thread_without_recursive_identity(self):
        self.start()
        self.ok(self.client.advance('device-drift'))
        with patch.object(identity, '_current_device', return_value=1):
            with self.assertRaisesRegex(RuntimeError, 'ROPE_HELD_WORKER_CONTEXT_CHANGED'):
                self.model(input_ids=torch.tensor([1, 2, 3]), positions=torch.arange(3))
        self.assertEqual(self.model.forward_calls, 0)
        self.ok(self.client.unregister())
        self.assert_clean()

    def test_real_forward_refuses_other_thread_and_retains_owned_cleanup(self):
        self.start()
        self.ok(self.client.advance('wrong-thread'))
        errors = []

        def wrong_thread():
            try:
                self.model(input_ids=torch.tensor([1, 2, 3]), positions=torch.arange(3))
            except RuntimeError as error:
                errors.append(str(error))

        thread = threading.Thread(target=wrong_thread)
        thread.start(); thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIn('THREAD_CHANGED', errors[0])
        self.assertEqual(self.model.forward_calls, 0)
        self.ok(self.client.unregister())
        self.assert_clean()

    def test_held_model_and_pid_are_rechecked_at_actual_root(self):
        self.start()
        self.ok(self.client.advance('model-drift'))
        self.worker.model_runner.model = torch.nn.Identity()
        try:
            with self.assertRaisesRegex(RuntimeError, 'MODEL_LIFETIME_CHANGED'):
                self.model(input_ids=torch.tensor([1, 2, 3]), positions=torch.arange(3))
        finally:
            self.worker.model_runner.model = self.model
        self.assertEqual(self.model.forward_calls, 0)
        self.ok(self.client.unregister())
        self.assert_clean()

    def test_pid_drift_between_rpc_and_actual_root_is_refused(self):
        self.start()
        self.ok(self.client.advance('pid-drift'))
        with patch.object(fault22.os, 'getpid', return_value=self.provider.identity['pid'] + 1):
            with self.assertRaisesRegex(RuntimeError, 'ROPE_HELD_WORKER_CONTEXT_CHANGED'):
                self.model(input_ids=torch.tensor([1, 2, 3]), positions=torch.arange(3))
        self.assertEqual(self.model.forward_calls, 0)
        self.ok(self.client.unregister())
        self.assert_clean()

    def test_partial_group_construction_retains_owned_base_hooks_for_cleanup(self):
        for action in ('prepare', 'register', 'check'):
            self.ok(getattr(self.client, action)())
        with patch('fault_injection.rope_serving.RoPEGroupProvider', side_effect=RuntimeError('init failed')):
            row = self.client.enable()
        self.assertEqual(row['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(row['ownership_retained'])
        self.assertTrue(self.model._forward_pre_hooks)
        self.ok(self.client.unregister())
        self.assert_clean()

    def test_failed_group_drain_retries_owned_cleanup_against_same_sealed_off(self):
        self.first_four()
        self.enqueue(4); self.enqueue(5)
        with patch.object(self.provider.session.provider, 'synchronize', side_effect=OSError('drain failed')):
            row = self.client.advance('c3-not-opened')
        self.assertEqual(row['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(row['ownership_retained'])
        group = self.provider.session.provider.group
        sealed = group.observer.evidence
        self.assertIsNotNone(sealed)
        self.assertFalse(group.recovery_report()['complete'])
        self.assertIsNone(self.provider.proof)
        self.ok(self.client.unregister())
        self.assertEqual(group.observer.evidence, sealed)
        self.assertTrue(group.recovery_report()['complete'])
        self.assertEqual(self.model.forward_calls, 6)
        self.assert_clean()

    def _scoped_removal_failure(self, kind):
        self.start()
        self.enqueue(0)
        self.ok(self.client.advance('partial-' + kind))
        hidden = self.model(input_ids=torch.tensor([1, 2, 3]), positions=torch.arange(3))
        resource = getattr(self.provider.session, kind)
        handles = resource.handles if kind == 'observer' else resource.hooks
        handle, _ = handles[0]
        with patch.object(handle, 'remove', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'removal|REMOVAL'):
                self.model.logits(hidden)
            if kind == 'trace':
                self.assertFalse(resource.report()['complete'])
            self.assertTrue(self.client.query()['ownership_retained'])
            self.assertEqual(self.client.unregister()['status'], 'PROVIDER_ACTION_FAILED')
        self.ok(self.client.unregister())
        self.assert_clean()

    def test_trace_partial_cleanup_retains_readback_and_can_retry(self):
        self._scoped_removal_failure('trace')

    def test_adapter_partial_cleanup_retains_readback_and_can_retry(self):
        self._scoped_removal_failure('adapter')

    def test_observer_partial_cleanup_retains_readback_and_can_retry(self):
        self._scoped_removal_failure('observer')

    def test_foreign_root_pre_hook_is_preserved_and_not_reordered(self):
        self.start()
        foreign_calls = []
        handle = self.model.register_forward_pre_hook(lambda *args: foreign_calls.append('foreign'))
        try:
            with self.assertRaisesRegex(RuntimeError, 'ROOT_PRE_HOOK_ORDER_UNREVIEWED'):
                self.enqueue(0)
            self.assertEqual(foreign_calls, [])
            self.ok(self.client.unregister())
            self.assertIn(handle.id, self.model._forward_pre_hooks)
        finally:
            handle.remove()
        self.assert_clean()


class TorchRootHookSnapshotTests(unittest.TestCase):
    def test_newly_registered_pre_hook_is_absent_from_current_forward_snapshot(self):
        model = torch.nn.Identity()
        events, handles = [], []

        def install(module, args):
            events.append('existing')
            if len(handles) == 1:
                handles.append(module.register_forward_pre_hook(lambda *args: events.append('new')))

        handles.append(model.register_forward_pre_hook(install))
        try:
            value = torch.tensor([1.])
            self.assertIs(model(value), value)
            self.assertEqual(events, ['existing'])
            self.assertIs(model(value), value)
            self.assertEqual(events, ['existing', 'existing', 'new'])
        finally:
            for handle in handles:
                handle.remove()


if __name__ == '__main__':
    unittest.main()
