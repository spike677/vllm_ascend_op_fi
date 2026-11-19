"""Ordinary actual CPU forwards with typed Native doubles, never CANN claims."""
import copy
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from fault_injection import quant_group_native as native
from demoncat.runtime.vllm_ascend.worker_extension import serving_quant_projection as ordinary
from demoncat.runtime.vllm_ascend.worker_extension.providers.quant import QuantProvider
from test_quant_group_c1 import CpuGroupBinding
import test_quant_group_c1 as c1_fixtures


class CpuBothOutputs(CpuGroupBinding):
    """Ownership fixture; these constants are not a quantization oracle."""
    def __init__(self):
        super().__init__()
        self.change_scale = True

    def _quant(self, x):
        finished = self.finished
        def done(value):
            if value.admitted and value.core.mode == 3 and self.change_scale and not self.no_effect:
                value.core.changed_scale_bytes = 4
            return finished(value)
        if finished is not None:
            self.finished = done
        try:
            q, scale = super()._quant(x)
            if self.lease is not None and self.lease.mode == 3 and self.change_scale and not self.no_effect:
                scale[0] = 2
                self.completion.core.changed_scale_bytes = 4
            return q, scale
        finally:
            self.finished = finished


class OrdinaryQuantTests(unittest.TestCase):
    def setUp(self):
        # Reuse only the explicit CPU Worker/model/native-layout fixtures. The
        # qualification session is closed before the ordinary session exists.
        fixture = c1_fixtures.QuantGroupC1Tests()
        fixture.setUp()
        fixture.session.close(abort=True)
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.binding = CpuBothOutputs()
        self.worker, self.model, self.runner, self.api = fixture.worker, fixture.model, fixture.runner, fixture.api
        self.runner.vllm_config.model_config.hf_config.model_type = 'qwen3'
        self.api.npu_dynamic_quant = self.binding.quant
        self.enterContext(patch.object(native.QuantGroupNative, 'load', return_value=self.binding))
        self.session = ordinary.QuantProjectionServingSession(self.worker, self.model, root='CPU_ONLY',
            build_sha256='b'*64, owner='ordinary-owner', generation=4, worker_identity=fixture.identity)
        self.provider = QuantProvider(self.worker, self.model, fixture.identity)
        self.batch_number = 0
        deployment = ordinary.QuantProjectionServingBinding(lambda *a, **k: self.session)
        self.enterContext(patch('fault_injection.providers.resolve_runtime', return_value=deployment))
        self.addCleanup(self.cleanup)
        for action in ('prepare', 'register', 'check', 'enable'):
            self.dispatch(action)

    def test_actual_schema_not_model_name_controls_candidate_discovery(self):
        self.session.close(abort=True)
        self.runner.vllm_config.model_config.hf_config.model_type = 'unqualified_cpu_structure_fixture'
        session = ordinary.QuantProjectionServingSession(self.worker, self.model, root='CPU_ONLY',
            build_sha256='b'*64, owner='new-scope', generation=5, worker_identity=self.fixture.identity)
        self.assertTrue(session.candidates)
        self.assertIsNone(session.target)  # Structure alone never selects a target.
        self.assertEqual(session.rows, [])
        self.assertFalse(session.recovered)

    def cleanup(self):
        if self.session.installed:
            # CPU fixture teardown only, never production recovery authority.
            self.binding.drain_result = True
            self.session.phase_attempts.discard('fault')
            self.session.error = None
            self.session.close(abort=True)

    def dispatch(self, action, settings=None):
        return self.provider.dispatch(action, settings or {}, owner='ordinary-owner', generation=4)

    def phase(self, *, collect=True, after_advance=None):
        index = self.batch_number
        self.batch_number += 1
        ids = ['ordinary-%d-%d' % (index, value) for value in range(8)]
        self.dispatch('advance', {'request_ids': ids})
        if after_advance is not None:
            after_advance()
        self.runner.requests = {value: NS(prompt_token_ids=[1, 2], output_token_ids=[]) for value in ids}
        self.runner.with_prefill = True
        with torch.inference_mode():
            hidden = self.model(input_ids=torch.tensor([1, 2]*8), positions=torch.arange(16))
            self.model.processor(self.model.head, hidden[-8:])
        self.runner.input_batch = NS(req_ids=ids)
        for value in ids:
            self.runner.requests[value].output_token_ids = [3]
        self.dispatch('advance', {'capture_decode': True})
        self.runner.with_prefill = False
        with torch.inference_mode():
            hidden = self.model(input_ids=torch.tensor([3]*8), positions=torch.tensor([2]*8))
            self.model.processor(self.model.head, hidden)
        if collect:
            self.dispatch('advance', {'collect': True})

    def calibrated(self):
        for _ in range(4):
            self.phase()

    def host_client(self):
        from demoncat.runtime.vllm_ascend.worker_extension import host
        from demoncat.runtime.vllm_ascend.worker_extension.runtime import WorkerFaultRuntime
        from demoncat.runtime.vllm_ascend.worker_extension.identity import WorkerContext
        self.session.close(abort=True)
        self.core = WorkerFaultRuntime(os.getpid())
        self.core._identity = NS(read=lambda worker: WorkerContext(self.fixture.identity, self.model, True))
        self.worker.demoncat_worker_identity = lambda: self.core.invoke(self.worker, 'worker_identity')
        def make(worker, model, *, owner, generation, worker_identity):
            self.session = ordinary.QuantProjectionServingSession(worker, model, root='CPU_ONLY',
                build_sha256='b'*64, owner=owner, generation=generation, worker_identity=worker_identity)
            return self.session
        self.enterContext(patch('fault_injection.providers.resolve_runtime',
                               return_value=ordinary.QuantProjectionServingBinding(make)))
        # Explicit CPU deployment double corresponding to the real installer.
        self.worker._dcat_quant_binding = lambda model: ordinary.QuantProjectionServingBinding(make)
        self.worker._dcat_quant_ordinary_model = self.model
        fixture = self
        class Executor:
            drop_action = None
            def collective_rpc(self, method, *, timeout, args=()):
                names = {'demoncat_worker_identity': 'worker_identity', 'demoncat_capabilities': 'capabilities',
                    'demoncat_attach': 'attach', 'demoncat_detach': 'detach', 'demoncat_fault_query': 'query'}
                action, payload = ((args[0], args[1]) if method == 'demoncat_fault_rpc'
                                   else (names[method], args[0] if args else None))
                result = fixture.core.invoke(fixture.worker, action, payload)
                if action == self.drop_action:
                    self.drop_action = None
                    raise ConnectionError('CPU_LOST_REPLY_AFTER_ACTUAL_ACTION')
                return [result]
        self.executor = Executor()
        client = host.WorkerFaultClient(host.HostAdapter(self.executor), host.FaultConfig('quant', 'batch_shared_scale'))
        def dispatch(action, settings=None):
            if action == 'advance':
                result = (client.advance_batch(settings['request_ids']) if 'request_ids' in settings else
                          client.collect_replay() if 'collect' in settings else client.begin_decode_capture())
            else:
                result = getattr(client, action)()
            self.assertEqual(result['status'], 'QUERY_OK' if action == 'query' else 'ACTION_COMPLETED', result)
            return result['response'].get('provider_state', {})
        self.dispatch = dispatch
        for action in ('prepare', 'register', 'check', 'enable'):
            self.dispatch(action)
        return client

    def test_actual_six_batches_normal_both_effects_off_without_census_or_dump(self):
        for _ in range(6):
            self.phase()
        result = self.dispatch('finish')['serving']
        self.assertEqual(result['status'], 'FUNCTIONAL_EFFECT_AND_OFF_OBSERVED')
        self.assertEqual(self.binding.selected_modes, [1, 2, 4, 3])
        self.assertEqual(result['actual_fault_hits'], 1)
        self.assertGreater(result['fault_effect']['quant']['changed_elements'], 0)
        self.assertGreater(result['fault_effect']['scale']['changed_elements'], 0)
        self.assertLessEqual(len(result['fault_effect']['quant']['samples']), 4)
        self.assertIsNone(result['current_request_effect'])
        self.assertEqual(result['rows'][0]['projection']['hashes'], result['rows'][-1]['projection']['hashes'])
        self.assertEqual(result['rows'][0]['decode']['logits_sha256'], result['rows'][-1]['decode']['logits_sha256'])
        self.assertFalse(result['all_target_inventory_run'] or result['c2'] or result['c3'])
        self.assertEqual(result['main_project_wiring'], 'NOT_AVAILABLE_NOT_ACCEPTED')
        self.assertEqual(result['owned_hooks'] + result['owned_method_patches'] + result['owned_api_patches']
                         + result['owned_row_modes'], 0)
        self.assertFalse(result['native_registered'])
        self.assertEqual(self.binding.calls[-4:], ['stop', 'drain', 'recover', 'remove'])
        self.dispatch('unregister')
        with self.assertRaisesRegex(RuntimeError, 'SINGLE_REVIEWED_ATTEMPT'):
            self.dispatch('enable')

    def test_pending_off_hides_old_current_effect_but_retains_fault_history(self):
        self.host_client()
        self.calibrated();self.phase()
        fault = self.session.report()['fault_effect']
        self.assertIsNotNone(fault)
        self.assertEqual(self.session.report()['current_request_effect'], fault)
        calls = list(self.binding.selected_modes)
        def pending_off():
            report = self.dispatch('query')['serving']
            self.assertEqual(report['next_phase'], 'off')
            self.assertIsNotNone(self.session.expected)
            self.assertIsNone(report['current_request_effect'])
            self.assertEqual(report['fault_effect'], fault)
            self.assertEqual(len(report['rows']), 5)
            self.assertEqual(self.binding.selected_modes, calls)
        self.phase(after_advance=pending_off)
        report = self.dispatch('finish')['serving']
        self.assertTrue(report['recovered'])
        self.assertEqual(report['rows'][-1]['phase'], 'off')
        self.assertIsNone(report['current_request_effect'])
        self.assertEqual(report['fault_effect'], fault)
        self.dispatch('unregister')

    def actual_dispatch(self):
        # Real CPU OpOverload -> real TorchDispatchMode -> typed Native double.
        # The production resolver still names only torch.ops.npu.npu_dynamic_quant.
        self.binding.dispatch_quant = True
        self.enterContext(patch.object(torch.ops.npu, 'npu_dynamic_quant',
            NS(default=torch.ops.cpu_quant_group_c1.quant.default), create=True))

    def test_actual_opoverload_plain_prefix_through_all_public_core_phases(self):
        client = self.host_client()
        self.actual_dispatch()
        snapshots = []
        original_query = self.session.row_observer.query_native_input
        def query(*args, **kwargs):
            graph = self.session.row_observer
            snapshots.append(dict(source_absent=graph.boundaries is None,
                before=len(kwargs['context']['operators']), current=len(graph.call_stack),
                outer_source_frames=len(kwargs['context']['source_operators'])))
            result = original_query(*args, **kwargs)
            self.assertEqual(result['native_input_dispatch']['operator'], 'cpu_quant_group_c1.quant.default')
            return result
        self.enterContext(patch.object(self.session.row_observer, 'query_native_input', query))
        for _ in range(6):
            self.phase()
        result = self.dispatch('finish')['serving']
        self.assertEqual(snapshots, [dict(source_absent=True, before=0, current=1, outer_source_frames=0)]*4)
        self.assertEqual(self.binding.selected_modes, [1, 2, 4, 3])
        self.assertTrue(result['functional_effect_observed'] and result['recovery_proven'])
        self.assertEqual(self.session.row_observer.call_stack, [])
        self.assertEqual(self.session.row_observer.dispatch_frames, [])
        self.dispatch('unregister')
        self.assertEqual(client.detach()['status'], 'DETACHED')
        self.assertFalse(client.snapshot()['ownership_retained'])

    def refuse_actual_dispatch(self, mutation=None, *, unknown=False, smooth=None):
        client = self.host_client()
        self.actual_dispatch()
        self.calibrated()
        self.binding.unknown_dispatch = unknown
        self.binding.dispatch_smooth = smooth
        observed = self.binding.observed
        def changed(call):
            if self.session.current_call is not None and mutation is not None:
                mutation(self.session.row_observer, self.session.current_call)
            return observed(call)
        self.binding.observed = changed
        with self.assertRaisesRegex(RuntimeError, 'NATIVE_ROW_'):
            self.phase()
        self.assertEqual(self.binding.selected_modes, [1, 2, 4])
        self.assertNotIn('fault', self.session.phase_attempts)
        self.assertFalse(any(row['phase'] == 'fault' for row in self.session.receipts))
        self.assertEqual(self.session.row_observer.dispatch_frames, [])
        self.assertEqual(self.session.row_observer.call_stack, [])
        self.assertTrue(client.snapshot()['ownership_retained'])
        self.binding.drain_result = False
        failed = client.recover()
        self.assertTrue(failed['ownership_retained'])
        self.assertTrue(self.session.installed)
        self.binding.drain_result = True
        self.dispatch('recover')
        self.dispatch('unregister')
        self.assertEqual(client.detach()['status'], 'DETACHED')
        self.assertFalse(client.snapshot()['ownership_retained'])
        self.assertFalse(self.session.installed or self.session.row_observer.active)

    def test_actual_dispatch_wrong_operator_never_admits_or_faults(self):
        self.refuse_actual_dispatch(unknown=True)

    def test_actual_dispatch_nondefault_tensor_argument_never_admits_or_faults(self):
        self.refuse_actual_dispatch(smooth=torch.ones(8))

    def test_actual_dispatch_error_after_public_query_still_refuses(self):
        self.refuse_actual_dispatch(lambda graph, context: graph.errors.append('CPU_ACTUAL_PENDING_DISPATCH_ERROR'))

    def test_actual_dispatch_current_frame_input_identity_cannot_be_replaced_by_same_shape(self):
        unrelated = torch.zeros(8, 4, dtype=torch.bfloat16)
        self.refuse_actual_dispatch(lambda graph, context: graph.dispatch_frames[-1].update(args=(unrelated,)))

    def test_actual_dispatch_write_epoch_change_still_refuses(self):
        self.refuse_actual_dispatch(lambda graph, context: graph._invalidate([context['input']]))

    def test_actual_dispatch_event_must_still_be_pending(self):
        self.refuse_actual_dispatch(lambda graph, context: graph.dispatch_frames[-1]['event'].update(status='OBSERVED'))

    def test_query_is_readonly_and_collect_is_explicit(self):
        self.phase(collect=False)
        before = copy.deepcopy(self.session.report())
        for _ in range(3):
            self.provider.query(owner='ordinary-owner', generation=4)
        self.assertEqual(before, self.session.report())
        self.assertEqual(len(self.session.rows), 0)
        self.dispatch('advance', {'collect': True})
        self.assertEqual(len(self.session.rows), 1)

    def test_normal_does_not_install_native_owner_or_qualification_inventory(self):
        self.phase()
        self.assertIsNone(self.binding.config)
        self.assertEqual(self.binding.calls, [])
        self.assertEqual(self.session.phase_name(), 'reference')
        self.assertFalse(hasattr(self.session, 'inventory') or hasattr(self.session, 'collector'))

    def test_roll_before_first_projection_cannot_be_selected_by_shape(self):
        self.model.unknown_first = True
        with self.assertRaisesRegex(RuntimeError, 'COMPLETE_ACTUAL_SINGLE_PROJECTION'):
            self.phase()
        self.assertFalse(self.binding.selected_modes)
        self.assertIsNone(self.session.target)

    def test_existing_target_cannot_switch_to_another_module(self):
        self.phase()
        self.model.first, self.model.second = self.model.second, self.model.first
        with self.assertRaises(RuntimeError):
            self.phase()
        self.assertFalse(self.binding.selected_modes)

    def test_changed_current_input_refuses_before_native_admission(self):
        self.phase()
        old = self.model.norm.forward
        self.model.norm.forward = lambda x: old(x) + 1
        with self.assertRaises(RuntimeError):
            self.phase()
        self.assertFalse(self.binding.selected_modes)

    def test_error_between_public_proof_and_native_admission_cannot_fault(self):
        self.calibrated()
        observed = self.binding.observed
        def invalidated(call):
            if self.session.current_call is not None:
                self.session.row_observer.errors.append('CPU_INVALIDATED_AFTER_PUBLIC_QUERY')
            return observed(call)
        self.binding.observed = invalidated
        with self.assertRaisesRegex(RuntimeError, 'NATIVE_ROW_OBSERVER_NOT_CURRENT_AND_HEALTHY'):
            self.phase()
        self.assertNotIn(3, self.binding.selected_modes)
        self.assertNotIn('fault', self.session.phase_attempts)

    def test_q_only_change_is_not_successful_product_effect(self):
        self.calibrated()
        self.binding.change_scale = False
        with self.assertRaisesRegex(RuntimeError, 'BOTH_ACTUAL_Q_AND_SCALE_MUST_CHANGE'):
            self.phase()
        self.assertIn(3, self.binding.selected_modes)
        first_error = self.session.error
        self.dispatch('recover')
        self.assertTrue(self.session.installed)
        self.assertFalse(self.session.recovered)
        self.assertTrue(self.session.recovery_only)
        self.phase()
        result = self.dispatch('finish')['serving']
        self.assertTrue(result['recovered'])
        self.assertEqual(result['error'], first_error)
        self.assertEqual(result['status'], 'PENDING_OR_REFUSED')
        self.assertEqual(len(result['failed_batches']), 1)
        self.assertEqual(self.binding.selected_modes.count(3), 1)
        self.dispatch('unregister')

    def test_failed_drain_preserves_hooks_and_spent_fault(self):
        self.calibrated()
        self.binding.drain_result = False
        with self.assertRaisesRegex(RuntimeError, 'ACTUAL_NATIVE_DRAIN_PENDING'):
            self.phase()
        self.assertIsNotNone(self.binding.config)
        self.assertTrue(self.session.installed)
        self.assertEqual(self.binding.selected_modes.count(3), 1)
        with self.assertRaisesRegex(RuntimeError, 'SINGLE_REVIEWED_ATTEMPT'):
            self.dispatch('enable')

    def test_unmatched_worker_identity_refuses_without_native_owner(self):
        self.worker.device.index = 1
        with self.assertRaisesRegex(RuntimeError, 'SAME_ACTUAL_TP1'):
            self.provider.query(owner='ordinary-owner', generation=4)
        self.assertIsNone(self.binding.config)
        self.worker.device.index = 0

    def test_actual_host_runtime_full_lifecycle_and_new_task_no_spent_reuse(self):
        client = self.host_client()
        for _ in range(6):
            self.phase()
        self.assertEqual(self.dispatch('finish')['serving']['status'], 'FUNCTIONAL_EFFECT_AND_OFF_OBSERVED')
        self.dispatch('disable')
        self.dispatch('unregister')
        self.assertEqual(client.detach()['status'], 'DETACHED')
        self.assertEqual(self.core._leases, {})
        self.assertEqual(self.session.report()['owned_normal_readback_bytes'], 0)
        old_session = self.session
        next_client = self.host_client()
        self.assertIsNot(self.session, old_session)
        self.assertFalse(self.session.phase_attempts)
        self.assertIsNone(self.session.target)
        self.phase()
        self.dispatch('disable')
        self.dispatch('unregister')
        self.assertEqual(next_client.detach()['status'], 'DETACHED')
        self.assertIn('fault', old_session.phase_attempts)

    def test_actual_host_query_keeps_normal_uncollected(self):
        self.host_client()
        self.phase(collect=False)
        before = self.session.report()
        with patch.object(self.session, 'collect', side_effect=AssertionError('QUERY_IS_NOT_AN_ACTION')):
            state = self.dispatch('query')
        self.assertEqual(state['serving'], before)
        self.assertEqual(self.session.rows, [])
        self.dispatch('advance', {'collect': True})

    def test_actual_host_failed_effect_recover_runs_off_and_retains_first_failure(self):
        client = self.host_client()
        self.calibrated()
        self.binding.change_scale = False
        with self.assertRaisesRegex(RuntimeError, 'BOTH_ACTUAL_Q_AND_SCALE_MUST_CHANGE'):
            self.phase()
        first = self.session.error
        self.assertTrue(client.query()['response']['recovery_required'])
        self.dispatch('recover')
        self.assertTrue(self.session.recovery_only)
        self.assertTrue(client.snapshot()['ownership_retained'])
        refused = client.unregister()
        self.assertNotEqual(refused['status'], 'ACTION_COMPLETED')
        self.dispatch('recover')
        self.phase()
        result = self.dispatch('finish')['serving']
        self.assertEqual(result['error'], first)
        self.assertTrue(result['recovered'])
        self.assertEqual(self.binding.selected_modes.count(3), 1)
        self.dispatch('unregister')
        self.assertEqual(client.detach()['status'], 'DETACHED')

    def test_actual_host_lost_fault_collect_reply_recovers_only_to_off(self):
        client = self.host_client()
        self.calibrated()
        self.phase(collect=False)
        self.executor.drop_action = 'advance'
        result = client.collect_replay()
        self.assertTrue(result['ownership_retained'])
        self.assertEqual(self.binding.selected_modes.count(3), 1)
        self.dispatch('recover')
        self.assertTrue(self.session.recovery_only)
        self.phase()
        self.dispatch('finish')
        self.dispatch('unregister')
        self.assertEqual(client.detach()['status'], 'DETACHED')
        self.assertEqual(self.binding.selected_modes.count(3), 1)

    def test_deployment_is_inert_and_wrong_model_cannot_construct_session(self):
        self.session.close(abort=True)
        calls = list(self.binding.calls)
        registered = []
        with patch.dict(os.environ, {'VLLM_ENABLE_V1_MULTIPROCESSING': '0', 'TASK_QUEUE_ENABLE': '0'}), \
             patch('fault_injection.providers.register_runtime', side_effect=lambda name, factory: registered.append(factory)):
            result = ordinary.install_quant_projection_binding(self.worker, root='CPU_ONLY', build_sha256='b'*64)
        self.assertEqual(result['status'], 'DEPLOYMENT_BOUND_NOT_ACCEPTED')
        self.assertFalse(result['native_owner_installed'])
        self.assertEqual(self.binding.calls, calls)
        with self.assertRaisesRegex(RuntimeError, 'FOREIGN_MODEL'):
            registered[0](object())
        self.assertEqual(len(registered), 1)
        del self.worker._dcat_quant_binding

    def test_actual_non_dynamic_method_refuses_before_native_load_even_with_qwen_name(self):
        self.session.close(abort=True)
        self.runner.vllm_config.model_config.hf_config.model_type = 'qwen3'
        # Recognized model name cannot admit a different quantization method.
        for _, module in self.model.named_modules():
            if hasattr(module, 'quant_method'):
                module.quant_method = NS(apply=lambda *args, **kwargs: None)
        with patch.object(native.QuantGroupNative, 'load', side_effect=AssertionError('MUST_REFUSE_BEFORE_NATIVE_LOAD')):
            with self.assertRaisesRegex(RuntimeError, 'NO_ORDINARY_TP1_W8A8_CANDIDATE'):
                ordinary.QuantProjectionServingSession(self.worker, self.model, root='CPU_ONLY', build_sha256='b'*64,
                    owner='next', generation=5, worker_identity=self.fixture.identity)

    def test_public_native_geometry_mismatch_does_not_backfill_from_public(self):
        original = self.binding.raw_range
        def changed(first, last):
            rows = original(first, last)
            if len(rows) == 5:
                rows[0]['parent']['quant']['columns'] += 1
            return rows
        self.binding.raw_range = changed
        with self.assertRaisesRegex(RuntimeError, 'ACTUAL_GROUP_PUBLIC_NATIVE_FORMALS_DIFFER'):
            self.phase()
        self.assertFalse(self.binding.selected_modes)
        self.assertIsNone(self.binding.config)

    def test_unknown_qmm_consumer_is_refused_without_native_fault(self):
        # Replace the actual owner-checked argument on entry, not just a report.
        from demoncat.runtime.vllm_ascend.worker_extension import serving20_qmm
        bind = serving20_qmm.bind_qmm_arguments
        def wrong(args, kwargs):
            value = bind(args, kwargs)
            value['pertoken_scale'] = value['pertoken_scale'].clone()
            return value
        with patch.object(ordinary, 'bind_qmm_arguments', side_effect=wrong):
            with self.assertRaisesRegex(RuntimeError, 'QMM_MUST_CONSUME_CURRENT_Q_AND_SCALE'):
                self.phase()
        self.assertIsNone(self.binding.config)
        self.assertFalse(self.binding.selected_modes)

    def test_norm_failure_is_preserved_before_collect_and_owner_can_abort(self):
        # The already-installed hook resolves its own real error; patch the
        # source-owner check exercised inside that production hook.
        with patch.object(ordinary, 'function_owner', side_effect=RuntimeError('CPU_SOURCE_OWNER_FAILURE')):
            with self.assertRaisesRegex(RuntimeError, 'CPU_SOURCE_OWNER_FAILURE'):
                self.phase()
        self.assertIn('CPU_SOURCE_OWNER_FAILURE', self.session.error)
        self.assertTrue(self.provider.query(owner='ordinary-owner', generation=4)['replay_failed'])
        self.dispatch('recover')
        self.assertFalse(self.session.installed)
        self.assertTrue(self.session.aborted)

    def test_frozen_b8_scope_rejects_fewer_or_reused_request_ids(self):
        with self.assertRaisesRegex(RuntimeError, 'EIGHT_FRESH_ACTUAL_ENQUEUED_REQUEST_IDS'):
            self.dispatch('advance', {'request_ids': ['request-' + str(i) for i in range(7)]})
        self.phase()
        with self.assertRaisesRegex(RuntimeError, 'EIGHT_FRESH_ACTUAL_ENQUEUED_REQUEST_IDS'):
            self.dispatch('advance', {'request_ids': list(self.session.seen)})
        self.assertEqual(self.binding.calls, [])

    def test_changed_prefill_is_refused_before_native_owner_or_fault(self):
        self.phase()
        self.session.prefill_result = {'logits_sha256': 'f'*64}
        self.session.capture_started = False
        with self.assertRaisesRegex(RuntimeError, 'NORMAL_PREFILL_CHANGED_BEFORE_SELECTED_DECODE'):
            self.session.begin_observation()
        self.assertEqual(self.binding.calls, [])


if __name__ == '__main__':
    unittest.main()
