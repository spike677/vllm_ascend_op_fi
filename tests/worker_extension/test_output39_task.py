"""Real CPU Host/Runtime/hooks/ops; Native and scheduler are explicit doubles.

No acceptance of NPU execution or actual inner main-project wiring is claimed.
"""
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

from demoncat.runtime.vllm_ascend.worker_extension import serving39_output, runtime
from demoncat.runtime.vllm_ascend.worker_extension.providers.output39_task import Output39TaskProvider
from fault_injection.deployments.postwrite import PostwriteRefused
from fault_injection.deployments.output39_execution import Output39SingleExecution
import test_core_output39 as fixture
from test_postwrite_providers import Projection
from output39_custom_fixture import CustomProjection


class SuperCustomProjection(CustomProjection):
    def forward(self, x):
        return super().forward(x)


class AlternateSuperParent(CustomProjection):
    pass


class WrongSelfSuperProjection(CustomProjection):
    def forward(self, x):
        if hasattr(self, 'guard_target'):
            from fault_injection.deployments.output39_execution import _ClassCellProjectionGuard
            return _ClassCellProjectionGuard(self.guard_target, 'projection')
        return super().forward(x)


def foreign_forward(original):
    def forward(self, x):
        return original(self, x)
    return forward


class EarlyUnconsumedModel(fixture.ActualProjectionModel):
    def __init__(self):
        super().__init__()
        self.early = Projection(consumer=False)

    def forward(self, input_ids, positions, inputs_embeds=None, intermediate_tensors=None):
        self.early(torch.ones(input_ids.numel(), 4, dtype=torch.bfloat16))
        return super().forward(input_ids, positions, inputs_embeds, intermediate_tensors)


class ReadbackTransport(fixture.OwnedCpuTransport):
    def snapshot(self, tensor):
        self.validate(tensor)
        return tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()


class Output39TaskTests(unittest.TestCase):
    snapshot = fixture.CoreOutput39Tests.snapshot
    root = fixture.CoreOutput39Tests.root
    forward = fixture.CoreOutput39Tests.forward
    dispose = fixture.CoreOutput39Tests.dispose
    ok = fixture.CoreOutput39Tests.ok
    enqueue = fixture.CoreOutput39Tests.enqueue

    def setUp(self):
        fixture.CoreOutput39Tests.setUp(self)
        self.transport = ReadbackTransport()
        def bind(actual, config):
            serving39_output.install_fault39_output_serving_binding(self.worker,
                transport=self.transport, prompt_token_ids=[1, 2, 3, 4], runtime_only=True,
                _simulation=True, _native_readback=lambda t: dict(abi=1, sha256=t.sha256))
            return {}
        self.client.host._binding = bind
        # Exercise the real public registry route, with no provider resolver
        # replacement. Only the Native transport and scheduler are CPU doubles.

    def start(self):
        before = self.model.forward_calls
        self.ok(self.client.prepare())
        self.provider = self.worker._demoncat_worker_fault_runtime._leases['39.output_proxy'].provider
        for action in ('register', 'check', 'enable'): self.ok(getattr(self.client, action)())
        self.assertEqual(self.model.forward_calls, before)

    def finish_task(self, first=0):
        for i in range(first, 4): self.enqueue(i)
        state = self.ok(self.client.finish())
        self.ok(self.client.unregister())
        self.assertEqual(self.client.detach()['status'], 'DETACHED')
        return state

    def test_normal_fault_off_health_single_target_no_census_or_files(self):
        from fault_injection.deployments.postwrite_inventory import ProjectionWorkloadInventory
        with patch.object(ProjectionWorkloadInventory, '__init__', side_effect=AssertionError('full census')):
            self.start()
            self.assertIsNone(self.provider.session.folder)
            self.enqueue(0)
            self.assertFalse(self.provider.p._qualified)  # No old qualification minted.
            self.assertEqual(self.provider.p.normal['path'], 'projection')
            state = self.finish_task(1)
        self.assertEqual(self.model.forward_calls, 4)
        self.assertEqual(len(self.transport.opened), 2)  # Native lossless + one F.
        self.assertTrue(all(lease.closed for lease in self.transport.opened))
        rows = state['evidence']['actual_requests']
        self.assertEqual([r['phase'] for r in rows], ['normal', 'fault', 'off', 'health'])
        effect = rows[1]['report']['effect']
        self.assertEqual((effect['changed_elements'], effect['finite_to_positive_inf']), (1, 1))
        self.assertTrue(effect['changed_only_selected'])
        self.assertTrue(rows[1]['report']['consumers'][0]['original_returned'])
        self.assertTrue(rows[2]['report']['exact_normal_output'])
        self.assertTrue(state['evidence']['matched_off_and_health_confirmed'])
        self.assertFalse(any(state['evidence'][key] for key in ('c1', 'c2', 'c3', 'E5')))
        self.assertEqual((state['owned_hooks'], state['leases'], state['inflight']), (0, 0, 0))
        self.assertFalse(self.evidence_root.exists())

    def test_normal_discovery_ignores_unconsumed_pending_candidate_then_never_switches(self):
        self.model = EarlyUnconsumedModel().eval()
        self.runner.model = self.model
        self.start()
        self.enqueue(0)
        selected = self.provider.p.normal['key']
        self.assertEqual(self.provider.p.normal['path'], 'projection')
        self.assertEqual(len(self.transport.opened), 1)
        state = self.finish_task(1)
        self.assertEqual(self.provider.p.normal['key'], selected)
        self.assertEqual(len(self.transport.opened), 2)
        self.assertIsNot(self.transport.opened[0].tensor, self.transport.opened[1].tensor)
        self.assertEqual(state['evidence']['last_fault_effect']['changed_elements'], 1)

    def _custom_protocol(self, consumer, projection_type=None):
        from output39_custom_fixture import CustomOperators, CustomProjection
        ops = self.enterContext(CustomOperators())
        self.enterContext(patch.dict(sys.modules, {'torch_npu': ops.npu}))
        projection = (projection_type or CustomProjection)(consumer)
        if consumer == 'swiglu':
            projection.weight = torch.nn.Parameter(torch.full((8, 4), .25, dtype=torch.bfloat16), requires_grad=False)
        self.model.projection = self.model.projection_alias = projection
        self.start()
        state = self.finish_task()
        self.assertEqual(ops.calls, ['gemm', consumer] * 4)
        rows = state['evidence']['actual_requests']
        self.assertEqual(rows[1]['report']['target']['operator'], 'vllm.unquantized_gemm.default')
        self.assertIsNotNone(rows[1]['report']['consumers'][0]['reviewed_consumer'])
        self.assertTrue(state['evidence']['matched_off_and_health_confirmed'])
        return state

    def test_source_bound_custom_gemm_and_rms_registration_cpu_contract(self):
        self._custom_protocol('rms')

    def test_source_bound_custom_gemm_and_swiglu_registration_cpu_contract(self):
        self._custom_protocol('swiglu')

    def test_actual_super_forward_chain_custom_gemm_no_full_census(self):
        from fault_injection.deployments.postwrite_inventory import ProjectionWorkloadInventory
        with patch.object(ProjectionWorkloadInventory, '__init__', side_effect=AssertionError('full census')):
            state = self._custom_protocol('rms', SuperCustomProjection)
        binding = state['evidence']['actual_requests'][0]['report']['target']['callable_binding']
        self.assertEqual(binding['policy'], 'SOURCE_BOUND_CLASS_CELL_AND_ACTUAL_SUPER_FRAMES_ONLY')
        self.assertEqual(len(binding['methods']), 2)
        self.assertFalse(binding['full_census_constructed'] or binding['profiler_installed'])

    def _start_super_normal(self):
        from output39_custom_fixture import CustomOperators
        self.enterContext(CustomOperators())
        self.model.projection = self.model.projection_alias = SuperCustomProjection('rms')
        self.start()
        self.enqueue(0)
        return self.provider.p.normal['callable']

    def test_same_class_mro_change_refused_with_original_target_retained(self):
        guard = self._start_super_normal()
        before = SuperCustomProjection.__bases__
        try:
            SuperCustomProjection.__bases__ = (AlternateSuperParent,)
            with self.assertRaisesRegex(PostwriteRefused, 'CLASS_MRO_CHANGED'): guard.verify()
        finally:
            SuperCustomProjection.__bases__ = before
        self.finish_task(1)

    def test_parent_forward_replacement_refused_even_when_child_function_unchanged(self):
        guard = self._start_super_normal()
        with patch.object(CustomProjection, 'forward', foreign_forward(CustomProjection.forward)):
            with self.assertRaisesRegex(PostwriteRefused, 'SUPER_FORWARD_OR_DEFAULTS_CHANGED'): guard.verify()
        self.finish_task(1)

    def test_foreign_forward_closure_is_not_a_class_cell_exception(self):
        from output39_custom_fixture import CustomOperators
        self.enterContext(CustomOperators())
        projection = CustomProjection('rms')
        projection.forward = types.MethodType(foreign_forward(CustomProjection.forward), projection)
        self.model.projection = self.model.projection_alias = projection
        self.start()
        with self.assertRaisesRegex(PostwriteRefused, 'CLASS_CELL_FORWARD_REQUIRED'): self.enqueue(0)
        self.assertFalse(self.provider.p._fault_attempted)
        self.assertEqual(self.transport.opened, [])
        self.ok(self.client.recover()); self.ok(self.client.unregister())

    def test_real_forward_frame_wrong_self_does_not_prove_selected_module(self):
        actual, foreign = WrongSelfSuperProjection(), WrongSelfSuperProjection()
        actual.guard_target = foreign
        with self.assertRaisesRegex(PostwriteRefused, 'FRAME_ACTUAL_OWNER_CHANGED'):
            actual.forward(torch.ones(1, 4, dtype=torch.bfloat16))

    def test_class_cell_must_be_exact_declaring_class(self):
        from fault_injection.deployments.output39_execution import _ClassCellProjectionGuard
        actual = SuperCustomProjection()
        original = SuperCustomProjection.forward
        # A real cell with the wrong class, not a string/name approximation.
        def closure(value):
            return (lambda: value).__closure__
        forged = types.FunctionType(original.__code__, original.__globals__,
            original.__name__, original.__defaults__, closure(AlternateSuperParent))
        forged.__qualname__ = original.__qualname__
        with patch.object(SuperCustomProjection, 'forward', forged):
            with self.assertRaisesRegex(PostwriteRefused, 'UNREVIEWED_FORWARD_CLOSURE'):
                _ClassCellProjectionGuard(actual, 'projection')

    def test_class_cell_loaded_code_must_match_source_bytes(self):
        from fault_injection.deployments.output39_execution import _ClassCellProjectionGuard
        original = SuperCustomProjection.forward
        previous = original.__code__
        try:
            original.__code__ = previous.replace(co_firstlineno=previous.co_firstlineno + 1)
            with self.assertRaisesRegex(PostwriteRefused, 'FRAME_CODE_SOURCE_CHANGED'):
                _ClassCellProjectionGuard(SuperCustomProjection(), 'projection')
        finally:
            original.__code__ = previous

    def test_query_never_drives_forward_and_spent_provider_cannot_reenable(self):
        self.start()
        before = self.model.forward_calls
        self.assertEqual(self.client.query()['status'], 'QUERY_OK')
        self.assertEqual(self.model.forward_calls, before)
        for i in range(4): self.enqueue(i)
        self.ok(self.client.finish())
        reply = self.client.enable()
        self.assertNotEqual(reply['status'], 'ACTION_COMPLETED')
        self.assertTrue(self.provider.attempted and self.provider.p._fault_attempted)
        self.ok(self.client.unregister())
        self.assertEqual(self.client.detach()['status'], 'DETACHED')

    def test_nonfinite_normal_is_refused_by_unchanged_native_guard(self):
        with torch.no_grad(): self.model.projection.weight.fill_(float('inf'))
        self.start()
        with self.assertRaisesRegex(PostwriteRefused, 'NONFINITE_ORIGINAL'):
            self.enqueue(0)
        self.assertEqual(len(self.transport.opened), 1)
        self.assertFalse(self.provider.p._fault_attempted)
        self.ok(self.client.recover())
        self.ok(self.client.unregister())

    def test_changed_actual_input_refuses_before_f_and_preserves_normal_target(self):
        self.start()
        self.enqueue(0)
        target = self.provider.p.normal['key']
        self.provider.session.prompt[0] = 7
        with self.assertRaisesRegex(PostwriteRefused, 'NORMAL_INPUT_BINDING'):
            self.enqueue(1)
        self.assertEqual(target, self.provider.p.normal['key'])
        self.assertEqual(len(self.transport.opened), 1)
        self.assertFalse(self.provider.p._fault_attempted)
        self.provider.session.prompt[0] = 1
        self.ok(self.client.recover())
        self.ok(self.client.unregister())

    def test_changed_weight_identity_refuses_without_switching_target(self):
        self.start()
        self.enqueue(0)
        original = self.model.projection.weight
        self.model.projection.weight = torch.nn.Parameter(original.clone(), requires_grad=False)
        try:
            with self.assertRaisesRegex(PostwriteRefused, 'MODEL_OR_STORAGE_CHANGED'):
                self.enqueue(1)
            self.assertEqual(len(self.transport.opened), 1)
        finally:
            self.model.projection.weight = original
        self.ok(self.client.recover())
        self.ok(self.client.unregister())

    def test_restore_failure_retains_native_and_runtime_owner_then_off_health(self):
        self.start()
        self.enqueue(0)
        self.transport.restore_failure = True
        with self.assertRaisesRegex(PostwriteRefused, 'CPU_RESTORE_FAILURE'):
            self.enqueue(1)
        self.assertEqual(self.provider.p.live_lease_count, 1)
        self.assertTrue(self.client.snapshot()['ownership_retained'])
        self.assertNotEqual(self.client.unregister()['status'], 'ACTION_COMPLETED')
        self.transport.restore_failure = False
        self.ok(self.client.recover())
        self.assertEqual(self.provider.phase, 'off')
        self.enqueue(2)
        self.enqueue(3)
        self.ok(self.client.finish())
        self.ok(self.client.unregister())
        self.assertEqual(self.client.detach()['status'], 'DETACHED')

    def test_absent_value_consumer_is_not_a_success_or_another_target(self):
        self.model.projection.consumer = False
        # The real logits computation consumes the final row only, while the
        # selected prefix is in the first row. Same storage is insufficient.
        self.start()
        with self.assertRaisesRegex(PostwriteRefused, 'VALUE_CONSUMER_NOT_OBSERVED'):
            self.enqueue(0)
        self.assertIsNone(self.provider.p.normal)
        self.ok(self.client.recover())
        self.ok(self.client.unregister())

    def test_host_side_tensor_read_between_root_and_logits_is_not_a_model_consumer(self):
        self.model.projection.consumer = False
        self.start()
        self.actual_request_id = 'host-side-read'
        self.ok(self.client.advance(self.actual_request_id))
        with torch.inference_mode():
            hidden = self.root(self.provider.session)
            discarded = hidden + 1
            self.assertTrue(torch.isfinite(discarded).all())
            with self.assertRaisesRegex(PostwriteRefused, 'VALUE_CONSUMER_NOT_OBSERVED'):
                self.model.processor(self.model.head, hidden[-1:].contiguous())
        self.assertIsNone(self.provider.p.normal)
        self.ok(self.client.recover())
        self.ok(self.client.unregister())

    def test_legacy_replay_entrypoints_are_not_available_on_runtime_execution(self):
        execution = Output39SingleExecution(self.model, self.transport, guard=lambda: None)
        for method in ('qualify', 'run'):
            with self.assertRaisesRegex(PostwriteRefused, 'ACTUAL_REQUEST_BOUNDARY_ONLY'):
                getattr(execution, method)()

    def test_foreign_dispatch_is_not_popped_and_owner_survives_until_retry(self):
        from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode_stack
        class Foreign(TorchDispatchMode):
            def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                return function(*args, **(kwargs or {}))
        self.start()
        self.actual_request_id = 'foreign-window'
        self.ok(self.client.advance(self.actual_request_id))
        with torch.inference_mode():
            hidden = self.root(self.provider.session)
            window = self.provider.session.window
            with Foreign() as foreign:
                with self.assertRaisesRegex(RuntimeError, 'FOREIGN_DISPATCH_MODE_ON_TOP'):
                    self.model.processor(self.model.head, hidden[-1:].contiguous())
                self.assertNotEqual(self.client.unregister()['status'], 'ACTION_COMPLETED')
                self.assertIs(_get_current_dispatch_mode_stack()[-1], foreign)
                self.assertTrue(window.active and self.client.snapshot()['ownership_retained'])
            self.ok(self.client.unregister())
        self.assertFalse(window.active)

    def test_mode_exit_failure_before_and_after_pop_remains_recoverable(self):
        for after in (False, True):
            with self.subTest(after=after):
                case = Output39TaskTests()
                case.setUp()
                try:
                    case.start()
                    case.actual_request_id = 'exit-failure'
                    case.ok(case.client.advance(case.actual_request_id))
                    with torch.inference_mode():
                        hidden = case.root(case.provider.session)
                        window = case.provider.session.window
                        original = window.__exit__
                        def broken(*args):
                            if after: original(*args)
                            raise RuntimeError('CPU_EXIT_FAILURE')
                        with patch.object(window, '__exit__', broken):
                            with case.assertRaisesRegex(RuntimeError, 'CPU_EXIT_FAILURE'):
                                case.model.processor(case.model.head, hidden[-1:].contiguous())
                        case.assertEqual(window.active, not after)
                    case.ok(case.client.unregister())
                    case.assertFalse(window.active)
                finally:
                    case.doCleanups()

    def test_new_task_uses_fresh_owner_provider_without_resetting_spent_flags(self):
        self.start()
        old, old_ticket = self.provider, self.provider._ticket
        self.finish_task()
        serving39_output.remove_fault39_output_serving_binding(self.worker)
        from demoncat.runtime.vllm_ascend.worker_extension.host import WorkerFaultClient
        self.client = WorkerFaultClient(self.client.host, self.client.config)
        self.start()
        self.assertIsNot(self.provider, old)
        self.assertNotEqual(self.provider._ticket, old_ticket)
        self.assertTrue(old.attempted and old.p._fault_attempted)
        self.assertFalse(self.provider.p._fault_attempted)
        self.finish_task()


if __name__ == '__main__':
    unittest.main()
