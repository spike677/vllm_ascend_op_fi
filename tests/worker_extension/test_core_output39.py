"""Real Host/Runtime and Torch dispatch; explicit CPU Native/scheduler doubles."""
from copy import deepcopy
from pathlib import Path
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from demoncat.runtime.vllm_ascend.worker_extension import identity, serving39_output
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
from fault_injection.deployments.postwrite import _storage, PostwriteRefused
from test_extension import _extended_worker
from test_core_host import ExistingExecutor
from test_postwrite_providers import CpuTransport, CpuLease, Projection
import test_serving_session_lifecycle as life


class OwnedCpuLease(CpuLease):
    def __init__(self, transport, tensor, owner, generation):
        super().__init__(tensor)
        self.transport, self.owner, self.generation = transport, owner, generation

    def _check(self):
        if _storage(self.tensor) != self.storage:
            raise PostwriteRefused('CPU_CURRENT_STORAGE_CHANGED')

    def recover(self):
        self._check()
        self.transport.restores += 1
        if self.transport.restore_failure:
            raise PostwriteRefused('CPU_RESTORE_FAILURE')
        super().recover()


class OwnedCpuTransport(CpuTransport):
    sha256 = 'a'*64
    def __init__(self):
        self.opened = []
        self.restores = 0
        self.restore_failure = False

    def open(self, tensor, owner, generation):
        self.validate(tensor)
        lease = OwnedCpuLease(self, tensor, owner, generation)
        self.opened.append(lease)
        return lease


class ActualProjectionModel(life.CpuModel):
    def __init__(self):
        super().__init__()
        self.projection = Projection()
        self.projection_alias = self.projection
        self.forward_calls = 0
        with torch.no_grad():
            self.head.weight.fill_(.25)

    def forward(self, input_ids, positions, inputs_embeds=None, intermediate_tensors=None):
        self.forward_calls += 1
        hidden = super().forward(input_ids, positions, inputs_embeds, intermediate_tensors)
        return self.projection(hidden)


class CoreOutput39Tests(unittest.TestCase):
    snapshot = life.ServingSessionLifecycleTests.snapshot
    root = life.ServingSessionLifecycleTests.root
    forward = life.ServingSessionLifecycleTests.forward

    def setUp(self):
        life.ServingSessionLifecycleTests.setUp(self)
        self.worker = _extended_worker(rank=0)
        self.worker.load_model()
        self.runner = self.worker.model_runner
        self.model = ActualProjectionModel().eval()
        self.runner.model = self.model
        self.runner.vllm_config = NS(model_config=NS(hf_config=NS(vocab_size=3)))
        self.transport = OwnedCpuTransport()
        self.enterContext(patch.object(identity, '_current_device', return_value=0))
        self.enterContext(patch.object(serving39_output, 'require_serial_eager'))
        self.enterContext(patch.object(serving39_output, 'Path', side_effect=lambda value:
            self.evidence_root if str(value) == '/evidence' else Path(value)))
        self.executor = ExistingExecutor([self.worker])
        def bind(actual, config):
            self.assertEqual(actual, self.worker.demoncat_worker_identity()['worker'])
            self.assertEqual(config.as_dict(), dict(fault_id=39, variant='output_proxy', implementation_version='1'))
            serving39_output.install_fault39_output_serving_binding(self.worker, transport=self.transport,
                prompt_token_ids=[1,2,3,4], evidence_dir=self.evidence_root/'output39', _simulation=True,
                _native_readback=lambda transport: dict(abi=1, sha256=transport.sha256))
            return {}
        self.client = WorkerFaultClient(HostAdapter(self.executor, prepare_binding=bind), FaultConfig(39, 'output_proxy'))
        self.provider = None
        self.addCleanup(self.dispose)

    def dispose(self):
        if self.provider and self.provider.session and (self.provider.session.handles or self.provider.session.window):
            self.transport.restore_failure = False
            self.provider.session.close()

    def ok(self, reply):
        self.assertEqual(reply['status'], 'ACTION_COMPLETED', repr(reply.get('response'))[:1400])
        return reply['response']['provider_state']

    def start(self):
        self.ok(self.client.prepare())
        self.provider = self.worker._demoncat_worker_fault_runtime._leases['39.output_proxy'].provider
        for action in ('register', 'check', 'enable'):
            self.ok(getattr(self.client, action)())
        self.assertEqual(self.model.forward_calls, 0)

    def enqueue(self, index):
        self.actual_request_id = 'actual-output39-'+str(index)
        calls = self.model.forward_calls
        self.ok(self.client.advance(self.actual_request_id))
        self.assertEqual(self.model.forward_calls, calls)
        with torch.inference_mode():
            return self.forward(self.provider.session)

    def test_six_real_requests_use_original_native_c1_then_off_health_and_public_cleanup(self):
        self.start()
        with patch.object(self.worker, 'demoncat_worker_identity', side_effect=AssertionError('recursive identity')):
            for i in range(6):
                self.enqueue(i)
            state = self.ok(self.client.finish())
        self.assertEqual(self.model.forward_calls, 6)
        self.assertEqual(state['evidence']['selection']['operator'], 'aten.mm.default')
        self.assertEqual([row['phase'] for row in state['evidence']['actual_requests']],
                         ['observe','observe_repeat','baseline','fault','off','health'])
        self.assertEqual(len(self.transport.opened), 2)
        self.assertTrue(all(lease.closed for lease in self.transport.opened))
        self.assertEqual(state['evidence']['actual_requests'][3]['report']['new_fault_hits'], 1)
        self.assertTrue(state['evidence']['actual_requests'][4]['report']['exact_baseline_output'])
        self.assertFalse(state['evidence']['c2'] or state['evidence']['c3']
                         or state['evidence']['model_local_qualification_inherited'])
        self.assertEqual((state['owned_hooks'], state['leases'], state['inflight']), (0,0,0))
        self.ok(self.client.unregister())
        self.assertEqual(self.client.detach()['status'], 'DETACHED')
        serving39_output.remove_fault39_output_serving_binding(self.worker)
        with torch.inference_mode():
            output = self.forward(self.provider.session)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(self.model.forward_calls, 7)

    def test_query_and_advance_without_forward_do_not_complete_or_write(self):
        self.start()
        self.ok(self.client.advance('not-run'))
        before = (self.model.forward_calls, len(self.transport.opened), self.transport.restores)
        reply = self.client.query()
        self.assertEqual(reply['status'], 'QUERY_OK', reply)
        self.assertEqual(reply['response']['provider_state']['evidence']['actual_requests'], [])
        self.assertEqual(before, (self.model.forward_calls, len(self.transport.opened), self.transport.restores))
        self.assertEqual(self.client.finish()['status'], 'PROVIDER_ACTION_FAILED')
        self.ok(self.client.unregister())

    def test_missing_deployment_is_refused_without_model_or_native(self):
        client = WorkerFaultClient(HostAdapter(self.executor), FaultConfig(39, 'output_proxy'))
        reply = client.prepare()
        self.assertEqual(reply['status'], 'PROVIDER_ACTION_FAILED', reply)
        self.assertEqual(self.model.forward_calls, 0)
        self.assertFalse(self.transport.opened)

    def test_foreign_owner_generation_and_unknown_user_layer_are_refused(self):
        self.start()
        ticket = self.client.snapshot()['ticket']
        for changed, reason in ((dict(owner='foreign'), 'OWNER_MISMATCH'),
                (dict(generation=ticket['generation']+1), 'STALE_GENERATION')):
            reply = self.worker.demoncat_fault_rpc('advance', dict(ticket, **changed, settings={'request_id':'bad'}))
            self.assertEqual(reply['reason'], reason)
        reply = self.worker.demoncat_fault_rpc('advance', dict(ticket,
            settings={'request_id':'bad', 'layer_name':'projection', 'address':123}))
        self.assertEqual(reply['status'], 'PROVIDER_ACTION_FAILED', reply)
        self.assertEqual(self.model.forward_calls, 0)
        self.ok(self.client.unregister())

    def test_after_apply_cleanup_failure_retains_owner_and_requires_real_off(self):
        self.start()
        for i in range(3):
            self.enqueue(i)
        self.transport.restore_failure = True
        with self.assertRaisesRegex(RuntimeError, 'CPU_RESTORE_FAILURE'):
            self.enqueue(3)
        self.assertTrue(self.provider.p.live_lease_count)
        before = self.transport.restores
        self.client.query()
        self.assertEqual(self.transport.restores, before)
        failed = self.client.unregister()
        self.assertEqual(failed['status'], 'PROVIDER_ACTION_FAILED', failed)
        self.assertTrue(failed['ownership_retained'])
        self.transport.restore_failure = False
        self.ok(self.client.recover())
        self.assertFalse(self.provider.p.live_lease_count)
        self.assertEqual(self.client.unregister()['status'], 'PROVIDER_ACTION_FAILED')
        self.ok(self.client.recover())
        self.enqueue('real-recovery-off')
        self.enqueue('real-recovery-health')
        self.ok(self.client.unregister())
        self.assertTrue(self.provider.recovered and self.provider.failures)
        self.assertFalse(self.provider.complete)

    def test_current_model_change_refuses_before_another_native_open(self):
        self.start()
        self.enqueue(0); self.enqueue(1)
        with torch.no_grad():
            self.model.projection.weight.add_(1)
        with self.assertRaisesRegex(RuntimeError, 'MODEL_IDENTITY_NOT_QUALIFIED'):
            self.enqueue(2)
        self.assertFalse(self.transport.opened)
        self.ok(self.client.unregister())

    def test_incomplete_native_effect_receipt_refuses_success_then_requires_real_off_health(self):
        self.start()
        for index in range(3):
            self.enqueue(index)
        original = OwnedCpuLease.apply
        def incomplete(lease, mode, indices):
            original(lease, mode, indices)
            return 0
        with patch.object(OwnedCpuLease, 'apply', incomplete):
            with self.assertRaisesRegex(RuntimeError, 'ACTUAL_NATIVE_EFFECT_UNPROVEN'):
                self.enqueue(3)
        self.assertFalse(self.provider.p.live_lease_count)
        self.assertTrue(self.provider.failures)
        self.assertEqual(self.client.unregister()['status'], 'PROVIDER_ACTION_FAILED')
        self.ok(self.client.recover())
        self.enqueue('off-after-incomplete-receipt')
        self.enqueue('health-after-incomplete-receipt')
        self.ok(self.client.unregister())
        self.assertFalse(self.provider.evidence()['c1_evidence_complete'])

    def test_unobserved_or_changed_scheduler_id_refuses_before_root_execution(self):
        self.start()
        self.ok(self.client.advance('different-enqueue'))
        before = self.model.forward_calls
        with torch.inference_mode(), self.assertRaisesRegex(RuntimeError, 'SCHEDULER_ID_MISMATCH'):
            self.forward(self.provider.session)
        self.assertEqual(self.model.forward_calls, before)
        self.assertFalse(self.transport.opened)
        self.ok(self.client.unregister())

    def test_saved_hook_after_real_request_cannot_reenter_or_write(self):
        self.start()
        self.actual_request_id = 'real-root'
        self.ok(self.client.advance(self.actual_request_id))
        with torch.inference_mode():
            hidden = self.root(self.provider.session)
            window = self.provider.session.window
            old = next(callback for callback in window.callbacks.values())
            self.model.processor(self.model.head, hidden[-1:].contiguous())
            with self.assertRaisesRegex(RuntimeError, 'LATE_OBSERVER_CALLBACK'):
                old(self.model.projection, ())
        self.assertFalse(self.transport.opened)
        self.ok(self.client.unregister())

    def test_query_does_not_restore_live_lease_between_root_and_logits(self):
        self.start()
        self.enqueue(0); self.enqueue(1)
        self.actual_request_id = 'lossless-between-boundaries'
        self.ok(self.client.advance(self.actual_request_id))
        with torch.inference_mode():
            hidden = self.root(self.provider.session)
            before = self.transport.restores
            self.assertEqual(self.provider.p.live_lease_count, 1)
            response = self.client.query()
            self.assertEqual(response['status'], 'QUERY_OK', response)
            self.assertEqual(self.transport.restores, before)
            self.assertEqual(response['response']['provider_state']['leases'], 1)
            self.model.processor(self.model.head, hidden[-1:].contiguous())
        self.ok(self.client.unregister())

    def test_later_logits_return_hook_is_refused_before_target_mutation(self):
        self.start()
        self.enqueue(0); self.enqueue(1)
        handle = self.model.processor.register_forward_hook(lambda module, args, output: output+1)
        try:
            with self.assertRaisesRegex(RuntimeError, 'FINAL_RETURN_OBSERVER_REQUIRED'):
                self.enqueue(2)
        finally:
            handle.remove()
        self.ok(self.client.unregister())

    def test_hook_remove_after_actual_removal_failure_retains_owner_then_retry(self):
        self.start()
        handle = self.provider.session.handles[-1]
        original = handle.remove
        def after_remove():
            original()
            raise RuntimeError('CPU_AFTER_ACTUAL_REMOVE')
        with patch.object(handle, 'remove', after_remove):
            result = self.client.unregister()
        self.assertEqual(result['status'], 'PROVIDER_ACTION_FAILED', result)
        self.assertTrue(result['ownership_retained'])
        self.assertNotIn(handle, self.provider.session.handles)
        self.ok(self.client.unregister())

    def test_hook_added_during_actual_forward_refuses_and_preserves_unwind_for_cleanup(self):
        for boundary in ('root', 'logits'):
            with self.subTest(boundary=boundary):
                case = CoreOutput39Tests()
                case.setUp()
                extra = []
                trigger = None
                try:
                    case.start()
                    case.enqueue(0); case.enqueue(1)
                    target = case.model if boundary == 'root' else case.model.processor
                    def add_return_hook(module, args, output):
                        extra.append(target.register_forward_hook(lambda m, a, result: result))
                    if boundary == 'root':
                        trigger = case.model.projection.register_forward_hook(add_return_hook)
                    else:
                        original = torch.nn.functional.linear
                        def after_actual_linear(*args, **kwargs):
                            result = original(*args, **kwargs)
                            add_return_hook(None, None, result)
                            return result
                        case.enterContext(patch.object(torch.nn.functional, 'linear', after_actual_linear))
                    with case.assertRaisesRegex(RuntimeError, 'FINAL_RETURN_OBSERVER_REQUIRED'):
                        case.enqueue(2)
                    case.assertFalse(case.provider.session.root_running or case.provider.session.logits_running)
                    case.assertTrue(case.provider.failures)
                    case.assertEqual(len(case.transport.opened), 1)
                    case.assertTrue(case.client.snapshot()['ownership_retained'])
                    if trigger is not None:
                        trigger.remove()
                    trigger = None
                    for handle in extra:
                        handle.remove()
                    extra.clear()
                    case.ok(case.client.unregister())
                    case.assertTrue(case.transport.opened[0].closed)
                finally:
                    if trigger is not None:
                        trigger.remove()
                    for handle in extra:
                        handle.remove()
                    case.doCleanups()

    def test_dispatch_before_and_after_pop_failure_remains_explicitly_recoverable(self):
        for after in (False, True):
            with self.subTest(after=after):
                case = CoreOutput39Tests()
                case.setUp()
                try:
                    case.start()
                    case.actual_request_id = 'mode-failure'
                    case.ok(case.client.advance(case.actual_request_id))
                    with torch.inference_mode():
                        hidden = case.root(case.provider.session)
                        window = case.provider.session.window
                        original = window.observer.__exit__
                        def broken(*args):
                            if after:
                                original(*args)
                            raise RuntimeError('CPU_MODE_EXIT_FAILURE')
                        with patch.object(window.observer, '__exit__', broken):
                            with case.assertRaisesRegex(RuntimeError, 'CPU_MODE_EXIT_FAILURE'):
                                case.model.processor(case.model.head, hidden[-1:].contiguous())
                        case.assertEqual(window.mode_active, not after)
                    case.ok(case.client.unregister())
                    case.assertFalse(window.mode_active or window.handles or window.locked)
                finally:
                    case.doCleanups()

    def test_unconsumed_selected_elements_remain_in_inventory_and_refuse_no_fallback(self):
        self.model.projection.consumer = False
        self.start()
        self.enqueue(0)
        with self.assertRaisesRegex(RuntimeError, 'NO_ELIGIBLE_WRITER_WITH_OBSERVED_VALUE_CONSUMER'):
            self.enqueue(1)
        self.assertFalse(self.transport.opened)
        observed = self.provider.p._last_observer.inventory.report()
        self.assertEqual(len(observed['calls']), 2)
        self.assertTrue(all(not row['eligible'] for row in observed['calls']))
        self.ok(self.client.unregister())

    def test_final_current_root_args_after_existing_pre_hook_are_verified(self):
        def change(module, args, kwargs):
            return args, dict(kwargs, positions=kwargs['positions']+1)
        handle = self.model.register_forward_pre_hook(change, with_kwargs=True)
        self.addCleanup(handle.remove)
        self.start()
        with self.assertRaisesRegex(RuntimeError, 'ACTUAL_DEVICE_INPUT_BYTES_NOT_FROZEN_PROMPT'):
            self.enqueue(0)
        self.assertEqual(self.model.forward_calls, 0)
        self.assertFalse(self.transport.opened)
        self.ok(self.client.unregister())

    def test_foreign_dispatch_top_is_preserved_until_explicit_retry(self):
        from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode_stack
        class Foreign(TorchDispatchMode):
            def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                return function(*args, **(kwargs or {}))
        self.start()
        self.actual_request_id = 'foreign-mode'
        self.ok(self.client.advance(self.actual_request_id))
        with torch.inference_mode():
            hidden = self.root(self.provider.session)
            window = self.provider.session.window
            with Foreign() as foreign:
                with self.assertRaisesRegex(RuntimeError, 'FOREIGN_DISPATCH_MODE_ON_TOP'):
                    self.model.processor(self.model.head, hidden[-1:].contiguous())
                failed = self.client.unregister()
                self.assertEqual(failed['status'], 'PROVIDER_ACTION_FAILED', failed)
                self.assertIs(_get_current_dispatch_mode_stack()[-1], foreign)
                self.assertTrue(window.mode_active and failed['ownership_retained'])
            self.ok(self.client.unregister())
        self.assertFalse(window.mode_active or window.handles)

    def test_current_storage_change_query_refuses_without_restore_write(self):
        self.start()
        self.enqueue(0); self.enqueue(1)
        self.actual_request_id = 'changed-storage'
        self.ok(self.client.advance(self.actual_request_id))
        with torch.inference_mode():
            hidden = self.root(self.provider.session)
            lease = self.provider.p._lease
            saved = lease.tensor.detach()
            lease.tensor.data = torch.zeros_like(lease.tensor)
            before = self.transport.restores
            try:
                response = self.client.query()
                self.assertEqual(response['status'], 'NATIVE_STATE_UNAVAILABLE', response['status'])
                self.assertEqual(self.transport.restores, before)
                self.assertTrue(response['ownership_retained'])
            finally:
                lease.tensor.data = saved
                self.model.processor(self.model.head, hidden[-1:].contiguous())
        self.ok(self.client.unregister())


if __name__ == '__main__':
    unittest.main()
