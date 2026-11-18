"""CPU-only public census RPC contracts; no serving runtime or device is loaded."""

from copy import deepcopy
import json
import sys
import threading
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from torch.utils.hooks import RemovableHandle

from demoncat.runtime.vllm_ascend.controller import DemoncatWorkerExtension
from demoncat.runtime.vllm_ascend.worker_extension import identity, runtime
from worker_extension.test_core_census import NaturalCPUModel
from worker_extension.test_extension import ReadbackProvider
from worker_extension.test_serving_context import fixture


class CPURunner:
    """Weakref-capable fixture for the real IdentityTracker."""


class CPUWorker(DemoncatWorkerExtension):
    def __init__(self, runner):
        self.rank = 0
        self.device = NS(type='npu', index=0)  # metadata only
        self.model_runner = runner
        self.load_model = Mock(side_effect=AssertionError('RPC must not create a Worker/model'))
        self.execute_model = Mock(side_effect=AssertionError('serving host owns execution'))


class WorkerCensusRPCTests(unittest.TestCase):
    def setUp(self):
        raw_runner, _, self.actual, self.context, prompt, request_id = fixture()
        self.model = NaturalCPUModel()
        self.runner = CPURunner()
        self.runner.__dict__.update(vars(raw_runner))
        self.runner.model = self.context.model_instance = self.model
        self.worker = CPUWorker(self.runner)
        self.captures = []
        self.settings = dict(phase='prefill', expected=dict(
            internal_request_id=request_id, prompt_token_ids=prompt))
        current = patch.object(identity, '_current_device', return_value=0)
        current.start()
        self.addCleanup(current.stop)
        forward_context = ModuleType('vllm.forward_context')
        forward_context.get_forward_context = lambda: self.context
        modules = patch.dict(sys.modules, {'vllm.forward_context': forward_context, 'torch_npu': None})
        modules.start()
        self.addCleanup(modules.stop)
        self.provider = ReadbackProvider()
        resolver = patch.object(runtime, '_resolve_provider', return_value=self.provider)
        self.resolve_provider = resolver.start()
        self.addCleanup(resolver.stop)
        self.identity = self.worker.demoncat_worker_identity()['worker']
        self.runtime = self.worker._demoncat_worker_fault_runtime
        self.tenant = self.attach('tenant-a')
        self.addCleanup(self.remove_fixture_hooks)

    def remove_fixture_hooks(self):
        # A model-replacement test deliberately leaves the runtime quarantined.
        # Teardown cleans its original CPU fixture directly; this is not a
        # product recovery API or evidence that its runtime owner was released.
        self.worker.model_runner = self.runner
        self.runner.model = self.model
        self.worker.rank = 0
        self.worker.device = NS(type='npu', index=0)
        for capture in self.captures:
            capture.cleanup(self.worker, capture.identity)

    def attach(self, name):
        row = self.worker.demoncat_attach(dict(tenant_id=name, worker=self.identity))
        self.assertEqual(row['status'], 'ATTACHED', row)
        return row['ticket']

    def start(self, tenant=None, **changes):
        payload = dict(tenant or self.tenant, owner='census-owner', settings=deepcopy(self.settings))
        payload.update(changes)
        row = self.worker.demoncat_census('start', payload)
        if self.runtime._census is not None:
            capture = self.runtime._census['capture']
            if capture not in self.captures:
                self.captures.append(capture)
        return row

    def rpc(self, action, ticket):
        return self.worker.demoncat_census(action, ticket)

    def normal_request(self):
        return self.model(**self.actual)

    def assert_released(self, row):
        self.assertEqual(row['status'], 'CENSUS_RELEASED', row)
        report = row['census']
        self.assertTrue(report['releasable'])
        self.assertTrue(report['resource_cleanup_confirmed'])
        self.assertTrue(report['cleanup_complete'])
        self.assertTrue(report['identity_valid'])
        self.assertTrue(report['stop_requested'])
        self.assertFalse(report['inflight'])
        self.assertFalse(report['foreign_hooks'])
        self.assertFalse(report['metadata_scope_active'])
        self.assertEqual(report['observer_hooks'], 0)
        self.assertFalse(report['qualification_granted'])
        self.assertIsNone(self.runtime._census)
        self.assertFalse(self.worker.demoncat_worker_identity()['quarantined'])
        self.assertEqual(sum(len(module._forward_hooks) + len(module._forward_pre_hooks)
                             for module in self.model.modules()), 0)

    def test_public_lifecycle_keeps_owner_until_actual_cleanup_and_same_worker_resumes(self):
        started = self.start()
        self.assertEqual(started['status'], 'CENSUS_SNAPSHOT')
        ticket = started['ticket']
        self.assertEqual(ticket['owner'], 'census-owner')
        self.assertIs(type(ticket['generation']), int)
        self.assertEqual(self.model.calls, 0)
        observed = self.normal_request()
        snapshot = self.rpc('query', ticket)
        self.assertTrue(snapshot['census']['request_binding_valid'])
        self.assertEqual(snapshot['census']['observation_status'], 'OBSERVED')
        self.assertFalse(self.rpc('stop', ticket)['census']['releasable'])
        self.assertIsNotNone(self.runtime._census)
        self.assert_released(self.rpc('cleanup', ticket))
        self.assertEqual(self.normal_request().tolist(), observed.tolist())
        self.assertEqual(self.rpc('query', ticket)['reason'], 'STALE_CENSUS_GENERATION')
        replacement = self.start()['ticket']
        self.assertGreater(replacement['generation'], ticket['generation'])
        self.assertEqual(self.rpc('cleanup', ticket)['reason'], 'STALE_CENSUS_GENERATION')
        self.assert_released(self.rpc('cleanup', replacement))
        self.worker.load_model.assert_not_called()
        self.worker.execute_model.assert_not_called()

    def test_wrong_tenant_nonce_owner_generation_and_worker_never_touch_capture(self):
        other = self.attach('tenant-b')
        ticket = self.start()['ticket']
        capture = self.runtime._census['capture']
        for changed, reason in (
            (dict(owner='foreign'), 'CENSUS_OWNER_MISMATCH'),
            (dict(generation=ticket['generation'] + 1), 'STALE_CENSUS_GENERATION'),
            (dict(generation=True), 'STALE_CENSUS_GENERATION'),
            (dict(attach_nonce='foreign'), 'STALE_ATTACHMENT'),
            (dict(tenant_id=other['tenant_id'], attach_nonce=other['attach_nonce']), 'CENSUS_OWNER_MISMATCH'),
            (dict(worker=dict(self.identity, rank=99)), 'RPC_NOT_TARGETED'),
            (dict(worker=dict(self.identity, model_instance_id='foreign')), 'WORKER_IDENTITY_MISMATCH'),
        ):
            with self.subTest(reason=reason), patch.object(capture, 'cleanup', wraps=capture.cleanup) as clean:
                row = self.rpc('cleanup', dict(ticket, **changed))
                self.assertEqual(row.get('reason', row['status']), reason)
                clean.assert_not_called()
                self.assertIs(self.runtime._census['capture'], capture)
        self.assertEqual(self.worker.demoncat_detach(other)['status'], 'DETACHED')
        self.assertIs(self.runtime._census['capture'], capture)
        self.assert_released(self.rpc('cleanup', ticket))

    def test_fault_lease_census_discovery_and_new_attachment_are_mutually_excluded(self):
        ticket = self.start()['ticket']
        fault = dict(self.tenant, owner='fault-owner', mechanism='20.buffer_substitute',
                     implementation_version='1')
        self.assertEqual(self.worker.demoncat_fault_rpc('prepare', fault)['reason'], 'WORKER_CENSUS_IN_PROGRESS')
        self.assertEqual(self.worker.demoncat_discover(self.tenant)['reason'],
                         'DISCOVERY_REQUIRES_IDLE_WORKER_OWNER_TABLE')
        self.assertEqual(self.worker.demoncat_attach(dict(tenant_id='new', worker=self.identity))['reason'],
                         'WORKER_CENSUS_IN_PROGRESS')
        self.assertEqual(self.start(owner='other')['reason'], 'CENSUS_REQUIRES_IDLE_WORKER_OWNER_TABLE')
        self.resolve_provider.assert_not_called()
        self.assert_released(self.rpc('cleanup', ticket))
        lease = self.worker.demoncat_fault_rpc('prepare', fault)['ticket']
        self.assertEqual(self.start()['reason'], 'CENSUS_REQUIRES_IDLE_WORKER_OWNER_TABLE')
        self.assertEqual(self.worker.demoncat_fault_rpc('unregister', lease)['status'], 'ACTION_COMPLETED')
        self.assert_released(self.rpc('cleanup', self.start()['ticket']))

    def test_partial_start_and_failed_cleanup_keep_owner_ticket_for_retry(self):
        from demoncat.runtime.vllm_ascend.worker_extension.call_census import WorkerCallCensus
        original_start = WorkerCallCensus.start
        def verify_owner_before_start(capture):
            self.assertIs(self.runtime._census['capture'], capture)
            return original_start(capture)
        with patch.object(WorkerCallCensus, 'start', verify_owner_before_start):
            with patch.object(self.model.activation, 'register_forward_hook',
                              side_effect=RuntimeError('CPU_RPC_INSTALL_FAILED')):
                with patch.object(RemovableHandle, 'remove', side_effect=RuntimeError('CPU_RPC_REMOVE_FAILED')):
                    failed = self.start()
        self.assertEqual(failed['status'], 'CENSUS_RECOVERY_REQUIRED')
        ticket = failed['ticket']
        self.assertTrue(self.worker.demoncat_worker_identity()['quarantined'])
        row = self.rpc('query', ticket)
        self.assertTrue(row['recovery_required'])
        self.assertGreater(row['census']['observer_hooks'], 0)
        self.assertEqual(row['census']['capture_errors']['workload_error_reason'], 'CPU_RPC_INSTALL_FAILED')
        self.assert_released(self.rpc('cleanup', ticket))

    def test_cleanup_failure_and_detach_failure_preserve_owner_and_attachment(self):
        ticket = self.start()['ticket']
        self.normal_request()
        with patch.object(RemovableHandle, 'remove', side_effect=RuntimeError('CPU_RPC_REMOVE_FAILED')):
            failed = self.rpc('cleanup', ticket)
            self.assertEqual(failed['status'], 'CENSUS_RECOVERY_REQUIRED')
            detached = self.worker.demoncat_detach(self.tenant)
            self.assertEqual(detached['status'], 'RECOVERY_REQUIRED')
            self.assertEqual(detached['census_recoveries'][0]['status'], 'CENSUS_RECOVERY_REQUIRED')
            self.assertIn(self.tenant['tenant_id'], self.runtime._tenants)
            self.assertIsNotNone(self.runtime._census)
        self.assert_released(self.rpc('cleanup', ticket))
        self.assertEqual(self.worker.demoncat_detach(self.tenant)['status'], 'DETACHED')
        self.assertNotIn(self.tenant['tenant_id'], self.runtime._tenants)

    def test_detach_runs_owned_cleanup_before_revoking_attachment(self):
        self.start()
        self.normal_request()
        detached = self.worker.demoncat_detach(self.tenant)
        self.assertEqual(detached['status'], 'DETACHED')
        self.assertEqual(detached['census_recoveries'][0]['status'], 'CENSUS_RELEASED')
        self.assertIsNone(self.runtime._census)
        self.assertEqual(self.worker.demoncat_discover(self.tenant)['reason'], 'STALE_ATTACHMENT')
        self.assertEqual(self.normal_request().tolist(), [0.0, 4.0])

    def test_runtime_model_generation_change_quarantines_and_preserves_old_capture(self):
        ticket = self.start()['ticket']
        capture = self.runtime._census['capture']
        self.runner.model = NaturalCPUModel()
        changed = self.worker.demoncat_worker_identity()
        self.assertTrue(changed['quarantined'])
        self.assertEqual(self.rpc('cleanup', ticket)['reason'], 'WORKER_IDENTITY_MISMATCH')
        self.assertEqual(self.rpc('cleanup', dict(ticket, worker=changed['worker']))['reason'],
                         'WORKER_IDENTITY_CHANGED_WITH_OWNED_RESOURCES')
        self.assertIs(self.runtime._census['capture'], capture)
        self.assertGreater(len(self.model._forward_pre_hooks), 0)

    def test_owner_thread_failure_is_recoverable_without_releasing_hooks(self):
        ticket = self.start()['ticket']
        rows = []
        thread = threading.Thread(target=lambda: rows.append(self.rpc('cleanup', ticket)))
        thread.start()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(rows[0]['status'], 'CENSUS_RECOVERY_REQUIRED')
        self.assertIsNotNone(self.runtime._census)
        self.assertGreater(self.rpc('query', ticket)['census']['observer_hooks'], 0)
        self.assert_released(self.rpc('cleanup', ticket))

    def test_inflight_rpc_stop_does_not_release_until_natural_call_returns(self):
        ticket = self.start()['ticket']
        during = []
        def stop_inside_model():
            during.append(self.rpc('stop', ticket))
            during.append(self.rpc('cleanup', ticket))
        self.model.during_forward = stop_inside_model
        self.normal_request()
        self.model.during_forward = None
        self.assertTrue(during[0]['census']['inflight'])
        self.assertFalse(during[0]['census']['releasable'])
        self.assertEqual(during[1]['status'], 'CENSUS_RECOVERY_REQUIRED')
        self.assertIsNotNone(self.runtime._census)
        self.assert_released(self.rpc('cleanup', ticket))

    def test_actual_request_failure_is_failed_evidence_and_does_not_permanently_lock_owner(self):
        ticket = self.start()['ticket']
        self.runner.input_batch.req_ids.append('CPU_ONLY_FOREIGN_REQUEST')
        self.runner.input_batch.num_reqs = 2
        self.normal_request()
        failed = self.rpc('query', ticket)['census']
        self.assertFalse(failed['request_binding_valid'])
        self.assertEqual(failed['observation_status'], 'FAILED')
        cleaned = self.rpc('cleanup', ticket)
        self.assert_released(cleaned)
        self.assertEqual(cleaned['census']['observation_status'], 'FAILED')
        self.assertTrue(cleaned['census']['request_failures'])
        self.assertFalse(cleaned['qualification_granted'])
        self.assertEqual(self.worker.demoncat_detach(self.tenant)['status'], 'DETACHED')

    def test_missing_contradictory_or_unserializable_release_readback_keeps_owner(self):
        ticket = self.start()['ticket']
        capture = self.runtime._census['capture']
        base = capture.query(self.worker, self.identity)
        base.update(releasable=True, cleanup_complete=True, resource_cleanup_confirmed=True,
                    identity_valid=True, stop_requested=True, inflight=False, observer_hooks=0,
                    foreign_hooks=False, metadata_scope_active=False)
        changes = [dict(observer_hooks=1), dict(observer_hooks=False), dict(inflight=True),
                   dict(foreign_hooks=True), dict(metadata_scope_active=True),
                   dict(identity_valid=False), dict(stop_requested=False),
                   dict(cleanup_complete=False), dict(resource_cleanup_confirmed=False),
                   dict(report={'cannot_serialize': object()}), dict(qualification_granted=True)]
        for index, changed in enumerate(changes):
            with self.subTest(index=index):
                report = dict(base, **changed)
                with patch.object(capture, 'cleanup', return_value=report), \
                        patch.object(capture, 'query', return_value=report):
                    row = self.rpc('cleanup', ticket)
                self.assertEqual(row['status'], 'CENSUS_RECOVERY_REQUIRED')
                self.assertIs(self.runtime._census['capture'], capture)
        missing = dict(base)
        missing.pop('resource_cleanup_confirmed')
        with patch.object(capture, 'cleanup', return_value=missing), \
                patch.object(capture, 'query', return_value=missing):
            self.assertEqual(self.rpc('cleanup', ticket)['status'], 'CENSUS_RECOVERY_REQUIRED')
        self.assertGreater(len(self.model._forward_pre_hooks), 0)
        self.assert_released(self.rpc('cleanup', ticket))

    def test_successful_cleanup_message_cannot_override_fresh_live_hook_readback(self):
        ticket = self.start()['ticket']
        capture = self.runtime._census['capture']
        claimed = capture.query(self.worker, self.identity)
        claimed.update(releasable=True, cleanup_complete=True, resource_cleanup_confirmed=True,
                       identity_valid=True, stop_requested=True, inflight=False, observer_hooks=0,
                       foreign_hooks=False, metadata_scope_active=False)
        with patch.object(capture, 'cleanup', return_value=claimed), \
                patch.object(capture, 'query', wraps=capture.query) as fresh:
            row = self.rpc('cleanup', ticket)
            fresh.assert_called_once_with(self.worker, self.identity)
        self.assertEqual(row['status'], 'CENSUS_SNAPSHOT')
        self.assertTrue(row['recovery_required'])
        self.assertFalse(row['census']['releasable'])
        self.assertGreater(row['census']['observer_hooks'], 0)
        self.assertIs(self.runtime._census['capture'], capture)
        self.assert_released(self.rpc('cleanup', ticket))

    def test_rpc_cannot_supply_native_factory_or_bypass_census_namespace(self):
        for changed in (dict(native_observer_factory='module.factory'), dict(max_events=True),
                        dict(max_events=100001), dict(max_events=0)):
            with self.subTest(changed=changed):
                row = self.start(settings=dict(self.settings, **changed))
                self.assertEqual(row['status'], 'REJECTED')
                self.assertIsNone(self.runtime._census)
        self.assertEqual(self.worker.demoncat_fault_rpc('census_start', {})['reason'], 'UNSUPPORTED_ACTION')
        self.assertEqual(self.worker.demoncat_census('prepare', {})['reason'], 'UNSUPPORTED_CENSUS_ACTION')
        self.assertEqual(len(self.model._forward_pre_hooks), 0)
        self.resolve_provider.assert_not_called()
        json.dumps(self.worker.demoncat_worker_identity(), allow_nan=False)


if __name__ == '__main__':
    unittest.main()
