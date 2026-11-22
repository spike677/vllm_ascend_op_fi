"""CPU checks for the existing-Worker extension's ownership and dispatch contract."""

import json
import os
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from demoncat.runtime.vllm_ascend.controller import DemoncatWorkerExtension
from demoncat.runtime.vllm_ascend.worker_extension import identity, runtime


MECHANISM = '20.buffer_substitute'
OTHER_MECHANISM = '39.matmul_range'
VERSION = '1'
_READ_CURRENT_DEVICE = identity._current_device


class Model:
    pass


class Runner:
    def __init__(self):
        self.model = Model()


class HostWorker:
    def __init__(self, rank=3, device=0):
        self.rank = rank
        self.rpc_rank = 1
        self.device = SimpleNamespace(type='npu', index=device)
        self.model_runner = None
        self.events = []

    def load_model(self):
        self.model_runner = Runner()
        self.events.append('load')

    def execute_model(self, batch):
        self.events.append(('execute', batch))
        return batch


def _extended_worker(rank=3, device=0):
    class AscendWorker(HostWorker):
        pass

    # Same dynamic Worker subclass mechanism used by vLLM's WorkerWrapperBase.
    extension_attributes = [name for name in dir(DemoncatWorkerExtension) if not name.startswith('__')]
    assert not any(hasattr(AscendWorker, name) for name in extension_attributes)
    AscendWorker.__bases__ += (DemoncatWorkerExtension,)
    return AscendWorker(rank, device)


class ReadbackProvider:
    def __init__(self):
        self.actions = []
        self.queries = []
        self.callbacks = {}
        self.state = dict(native_loaded=True, registered=False, native_registered=False,
                          enabled=False, inflight=0, leases=0)

    def dispatch(self, action, settings, *, owner, generation):
        self.actions.append((action, settings, owner, generation))
        callback = self.callbacks.get(action)
        if callback is not None:
            callback()
        if action == 'register':
            self.state['registered'] = True
        if action in ('disable', 'abort', 'clean', 'unregister'):
            self.state.update(enabled=False, inflight=0, leases=0)
        if action in ('clean', 'unregister'):
            self.state.update(registered=False, native_registered=False)
        return {'dispatch': action}

    def query(self, *, owner, generation):
        self.queries.append((owner, generation))
        return self.state.copy()


class WorkerExtensionTest(unittest.TestCase):
    def setUp(self):
        self.worker = _extended_worker()
        self.provider = ReadbackProvider()
        self.resolver = Mock(return_value=self.provider)
        current_patch = patch.object(identity, '_current_device', return_value=0)
        self.addCleanup(current_patch.stop)
        self.current = current_patch.start()
        resolver_patch = patch.object(runtime, '_resolve_provider', self.resolver)
        self.addCleanup(resolver_patch.stop)
        resolver_patch.start()

    def _attach(self, tenant='tenant-a'):
        row = self.worker.demoncat_worker_identity()
        self.assertEqual(row['status'], 'WORKER_IDENTIFIED')
        attached = self.worker.demoncat_attach({'tenant_id': tenant, 'worker': row['worker']})
        self.assertEqual(attached['status'], 'ATTACHED', attached)
        return json.loads(json.dumps(attached['ticket']))

    def _prepare(self, ticket, mechanism=MECHANISM, owner='owner-a'):
        return self.worker.demoncat_fault_rpc('prepare', dict(
            ticket, mechanism=mechanism, implementation_version=VERSION, owner=owner,
        ))

    def _check(self, attach):
        prepared = self._prepare(attach)
        self.assertEqual(prepared['status'], 'ACTION_COMPLETED', prepared)
        lease = prepared['ticket']
        for action in ('register', 'check'):
            response = self.worker.demoncat_fault_rpc(action, lease)
            self.assertEqual(response['status'], 'ACTION_COMPLETED', response)
        return lease

    def test_only_namespaced_rpc_methods_and_live_worker_identity(self):
        self.assertEqual(
            sorted(name for name in vars(DemoncatWorkerExtension) if not name.startswith('__')),
            ['demoncat_attach', 'demoncat_capabilities', 'demoncat_census', 'demoncat_detach', 'demoncat_discover', 'demoncat_fault_query',
             'demoncat_fault_rpc', 'demoncat_inspect_readonly', 'demoncat_product_binding', 'demoncat_worker_identity'],
        )
        before = self.worker.demoncat_worker_identity()
        self.assertEqual(before['status'], 'WORKER_CONTEXT_NOT_READY')
        self.current.assert_not_called()
        self.worker.load_model()
        with self.assertRaisesRegex(RuntimeError,'PRODUCT_RPC_OPERATOR_NOT_AUTHORIZED'):
            self.worker.demoncat_product_binding('attach',{'profile':'user-supplied','operator':'user'})
        with patch.dict(os.environ, {'ASCEND_RT_VISIBLE_DEVICES': '5,2'}):
            actual = self.worker.demoncat_worker_identity()
            self.assertEqual(actual['status'], 'WORKER_IDENTIFIED')
            expected = actual['worker']
            self.assertEqual(expected['rank'], 3)
            self.assertEqual(expected['logical_device'], 0)
            self.assertEqual(expected['visible_devices'], [5, 2])
            self.assertEqual(expected['visible_devices_source'], 'ASCEND_RT_VISIBLE_DEVICES')
            self.assertEqual(actual, self.worker.demoncat_worker_identity())
            self.current.return_value = 1
            mismatch = self.worker.demoncat_worker_identity()
            self.assertEqual(mismatch['status'], 'WORKER_CONTEXT_NOT_READY')
            self.assertIsNone(mismatch['worker']['logical_device'])
            self.assertEqual(mismatch['worker']['observed_logical_device'], 1)
        self.assertEqual(self.worker.execute_model('request-a'), 'request-a')
        self.assertEqual(self.worker.events, ['load', ('execute', 'request-a')])
        self.resolver.assert_not_called()

    def test_uninitialized_npu_device_is_never_selected_or_initialized(self):
        torch = ModuleType('torch')
        torch.npu = SimpleNamespace(is_initialized=Mock(return_value=False), current_device=Mock(return_value=2))
        with patch.dict(sys.modules, {'torch': torch}):
            self.assertIsNone(_READ_CURRENT_DEVICE())
            torch.npu.current_device.assert_not_called()
            torch.npu.is_initialized.return_value = True
            self.assertEqual(_READ_CURRENT_DEVICE(), 2)

    def test_full_lease_lifecycle_and_live_native_readback(self):
        self.worker.load_model()
        attached = self._attach()
        self.resolver.assert_not_called()
        lease = self._check(attached)
        mech, version, worker, model, context = self.resolver.call_args.args
        self.assertEqual((mech, version), (MECHANISM, VERSION))
        self.assertIs(worker, self.worker)
        self.assertIs(model, self.worker.model_runner.model)
        self.assertEqual(context, attached['worker'])
        query = self.worker.demoncat_fault_query(lease)
        self.assertEqual((query['status'], query['native']['inflight']), ('QUERY_OK', 0))
        self.assertEqual(self.worker.demoncat_fault_rpc('execute', lease)['phase'], 'ATTEMPTED')
        self.assertEqual(self.worker.demoncat_fault_rpc('execute', lease)['reason'], 'ATTEMPT_CONSUMED')
        self.assertEqual(self.worker.demoncat_fault_rpc('finish', lease)['phase'], 'FINISHED')
        self.assertEqual(self.worker.demoncat_fault_rpc('finish', lease)['reason'], 'INVALID_ACTION_FOR_PHASE')
        cleaned = self.worker.demoncat_fault_rpc('clean', lease)
        self.assertEqual(cleaned['phase'], 'CLEANED')
        self.assertIs(cleaned['native']['native_registered'], False)
        self.assertEqual(self.worker.demoncat_detach(attached)['status'], 'DETACHED')
        self.assertEqual([entry[0] for entry in self.provider.actions],
                         ['prepare', 'register', 'check', 'execute', 'finish', 'clean', 'unregister'])
        actual_owners = {entry[2] for entry in self.provider.actions}
        self.assertEqual(len(actual_owners), 1)
        self.assertNotIn(lease['owner'], actual_owners)

    def test_host_driven_pending_replay_excludes_other_owners(self):
        self.worker.load_model()
        tenant = self._attach()
        lease = self._check(tenant)
        self.provider.callbacks['execute'] = lambda: self.provider.state.update(replay_pending=True)
        self.assertEqual(self.worker.demoncat_fault_rpc('execute', lease)['status'], 'ACTION_COMPLETED')
        self.assertFalse(self.worker.demoncat_fault_query(lease)['quarantined'])
        self.assertEqual(self._prepare(tenant, mechanism=OTHER_MECHANISM)['reason'], 'WORKER_REPLAY_IN_PROGRESS')
        other = dict(tenant_id='other', worker=tenant['worker'])
        self.assertEqual(self.worker.demoncat_attach(other)['reason'], 'WORKER_REPLAY_IN_PROGRESS')
        self.assertEqual(self.worker.demoncat_fault_rpc('advance', dict(lease, settings={'request_id':'mock-enqueued'}))['status'],
                         'ACTION_COMPLETED')
        self.provider.callbacks['finish'] = lambda: self.provider.state.update(replay_pending=False)
        self.assertEqual(self.worker.demoncat_fault_rpc('finish', lease)['phase'], 'FINISHED')

    def test_pending_hook_cannot_be_hidden_by_idle_native_on_finish(self):
        self.worker.load_model()
        lease = self._check(self._attach())
        self.provider.callbacks['execute'] = lambda: self.provider.state.update(replay_pending=True)
        self.worker.demoncat_fault_rpc('execute', lease)
        failed = self.worker.demoncat_fault_rpc('finish', lease)
        self.assertEqual(failed['status'], 'RECOVERY_REQUIRED')
        self.assertTrue(self.worker.demoncat_fault_query(lease)['quarantined'])
        self.provider.callbacks['unregister'] = lambda: self.provider.state.update(replay_pending=False)
        self.assertEqual(self.worker.demoncat_fault_rpc('unregister', lease)['status'], 'ACTION_COMPLETED')

    def test_failed_serving_replay_requires_off_only_recovery(self):
        self.worker.load_model()
        lease = self._check(self._attach())
        self.worker.demoncat_fault_rpc('execute', lease)
        self.provider.state.update(replay_pending=True, replay_failed=True)
        self.assertTrue(self.worker.demoncat_fault_query(lease)['quarantined'])
        self.assertEqual(self.worker.demoncat_fault_rpc('advance', lease)['reason'], 'RECOVERY_REQUIRED')
        self.provider.callbacks['recover'] = lambda: self.provider.state.update(replay_failed=False)
        reply = self.worker.demoncat_fault_rpc('recover', lease)
        self.assertEqual(reply['status'], 'ACTION_COMPLETED')
        self.assertEqual(reply['phase'], 'ATTEMPTED')
        self.assertEqual(self.worker.demoncat_fault_rpc('execute', lease)['reason'], 'ATTEMPT_CONSUMED')

    def test_wrong_worker_owner_version_or_generation_does_not_dispatch(self):
        self.worker.load_model()
        attached = self._attach()
        prepared = self._prepare(attached)
        lease = prepared['ticket']
        for changed, reason in (
            (dict(rank=9), 'RPC_NOT_TARGETED'),
            (dict(model_instance_id='forged'), 'WORKER_IDENTITY_MISMATCH'),
        ):
            row = self.worker.demoncat_fault_query(dict(lease, worker=dict(lease['worker'], **changed)))
            self.assertEqual(row.get('reason', row['status']), reason)
        for changed, reason in (
            (dict(generation=lease['generation'] + 1), 'STALE_GENERATION'),
            (dict(generation=True), 'STALE_GENERATION'),
            (dict(owner='wrong'), 'OWNER_MISMATCH'),
            (dict(implementation_version='wrong'), 'PROVIDER_IDENTITY_MISMATCH'),
            (dict(attach_nonce='wrong'), 'STALE_ATTACHMENT'),
        ):
            row = self.worker.demoncat_fault_rpc('register', dict(lease, **changed))
            self.assertEqual(row['reason'], reason)
        row = self.worker.demoncat_fault_rpc('query', lease)
        self.assertEqual(row['reason'], 'UNSUPPORTED_ACTION')
        self.assertEqual([entry[0] for entry in self.provider.actions], ['prepare'])
        self.assertEqual(len(self.provider.queries), 1)

    def test_detach_releases_only_own_lease_and_revokes_ticket(self):
        self.worker.load_model()
        tenant_a = self._attach()
        tenant_b = self._attach('tenant-b')
        lease_a = self._prepare(tenant_a)['ticket']
        self.assertEqual(self._prepare(tenant_b)['reason'], 'MECHANISM_ALREADY_OWNED')
        other = ReadbackProvider()
        self.resolver.return_value = other
        lease_b = self._prepare(tenant_b, OTHER_MECHANISM)['ticket']
        forged = dict(lease_b, tenant_id=tenant_a['tenant_id'], attach_nonce=tenant_a['attach_nonce'])
        self.assertEqual(self.worker.demoncat_fault_rpc('unregister', forged)['reason'], 'OWNER_MISMATCH')
        self.assertEqual(self.worker.demoncat_detach(tenant_a)['status'], 'DETACHED')
        self.assertEqual([entry[0] for entry in self.provider.actions], ['prepare', 'unregister'])
        self.assertEqual([entry[0] for entry in other.actions], ['prepare'])
        self.assertEqual(self.worker.demoncat_fault_query(lease_b)['status'], 'QUERY_OK')
        replacement = self._attach()
        self.assertNotEqual(tenant_a['attach_nonce'], replacement['attach_nonce'])
        self.assertEqual(self.worker.demoncat_fault_query(lease_a)['reason'], 'STALE_ATTACHMENT')

    def test_preload_absence_before_mutation_allows_detach(self):
        self.worker.load_model()
        tenant = self._attach()
        self.resolver.side_effect = runtime.NativePreloadNotAttached
        self.assertEqual(self._prepare(tenant)['status'], 'NATIVE_PRELOAD_NOT_ATTACHED')
        self.resolver.side_effect = None
        self.provider.callbacks['prepare'] = Mock(side_effect=runtime.NativePreloadNotAttached)
        self.assertEqual(self._prepare(tenant), {'status': 'NATIVE_PRELOAD_NOT_ATTACHED', 'action': 'prepare'})
        self.assertEqual(self.worker.demoncat_detach(tenant)['status'], 'DETACHED')
        self.assertEqual(self.provider.queries, [])

    def test_unverified_recovery_preserves_lease_and_tenant(self):
        self.worker.load_model()
        tenant = self._attach()
        self.provider.callbacks['prepare'] = Mock(side_effect=RuntimeError)
        failed = self._prepare(tenant)
        self.assertEqual((failed['status'], failed['recovery_required']), ('PROVIDER_ACTION_FAILED', True))
        lease = failed['ticket']
        self.provider.state['native_loaded'] = False
        self.assertEqual(self.worker.demoncat_fault_query(lease)['status'], 'NATIVE_STATE_UNAVAILABLE')
        self.assertEqual(self.worker.demoncat_fault_rpc('register', lease)['reason'], 'RECOVERY_REQUIRED')
        self.assertEqual(self.worker.demoncat_detach(tenant)['status'], 'RECOVERY_REQUIRED')
        self.assertEqual(self._prepare(tenant)['reason'], 'MECHANISM_ALREADY_OWNED')
        self.provider.state['native_loaded'] = True
        self.assertEqual(self.worker.demoncat_detach(tenant)['status'], 'DETACHED')

    def test_malformed_rpc_and_safe_provider_failure(self):
        self.worker.load_model()
        tenant = self._attach()
        settings = {}
        settings['cycle'] = settings
        fields = dict(tenant, mechanism=MECHANISM, implementation_version=VERSION, owner='owner-a')
        row = self.worker.demoncat_fault_rpc('prepare', dict(fields, settings=settings))
        self.assertEqual(row['reason'], 'INVALID_PROVIDER_SETTINGS')
        row = self.worker.demoncat_fault_rpc('prepare', dict(fields, worker=dict(tenant['worker'], cycle=settings)))
        self.assertEqual(row['reason'], 'INVALID_WORKER_IDENTITY')
        self.resolver.assert_not_called()
        lease = self._prepare(tenant)['ticket']
        error = RuntimeError('private or very large provider diagnostic')
        error.code = 'TRUSTED_REPLAY_REQUIRED'
        self.provider.callbacks['register'] = Mock(side_effect=error)
        row = self.worker.demoncat_fault_rpc('register', lease)
        self.assertEqual(row['provider_code'], 'TRUSTED_REPLAY_REQUIRED')
        self.assertNotIn(str(error), json.dumps(row))

    def test_missing_native_registration_readback_does_not_prove_off(self):
        self.worker.load_model()
        tenant = self._attach()
        self.provider.state.pop('native_registered')
        row = self._prepare(tenant)
        self.assertEqual(row['status'], 'NATIVE_STATE_UNAVAILABLE')
        self.assertTrue(row['recovery_required'])
        self.provider.state['native_registered'] = True
        self.provider.callbacks['unregister'] = Mock(side_effect=RuntimeError)
        self.assertEqual(self.worker.demoncat_detach(tenant)['status'], 'RECOVERY_REQUIRED')

    def test_execute_requires_off_and_drained_and_consumes_attempt_on_failure(self):
        self.worker.load_model()
        lease = self._check(self._attach())
        self.provider.callbacks['execute'] = lambda: self.provider.state.update(enabled=True, inflight=1)
        row = self.worker.demoncat_fault_rpc('execute', lease)
        self.assertEqual(row['status'], 'RECOVERY_REQUIRED')
        self.assertTrue(row['native']['enabled'])
        blocked = self.worker.demoncat_fault_rpc('enable', lease)
        self.assertEqual(blocked['reason'], 'RECOVERY_REQUIRED')
        self.assertEqual(self.worker.demoncat_fault_rpc('disable', lease)['status'], 'ACTION_COMPLETED')
        self.assertEqual(self.worker.demoncat_fault_rpc('execute', lease)['reason'], 'ATTEMPT_CONSUMED')

    def test_model_replacement_and_process_restart_revoke_old_tickets(self):
        self.worker.load_model()
        old = self._attach()
        self.worker.load_model()
        fresh = self._attach()
        self.assertNotEqual(old['worker']['model_instance_id'], fresh['worker']['model_instance_id'])
        self.assertEqual(self._prepare(old)['reason'], 'WORKER_IDENTITY_MISMATCH')
        self.assertEqual(self.worker.demoncat_detach(fresh)['status'], 'DETACHED')
        with patch('demoncat.runtime.vllm_ascend.worker_extension.extension.os.getpid',
                   return_value=old['worker']['pid'] + 10000):
            restarted = self._attach()
            self.assertNotEqual(old['worker']['worker_start_identity'], restarted['worker']['worker_start_identity'])
            self.assertEqual(self._prepare(old)['reason'], 'WORKER_IDENTITY_MISMATCH')
        self.resolver.assert_not_called()

    def test_model_replacement_with_lease_quarantines_the_worker(self):
        self.worker.load_model()
        tenant = self._attach()
        lease = self._prepare(tenant)['ticket']
        first_model = self.worker.model_runner.model
        self.worker.model_runner.model = Model()
        replacement = self.worker.demoncat_worker_identity()['worker']
        row = self.worker.demoncat_attach({'tenant_id': 'new', 'worker': replacement})
        self.assertEqual(row['reason'], 'WORKER_IDENTITY_CHANGED_WITH_OWNED_RESOURCES')
        self.worker.model_runner.model = first_model
        current = self.worker.demoncat_worker_identity()['worker']
        self.assertEqual(
            self.worker.demoncat_fault_rpc('clean', dict(lease, worker=current))['reason'],
            'WORKER_IDENTITY_CHANGED_WITH_OWNED_RESOURCES',
        )
        self.assertEqual([entry[0] for entry in self.provider.actions], ['prepare'])

    def test_overlapping_rpc_is_rejected_before_provider_access(self):
        self.worker.load_model()
        lease = self._check(self._attach())
        entered, release = threading.Event(), threading.Event()
        receipts = []

        def wait_for_release():
            entered.set()
            if not release.wait(5):
                raise TimeoutError

        self.provider.callbacks['execute'] = wait_for_release
        running = threading.Thread(target=lambda: receipts.append(self.worker.demoncat_fault_rpc('execute', lease)))
        running.start()
        try:
            self.assertTrue(entered.wait(5))
            before = len(self.provider.queries)
            row = self.worker.demoncat_fault_query(lease)
            self.assertEqual(row, {'status': 'REJECTED', 'reason': 'WORKER_RPC_BUSY'})
            self.assertEqual(before, len(self.provider.queries))
        finally:
            release.set()
            running.join(5)
        self.assertFalse(running.is_alive())
        self.assertEqual([row['status'] for row in receipts], ['ACTION_COMPLETED'])

    def test_capabilities_survive_a_missing_optional_provider_dependency(self):
        with patch.object(runtime, '_provider_capabilities',
                          side_effect=ModuleNotFoundError(name='fault_injection')):
            row = self.worker.demoncat_capabilities()
        self.assertEqual(row['status'], 'CAPABILITIES')
        self.assertEqual(row['provider_resolver_status'], 'DEPENDENCY_UNAVAILABLE')
        self.assertEqual(row['providers'], [])
        self.resolver.assert_not_called()

    def test_failed_provider_quarantines_other_mechanisms_and_new_tenants(self):
        self.worker.load_model()
        tenant = self._attach()
        failed = self._check(tenant)
        other = ReadbackProvider()
        self.resolver.return_value = other
        other_lease = self._prepare(tenant, OTHER_MECHANISM)['ticket']
        for action in ('register', 'check'):
            self.assertEqual(self.worker.demoncat_fault_rpc(action, other_lease)['status'], 'ACTION_COMPLETED')
        self.provider.callbacks['execute'] = Mock(side_effect=RuntimeError)
        self.assertEqual(self.worker.demoncat_fault_rpc('execute', failed)['status'], 'PROVIDER_ACTION_FAILED')
        before = len(other.actions)
        blocked = self.worker.demoncat_fault_rpc('execute', other_lease)
        self.assertEqual(blocked['reason'], 'WORKER_RECOVERY_REQUIRED')
        self.assertEqual(len(other.actions), before)
        current = self.worker.demoncat_worker_identity()
        self.assertTrue(current['quarantined'])
        attached = self.worker.demoncat_attach({'tenant_id': 'new', 'worker': current['worker']})
        self.assertEqual(attached['reason'], 'WORKER_RECOVERY_REQUIRED')
        fresh = self._prepare(tenant, '26.residual_stomp')
        self.assertEqual(fresh['reason'], 'WORKER_RECOVERY_REQUIRED')
        self.assertEqual(self.worker.demoncat_fault_rpc('disable', failed)['status'], 'ACTION_COMPLETED')
        self.assertFalse(self.worker.demoncat_capabilities()['quarantined'])
        self.assertEqual(self.worker.demoncat_fault_rpc('execute', other_lease)['status'], 'ACTION_COMPLETED')
        self.assertEqual(self.worker.demoncat_fault_rpc('execute', failed)['reason'], 'ATTEMPT_CONSUMED')

    def test_query_unknown_native_state_requires_explicit_recovery(self):
        self.worker.load_model()
        tenant = self._attach()
        lease = self._prepare(tenant)['ticket']
        self.provider.state['native_loaded'] = False
        missing = self.worker.demoncat_fault_query(lease)
        self.assertEqual(missing['status'], 'NATIVE_STATE_UNAVAILABLE')
        self.assertTrue(missing['recovery_required'])
        self.assertTrue(missing['quarantined'])
        self.provider.state['native_loaded'] = True
        self.assertTrue(self.worker.demoncat_fault_query(lease)['quarantined'])
        self.assertEqual(self._prepare(tenant, OTHER_MECHANISM)['reason'], 'WORKER_RECOVERY_REQUIRED')
        self.assertEqual(self.worker.demoncat_fault_rpc('clean', lease)['status'], 'ACTION_COMPLETED')
        self.assertFalse(self.worker.demoncat_capabilities()['quarantined'])

    def test_query_busy_native_state_cannot_admit_other_work(self):
        self.worker.load_model()
        tenant = self._attach()
        lease = self._check(tenant)
        self.provider.state.update(inflight=2, leases=1)
        current = self.worker.demoncat_fault_query(lease)
        self.assertEqual(current['native']['inflight'], 2)
        self.assertTrue(current['quarantined'])
        self.assertEqual(self.worker.demoncat_fault_rpc('execute', lease)['reason'], 'RECOVERY_REQUIRED')
        self.provider.state.update(inflight=0, leases=0)
        self.assertTrue(self.worker.demoncat_fault_query(lease)['quarantined'])
        self.assertEqual(self.worker.demoncat_fault_rpc('abort', lease)['status'], 'ACTION_COMPLETED')
        self.assertFalse(self.worker.demoncat_worker_identity()['quarantined'])

    def test_recovering_one_provider_does_not_release_another_failed_provider(self):
        self.worker.load_model()
        tenant = self._attach()
        a = self._prepare(tenant)['ticket']
        other = ReadbackProvider()
        self.resolver.return_value = other
        b = self._prepare(tenant, OTHER_MECHANISM)['ticket']
        self.provider.state['native_loaded'] = other.state['native_loaded'] = False
        self.worker.demoncat_fault_query(a)
        self.worker.demoncat_fault_query(b)
        self.provider.state['native_loaded'] = True
        self.assertEqual(self.worker.demoncat_fault_rpc('unregister', a)['status'], 'ACTION_COMPLETED')
        self.assertTrue(self.worker.demoncat_worker_identity()['quarantined'])
        self.assertEqual(self._prepare(tenant)['reason'], 'WORKER_RECOVERY_REQUIRED')
        other.state['native_loaded'] = True
        self.assertEqual(self.worker.demoncat_detach(tenant)['status'], 'DETACHED')
        self.assertFalse(self.worker.demoncat_worker_identity()['quarantined'])

    def test_interrupted_action_preserves_quarantine_and_consumes_attempt(self):
        self.worker.load_model()
        tenant = self._attach()
        lease = self._check(tenant)
        self.provider.callbacks['execute'] = Mock(side_effect=KeyboardInterrupt)
        with self.assertRaises(KeyboardInterrupt):
            self.worker.demoncat_fault_rpc('execute', lease)
        self.assertTrue(self.worker.demoncat_worker_identity()['quarantined'])
        self.assertEqual(self._prepare(tenant, OTHER_MECHANISM)['reason'], 'WORKER_RECOVERY_REQUIRED')
        self.assertEqual(self.worker.demoncat_fault_rpc('disable', lease)['status'], 'ACTION_COMPLETED')
        self.assertEqual(self.worker.demoncat_fault_rpc('execute', lease)['reason'], 'ATTEMPT_CONSUMED')


if __name__ == '__main__':
    unittest.main()
