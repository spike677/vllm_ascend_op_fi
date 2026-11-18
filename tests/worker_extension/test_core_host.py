"""CPU contracts against the real WorkerFaultRuntime, without native loading."""

from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

from demoncat.runtime.vllm_ascend.worker_extension import identity, runtime
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient

try:
    from . import test_extension as fixtures
except ImportError:
    import test_extension as fixtures


class ExistingExecutor:
    """Deliver through the extension exactly as collective_rpc does."""

    def __init__(self, workers):
        self.workers = workers
        self.calls = []
        self.after = None

    def collective_rpc(self, method, timeout=None, args=()):
        self.calls.append((method, deepcopy(args)))
        replies = [getattr(worker, method)(*deepcopy(args)) for worker in self.workers]
        return self.after(method, args, replies) if self.after else list(reversed(replies))


class CoreHostTests(unittest.TestCase):
    def setUp(self):
        self.worker = fixtures._extended_worker(rank=7)
        self.worker.load_model()
        self.workers = [self.worker]
        self.providers = {}
        self.executor = ExistingExecutor(self.workers)
        device = patch.object(identity, '_current_device', return_value=0)
        device.start()
        self.addCleanup(device.stop)
        resolver = patch.object(runtime, '_resolve_provider', side_effect=self._resolve)
        self.resolver = resolver.start()
        self.addCleanup(resolver.stop)

    def _resolve(self, mechanism, version, worker, model, worker_identity):
        self.assertIs(model, worker.model_runner.model)
        self.assertEqual(worker_identity['rank'], worker.rank)
        self.assertEqual(worker_identity['configured_logical_device'], worker.device.index)
        return self.providers.setdefault((worker, mechanism), fixtures.ReadbackProvider())

    def client(self, fault_id=20, variant='buffer_substitute', **host_options):
        return WorkerFaultClient(HostAdapter(self.executor, **host_options),
                                 FaultConfig(fault_id, variant))

    def checked(self, client):
        for action in ('prepare', 'register', 'check'):
            row = getattr(client, action)()
            self.assertEqual(row['status'], 'ACTION_COMPLETED', row)
        return row

    def actions(self, worker=None, mechanism='20.buffer_substitute'):
        provider = self.providers.get((worker or self.worker, mechanism))
        return [] if provider is None else [action[0] for action in provider.actions]

    def fail_once_after(self, action):
        def lose_reply(method, args, replies):
            if method == 'demoncat_fault_rpc' and args[0] == action:
                self.executor.after = None
                raise TimeoutError('response lost after Worker applied action')
            return replies
        self.executor.after = lose_reply

    def test_batch_and_decode_controls_reuse_existing_owned_advance(self):
        client = self.client()
        self.checked(client)
        self.assertEqual(client.execute()['status'], 'ACTION_COMPLETED')
        ids = ['actual-' + str(i) for i in range(8)]
        self.assertEqual(client.advance_batch(ids)['status'], 'ACTION_COMPLETED')
        self.assertEqual(client.begin_decode_capture()['status'], 'ACTION_COMPLETED')
        self.assertEqual(client.collect_replay()['status'], 'ACTION_COMPLETED')
        calls = [args[-1]['settings'] for method, args in self.executor.calls
                 if method == 'demoncat_fault_rpc' and args[0] == 'advance']
        self.assertEqual(calls, [{'request_ids': ids}, {'capture_decode': True}, {'collect': True}])
        self.assertEqual(client.detach()['status'], 'DETACHED')

    def test_invalid_batch_mapping_is_rejected_without_rpc(self):
        client = self.client()
        for value in ([], ['same', 'same'], [''], ['x', 1], 'not-a-list', list(map(str, range(257)))):
            with self.subTest(value=str(value)[:20]), self.assertRaises(ValueError):
                client.advance_batch(value)
        self.assertEqual(self.executor.calls, [])
        self.assertEqual(client.begin_decode_capture()['response']['reason'],
                         'LEASE_TICKET_REQUIRED_USE_DETACH_FOR_UNCERTAIN_PREPARE')

    def test_config_selects_existing_catalog_and_keeps_legacy_fault39_contract(self):
        from fault_injection.catalog import CAPABILITIES
        from fault_injection.lifecycle import FaultConfig as ModelFaultConfig

        for capability in CAPABILITIES:
            family, variant = capability.name.split('.', 1)
            config = FaultConfig(int(family) if family.isdecimal() else family, variant)
            self.assertEqual(config.mechanism, capability.name)
        with self.assertRaisesRegex(ValueError, 'separately validated'):
            ModelFaultConfig('39.matmul_range')
        self.assertEqual(FaultConfig(39, 'matmul_range').mechanism, '39.matmul_range')
        for key in ('layer_name', 'rank', 'device', 'operator', 'tensor_address', 'settings',
                    'native_library', 'build_profile', 'target', 'auto_retry'):
            with self.subTest(key=key), self.assertRaises(TypeError):
                FaultConfig.from_dict(dict(fault_id=20, variant='buffer_substitute', **{key: 0}))
        for fault_id in (True, '20', 0, -1):
            with self.subTest(fault_id=fault_id), self.assertRaises(ValueError):
                FaultConfig(fault_id, 'buffer_substitute')

    def test_default_uses_actual_identity_and_preserves_runtime_ticket(self):
        client = self.client()
        checked = self.checked(client)
        self.assertEqual(checked['worker']['rank'], 7)
        self.assertEqual(checked['worker']['logical_device'], 0)
        self.assertEqual(checked['ticket']['worker'], checked['worker'])
        self.assertEqual(checked['ticket']['owner'], client.owner)
        self.assertEqual(checked['provider_check'], 'COMPLETED')
        self.assertEqual(client.enable()['status'], 'ACTION_COMPLETED')
        self.assertEqual(client.finish()['status'], 'ACTION_COMPLETED')
        self.assertEqual(client.disable()['status'], 'ACTION_COMPLETED')
        self.assertEqual(client.unregister()['status'], 'ACTION_COMPLETED')
        detached = client.detach()
        self.assertEqual(detached['status'], 'DETACHED')
        self.assertFalse(detached['ownership_retained'])
        self.assertEqual(detached['assessments']['Recovery']['status'], 'LEASE_RELEASE_CONFIRMED')
        self.assertEqual(self.worker.events, ['load'])
        self.assertEqual(self.actions(), ['prepare', 'register', 'check', 'enable',
                                         'finish', 'disable', 'unregister'])

    def test_multiple_workers_require_host_policy_and_never_default_to_rank_zero(self):
        second = fixtures._extended_worker(rank=0)
        second.load_model()
        self.workers.append(second)
        row = self.client().prepare()
        self.assertEqual(row['response']['reason'], 'TRUSTED_WORKER_ROUTING_REQUIRED')
        self.resolver.assert_not_called()
        self.assertEqual([call[0] for call in self.executor.calls], ['demoncat_worker_identity'])

    def test_host_selects_complete_identity_and_broadcast_never_crosses_workers(self):
        other = fixtures._extended_worker(rank=2)
        other.load_model()
        self.workers.insert(0, other)
        # Deployment policy chooses an actual discovered identity. Client
        # configuration contains no rank, layer, device or address selector.
        chosen = self.worker.demoncat_worker_identity()['worker']
        selector = Mock(side_effect=lambda config, workers: next(worker for worker in workers if worker == chosen))
        client = self.client(select_worker=selector)
        self.checked(client)
        original_ticket = client.snapshot()['ticket']
        self.workers.reverse()  # Response ordering is not a route identifier.
        self.assertEqual(client.execute()['status'], 'ACTION_COMPLETED')
        self.assertEqual(client.detach()['status'], 'DETACHED')
        self.assertEqual(self.actions(other), [])
        self.assertEqual(self.actions(), ['prepare', 'register', 'check', 'execute', 'unregister'])
        self.assertEqual(selector.call_count, 1)
        for method, args in self.executor.calls:
            if method == 'demoncat_fault_rpc' and args[0] != 'prepare':
                self.assertEqual(args[-1], original_ticket)

    def test_same_rank_broadcast_is_rejected_before_any_mutation(self):
        duplicate = fixtures._extended_worker(rank=self.worker.rank)
        duplicate.load_model()
        self.workers.append(duplicate)
        row = self.client(select_worker=lambda config, workers: workers[0]).prepare()
        self.assertEqual(row['response']['reason'], 'AMBIGUOUS_BROADCAST_RANK')
        self.resolver.assert_not_called()
        self.assertNotIn('demoncat_attach', [call[0] for call in self.executor.calls])

    def test_host_policy_cannot_forge_an_unobserved_identity(self):
        row = self.client(select_worker=lambda config, workers: dict(workers[0], pid=-1)).prepare()
        self.assertEqual(row['response']['reason'], 'SELECTED_WORKER_NOT_DISCOVERED')
        self.resolver.assert_not_called()

    def test_incomplete_worker_context_is_not_selected(self):
        self.worker.model_runner = None
        row = self.client().prepare()
        self.assertEqual(row['response']['reason'], 'WORKER_CONTEXT_NOT_READY')
        self.resolver.assert_not_called()

    def test_capability_listing_is_not_provider_or_coverage_qualification(self):
        client = self.client()
        row = client.attach()
        self.assertEqual(row['status'], 'ATTACHED')
        self.assertTrue(row['capability_enumeration']['providers'])
        self.assertEqual(row['provider_check'], 'NOT_ASSESSED')
        self.assertEqual({key: row['assessments'][key] for key in ('C2', 'C3', 'E5')},
                         {'C2': 'NOT_ASSESSED', 'C3': 'NOT_ASSESSED', 'E5': 'NOT_ASSESSED'})
        self.resolver.assert_not_called()
        self.assertEqual(client.detach()['status'], 'DETACHED')

    def test_unadapted_provider_version_refuses_without_native_resolution(self):
        client = WorkerFaultClient(HostAdapter(self.executor),
                                   FaultConfig(28, 'fa.legal_tiling', 'unsupported'))
        row = client.prepare()
        self.assertEqual(row['response']['reason'], 'WORKER_PROVIDER_NOT_ADAPTED')
        self.resolver.assert_not_called()
        self.assertTrue(row['ownership_retained'])
        self.assertEqual(client.detach()['status'], 'DETACHED')

    def test_trusted_binding_supplies_fault20_actual_operator_without_user_fields(self):
        binding = Mock(return_value={'prepare': {'operator': 'aclnnQuantMatmulWeightNz'}})
        client = self.client(prepare_binding=binding)
        row = client.prepare()
        self.assertEqual(row['status'], 'ACTION_COMPLETED')
        binding.assert_called_once_with(row['worker'], client.config)
        provider = self.providers[(self.worker, client.config.mechanism)]
        self.assertEqual(provider.actions[0][1], {'operator': 'aclnnQuantMatmulWeightNz'})
        self.assertEqual(client.config.as_dict(),
                         dict(fault_id=20, variant='buffer_substitute', implementation_version='1'))

    def test_fault39_uses_same_client_but_keeps_distinct_provider(self):
        client = self.client(39, 'matmul_range')
        self.checked(client)
        self.assertEqual(client.execute()['status'], 'ACTION_COMPLETED')
        self.assertEqual(client.detach()['status'], 'DETACHED')
        self.assertEqual(self.actions(), [])
        self.assertEqual(self.actions(mechanism='39.matmul_range'),
                         ['prepare', 'register', 'check', 'execute', 'unregister'])

    def test_lost_prepare_reply_preserves_owner_and_detach_recovers_same_tenant(self):
        client = self.client()
        self.fail_once_after('prepare')
        uncertain = client.prepare()
        self.assertEqual(uncertain['status'], 'RPC_UNCERTAIN')
        self.assertIsNone(uncertain['ticket'])
        self.assertIsNotNone(uncertain['attachment'])
        self.assertEqual(uncertain['assessments']['Recovery']['status'], 'REQUIRED')
        self.assertEqual(client.prepare()['status'], 'HOST_REJECTED')
        self.assertEqual(client.enable()['status'], 'HOST_REJECTED')
        self.assertEqual(self.actions(), ['prepare'])
        detached = client.detach()
        self.assertEqual(detached['status'], 'DETACHED')
        self.assertEqual(detached['response']['recoveries'][0]['ticket']['owner'], client.owner)
        self.assertFalse(detached['ownership_retained'])

    def test_lost_attach_reply_does_not_invent_nonce_or_retry_attachment(self):
        def lose_attach(method, args, replies):
            if method == 'demoncat_attach':
                raise TimeoutError('attachment reply lost')
            return replies
        self.executor.after = lose_attach
        client = self.client()
        uncertain = client.prepare()
        self.assertEqual(uncertain['status'], 'RPC_UNCERTAIN')
        self.assertIsNone(uncertain['attachment'])
        self.assertEqual(client.attach()['status'], 'HOST_REJECTED')
        self.assertEqual(client.detach()['status'], 'HOST_REJECTED')
        self.assertEqual(client.snapshot()['assessments']['Recovery']['status'], 'REQUIRED')
        self.assertEqual(len(self.worker._demoncat_worker_fault_runtime._tenants), 1)
        self.assertEqual(sum(method == 'demoncat_attach' for method, _ in self.executor.calls), 1)
        self.resolver.assert_not_called()

    def test_lost_enable_reply_never_retries_and_keeps_exact_generation(self):
        client = self.client()
        checked = self.checked(client)
        self.fail_once_after('enable')
        uncertain = client.enable()
        self.assertEqual(uncertain['status'], 'RPC_UNCERTAIN')
        self.assertEqual(uncertain['ticket'], checked['ticket'])
        self.assertEqual(client.enable()['response']['reason'], 'RECOVERY_REQUIRED')
        self.assertEqual(self.actions().count('enable'), 1)
        self.assertEqual(client.detach()['status'], 'DETACHED')

    def test_unexpected_broadcast_response_keeps_applied_action_owned(self):
        client = self.client()
        self.checked(client)
        def extra_response(method, args, replies):
            if method == 'demoncat_fault_rpc' and args[0] == 'execute':
                return replies + [{'status': 'REJECTED', 'reason': 'WORKER_IDENTITY_MISMATCH'}]
            return replies
        self.executor.after = extra_response
        row = client.execute()
        self.assertEqual(row['response']['reason'], 'RPC_TARGET_NOT_UNIQUE')
        self.assertEqual(len(row['response']['observed_response']), 2)
        self.assertEqual(row['assessments']['Recovery']['status'], 'REQUIRED')
        self.assertTrue(row['ownership_retained'])
        self.assertEqual(client.execute()['status'], 'HOST_REJECTED')
        self.assertEqual(self.actions().count('execute'), 1)

    def test_failed_recovery_preserves_owner_and_does_not_drain_foreign_tenant(self):
        client, other = self.client(), self.client(39, 'matmul_range')
        self.checked(client)
        self.checked(other)
        provider = self.providers[(self.worker, client.config.mechanism)]
        provider.callbacks['unregister'] = Mock(side_effect=RuntimeError('drain incomplete'))
        failed = client.unregister()
        owner_ticket = failed['ticket']
        self.assertEqual(failed['status'], 'PROVIDER_ACTION_FAILED')
        self.assertEqual(failed['assessments']['Recovery']['status'], 'REQUIRED')
        detached = client.detach()
        self.assertEqual(detached['status'], 'RECOVERY_REQUIRED')
        self.assertEqual(detached['ticket'], owner_ticket)
        self.assertTrue(detached['ownership_retained'])
        self.assertEqual(self.actions(mechanism='39.matmul_range'), ['prepare', 'register', 'check'])
        provider.callbacks.clear()
        self.assertEqual(client.detach()['status'], 'DETACHED')
        self.assertEqual(other.query()['status'], 'QUERY_OK')
        self.assertEqual(other.detach()['status'], 'DETACHED')

    def test_replaced_model_keeps_old_identity_and_never_routes_to_new_generation(self):
        client = self.client()
        before = self.checked(client)
        self.worker.model_runner.model = fixtures.Model()
        row = client.query()
        self.assertEqual(row['response']['reason'], 'WORKER_IDENTITY_MISMATCH')
        self.assertEqual(row['worker'], before['worker'])
        self.assertEqual(row['ticket'], before['ticket'])
        self.assertEqual(row['assessments']['Recovery']['status'], 'REQUIRED')
        self.assertEqual(client.disable()['status'], 'REJECTED')
        self.assertEqual(self.actions(), ['prepare', 'register', 'check'])

    def test_receipt_mutation_cannot_change_authoritative_ticket(self):
        client = self.client()
        row = self.checked(client)
        original = deepcopy(row['ticket'])
        row['ticket']['generation'] += 100
        row['ticket']['owner'] = 'foreign-owner'
        row['ticket']['worker']['rank'] = 0
        queried = client.query()
        self.assertEqual(queried['status'], 'QUERY_OK')
        self.assertEqual(queried['ticket'], original)

    def test_forged_returned_generation_is_not_adopted(self):
        client = self.client()
        before = self.checked(client)
        def wrong_ticket(method, args, replies):
            if method == 'demoncat_fault_query':
                replies[0]['ticket']['generation'] += 1
            return replies
        self.executor.after = wrong_ticket
        row = client.query()
        self.assertEqual(row['response']['reason'], 'LEASE_TICKET_MISMATCH')
        self.assertEqual(row['ticket'], before['ticket'])
        self.assertEqual(row['assessments']['Recovery']['status'], 'REQUIRED')
        self.assertEqual(row['response']['observed_response']['ticket']['generation'],
                         before['ticket']['generation'] + 1)

    def test_failed_prepare_returns_owned_ticket_and_explicit_detach_can_recover(self):
        provider = fixtures.ReadbackProvider()
        provider.callbacks['prepare'] = Mock(side_effect=RuntimeError('partial prepare'))
        self.providers[(self.worker, '20.buffer_substitute')] = provider
        client = self.client()
        row = client.prepare()
        self.assertEqual(row['status'], 'PROVIDER_ACTION_FAILED')
        self.assertEqual(row['ticket']['owner'], client.owner)
        self.assertEqual(row['assessments']['Recovery']['status'], 'REQUIRED')
        self.assertEqual(client.register()['response']['reason'], 'RECOVERY_REQUIRED')
        self.assertEqual(client.detach()['status'], 'DETACHED')

    def test_lost_prepare_and_failed_detach_recovers_ticket_without_releasing_owner(self):
        client = self.client()
        self.fail_once_after('prepare')
        self.assertIsNone(client.prepare()['ticket'])
        provider = self.providers[(self.worker, client.config.mechanism)]
        provider.callbacks['unregister'] = Mock(side_effect=RuntimeError('busy callback'))
        row = client.detach()
        self.assertEqual(row['status'], 'RECOVERY_REQUIRED')
        self.assertEqual(row['ticket']['owner'], client.owner)
        self.assertEqual(row['assessments']['Recovery']['status'], 'REQUIRED')
        provider.callbacks.clear()
        self.assertEqual(client.unregister()['status'], 'ACTION_COMPLETED')
        self.assertEqual(client.detach()['status'], 'DETACHED')

    def test_detach_missing_known_recovery_never_releases_owner(self):
        client = self.client()
        original = self.checked(client)['ticket']
        def hide_recovery(method, args, replies):
            if method == 'demoncat_detach':
                replies[0]['recoveries'] = []
            return replies
        self.executor.after = hide_recovery
        row = client.detach()
        self.assertEqual(row['status'], 'RPC_UNCERTAIN')
        self.assertEqual(row['response']['reason'], 'DETACH_RECOVERY_MISSING')
        self.assertEqual(row['ticket'], original)
        self.assertTrue(row['ownership_retained'])

    def test_conflicting_success_and_recovery_failure_never_releases_owner(self):
        client = self.client()
        original = self.checked(client)['ticket']
        def conflict(method, args, replies):
            if method == 'demoncat_fault_rpc' and args[0] == 'unregister':
                replies[0]['recovery_required'] = True
            return replies
        self.executor.after = conflict
        row = client.unregister()
        self.assertEqual(row['status'], 'RPC_UNCERTAIN')
        self.assertEqual(row['response']['reason'], 'CONTRADICTORY_ACTION_RECOVERY')
        self.assertEqual(row['ticket'], original)
        self.assertEqual(row['assessments']['Recovery']['status'], 'REQUIRED')

    def test_missing_optional_discovery_does_not_claim_qualification(self):
        client = self.client()
        invoke = client.host.invoke
        def old_extension(worker, method, *args):
            if method == 'demoncat_discover':
                raise AttributeError('older extension has no optional discovery method')
            return invoke(worker, method, *args)
        with patch.object(client.host, 'invoke', side_effect=old_extension):
            row = client.discover()
        self.assertEqual(row['status'], 'HOST_REJECTED')
        self.assertEqual(row['response']['error_type'], 'AttributeError')
        self.assertEqual(row['provider_check'], 'NOT_ASSESSED')
        self.resolver.assert_not_called()
        self.assertEqual(client.detach()['status'], 'DETACHED')

    def test_discovery_receipt_is_bound_to_actual_worker_and_is_only_c0(self):
        client = self.client()
        invoke = client.host.invoke
        def discovery(worker, method, *args):
            if method == 'demoncat_discover':
                return dict(status='DISCOVERY_SNAPSHOT', discovery=dict(
                    worker=deepcopy(worker), read_only=True, qualification_granted=False,
                    candidates=[{'name': 'observed-structure', 'eligible': False}]))
            return invoke(worker, method, *args)
        with patch.object(client.host, 'invoke', side_effect=discovery):
            row = client.discover()
        self.assertEqual(row['status'], 'DISCOVERY_SNAPSHOT')
        self.assertEqual(row['provider_check'], 'NOT_ASSESSED')
        self.assertEqual(row['assessments']['C2'], 'NOT_ASSESSED')
        self.resolver.assert_not_called()
        self.assertEqual(client.prepare()['status'], 'ACTION_COMPLETED')

    def test_discovery_does_not_accept_a_foreign_worker_receipt(self):
        client = self.client()
        invoke = client.host.invoke
        def wrong_worker(worker, method, *args):
            if method == 'demoncat_discover':
                return dict(status='DISCOVERY_SNAPSHOT', discovery=dict(
                    worker=dict(worker, model_instance_id='foreign-model'),
                    read_only=True, qualification_granted=False))
            return invoke(worker, method, *args)
        with patch.object(client.host, 'invoke', side_effect=wrong_worker):
            row = client.discover()
        self.assertEqual(row['response']['reason'], 'DISCOVERY_SCOPE_MISMATCH')
        self.assertEqual(row['provider_check'], 'NOT_ASSESSED')
        self.resolver.assert_not_called()

    def test_resource_release_does_not_promote_coverage_health_or_e5(self):
        client = self.client()
        self.checked(client)
        provider = self.providers[(self.worker, client.config.mechanism)]
        original = provider.dispatch
        def evidence(action, settings, **scope):
            return dict(original(action, settings, **scope), C2='PASS', C3='PASS', E5='PASS',
                        model_output_changed=True, off_model_output_verified=True)
        provider.dispatch = evidence
        self.assertEqual(client.enable()['status'], 'ACTION_COMPLETED')
        self.assertEqual(client.disable()['assessments']['Recovery']['status'], 'NOT_ASSESSED')
        row = client.unregister()
        self.assertEqual(row['response']['provider']['E5'], 'PASS')
        self.assertEqual(row['assessments'], {
            'C2': 'NOT_ASSESSED', 'C3': 'NOT_ASSESSED', 'E5': 'NOT_ASSESSED',
            'Recovery': {'status': 'LEASE_RELEASE_CONFIRMED', 'healthy_output': 'NOT_ASSESSED'}})

    def test_binding_failure_remains_explicit_and_is_not_retried(self):
        binding = Mock(side_effect=RuntimeError('reviewed deployment unavailable'))
        client = self.client(prepare_binding=binding)
        row = client.prepare()
        self.assertEqual(row['assessments']['Recovery']['status'], 'REQUIRED')
        self.assertEqual(client.prepare()['status'], 'HOST_REJECTED')
        binding.assert_called_once()
        self.resolver.assert_not_called()
        self.assertEqual(client.detach()['status'], 'DETACHED')

    def test_advance_forwards_only_actual_request_id_and_recovery_is_explicit(self):
        client = self.client()
        self.checked(client)
        self.assertEqual(client.enable()['status'], 'ACTION_COMPLETED')
        self.assertEqual(client.advance('scheduled-request-31')['status'], 'ACTION_COMPLETED')
        self.assertEqual(client.recover()['status'], 'ACTION_COMPLETED')
        provider = self.providers[(self.worker, client.config.mechanism)]
        self.assertEqual(provider.actions[-2][1], {'request_id': 'scheduled-request-31'})
        self.assertEqual(provider.actions[-1][0], 'recover')
        self.assertEqual(client.finish()['status'], 'ACTION_COMPLETED')


if __name__ == '__main__':
    unittest.main()
