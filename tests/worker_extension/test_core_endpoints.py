"""Routing tests through real Extension/Runtime; no NPU qualification."""

import unittest
from unittest.mock import patch

from demoncat.runtime.vllm_ascend.worker_extension import identity, runtime
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.endpoints import WorkerEndpointSet
from demoncat.runtime.vllm_ascend.worker_extension.host import HostContractError

try:
    from . import test_extension as fixtures
    from .test_core_host import ExistingExecutor
except ImportError:
    import test_extension as fixtures
    from test_core_host import ExistingExecutor


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.workers = [fixtures._extended_worker(rank=rank) for rank in (7, 2)]
        for worker in self.workers:
            worker.load_model()
        self.executor = ExistingExecutor(self.workers)
        self.providers = {}
        device = patch.object(identity, '_current_device', return_value=0)
        device.start()
        self.addCleanup(device.stop)
        resolve = patch.object(runtime, '_resolve_provider', side_effect=self.resolve)
        resolve.start()
        self.addCleanup(resolve.stop)
        self.config = FaultConfig(20, 'buffer_substitute')

    def resolve(self, mechanism, version, worker, model, worker_identity):
        return self.providers.setdefault(worker.rank, fixtures.ReadbackProvider())

    def endpoints(self):
        return WorkerEndpointSet(self.executor, expected_ranks=[2, 7])

    def checked(self, client):
        for name in ('prepare', 'register', 'check'):
            result = getattr(client, name)()
            self.assertEqual(result['status'], 'ACTION_COMPLETED', result)

    def test_explicit_set_pins_real_identity_and_per_rank_leases(self):
        endpoints = self.endpoints()
        self.assertEqual([e.rank for e in endpoints.endpoints], [2, 7])
        self.assertEqual([c[0] for c in self.executor.calls], ['demoncat_worker_identity'])
        detached_copy = endpoints.endpoints[0].identity
        detached_copy['pid'] = -1
        self.assertNotEqual(endpoints.endpoints[0].identity['pid'], -1)
        clients = [endpoints.client_for(rank, self.config) for rank in (2, 7)]
        self.workers.reverse()
        for client in clients:
            self.checked(client)
        self.assertNotEqual(clients[0].owner, clients[1].owner)
        self.assertNotEqual(clients[0].tenant_id, clients[1].tenant_id)
        for rank, client in zip((2, 7), clients):
            self.assertEqual(client.snapshot()['ticket']['worker']['rank'], rank)
            self.assertEqual(client.execute()['status'], 'ACTION_COMPLETED')
        self.assertTrue(endpoints.snapshot()['ownership_retained'])
        self.assertEqual(clients[0].detach()['status'], 'DETACHED')
        self.assertTrue(endpoints.snapshot()['ownership_retained'])
        self.assertEqual(clients[1].detach()['status'], 'DETACHED')
        self.assertFalse(endpoints.snapshot()['ownership_retained'])
        self.assertEqual([worker.events for worker in self.workers], [['load'], ['load']])

    def test_missing_extra_duplicate_and_invalid_ranks_fail_without_attach(self):
        for expected in ([2], [2, 7, 9], [], [True, 7], [2, 2]):
            with self.subTest(expected=expected), self.assertRaises((ValueError, HostContractError)):
                WorkerEndpointSet(self.executor, expected_ranks=expected)
        duplicate = fixtures._extended_worker(rank=2)
        duplicate.load_model()
        self.workers.append(duplicate)
        with self.assertRaisesRegex(HostContractError, 'AMBIGUOUS_BROADCAST_RANK'):
            self.endpoints()
        self.assertFalse(self.providers)
        self.assertNotIn('demoncat_attach', [c[0] for c in self.executor.calls])

    def test_restart_before_attach_rejects_pinned_identity(self):
        endpoints = self.endpoints()
        client = endpoints.client_for(2, self.config)
        replacement = fixtures._extended_worker(rank=2)
        replacement.load_model()
        self.workers[1] = replacement
        result = client.prepare()
        self.assertEqual(result['response']['reason'], 'WORKER_ENDPOINT_SET_CHANGED')
        self.assertFalse(self.providers)
        self.assertNotIn('demoncat_attach', [c[0] for c in self.executor.calls])

    def test_client_access_never_replaces_or_retries_uncertain_rank(self):
        endpoints = self.endpoints()
        a = endpoints.client_for(2, self.config)
        b = endpoints.client_for(7, self.config)
        self.checked(a)
        self.checked(b)
        ticket = a.snapshot()['ticket']

        def lose_enable(method, args, replies):
            if method == 'demoncat_fault_rpc' and args[0] == 'enable':
                self.executor.after = None
                raise TimeoutError('lost response after enable')
            return replies

        self.executor.after = lose_enable
        self.assertEqual(a.enable()['status'], 'RPC_UNCERTAIN')
        self.assertIs(endpoints.client_for(2, self.config), a)
        self.assertEqual(a.snapshot()['ticket'], ticket)
        self.assertEqual(a.enable()['status'], 'HOST_REJECTED')
        self.assertEqual(b.enable()['status'], 'ACTION_COMPLETED')
        per_rank = {r['rank']: r['client'] for r in endpoints.snapshot()['endpoints']}
        self.assertEqual(per_rank[2]['assessments']['Recovery']['status'], 'REQUIRED')
        self.assertTrue(per_rank[7]['ownership_retained'])
        self.assertEqual(a.detach()['status'], 'DETACHED')
        self.assertTrue(endpoints.snapshot()['ownership_retained'])
        self.assertEqual(b.detach()['status'], 'DETACHED')
        self.assertFalse(endpoints.snapshot()['ownership_retained'])
        self.assertEqual([n for n, *_ in self.providers[2].actions].count('enable'), 1)

    def test_rank_and_config_changes_cannot_rebind_existing_client(self):
        endpoints = self.endpoints()
        a = endpoints.client_for(2, self.config)
        with self.assertRaisesRegex(HostContractError, 'NOT_SELECTED'):
            endpoints.client_for(0, self.config)
        with self.assertRaisesRegex(HostContractError, 'ALREADY_BOUND'):
            endpoints.client_for(2, FaultConfig(39, 'matmul_range'))
        self.assertIs(endpoints.client_for(2, self.config.as_dict()), a)
        self.assertFalse(self.providers)


if __name__ == '__main__':
    unittest.main()
