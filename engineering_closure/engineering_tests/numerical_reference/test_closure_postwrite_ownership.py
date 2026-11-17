from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import unittest

import torch

from engineering_closure.operator_adapters import OperatorAdapter
from fault_injection.coverage_session import CoverageScope
from fault_injection.deployments.postwrite import Fault39OutputProxyProvider, PostwriteRefused
from fault_injection.deployments.postwrite_coverage import ProjectionCoverage
from fault_injection.deployments.postwrite_kv import LiveKVSlotLease, live_value_tail_plan
from tests.test_postwrite_coverage import TwoLayers
from tests.test_postwrite_natural_kv import CpuSpanTransport, fixture
from tests.test_postwrite_providers import CpuLease, CpuTransport, Projection


class ParallelProjections(TwoLayers):
    def forward(self, value):
        return self.first(value) + self.second(value)


class RecordedLease(CpuLease):
    def __init__(self, tensor, transport):
        super().__init__(tensor)
        self.transport = transport
        self.applied = False

    def apply(self, mode, indices):
        count = super().apply(mode, indices)
        if mode:
            self.applied = True
            self.transport.applied += 1
            self.transport.peak = max(self.transport.peak, self.transport.applied)
        return count

    def recover(self):
        if self.applied and self.transport.fail_once:
            self.transport.fail_once = False
            raise PostwriteRefused('CPU_MOCK_DRAIN_FAILED')
        super().recover()
        if self.applied:
            self.transport.applied -= 1
            self.applied = False


class MultiLeaseTransport(CpuTransport):
    lease_capacity = 2

    def __init__(self):
        self.applied = self.peak = 0
        self.fail_once = False

    def open(self, tensor, owner, generation):
        self.validate(tensor)
        return RecordedLease(tensor, self)


class PostwriteOwnershipTests(unittest.TestCase):
    def test_real_matmul_boundary_retains_mutation_until_owned_recovery(self):
        class OwnerProbe(Projection):
            action = None

            def forward(self, value):
                output = torch.mm(value, self.weight)
                if self.action is not None:
                    self.action()
                return output + 1

        model = OwnerProbe()
        provider = Fault39OutputProxyProvider(model, CpuTransport())
        adapter = OperatorAdapter('39.output_proxy').bind_target(provider)
        value = torch.ones(2, 4, dtype=torch.bfloat16)
        self.assertEqual(adapter.discover(value)['operator'], 'aten.mm.default')
        baseline, _ = adapter.observe(value)

        def during_consumer_window():
            with self.assertRaisesRegex(PostwriteRefused, 'CONSUMER_WINDOW_ACTIVE'):
                provider.recover()
            with ThreadPoolExecutor(max_workers=1) as executor:
                with self.assertRaisesRegex(PostwriteRefused, 'SYNCHRONOUS_OWNER_REQUIRED'):
                    executor.submit(provider.recover).result(timeout=5)
            self.assertTrue(provider.query()['recovery_pending'])

        model.action = during_consumer_window
        @contextmanager
        def observer_window(actual_model):
            self.assertIs(actual_model, model)
            yield
            with self.assertRaisesRegex(PostwriteRefused, 'CONSUMER_WINDOW_ACTIVE'):
                provider.recover()
            self.assertEqual(provider.live_lease_count, 1)

        fault, receipt = adapter.inject(value, model_observation=observer_window)
        model.action = None
        self.assertEqual(int(torch.isinf(fault).sum()), 1)
        self.assertTrue(receipt['recovery_confirmed'])
        self.assertEqual(adapter.verify_effect()['leases'], 0)
        off, _ = adapter.restore(value)
        self.assertTrue(torch.equal(baseline, off))
        adapter.cleanup()

    def test_kv_lease_foreign_arm_and_recovery_cannot_touch_cache(self):
        plan = live_value_tail_plan(**fixture(), tail_elements=2)
        baseline = plan.cache.clone()
        lease = LiveKVSlotLease(CpuSpanTransport(), plan, 12, 1)
        with ThreadPoolExecutor(max_workers=1) as executor:
            with self.assertRaisesRegex(PostwriteRefused, 'LEASE_OWNER_REQUIRED'):
                executor.submit(lease.arm, fault=True).result(timeout=5)
            self.assertTrue(torch.equal(plan.cache, baseline))
            lease.arm(fault=True)
            mutated = plan.cache.clone()
            self.assertFalse(torch.equal(mutated, baseline))
            with self.assertRaisesRegex(PostwriteRefused, 'LEASE_OWNER_REQUIRED'):
                executor.submit(lease.recover).result(timeout=5)
            self.assertTrue(torch.equal(plan.cache, mutated))
            self.assertTrue(lease.live)
        consumer = lease.consume(lambda: plan.cache.clone(), (), {})
        self.assertTrue(torch.equal(consumer, mutated))
        self.assertTrue(torch.equal(plan.cache, baseline))
        self.assertFalse(lease.live)

    def test_disjoint_every_match_waits_for_both_consumers_and_preserves_c2_gate(self):
        provider, coverage, transport, value = self._group()
        with self.assertRaisesRegex(PostwriteRefused, 'LIVE_C2_PROOF_REQUIRED'):
            coverage.run('every_match', value)
        @contextmanager
        def owner_window(_):
            generation = coverage.generation
            with self.assertRaisesRegex(PostwriteRefused, 'GROUP_SYNCHRONOUS_OWNER_REQUIRED'):
                coverage.run('once', value)
            with ThreadPoolExecutor(max_workers=1) as executor:
                with self.assertRaisesRegex(PostwriteRefused, 'GROUP_SYNCHRONOUS_OWNER_REQUIRED'):
                    executor.submit(coverage.run, 'once', value).result(timeout=5)
            self.assertEqual(coverage.generation, generation)
            yield

        c2 = coverage.run('once', value, model_observation=owner_window)
        self.assertTrue(c2['contract_satisfied'], c2['rejection_reasons'])
        self.assertEqual(transport.peak, 1)
        c3 = coverage.run('every_match', value)
        self.assertTrue(c3['contract_satisfied'], c3['rejection_reasons'])
        self.assertEqual(c3['hits'], 4)
        self.assertEqual(transport.peak, 2)
        self.assertEqual(provider.live_lease_count, 0)
        self.assertTrue(all(row['consumers'] and row['native_recovered'] for row in c3['execution_artifacts']))
        self.assertFalse(c3['public_qualification_granted'])

    def test_partial_restore_failure_retains_only_failed_storage_and_can_retry(self):
        provider, coverage, transport, value = self._group()
        self.assertTrue(coverage.run('once', value)['contract_satisfied'])
        transport.fail_once = True
        failed = coverage.run('every_match', value)
        self.assertIn('CPU_MOCK_DRAIN_FAILED', failed['provider_failure'])
        self.assertFalse(failed['contract_satisfied'])
        self.assertFalse(failed['recovery']['complete'])
        self.assertEqual(provider.live_lease_count, 1)
        self.assertEqual(transport.applied, 1)
        with self.assertRaisesRegex(PostwriteRefused, 'PREVIOUS_GROUP_RECOVERY_PENDING'):
            coverage.run('every_match', value)
        self.assertTrue(coverage.retry_recovery(value)['complete'])
        self.assertEqual(provider.live_lease_count, 0)
        self.assertFalse(coverage.session.report()['contract_satisfied'])

    def _group(self):
        transport = MultiLeaseTransport()
        provider = Fault39OutputProxyProvider(ParallelProjections(), transport)
        value = torch.ones(1, 4, dtype=torch.bfloat16)
        provider.qualify(value)
        for phase in ('baseline', 'fault', 'off'):
            provider.run(phase, value)
        scope = CoverageScope(provider.mechanism, 'a' * 64, 'b' * 64, 'c' * 64, 'cpu_contract')
        return provider, ProjectionCoverage(provider, scope, max_live_leases=2), transport, value
