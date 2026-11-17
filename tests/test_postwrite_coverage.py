"""Real CPU operator coverage contracts, distinct from NPU/model qualification."""
import unittest
import uuid

import torch

from fault_injection.coverage_session import CoverageScope
from fault_injection.deployments.postwrite import PostwriteProvider, PostwriteRefused
from fault_injection.deployments.postwrite_coverage import ProjectionCoverage
from fault_injection.deployments.postwrite_evidence import digest
from tests.test_postwrite_providers import CpuTransport, Projection


class TwoLayers(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = Projection()
        self.second = Projection()
    def forward(self, x):return self.second(self.first(x))


def prepared():
    p = PostwriteProvider(TwoLayers(), CpuTransport(), '39.output_proxy')
    x = torch.ones(1, 4, dtype=torch.bfloat16)
    p.qualify(x)
    p.run('baseline', x)
    p.run('fault', x)
    p.run('off', x)
    scope = CoverageScope(p.mechanism, digest(('fixed-input', str(uuid.uuid4()))), 'b'*64, 'c'*64, 'cpu_contract')
    return p, x, scope


class ProjectionCoverageTests(unittest.TestCase):
    def test_same_shape_different_modules_all_targets_once_independent_denominator_and_off(self):
        p, x, scope = prepared()
        group = ProjectionCoverage(p, scope)
        report = group.run('once', x)
        self.assertTrue(report['contract_satisfied'], report['rejection_reasons'])
        self.assertEqual(report['planned_targets'], 2)
        self.assertEqual(report['planned_targets_hit'], 2)
        self.assertEqual(report['hits'], 2)
        self.assertEqual(report['observed_eligible_calls'], 4)
        self.assertEqual(report['completed_fault_forwards'], 2)
        reasons = [call['decision']['reason'] for call in report['calls']]
        self.assertIn('CONCURRENCY_BUDGET_EXHAUSTED', reasons)
        self.assertIn('TARGET_ALREADY_ATTEMPTED', reasons)
        self.assertEqual(sum(reason == 'OFF_RECOVERY' for reason in reasons), 2)
        self.assertTrue(report['recovery']['complete'])
        self.assertIsNotNone(group.c2_live_proof)
        self.assertEqual(p.query()['leases'], 0)
        self.assertFalse(report['public_qualification_granted'])
        self.assertFalse(report['real_worker_evidence'])
        self.assertFalse(report['real_pretrained_model_evidence'])

    def test_c3_refuses_each_match_lease_limit_without_shrinking_targets_and_recovers(self):
        p, x, scope = prepared(); group = ProjectionCoverage(p, scope)
        self.assertTrue(group.run('once', x)['contract_satisfied'])
        report = group.run('every_match', x)
        self.assertFalse(report['contract_satisfied'])
        self.assertEqual(report['planned_targets'], 2)
        self.assertIn('CONCURRENCY_BUDGET_EXHAUSTED', report['provider_failure'])
        self.assertIn('EVERY_MATCH_NOT_PROVEN', report['rejection_reasons'])
        self.assertTrue(report['recovery']['complete'])
        self.assertEqual(p.query()['leases'], 0)

    def test_c2_requires_actual_c1_effect_and_off_in_same_model_lifetime(self):
        p = PostwriteProvider(TwoLayers(), CpuTransport(), '39.output_proxy')
        x = torch.ones(1, 4, dtype=torch.bfloat16)
        p.qualify(x); p.run('baseline', x)
        scope = CoverageScope(p.mechanism, 'a'*64, 'b'*64, 'c'*64, 'cpu_contract')
        with self.assertRaisesRegex(PostwriteRefused, 'SAME_PROCESS_C1'):
            ProjectionCoverage(p, scope)

    def test_c3_requires_live_common_c2_proof(self):
        p, x, scope = prepared(); group = ProjectionCoverage(p, scope)
        with self.assertRaisesRegex(PostwriteRefused, 'LIVE_C2_PROOF'):
            group.run('every_match', x)

    def test_unexecuted_module_keeps_candidate_denominator_incomplete(self):
        p, x, scope = prepared()
        p.model.unused = Projection()
        # A fresh model binding is required before discovery can consider this
        # real but unexecuted module; do not edit an existing identity receipt.
        fresh = PostwriteProvider(p.model, CpuTransport(), '39.output_proxy')
        fresh.qualify(x); fresh.run('baseline', x); fresh.run('fault', x); fresh.run('off', x)
        group = ProjectionCoverage(fresh, scope)
        with self.assertRaisesRegex(PostwriteRefused, 'DISCOVERY_INCOMPLETE'):
            group.run('once', x)
        self.assertEqual(fresh.query()['leases'], 0)

    def test_changed_input_or_weights_refused_before_group_mutation(self):
        p, x, scope = prepared(); group = ProjectionCoverage(p, scope)
        with self.assertRaisesRegex(PostwriteRefused, 'FIXED_WORKLOAD'):
            group.run('once', x + 1)
        with torch.no_grad():p.model.first.weight.add_(1)
        with self.assertRaisesRegex(PostwriteRefused, 'FIXED_WORKLOAD'):
            group.run('once', x)


if __name__ == '__main__':unittest.main()
