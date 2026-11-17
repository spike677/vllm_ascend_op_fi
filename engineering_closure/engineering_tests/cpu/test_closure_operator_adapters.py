import unittest

from engineering_closure.operator_adapters import OPERATOR_PATHS, OperatorAdapter
from engineering_closure.queue import prepare_cases
from fault_injection.core import UnsupportedCall
from fault_injection.coverage import Candidate, Eligibility, EligibilityResult, TargetIdentity
from fault_injection.coverage_session import IndependentCallObserver
from fault_injection.lifecycle import FaultConfig, ModelFaultBackend
from fault_injection.runtime_discovery import RuntimeCandidateInventory


class OperatorRegistryTests(unittest.TestCase):
    def test_case_factory_forwards_reviewed_lease_limit_to_fault_only(self):
        options = dict(source='/reviewed/source', native='/reviewed/native', sha256='a' * 64,
            build_fingerprint='b' * 64, model='/reviewed/model')
        baseline, fault = prepare_cases('39.output_proxy', dict(options, max_live_leases=140))
        self.assertNotIn('--max-live-leases', baseline['command'])
        self.assertIn('--try-c3 --max-live-leases 140', fault['command'])
        self.assertEqual(fault['review']['c3_max_live_leases'], 140)
        self.assertNotIn('--max-live-leases', prepare_cases('39.output_proxy', options)[1]['command'])
        with self.assertRaisesRegex(ValueError, 'BOUNDED_LIVE_LEASE_COUNT'):
            prepare_cases('39.output_proxy', dict(options, max_live_leases=257))

    def test_independent_discovery_keeps_unseen_calls_unresolved(self):
        target = TargetIdentity(5, 'worker', 0, 0, 'model', 6, 7, ('LINEAR',),
                                'aclnnMatmul', 'cache', 'executor', 'live', 'bf16', 'a' * 64)
        candidate = Candidate('projection', EligibilityResult(Eligibility.SUPPORTED), target)
        adapter = OperatorAdapter('39.matmul_range')
        for modules in ((7,), (7, 8)):
            with self.subTest(modules=modules):
                inventory = RuntimeCandidateInventory(expected_modules=modules, expected_workers=(target.worker,))
                observer = IndependentCallObserver(source='independent_native_trace', observer_id='trace',
                    implementation_sha256='b' * 64, workload_sha256='c' * 64, max_calls=8)
                for forward in ('first', 'second'):
                    inventory.observe(candidate, forward_id=forward)
                    inventory.finish_forward(forward, healthy=True)
                candidates = adapter.discover(inventory, observer, artifact_sha256='d' * 64)
                if modules == (7,):
                    self.assertEqual(adapter.identify(candidates), (target,))
                    metadata = adapter.inspect_metadata(target)
                    self.assertEqual(metadata['target']['operator_api'], 'aclnnMatmul')
                    self.assertFalse(metadata['execution_authorized'])
                else:
                    with self.assertRaisesRegex(UnsupportedCall, 'incomplete'):
                        adapter.identify(candidates, 'all_eligible')

    def test_distinct_routes_and_mismatched_model_config_are_refused(self):
        self.assertEqual(len(OPERATOR_PATHS), 17)
        self.assertNotEqual(OPERATOR_PATHS['28.fa.legal_tiling'], OPERATOR_PATHS['28.fia.legal_tiling'])
        self.assertNotEqual(OPERATOR_PATHS['39.matmul_range'], OPERATOR_PATHS['39.output_proxy'])
        model = type('Model', (), {'named_modules': lambda self: ()})()
        backend = ModelFaultBackend(model)
        backend.register(FaultConfig('17.spatial_padding'))
        with self.assertRaisesRegex(UnsupportedCall, 'CONFIG_DOES_NOT_MATCH'):
            OperatorAdapter('18.api_half').bind_target(backend)
        bound = OperatorAdapter('17.spatial_padding').bind_target(backend)
        with self.assertRaisesRegex(UnsupportedCall, 'READ_ONLY_OBSERVATION'):
            bound.observe()
        bound.cleanup()
        with self.assertRaisesRegex(UnsupportedCall, 'CONFIG_CHANGED'):
            bound.inject()
