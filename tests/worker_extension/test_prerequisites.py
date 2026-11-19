"""Read-only CPU prerequisite reports; neither names nor reports grant NPU use."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from demoncat.runtime.vllm_ascend.worker_extension.prerequisites import (
    assessment, expectation, serial_eager_assessment)
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import ContextRefused, require_serial_eager
from test_serving_context import fixture

class PrerequisitesTests(unittest.TestCase):
    def test_all_simultaneous_mismatches_with_expected_actual(self):
        runner = fixture()[0]
        runner.vllm_config.parallel_config.tensor_parallel_size = 8
        runner.use_async_scheduling = True
        runner.vllm_config.cache_config.enable_prefix_caching = True
        runner.vllm_config.model_config.enforce_eager = False
        report = serial_eager_assessment(runner)
        self.assertEqual(len(report['unmet']), 4)
        with self.assertRaises(ContextRefused) as caught:
            require_serial_eager(runner)
        self.assertEqual(caught.exception.prerequisites, report)
        self.assertEqual(report['primary']['actual'], 8)
        self.assertEqual(report['primary']['expected'], 1)
        self.assertFalse(report['execution_authorized'])

    def test_missing_nested_fields_are_all_pending_and_not_passed(self):
        report = serial_eager_assessment(NS())
        self.assertEqual(len(report['unmet']), 16)
        self.assertEqual(report['status'], 'pending')
        self.assertTrue(all(row['actual']=={'missing':True} for row in report['unmet']))

    def test_missing_pcp_is_aggregated_with_other_mismatches_not_defaulted(self):
        runner = fixture()[0]
        del runner.pcp_size
        runner.use_async_scheduling = True
        runner.is_multimodal_model = True
        with self.assertRaises(ContextRefused) as caught:
            require_serial_eager(runner)
        report = caught.exception.prerequisites
        rows = {row['field']: row for row in report['unmet']}
        self.assertEqual(set(rows), {'pcp_size', 'use_async_scheduling', 'is_multimodal_model'})
        self.assertEqual(rows['pcp_size']['expected'], 1)
        self.assertEqual(rows['pcp_size']['actual'], {'missing': True})
        self.assertEqual(rows['pcp_size']['status'], 'pending')
        self.assertFalse(hasattr(runner, 'pcp_size'))
        self.assertFalse(report['execution_authorized'])

    def test_boolean_does_not_satisfy_integer_parallel_contract(self):
        runner = fixture()[0]
        runner.vllm_config.parallel_config.tensor_parallel_size = True
        with self.assertRaises(ContextRefused): require_serial_eager(runner)

    def test_valid_runtime_is_not_an_admission_certificate(self):
        runner = fixture()[0]
        self.assertEqual(require_serial_eager(runner)['status'], 'matched')
        self.assertFalse(require_serial_eager(runner)['execution_authorized'])
        runner.vllm_config.scheduler_config.max_num_seqs = 8
        self.assertEqual(require_serial_eager(runner, expected_max_seqs=8)['unmet'], [])

    def test_deployment_identity_precedes_missing_consumer(self):
        checks = [expectation('actual_call','consumer',True,False,'NO_CONSUMER'),
                  expectation('deployment_identity','library_sha256','a'*64,'b'*64,'BUILD_MISMATCH')]
        report = assessment(checks)
        self.assertEqual(report['primary']['code'], 'BUILD_MISMATCH')
        self.assertEqual(len(report['unmet']), 2)

if __name__=='__main__': unittest.main()
