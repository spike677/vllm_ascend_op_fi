import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from experiments.c2c3_closure_20261117 import providers_quant_c1_probe as probe
from fault_injection import quant_group_diagnostics as diagnostic


def module(name, **fields):
    value = ModuleType(name)
    vars(value).update(fields)
    return value


class QuantC1InitializationDiagnosticTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.evidence = Path(directory.name)
        self.enterContext(patch.object(probe, 'EVIDENCE', self.evidence))
        self.llm = Mock(side_effect=RuntimeError('aclnnDynamicQuantV2 failed with 161001'))
        self.client = Mock()
        prefix = 'demoncat.runtime.vllm_ascend.worker_extension.'
        self.enterContext(patch.dict(sys.modules, {
            'vllm': module('vllm', LLM=self.llm, SamplingParams=Mock()),
            prefix + 'config': module(prefix + 'config', FaultConfig=Mock()),
            prefix + 'host': module(prefix + 'host', HostAdapter=Mock(), WorkerFaultClient=self.client),
            prefix + 'serving_context': module(prefix + 'serving_context', require_host_driven_core=Mock(),
                                               actual_driver_request_map=Mock()),
        }))
        self.enterContext(patch.object(sys, 'argv', ['probe.py', '--build-sha256', 'a' * 64, '--group-abi']))
        self.capture = self.enterContext(patch.object(diagnostic, 'capture_loaded_quant_state',
            return_value=dict(status='READ_ONLY_SNAPSHOT_COMPLETE', process_id=os.getpid(), native_thread_id=123,
                snapshot_complete=True, records=[], read_errors=[], qualification_authority=False)))

    def test_constructor_failure_captures_same_process_before_any_binding_or_cleanup(self):
        def captured(*args, **kwargs):
            checkpoint = json.loads((self.evidence / 'result.json').read_text())
            self.assertEqual(checkpoint['failure_stage'], 'LLM_CONSTRUCTION')
            self.assertIn('161001', checkpoint['error'])
            self.client.assert_not_called()
            return self.capture.return_value
        self.capture.side_effect = captured
        self.assertEqual(probe.main(), 2)
        report = json.loads((self.evidence / 'result.json').read_text())
        self.assertEqual(report['failure_context'], dict(llm_constructed=False, client_constructed=False, binding_installed=False))
        self.assertEqual(report['failure_process_id'], report['initialization_native_diagnostic']['process_id'])
        self.assertFalse(report['c1'])
        self.assertEqual(report['phases'], [])
        self.assertNotIn('cleanup', report)
        self.assertTrue((self.evidence / 'quant-native-init-diagnostic.json').is_file())
        self.capture.assert_called_once_with('/quant', approved_build_sha256='a' * 64)
        self.client.assert_not_called()

    def test_diagnostic_exception_does_not_replace_original_constructor_error(self):
        self.capture.side_effect = RuntimeError('DIAGNOSTIC_TEST_FAILURE')
        self.assertEqual(probe.main(), 2)
        report = json.loads((self.evidence / 'result.json').read_text())
        self.assertIn('161001', report['error'])
        self.assertIn('161001', report['traceback'])
        self.assertIn('DIAGNOSTIC_TEST_FAILURE', report['initialization_native_diagnostic']['error'])
        self.assertEqual(report['initialization_native_diagnostic']['status'], 'DIAGNOSTIC_FAILED_PRIMARY_ERROR_PRESERVED')
        self.client.assert_not_called()


if __name__ == '__main__':
    unittest.main()
