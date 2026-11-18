"""Experimental probe through real public Core, with explicit CPU fixtures."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import test_tensor_worker_core_probe as public
from test_core_tensor_statistics_capture import FakeCollector, FakeVerifier
from experiments.worker_core_v11_20261118 import core_tensor_statistics_capture as capture
from experiments.worker_core_v11_20261118 import core_tensor_statistics_probe as probe
from experiments.worker_core_v11_20261118 import prepare_core_tensor_statistics as prepare
from experiments.c2c3_closure_20261117.queue_contract import validate_cases


class CPUCapture(capture.PublicTensorStatisticsSidecar):
    def __init__(self, session, folder, *, source_pins):
        super().__init__(session, folder, source_pins=source_pins,
                         _capture=FakeCollector(), _hook_verifier=FakeVerifier())


class StatisticsProbeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = public.TensorPublicProbeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.case, self.driver = self.fixture.case, self.fixture.driver
        self.driver.collective_rpc = lambda function, args=(), timeout=300: [function(self.case.worker, *args)]
        self.enterContext(patch.object(probe.original, 'actual', lambda worker: worker))
        self.enterContext(patch.object(probe, 'PublicTensorStatisticsSidecar', CPUCapture))
        self.manifest = self.fixture.out/'source_manifest.json'
        files = [dict(path=name, sha256=capture.sha(probe.ROOT/name)) for name in capture.RUNTIME_FILES+probe.EXTRA_FILES]
        self.manifest.write_text(json.dumps(dict(schema='demoncat.c2c3.source.v1', files=files)))
        self.manifest_sha = capture.sha(self.manifest)

    def run_probe(self):
        return probe.drive(self.driver, self.case.client, prompt={'prompt_token_ids': [1, 2, 3]}, params=object(),
            out=self.fixture.out, report=self.fixture.report, manifest_path=self.manifest,
            manifest_sha256=self.manifest_sha)

    def test_actual_public_driver_same_worker_three_capture_steps_and_off_health(self):
        result = self.run_probe()
        self.assertEqual(result['status'], probe.original.SUCCESS, result.get('error'))
        self.assertEqual(result['statistics_status'], 'THREE_ORIGINAL_WINDOWS_READY_FOR_INDEPENDENT_RAW_REVIEW')
        self.assertEqual(self.driver.enqueued, self.driver.executed)
        self.assertEqual(len(self.driver.enqueued), 10)
        self.assertEqual((result['c2']['hits'], result['c3']['hits']), (2, 4))
        self.assertEqual(result['healthy_before'], result['healthy_after'])
        self.assertEqual([r['actual_step'] for r in result['collector_final']['phase_steps']], [0, 1, 2])
        self.assertTrue(all(r['completed'] for r in result['collector_final']['phase_steps']))
        self.assertFalse(result['public_e5'])
        self.assertFalse(result['collector_final']['global_api_wrappers_restored'])
        self.assertFalse(hasattr(self.case.worker, probe.ATTRIBUTE))
        self.assertIsNone(sys.getprofile())
        self.fixture.receipts()
        self.case.assert_clean()

    def test_failed_logits_keeps_partial_f_and_public_recovery(self):
        self.driver.fail_index = 4
        result = self.run_probe()
        self.assertEqual(result['status'], 'REFUSED_OR_FAILED')
        self.assertEqual(result['statistics_status'], 'PARTIAL_OR_FAILED_CAPTURE_RETAINED')
        self.assertEqual([r['completed'] for r in result['collector_final']['phase_steps']], [True, True, False])
        self.assertEqual(result['recovery_requests'][0]['phase'], 'recovery_off')
        self.assertEqual(result['health_request']['phase'], 'recovery_health')
        self.assertFalse(result['worker_core']['ownership_retained'])
        self.assertIsNone(sys.getprofile())
        self.case.assert_clean()

    def test_manifest_or_collector_code_change_is_not_adopted(self):
        with self.assertRaisesRegex(RuntimeError, 'FROZEN_SOURCE_MANIFEST_CHANGED'):
            probe.source_pins(self.manifest, '0'*64)
        doc = json.loads(self.manifest.read_text())
        doc['files'][-1]['sha256'] = '0'*64
        self.manifest.write_text(json.dumps(doc))
        with self.assertRaisesRegex(RuntimeError, 'CAPTURE_SOURCE_BYTES_NOT_FROZEN'):
            probe.source_pins(self.manifest, capture.sha(self.manifest))

    def test_one_disposable_case_preparation_is_not_a_launch(self):
        rows = prepare.cases('/home/frozen-source', self.manifest_sha)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['review']['actual_step_mapping'],
                         {'step0': 'baseline', 'step1': 'baseline_repeat', 'step2': 'c2_fault'})
        self.assertTrue(all(mount[2] == 'ro' for mount in row['mounts']))
        self.assertFalse(any(mount[1] == '/msprobe_packages' for mount in row['mounts']))
        dependency = row['review']['required_queue_collector_mount']
        self.assertEqual((dependency['source'], dependency['destination'], dependency['mode']),
                         (prepare.COLLECTOR_SOURCE, '/msprobe_packages', 'ro'))
        self.assertEqual(dependency['queue_script_sha256'], capture.sha(prepare.QUEUE_SCRIPT))
        self.assertEqual(validate_cases(rows, [5]), rows)
        self.assertIn('core_tensor_statistics_probe.py', row['command'])
        self.assertIn('--source-manifest-sha256 '+self.manifest_sha, row['command'])
        self.assertIn('/msprobe_packages', row['env']['PYTHONPATH'])
        self.assertTrue(row['review']['disposable_worker_required'])
        with self.assertRaises(ValueError):
            prepare.cases('/home/frozen-source', 'not-a-sha')

    def test_collector_source_override_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'QUEUE_COLLECTOR_SOURCE_OVERRIDE_FORBIDDEN'):
            prepare.cases('/home/frozen-source', self.manifest_sha, collector='/home/another-collector')

    def test_changed_queue_collector_mode_or_source_is_refused(self):
        original = prepare.QUEUE_SCRIPT.read_text()
        path = self.fixture.out/'queue_contract_fixture.py'
        changes = (original.replace("'/msprobe_packages', 'ro'", "'/msprobe_packages', 'rw'"),
                   original.replace(prepare.COLLECTOR_SOURCE, '/home/foreign-collector'))
        for changed in changes:
            self.assertNotEqual(changed, original)
            path.write_text(changed)
            with patch.object(prepare, 'QUEUE_SCRIPT', path):
                with self.assertRaisesRegex(ValueError, 'QUEUE_FIXED_READ_ONLY_COLLECTOR_MOUNT_REQUIRED'):
                    prepare.cases('/home/frozen-source', self.manifest_sha)

    def test_queue_contract_still_rejects_duplicate_collector_destination(self):
        rows = prepare.cases('/home/frozen-source', self.manifest_sha)
        rows[0]['mounts'].append([prepare.COLLECTOR_SOURCE, '/msprobe_packages', 'ro'])
        with self.assertRaisesRegex(ValueError, 'MOUNT_PATH_INVALID'):
            validate_cases(rows, [5])

    def test_probe_composition_restores_entry_and_does_not_replace_core_factory(self):
        factory = capture.request_module.ActualTensorRequest
        entry = probe.original.drive
        with patch.object(probe.original, 'main', return_value=0) as main:
            self.assertEqual(probe.main(['--source-manifest', str(self.manifest),
                '--source-manifest-sha256', self.manifest_sha, '--mechanism', '18.api_half']), 0)
            main.assert_called_once_with(['--mechanism', '18.api_half'])
        self.assertIs(probe.original.drive, entry)
        self.assertIs(capture.request_module.ActualTensorRequest, factory)


if __name__ == '__main__':
    unittest.main()
