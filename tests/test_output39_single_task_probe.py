"""Four-request acceptance driver contracts, CPU only with explicit ABI doubles."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import test_output39_task as task_fixture
from test_output39_worker_core_probe import Driver
from test_postwrite_native_recovery import CpuNativeTransport
from experiments.worker_core_v11_20261118 import output39_single_task_probe as probe
from experiments.worker_core_v11_20261118 import prepare_output39_single_task as prepare


class ReadbackABI(CpuNativeTransport):
    snapshot = task_fixture.ReadbackTransport.snapshot


class SingleTaskProbeTests(unittest.TestCase):
    def setUp(self):
        self.case = task_fixture.Output39TaskTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.case.transport = ReadbackABI()
        self.out = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.driver = Driver(self.case)
        self.report = dict(status='STARTED', C1=False, C2=False, C3=False, E5=False)

    def drive(self):
        return probe.drive(self.driver, self.case.client, prompt=dict(prompt_token_ids=[1, 2, 3, 4]),
            params=object(), out=self.out, report=self.report, cpu_contract=True)

    def test_exactly_four_real_requests_original_native_receipts_and_separate_recovery(self):
        result = self.drive()
        self.assertEqual(result['status'], probe.CPU_SUCCESS, result.get('error'))
        self.assertEqual(self.driver.enqueued, self.driver.executed)
        self.assertEqual(len(self.driver.executed), 4)
        self.assertTrue(all(row['completed'] for row in result['requests']))
        self.assertEqual(result['checks']['typed_effect']['finite_to_positive_inf'], 1)
        self.assertTrue(result['checks']['exact_off_hidden_logits'] and result['checks']['exact_health_hidden_logits'])
        self.assertFalse(result['ownership_retained'])
        self.assertEqual(result['resource_cleanup']['native_leases'], 0)
        self.assertFalse(any(result[key] for key in ('C1', 'C2', 'C3', 'E5')))
        self.assertEqual(probe.sha(self.out/'host_receipt.json'), result['host_receipt']['sha256'])

    def test_failed_f_keeps_incomplete_mapping_and_recovers_without_retry(self):
        self.driver.fail_index = 1
        result = self.drive()
        self.assertEqual(result['status'], 'REFUSED_OR_FAILED')
        self.assertIn('CPU_ROOT_FORWARD_FAILED', result['error'])
        self.assertEqual([row['phase'] for row in result['requests']], ['normal', 'fault'])
        self.assertFalse(result['requests'][1]['completed'])
        self.assertEqual([row['phase'] for row in result['recovery_requests']], ['off', 'health'])
        self.assertEqual(len(self.driver.executed), 4)
        self.assertEqual(self.driver.aborted, [self.driver.enqueued[1]])
        self.assertFalse(result['ownership_retained'], result.get('recovery_error'))

    def test_uncertain_advance_records_owned_id_aborts_and_does_not_execute_f(self):
        def lose(method, args, responses):
            if method == 'demoncat_fault_rpc' and args[0] == 'advance' and len(self.driver.enqueued) == 2:
                self.case.executor.after = None
                raise TimeoutError('LOST_ARM_REPLY')
            return responses
        self.case.executor.after = lose
        result = self.drive()
        self.assertEqual(result['status'], 'REFUSED_OR_FAILED')
        self.assertEqual(len(self.driver.executed), 1)
        self.assertEqual(self.driver.aborted, [self.driver.enqueued[1]])
        self.assertFalse(result['requests'][1]['completed'])
        self.assertFalse(result['ownership_retained'], result.get('recovery_error'))

    def test_normal_operator_rejection_happens_before_fault_request(self):
        with patch.object(probe, 'verify_normal', side_effect=RuntimeError('CUSTOM_GEMM_UNADAPTED')):
            result = self.drive()
        self.assertEqual(result['status'], 'REFUSED_OR_FAILED')
        self.assertEqual(len(self.driver.executed), 1)
        self.assertEqual(result['requests'][0]['phase'], 'normal')
        self.assertFalse(self.case.provider.p._fault_attempted)

    def test_receipt_io_failure_keeps_original_forward_error(self):
        self.driver.fail_index = 1
        original = probe.write
        def fail_receipt(path, *args, **kwargs):
            if path.name == 'host_receipt.json': raise OSError('CPU_DISK_FULL')
            return original(path, *args, **kwargs)
        with patch.object(probe, 'write', side_effect=fail_receipt): result = self.drive()
        self.assertIn('CPU_ROOT_FORWARD_FAILED', result['error'])
        self.assertIn('CPU_DISK_FULL', result['receipt_error'])
        self.assertEqual(result['status'], 'REFUSED_OR_FAILED')


class CasePreparationTests(unittest.TestCase):
    def case(self, **changes):
        values = dict(source='/home/17-fault-c2c3-20261117/source-new', installed='/home/installed',
            library='/home/old-native/libdcat_postwrite.so', sha256='a'*64, build_fingerprint='b'*64,
            source_manifest_sha256='c'*64)
        values.update(changes)
        return prepare.make_case(**values)

    def test_one_fixed_image_case_uses_existing_queue_and_no_duplicate_collector_mount(self):
        row, = self.case()
        self.assertEqual(row['image'], probe.IMAGE)
        self.assertEqual(row['timeout'], row['review']['resource_budget']['timeout_seconds'])
        self.assertEqual(row['review']['resource_budget']['nominal_actual_requests'], 4)
        self.assertEqual(row['review']['resource_budget']['max_actual_requests_with_recovery'], 6)
        self.assertEqual(row['review']['physical_cards'], [5])
        self.assertEqual(row['env']['PYTHONPATH'].split(':')[0], '/installed')
        self.assertNotIn('/msprobe_packages', [m[1] for m in row['mounts']])
        self.assertNotIn('--statistics', row['command'])
        self.assertNotIn('--stage', row['command'])
        self.assertIn('--deadline-seconds 480', row['command'])
        self.assertFalse(row['review']['full_census'])

    def test_invalid_native_identity_model_card_timeout_and_mount_are_rejected(self):
        for changed in (dict(sha256='x'*64), dict(model='/other-model'), dict(card=0),
                        dict(timeout=1801), dict(timeout=120), dict(library='/tmp/a,b.so')):
            with self.subTest(changed=changed), self.assertRaises((RuntimeError, ValueError)):
                self.case(**changed)

    def test_source_manifest_exact_bytes_and_installed_mismatch_fail_before_case(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        files = []
        for relative in probe.SOURCES:
            target = root/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            # The actual preparer identity is independently pinned too.
            raw = Path(prepare.__file__).read_bytes() if target.name == 'prepare_output39_single_task.py' else b'# fixture\n'
            target.write_bytes(raw)
            files.append(dict(path=relative, bytes=len(raw), sha256=probe.sha(target)))
        manifest = root/'source_manifest.json'
        manifest.write_text(json.dumps(dict(base_commit='d'*40, files=files)))
        pins = probe.source_inputs(root, probe.sha(manifest))
        self.assertEqual(set(pins['pins']), set(probe.SOURCES))
        installed = root/'installed'
        for relative in pins['pins']:
            target = (relative.removeprefix('src/') if relative.startswith('src/') else
                      relative if relative.startswith('fault_injection/') else None)
            if target:
                destination = installed/target
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((root/relative).read_bytes())
        native = root/'native.so'; native.write_bytes(b'CPU file identity only')
        (installed/'fault_injection/deployments/output39_execution.py').write_bytes(b'# changed\n')
        args = NS(source=root, installed=installed, library=native, sha256=probe.sha(native),
            source_manifest_sha256=probe.sha(manifest), build_fingerprint='e'*64,
            model=prepare.MODEL, card=5, timeout=600)
        with self.assertRaisesRegex(RuntimeError, 'INSTALLED_TASK_MODULE_DIFFERS'): prepare.prepare(args)
        (root/probe.SOURCES[0]).write_bytes(b'# changed\n')
        with self.assertRaisesRegex(RuntimeError, 'FROZEN_TASK_SOURCE_CHANGED'):
            probe.source_inputs(root, probe.sha(manifest))


if __name__ == '__main__':
    unittest.main()
