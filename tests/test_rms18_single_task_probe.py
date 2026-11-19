"""Four-request acceptance driver through real CPU Host/Runtime, no NPU claims."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import test_rms18_task as fixture
from test_output39_worker_core_probe import Driver as QueueFixture
from experiments.worker_core_v11_20261118 import rms18_single_task_probe as probe
from experiments.worker_core_v11_20261118 import prepare_rms18_single_task as prepare


class Driver(QueueFixture):
    def wait_for_completion(self, use_tqdm):
        self.case.provider = self.case.worker._demoncat_worker_fault_runtime._leases['18.api_half'].provider
        return super().wait_for_completion(use_tqdm)


class Rms18ProbeTests(unittest.TestCase):
    def setUp(self):
        self.case = fixture.Rms18TaskTests()
        self.case.setUp(); self.addCleanup(self.case.doCleanups)
        self.out = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.driver = Driver(self.case)
        self.report = dict(status='STARTED', C1=False, C2=False, C3=False, E5=False)

    def test_two_scopes_keep_actual_worker_with_distinct_owners(self):
        from demoncat.runtime.vllm_ascend.worker_extension.host import WorkerFaultClient
        from demoncat.runtime.vllm_ascend.worker_extension import serving_tensor_binding as bindings
        first_client = self.case.client
        first = self.drive()
        self.assertEqual(first['status'], probe.CPU_SUCCESS, first)
        first_owner = first_client.snapshot()['owner']
        first_worker = first_client.snapshot()['worker']
        bindings.remove_tensor_serving_binding(self.case.worker, '18.api_half')
        self.out = self.out / 'scope1'; self.out.mkdir()
        self.report = dict(status='STARTED')
        self.case.client = WorkerFaultClient(first_client.host, first_client.config)
        second = self.drive()
        self.assertEqual(second['status'], probe.CPU_SUCCESS, second)
        self.assertEqual(self.case.client.snapshot()['worker'], first_worker)
        self.assertNotEqual(self.case.client.snapshot()['owner'], first_owner)
        self.assertEqual(self.case.model.forward_calls, 8)
        self.assertFalse(self.case.client.snapshot()['ownership_retained'])

    def drive(self):
        return probe.drive(self.driver, self.case.client, prompt=dict(prompt_token_ids=[1, 2, 3, 4]),
            params=object(), out=self.out, report=self.report, cpu_contract=True)

    def test_four_real_requests_one_event_and_exact_off_health(self):
        result = self.drive()
        self.assertEqual(result['status'], probe.CPU_SUCCESS, result.get('error'))
        self.assertEqual(self.driver.enqueued, self.driver.executed)
        self.assertEqual(len(self.driver.executed), 4)
        self.assertEqual(result['checks']['normal_fault_observed_returns'], [1, 1])
        self.assertEqual(result['checks']['off_health_rms_observers'], [0, 0])
        self.assertTrue(result['checks']['exact_off_hidden_logits'] and result['checks']['exact_health_hidden_logits'])
        self.assertGreater(result['checks']['typed_effect']['changed_elements'], 0)
        self.assertFalse(result['ownership_retained'])
        self.assertFalse(any(result[k] for k in ('C1', 'C2', 'C3', 'E5')))
        self.assertEqual(probe.sha(self.out/'host_receipt.json'), result['host_receipt']['sha256'])

    def test_failed_f_keeps_mapping_and_recovers_without_retry(self):
        self.driver.fail_index = 1
        result = self.drive()
        self.assertEqual(result['status'], 'REFUSED_OR_FAILED')
        self.assertIn('CPU_ROOT_FORWARD_FAILED', result['error'])
        self.assertEqual([r['phase'] for r in result['requests']], ['normal', 'fault'])
        self.assertFalse(result['requests'][1]['completed'])
        self.assertEqual([r['phase'] for r in result['recovery_requests']], ['off', 'health'])
        self.assertEqual(self.driver.aborted, [self.driver.enqueued[1]])
        self.assertEqual(len(self.driver.executed), 4)
        self.assertFalse(result['ownership_retained'], result.get('recovery_error'))

    def test_uncertain_advance_does_not_execute_or_retry_f(self):
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
        self.assertFalse(result['ownership_retained'], result.get('recovery_error'))

    def test_normal_unobservable_refuses_before_f(self):
        with patch.object(probe, 'verify_normal', side_effect=RuntimeError('NORMAL_UNOBSERVABLE')):
            result = self.drive()
        self.assertEqual(result['status'], 'REFUSED_OR_FAILED')
        self.assertEqual(len(self.driver.executed), 1)
        self.assertFalse(self.case.provider.fault_started)

    def test_original_failure_survives_receipt_io_failure(self):
        self.driver.fail_index = 1
        original = probe.write
        def fail(path, *args, **kwargs):
            if path.name == 'host_receipt.json': raise OSError('CPU_DISK_FULL')
            return original(path, *args, **kwargs)
        with patch.object(probe, 'write', side_effect=fail): result = self.drive()
        self.assertIn('CPU_ROOT_FORWARD_FAILED', result['error'])
        self.assertIn('CPU_DISK_FULL', result['receipt_error'])
        self.assertEqual(result['status'], 'REFUSED_OR_FAILED')


class Rms18PrepareTests(unittest.TestCase):
    def case(self, **changes):
        values = dict(source='/home/source-new', installed='/home/installed', source_manifest_sha256='c'*64)
        values.update(changes)
        return prepare.make_case(**values)

    def test_one_case_no_native_build_no_census_uses_existing_queue(self):
        row, = self.case()
        self.assertEqual(row['image'], probe.IMAGE)
        self.assertEqual(row['review']['resource_budget']['nominal_actual_requests'], 4)
        self.assertEqual(row['review']['resource_budget']['normal_return_observation_limit'], 1)
        self.assertEqual(row['review']['physical_cards'], [5])
        self.assertFalse(row['review']['full_census'] or row['review']['statistics'])
        self.assertNotIn('--library', row['command'])
        self.assertNotIn('/msprobe_packages', [m[1] for m in row['mounts']])
        self.assertEqual(row['env']['PYTHONPATH'].split(':')[0], '/installed')

    def test_invalid_identity_model_card_budget_rejected(self):
        for changed in (dict(source_manifest_sha256='x'*64), dict(model='/other'), dict(card=0),
                        dict(timeout=1801), dict(timeout=120), dict(source='/tmp/a,b')):
            with self.subTest(changed=changed), self.assertRaises((ValueError, RuntimeError)):
                self.case(**changed)

    def test_own_source_manifest_and_installed_source_mismatch_refuse(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        files = []
        for relative in probe.SOURCES:
            target = root/relative; target.parent.mkdir(parents=True, exist_ok=True)
            raw = Path(prepare.__file__).read_bytes() if target.name == 'prepare_rms18_single_task.py' else b'# CPU fixture\n'
            target.write_bytes(raw)
            files.append(dict(path=relative, bytes=len(raw), sha256=probe.sha(target)))
        manifest = root/'source_manifest.json'
        manifest.write_text(json.dumps(dict(base_commit='d'*40, files=files)))
        source = probe.source_inputs(root, probe.sha(manifest))
        self.assertEqual(set(source['pins']), set(probe.SOURCES))
        installed = root/'installed'
        for relative in source['pins']:
            target = (relative.removeprefix('src/') if relative.startswith('src/') else
                      relative if relative.startswith('fault_injection/') else None)
            if target:
                destination = installed/target; destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((root/relative).read_bytes())
        (installed/'fault_injection/rmsnorm_return.py').write_bytes(b'# changed\n')
        args = NS(source=root, installed=installed, source_manifest_sha256=probe.sha(manifest),
                  model=prepare.MODEL, card=5, timeout=600)
        with self.assertRaisesRegex(RuntimeError, 'INSTALLED_TASK_MODULE_DIFFERS'): prepare.prepare(args)
        (root/probe.SOURCES[0]).write_bytes(b'# changed\n')
        with self.assertRaisesRegex(RuntimeError, 'FROZEN_TASK_SOURCE_CHANGED'):
            probe.source_inputs(root, probe.sha(manifest))


if __name__ == '__main__': unittest.main()
