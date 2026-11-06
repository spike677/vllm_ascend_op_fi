"""CPU-only safety boundaries for the reviewed card-5 ATK validator.

Importing the helper cannot start its main function. Every process API is mocked;
negative main-path checks use a checked temporary directory inside analysis.
"""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import launch_model_case as launch


HERE = Path(__file__).resolve().parent
HELPER = HERE / 'launch_model_case.py'
IDENTITY = 'a' * 64
IMAGE = 'sha256:' + 'b' * 64
ALLOWED = ('/dev/davinci5', '/dev/davinci_manager', '/dev/devmm_svm', '/dev/hisi_hdc')


def inspected():
    return {
        'Id': IDENTITY, 'Name': '/atk_q_cpu_mock', 'Image': IMAGE,
        'Config': {'Labels': {'atk.queue.job_id': 'mock-card5-job'}},
        'HostConfig': {
            'Devices': [{'PathOnHost': path, 'PathInContainer': path, 'CgroupPermissions': 'rwm'}
                        for path in ALLOWED],
            'Privileged': False, 'DeviceRequests': None, 'DeviceCgroupRules': None,
        },
        'Mounts': [], 'State': {'Running': True, 'Status': 'running'},
    }


class Card5AtkValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper_before = HELPER.read_bytes()

    @classmethod
    def tearDownClass(cls):
        if HELPER.read_bytes() != cls.helper_before:
            raise AssertionError('Launcher source changed during CPU safety tests')

    def setUp(self):
        self.process_run = patch.object(launch.subprocess, 'run', side_effect=AssertionError('Real process forbidden')).start()
        self.addCleanup(patch.stopall)
        self.process_call = patch.object(launch.subprocess, 'call', side_effect=AssertionError('Controller execution forbidden')).start()

    def reject(self, obj, identity=IDENTITY):
        with self.assertRaises(ValueError):
            launch.validate(obj, identity)
        self.process_run.assert_not_called()
        self.process_call.assert_not_called()

    def test_accept_exact_card5_and_manager_devices(self):
        obj = inspected()
        before = copy.deepcopy(obj)
        receipt = launch.validate(obj, IDENTITY)
        self.assertEqual(obj, before)
        self.assertEqual(receipt, {'id': IDENTITY, 'name': obj['Name'], 'image': IMAGE,
                                  'queue_job_id': 'mock-card5-job', 'devices': sorted(ALLOWED),
                                  'initial_state': obj['State']})

    def test_manager_devices_are_optional_but_card5_is_required(self):
        obj = inspected()
        obj['HostConfig']['Devices'] = [obj['HostConfig']['Devices'][0]]
        self.assertEqual(launch.validate(obj, IDENTITY)['devices'], ['/dev/davinci5'])

    def test_accept_only_allowed_device_bind_sources(self):
        obj = inspected()
        obj['Mounts'] = [{'Source': path, 'Destination': path} for path in ALLOWED]
        self.assertEqual(launch.validate(obj, IDENTITY)['id'], IDENTITY)

    def test_exact_full_identity_must_match(self):
        for identity in ('c' * 64, IDENTITY[:12], '', None):
            with self.subTest(identity=identity):
                self.reject(inspected(), identity)

    def test_unrelated_container_names_are_rejected(self):
        for name in ('/training', '/atk_q', '/xatk_q_cpu_mock', '/atk_queue_job'):
            with self.subTest(name=name):
                obj = inspected()
                obj['Name'] = name
                self.reject(obj)

    def test_missing_or_empty_queue_job_label_is_rejected(self):
        for labels in (None, {}, {'atk.queue.job_id': ''}, {'atk.queue.other': 'mock'}):
            with self.subTest(labels=labels):
                obj = inspected()
                obj['Config']['Labels'] = labels
                self.reject(obj)

    def test_missing_card5_or_devices_is_rejected(self):
        for devices in (None, [], [{'PathOnHost': '/dev/davinci_manager'}]):
            with self.subTest(devices=devices):
                obj = inspected()
                obj['HostConfig']['Devices'] = devices
                self.reject(obj)

    def test_any_other_physical_card_is_rejected(self):
        for card in (0, 1, 2, 3, 4, 6, 7):
            with self.subTest(card=card):
                obj = inspected()
                obj['HostConfig']['Devices'].append({'PathOnHost': '/dev/davinci' + str(card)})
                self.reject(obj)

    def test_any_other_device_path_is_rejected(self):
        for device in ('/dev', '/dev/kfd', '/dev/dri/renderD128', '/dev/davinci_manager_extra'):
            with self.subTest(device=device):
                obj = inspected()
                obj['HostConfig']['Devices'].append({'PathOnHost': device})
                self.reject(obj)

    def test_privileged_device_requests_and_cgroup_rules_are_rejected(self):
        for key, value in (('Privileged', True), ('DeviceRequests', [{'Driver': 'ascend', 'Count': -1}]),
                           ('DeviceCgroupRules', ['c *:* rwm'])):
            with self.subTest(key=key):
                obj = inspected()
                obj['HostConfig'][key] = value
                self.reject(obj)

    def test_root_and_dev_directory_binds_are_rejected(self):
        for source in ('/', '//', '/dev', '/dev/', '/dev///'):
            with self.subTest(source=source):
                obj = inspected()
                obj['Mounts'] = [{'Source': source}]
                self.reject(obj)

    def test_other_device_binds_and_device_subdirectories_are_rejected(self):
        for source in ('/dev/davinci0', '/dev/davinci4/', '/dev/davinci6', '/dev/dri',
                       '/dev/davinci5/child', '/dev/davinci_manager_extra'):
            with self.subTest(source=source):
                obj = inspected()
                obj['Mounts'] = [{'Source': source}]
                self.reject(obj)

    def test_invalid_inspect_never_reaches_stop_or_controller(self):
        # Exercise main's ordering, not a second implementation of the guard.
        mutations = (
            lambda o: o.update(Id='c' * 64),
            lambda o: o.update(Name='/unrelated'),
            lambda o: o['Config'].update(Labels={}),
            lambda o: o['HostConfig']['Devices'].append({'PathOnHost': '/dev/davinci0'}),
            lambda o: o['HostConfig'].update(Privileged=True),
            lambda o: o.update(Mounts=[{'Source': '/dev'}]),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                obj = inspected()
                mutate(obj)
                temp = tempfile.TemporaryDirectory(prefix='launch_guard_', dir=HERE)
                checked = Path(temp.name).resolve()
                self.assertEqual(checked.parent, HERE.resolve())
                try:
                    response = subprocess.CompletedProcess(['docker', 'inspect', IDENTITY], 0,
                                                           json.dumps([obj]), '')
                    with patch.object(launch, 'ROOT', checked), \
                         patch.object(launch, 'run', return_value=response) as backend, \
                         patch.object(sys, 'argv', ['launch_model_case.py', '--case', 'model_baseline',
                                                   '--attempt', 'cpu_guard', '--stop-atk-id', IDENTITY]):
                        with self.assertRaises(ValueError):
                            launch.main()
                    self.assertEqual(backend.call_args_list[0].args, (['docker', 'inspect', IDENTITY],))
                    self.assertEqual(backend.call_count, 1)
                    self.process_call.assert_not_called()
                    self.assertFalse((checked / 'outputs/manual_launch/cpu_guard.json').exists())
                finally:
                    temp.cleanup()


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Card5AtkValidation))
    unchanged = HELPER.read_bytes() == Card5AtkValidation.helper_before
    report = {
        'schema': 'fault33.card5_atk_validator.cpu_mock.v1',
        'status': 'PASS_CPU_MOCKS' if result.wasSuccessful() and unchanged else 'FAIL_CPU_MOCKS',
        'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'skipped': len(result.skipped), 'helper_unchanged': unchanged,
        'helper_sha256': hashlib.sha256(HELPER.read_bytes()).hexdigest(),
        'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'test_scope': 'Actual validate function and refused main branches, inspect fixtures and blocked process APIs',
        'docker_or_hardware_commands_executed': False, 'ssh_executed': False, 'npu_executed': False,
    }
    (HERE / 'launch_model_case_cpu_report.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps(report))
    raise SystemExit(0 if report['status'] == 'PASS_CPU_MOCKS' else 1)
