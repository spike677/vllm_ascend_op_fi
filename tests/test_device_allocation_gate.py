"""CPU/read-only device gate contracts; no Docker, CANN, NPU or process actions."""
import copy
import importlib.util
import json
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


TOOLS = Path(__file__).resolve().parents[1] / 'fault39/tools'
sys.path.insert(0, str(TOOLS))
try:
    try:
        import fcntl
        replacements = {}
    except ImportError:
        replacements = {'fcntl': SimpleNamespace()}
    with patch.dict(sys.modules, replacements):
        spec = importlib.util.spec_from_file_location('device_allocation_gate_under_test', TOOLS / 'run_124.py')
        gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gate)
finally:
    sys.path.remove(str(TOOLS))


def checkpoint(ids=None, *, entries=True):
    return {'Data': {'PodDeviceEntries': [dict(PodUID='pod-lifetime', ContainerName='ascend',
        ResourceName='huawei.com/Ascend910', DeviceIDs={'-1': ids or ['Ascend910-2', 'Ascend910-5']})]
        if entries else [], 'RegisteredDevices': {'huawei.com/Ascend910': ['Ascend910-2', 'Ascend910-5']}}}


class AllocationSchemaTests(unittest.TestCase):
    def test_per_pod_allocation_reserves_selected_cards_without_process_evidence(self):
        blocks, allocations = gate.checkpoint_allocations(checkpoint(), [2, 5])
        self.assertEqual([row['card'] for row in blocks], [2, 5])
        self.assertEqual(len(allocations), 2)
        self.assertTrue(all(row['reason'] == 'KUBELET_DEVICE_ALLOCATION' for row in blocks))
        self.assertTrue(all(row['status'] == 'ALLOCATION_RECORDED_RELEASE_UNVERIFIED' for row in blocks))

    def test_global_registration_is_not_a_pod_reservation(self):
        self.assertEqual(gate.checkpoint_allocations(checkpoint(entries=False), [2, 5]), ([], []))

    def test_unselected_physical_card_does_not_reserve_selected_card(self):
        blocks, inventory = gate.checkpoint_allocations(checkpoint(['Ascend910-7']), [2, 5])
        self.assertEqual(blocks, [])
        self.assertEqual(inventory[0]['card'], 7)

    def test_list_and_encoded_data_checkpoint_forms(self):
        value = checkpoint()
        value['Data']['PodDeviceEntries'][0]['DeviceIDs'] = ['Ascend910-2']
        value['Data'] = json.dumps(value['Data'])
        self.assertEqual(len(gate.checkpoint_allocations(value, [2])[0]), 1)

    def test_missing_or_malformed_allocation_schema_refuses(self):
        for value in ([], {}, {'Data': {}}, {'Data': {'RegisteredDevices': {}}},
                      {'Data': {'PodDeviceEntries': {}, 'RegisteredDevices': {}}}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                gate.checkpoint_allocations(value, [2])

    def test_unknown_device_id_or_accelerator_resource_cannot_mean_unallocated(self):
        for key, value in (('DeviceIDs', {'-1': ['Ascend910-virtual-2']}),
                           ('DeviceIDs', {'-1': ['unknown']}), ('DeviceIDs', {'-1': []}),
                           ('DeviceIDs', {'bad-numa': ['Ascend910-2']}),
                           ('ResourceName', 'huawei.com/Ascend910-2c'), ('PodUID', None)):
            data = checkpoint()
            data['Data']['PodDeviceEntries'][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                gate.checkpoint_allocations(data, [2, 5])

    def test_other_resource_does_not_turn_into_ascend_allocation(self):
        value = checkpoint()
        value['Data']['PodDeviceEntries'][0]['ResourceName'] = 'example.com/fpga'
        self.assertEqual(gate.checkpoint_allocations(value, [2, 5]), ([], []))


class RuntimePathTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.proc = self.root / 'proc'; self.proc.mkdir()
        self.default = self.root / 'default/device-plugins/kubelet_internal_checkpoint'
        self.devices = {238002: 2, 238005: 5}
        self.enterContext(patch.object(gate.os, 'makedev', side_effect=lambda major, minor: major * 1000 + minor, create=True))

    def process(self, pid, name, args):
        folder = self.proc / str(pid); folder.mkdir()
        (folder / 'comm').write_text(name + '\n')
        (folder / 'cmdline').write_bytes(('\x00'.join([name, *args]) + '\x00').encode())
        (folder / 'fd').mkdir()
        (folder / 'net').mkdir()
        (folder / 'net/unix').write_text('Num RefCount Protocol Flags Type St Inode Path\n')
        return folder

    def write_checkpoint(self, path=None, value=None):
        path = path or self.default; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value or checkpoint()))
        return path

    def inventory(self):
        return gate.runtime_device_inventory([2, 5], self.devices, proc_root=self.proc, default_checkpoint=self.default)

    def test_actual_containerd_config_state_and_both_runtime_versions_are_inventory(self):
        state = self.root / 'mindcluster-state'; state.mkdir()
        config = self.root / 'containerd.toml'
        config.write_text('version = 2\nstate = ' + json.dumps(str(state)) + '\n')
        self.process(10, 'containerd', ['--config', str(config)])
        for runtime in ('io.containerd.runtime.v1.linux', 'io.containerd.runtime.v2.task'):
            task = state / runtime / 'k8s.io' / 'task-id'; task.mkdir(parents=True)
            (task / 'config.json').write_text(json.dumps({'linux': {'devices': [
                dict(type='c', major=238, minor=2, path='/dev/davinci0')]}}))
        self.write_checkpoint(value=checkpoint(entries=False))
        report = self.inventory()
        self.assertEqual(report['blockers'], [])
        self.assertEqual(len(report['device_access_grants']), 2)
        self.assertEqual({row['card'] for row in report['device_access_grants']}, {2})
        self.assertEqual(report['containerd_states'][0]['state'], str(state))

    def test_explicit_state_option_takes_precedence(self):
        actual = self.root / 'actual-state'; actual.mkdir()
        state, reason = gate.containerd_state_path(['containerd', '--config', 'missing.toml', '--state=' + str(actual)])
        self.assertEqual(state, actual)
        self.assertEqual(reason['basis'], 'containerd_argv_state')

    def test_relative_missing_and_imported_state_paths_are_unknown(self):
        for args in (['--state', 'relative'], ['--config', str(self.root / 'missing.toml')],
                     ['--state'], ['--state', str(self.root), '--state', str(self.proc)]):
            with self.subTest(args=args), self.assertRaises((ValueError, OSError)):
                gate.containerd_state_path(['containerd', *args])
        config = self.root / 'imports.toml'; config.write_text('imports = ["other.toml"]\n')
        with self.assertRaisesRegex(ValueError, 'IMPORTED_STATE'):
            gate.containerd_state_path(['containerd', '-c', str(config)])

    def test_missing_stdlib_toml_reader_is_unknown_not_default_state(self):
        config = self.root / 'containerd.toml'; config.write_text('state = "/actual/state"\n')
        self.process(10, 'containerd', ['--config', str(config)])
        with patch.dict(sys.modules, {'tomllib': None}):
            report = self.inventory()
        self.assertEqual(report['containerd_states'], [])
        self.assertEqual(report['blockers'][0]['reason'], 'RUNTIME_ALLOCATION_SOURCE_UNKNOWN')
        self.assertIn('TOML_READER_UNAVAILABLE', report['blockers'][0]['error'])

    def test_actual_kubelet_root_dir_is_used(self):
        custom = self.root / 'custom-kubelet'
        self.process(20, 'kubelet', ['--root-dir=' + str(custom)])
        selected = self.write_checkpoint(custom / 'device-plugins/kubelet_internal_checkpoint')
        report = self.inventory()
        self.assertEqual(len(report['blockers']), 2)
        self.assertEqual(report['blockers'][0]['checkpoint'], str(selected))

    def test_retained_checkpoint_reserves_after_workload_and_kubelet_exit(self):
        self.write_checkpoint()
        report = self.inventory()
        self.assertEqual([row['card'] for row in report['blockers']], [2, 5])

    def test_owned_device_plugin_socket_can_use_default_despite_custom_root_dir(self):
        proc = self.process(20, 'kubelet', ['--root-dir', str(self.root / 'custom-root')])
        self.write_checkpoint()
        socket = (self.default.parent / 'kubelet.sock').as_posix()
        (proc / 'net/unix').write_text('Num RefCount Protocol Flags Type St Inode Path\n'
            '000: 0002 0000 0001 0001 01 120 ' + socket + '\n')
        (proc / 'fd/17').write_text('test socket fixture')
        with patch.object(gate.os, 'readlink', return_value='socket:[120]'):
            report = self.inventory()
        self.assertEqual([row['reason'] for row in report['blockers']], ['KUBELET_DEVICE_ALLOCATION'] * 2)
        self.assertEqual(report['kubelet_checkpoint_sources'][0]['basis'], 'owned_device_plugin_socket')
        self.assertEqual(report['kubelet_checkpoint_sources'][0]['checkpoint'], str(self.default))

    def test_unowned_socket_does_not_silently_override_actual_root(self):
        proc = self.process(20, 'kubelet', ['--root-dir', str(self.root / 'custom-root')])
        socket = (self.default.parent / 'kubelet.sock').as_posix()
        (proc / 'net/unix').write_text('Num RefCount Protocol Flags Type St Inode Path\n'
            '000: 0002 0000 0001 0001 01 120 ' + socket + '\n')
        (proc / 'fd/17').write_text('test socket fixture')
        with patch.object(gate.os, 'readlink', return_value='socket:[999]'):
            report = self.inventory()
        self.assertEqual(report['blockers'][0]['reason'], 'KUBELET_ALLOCATION_UNKNOWN')
        self.assertEqual(report['kubelet_checkpoint_sources'][0]['basis'], 'kubelet_root_dir_fallback')

    def test_active_kubelet_missing_or_corrupt_checkpoint_fails_closed(self):
        custom = self.root / 'kubelet'
        self.process(20, 'kubelet', ['--root-dir', str(custom)])
        for contents in (None, '{not JSON}', json.dumps({'Data': {}}),
                         json.dumps(checkpoint(['unknown-id']))):
            path = custom / 'device-plugins/kubelet_internal_checkpoint'
            if contents is not None:
                path.parent.mkdir(parents=True, exist_ok=True); path.write_text(contents)
            report = self.inventory()
            self.assertEqual(report['blockers'][0]['reason'], 'KUBELET_ALLOCATION_UNKNOWN')
            self.assertEqual(report['blockers'][0]['status'], 'UNKNOWN')

    def test_active_containerd_missing_config_is_diagnostic_unknown(self):
        self.process(10, 'containerd', ['--config', str(self.root / 'absent.toml')])
        report = self.inventory()
        self.assertEqual(report['blockers'][0]['reason'], 'RUNTIME_ALLOCATION_SOURCE_MISSING')

    def test_device_bind_mount_is_inventory_not_reservation(self):
        item = {'mounts': [dict(source='/dev/davinci5', destination='/dev/davinci0', type='bind')]}
        with patch.object(gate.Path, 'stat', return_value=SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=238005)):
            grants = gate.oci_device_grants(item, self.devices, task='task', config='config.json')
        self.assertEqual(grants[0]['card'], 5)
        self.assertEqual(grants[0]['kind'], 'OCI_DEVICE_MOUNT_PERMISSION')
        self.assertNotIn('reason', grants[0])

    def test_idle_persistent_docker_permission_remains_only_a_permission(self):
        item = dict(Name='/gyl_persistent', State={'Running': True},
                    HostConfig={'Devices': [{'PathOnHost': '/dev/davinci2'}]})
        self.assertEqual(gate.dedicated_atk_reservations(item, [2, 5]), [])
        dedicated = copy.deepcopy(item); dedicated['Name'] = '/ptq_running_job'
        self.assertEqual(gate.dedicated_atk_reservations(dedicated, [2, 5])[0]['card'], 2)
        dedicated['Name'] = '/atk_q_ops-math-v900-aclnnAtan2_dev2_slot0'
        self.assertEqual(gate.dedicated_atk_reservations(dedicated, [2, 5])[0]['card'], 2)
        dedicated['State']['Running'] = False
        self.assertEqual(gate.dedicated_atk_reservations(dedicated, [2, 5]), [])


if __name__ == '__main__':
    unittest.main()
