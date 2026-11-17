"""CPU contracts only: no API, socket, credential or device access."""
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

TOOLS = Path(__file__).resolve().parents[1] / 'fault39/tools'
spec = importlib.util.spec_from_file_location('kubelet_allocations', TOOLS / 'kubelet_allocations.py')
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)


def varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def field(number, value):
    if isinstance(value, str):
        value = value.encode()
    return varint(number * 8 + 2) + varint(len(value)) + value


HEADERS = b'HTTP/2 200\r\ncontent-type: application/grpc\r\ngrpc-status: 0\r\n'


def reply(extra=b''):
    container = field(1, 'worker') + extra
    pod = field(1, 'pod') + field(2, 'ns') + field(3, container)
    payload = field(1, pod)
    return b'\0' + len(payload).to_bytes(4, 'big') + payload


def api_fixture():
    node = dict(apiVersion='v1', kind='Node', metadata=dict(name='node', uid='node-uid', resourceVersion='12'),
        status=dict(nodeInfo=dict(machineID='machine', bootID='boot', kubeletVersion='v1.28.15'),
                    conditions=[dict(type='Ready', status='True', lastHeartbeatTime=datetime.now(timezone.utc).isoformat())]))
    pods = dict(apiVersion='v1', kind='PodList', metadata=dict(resourceVersion='13'), items=[dict(
        metadata=dict(uid='pod-uid', name='pod', namespace='ns'), spec=dict(nodeName='node', containers=[dict(name='worker')]),
        status=dict(phase='Running'))])
    return node, pods


class ProtoTests(unittest.TestCase):
    def test_exact_list_not_allocatable_decodes_containers(self):
        rows = live.parse_list_response(reply(), HEADERS)
        self.assertEqual(rows, [dict(name='pod', namespace='ns', containers=[dict(name='worker', devices=[])])])
        self.assertEqual(live.RPC, '/v1.PodResourcesLister/List')

    def test_device_assignment_remains_visible(self):
        device = field(1, 'huawei.com/Ascend910') + field(2, 'Ascend910-5')
        rows = live.parse_list_response(reply(field(2, device)), HEADERS)
        self.assertEqual(rows[0]['containers'][0]['devices'][0]['device_ids'], ['Ascend910-5'])

    def test_truncation_compression_multiple_messages_and_nonzero_status_refuse(self):
        for body in (reply()[:-1], b'\1' + reply()[1:], reply() + b'\0' * 5, b'\0' * 4):
            with self.subTest(body=body[:5]), self.assertRaises(ValueError):
                live.parse_list_response(body, HEADERS)
        for headers in (HEADERS.replace(b'grpc-status: 0', b'grpc-status: 13'),
                        HEADERS.replace(b'HTTP/2 200', b'HTTP/2 503'),
                        HEADERS.replace(b'grpc-status: 0\r\n', b''), HEADERS + b'grpc-status: 0\r\n'):
            with self.subTest(headers=headers), self.assertRaises(ValueError):
                live.parse_list_response(reply(), headers)

    def test_unknown_fields_dra_and_invalid_utf8_refuse(self):
        for extra in (field(6, b'x'), field(5, b''), field(1, b'\xff')):
            with self.subTest(extra=extra), self.assertRaises((ValueError, UnicodeError)):
                live.parse_list_response(reply(extra), HEADERS)

    def test_duplicate_pods_and_malformed_varints_refuse(self):
        body = reply()
        double = b'\0' + (2 * (len(body) - 5)).to_bytes(4, 'big') + body[5:] * 2
        for value in (double, b'\0\0\0\0\1\x80', b'\0\0\0\0\1\x00'):
            with self.assertRaises(ValueError):
                live.parse_list_response(value, HEADERS)


class NodeApiTests(unittest.TestCase):
    def snapshot(self, node=None, pods=None):
        fallback_node, fallback_pods = api_fixture()
        node = fallback_node if node is None else node
        pods = fallback_pods if pods is None else pods
        return live.node_snapshot(lambda path: (copy.deepcopy(node if '/nodes/' in path else pods), 'a' * 64),
                                  {'node_name': 'node'}, 'machine', 'boot')

    def test_node_certificate_scope_machine_boot_and_complete_pod_list(self):
        rows, proof = self.snapshot()
        self.assertEqual(rows[0]['uid'], 'pod-uid')
        self.assertTrue(proof['machine_id_matches'] and proof['boot_id_matches'])

    def test_machine_boot_version_readiness_and_heartbeat_refuse(self):
        for member, value in (('machineID', 'other'), ('bootID', 'other'), ('kubeletVersion', 'v1.29.0')):
            node, _ = api_fixture(); node['status']['nodeInfo'][member] = value
            with self.subTest(member=member), self.assertRaises(ValueError):
                self.snapshot(node=node)
        for member, value in (('status', 'False'), ('lastHeartbeatTime', '2000-01-01T00:00:00Z')):
            node, _ = api_fixture(); node['status']['conditions'][0][member] = value
            with self.assertRaises(ValueError):
                self.snapshot(node=node)

    def test_pagination_foreign_node_unknown_phase_and_duplicate_pod_refuse(self):
        for failure in range(4):
            _, pods = api_fixture()
            if failure == 0: pods['metadata']['continue'] = 'more'
            if failure == 1: pods['items'][0]['spec']['nodeName'] = 'another'
            if failure == 2: pods['items'][0]['status']['phase'] = 'Unknown'
            if failure == 3: pods['items'] *= 2
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                self.snapshot(pods=pods)

    def test_pending_npu_request_in_regular_init_or_ephemeral_container_refuses(self):
        for kind in ('containers', 'initContainers', 'ephemeralContainers'):
            _, pods = api_fixture()
            pods['items'][0]['status']['phase'] = 'Pending'
            pods['items'][0]['spec'][kind] = [dict(name='worker', resources=dict(limits={'huawei.com/Ascend910': '1'}))]
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'ACCELERATOR_REQUEST'):
                self.snapshot(pods=pods)

    def test_resource_claim_or_unknown_accelerator_quantity_refuses(self):
        for failure in range(5):
            _, pods = api_fixture(); pod = pods['items'][0]
            if failure == 0: pod['spec']['resourceClaims'] = [dict(name='claim')]
            else:
                key = 'huawei.com/Ascend310' if failure == 1 else 'vendor.example/opaque-device' if failure == 4 else 'huawei.com/Ascend910'
                value = '1.0' if failure == 2 else True if failure == 3 else '1'
                pod['spec']['containers'][0]['resources'] = dict(requests={key: value})
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                self.snapshot(pods=pods)


class LiveIdentityTests(unittest.TestCase):
    def test_socket_peer_change_or_query_failure_cannot_prove_no_owner(self):
        peer = dict(pid=100, uid=0, gid=0, device=1, inode=2)
        response = SimpleNamespace(returncode=0, stdout=reply(), stderr=HEADERS)
        with patch.object(live, '_peer', side_effect=[peer, dict(peer, inode=3)]), \
             patch.object(live.subprocess, 'run', return_value=response):
            with self.assertRaisesRegex(ValueError, 'SOCKET_CHANGED'):
                live.live_list(Path('/observed/socket'), 100)
        response.returncode = 28
        with patch.object(live, '_peer', return_value=peer), \
             patch.object(live.subprocess, 'run', return_value=response):
            with self.assertRaisesRegex(ValueError, 'QUERY_FAILED'):
                live.live_list(Path('/observed/socket'), 100)

    def test_non_kubelet_socket_peer_refuses(self):
        path = Mock()
        path.stat.return_value = SimpleNamespace(st_dev=1, st_ino=2)
        with patch.object(live.socket, 'socket') as connection, \
             patch.object(live.socket, 'AF_UNIX', 1, create=True), \
             patch.object(live.socket, 'SO_PEERCRED', 17, create=True):
            connection.return_value.__enter__.return_value.getsockopt.return_value = live.struct.pack('3i', 101, 0, 0)
            with self.assertRaisesRegex(ValueError, 'PEER_IDENTITY'):
                live._peer(path, 100)

    def test_non_node_certificate_or_insecure_auth_refuses_before_network(self):
        temp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        config_path = temp / 'kubelet.conf'
        config_path.write_bytes(b'fixture without credentials')
        user = {'client-certificate': str(temp / 'client.crt'), 'client-key': str(temp / 'client.key')}
        config = dict(contexts=[dict(name='current', context=dict(cluster='cluster', user='user'))],
            clusters=[dict(name='cluster', cluster=dict(server='https://example.invalid:6443'))],
            users=[dict(name='user', user=user)], **{'current-context': 'current'})
        with patch.dict(sys.modules, {'yaml': SimpleNamespace(safe_load=lambda raw: config)}), \
             patch.object(live.ssl._ssl, '_test_decode_cert', return_value={'subject': ((('commonName', 'kubernetes-admin'),),)}), \
             patch.object(live.urllib.request, 'build_opener') as network:
            with self.assertRaisesRegex(ValueError, 'NODE_CLIENT_CN'):
                live.node_client(config_path)
            config['clusters'][0]['cluster']['insecure-skip-tls-verify'] = True
            with self.assertRaisesRegex(ValueError, 'AUTHENTICATION_REQUIRED'):
                live.node_client(config_path)
            config['clusters'][0]['cluster'].pop('insecure-skip-tls-verify')
            user['token'] = 'fixture-token'
            with self.assertRaisesRegex(ValueError, 'AUTHENTICATION_REQUIRED'):
                live.node_client(config_path)
            network.assert_not_called()


class JointReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.checkpoint = self.temp / 'checkpoint'; self.checkpoint.write_bytes(b'preserved checkpoint')
        self.config = self.temp / 'client.conf'; self.config.write_bytes(b'fixture, no credentials')
        self.blockers = [dict(reason='KUBELET_DEVICE_ALLOCATION', card=5, pod_uid='old-pod',
            checkpoint=str(self.checkpoint), checkpoint_sha256=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest())]
        self.sources = [dict(pid=100, root_dir='/observed/root', checkpoint=str(self.checkpoint), kubeconfig=str(self.config))]
        self.oci = [dict(task='infra', pod_uid='another-pod')]
        self.rows = [dict(uid='current', name='pod', namespace='ns', containers=[dict(kind='containers', name='worker')])]
        self.proof = dict(node_uid='node-uid', pod_count=1)
        self.identity = dict(node_name='node', client_cn='system:node:node',
                             kubeconfig_sha256=hashlib.sha256(self.config.read_bytes()).hexdigest())
        self.enterContext(patch.object(live, 'process_lifetime', return_value=dict(pid=100, start_ticks='20')))
        self.enterContext(patch.object(live.subprocess, 'check_output', return_value='Kubernetes v1.28.15\n'))
        self.enterContext(patch.object(live.Path, 'read_text', return_value='machine-or-boot'))
        self.enterContext(patch.object(live, 'node_client', return_value=(None, self.identity)))
        self.snapshot = self.enterContext(patch.object(live, 'node_snapshot', side_effect=lambda *a: (copy.deepcopy(self.rows), dict(self.proof))))
        self.query = self.enterContext(patch.object(live, 'live_list', return_value=(
            [dict(name='pod', namespace='ns', containers=[dict(name='worker', devices=[])])], dict(grpc_status=0))))

    def review(self):
        return live.review_stale_allocations(self.blockers, self.sources, self.oci)

    def test_no_live_owner_requires_joint_evidence_and_keeps_checkpoint_unchanged(self):
        raw = self.checkpoint.read_bytes()
        proof = self.review()
        self.assertEqual(proof['status'], 'NO_LIVE_OWNER_CORROBORATED')
        self.assertFalse(proof['internal_allocation_clearance_proven'])
        self.assertFalse(proof['driver_initialization_available_proven'])
        self.assertFalse(proof['exclusive_device_reservation'])
        self.assertEqual(proof['checkpoint_pod_uids'], ['old-pod'])
        self.assertEqual(self.checkpoint.read_bytes(), raw)
        self.assertEqual(self.snapshot.call_count, 2)

    def test_visible_allocation_overrides_absent_api_pod(self):
        self.query.return_value[0][0]['containers'][0]['devices'] = [dict(resource='huawei.com/Ascend910', device_ids=['Ascend910-5'])]
        with self.assertRaisesRegex(ValueError, 'LIVE_ACCELERATOR'):
            self.review()

    def test_unknown_live_resource_or_device_id_never_exempts_checkpoint(self):
        for resource, device_id in (('vendor.example/opaque-device', 'device'), ('huawei.com/Ascend910', 'opaque-id')):
            self.query.return_value[0][0]['containers'][0]['devices'] = [dict(resource=resource, device_ids=[device_id])]
            with self.subTest(resource=resource), self.assertRaisesRegex(ValueError, 'UNREVIEWED'):
                self.review()

    def test_checkpoint_pod_in_api_or_oci_cannot_be_exempted(self):
        self.rows[0]['uid'] = 'old-pod'
        with self.assertRaisesRegex(ValueError, 'STILL_IN_NODE_API'):
            self.review()
        self.rows[0]['uid'] = 'current'; self.oci[0]['pod_uid'] = 'old-pod'
        with self.assertRaisesRegex(ValueError, 'OCI_PRESENT'):
            self.review()

    def test_unknown_oci_owner_missing_client_or_ambiguous_source_refuses(self):
        self.oci[0]['pod_uid'] = None
        with self.assertRaises(ValueError): self.review()
        self.oci[0]['pod_uid'] = 'infra'; self.sources[0]['kubeconfig'] = None
        with self.assertRaises(ValueError): self.review()
        self.sources[0]['kubeconfig'] = str(self.config)
        self.sources.append(dict(self.sources[0], pid=101))
        with self.assertRaises(ValueError): self.review()

    def test_api_changes_or_podresources_scope_mismatch_refuse(self):
        different = copy.deepcopy(self.rows); different[0]['uid'] = 'new-pod'
        self.snapshot.side_effect = [(self.rows, self.proof), (different, self.proof)]
        with self.assertRaisesRegex(ValueError, 'SCOPE_CHANGED'): self.review()
        self.snapshot.side_effect = lambda *a: (self.rows, self.proof)
        self.query.return_value[0][0]['name'] = 'new-pod'
        with self.assertRaisesRegex(ValueError, 'SCOPE_DIFFER'): self.review()

    def test_checkpoint_change_query_failure_or_process_reuse_refuses(self):
        with patch.object(live, 'process_lifetime', side_effect=[dict(pid=100, start_ticks='20'), dict(pid=100, start_ticks='21')]):
            with self.assertRaisesRegex(ValueError, 'PROCESS_OR_CLIENT_CHANGED'): self.review()
        self.query.side_effect = TimeoutError('fixture timeout')
        with self.assertRaises(TimeoutError): self.review()
        self.query.side_effect = None; self.checkpoint.write_bytes(b'new allocation')
        with self.assertRaisesRegex(ValueError, 'CHECKPOINT_CHANGED'): self.review()

    def test_other_blocker_cannot_enter_stale_review(self):
        self.blockers.append(dict(reason='NPU_PROCESS', card=5))
        with self.assertRaisesRegex(ValueError, 'OTHER_GATE_BLOCKERS'): self.review()


class RunnerIntegrationTests(unittest.TestCase):
    def run_gate(self, *, extra_blocker=False, query_error=False, post_busy=False):
        from test_device_allocation_gate import gate
        original = gate.idle_gate
        blocker = dict(reason='KUBELET_DEVICE_ALLOCATION', card=5, pod_uid='old-pod')
        runtime = dict(blockers=[blocker], device_access_grants=[], oci_tasks=[], containerd_states=[],
                       kubelet_checkpoint_sources=[], kubelet_allocation_checks=[])
        if extra_blocker:
            runtime['blockers'].append(dict(reason='RUNTIME_ALLOCATION_SOURCE_UNKNOWN'))
        smi = 'Health\n| 5 910B4 | OK |\nProcess id\nNo running processes'
        post = dict(copy.deepcopy(runtime), time='post', npu_smi=smi)
        if post_busy:
            post['blockers'].append(dict(reason='NPU_PROCESS', card=5, pid=77))
        with patch.dict(sys.modules, {'kubelet_allocations': live}), \
             patch.object(live, 'review_stale_allocations', side_effect=TimeoutError('timeout') if query_error else None,
                          return_value=dict(status='NO_LIVE_OWNER_CORROBORATED', evidence_sha256='a' * 64)) as review, \
             patch.object(gate, 'runtime_device_inventory', return_value=runtime), \
             patch.object(gate, 'command', side_effect=lambda args: smi if args[0] == 'npu-smi' else ''), \
             patch.object(gate.Path, 'stat', return_value=SimpleNamespace(st_rdev=99)), \
             patch.object(gate.Path, 'iterdir', return_value=iter(())), \
             patch.object(gate, 'idle_gate', return_value=post):
            result = original([5])
        return result, blocker, review.call_count

    def test_verified_stale_preserves_original_blocker_and_release_proof(self):
        report, blocker, calls = self.run_gate()
        self.assertEqual(calls, 1)
        self.assertTrue(report['idle'])
        self.assertEqual(report['blockers'], [])
        stale = report['stale_checkpoint_allocations'][0]
        self.assertEqual(stale['original_blocker'], blocker)
        self.assertEqual(stale['release_evidence_sha256'], report['stale_checkpoint_review']['evidence_sha256'])

    def test_unknown_query_keeps_original_allocation_blocking(self):
        report, blocker, _ = self.run_gate(query_error=True)
        self.assertFalse(report['idle'])
        self.assertIn(blocker, report['blockers'])
        self.assertNotIn('stale_checkpoint_allocations', report)

    def test_new_compute_in_post_check_prevents_stale_exemption(self):
        report, blocker, _ = self.run_gate(post_busy=True)
        self.assertFalse(report['idle'])
        self.assertIn(blocker, report['blockers'])
        self.assertEqual(report['stale_checkpoint_post_gate']['blockers'][-1]['pid'], 77)

    def test_any_other_blocker_skips_live_exemption(self):
        report, _, calls = self.run_gate(extra_blocker=True)
        self.assertFalse(report['idle'])
        self.assertEqual(calls, 0)


if __name__ == '__main__':
    unittest.main()
