"""Read-only corroboration of stale kubelet checkpoints; unknown is never idle.

Only Kubernetes v1.28.15's v1 PodResources List is reviewed here. Its response
is joined to two complete, node-authenticated API snapshots. GetAllocatable
is deliberately not used: it reports capacity, not current free devices.
"""
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import ssl
import struct
import subprocess
import time
import urllib.parse
import urllib.request

VERSION = 'v1.28.15'
PROTO_URL = ('https://github.com/kubernetes/kubernetes/blob/v1.28.15/'
             'staging/src/k8s.io/kubelet/pkg/apis/podresources/v1/api.proto')
RPC = '/v1.PodResourcesLister/List'
MAX_REPLY = 1024 * 1024


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _text(value):
    require(isinstance(value, str) and 0 < len(value) <= 512 and '\x00' not in value,
            'LIVE_IDENTITY_UNRECOGNIZED')
    return value


def _varint(raw, offset):
    value = 0
    for shift in range(0, 70, 7):
        require(offset < len(raw), 'PROTOBUF_TRUNCATED_VARINT')
        byte = raw[offset]
        offset += 1
        require(shift < 63 or byte <= 1, 'PROTOBUF_VARINT_OVERFLOW')
        value |= (byte & 127) << shift
        if not byte & 128:
            return value, offset
    raise ValueError('PROTOBUF_VARINT_OVERFLOW')


def _fields(raw, allowed):
    offset = 0
    while offset < len(raw):
        key, offset = _varint(raw, offset)
        number, wire = key >> 3, key & 7
        require(number in allowed and wire in allowed[number], 'PODRESOURCES_UNKNOWN_FIELD_OR_WIRE')
        if wire == 0:
            value, offset = _varint(raw, offset)
        else:
            require(wire == 2, 'PODRESOURCES_UNREVIEWED_WIRE')
            size, offset = _varint(raw, offset)
            require(size <= len(raw) - offset, 'PROTOBUF_TRUNCATED_FIELD')
            value = raw[offset:offset + size]
            offset += size
        yield number, wire, value


def _topology(raw):
    for _, _, node in _fields(raw, {1: (2,)}):
        list(_fields(node, {1: (0,)}))


def _device(raw):
    result = {'resource': None, 'device_ids': []}
    for field, _, value in _fields(raw, {1: (2,), 2: (2,), 3: (2,)}):
        if field == 1:
            require(result['resource'] is None, 'DUPLICATE_RESOURCE_NAME')
            result['resource'] = _text(value.decode('utf-8'))
        elif field == 2:
            result['device_ids'].append(_text(value.decode('utf-8')))
        else:
            _topology(value)
    require(result['resource'] and result['device_ids']
            and len(set(result['device_ids'])) == len(result['device_ids']), 'INCOMPLETE_DEVICE_ASSIGNMENT')
    return result


def _container(raw):
    result = {'name': None, 'devices': []}
    for field, wire, value in _fields(raw, {1: (2,), 2: (2,), 3: (0, 2), 4: (2,), 5: (2,)}):
        if field == 1:
            require(result['name'] is None, 'DUPLICATE_CONTAINER_NAME')
            result['name'] = _text(value.decode('utf-8'))
        elif field == 2:
            result['devices'].append(_device(value))
        elif field == 3 and wire == 2:
            offset = 0
            while offset < len(value):
                _, offset = _varint(value, offset)
        elif field == 4:
            for member, _, data in _fields(value, {1: (2,), 2: (0,), 3: (2,)}):
                if member == 1:
                    _text(data.decode('utf-8'))
                elif member == 3:
                    _topology(data)
        elif field == 5:
            raise ValueError('DYNAMIC_RESOURCE_ASSIGNMENT_UNREVIEWED')
    require(result['name'], 'CONTAINER_NAME_MISSING')
    return result


def parse_list_response(body, headers):
    require(isinstance(body, bytes) and 5 <= len(body) <= MAX_REPLY, 'GRPC_REPLY_SIZE_UNPROVEN')
    lines = [line.strip().lower() for line in headers.decode('ascii').splitlines() if line.strip()]
    require([line for line in lines if line.startswith('http/')] == ['http/2 200']
            and [line for line in lines if line.startswith('grpc-status:')] == ['grpc-status: 0']
            and [line for line in lines if line.startswith('content-type:')] == ['content-type: application/grpc'],
            'GRPC_TRANSPORT_OR_STATUS_UNPROVEN')
    require(body[0] == 0 and int.from_bytes(body[1:5], 'big') == len(body) - 5,
            'GRPC_TRUNCATED_COMPRESSED_OR_MULTIPLE_MESSAGES')
    pods = []
    for _, _, raw in _fields(body[5:], {1: (2,)}):
        row = {'name': None, 'namespace': None, 'containers': []}
        for field, _, value in _fields(raw, {1: (2,), 2: (2,), 3: (2,)}):
            if field == 3:
                row['containers'].append(_container(value))
            else:
                name = 'name' if field == 1 else 'namespace'
                require(row[name] is None, 'DUPLICATE_POD_IDENTITY')
                row[name] = _text(value.decode('utf-8'))
        require(row['name'] and row['namespace'] and row['containers'], 'POD_IDENTITY_OR_CONTAINERS_MISSING')
        require(len({c['name'] for c in row['containers']}) == len(row['containers']), 'DUPLICATE_CONTAINER')
        pods.append(row)
    require(len(pods) <= 8192 and len({(p['namespace'], p['name']) for p in pods}) == len(pods),
            'DUPLICATE_OR_UNBOUNDED_POD_LIST')
    return pods


def process_lifetime(pid):
    proc = Path('/proc') / str(pid)
    require((proc / 'comm').read_text().strip() == 'kubelet', 'SOCKET_PROCESS_IS_NOT_KUBELET')
    fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=fields[19], executable=os.readlink(proc / 'exe'),
                boot_id_sha256=hashlib.sha256(Path('/proc/sys/kernel/random/boot_id').read_bytes()).hexdigest())


def _peer(path, pid):
    before = path.stat()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(5)
        connection.connect(str(path))
        peer = struct.unpack('3i', connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
    require(peer == (pid, 0, 0), 'PODRESOURCES_PEER_IDENTITY_UNPROVEN')
    return dict(pid=peer[0], uid=peer[1], gid=peer[2], device=before.st_dev, inode=before.st_ino)


def live_list(path, pid):
    peer = _peer(path, pid)
    command = ['curl', '--silent', '--show-error', '--noproxy', '*', '--http2-prior-knowledge',
        '--unix-socket', str(path), '--max-time', '10', '--max-filesize', str(MAX_REPLY),
        '--dump-header', '/dev/stderr', '--request', 'POST', '--header', 'Content-Type: application/grpc',
        '--header', 'TE: trailers', '--data-binary', '@-', 'http://localhost' + RPC]
    result = subprocess.run(command, input=b'\0' * 5, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=15, check=False)
    require(result.returncode == 0, 'PODRESOURCES_QUERY_FAILED')
    rows = parse_list_response(result.stdout, result.stderr)
    require(_peer(path, pid) == peer, 'PODRESOURCES_SOCKET_CHANGED')
    return rows, dict(rpc=RPC, official_proto=PROTO_URL, socket=str(path), peer=peer,
        payload_sha256=hashlib.sha256(result.stdout).hexdigest(), headers_sha256=hashlib.sha256(result.stderr).hexdigest(),
        pod_count=len(rows), container_count=sum(len(p['containers']) for p in rows), http_status=200, grpc_status=0)


def node_client(path):
    import yaml
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    def unique(rows, name, member):
        matches = [row[member] for row in rows if row['name'] == name]
        require(len(matches) == 1, 'KUBELET_CLIENT_CONTEXT_AMBIGUOUS')
        return matches[0]
    context = unique(config['contexts'], config['current-context'], 'context')
    cluster = unique(config['clusters'], context['cluster'], 'cluster')
    user = unique(config['users'], context['user'], 'user')
    require(set(user) == {'client-certificate', 'client-key'} and not cluster.get('insecure-skip-tls-verify'),
            'NODE_CERTIFICATE_AUTHENTICATION_REQUIRED')
    cert, key = Path(user['client-certificate']), Path(user['client-key'])
    require(cert.is_absolute() and key.is_absolute(), 'ABSOLUTE_CLIENT_CERTIFICATE_PATH_REQUIRED')
    common_names = [value for group in ssl._ssl._test_decode_cert(str(cert))['subject']
                    for name, value in group if name == 'commonName']
    require(len(common_names) == 1 and common_names[0].startswith('system:node:'), 'NODE_CLIENT_CN_UNPROVEN')
    node_name = _text(common_names[0][len('system:node:'):])
    server = cluster['server'].rstrip('/')
    parsed = urllib.parse.urlsplit(server)
    require(parsed.scheme == 'https' and parsed.netloc and not parsed.username and not parsed.password
            and not parsed.path and not parsed.query and not parsed.fragment, 'VERIFIED_API_SERVER_REQUIRED')
    if 'certificate-authority-data' in cluster:
        ca = base64.b64decode(cluster['certificate-authority-data'], validate=True).decode('ascii')
        tls = ssl.create_default_context(cadata=ca)
    else:
        ca_path = Path(cluster['certificate-authority'])
        require(ca_path.is_absolute(), 'ABSOLUTE_CA_PATH_REQUIRED')
        tls = ssl.create_default_context(cafile=str(ca_path))
    tls.load_cert_chain(str(cert), str(key))
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=tls))
    def get(suffix):
        with client.open(server + suffix, timeout=10) as response:
            value = response.read(20 * MAX_REPLY + 1)
            require(response.status == 200 and len(value) <= 20 * MAX_REPLY, 'NODE_API_RESPONSE_INCOMPLETE')
        return json.loads(value), hashlib.sha256(value).hexdigest()
    return get, dict(client_cn=common_names[0], node_name=node_name, server=server,
                     kubeconfig=str(path), kubeconfig_sha256=hashlib.sha256(raw).hexdigest())


def node_snapshot(get, identity, machine_id, boot_id):
    name = identity['node_name']
    node, node_sha = get('/api/v1/nodes/' + urllib.parse.quote(name, safe=''))
    pods, pods_sha = get('/api/v1/pods?' + urllib.parse.urlencode({'fieldSelector': 'spec.nodeName=' + name, 'limit': 500}))
    require(node.get('kind') == 'Node' and node.get('apiVersion') == 'v1' and node['metadata']['name'] == name,
            'NODE_API_IDENTITY_MISMATCH')
    info = node['status']['nodeInfo']
    require(info['machineID'] == machine_id and info['bootID'] == boot_id and info['kubeletVersion'] == VERSION,
            'NODE_MACHINE_BOOT_OR_VERSION_MISMATCH')
    ready = [row for row in node['status']['conditions'] if row.get('type') == 'Ready']
    require(len(ready) == 1 and ready[0]['status'] == 'True', 'NODE_READINESS_UNPROVEN')
    heartbeat = datetime.fromisoformat(ready[0]['lastHeartbeatTime'].replace('Z', '+00:00'))
    require(-30 <= (datetime.now(timezone.utc) - heartbeat).total_seconds() <= 600, 'NODE_HEARTBEAT_STALE')
    require(pods.get('kind') == 'PodList' and pods.get('apiVersion') == 'v1'
            and isinstance(pods.get('items'), list) and not pods['metadata'].get('continue'), 'NODE_POD_LIST_INCOMPLETE')
    rows = []
    for pod in pods['items']:
        meta, spec, status = pod['metadata'], pod['spec'], pod['status']
        require(spec['nodeName'] == name and status.get('phase') in ('Pending', 'Running', 'Succeeded', 'Failed'),
                'POD_NODE_OR_PHASE_UNKNOWN')
        require(not spec.get('resourceClaims'), 'POD_DYNAMIC_RESOURCE_CLAIM_UNREVIEWED')
        row = dict(uid=_text(meta['uid']), name=_text(meta['name']), namespace=_text(meta['namespace']),
                   phase=status['phase'], deleting=meta.get('deletionTimestamp'), containers=[])
        for kind in ('containers', 'initContainers', 'ephemeralContainers'):
            entries = spec.get(kind, [])
            require(isinstance(entries, list), 'CONTAINER_SCHEMA_UNKNOWN')
            for container in entries:
                resources = container.get('resources', {})
                require(isinstance(resources, dict) and not resources.get('claims'), 'CONTAINER_RESOURCE_CLAIM_UNREVIEWED')
                for bucket in ('requests', 'limits'):
                    values = resources.get(bucket, {})
                    require(isinstance(values, dict), 'RESOURCE_QUANTITY_SCHEMA_UNKNOWN')
                    for resource, quantity in values.items():
                        require(isinstance(resource, str), 'RESOURCE_NAME_UNKNOWN')
                        require(resource in ('cpu', 'memory', 'ephemeral-storage', 'huawei.com/Ascend910')
                                or resource.startswith('hugepages-'), 'EXTENDED_RESOURCE_REQUEST_UNREVIEWED')
                        if any(token in resource.lower() for token in ('ascend', 'npu')):
                            require(resource == 'huawei.com/Ascend910' and type(quantity) in (int, str)
                                    and re.fullmatch(r'0|[1-9]\d*', str(quantity)), 'ACCELERATOR_REQUEST_UNKNOWN')
                            require(not int(quantity) or status['phase'] in ('Succeeded', 'Failed'),
                                    'CURRENT_NODE_ACCELERATOR_REQUEST')
                row['containers'].append(dict(kind=kind, name=_text(container['name']), resources=resources))
        require(any(c['kind'] == 'containers' for c in row['containers']), 'REGULAR_CONTAINERS_MISSING')
        require(len({(c['kind'], c['name']) for c in row['containers']}) == len(row['containers']), 'DUPLICATE_API_CONTAINER')
        row['containers'].sort(key=lambda c: (c['kind'], c['name']))
        rows.append(row)
    require(len({(r['namespace'], r['name']) for r in rows}) == len(rows)
            and len({r['uid'] for r in rows}) == len(rows), 'DUPLICATE_API_POD')
    rows.sort(key=lambda row: (row['namespace'], row['name']))
    return rows, dict(node_uid=_text(node['metadata']['uid']), node_name=name, node_sha256=node_sha,
        pods_sha256=pods_sha, pods_resource_version=pods['metadata']['resourceVersion'],
        node_resource_version=node['metadata']['resourceVersion'], pod_count=len(rows),
        pod_scope_sha256=digest(rows), machine_id_matches=True, boot_id_matches=True, ready=True, kubelet_version=VERSION)


def review_stale_allocations(blockers, sources, oci_tasks):
    started = time.monotonic()
    require(blockers and all(row.get('reason') == 'KUBELET_DEVICE_ALLOCATION' for row in blockers),
            'OTHER_GATE_BLOCKERS_PRESENT')
    paths = {row['checkpoint']: row['checkpoint_sha256'] for row in blockers}
    matches = [row for row in sources if row['checkpoint'] in paths]
    require({row['checkpoint'] for row in matches} == set(paths)
            and len({(row['pid'], row['root_dir'], row.get('kubeconfig')) for row in matches}) == 1,
            'ACTIVE_CHECKPOINT_SOURCE_AMBIGUOUS')
    source = matches[0]
    require(source.get('kubeconfig') and Path(source['kubeconfig']).is_absolute(), 'ACTIVE_KUBELET_CLIENT_UNPROVEN')
    target_uids = {row['pod_uid'] for row in blockers}
    require(all(row.get('pod_uid') and row['pod_uid'] not in target_uids for row in oci_tasks),
            'CHECKPOINT_POD_OCI_PRESENT_OR_UNKNOWN')
    def checkpoints_unchanged():
        require(all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha for path, sha in paths.items()),
                'CHECKPOINT_CHANGED_DURING_REVIEW')
    checkpoints_unchanged()
    pid = source['pid']
    lifetime = process_lifetime(pid)
    version = subprocess.check_output([str(Path('/proc') / str(pid) / 'exe'), '--version'], text=True, timeout=5).strip()
    require(version == 'Kubernetes ' + VERSION, 'LIVE_KUBELET_VERSION_UNREVIEWED')
    config_path = Path(source['kubeconfig'])
    get, identity = node_client(config_path)
    machine = Path('/etc/machine-id').read_text().strip()
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    before, before_proof = node_snapshot(get, identity, machine, boot)
    live, live_proof = live_list(Path(source['root_dir']) / 'pod-resources/kubelet.sock', pid)
    after, after_proof = node_snapshot(get, identity, machine, boot)
    require(before == after and before_proof['node_uid'] == after_proof['node_uid'], 'NODE_SCOPE_CHANGED_DURING_REVIEW')
    require(not target_uids.intersection(row['uid'] for row in after), 'CHECKPOINT_POD_STILL_IN_NODE_API')
    expected = {(p['namespace'], p['name']): sorted(c['name'] for c in p['containers'] if c['kind'] == 'containers') for p in after}
    observed = {(p['namespace'], p['name']): sorted(c['name'] for c in p['containers']) for p in live}
    require(observed == expected, 'PODRESOURCES_AND_NODE_API_SCOPE_DIFFER')
    for pod in live:
        for container in pod['containers']:
            for device in container['devices']:
                require(device['resource'] == 'huawei.com/Ascend910'
                        and all(re.fullmatch(r'Ascend910-(0|[1-9]\d*)', value) for value in device['device_ids']),
                        'LIVE_DEVICE_RESOURCE_UNREVIEWED')
                raise ValueError('LIVE_ACCELERATOR_ALLOCATION_PRESENT')
    require(process_lifetime(pid) == lifetime and hashlib.sha256(config_path.read_bytes()).hexdigest() == identity['kubeconfig_sha256'],
            'KUBELET_PROCESS_OR_CLIENT_CHANGED')
    checkpoints_unchanged()
    require(time.monotonic() - started <= 30, 'ALLOCATION_REVIEW_WINDOW_EXCEEDED')
    proof = dict(schema='demoncat.kubelet_stale_checkpoint_review.v1', status='NO_LIVE_OWNER_CORROBORATED',
        time=datetime.now(timezone.utc).isoformat(), kubelet=lifetime, client=identity, before=before_proof,
        podresources=live_proof, after=after_proof, checkpoint_sha256=paths,
        checkpoint_pod_uids=sorted(target_uids), oci_tasks_sha256=digest(oci_tasks),
        machine_id_sha256=hashlib.sha256(machine.encode()).hexdigest(), boot_id_sha256=hashlib.sha256(boot.encode()).hexdigest(),
        internal_allocation_clearance_proven=False, driver_initialization_available_proven=False,
        exclusive_device_reservation=False,
        limitation='READ_ONLY_NO_LIVE_OWNER_SNAPSHOT_NOT_DRIVER_OR_INTERNAL_ALLOCATION_CLEARANCE')
    return dict(proof, evidence_sha256=digest(proof))
