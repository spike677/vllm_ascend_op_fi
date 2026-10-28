"""Bounded all-eligible-hit B/F runner; never stop or remove external containers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time
import traceback
import uuid

ROOT = Path('/home/fault15-alias-detach-allhit-20261028')
IMAGE = 'sha256:8988c3cd3a971a72ebb2e5baae24e0091cde55146f96d2b4c8229448210735b9'
TASK = 'fault15-alias-detach-allhit-20261028'
PROVEN_OWNED_PROCESSES = {}
OWNED_TEARDOWN_SINCE = {}
OWNED_TEARDOWN_GRACE_SECONDS = 60
ALLOW_IDLE_IDS = set()
IDLE_BASELINES = {}
ATTEMPT_EVIDENCE = None
REVIEWED_IDLE_CONTAINERS = {
    'a8eeda3a5db34389fbcc83fdd8cabc99d687ac1db5aa4011a0586f2a9f0ddc4c': {
        'name': 'gyl_test', 'image': 'sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0',
        'command': ['/bin/bash']},
    'db21bd452c30c64dfd61152b6cf0a98bebfcea468a0f1c8a18452d2cb0e99de1': {
        'name': 'catmonitor-demo-control', 'image': 'sha256:9858f14b8b562a21e0429e222064998e260085dc9d58c07519879b462b806acc',
        'command': ['daemon', '--config', '/etc/catmonitor/catmonitor.yaml']},
}
IDLE_REVIEW_SOURCE = {'utc': '2026-10-28T07:30:13.196756+00:00',
    'source': 'Read-only docker inspect of exact full IDs on npu-dual; no Config.Env readout. Earlier proc-root stat confirmed both physical device nodes visible.'}


def run(argv, timeout=25):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        details = f'{argv[:3]!r}: rc={result.returncode}: {result.stderr[-2000:]}'
        if len(argv) > 1 and Path(argv[1]).name == 'analyze_alias.py':
            details += '\nComplete analyzer stdout:\n' + result.stdout
        raise RuntimeError(details)
    return result.stdout


def require(value, message):
    if not value:
        raise RuntimeError(message)


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)


def source_manifest():
    names = sorted(p.name for p in ROOT.glob('*.py')) + [
        'env.sh', 'setup/logical_reference.json', 'setup/acceptance_contract.json',
        'tools/post_resource_check.py']
    for needed in ('alias_probe.py', 'analyze_alias.py', 'controller.py'):
        require(needed in names, 'Missing runtime source: ' + needed)
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def parse_proc_stat(text):
    prefix, closing, tail = text.rpartition(')')
    require(closing and '(' in prefix, 'Unrecognized /proc stat')
    fields = tail.split()
    require(len(fields) > 19, 'Incomplete /proc stat')
    flags = int(fields[6])  # /proc/PID/stat field 9; PF_EXITING == 0x4.
    return {'host_pid': int(prefix.split('(', 1)[0].strip()),
            'process_state': fields[0], 'starttime_ticks': int(fields[19]),
            'process_flags': flags, 'pf_exiting': bool(flags & 0x4)}


def read_process_identity(pid):
    path = Path('/proc') / str(pid)
    before = parse_proc_stat((path / 'stat').read_text())
    cgroup = (path / 'cgroup').read_text()
    status = (path / 'status').read_text().splitlines()
    vectors = [line for line in status if line.startswith('NSpid:')]
    require(len(vectors) == 1, 'Missing PID namespace vector')
    vector = [int(value) for value in vectors[0].split()[1:]]
    namespace = os.readlink(path / 'ns/pid')
    after = parse_proc_stat((path / 'stat').read_text())
    require(before['host_pid'] == after['host_pid'] == pid
            and before['starttime_ticks'] == after['starttime_ticks']
            and vector and vector[0] == pid and re.fullmatch(r'pid:\[\d+\]', namespace),
            'Process identity changed during observation')
    return {**after, 'cgroup': cgroup, 'nspid_vector': vector, 'pid_namespace': namespace,
            'state_before_read': before['process_state']}


def classify_process_identity(identity, owned, proven=None, teardown_since=None, now=None):
    proven = PROVEN_OWNED_PROCESSES if proven is None else proven
    teardown_since = OWNED_TEARDOWN_SINCE if teardown_since is None else teardown_since
    now = time.monotonic() if now is None else now
    fields = ('host_pid', 'starttime_ticks', 'pid_namespace', 'nspid_vector')
    key = (owned, identity['host_pid'])
    teardown_key = (*key, identity['starttime_ticks'])
    if owned is not None and owned in identity['cgroup']:
        proven[key] = {field: identity[field] for field in fields}
        teardown_since.pop(teardown_key, None)
        return True, 'CURRENT_OWNED_CGROUP', None
    previous = proven.get(key)
    groups = [line.split(':', 2) for line in identity['cgroup'].splitlines()]
    root_only = bool(groups) and all(len(parts) == 3 and parts[2] == '/' for parts in groups)
    same = previous is not None and all(previous[field] == identity[field] for field in fields)
    exited = identity['process_state'] in ('Z', 'X', 'x')
    exiting = bool(identity.get('process_flags', 0) & 0x4)
    if owned is not None and root_only and same and (exited or exiting):
        age = now - teardown_since.setdefault(teardown_key, now)
        identity.update(teardown_age_seconds=age, teardown_grace_seconds=OWNED_TEARDOWN_GRACE_SECONDS)
        if 0 <= age <= OWNED_TEARDOWN_GRACE_SECONDS:
            return True, ('PROVEN_OWNED_EXITED_CGROUP_TEARDOWN' if exited
                          else 'PROVEN_OWNED_EXITING_CGROUP_TEARDOWN'), previous.copy()
        return False, 'PROVEN_OWNED_TEARDOWN_GRACE_EXCEEDED', previous.copy()
    return False, 'UNPROVEN_OR_LIVE_FOREIGN_PROCESS', None


def verify_owned(obj, token, name, expected_id):
    labels = obj['Config'].get('Labels') or {}
    require(obj['Id'] == expected_id and obj['Image'] == IMAGE
            and labels.get('demoncat.controller_token') == token
            and labels.get('demoncat.task') == TASK and obj['Name'].lstrip('/') == name,
            'Refuse cleanup of unowned container')


def recover_owned_id(token, name):
    """Recover an uncertain docker-run result only through this random token."""
    ids = run(['docker', 'ps', '-a', '-q', '--no-trunc',
               '--filter', 'label=demoncat.controller_token=' + token,
               '--filter', 'label=demoncat.task=' + TASK]).split()
    require(len(ids) <= 1 and all(re.fullmatch('[0-9a-f]{64}', value) for value in ids),
            'Ambiguous token-owned container recovery')
    if not ids:
        return None
    objects = json.loads(run(['docker', 'inspect', ids[0]]))
    require(len(objects) == 1, 'Ambiguous recovered inspect result')
    verify_owned(objects[0], token, name, ids[0])
    return ids[0]


def device_capability(obj, card=None):
    """Describe all NPU routes, or routes capable of the selected physical card.

    Shared manager nodes do not grant a different card's device-node access to
    an otherwise restricted container. Unknown cgroup/request grants fail closed.
    """
    h = obj['HostConfig']
    reasons = []
    if h.get('Privileged'):
        reasons.append('privileged_all_host_devices_possible')
    if h.get('DeviceCgroupRules'):
        reasons.append('device_cgroup_rules_selected_access_not_proven_restricted')
    if h.get('DeviceRequests'):
        reasons.append('device_requests_selected_access_not_proven_restricted')
    for item in h.get('Devices') or []:
        path = item.get('PathOnHost', '')
        if re.fullmatch(r'/dev/(?:davinci(?:[0-9]+|_manager)|devmm_svm|hisi_hdc)', path):
            if card is None or path == f'/dev/davinci{card}':
                reasons.append('explicit_device:' + path)
    for item in obj.get('Mounts') or []:
        path = item.get('Source', '').rstrip('/') or '/'
        if path in ('/', '/dev'):
            reasons.append('device_capable_bind:' + path)
        elif re.fullmatch(r'/dev/(?:davinci(?:[0-9]+|_manager)|devmm_svm|hisi_hdc)', path):
            if card is None or path == f'/dev/davinci{card}':
                reasons.append('device_capable_bind:' + path)
    return sorted(set(reasons))


def idle_metadata(obj):
    h = obj['HostConfig']
    return {'id': obj['Id'], 'name': obj['Name'].lstrip('/'), 'image': obj['Image'],
            'command': obj['Config'].get('Cmd'), 'entrypoint': obj['Config'].get('Entrypoint'),
            'privileged': bool(h.get('Privileged')),
            'device_cgroup_rules': sorted(h.get('DeviceCgroupRules') or []),
            'device_requests': h.get('DeviceRequests') or [],
            'devices': sorted(h.get('Devices') or [], key=lambda x: (x.get('PathOnHost', ''), x.get('PathInContainer', ''))),
            'mounts': sorted(obj.get('Mounts') or [], key=lambda x: (x.get('Destination', ''), x.get('Source', ''))),
            'init_host_pid': obj['State'].get('Pid'), 'started_at': obj['State'].get('StartedAt'),
            'network_mode': h.get('NetworkMode'), 'pid_mode': h.get('PidMode'), 'ipc_mode': h.get('IpcMode')}


def append_resource_check(record):
    if ATTEMPT_EVIDENCE is not None:
        with (ATTEMPT_EVIDENCE / 'resource_checks.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
            stream.flush()


def resources(card, owned=None, strict=True):
    ids = run(['docker', 'ps', '-q', '--no-trunc']).split()
    require(all(re.fullmatch('[0-9a-f]{64}', value) for value in ids), 'Invalid running container identity')
    objects = json.loads(run(['docker', 'inspect', *ids])) if ids else []
    capable, unapproved = [], []
    for obj in objects:
        if obj['Id'] == owned:
            continue
        reasons = device_capability(obj)
        if not reasons:
            continue
        selected_reasons = device_capability(obj, card)
        metadata = idle_metadata(obj)
        allowed = obj['Id'] in ALLOW_IDLE_IDS
        expected = REVIEWED_IDLE_CONTAINERS.get(obj['Id'])
        reviewed_match = expected is None or all(metadata[key] == value for key, value in expected.items())
        if allowed and reviewed_match and obj['Id'] not in IDLE_BASELINES:
            IDLE_BASELINES[obj['Id']] = metadata
        unchanged = allowed and IDLE_BASELINES.get(obj['Id']) == metadata
        record = {'id': obj['Id'], 'capability_reasons': reasons, 'metadata': metadata,
                  'selected_card_capable': bool(selected_reasons),
                  'selected_card_capability_reasons': selected_reasons,
                  'explicit_idle_id_allowed': allowed, 'reviewed_identity_matches': reviewed_match,
                  'metadata_unchanged': unchanged,
                  'metadata_sha256': hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()}
        capable.append(record)
        if selected_reasons and not (allowed and reviewed_match and unchanged):
            unapproved.append(record)
    smi = run(['npu-smi', 'info'])
    require('Process id' in smi, 'Unrecognized npu-smi output')
    require(re.search(rf'^\|\s*{card}\s+910B4', smi, re.M), 'Requested 910B4 not visible')
    processes, foreign, accounted = [], [], set()
    for line in smi.split('Process id', 1)[1].splitlines():
        match = re.match(r'^\|\s*([25])\s+\d+\s*\|\s*(\d+)\s*\|', line)
        if not match:
            continue
        physical, pid = map(int, match.groups())
        accounted.add(physical)
        try:
            identity = read_process_identity(pid)
            identity['owned'], identity['ownership_evidence'], previous = classify_process_identity(identity, owned)
            if previous is not None:
                identity['prior_owned_identity'] = previous
        except FileNotFoundError:
            identity = {'host_pid': pid, 'owned': False, 'status': 'identity_unavailable_for_recorded_NPU_PID'}
        identity['physical_device'] = physical
        processes.append(identity)
        if not identity['owned'] or physical != card:
            foreign.append(identity)
    for device in (2, 5):
        require(device in accounted or f'No running processes found in NPU {device}' in smi,
                'NPU process table does not account for physical card ' + str(device))
    selected_foreign = [item for item in foreign if item['physical_device'] == card]
    idle = owned is not None or f'No running processes found in NPU {card}' in smi
    report = {'epoch': time.time(), 'selected_card': card, 'monitored_physical_cards': [2, 5],
              'npu_smi': smi, 'processes': processes, 'foreign_npu_processes': foreign,
              'selected_card_foreign_npu_processes': selected_foreign,
              'other_card_foreign_npu_processes': [item for item in foreign if item['physical_device'] != card],
              'device_capable_containers': capable, 'unapproved_device_capable_containers': unapproved,
              'idle_allowlist_ids': sorted(ALLOW_IDLE_IDS),
              'allowlisted_ids_not_running': sorted(ALLOW_IDLE_IDS - set(ids)),
              'eligible': not unapproved and not selected_foreign and idle,
              'scope': 'Selected-card resource gate; both cards recorded. Explicit idle coexistence is not device exclusivity.'}
    append_resource_check(report)
    if strict:
        require(report['eligible'], 'Resource gate rejected: ' + json.dumps({
            'unapproved_container_ids': [item['id'] for item in unapproved],
            'selected_card': card,
            'foreign_npu_pids': [item['host_pid'] for item in selected_foreign], 'preflight_idle': idle}))
    return report


def main():
    global ALLOW_IDLE_IDS, ATTEMPT_EVIDENCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--card', type=int, choices=(2, 5), default=5)
    parser.add_argument('--attempt')
    parser.add_argument('--max-seconds', type=int, default=600)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--allow-idle-container-ID', action='append', default=[], dest='allow_idle_ids',
                        help='Explicit full ID of an idle device-capable background container; repeat for each ID')
    args = parser.parse_args()
    PROVEN_OWNED_PROCESSES.clear()
    OWNED_TEARDOWN_SINCE.clear()
    ALLOW_IDLE_IDS = set(args.allow_idle_ids)
    require(len(ALLOW_IDLE_IDS) == len(args.allow_idle_ids) and all(re.fullmatch('[0-9a-f]{64}', value) for value in ALLOW_IDLE_IDS),
            'Idle allowlist requires unique exact 64-character IDs')
    IDLE_BASELINES.clear()
    sources = source_manifest()
    if args.prepare_only:
        print(json.dumps({'status': 'PREPARED_NO_NPU', 'sources': sources}))
        return 0
    require(args.attempt is not None and re.fullmatch(r'[A-Za-z0-9_-]{1,40}', args.attempt),
            'Fresh explicit attempt required')
    require(180 <= args.max_seconds <= 1200, 'Unsupported duration')
    import fcntl
    evidence = ROOT / 'outputs' / args.attempt
    evidence.mkdir(parents=True, exist_ok=False)
    ATTEMPT_EVIDENCE = evidence
    name = 'dcat-f15-allhit-' + args.attempt
    token = uuid.uuid4().hex
    lock = None
    lock_acquired = False
    create_attempted = False
    owned = None
    child = None
    start = time.monotonic()
    result = {'status': 'RUNNING', 'card': args.card, 'stages': [], 'cleanup_errors': [],
              'start_epoch': time.time(), 'scope': 'mechanism microbenchmark, not model tokens',
              'idle_allowlist_ids': sorted(ALLOW_IDLE_IDS), 'external_container_mutations': False}

    def interrupted(signum, frame):
        raise RuntimeError('Interrupted: ' + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        lock = open('/var/tmp/fault22_npu_2_5.lock', 'a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_acquired = True
        save(evidence / 'before.json', resources(args.card))
        save(evidence / 'resource_policy.json', {'idle_allowlist_ids': sorted(ALLOW_IDLE_IDS),
             'baseline_metadata': IDLE_BASELINES, 'review_source': IDLE_REVIEW_SOURCE,
             'reviewed_identities': {key: value for key, value in REVIEWED_IDLE_CONTAINERS.items() if key in ALLOW_IDLE_IDS},
             'scope': 'No external container stop/remove; selected-card foreign NPU activity or unapproved selected-card-capable container blocks; other-card metadata is recorded.'})
        save(evidence / 'source_manifest.json', sources)
        for relative, expected_sha in sources.items():
            payload = (ROOT / relative).read_bytes()
            require(hashlib.sha256(payload).hexdigest() == expected_sha, 'Source changed while freezing: ' + relative)
            destination = evidence / 'source' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(payload)
            require(hashlib.sha256(destination.read_bytes()).hexdigest() == expected_sha,
                    'Frozen source copy differs: ' + relative)
        mounts = [('/usr/local/Ascend/driver', '/usr/local/Ascend/driver', True),
                  ('/usr/local/dcmi', '/usr/local/dcmi', True),
                  ('/usr/local/sbin/npu-smi', '/usr/local/sbin/npu-smi', True),
                  ('/home/tik-constfold-cann91-overnight-20261022', '/work', True),
                  ('/home/batch-scale-dynamic-msprobe-20261023/python_packages', '/msprobe_packages', True),
                  ('/dev/null', '/etc/ld.so.preload', True),
                  (str(ROOT), '/audit', True), (str(evidence), '/evidence', False)]
        command = ['docker', 'run', '-d', '--rm', '--name', name,
                   '--label', 'demoncat.task=' + TASK, '--label', 'demoncat.controller_token=' + token,
                   '--network', 'none', '--memory', '8g', '--cpus', '4', '--pids-limit', '512',
                   '--shm-size', '1g']
        for device in (f'/dev/davinci{args.card}', '/dev/davinci_manager', '/dev/devmm_svm', '/dev/hisi_hdc'):
            command += ['--device', device]
        for source, dest, readonly in mounts:
            command += ['--mount', f'type=bind,src={source},dst={dest}' + (',readonly' if readonly else '')]
        command += ['--entrypoint', '/bin/sleep', IMAGE, str(args.max_seconds)]
        save(evidence / 'launch_command.json', command)
        resources(args.card)
        create_attempted = True
        returned_id = run(command, 40).strip()
        require(re.fullmatch('[0-9a-f]{64}', returned_id), 'Invalid owned container id')
        owned = returned_id
        obj = json.loads(run(['docker', 'inspect', owned]))[0]
        require(obj['Image'] == IMAGE and obj['Config']['Labels']['demoncat.controller_token'] == token,
                'Owned container identity mismatch')
        require(not obj['HostConfig']['Privileged'] and obj['HostConfig']['NetworkMode'] == 'none',
                'Isolation mismatch')
        save(evidence / 'container_identity.json', {'id': owned, 'name': name, 'image': obj['Image'],
             'labels': obj['Config']['Labels'], 'devices': obj['HostConfig']['Devices']})
        for arm in ('B', 'F'):
            require(source_manifest() == sources, 'Source changed during experiment')
            require(not (ROOT / 'STOP').exists(), 'Explicit STOP marker')
            remaining = args.max_seconds - int(time.monotonic() - start) - 30
            require(remaining >= 60, 'Insufficient time for next arm')
            invocation = ['timeout', '--signal=TERM', '--kill-after=5s', str(min(remaining, 180)),
                          'python', '/audit/alias_probe.py', '--arm', arm, '--output', '/evidence/' + arm,
                          '--reference', '/audit/setup/logical_reference.json', '--capture']
            argv = ['docker', 'exec', owned, 'bash', '-c', 'source /audit/env.sh && exec ' + shlex.join(invocation)]
            pre_arm = resources(args.card, owned)
            record = {'arm': arm, 'start': time.time(), 'resources': [], 'pre_arm_resources': pre_arm}
            with (evidence / (arm + '.log')).open('xb') as log:
                child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
                try:
                    while child.poll() is None:
                        require(time.monotonic() - start < args.max_seconds - 10, 'Controller timeout')
                        require(not (ROOT / 'STOP').exists(), 'Explicit STOP marker')
                        record['resources'].append(resources(args.card, owned))
                        time.sleep(2)
                    record['rc'] = child.wait()
                finally:
                    record['end'] = time.time()
                    save(evidence / (arm + '.controller.json'), record)
            require(record['rc'] == 0, 'Probe failed: ' + arm)
            check = ['python3', str(ROOT / 'analyze_alias.py'), '--baseline', str(evidence / 'B'),
                     '--reference', str(ROOT / 'setup/logical_reference.json'),
                     '--output', str(evidence / ('baseline_gate.json' if arm == 'B' else 'acceptance.json'))]
            if arm == 'F':
                check += ['--fault', str(evidence / 'F')]
            save(evidence / (arm + '.analysis_stdout.json'), {'stdout': run(check, 90)})
            result['stages'].append({'arm': arm, 'rc': 0, 'independent_acceptance': True})
            require(source_manifest() == sources, 'Source changed during experiment')
        result['status'] = 'PASS'
    except BaseException:
        result['status'] = 'FAIL'
        result['traceback'] = traceback.format_exc()
    finally:
        try:
            if owned is None and create_attempted:
                owned = recover_owned_id(token, name)
                save(evidence / 'uncertain_create_recovery.json',
                     {'container_id': owned, 'controller_token': token,
                      'task': TASK, 'name': name, 'token_query_completed': True})
            if owned:
                ids = run(['docker', 'ps', '-a', '-q', '--no-trunc']).split()
                if owned in ids:
                    obj = json.loads(run(['docker', 'inspect', owned]))[0]
                    verify_owned(obj, token, name, owned)
                    if obj['State']['Running']:
                        run(['docker', 'stop', '--time', '5', owned], 20)
                    else:
                        run(['docker', 'rm', owned])
                require(owned not in run(['docker', 'ps', '-a', '-q', '--no-trunc']).split(), 'Owned container remains')
                result['owned_container_removed'] = True
        except BaseException:
            result['cleanup_errors'].append(traceback.format_exc())
        if child is not None and child.poll() is None:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.terminate()
        if lock is not None:
            lock.close()
        result['device_lock_acquired'] = lock_acquired
        result['device_lock_released'] = lock_acquired
        result['owned_container_id'] = owned
        result['end_epoch'] = time.time()
        if result['cleanup_errors']:
            result['status'] = 'FAIL_CLEANUP'
        save(evidence / 'completion.json', result)
    print(json.dumps(result))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
