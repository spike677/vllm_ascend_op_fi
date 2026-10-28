"""Bounded owned single-NPU B/F runner; optional exact authorized pair stops only."""
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

ROOT = Path('/home/fault27-alias-detach-20261028')
IMAGE = 'sha256:8988c3cd3a971a72ebb2e5baae24e0091cde55146f96d2b4c8229448210735b9'
TASK = 'fault27-alias-detach-20261028'
PROVEN_OWNED_PROCESSES = {}
OWNED_TEARDOWN_SINCE = {}
OWNED_TEARDOWN_GRACE_SECONDS = 60
STOP_AUTHORIZED_VLLM_PAIR = False
AUTHORIZED_STOP_COUNT = 0
AUTHORIZED_STOP_LIMIT = 8
CONTROLLER_DEADLINE = None
ATTEMPT_EVIDENCE = None


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
        'env.sh', 'setup/logical_reference.json', 'setup/acceptance_contract.json']
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


def matches_authorized_pair(obj):
    labels = obj.get('Config', {}).get('Labels') or {}
    mapped = {item.get('PathOnHost') for item in obj.get('HostConfig', {}).get('Devices') or []}
    return bool(re.fullmatch('[0-9a-f]{64}', obj.get('Id', ''))
                and re.fullmatch(r'vllm-pair-20261028-[ab]-candidate', obj.get('Name', '').lstrip('/'))
                and labels.get('experiment') == 'vllm-pair-20261028'
                and mapped.intersection({'/dev/davinci2', '/dev/davinci5'}))


def inspect_pair_id(container_id):
    require(re.fullmatch('[0-9a-f]{64}', container_id), 'Invalid authorized candidate ID')
    result = subprocess.run(['docker', 'inspect', container_id], capture_output=True, text=True, timeout=15)
    if result.returncode:
        require(result.returncode == 1 and result.stdout.strip() in ('', '[]')
                and result.stderr.strip() in ('Error: No such object: ' + container_id,
                    'Error: No such container: ' + container_id,
                    'Error response from daemon: No such container: ' + container_id),
                'Cannot establish authorized candidate state: ' + result.stderr[-2000:])
        return None
    objects = json.loads(result.stdout)
    require(len(objects) == 1 and objects[0]['Id'] == container_id, 'Candidate inspect identity differs')
    return objects[0]


def pair_metadata(obj):
    return {'container_id': obj['Id'], 'name': obj['Name'], 'image': obj['Image'],
            'experiment_label': (obj['Config'].get('Labels') or {}).get('experiment'),
            'command': obj['Config'].get('Cmd'), 'entrypoint': obj['Config'].get('Entrypoint'),
            'devices': obj['HostConfig'].get('Devices'), 'state': obj['State'],
            'auto_remove': obj['HostConfig'].get('AutoRemove')}


def append_authorized_stop(event):
    require(ATTEMPT_EVIDENCE is not None, 'No attempt evidence for authorized stop')
    with (ATTEMPT_EVIDENCE / 'authorized_stops.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def stop_authorized_pair_candidates():
    """Default off; only this named experiment, never docker rm or a broad stop."""
    global AUTHORIZED_STOP_COUNT
    if not STOP_AUTHORIZED_VLLM_PAIR:
        return
    ids = run(['docker', 'ps', '-q', '--no-trunc',
               '--filter', 'label=experiment=vllm-pair-20261028']).split()
    require(all(re.fullmatch('[0-9a-f]{64}', value) for value in ids), 'Invalid candidate enumeration')
    for container_id in ids:
        candidate = inspect_pair_id(container_id)
        if candidate is None or not candidate['State']['Running'] or not matches_authorized_pair(candidate):
            continue
        require(not (ROOT / 'STOP').exists(), 'STOP requested before authorized stop')
        require(CONTROLLER_DEADLINE is not None and time.monotonic() + 35 < CONTROLLER_DEADLINE,
                'Insufficient controller lifetime for authorized stop and state verification')
        require(AUTHORIZED_STOP_COUNT < AUTHORIZED_STOP_LIMIT, 'Authorized stop limit reached')
        npu_before = run(['npu-smi', 'info'], 15)
        current = inspect_pair_id(container_id)
        if current is None or not current['State']['Running']:
            continue
        require(current['Id'] == candidate['Id'] and current['Name'] == candidate['Name']
                and matches_authorized_pair(current), 'Authorized stop target changed on reinspection')
        require(not (ROOT / 'STOP').exists() and time.monotonic() + 35 < CONTROLLER_DEADLINE,
                'Stop requested or deadline too close before authorized action')
        AUTHORIZED_STOP_COUNT += 1
        identity = {'sequence': AUTHORIZED_STOP_COUNT, 'container_id': container_id,
                    'name': current['Name'], 'metadata': pair_metadata(current),
                    'scope': 'explicit vllm-pair-20261028 A/B candidate stop; never remove'}
        append_authorized_stop({**identity, 'event': 'stop_started', 'epoch': time.time(), 'npu_before': npu_before})
        argv = ['docker', 'stop', '--time', '5', container_id]
        command = {'argv': argv}
        try:
            stopped = subprocess.run(argv, capture_output=True, text=True, timeout=20)
            command.update(rc=stopped.returncode, stdout=stopped.stdout, stderr=stopped.stderr)
        except Exception as error:
            command.update(rc=None, error=repr(error))
        after = None
        state_error = None
        try:
            after = inspect_pair_id(container_id)
        except Exception as error:
            state_error = repr(error)
        confirmed_stopped = state_error is None and (after is None or not after['State']['Running'])
        append_authorized_stop({**identity, 'event': 'stop_finished', 'epoch': time.time(),
                                'command': command, 'post_state': pair_metadata(after) if after else None,
                                'post_inspect_error': state_error, 'stopped_or_absent': confirmed_stopped})
        require(confirmed_stopped, 'Authorized candidate did not stop or its state is unknown')


def resources(card, owned=None):
    stop_authorized_pair_candidates()
    ids = run(['docker', 'ps', '-q', '--no-trunc']).split()
    objects = json.loads(run(['docker', 'inspect', *ids])) if ids else []
    reservations = []
    for obj in objects:
        if obj['Id'] == owned:
            continue
        mapped = any(v.get('PathOnHost') == f'/dev/davinci{card}'
                     for v in obj['HostConfig'].get('Devices', []))
        mapped = mapped or any(v.get('Source') in ('/dev', f'/dev/davinci{card}')
                               for v in obj.get('Mounts', []))
        # This longstanding general container is allowed only with no NPU process.
        if mapped and obj['Name'].lstrip('/') != 'gyl_test':
            reservations.append({'id': obj['Id'], 'name': obj['Name']})
    require(not reservations, 'Device reserved: ' + repr(reservations))
    smi = run(['npu-smi', 'info'])
    require('Process id' in smi, 'Unrecognized npu-smi output')
    require(re.search(rf'^\|\s*{card}\s+910B4', smi, re.M), 'Requested 910B4 not visible')
    pids, processes = [], []
    for line in smi.split('Process id', 1)[1].splitlines():
        match = re.match(r'^\|\s*(\d+)\s+\d+\s*\|\s*(\d+)\s*\|', line)
        if match and int(match[1]) == card:
            pid = int(match[2])
            try:
                identity = read_process_identity(pid)
            except FileNotFoundError:
                require(not Path(f'/proc/{pid}').exists(), f'Incomplete identity while NPU PID still exists: {pid}')
                processes.append({'host_pid': pid, 'status': 'exited_before_identity_read'})
                continue
            identity['owned'], identity['ownership_evidence'], previous = classify_process_identity(identity, owned)
            if previous is not None:
                identity['prior_owned_identity'] = previous
            processes.append(identity)
            require(identity['owned'], 'Foreign/unproven NPU process: ' + repr(identity))
            pids.append(pid)
    if owned is None:
        require(f'No running processes found in NPU {card}' in smi, 'Card is not confirmed idle')
    return {'epoch': time.time(), 'npu_smi': smi, 'owned_pids': pids,
            'processes': processes, 'foreign_reservations': reservations,
            'authorized_stop_count': AUTHORIZED_STOP_COUNT}


def main():
    global STOP_AUTHORIZED_VLLM_PAIR, AUTHORIZED_STOP_COUNT, CONTROLLER_DEADLINE, ATTEMPT_EVIDENCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--card', type=int, choices=(2, 5), default=5)
    parser.add_argument('--attempt')
    parser.add_argument('--max-seconds', type=int, default=600)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--stop-authorized-vllm-pair', action='store_true',
                        help='Stop only experiment=vllm-pair-20261028 exact A/B candidate names mapped to cards2/5; never remove')
    args = parser.parse_args()
    PROVEN_OWNED_PROCESSES.clear()
    OWNED_TEARDOWN_SINCE.clear()
    STOP_AUTHORIZED_VLLM_PAIR = args.stop_authorized_vllm_pair
    AUTHORIZED_STOP_COUNT = 0
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
    name = 'dcat-f27-alias-' + args.attempt
    token = uuid.uuid4().hex
    lock = None
    lock_acquired = False
    create_attempted = False
    owned = None
    child = None
    start = time.monotonic()
    CONTROLLER_DEADLINE = start + args.max_seconds
    result = {'status': 'RUNNING', 'card': args.card, 'stages': [], 'cleanup_errors': [],
              'start_epoch': time.time(), 'scope': 'mechanism microbenchmark, not model tokens',
              'stop_authorized_vllm_pair': STOP_AUTHORIZED_VLLM_PAIR,
              'authorized_stop_limit': AUTHORIZED_STOP_LIMIT}

    def interrupted(signum, frame):
        raise RuntimeError('Interrupted: ' + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        lock = open('/var/tmp/fault22_npu_2_5.lock', 'a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_acquired = True
        save(evidence / 'before.json', resources(args.card))
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
            record = {'arm': arm, 'start': time.time(), 'resources': [], 'pre_arm_resources': pre_arm,
                      'authorized_stops_before': AUTHORIZED_STOP_COUNT}
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
                    record['authorized_stops_after'] = AUTHORIZED_STOP_COUNT
                    save(evidence / (arm + '.controller.json'), record)
            require(record['authorized_stops_after'] == record['authorized_stops_before'],
                    'INVALIDATED_EXTERNAL_INTERFERENCE: an authorized external stop occurred during this arm')
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
        result['authorized_stop_count'] = AUTHORIZED_STOP_COUNT
        result['end_epoch'] = time.time()
        if result['cleanup_errors']:
            result['status'] = 'FAIL_CLEANUP'
        save(evidence / 'completion.json', result)
    print(json.dumps(result))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
