"""Bounded full-model capture runner; never stop or remove external containers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import time
import traceback
import uuid

ROOT = Path('/home/fault28-model-coverage-20261030')
IMAGE = 'sha256:8988c3cd3a971a72ebb2e5baae24e0091cde55146f96d2b4c8229448210735b9'
TASK = 'fault28-model-coverage-20261030'
EXPECTED_RUNTIME_MANIFEST_SHA256 = 'ca0bf188bdc031ea3e4249c03ac13a9a8f751cd5ae8485655d9048c9f6cdba3b'
MAX_CASE_SECONDS = 3600
MAX_CONTROLLER_SECONDS = 7200
MAX_LOG_MIB = 16384
SIDECAR_RESERVE_BYTES_PER_CASE = 2 * 1024 ** 3
DISK_SAFETY_RESERVE_BYTES = 8 * 1024 ** 3
PROVEN_OWNED_PROCESSES = {}
OWNED_TEARDOWN_SINCE = {}
OWNED_TEARDOWN_GRACE_SECONDS = 60
OWNED_MISSING_PID_SINCE = {}
OWNED_MISSING_PID_GRACE_SECONDS = 10
OWNED_MISSING_PID_RETRY_SECONDS = 0.5
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
IDLE_REVIEW_SOURCE = {'utc': '2026-10-30T03:37:50.349003+00:00',
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


def execution_budget(case_seconds, max_seconds, max_log_mib, case_count):
    require(type(case_seconds) is int and 60 <= case_seconds <= MAX_CASE_SECONDS
            and type(max_seconds) is int and 180 <= max_seconds <= MAX_CONTROLLER_SECONDS,
            'Unsupported bounded duration')
    require(type(max_log_mib) is int and 16 <= max_log_mib <= MAX_LOG_MIB,
            'Unsupported bounded log size')
    require(type(case_count) is int and 1 <= case_count <= 8, 'Invalid finite case count')
    return {'case_seconds': case_seconds, 'controller_seconds': max_seconds,
            'max_log_mib_per_case': max_log_mib, 'max_log_bytes_per_case': max_log_mib * 1024 ** 2,
            'case_count': case_count, 'sidecar_reserve_bytes_per_case': SIDECAR_RESERVE_BYTES_PER_CASE,
            'disk_safety_reserve_bytes': DISK_SAFETY_RESERVE_BYTES,
            'maximum_supported_case_seconds': MAX_CASE_SECONDS,
            'maximum_supported_controller_seconds': MAX_CONTROLLER_SECONDS,
            'maximum_supported_log_mib_per_case': MAX_LOG_MIB,
            'pre_case_disk_formula': 'remaining_cases * (log_cap_bytes + sidecar_reserve_bytes) + safety_reserve_bytes',
            'running_disk_formula': 'free_bytes >= safety_reserve_bytes',
            'disk_budget_is_reservation_estimate_not_filesystem_quota': True}


def disk_snapshot(budget, remaining_cases, phase):
    require(type(remaining_cases) is int and 1 <= remaining_cases <= budget['case_count'],
            'Invalid remaining case count for disk guard')
    require(phase in ('before_container', 'before_case', 'running'), 'Invalid disk guard phase')
    required = budget['disk_safety_reserve_bytes']
    if phase != 'running':
        required += remaining_cases * (budget['max_log_bytes_per_case'] + budget['sidecar_reserve_bytes_per_case'])
    usage = shutil.disk_usage(ROOT)
    record = {'epoch': time.time(), 'filesystem_path': str(ROOT), 'phase': phase,
              'remaining_cases': remaining_cases, 'total_bytes': usage.total, 'used_bytes': usage.used,
              'free_bytes': usage.free, 'required_free_bytes': required,
              'eligible': usage.free >= required}
    if ATTEMPT_EVIDENCE is not None:
        with (ATTEMPT_EVIDENCE / 'disk_checks.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
            stream.flush()
    return record


def source_manifest():
    names = sorted(p.name for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py', '.sh'))
    names += sorted(str(p.relative_to(ROOT)) for p in (ROOT / 'setup').rglob('*.json'))
    require('controller.py' in names and 'env.sh' in names, 'Missing controller/environment source')
    required_metadata = ('memory_api_source_audit.json', 'inputs/manifest.json',
                         'model_manifest.json', 'tokenizer_manifest.json', 'source_fingerprints.json')
    for relative in required_metadata:
        require((ROOT / relative).is_file() and not (ROOT / relative).is_symlink(),
                'Missing or symlinked model/input/source fingerprint: ' + relative)
        names.append(relative)
    runtime_root = ROOT / 'runtime'
    manifest_path = ROOT / 'runtime_manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    require(hashlib.sha256(manifest_path.read_bytes()).hexdigest() == EXPECTED_RUNTIME_MANIFEST_SHA256,
            'Validated runtime manifest changed; do not edit core or certificate in this experiment')
    require(manifest.get('schema') == 'nes.runtime_snapshot.v1', 'Unexpected runtime snapshot schema')
    entries = manifest.get('files')
    require(isinstance(entries, dict) and len(entries) == 23,
            'Expected exact validated23-file runtime inventory')
    actual = {p.relative_to(runtime_root).as_posix() for p in runtime_root.rglob('*') if p.is_file()}
    require(actual == set(entries) and not any(p.is_symlink() for p in runtime_root.rglob('*')),
            'Runtime files differ from frozen inventory or contain symlinks')
    for relative, row in entries.items():
        path = runtime_root / relative
        require(isinstance(relative, str) and not Path(relative).is_absolute()
                and '..' not in Path(relative).parts and '\\' not in relative
                and path.resolve(strict=True).is_relative_to(runtime_root.resolve(strict=True))
                and path.suffix in ('.py', '.json'), 'Unsafe runtime artifact: ' + str(relative))
        require(isinstance(row, dict) and type(row.get('bytes')) is int
                and path.stat().st_size == row['bytes']
                and hashlib.sha256(path.read_bytes()).hexdigest() == row.get('sha256'),
                'Frozen runtime artifact changed: ' + relative)
        names.append('runtime/' + relative)
    components = manifest.get('native_components_sha256')
    require(isinstance(components, dict) and components
            and all(isinstance(key, str) and isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value)
                    for key, value in components.items()), 'Native component source fingerprints required')
    names.append('runtime_manifest.json')
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


def process_directory_absent(pid):
    """A missing stat/ns child is insufficient: confirm /proc/PID itself is gone."""
    try:
        (Path('/proc') / str(pid)).stat()
    except FileNotFoundError:
        return True
    return False


def missing_process_identity(pid, owned, absent, proven=None, missing_since=None, now=None):
    """Never approve a missing identity; only identify a bounded retry candidate."""
    proven = PROVEN_OWNED_PROCESSES if proven is None else proven
    missing_since = OWNED_MISSING_PID_SINCE if missing_since is None else missing_since
    now = time.monotonic() if now is None else now
    identity = {'host_pid': pid, 'owned': False,
                'status': 'identity_unavailable_for_recorded_NPU_PID',
                'proc_directory_absent': absent, 'retry_pending': False}
    previous = proven.get((owned, pid)) if owned is not None else None
    if not absent or previous is None:
        identity['ownership_evidence'] = 'MISSING_IDENTITY_NOT_PROVEN_OWNED_OR_PROC_STILL_PRESENT'
        return identity
    key = (owned, pid, previous['starttime_ticks'])
    since = missing_since.setdefault(key, now)
    age = now - since
    pending = 0 <= age < OWNED_MISSING_PID_GRACE_SECONDS
    identity.update(prior_owned_identity=previous.copy(), retry_pending=pending,
                    missing_pid_since_monotonic=since, missing_pid_age_seconds=age,
                    missing_pid_grace_seconds=OWNED_MISSING_PID_GRACE_SECONDS,
                    ownership_evidence=('PROVEN_OWNED_PROC_GONE_DRIVER_ROW_RETRY' if pending
                                        else 'PROVEN_OWNED_PROC_GONE_GRACE_EXCEEDED'))
    return identity


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


def resource_snapshot(card, owned=None, require_idle=False, query_deadline=None):
    def query(argv):
        timeout = 25
        if query_deadline is not None:
            remaining = query_deadline - time.monotonic()
            require(remaining > 0, 'Owned missing-PID retry deadline reached')
            timeout = min(timeout, remaining)
        value = run(argv, timeout)
        if query_deadline is not None:
            require(time.monotonic() < query_deadline, 'Owned missing-PID retry deadline reached')
        return value

    ids = query(['docker', 'ps', '-q', '--no-trunc']).split()
    require(all(re.fullmatch('[0-9a-f]{64}', value) for value in ids), 'Invalid running container identity')
    objects = json.loads(query(['docker', 'inspect', *ids])) if ids else []
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
    smi = query(['npu-smi', 'info'])
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
            identity = missing_process_identity(pid, owned, process_directory_absent(pid))
        identity['physical_device'] = physical
        processes.append(identity)
        if not identity['owned'] or physical != card:
            foreign.append(identity)
    for device in (2, 5):
        require(device in accounted or f'No running processes found in NPU {device}' in smi,
                'NPU process table does not account for physical card ' + str(device))
    selected_foreign = [item for item in foreign if item['physical_device'] == card]
    selected_processes = [item for item in processes if item['physical_device'] == card]
    retry_pending = [item for item in selected_foreign if item.get('retry_pending')]
    blocking_foreign = [item for item in selected_foreign if not item.get('retry_pending')]
    table_idle = not selected_processes and f'No running processes found in NPU {card}' in smi
    idle = table_idle if require_idle else (owned is not None or table_idle)
    # Retire proof once a previously missing process also disappears from the driver.
    # A later PID reuse must establish a fresh live identity, not inherit old proof.
    present_pids = {item['host_pid'] for item in processes}
    retired = []
    for key in list(OWNED_MISSING_PID_SINCE):
        if key[0] == owned and key[1] not in present_pids:
            previous = PROVEN_OWNED_PROCESSES.get((owned, key[1]))
            if previous is not None and previous['starttime_ticks'] == key[2]:
                PROVEN_OWNED_PROCESSES.pop((owned, key[1]))
            OWNED_MISSING_PID_SINCE.pop(key)
            retired.append({'host_pid': key[1], 'starttime_ticks': key[2]})
    report = {'epoch': time.time(), 'selected_card': card, 'monitored_physical_cards': [2, 5],
              'npu_smi': smi, 'processes': processes, 'foreign_npu_processes': foreign,
              'selected_card_foreign_npu_processes': selected_foreign,
              'other_card_foreign_npu_processes': [item for item in foreign if item['physical_device'] != card],
              'device_capable_containers': capable, 'unapproved_device_capable_containers': unapproved,
              'idle_allowlist_ids': sorted(ALLOW_IDLE_IDS),
              'allowlisted_ids_not_running': sorted(ALLOW_IDLE_IDS - set(ids)),
              'require_selected_card_idle': require_idle, 'selected_card_table_idle': table_idle,
              'retry_pending_owned_missing_processes': retry_pending,
              'blocking_selected_card_foreign_npu_processes': blocking_foreign,
              'retired_missing_process_identities': retired,
              'eligible': not unapproved and not selected_foreign and idle,
              'scope': 'Selected-card resource gate; both cards recorded. Explicit idle coexistence is not device exclusivity.'}
    append_resource_check(report)
    return report


def resources(card, owned=None, strict=True, require_idle=False):
    deadline = None
    retry_count = 0
    while True:
        try:
            report = resource_snapshot(card, owned, require_idle, deadline)
        except BaseException:
            if deadline is not None:
                append_resource_check({'event': 'owned_missing_pid_retry_failed', 'epoch': time.time(),
                                       'selected_card': card, 'owned_container_id': owned,
                                       'retry_count': retry_count, 'deadline_monotonic': deadline,
                                       'traceback': traceback.format_exc()})
            raise
        if not strict or report['eligible']:
            return report
        pending = report['retry_pending_owned_missing_processes']
        blockers = report['blocking_selected_card_foreign_npu_processes']
        # Missing rows are never approved: retry only when they are the sole blocker.
        retry_only = (pending and not blockers and not report['unapproved_device_capable_containers']
                      and (not require_idle or
                           len(pending) == len([p for p in report['processes'] if p['physical_device'] == card])))
        if not retry_only:
            raise RuntimeError('Resource gate rejected: ' + json.dumps({
                'unapproved_container_ids': [item['id'] for item in report['unapproved_device_capable_containers']],
                'selected_card': card,
                'foreign_npu_pids': [item['host_pid'] for item in report['selected_card_foreign_npu_processes']],
                'require_selected_card_idle': require_idle,
                'selected_card_table_idle': report['selected_card_table_idle']}))
        pending_deadline = min(item['missing_pid_since_monotonic'] + OWNED_MISSING_PID_GRACE_SECONDS
                               for item in pending)
        deadline = pending_deadline if deadline is None else min(deadline, pending_deadline)
        remaining = deadline - time.monotonic()
        require(remaining > 0, 'Owned missing-PID driver row did not clear within 10 seconds')
        retry_count += 1
        append_resource_check({'event': 'owned_missing_pid_retry_wait', 'epoch': time.time(),
                               'selected_card': card, 'owned_container_id': owned,
                               'host_pids': [item['host_pid'] for item in pending],
                               'retry_count': retry_count, 'remaining_seconds': remaining,
                               'deadline_monotonic': deadline,
                               'next_delay_seconds': min(OWNED_MISSING_PID_RETRY_SECONDS, remaining)})
        time.sleep(min(OWNED_MISSING_PID_RETRY_SECONDS, remaining))


def load_cases(relative_paths):
    cases = []
    ids = set()
    for relative in relative_paths:
        path = (ROOT / relative).resolve()
        require((ROOT / 'setup').resolve() in path.parents and path.suffix == '.json',
                'Case must be a JSON file under this task setup directory')
        case = json.loads(path.read_text())
        require(case.get('schema') == 'nes.model_coverage.case.v1'
                and case.get('phase') == 'model_coverage'
                and case.get('operator_variant') == 'private_vendor',
                'Only full-model coverage cases using the immutable private vendor are accepted')
        case_id = case.get('case_id')
        require(isinstance(case_id, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,60}', case_id)
                and case_id not in ids, 'Case IDs must be unique and filesystem safe')
        ids.add(case_id)
        cases.append({'case_id': case_id, 'file': str(path.relative_to(ROOT.resolve())), 'case': case})
    require(1 <= len(cases) <= 8, 'Select between one and eight finite cases')
    return cases


def build_probe_command(probe, case, capture):
    variant = case['case'].get('operator_variant', 'stock')
    require(variant in ('stock', 'private_vendor'), 'Unknown operator variant')
    selected_entrypoint = probe
    if variant == 'private_vendor':
        require(probe == 'model_probe.py', 'Private vendor launcher targets the reviewed model probe only')
        selected_entrypoint = 'vendor_exec.py'
    args = ['python', '/audit/' + selected_entrypoint, '--case-file', '/audit/' + case['file'],
            '--output', '/evidence/' + case['case_id']]
    if capture == 'msprof':
        return ['/usr/local/Ascend/cann-9.1.0/tools/profiler/bin/msprof',
                '--output=/evidence/' + case['case_id'] + '_native_profile', '--task-time=on',
                '--application=' + shlex.join(args)]
    return args


def main():
    global ALLOW_IDLE_IDS, ATTEMPT_EVIDENCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--card', type=int, choices=(5,), default=5)
    parser.add_argument('--attempt')
    parser.add_argument('--max-seconds', type=int, default=1800)
    parser.add_argument('--case-seconds', type=int, default=900)
    parser.add_argument('--probe', choices=('model_probe.py',), default='model_probe.py')
    parser.add_argument('--case-file', action='append', required=True)
    parser.add_argument('--capture', choices=('native-logs', 'msprof', 'none'), default='native-logs')
    parser.add_argument('--max-log-mib', type=int, default=2048)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--allow-idle-container-ID', action='append', default=[], dest='allow_idle_ids',
                        help='Explicit full ID of an idle device-capable background container')
    args = parser.parse_args()
    PROVEN_OWNED_PROCESSES.clear()
    OWNED_TEARDOWN_SINCE.clear()
    OWNED_MISSING_PID_SINCE.clear()
    ALLOW_IDLE_IDS = set(args.allow_idle_ids)
    require(len(ALLOW_IDLE_IDS) == len(args.allow_idle_ids) and all(re.fullmatch('[0-9a-f]{64}', value) for value in ALLOW_IDLE_IDS),
            'Idle allowlist requires unique exact 64-character IDs')
    IDLE_BASELINES.clear()
    cases = load_cases(args.case_file)
    budget = execution_budget(args.case_seconds, args.max_seconds, args.max_log_mib, len(cases))
    sources = source_manifest()
    require(args.probe in sources, 'Selected probe source is missing')
    if any(case['case'].get('operator_variant', 'stock') == 'private_vendor' for case in cases):
        require(args.probe == 'model_probe.py' and 'vendor_exec.py' in sources,
                'Private model cases require frozen probe and vendor launcher sources')
    if args.prepare_only:
        print(json.dumps({'status': 'PREPARED_NO_NPU', 'sources': sources, 'cases': cases, 'execution_budget': budget,
                          'commands': [build_probe_command(args.probe, case, args.capture) for case in cases]}))
        return 0
    require(args.attempt is not None and re.fullmatch(r'[A-Za-z0-9_-]{1,40}', args.attempt),
            'Fresh explicit attempt required')
    import fcntl
    evidence = ROOT / 'outputs' / args.attempt
    evidence.mkdir(parents=True, exist_ok=False)
    ATTEMPT_EVIDENCE = evidence
    name = 'dcat-model-coverage-' + args.attempt
    token = uuid.uuid4().hex
    lock = None
    lock_acquired = False
    create_attempted = False
    owned = None
    child = None
    start = time.monotonic()
    result = {'status': 'RUNNING', 'card': args.card, 'cases': [], 'cleanup_errors': [],
              'start_epoch': time.time(), 'probe': args.probe, 'capture': args.capture,
              'scope': 'Full-model coverage capture only; per-layer native eligibility, strategy effects and model effects require independent acceptance',
              'mechanism_accepted': False, 'idle_allowlist_ids': sorted(ALLOW_IDLE_IDS),
              'external_container_mutations': False, 'execution_budget': budget}

    def interrupted(signum, frame):
        raise RuntimeError('Interrupted: ' + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        lock = open('/var/tmp/fault22_npu_2_5.lock', 'a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_acquired = True
        save(evidence / 'before.json', resources(args.card))
        mem = Path('/proc/meminfo').read_text()
        available = re.search(r'^MemAvailable:\s+(\d+) kB$', mem, re.M)
        require(available and int(available.group(1)) >= 24 * 1024 * 1024,
                'At least 24 GiB host memory required for 16 GiB container plus headroom')
        before_disk = disk_snapshot(budget, len(cases), 'before_container')
        save(evidence / 'before_disk.json', before_disk)
        save(evidence / 'resource_policy.json', {'idle_allowlist_ids': sorted(ALLOW_IDLE_IDS),
             'baseline_metadata': IDLE_BASELINES, 'review_source': IDLE_REVIEW_SOURCE,
             'host_mem_available_kib': int(available.group(1)), 'container_memory_gib': 16,
             'execution_budget': budget, 'before_disk': before_disk,
             'proven_owned_missing_pid_retry_seconds': OWNED_MISSING_PID_GRACE_SECONDS,
             'proven_owned_missing_pid_requires_proc_directory_absent': True,
             'pre_post_case_requires_selected_card_table_idle': True,
             'reviewed_identities': {key: value for key, value in REVIEWED_IDLE_CONTAINERS.items() if key in ALLOW_IDLE_IDS},
             'scope': 'Selected-card gate; other-card activity recorded. No external stop/remove.'})
        require(before_disk['eligible'], 'Insufficient disk space for remaining cases and safety reserve')
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
        save(evidence / 'selected_cases.json', cases)
        mounts = [('/usr/local/Ascend/driver', '/usr/local/Ascend/driver', True),
                  ('/usr/local/dcmi', '/usr/local/dcmi', True),
                  ('/usr/local/sbin/npu-smi', '/usr/local/sbin/npu-smi', True),
                  ('/home/tik-constfold-cann91-overnight-20261022', '/work', True),
                  ('/home/gyl/models/Qwen3-0.6B', '/models/Qwen3-0.6B', True),
                  ('/home/batch-scale-dynamic-msprobe-20261023/python_packages', '/msprobe_packages', True),
                  ('/home/fault28-native-tiling-20261029/collector_high_a1', '/audit/collector_high_a1', True),
                  ('/dev/null', '/etc/ld.so.preload', True),
                  (str(ROOT), '/audit', True), (str(evidence), '/evidence', False)]
        command = ['docker', 'run', '-d', '--rm', '--name', name,
                   '--label', 'demoncat.task=' + TASK, '--label', 'demoncat.controller_token=' + token,
                   '--network', 'none', '--memory', '16g', '--cpus', '8', '--pids-limit', '1024',
                   '--shm-size', '2g']
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
        verify_owned(obj, token, name, owned)
        require(not obj['HostConfig']['Privileged'] and obj['HostConfig']['NetworkMode'] == 'none',
                'Isolation mismatch')
        save(evidence / 'container_identity.json', {'id': owned, 'name': name, 'image': obj['Image'],
             'labels': obj['Config']['Labels'], 'devices': obj['HostConfig']['Devices']})
        if args.capture == 'msprof':
            help_command = ['docker', 'exec', owned, 'bash', '-c',
                            'source /audit/env.sh && /usr/local/Ascend/cann-9.1.0/tools/profiler/bin/msprof --help']
            save(evidence / 'msprof_help.json', {'stdout': run(help_command, 20)})
        for case_index, case in enumerate(cases):
            require(source_manifest() == sources, 'Source changed during capture')
            require(not (ROOT / 'STOP').exists(), 'Explicit STOP marker')
            remaining = args.max_seconds - int(time.monotonic() - start) - 30
            require(remaining >= 60, 'Insufficient time for next case')
            invocation = ['timeout', '--signal=TERM', '--kill-after=5s', str(min(remaining, args.case_seconds))]
            if args.capture != 'none':
                invocation += ['env', 'ASCEND_GLOBAL_LOG_LEVEL=0', 'ASCEND_SLOG_PRINT_TO_STDOUT=1']
            invocation += build_probe_command(args.probe, case, args.capture)
            argv = ['docker', 'exec', owned, 'bash', '-c', 'source /audit/env.sh && exec ' + shlex.join(invocation)]
            pre_case = resources(args.card, owned, require_idle=True)
            pre_case_disk = disk_snapshot(budget, len(cases) - case_index, 'before_case')
            save(evidence / (case['case_id'] + '.before_disk.json'), pre_case_disk)
            require(pre_case_disk['eligible'], 'Insufficient disk space before case: ' + case['case_id'])
            record = {'case_id': case['case_id'], 'start': time.time(), 'resources': [],
                      'pre_case_resources': pre_case, 'pre_case_disk': pre_case_disk,
                      'disk_checks': [], 'command': argv}
            result['cases'].append(record)
            log_path = evidence / (case['case_id'] + '.log')
            with log_path.open('xb') as log:
                child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
                try:
                    while child.poll() is None:
                        require(time.monotonic() - start < args.max_seconds - 10, 'Controller timeout')
                        require(not (ROOT / 'STOP').exists(), 'Explicit STOP marker')
                        require(log_path.stat().st_size < args.max_log_mib * 1024 * 1024, 'Case log size limit reached')
                        running_disk = disk_snapshot(budget, len(cases) - case_index, 'running')
                        record['disk_checks'].append(running_disk)
                        require(running_disk['eligible'], 'Disk safety reserve below8GiB during case')
                        record['resources'].append(resources(args.card, owned))
                        time.sleep(2)
                    record['rc'] = child.wait()
                    record['final_log_bytes'] = log_path.stat().st_size
                    record['post_case_disk'] = disk_snapshot(budget, len(cases) - case_index, 'running')
                    require(record['final_log_bytes'] < budget['max_log_bytes_per_case'], 'Final case log size limit reached')
                    require(record['post_case_disk']['eligible'], 'Disk safety reserve below8GiB at case completion')
                finally:
                    record['end'] = time.time()
                    save(evidence / (case['case_id'] + '.controller.json'), record)
            require(record['rc'] == 0, 'Probe process failed: ' + case['case_id'])
            case_result_path = evidence / case['case_id'] / 'result.json'
            require(case_result_path.is_file(), 'Probe result.json missing: ' + case['case_id'])
            case_result = json.loads(case_result_path.read_text())
            require(case_result.get('case_id') == case['case_id'] and case_result.get('status') in ('completed', 'CAPTURE_COMPLETED'),
                    'Probe did not report completed capture: ' + case['case_id'])
            record['result_sha256'] = hashlib.sha256(case_result_path.read_bytes()).hexdigest()
            record['status'] = 'CAPTURE_COMPLETED_NOT_MECHANISM_ACCEPTANCE'
            require(source_manifest() == sources, 'Source changed during capture')
            record['post_case_resources'] = resources(args.card, owned, require_idle=True)
        result['status'] = 'CAPTURE_COMPLETED'
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
    print(json.dumps({
        'status': result['status'], 'attempt': args.attempt, 'card': args.card,
        'probe': args.probe, 'capture': args.capture, 'mechanism_accepted': False,
        'owned_container_id': result.get('owned_container_id'),
        'owned_container_removed': result.get('owned_container_removed', False),
        'cleanup_error_count': len(result['cleanup_errors']),
        'cases': [{key: case[key] for key in ('case_id', 'rc', 'status') if key in case}
                  for case in result['cases']],
        'completion_file': str(evidence / 'completion.json'),
        'error_tail': result.get('traceback', '')[-1200:],
    }))
    return 0 if result['status'] == 'CAPTURE_COMPLETED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
