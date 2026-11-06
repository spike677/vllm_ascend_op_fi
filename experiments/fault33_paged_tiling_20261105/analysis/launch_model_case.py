"""Run one sealed model case; optionally stop one explicitly identified card-5 ATK.

The user authorized clearing conflicting ATK work. This helper acts only before
the controller starts. It does not change queue files, device nodes or services.
The controller retains its own resource lock and repeated fail-closed gates.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path('/home/fault33-paged-tiling-20261105')
CASES = ('model_baseline', 'model_fault_r0', 'model_fault_r1', 'model_restore')


def run(args, timeout=30):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def validate(obj, identity):
    labels = obj['Config'].get('Labels') or {}
    host = obj['HostConfig']
    devices = {v['PathOnHost'] for v in host.get('Devices') or []}
    allowed = {'/dev/davinci5', '/dev/davinci_manager', '/dev/devmm_svm', '/dev/hisi_hdc'}
    if not (obj['Id'] == identity and obj['Name'].lstrip('/').startswith('atk_q_')
            and labels.get('atk.queue.job_id') and devices <= allowed and '/dev/davinci5' in devices
            and not host.get('Privileged') and not host.get('DeviceRequests') and not host.get('DeviceCgroupRules')):
        raise ValueError('Refuse cleanup outside one explicitly identified card-5-only ATK container')
    for mount in obj.get('Mounts') or []:
        source = mount.get('Source', '').rstrip('/') or '/'
        if source in ('/', '/dev') or (source.startswith('/dev/') and source not in allowed):
            raise ValueError('Unrestricted/other device bind; refuse cleanup')
    return {'id': identity, 'name': obj['Name'], 'image': obj['Image'],
            'queue_job_id': labels['atk.queue.job_id'], 'devices': sorted(devices),
            'initial_state': obj['State']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True, choices=CASES)
    p.add_argument('--attempt', required=True)
    p.add_argument('--stop-atk-id', help='Exact full ID observed at a failed card-5 gate')
    p.add_argument('--clear-card5-atk', action='store_true',
                   help='Discover at most one running card-5-only ATK, verify it, then stop before launch')
    args = p.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]{1,40}', args.attempt):
        raise ValueError('Unsafe attempt')
    if args.stop_atk_id and not re.fullmatch('[0-9a-f]{64}', args.stop_atk_id):
        raise ValueError('Full exact container ID required')
    if args.clear_card5_atk and args.stop_atk_id:
        raise ValueError('Choose explicit ID or bounded discovery')
    if args.clear_card5_atk:
        listing = run(['docker', 'ps', '-q', '--no-trunc'])
        if listing.returncode:
            raise RuntimeError('Cannot discover running containers')
        candidates = []
        for identity in listing.stdout.split():
            if not re.fullmatch('[0-9a-f]{64}', identity):
                raise ValueError('Invalid discovery ID')
            inspected = run(['docker', 'inspect', identity])
            if inspected.returncode:
                if 'No such' in inspected.stderr:
                    continue
                raise RuntimeError('Container discovery failed')
            objects = json.loads(inspected.stdout)
            if len(objects) != 1:
                raise ValueError('Ambiguous discovered container')
            obj = objects[0]
            labels = obj['Config'].get('Labels') or {}
            paths = {v['PathOnHost'] for v in obj['HostConfig'].get('Devices') or []}
            if obj['Name'].lstrip('/').startswith('atk_q_') and labels.get('atk.queue.job_id') and '/dev/davinci5' in paths:
                validate(obj, identity)
                candidates.append(identity)
        if len(candidates) > 1:
            raise ValueError('More than one card-5 ATK found; do not mutate an ambiguous resource state')
        args.stop_atk_id = candidates[0] if candidates else None
    output = ROOT / 'outputs/manual_launch' / (args.attempt + '.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or (ROOT / 'outputs' / args.attempt).exists():
        raise ValueError('Fresh attempt and receipt required')
    receipt = {'schema': 'fault33.single_case.manual_launch.v1',
               'recorded_utc': datetime.now(timezone.utc).isoformat(),
               'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'case': args.case, 'attempt': args.attempt, 'cleanup': None,
               'scope': 'At most one exact ATK stop before controller; no queue/result file edits.'}
    if args.stop_atk_id:
        found = run(['docker', 'inspect', args.stop_atk_id])
        if found.returncode:
            if 'No such' not in found.stderr:
                raise RuntimeError(found.stderr[-2000:])
            receipt['cleanup'] = {'id': args.stop_atk_id, 'status': 'ALREADY_GONE_NO_STOP'}
        else:
            objects = json.loads(found.stdout)
            if len(objects) != 1:
                raise ValueError('Ambiguous exact container inspect')
            row = validate(objects[0], args.stop_atk_id)
            stopped = run(['docker', 'stop', '--time', '5', args.stop_atk_id], timeout=20)
            row.update(status='STOP_SUCCEEDED' if stopped.returncode == 0 else 'STOP_FAILED',
                       returncode=stopped.returncode, stdout=stopped.stdout, stderr=stopped.stderr)
            receipt['cleanup'] = row
            if stopped.returncode and 'No such' not in stopped.stderr:
                output.write_text(json.dumps(receipt, indent=2) + '\n')
                raise RuntimeError('Exact ATK stop failed')
    sys.path.insert(0, str(ROOT))
    import controller
    command = [sys.executable, '-B', str(ROOT / 'controller.py'), '--card', '5',
               '--attempt', args.attempt, '--probe', 'model_probe.py', '--case-file',
               'setup/model_a4/' + args.case + '.json', '--capture', 'native-logs',
               '--case-seconds', '900', '--max-seconds', '1200', '--max-log-mib', '8192']
    for identity in controller.REVIEWED_IDLE_CONTAINERS:
        command += ['--allow-idle-container-ID', identity]
    receipt['controller_command'] = command
    with output.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    # The controller owns its timeout, signal cleanup and lock lifecycle.
    raise SystemExit(subprocess.call(command))


if __name__ == '__main__':
    main()
