"""Root-owned 124 queue; cooperative lock, per-case gate, owned cleanup only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from queue_contract import ROOT, LABEL, LOCK, validate_cases


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args, timeout=30):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT,
                                   timeout=timeout)


def inspect_owned(name):
    result = subprocess.run(['docker', 'inspect', name], text=True,
                            capture_output=True, timeout=30)
    if result.returncode:
        # Absence is not interpreted as successful Worker recovery.
        return None
    info = json.loads(result.stdout)[0]
    if info.get('Config', {}).get('Labels', {}).get('dcat.task') != LABEL:
        raise RuntimeError('CONTAINER_OWNERSHIP_CHANGED')
    return info


def cleanup_owned(name):
    info = inspect_owned(name)
    if info is None:
        return {'status': 'ABSENT_OR_INSPECTION_FAILED', 'recovery_proven': False}
    initial = info['State']
    if initial['Running']:
        command(['docker', 'stop', '--time', '20', name], timeout=35)
    after = inspect_owned(name)
    if after is None or after['State']['Running']:
        raise RuntimeError('OWNED_CONTAINER_STOP_UNPROVEN')
    command(['docker', 'rm', name])
    return {'status': 'OWNED_CONTAINER_REMOVED', 'state_before_cleanup': initial,
            'final_state': after['State'], 'recovery_proven': False,
            'note': 'Container cleanup does not prove same-process OFF recovery.'}


def verify_source_mount(source):
    path = Path(source)
    manifest_path = path / 'source_manifest.json'
    if not manifest_path.is_file():
        raise RuntimeError('SOURCE_MANIFEST_REQUIRED')
    manifest = json.loads(manifest_path.read_text())
    for item in manifest['files']:
        target = (path / item['path']).resolve()
        if (not target.is_relative_to(path.resolve()) or target.is_symlink()
                or digest(target) != item['sha256']):
            raise RuntimeError('SOURCE_BYTES_CHANGED:' + item['path'])
    return {'manifest_sha256': digest(manifest_path),
            'base_commit': manifest.get('base_commit'), 'files': len(manifest['files'])}


def run_case(case, cards, folder, save, row):
    name = 'c23-' + case['id'] + '-' + str(os.getpid())
    budget = case['review']['resource_budget']
    row['image_id'] = command(['docker', 'image', 'inspect', '--format', '{{.Id}}',
                               case['image']]).strip()
    args = ['docker', 'run', '--runtime', 'runc', '--name', name,
            '--label', 'dcat.task=' + LABEL, '--network', 'none',
            '--memory', str(budget['memory_gib']) + 'g', '--cpus', str(budget['cpus']),
            '--pids-limit', '4096', '--shm-size', '4g']
    for device in [*(f'davinci{x}' for x in cards),
                   'davinci_manager', 'devmm_svm', 'hisi_hdc']:
        args += ['--device', '/dev/' + device]
    mounts = [('/usr/local/Ascend/driver', '/usr/local/Ascend/driver', 'ro'),
              ('/usr/local/dcmi', '/usr/local/dcmi', 'ro'),
              ('/dev/null', '/etc/ld.so.preload', 'ro'),
              (str(folder), '/evidence', 'rw'),
              ('/home/batch-scale-dynamic-msprobe-20261023/python_packages',
               '/msprobe_packages', 'ro'), *case.get('mounts', [])]
    row['frozen_scripts'] = []
    row['source_manifests'] = []
    for source, dest, mode in mounts:
        source_path = Path(source)
        if not source_path.exists():
            raise RuntimeError('MOUNT_SOURCE_MISSING:' + source)
        if dest == '/source':
            row['source_manifests'].append(verify_source_mount(source))
        if mode == 'ro' and source_path.is_file() and source_path.suffix == '.py':
            if source_path.stat().st_size > 2 * 1024 * 1024:
                raise RuntimeError('SCRIPT_INPUT_TOO_LARGE')
            raw = source_path.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            frozen = folder / 'inputs' / (sha + '-' + source_path.name)
            frozen.parent.mkdir(exist_ok=True)
            if not frozen.exists():
                with frozen.open('xb') as out:
                    out.write(raw)
            row['frozen_scripts'].append(dict(original=source, destination=dest,
                                              sha256=sha, frozen=str(frozen)))
            source = str(frozen)
        args += ['--mount', f'type=bind,src={source},dst={dest}' +
                 (',readonly' if mode == 'ro' else '')]
    env = dict(PYTHONDONTWRITEBYTECODE='1', TASK_QUEUE_ENABLE='0',
               VLLM_SERVER_DEV_MODE='1', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    env.update(case.get('env', {}))
    for key, value in env.items():
        args += ['-e', key + '=' + value]
    args += ['--entrypoint', '/bin/bash', row['image_id'], '-c', case['command']]
    row['command'] = args
    row['started_utc'] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps({'status': 'CASE_START', 'id': case['id'], 'cards': cards}), flush=True)
    try:
        with (folder / 'process.log').open('xb') as log:
            proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
            try:
                row['exit_code'] = proc.wait(timeout=budget['timeout_seconds'])
            except subprocess.TimeoutExpired:
                row['timeout'] = True
                info = inspect_owned(name)
                if info is not None and info['State']['Running']:
                    command(['docker', 'stop', '--time', '20', name], timeout=35)
                row['exit_code'] = proc.wait(timeout=35)
        info = inspect_owned(name)
        row['container_state'] = info['State'] if info else None
    finally:
        try:
            row['cleanup'] = cleanup_owned(name)
        finally:
            row['ended_utc'] = datetime.now(timezone.utc).isoformat()
            save()
    print(json.dumps({'status': 'CASE_END', 'id': case['id'],
                      'exit_code': row.get('exit_code')}), flush=True)


def main():
    import fcntl
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'fault39' / 'tools'))
    from run_124 import idle_gate
    p = argparse.ArgumentParser()
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--attempt', required=True)
    p.add_argument('--cards', nargs='+', type=int, choices=[2, 5], required=True)
    a = p.parse_args()
    if not re.fullmatch(r'[a-z0-9_-]{1,80}', a.attempt):
        p.error('Invalid attempt')
    raw_manifest = a.cases.read_bytes()
    cases = validate_cases(json.loads(raw_manifest), a.cards)
    folder = Path(ROOT) / a.attempt
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'reviewed_cases.json').write_bytes(raw_manifest)
    receipt = dict(schema='demoncat.c2c3.queue.v1', cards=a.cards, cases=[],
                   status='WAITING_FOR_LOCK', controller_sha256=digest(__file__),
                   contract_sha256=digest(Path(__file__).with_name('queue_contract.py')),
                   case_manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
                   grants_mechanism_qualification=False)

    def save():
        temporary = folder / 'controller.tmp'
        temporary.write_text(json.dumps(receipt, indent=2) + '\n')
        temporary.replace(folder / 'controller.json')

    save()
    try:
        with open(LOCK, 'a') as lock:
            # Fail fast. Root can enqueue a new attempt after a previous owner exits.
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                receipt['status'] = 'RESOURCE_LOCK_BUSY'
                save()
                return 3
            for case in cases:
                receipt['status'] = 'CHECKING_DEVICE'
                gate = idle_gate(a.cards)
                row = dict(id=case['id'], review=case['review'], gate=gate)
                receipt['cases'].append(row)
                save()
                if not gate['idle']:
                    receipt['status'] = 'DEVICE_BUSY'
                    save()
                    return 3
                case_folder = folder / case['id']
                case_folder.mkdir()
                receipt['status'] = 'RUNNING_CASE'
                save()
                run_case(case, a.cards, case_folder, save, row)
                row['post_cleanup_gate'] = idle_gate(a.cards)
                save()
                if row.get('exit_code') != 0:
                    receipt['status'] = 'CASE_FAILED'
                    save()
                    return 2
                if not row['post_cleanup_gate']['idle']:
                    receipt['status'] = 'POST_CASE_DEVICE_BUSY'
                    save()
                    return 3
            receipt['status'] = 'CASES_COMPLETED_REQUIRES_QUALIFICATION_REVIEW'
            save()
            return 0
    except BaseException as error:
        receipt['status'] = 'QUEUE_FAILED'
        receipt['error'] = type(error).__name__ + ':' + str(error)
        save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
