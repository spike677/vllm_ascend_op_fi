"""Host-side session wrapper around the two ALREADY VERIFIED launchers.

No model parameters or native algorithms are defined here. The existing
NPU_LOCK/owner.json remains the coordination authority. Run on the A2 host.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import urllib.request

ROOT = Path('/home/haoran/demoncat_overnight_20260907')
LOCK = ROOT / 'coordination/NPU_LOCK'
REF = ROOT / 'fake_requant/reference_experiment'
EXPERIMENT = ROOT / 'evaluation/fake_requant_effect_evaluation_v1_complete_20260908T074102Z/scripts/run_v17_fake_requant_summary_rootfs.sh'
ORIGINAL = ROOT / 'evaluation/module_injection_surface_discovery_v1_20260909T020000Z/run_v17_bundle_summary_rootfs.sh'


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return {'http_status': response.status, 'body': response.read().decode()}
    except Exception as exc:
        return {'http_status': None, 'error': str(exc)}


def launch(output, original=False):
    old = json.loads(Path('/home/haoran/fev_20260908_074102/environment.json').read_text())
    spec = json.loads((REF / 'r1_launch_spec.json').read_text())
    env = dict(os.environ)
    env.update(FAULT20_ROOTFS=old['rootfs'], FAULT20_RUN_DIR=str(output),
               FAULT20_HOST_MODEL_PATH=old['host_model_path'], FAULT20_BUNDLE=old['bundle'],
               FAULT20_PHYSICAL_DEVICES=','.join(map(str, spec['physical_npu_devices'])),
               FAULT20_ASCEND_RT_VISIBLE_DEVICES=','.join(map(str, spec['visible_npu_devices'])),
               FAULT20_PORT=str(spec['port']), FAULT20_MODEL_NAME=spec['model_name'],
               FAULT20_ROOTFS_SOURCE_CONTAINER=old['rootfs_source_container'],
               FAKE_REQUANT_HOST_SOURCE=str(REF), FAULT20_MIN_HOME_FREE_KB=str(12*1024*1024))
    launcher = ORIGINAL if original else EXPERIMENT
    with output.with_suffix('.startup.log').open('xb') as log:
        child = subprocess.Popen([str(launcher)], env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return {'launcher': str(launcher), 'sha256': digest(launcher), 'pid': child.pid,
            'run_dir': str(output), 'env': {k: v for k, v in env.items() if k.startswith(('FAULT20_', 'FAKE_REQUANT_'))}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'status', 'restore'])
    parser.add_argument('--session', required=True, type=Path)
    args = parser.parse_args()
    session = args.session.resolve()
    if session.parent != Path('/home/haoran') or not session.name.startswith('distlive_'):
        raise ValueError('session must be a new /home/haoran/distlive_* directory')
    if args.action == 'start':
        assert not session.exists(), 'preserve existing attempt'
        assert shutil.disk_usage('/home').free >= 12*1024**3
        assert not subprocess.check_output(['ss', '-H', '-ltn', 'sport = :18031']).strip()
        smi = subprocess.check_output(['npu-smi', 'info'], text=True)
        assert all(f'No running processes found in NPU {i}' in smi for i in (2, 5)), smi
        LOCK.mkdir(exist_ok=True)
        with (LOCK / 'owner.json').open('r+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            previous = json.load(handle)
            assert previous.get('owner') is None and previous.get('status') == 'RELEASED', previous
            session.mkdir()
            write(session / 'previous_lock.json', previous)
            owner = {'owner': 'EVALUATION', 'branch': 'feature/distortion-evaluation-v1',
                     'phase': 'NORMAL_DIAGNOSTIC_CASES_LIVE', 'status': 'ACTIVE',
                     'result_dir': str(session), 'updated_at': now()}
            # Keep the same inode under advisory lock; mkdir-based peers also see NPU_LOCK.
            handle.seek(0); json.dump(owner, handle); handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        audit = {'started_at': now(), 'npu_before': smi, 'free_bytes': shutil.disk_usage('/home').free,
                 'original_launcher': {'path': str(ORIGINAL), 'sha256': digest(ORIGINAL)},
                 'extension_sha256': digest(REF / 'fake_requant_reference_extension.py'),
                 'launch_spec': json.loads((REF / 'r1_launch_spec.json').read_text())}
        try:
            audit['experiment'] = launch(session / 'service')
        except Exception as exc:
            audit['start_error'] = repr(exc)
            write(session / 'service_startup_audit.json', audit)
            raise
        write(session / 'service_startup_audit.json', audit)
        print(json.dumps(audit))
    elif args.action == 'status':
        audit = json.loads((session / 'service_startup_audit.json').read_text())
        print(json.dumps({'health': get('http://127.0.0.1:18031/health'),
                          'models': get('http://127.0.0.1:18031/v1/models'), 'audit': audit}))
        for path in [session / 'service.startup.log', session / 'service/harness.log', session / 'service/server.log']:
            if path.exists():
                with path.open('rb') as handle:
                    handle.seek(max(0, path.stat().st_size-5000)); print(str(path), handle.read().decode(errors='replace'))
    else:
        owner = json.loads((LOCK / 'owner.json').read_text())
        assert owner.get('result_dir') == str(session) and owner.get('owner') == 'EVALUATION'
        audit = json.loads((session / 'service_startup_audit.json').read_text())
        pid = audit['experiment']['pid']
        command = Path(f'/proc/{pid}/cmdline')
        if command.exists():
            assert str(EXPERIMENT).encode() in command.read_bytes(), 'PID identity changed; do not signal'
            os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            smi = subprocess.check_output(['npu-smi', 'info'], text=True)
            if not subprocess.check_output(['ss', '-H', '-ltn', 'sport = :18031']).strip() and all(
                    f'No running processes found in NPU {i}' in smi for i in (2, 5)):
                break
            time.sleep(2)
        else:
            raise RuntimeError('owned service has not stopped safely; do not launch duplicate')
        restored = launch(session / 'original_service', original=True)
        write(session / 'restoration_start.json', restored)
        print(json.dumps(restored))
        # Health/final no-extension checks and lock release are intentionally a
        # separate explicit verification step, never inferred from Popen success.


if __name__ == '__main__':
    main()
