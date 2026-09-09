"""Bounded host supervisor: await clean calibration, run cycles, always restore.

Only drives the existing evaluation client and verified service-session wrapper.
It never defines a fault or silently retries a failed experimental request.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2); handle.write('\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--calibration-pid', type=int, required=True)
    a = p.parse_args()
    session, out = a.session, a.session / 'normal_v2'
    lock = Path('/home/haoran/demoncat_overnight_20260907/coordination/NPU_LOCK/owner.json')
    def owned():
        owner = json.loads(lock.read_text())
        if owner.get('owner') != 'EVALUATION' or owner.get('result_dir') != str(session):
            raise RuntimeError('lock ownership changed: stop touching service')
    owned()
    started = time.monotonic()
    child = None
    final = {'started_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'hard_deadline_hours': 12}
    try:
        # No fault is allowed until BOTH full calibration runs have completed.
        while not (out / 'calibrate_final.json').exists():
            owned()
            proc = Path(f'/proc/{a.calibration_pid}/cmdline')
            if not proc.exists() or b'run_normal_diagnostic_live.py' not in proc.read_bytes():
                raise RuntimeError('calibration exited without a final record')
            if time.monotonic() - started > 3*3600:
                os.kill(a.calibration_pid, signal.SIGINT)
                raise RuntimeError('calibration deadline exceeded')
            time.sleep(10)
        calibration_final = json.loads((out / 'calibrate_final.json').read_text())
        if any(k in calibration_final for k in ('error', 'cleanup_error')) or calibration_final.get('health') is not True:
            raise RuntimeError('calibration cleanup or health failed')
        calibration = json.loads((out / 'calibration.json').read_text())
        assert len(calibration['matrix']) == 16
        save(out / 'pre_fault_freeze.json', {
            'calibration_sha256': hashlib.sha256((out / 'calibration.json').read_bytes()).hexdigest(),
            'suite_sha256': hashlib.sha256((out / 'suite.json').read_bytes()).hexdigest(),
            'frozen_at': dt.datetime.now(dt.timezone.utc).isoformat(),
            'threshold_retuning': False, 'fault_parameter_changes': False,
            'source_sha256': {str(f.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(f.read_bytes()).hexdigest()
                              for f in Path(__file__).resolve().parents[1].rglob('*.py') if '__pycache__' not in f.parts}})
        script = Path(__file__).with_name('run_normal_diagnostic_live.py')
        with (session / 'cycles_v2_runner.log').open('xb') as log:
            child = subprocess.Popen([sys.executable, str(script), '--session', str(session),
                                      '--stage', 'cycles', '--suite', str(out / 'suite.json')],
                                     stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            save(out / 'cycles_process.json', {'pid': child.pid, 'script': str(script)})
            while child.poll() is None:
                owned()
                if time.monotonic() - started > 12*3600:
                    child.send_signal(signal.SIGINT)
                    try:
                        child.wait(timeout=240)
                    except subprocess.TimeoutExpired:
                        raise RuntimeError('cycle timeout; owned service will be stopped for cleanup')
                    raise RuntimeError('full run deadline exceeded; partial artifacts preserved')
                time.sleep(10)
        final['cycle_exit_code'] = child.returncode
        if child.returncode != 0:
            raise RuntimeError('cycle runner failed; see preserved raw attempt and cycles_final.json')
        final['live_complete'] = True
    except BaseException as exc:
        final['error'] = repr(exc)
    finally:
        owned()
        # Stop our client before switching our service. Never signal another task.
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=240)
            except subprocess.TimeoutExpired:
                final['client_timeout'] = True
        restore = Path(__file__).with_name('distortion_service_session.py')
        with (session / 'restore_runner.log').open('xb') as log:
            result = subprocess.run([sys.executable, str(restore), 'restore', '--session', str(session)],
                                    stdout=log, stderr=subprocess.STDOUT, timeout=180)
        final['restore_launcher_exit'] = result.returncode
        healthy = False
        if result.returncode == 0:
            for _ in range(180):
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18031/health', timeout=5) as response:
                        healthy = response.status == 200
                    if healthy:
                        break
                except Exception:
                    pass
                time.sleep(5)
        final['original_service_healthy'] = healthy
        if healthy:
            with urllib.request.urlopen('http://127.0.0.1:18031/v1/models', timeout=10) as response:
                final['original_models'] = json.load(response)
            req = urllib.request.Request('http://127.0.0.1:18031/collective_rpc',
                data=json.dumps({'method': 'fake_requant_query', 'args': [], 'kwargs': {}}).encode(),
                headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(req, timeout=20) as response:
                    final['unexpected_fake_requant_rpc'] = response.read().decode()
            except Exception as exc:
                body = exc.read().decode() if hasattr(exc, 'read') else str(exc)
                final['original_rpc_absence'] = {'status': getattr(exc, 'code', None), 'body': body}
            bundle = Path(json.loads((session / 'service_startup_audit.json').read_text())['experiment']['env']['FAULT20_BUNDLE'])
            ctl = subprocess.run([str(bundle / 'ctl/fault20_buffer_ctl'), 'status'],
                                 env={**os.environ, 'VLLM_SERVER_DEV_MODE': '1'}, text=True, capture_output=True)
            final['fault20_control'] = {'returncode': ctl.returncode, 'stdout': ctl.stdout, 'stderr': ctl.stderr}
            try:
                with urllib.request.urlopen('http://127.0.0.1:18031/health', timeout=10) as response:
                    final['original_service_healthy_after_rpc_check'] = response.status == 200
            except Exception as exc:
                final['original_service_healthy_after_rpc_check'] = False
                final['post_rpc_health_error'] = repr(exc)
        final['completed_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        # Active faults are removed by the client finally, then the exact owned
        # service is stopped. New original service has no FakeRequant extension.
        final['fault_state_basis'] = 'client cleanup plus original no-extension launcher restart'
        save(session / 'final_environment.json', final)
        with lock.open('r+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            owner = json.load(handle)
            assert owner.get('result_dir') == str(session)
            released = {**owner, 'owner': None, 'status': 'RELEASED', 'phase': 'NORMAL_DIAGNOSTIC_LIVE_FINISHED',
                        'final_environment': str(session / 'final_environment.json'), 'updated_at': final['completed_at']}
            handle.seek(0); json.dump(released, handle); handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        print(json.dumps(final), flush=True)


if __name__ == '__main__':
    main()
