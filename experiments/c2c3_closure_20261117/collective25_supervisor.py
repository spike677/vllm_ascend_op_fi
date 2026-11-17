"""Observe true torchrun exit; bounded timeout is failure, never normal drain."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def assess(folder, exit_code, timed_out, cpu=False):
    ranks = []
    for rank in range(2):
        receipt_path = folder / f'rank{rank}.json'
        lifecycle_path = folder / f'rank{rank}.lifecycle.jsonl'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
        events = [json.loads(line) for line in lifecycle_path.read_text().splitlines()] if lifecycle_path.is_file() else []
        stages = [event['stage'] for event in events]
        cases = receipt.get('cases', [])
        ranks.append(dict(rank=rank, pid=receipt.get('pid'), status=receipt.get('status'),
            same_process_off_exact=bool(cases) and all(case['off_exact'] for case in cases),
            group_drain_returned=receipt.get('group_drain', {}).get('status') == 'CROSS_RANK_DRAIN_RETURNED',
            destroy_returned='destroy_process_group_returned' in stages,
            main_returned='main_returned' in stages,
            native_shutdown_returned=cpu or 'native_shutdown_returned' in stages,
            atexit_tail='python_atexit_tail' in stages,
            last_event=events[-1] if events else None,
            receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest() if receipt_path.is_file() else None,
            lifecycle_sha256=hashlib.sha256(lifecycle_path.read_bytes()).hexdigest() if lifecycle_path.is_file() else None))
    normal_exit = exit_code == 0 and not timed_out
    lifecycle_ok = normal_exit and all(all(row[key] for key in (
        'same_process_off_exact', 'group_drain_returned', 'destroy_returned',
        'main_returned', 'native_shutdown_returned', 'atexit_tail')) for row in ranks)
    return dict(status='CALIBRATION_LIFECYCLE_PASSED' if lifecycle_ok else 'CALIBRATION_LIFECYCLE_FAILED',
                torchrun_exit_code=exit_code, timed_out=timed_out, normal_exit_observed=normal_exit,
                lifecycle_pass=lifecycle_ok, ranks=ranks, c2_pass=False, c3_pass=False,
                actual_fault_recovery_proven=False, injection_attempted=False,
                note='Calibration/OFF/health/drain/exit only; a no-sample run grants no injection qualification.')


def stop_owned(proc):
    """Terminate only the new process group created by this supervisor."""
    if proc.poll() is not None:
        return
    if os.name == 'posix':
        if os.getpgid(proc.pid) != proc.pid:
            raise RuntimeError('OWNED_LAUNCHER_PROCESS_GROUP_CHANGED')
        os.killpg(proc.pid, signal.SIGTERM)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if os.name == 'posix':
            if os.getpgid(proc.pid) != proc.pid:
                raise RuntimeError('OWNED_LAUNCHER_PROCESS_GROUP_CHANGED')
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        proc.wait(timeout=15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--suite', choices=('smoke', 'bounded'), default='smoke')
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--stage-timeout', type=float, default=60)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--master-port', type=int, default=29500)
    args = parser.parse_args()
    if not 0 < args.timeout <= 1500 or not 0 < args.stage_timeout < args.timeout:
        parser.error('Bounded positive stage and supervisor timeouts required')
    args.out.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name('collective25_closure_probe.py')
    command = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1',
        '--master-addr=127.0.0.1', f'--master-port={args.master_port}', '--nproc-per-node=2',
        str(script), '--out', str(args.out), '--suite', args.suite,
        '--repeats', str(args.repeats), '--timeout', str(args.stage_timeout)]
    if args.cpu:
        command.append('--cpu')
    started = time.monotonic()
    timed_out = False
    with (args.out / 'torchrun.log').open('x') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=os.name == 'posix')
        try:
            exit_code = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_owned(proc)
            exit_code = proc.returncode
        except BaseException:
            stop_owned(proc)
            raise
    result = assess(args.out, exit_code, timed_out, cpu=args.cpu)
    result.update(command=command, launcher_pid=proc.pid, elapsed_seconds=time.monotonic() - started,
                  supervisor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  probe_sha256=hashlib.sha256(script.read_bytes()).hexdigest())
    (args.out / 'supervisor.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return 0 if result['lifecycle_pass'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
