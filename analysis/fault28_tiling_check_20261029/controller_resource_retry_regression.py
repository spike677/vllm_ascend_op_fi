"""CPU-only mocked resource-gate regression; does not run Docker, SSH, or NPU."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch


OWNED = 'c3f246a08c31f19351abf8d1ae4f56658c4625a69966bbbc0a5861c9fe824ddf'
PID = 421611


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('controller', type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('fault28_controller_tested', args.controller)
    c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c)
    completed = []

    def live(pid=PID, owner=OWNED, start=230444979):
        return {'host_pid': pid, 'starttime_ticks': start, 'pid_namespace': 'pid:[4026538356]',
                'nspid_vector': [pid, 105], 'process_state': 'S', 'process_flags': 0,
                'cgroup': '0::/docker/' + owner, 'pf_exiting': False, 'state_before_read': 'S'}

    def reset(prove=True):
        c.PROVEN_OWNED_PROCESSES.clear()
        c.OWNED_MISSING_PID_SINCE.clear()
        c.OWNED_TEARDOWN_SINCE.clear()
        c.IDLE_BASELINES.clear()
        c.ALLOW_IDLE_IDS.clear()
        if prove:
            assert c.classify_process_identity(live(), OWNED, now=99)[0]

    def reject(fn):
        try:
            fn()
        except RuntimeError:
            return
        raise AssertionError('Expected fail-closed rejection')

    class Harness:
        def __init__(self, frames, absent=True):
            self.frames, self.absent = frames, absent
            self.index, self.current = -1, None
            self.now = 100.0
            self.records, self.sleeps, self.query_timeouts = [], [], []

        def sleep(self, seconds):
            assert 0 < seconds <= 0.5
            self.sleeps.append(seconds)
            self.now += seconds

        def run(self, argv, timeout=25):
            self.query_timeouts.append(timeout)
            if argv[:2] == ['docker', 'ps']:
                return ''
            assert argv == ['npu-smi', 'info']
            self.index += 1
            self.current = self.frames[min(self.index, len(self.frames) - 1)]
            lines = ['| 2 910B4 |', '| 5 910B4 |', '| NPU Chip | Process id |',
                     '| No running processes found in NPU 2 |']
            lines += [f'| 5 0 | {pid} | worker | 10 |' for pid in self.current]
            if not self.current:
                lines.append('| No running processes found in NPU 5 |')
            return '\n'.join(lines)

        def identity(self, pid):
            mode = self.current[pid]
            if mode == 'missing':
                raise FileNotFoundError('/proc/' + str(pid) + '/stat')
            if mode == 'foreign':
                return live(pid, 'f' * 64)
            if mode == 'reused':
                return live(pid, 'f' * 64, start=999999999)
            assert mode == 'owned'
            return live(pid)

        def call(self, **kwargs):
            with patch.object(c, 'run', self.run), patch.object(c, 'read_process_identity', self.identity), \
                 patch.object(c, 'process_directory_absent', lambda pid: self.absent), \
                 patch.object(c, 'append_resource_check', lambda x: self.records.append(copy.deepcopy(x))), \
                 patch.object(c.time, 'monotonic', lambda: self.now), \
                 patch.object(c.time, 'time', lambda: self.now), patch.object(c.time, 'sleep', self.sleep):
                return c.resources(5, kwargs.pop('owned', OWNED), **kwargs)

    reset()
    h = Harness([{PID: 'missing'}, {}])
    result = h.call(require_idle=True)
    assert result['eligible'] and result['selected_card_table_idle'] and h.now == 100.5
    assert h.records[0]['eligible'] is False
    assert h.records[0]['retry_pending_owned_missing_processes'][0]['owned'] is False
    assert (OWNED, PID) not in c.PROVEN_OWNED_PROCESSES and not c.OWNED_MISSING_PID_SINCE
    assert len([r for r in h.records if r.get('event') == 'owned_missing_pid_retry_wait']) == 1
    completed.append('proven_proc_gone_driver_row_retries_then_clears_and_retires_proof')

    h = Harness([{PID: 'missing'}])
    reject(h.call)
    assert not h.sleeps
    completed.append('retired_pid_proof_cannot_approve_later_unseen_reuse')

    reset()
    h = Harness([{PID: 'missing'}])
    reject(lambda: h.call(require_idle=True))
    assert h.now == 110 and sum(h.sleeps) == 10
    assert any(r.get('event') == 'owned_missing_pid_retry_failed' for r in h.records)
    assert all(t <= 10 for t in h.query_timeouts[2:])
    completed.append('persistent_driver_residual_fails_at_10_second_deadline')

    for name, proven, absent, owner in [
        ('never_proven_pid', False, True, OWNED),
        ('proc_directory_still_exists', True, False, OWNED),
        ('different_attempt_container', True, True, 'e' * 64),
        ('no_owned_container', True, True, None),
    ]:
        reset(proven)
        h = Harness([{PID: 'missing'}], absent=absent)
        reject(lambda: h.call(owned=owner))
        assert not h.sleeps
        completed.append(name + '_fails_without_retry')

    reset()
    h = Harness([{PID: 'missing', 700001: 'foreign'}])
    reject(h.call)
    assert not h.sleeps
    completed.append('mixed_foreign_and_proven_missing_fails_immediately')

    for mode in ('foreign', 'reused'):
        reset()
        h = Harness([{PID: 'missing'}, {PID: mode}])
        reject(h.call)
        assert h.now == 100.5
        completed.append(mode + '_appearing_during_retry_fails_closed')

    reset()
    h = Harness([{PID: 'owned'}])
    assert h.call()['eligible']
    reject(lambda: h.call(require_idle=True))
    assert not h.sleeps
    completed.append('live_owned_allowed_during_case_but_blocks_next_case')

    reset()
    h = Harness([{PID: 'missing'}])
    result = h.call(strict=False)
    assert not result['eligible'] and not h.sleeps
    completed.append('non_strict_snapshot_never_labels_pending_eligible')

    reset()
    c.OWNED_MISSING_PID_SINCE[(OWNED, PID, 230444979)] = 90
    h = Harness([{PID: 'missing'}])
    reject(h.call)
    assert not h.sleeps
    completed.append('expired_prior_retry_window_does_not_restart')

    reset()
    with patch.object(c.Path, 'stat', side_effect=PermissionError('denied')):
        try:
            c.process_directory_absent(PID)
        except PermissionError:
            pass
        else:
            raise AssertionError('Permission failure must not mean absent')
    completed.append('permission_error_not_misclassified_as_absent')

    previous = live()
    previous.update(cgroup='0::/', process_state='Z')
    assert c.classify_process_identity(previous, OWNED, now=100)[0]
    assert not c.classify_process_identity(previous, OWNED, now=161)[0]
    completed.append('existing_strong_identity_zombie_grace_retains_bound')

    source = args.controller.read_bytes()
    assert "'hf_runtime_probe.py'" in source.decode()
    print(json.dumps({'status': 'PASS_CPU_RESOURCE_RETRY_REGRESSION', 'tests': completed,
                      'count': len(completed), 'controller_sha256': hashlib.sha256(source).hexdigest(),
                      'NPU_executed': False, 'external_commands_executed': False}, indent=2))


if __name__ == '__main__':
    main()
