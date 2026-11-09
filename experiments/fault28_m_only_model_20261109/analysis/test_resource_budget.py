"""CPU-only finite-budget boundaries; never executes the controller or devices."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(HERE))
import controller
import native_launch_census as census


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rejected(fn):
    try: fn()
    except (RuntimeError, ValueError): return
    raise AssertionError('Expected rejected bounded input')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    checks = []
    def check(name, fn):
        fn(); checks.append({'name': name, 'passed': True})

    def durations():
        assert controller.execution_budget(60, 180, 16, 1)['case_seconds'] == 60
        maximum = controller.execution_budget(3600, 7200, 16384, 8)
        assert maximum['max_log_bytes_per_case'] == 16 * 1024 ** 3
        for case, total in [(59, 180), (3601, 7200), (60, 179), (3600, 7201), (True, 180), (60, 180.)]:
            rejected(lambda case=case, total=total: controller.execution_budget(case, total, 16, 1))
    check('duration_lower_upper_edges_and_invalid_types', durations)

    def log_bounds():
        for log in (15, 16385, 0, -1, True, 2048.):
            rejected(lambda log=log: controller.execution_budget(900, 1800, log, 1))
        for count in (0, 9, True, 2.):
            rejected(lambda count=count: controller.execution_budget(900, 1800, 2048, count))
    check('finite_log_and_case_count_bounds', log_bounds)

    def defaults():
        tree = ast.parse((EXPERIMENT / 'controller.py').read_text())
        found = {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'add_argument' and node.args and isinstance(node.args[0], ast.Constant)):
                for keyword in node.keywords:
                    if keyword.arg == 'default': found[node.args[0].value] = ast.literal_eval(keyword.value)
        assert found['--case-seconds'] == 900 and found['--max-seconds'] == 1800 and found['--max-log-mib'] == 2048
    check('pilot_default900_1800_2048_unchanged', defaults)

    budget = controller.execution_budget(3600, 7200, 16384, 2)
    def disk(free, remaining, phase):
        usage = SimpleNamespace(total=200 * 1024 ** 3, used=200 * 1024 ** 3 - free, free=free)
        with patch.object(controller.shutil, 'disk_usage', return_value=usage) as reader:
            row = controller.disk_snapshot(budget, remaining, phase)
        reader.assert_called_once_with(controller.ROOT)
        return row

    def before_disk():
        required = 52 * 1024 ** 3  #2*(16GiB log+6GiB sidecars)+8GiB.
        row = disk(required, 2, 'before_container')
        assert row['eligible'] is True and row['required_free_bytes'] == required
        assert disk(required - 1, 2, 'before_container')['eligible'] is False
        assert disk(required + 1, 2, 'before_container')['eligible'] is True
    check('initial_disk_exact52GiB_threshold_and_one_byte_rejection', before_disk)

    def next_case_disk():
        required = 30 * 1024 ** 3
        assert disk(required, 1, 'before_case')['eligible'] is True
        assert disk(required - 1, 1, 'before_case')['eligible'] is False
        assert disk(required, 2, 'before_case')['eligible'] is False
        assert disk(required, 1, 'before_case')['required_free_bytes'] == required
    check('remaining_case_reservation_reduces_only_after_completed_case', next_case_disk)

    def running_disk():
        floor = 8 * 1024 ** 3
        assert disk(floor, 2, 'running')['eligible'] is True
        assert disk(floor - 1, 2, 'running')['eligible'] is False
        assert disk(floor, 1, 'running')['required_free_bytes'] == floor
    check('running_disk8GiB_floor_inclusive', running_disk)

    def invalid_disk():
        for count in (0, 3, True):
            rejected(lambda count=count: disk(100 * 1024 ** 3, count, 'before_case'))
        rejected(lambda: disk(100 * 1024 ** 3, 1, 'unknown'))
    check('disk_unknown_phase_and_remaining_count_rejected', invalid_disk)

    def census_bounds():
        for size in (1, 2 * 1024 ** 3 + 1, 16 * 1024 ** 3): census.check_log_size(size)
        for size in (0, -1, 16 * 1024 ** 3 + 1, True, 1.):
            rejected(lambda size=size: census.check_log_size(size))
        source = (HERE / 'native_launch_census.py').read_text()
        assert "with log_path.open('rb') as stream:" in source and 'for number, payload in enumerate(stream, 1):' in source
        assert 'digest.update(payload)' in source
    check('census16GiB_finite_edges_preserve_streamed_byte_hashing', census_bounds)

    def unchanged_devices():
        old = Path('/home/fault28-model-coverage-20261030/controller.py')
        def functions(path):
            tree = ast.parse(path.read_text())
            return {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}
        before, after = functions(old), functions(EXPERIMENT / 'controller.py')
        for name in ('device_capability', 'idle_metadata', 'resource_snapshot', 'resources', 'verify_owned',
                     'classify_process_identity', 'missing_process_identity'):
            assert before[name] == after[name], name
        latest=functions(Path('/home/fault28-fia-graph-20261106/controller.py'))
        assert latest['recover_owned_id']==after['recover_owned_id']
    check('seven_original_gates_and_later_bounded_recovery_unchanged', unchanged_devices)
    report = {'schema': 'nes.model_coverage.budget_cpu.v1', 'status': 'PASS_CPU_BUDGET_CHECKS',
              'checks': checks, 'npu_calls': 0, 'containers_started': 0, 'ssh_connections': 0,
              'sources': {str(p.relative_to(EXPERIMENT)).replace('\\', '/'): sha(p) for p in
                  (EXPERIMENT / 'controller.py', HERE / 'native_launch_census.py', Path(__file__))}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps({'status': report['status'], 'checks': len(checks), 'sha256': sha(args.output)}))


if __name__ == '__main__':
    main()
