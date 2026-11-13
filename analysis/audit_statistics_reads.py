"""Recompute all three steps with a strict read allowlist, no runtime input."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def run(baseline, fault, source, output):
    sys.path.insert(0, str(source.resolve()))
    from compare_statistics import compare
    from accept_statistics import accept
    roots = {'baseline': baseline.resolve(), 'fault': fault.resolve()}
    allowed = {}
    for arm, root in roots.items():
        for name in ('config.json', 'collector_provenance.json'):
            allowed[(root / name).resolve()] = arm + '/' + name
        for step in range(3):
            for name in ('dump.json', 'stack.json', 'construct.json'):
                relative = Path(f'observation/step{step}/rank0') / name
                allowed[(root / relative).resolve()] = arm + '/' + relative.as_posix()
    accesses = []
    active = True
    def hook(event, args):
        if not active:
            return
        if event == 'open':
            path, mode, flags = args
            if isinstance(path, int):
                raise RuntimeError('File descriptor input is not allowed')
            resolved = Path(os.fsdecode(path)).resolve()
            if resolved not in allowed or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC):
                raise RuntimeError('Non-statistical file read/write rejected: ' + str(resolved))
            accesses.append(allowed[resolved])
        elif event in ('subprocess.Popen', 'os.system', 'socket.connect'):
            raise RuntimeError('External process/network input is not allowed during localization')
    sys.addaudithook(hook)
    rows = []
    for step in range(3):
        relative = Path(f'observation/step{step}/rank0/dump.json')
        result = compare(roots['baseline'] / relative, roots['fault'] / relative, rtol=1e-5, atol=1e-8)
        self_result = compare(roots['baseline'] / relative, roots['baseline'] / relative, rtol=1e-5, atol=1e-8)
        assert self_result['first_exact_api_difference'] is None
        assert self_result['first_observed_api_difference'] is None
        rows.append({'step': step, 'status': result['status'], 'alignment': result['alignment'],
                     'first_exact': result['first_exact_api_difference'],
                     'first_above': result['first_observed_api_difference'],
                     'accepted': accept(result)['accepted'],
                     'unobservable_fields': result['unobservable_fields'],
                     'baseline_self_comparison_exact': True})
    active = False
    value = {'schema': 'fault28.visible.statistics_only_read_audit.v1',
             'status': 'STATISTICS_ONLY_FILE_ACCESS_AUDIT_COMPLETED',
             'actual_steps': [0, 1, 2], 'rtol': 1e-5, 'atol': 1e-8,
             'read_allowlist_types': ['dump.json', 'stack.json', 'construct.json', 'config.json', 'collector_provenance.json'],
             'unique_files_read': len(set(accesses)), 'read_operations': len(accesses),
             'read_paths': sorted(set(accesses)), 'forbidden_reads': 0,
             'model_injector_runtime_log_or_tensor_read': False, 'results': rows,
             'source_sha256': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                               for name in ('compare_statistics.py', 'accept_statistics.py', 'statistics_values.py')},
             'auditor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: value[key] for key in ('status', 'unique_files_read', 'read_operations', 'forbidden_reads')}))
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--fault', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.baseline, args.fault, args.source, args.output)
