"""Emit allowlisted receipts; raw statistics and storage identities stay on 124.

This summarizes execution/recovery only. It never chooses a statistical first
difference. Run the independent comparator separately, before correlation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path):
    return dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size)


def summarize_case(case):
    result = case / 'result.json'
    row = dict(case=str(case), raw_statistics_published=False)
    log = case / 'process.log'
    if log.is_file():
        row['process_log'] = artifact(log)
    if not result.is_file():
        row['status'] = 'NO_RESULT_JSON'
        return row
    raw = json.loads(result.read_text())
    row['result'] = artifact(result)
    for key in ('fault', 'arm', 'status', 'scope', 'pretrained_model', 'worker_extension',
                'actual_device', 'torch', 'torch_npu', 'alias_check', 'off_exact',
                'repeat_stable', 'input_unchanged', 'changed', 'finite', 'max_abs',
                'steps', 'exact_half', 'error', 'actual_fault_injection', 'tp',
                'memory_utilization', 'runtime_sha256'):
        if key in raw:
            row[key] = raw[key]
    if 'session' in raw:
        session = raw['session']
        events = session.get('events', [])
        row['routing'] = dict(candidate_modules=session.get('candidate_modules'),
            statuses=dict(Counter(e.get('status', 'unknown') for e in events)),
            per_module=dict(Counter(e.get('path', 'unknown') for e in events)),
            forwards=sorted({e['forward'] for e in events if 'forward' in e}),
            snapshot_scope='inside session, before context exit; off_exact measured after exit')
        if events and 'aliases_original_base' in events[0]:
            row['alias_observations'] = [
                {k: e.get(k) for k in ('forward', 'eligible', 'actual_format',
                    'initial_values_correct', 'aliases_original_base', 'writes',
                    'consumers', 'observation_errors')} for e in events]
    if 'workers' in raw:
        row['worker_count'] = len(raw['workers'])
        row['rank_ids'] = [w.get('rank') for w in raw['workers']]
        row['normal_generation_exact_after'] = raw.get('baseline') == raw.get('after') if 'after' in raw else None
        row['observed_generation_exact'] = (all(x == raw['baseline'] for x in raw['observed_requests'])
            if 'observed_requests' in raw else None)
        for key in ('prepared', 'register', 'other_provider', 'recovery', 'detach'):
            if key in raw:
                row[key] = {k: raw[key][k] for k in ('status', 'reason', 'provider_code',
                    'recovery_required', 'action') if k in raw[key]}
        if 'discovery' in raw:
            row['discovery'] = raw['discovery']
    capture = case / 'capture'
    if capture.is_dir():
        row['capture_artifacts'] = [artifact(p) for p in sorted(capture.rglob('*')) if p.is_file()]
        configs = [p for p in capture.rglob('*.json') if 'config' in p.name]
        row['capture_configs'] = [{**artifact(p), 'configuration': json.loads(p.read_text())} for p in configs]
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--attempts', nargs='+', required=True)
    args = parser.parse_args()
    rows = []
    for attempt in args.attempts:
        directory = args.root / attempt
        if not directory.is_dir():
            raise RuntimeError('Missing attempt: ' + attempt)
        rows.extend(summarize_case(p) for p in sorted(directory.iterdir()) if p.is_dir())
    output = dict(schema='demoncat.overnight.public_receipts.v1',
                  summary_tool_sha256=digest(Path(__file__)), cases=rows,
                  independent_first_difference_selected=False,
                  new_raw_statistics_published=False)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(cases=len(rows), output=str(args.out))))


if __name__ == '__main__':
    main()
