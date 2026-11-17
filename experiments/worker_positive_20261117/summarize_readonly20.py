"""Compare observed target identities across two real read-only forwards."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report_path, result_path = args.case/'observer-rank0.json', args.case/'result.json'
    report, result = json.loads(report_path.read_text()), json.loads(result_path.read_text())
    by_path = defaultdict(lambda: defaultdict(list))
    events = [e for e in report['report']['events'] if e.get('native_facts')]
    for event in events:
        by_path[event['path']][event['forward']].append(event)
    identity_fields = ('native_signature', 'cache_identity', 'cache_lifecycle',
                       'build_profile_sha256', 'layout', 'slots')
    rows = []
    for path, forwards in sorted(by_path.items()):
        a, b = forwards.get(0, []), forwards.get(1, [])
        identity = lambda event: {key: event['native_facts'].get(key) for key in identity_fields}
        rows.append(dict(path=path, observed_calls=[len(a), len(b)],
            observed_identity_stable=len(a) == len(b) == 1 and identity(a[0]) == identity(b[0]),
            canonical_roles=(a or b)[0]['canonical_roles'],
            observed_layout=(a or b)[0]['native_facts']['layout']))
    out = dict(schema='demoncat.worker_positive.readonly20.v1', status=result['status'],
        source=str(args.case), source_hashes={'result.json': sha(result_path),
                                            'observer-rank0.json': sha(report_path)},
        operator=result['operator'], actual_fault_injection=False,
        observed_forwards=2, observed_events=len(events), observed_modules=len(rows),
        actual_original_calls=sum(e['actual_original_calls'] for e in events),
        all_observed_identities_stable=bool(rows) and all(r['observed_identity_stable'] for r in rows),
        native_idle=all(row['native_idle'] for row in result.get('finish', [])),
        generated_tokens=dict(before=result['before'], observed=result['observed'], after=result['after']),
        rows=rows, limitations=[
            'Only the observed current-call paths; no all_eligible denominator or qualification',
            'No lossless/fault/OFF substitution trial for QMM in the common serving interface',
            'Token equality in this read-only probe does not prove full logits equality',
            'The pretrained output head is FLOAT; its natural invocation cannot be relabeled QMM'])
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
