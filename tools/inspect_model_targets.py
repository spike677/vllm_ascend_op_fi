"""Read-only model-aware discovery / plan export. Never install, arm or launch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ascend_fi.inspect.client import InspectClient
from ascend_fi.inspect.model_descriptor import digest
from ascend_fi.inspect.plan_builder import build_target_plan


def collect_reports(client):
    status = client.call('status')
    reports = []
    for s in status:
        epoch, offset, report, targets = s['service_epoch'], 0, None, []
        while True:
            rows = client.call('auto_targets', target_service_epochs=[epoch], offset=offset, limit=100)
            row = next(r['result'] for r in rows if not r.get('skipped'))
            if report and row['report_sha256'] != report['report_sha256']:
                raise ValueError('REPORT_CHANGED_DURING_PAGINATION')
            report = {k: v for k, v in row.items() if k not in ('items', 'total', 'next_offset')}
            targets.extend(row['items'])
            if row['next_offset'] is None:
                if len(targets) != row['total']:
                    raise ValueError('PAGINATION_INCOMPLETE')
                break
            offset = row['next_offset']
        report['targets'] = targets
        check = dict(report)
        if check.pop('report_sha256') != digest(check):
            raise ValueError('REPORT_CHECKSUM_MISMATCH')
        reports.append(report)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:18031')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--roles', nargs='+')
    parser.add_argument('--rank', type=int, nargs='+', default=[0])
    parser.add_argument('--reference-arm', type=Path)
    parser.add_argument('--multiplier', type=float, default=20)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--max-prompt-tokens', type=int)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    reports = collect_reports(InspectClient(args.base_url))
    with (args.out / 'model_target_reports.json').open('x', encoding='utf-8') as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    if args.roles:
        if not args.reference_arm or not args.max_prompt_tokens:
            parser.error('plan export needs --reference-arm and --max-prompt-tokens')
        plan = build_target_plan(reports, roles=args.roles, rank_filter=args.rank,
            scale_multiplier=args.multiplier, reference_arm=json.loads(args.reference_arm.read_text()),
            max_tokens=args.max_tokens, max_prompt_tokens=args.max_prompt_tokens)
        with (args.out / 'target_plan.json').open('x', encoding='utf-8') as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
    print(str(args.out))


if __name__ == '__main__':
    main()
