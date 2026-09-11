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
from ascend_fi.inspect.selection import parse_ranks, target_view, render_target_view


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
    parser.add_argument('--api-key-file', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--roles', nargs='+')
    parser.add_argument('--selection-mode', choices=['manual', 'all_bindable'])
    parser.add_argument('--layers', type=int, nargs='+')
    parser.add_argument('--modules', nargs='+', help='Exact canonical names; AND with roles/layers')
    parser.add_argument('--rank', '--ranks', dest='rank', nargs='+', default=['0'], help='Global rank list, or explicit all')
    parser.add_argument('--reports', type=Path, help='Offline report replay only; never contact service')
    parser.add_argument('--reference-arm', type=Path)
    parser.add_argument('--multiplier', type=float, default=20)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--max-prompt-tokens', type=int)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    reports = (json.loads(args.reports.read_text(encoding='utf-8')) if args.reports else
               collect_reports(InspectClient(args.base_url, api_key_file=args.api_key_file)))
    for report in reports:
        check = dict(report)
        if check.pop('report_sha256', None) != digest(check):
            raise ValueError('TARGET_REPORT_INTEGRITY_MISMATCH')
    with (args.out / 'model_target_reports.json').open('x', encoding='utf-8') as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    plan = None
    if args.roles or args.modules or args.selection_mode:
        if not args.reference_arm or not args.max_prompt_tokens:
            parser.error('plan export needs --reference-arm and --max-prompt-tokens')
        plan = build_target_plan(reports, roles=args.roles, rank_filter=parse_ranks(args.rank),
            selection_mode=args.selection_mode or 'manual', layers=args.layers, module_names=args.modules,
            scale_multiplier=args.multiplier, reference_arm=json.loads(args.reference_arm.read_text()),
            max_tokens=args.max_tokens, max_prompt_tokens=args.max_prompt_tokens)
        with (args.out / 'target_plan.json').open('x', encoding='utf-8') as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
    view = target_view(reports, plan=plan)
    view['observation_source'] = 'OFFLINE_REPLAY_CURRENT_EPOCH_UNVERIFIED' if args.reports else 'LIVE_READONLY'
    with (args.out / 'target_preview.json').open('x', encoding='utf-8') as f:
        json.dump(view, f, ensure_ascii=False, indent=2, allow_nan=False)
    with (args.out / 'target_preview.md').open('x', encoding='utf-8') as f:
        f.write(render_target_view(view))
    print(str(args.out))


if __name__ == '__main__':
    main()
