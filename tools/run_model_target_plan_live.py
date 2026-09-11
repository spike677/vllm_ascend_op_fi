"""Run an explicit plan through the existing leased lab owner (no launch/lock)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ascend_fi.inspect.client import InspectClient
from ascend_fi.inspect.qwen_r1_adapter import create_owner
from ascend_fi.inspect.plan_executor import TargetPlanExecutor
from ascend_fi.inspect.plan_builder import build_target_plan
from ascend_fi.inspect.dataset_effect import evaluate_dataset
from ascend_fi.evaluation.distortion.report import write_report
from ascend_fi.evaluation.distortion.contracts import content_hash
from ascend_fi.contracts import EvaluationResult
from inspect_model_targets import collect_reports


def save(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def observe_and_build(client, owner, probe, out, *, roles, multiplier, max_prompt_tokens):
    """Actual structural proposals -> actual request -> automatic input binding."""
    proposals = collect_reports(client)
    save(out / 'structural_reports.json', proposals)
    active = {}
    try:
        for r in proposals:
            names = [t['module_name'] for t in r['targets'] if set(t['roles']) & set(roles)]
            if not names:
                raise ValueError('SELECTED_ROLES_NOT_FOUND; no alternative target fallback')
            epoch = r['descriptor']['service_epoch']
            values = client.call('observe_start', target_service_epochs=[epoch], service_epoch=epoch,
                module_names=names, request_window='auto-target-binding', exclusive=True,
                ttl_seconds=300, max_events=min(2048, 2 * len(names) + 1),
                max_bytes=2097152, samples_per_module=2)
            active[epoch] = next(v['result']['session'] for v in values if not v.get('skipped'))
        # Existing owner enforces the lease; no invented discovery prompts.
        save(out / 'observation_response.json', owner.request(probe, 'observation', 0))
    finally:
        for epoch, session in active.items():
            client.call('observe_stop', target_service_epochs=[epoch], session=session)
    reports = collect_reports(client)
    save(out / 'model_target_reports.json', reports)
    return build_target_plan(reports, roles=roles, rank_filter=[0],
        scale_multiplier=multiplier, reference_arm=owner.manifest['r1_arm_config'],
        max_tokens=probe['generation_config']['max_tokens'], max_prompt_tokens=max_prompt_tokens)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--deployment', type=Path, help='Explicit private-service deployment; no legacy lab paths')
    p.add_argument('--discover-only', action='store_true', help='Observe real clean input and export disabled plan; never install/arm')
    p.add_argument('--plan', type=Path)
    p.add_argument('--roles', nargs='+')
    p.add_argument('--multiplier', type=float, default=20)
    p.add_argument('--max-prompt-tokens', type=int)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--identity-probe', required=True)
    p.add_argument('--probe', required=True, nargs='+')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    suite = json.loads(args.suite.read_text())
    if suite.get('hash') != content_hash(suite):
        raise ValueError('FROZEN_SUITE_HASH_MISMATCH')
    plan = json.loads(args.plan.read_text()) if args.plan else None
    if not plan and (not args.roles or not args.max_prompt_tokens):
        p.error('automatic plan needs --roles and --max-prompt-tokens')
    probes = {r['probe_id']: r for r in suite['probes']}
    if args.identity_probe not in probes or any(k not in probes for k in args.probe):
        raise ValueError('USE_FROZEN_SUITE_PROBE_IDS')
    args.out.mkdir(parents=True, exist_ok=False)
    if args.deployment:
        from private_service_owner import PrivateServiceOwner, InspectTransport
        owner = PrivateServiceOwner(args.session, suite, json.loads(args.deployment.read_text()))
        client = InspectClient(owner.base, transport=InspectTransport(owner))
    else:
        owner = create_owner(args.session, suite)
        client = InspectClient('http://127.0.0.1:18031')
    owner.rawroot = args.out / 'raw'
    def current():
        return [r['result']['descriptor'] for r in client.call('describe_model', limit=1)]
    leased = False
    try:
        preflight = owner.preflight()
        leased = True
        if not owner.access_token:
            raise ValueError('PRIVATE_SINGLE_REQUEST_SERVICE_REQUIRED')
        import urllib.request, urllib.error
        try:
            urllib.request.urlopen(owner.base + '/v1/models', timeout=15)
        except urllib.error.HTTPError as error:
            if error.code != 401:
                raise
        else:
            raise ValueError('SERVICE_PRIVATE_AUTH_NOT_PROVEN')
        owner.plan_discovery_window = True
        save(args.out / 'preflight.json', preflight)
        if plan is None:
            plan = observe_and_build(client, owner, probes[args.identity_probe], args.out,
                roles=args.roles, multiplier=args.multiplier, max_prompt_tokens=args.max_prompt_tokens)
        for key in [args.identity_probe, *args.probe]:
            if probes[key]['generation_config']['max_tokens'] != plan['request_window']['max_tokens']:
                raise ValueError('PLAN_GENERATION_BUDGET_MISMATCH')
        if args.discover_only:
            save(args.out / 'target_plan.json', plan)
            print(json.dumps({'event': 'disabled_plan_exported', 'targets': len(plan['selected_modules']),
                'roles': plan['roles'], 'path': str(args.out / 'target_plan.json')}), flush=True)
            return
        runner = TargetPlanExecutor(owner, plan, current)
        save(args.out / 'target_plan.json', plan)
        save(args.out / 'identity.json', runner.validate_identity(probes[args.identity_probe]))
        before = owner.clean_query()
        pairs, runs = {}, {}
        for key in args.probe:
            pair = runner.run_pair(probes[key])
            save(args.out / ('pair.json' if len(args.probe) == 1 else key + '_pair.json'), pair)
            runs[key], pairs[key] = pair, pair['pair']
            print(json.dumps({'event': 'pair_complete', 'probe': key, 'status': pair['status'],
                'recovery_exact': pair['recovery_exact']}), flush=True)
            if pair['status'] != 'PASS':
                raise RuntimeError('PAIR_GATE_FAILED_STOP_SUBSEQUENT_REQUESTS:' + key)
        cycle = {'cycle_id': 'cycle_01', 'before_baseline': before, 'service_healthy_before': True,
                 'pairs': pairs, 'after_remove': owner.clean_query(),
                 'service_healthy': owner.service_healthy()}
        subset = {**suite, 'probes': [probes[k] for k in args.probe], 'probe_count': len(args.probe)}
        subset['hash'] = content_hash(subset)
        result = evaluate_dataset(subset, runner.config, [cycle], evaluation_id=plan['plan_sha256'][:16])
        data = result.to_dict()
        data['model_targeting'] = {'plan_sha256': plan['plan_sha256'], 'window': pair['window'],
                                  'recovery_exact': all(r['recovery_exact'] for r in runs.values()),
                                  'per_probe_windows': {k: r['window'] for k, r in runs.items()}}
        if pair['status'] != 'PASS' or pair['window']['full_request_window'] != 'PASS':
            # Keep the existing EvaluationStatus enum; PARTIAL is a profile stage,
            # not a new competing result schema/status enumeration.
            data['model_targeting']['status'] = 'PARTIAL'
            data.update(status='UNKNOWN', reason='MODEL_AWARE_EXECUTION_OR_FULL_WINDOW_VALIDATION_INCOMPLETE')
        write_report(args.out / 'effect', EvaluationResult.from_dict(data), {'complete_cycles': 1})
        save(args.out / 'replay_input.json', {'suite': subset, 'config': runner.config.to_dict(), 'cycles': [cycle]})
    finally:
        if leased:
            owner.disarm_and_remove()
            save(args.out / 'final_state.json', {'query': owner.clean_query(), 'healthy': owner.service_healthy()})
    # The service supervisor owns eventual service restoration and lease release.


if __name__ == '__main__':
    main()
