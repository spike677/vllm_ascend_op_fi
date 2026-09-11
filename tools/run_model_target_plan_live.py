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
from ascend_fi.inspect.selection import scoped_targets, parse_ranks, target_view, render_target_view
from ascend_fi.inspect.dataset_effect import evaluate_dataset
from ascend_fi.evaluation.distortion.report import write_report
from ascend_fi.evaluation.distortion.contracts import content_hash
from ascend_fi.contracts import EvaluationResult
from inspect_model_targets import collect_reports


def save(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def observe_and_build(client, owner, probe, out, *, roles, multiplier, max_prompt_tokens,
                      selection_mode='manual', layers=None, module_names=None, ranks=None):
    """Actual structural proposals -> actual request -> automatic input binding."""
    proposals = collect_reports(client)
    save(out / 'structural_reports.json', proposals)
    active = {}
    try:
        for r in proposals:
            names = sorted({t['module_name'] for t in scoped_targets(r, roles=roles,
                            layers=layers, module_names=module_names)})
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
    plan = build_target_plan(reports, roles=roles, rank_filter=[0] if ranks is None else ranks,
        selection_mode=selection_mode, layers=layers, module_names=module_names,
        scale_multiplier=multiplier, reference_arm=owner.manifest['r1_arm_config'],
        max_tokens=probe['generation_config']['max_tokens'], max_prompt_tokens=max_prompt_tokens)
    save(out / 'target_preview.json', target_view(reports, plan=plan))
    with (out / 'target_preview.md').open('x', encoding='utf-8') as f:
        f.write(render_target_view(target_view(reports, plan=plan)))
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--deployment', type=Path, help='Explicit private-service deployment; no legacy lab paths')
    p.add_argument('--discover-only', action='store_true', help='Observe real clean input and export disabled plan; never install/arm')
    p.add_argument('--execution-policy', choices=['same-process-bfr', 'stop-instance'], default='same-process-bfr',
                   help='stop-instance is invoked by the owning host wrapper; one lossy call, no recovery inference')
    p.add_argument('--plan', type=Path)
    p.add_argument('--roles', nargs='+')
    p.add_argument('--selection-mode', choices=['manual', 'all_bindable'], default='manual')
    p.add_argument('--layers', type=int, nargs='+')
    p.add_argument('--modules', nargs='+')
    p.add_argument('--ranks', nargs='+', default=['0'], help='Global ranks or all participating pure-TP ranks')
    p.add_argument('--multiplier', type=float, default=20)
    p.add_argument('--max-prompt-tokens', type=int)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--identity-probe', required=True)
    p.add_argument('--probe', required=True, nargs='+')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.execution_policy == 'stop-instance' and (not args.deployment or len(args.probe) != 1 or args.discover_only):
        p.error('stop-instance requires private deployment and exactly one probe; not discover-only')
    if args.execution_policy == 'stop-instance':
        contract = json.loads((args.session / 'single_lossy_host_contract.json').read_text())
        deployment = json.loads(args.deployment.read_text())
        if (contract.get('output') != str(args.out.resolve())
                or contract.get('lease_id') != deployment['lease_id']):
            raise ValueError('OWNING_HOST_STOP_CONTRACT_REQUIRED; use run_private_demo_once.py')
    suite = json.loads(args.suite.read_text())
    if suite.get('hash') != content_hash(suite):
        raise ValueError('FROZEN_SUITE_HASH_MISMATCH')
    plan = json.loads(args.plan.read_text()) if args.plan else None
    if not plan and ((args.selection_mode == 'manual' and not (args.roles or args.modules)) or not args.max_prompt_tokens):
        p.error('manual plan needs --roles/--modules; all plans need --max-prompt-tokens')
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
        if args.execution_policy == 'stop-instance':
            owner.reserve_single_instance()
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
                roles=args.roles, multiplier=args.multiplier, max_prompt_tokens=args.max_prompt_tokens,
                selection_mode=args.selection_mode, layers=args.layers, module_names=args.modules,
                ranks=parse_ranks(args.ranks))
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
        preview = {'arm': False, 'plan_sha256': plan['plan_sha256'], 'selection': plan.get('selection'),
            'per_worker_target_count': {str(w['rank']): len(w['targets']) if w['targeted'] else 0 for w in plan['workers']},
            'worker_epochs': [w['service_epoch'] for w in plan['workers']],
            'arm_config': plan['prospective_arm'], 'budget': plan['request_window'],
            'output_directory': str(args.out), 'execution_policy': args.execution_policy}
        save(args.out / 'execution_preview.json', preview)
        print(json.dumps({'event': 'pre_identity_plan_preview', **preview}, ensure_ascii=False), flush=True)
        save(args.out / 'identity.json', runner.validate_identity(probes[args.identity_probe]))
        if args.execution_policy == 'stop-instance':
            from ascend_fi.inspect.single_effect import evaluate_single
            once = runner.run_once(probes[args.probe[0]])
            save(args.out / 'single_run.json', once)
            result = evaluate_single(suite, runner.config, plan, once, probe_id=args.probe[0])
            write_report(args.out / 'effect', result, {'complete_cycles': 0, 'source': str(args.out / 'single_run.json')})
            if once['status'] == 'PARTIAL':
                raise RuntimeError('SINGLE_RUN_PARTIAL; stop owned model instance')
            return
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
            if args.execution_policy == 'stop-instance':
                final = {'policy': 'STOP_INSTANCE_REQUIRED', 'hot_recovery': 'NOT_MEASURED'}
                for key, action in [('cleanup_attempt', owner.disarm_and_remove),
                                    ('query', owner.clean_query), ('healthy', owner.service_healthy)]:
                    try:
                        final[key] = action()
                    except Exception as exc:
                        final[key] = {'error': repr(exc)}
                save(args.out / 'final_state.json', final)
            else:
                owner.disarm_and_remove()
                save(args.out / 'final_state.json', {'query': owner.clean_query(), 'healthy': owner.service_healthy()})
    # The service supervisor owns eventual service restoration and lease release.


if __name__ == '__main__':
    main()
