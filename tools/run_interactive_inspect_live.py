"""Small real-service acceptance, delegating all mutation to existing R1 lifecycle.

Requires an already acquired exclusive session and the combined startup extension.
Never starts a service, acquires a lock, changes R1 strength, or pads a hit budget.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ascend_fi.inspect.client import InspectClient
from ascend_fi.inspect.qwen_r1_adapter import create_owner, factory
from ascend_fi.inspect.plans import validate_plan
from ascend_fi.evaluation.distortion.contracts import content_hash
from ascend_fi.evaluation.distortion.mechanism import cleanup_gate


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'saved': str(path)}), flush=True)


def selected(rows):
    return [r['result'] for r in rows if not r.get('skipped')]


def owned_lease(session):
    path = Path('/home/haoran/demoncat_overnight_20260907/coordination/NPU_LOCK/owner.json')
    state = json.loads(path.read_text())
    return state.get('owner') == 'EVALUATION' and state.get('result_dir') == str(session) and state.get('status') == 'ACTIVE'


def paged(client, action, epoch, **kwargs):
    result, offset = [], 0
    while True:
        p = selected(client.call(action, target_service_epochs=[epoch], offset=offset, limit=100, **kwargs))
        assert len(p) == 1
        result.extend(p[0]['items'])
        offset = p[0]['next_offset']
        if offset is None:
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--stage', choices=['inspect-identity', 'smoke'], required=True)
    parser.add_argument('--prompt', help='User-provided normal prompt; required for lossy smoke')
    args = parser.parse_args()
    if args.stage == 'smoke' and not args.prompt:
        parser.error('lossy smoke requires the user prompt, not a benchmark substitute')
    output = args.session / 'acceptance'
    output.mkdir(exist_ok=True)
    frozen = json.loads(Path('/home/haoran/distlive_20260909_v1/normal_v2/suite.json').read_text())
    suite = {'request_protocol': frozen['request_protocol']}
    probe = {'probe_id': 'inspect_normal', 'category': 'interactive_normal',
             'prompt': args.prompt or '请只回答结果：12乘7等于多少？',
             'generation_config': {'temperature': 0, 'seed': 42, 'max_tokens': 128}}
    owner = create_owner(args.session, suite)
    client = InspectClient('http://127.0.0.1:18031')
    final = {'stage': args.stage}
    preflight_passed = False
    try:
        preflight = owner.preflight()
        preflight_passed = True
        save(output / (args.stage + '_preflight.json'), preflight)
        status = selected(client.call('status'))
        assert len(status) == 2
        assert {s['ranks']['global']['value'] for s in status} == {0, 1}, status
        assert all(s['reference_backend_available'] for s in status)
        epochs = [s['service_epoch'] for s in status]
        if args.stage == 'smoke':
            gate = json.loads((output / 'identity_result.json').read_text())
            assert gate['lossless_passed'] and gate['restore_exact']
            plan = json.loads((output / 'draft_plan.json').read_text())
            current = [s for s in status if s['service_epoch'] == plan['service_epoch']]
            assert len(current) == 1 and current[0]['runtime_fingerprint'] == plan['runtime_fingerprint']
            assert all(not s['active'] and not s['installed_handles'] for s in status)
            save(output / 'smoke_frozen_request.json', {'probe': probe, 'suite': suite,
                 'frozen_arm': owner.arm, 'thresholds': frozen['thresholds'], 'complete_cycles': 1})
            baseline = owner.request(probe, 'smoke_baseline', 1)
            before = owner.clean_query()
            try:
                config = owner.install()
                assert set(config.actual_module_names) == set(plan['selected_modules'])
                fault = owner.fault_request(probe, config, 1)
            finally:
                owner.disarm_and_remove()
            clean = cleanup_gate(owner.clean_query(), owner.service_healthy(), 2)
            assert clean['status'] == 'PASS'
            recovery = owner.request(probe, 'smoke_recovery', 1)
            from ascend_fi.evaluation.distortion.evaluator import evaluate_cycles
            from ascend_fi.evaluation.distortion.report import write_report
            # Reuse current evaluator/thresholds; a one-request profile is not repeatability.
            effect_probe = {**probe, 'expected': {},
                           'applicable_detectors': ['length', 'garbage', 'crash']}
            effect_suite = {**frozen, 'version': 'INTERACTIVE_INSPECT_SINGLE_REQUEST_V1',
                            'probes': [effect_probe], 'probe_count': 1}
            effect_suite['hash'] = content_hash(effect_suite)
            cycle = {'cycle_id': 'cycle_01', 'complete': True, 'status': 'COMPLETE',
                     'before_baseline': before, 'service_healthy_before': True,
                     'after_remove': owner.clean_query(), 'service_healthy': owner.service_healthy(),
                     'pairs': {probe['probe_id']: {'baseline': baseline, 'fault': fault, 'recovery': recovery}}}
            # Save raw even if the profile builder reports a schema/compatibility gap.
            save(output / 'smoke_raw.json', {'suite': effect_suite, 'config': config.to_dict(), 'cycles': [cycle]})
            report = evaluate_cycles(effect_suite, config, [cycle], evaluation_id='inspect-single-request')
            write_report(output / 'effect', report, {'kind': 'LIVE_SINGLE_REQUEST', 'repeatability_proven': False})
            final.update(mechanism=fault.get('mechanism_summary'),
                         response_equal=baseline.get('response_text') == fault.get('response_text'),
                         recovery_exact=baseline.get('response_text') == recovery.get('response_text'))
            return
        save(output / 'initial_status.json', status)
        modules = {}
        for s in status:
            modules[s['service_epoch']] = paged(client, 'list_modules', s['service_epoch'])
        save(output / 'model_modules.json', modules)
        rank0 = next(s for s in status if s['ranks']['global']['value'] == 0)
        rows = modules[rank0['service_epoch']]
        layers = sorted({r['layer_index'] for r in rows if r['logical_role'] == 'MLP_DOWN'})
        assert layers
        choices = {layers[0], layers[len(layers)//2], layers[-1]}
        roles = {'ATTN_QKV_FUSED', 'ATTN_OUTPUT', 'MLP_GATE_UP_FUSED', 'MLP_DOWN'}
        representative = [r['module_name'] for r in rows if r['logical_role'] in roles and r['layer_index'] in choices]
        down = [r['module_name'] for r in rows if r['logical_role'] == 'MLP_DOWN']
        descriptions = []
        for epoch in epochs:
            for name in representative:
                descriptions.extend(selected(client.call('describe_target', target_service_epochs=[epoch], module_name=name)))
        save(output / 'quantization_inventory.json', descriptions)
        assert all(d['quantization']['resolution'] == 'AUDITED_IMPLEMENTATION' for d in descriptions), 'current source contract unresolved'
        # TTL cleanup without any model request.
        client.call('observe_start', target_service_epochs=epochs, module_names=representative[:1],
                    request_window='ttl_no_request', exclusive=True, ttl_seconds=3, max_events=8, samples_per_module=2)
        time.sleep(4)
        ttl = selected(client.call('status'))
        save(output / 'ttl_cleanup.json', ttl)
        assert all(s['stop_reason'] == 'TTL_EXPIRED' and not s['installed_handles'] and not s['active'] for s in ttl)
        baseline = owner.request(probe, 'inspect_clean', 0)
        transparent = []
        for name, targets in [('representative', representative), ('full_r1_targets', down)]:
            active = selected(client.call('observe_start', target_service_epochs=epochs, module_names=targets,
                              request_window=name, exclusive=True, ttl_seconds=180,
                              max_events=512, max_bytes=2097152, samples_per_module=2))
            sessions = {s['service_epoch']: s['session'] for s in active}
            try:
                response = owner.request(probe, 'observe_' + name, 0)
                events = {epoch: paged(client, 'observe_poll', epoch, session=sessions[epoch]) for epoch in epochs}
            finally:
                stopped = selected(client.call('observe_stop', target_service_epochs=epochs, session_by_epoch=sessions))
            save(output / (name + '_events.json'), events)
            save(output / (name + '_stop.json'), stopped)
            assert all(not s['installed_handles'] and not s['active'] for s in stopped)
            assert all(set(e['module_name'] for e in ev) == set(targets) for ev in events.values())
            assert response['http_status'] == 200
            transparent.append(response['response_text'] == baseline['response_text'])
        save(output / 'observer_transparency.json', {'both_equal': all(transparent), 'checks': transparent})
        assert all(transparent), 'observer transparency failed'
        candidates = paged(client, 'list_candidates', rank0['service_epoch'], filters={'logical_role': 'MLP_DOWN'})
        save(output / 'injection_candidates.json', candidates)
        assert len(candidates) == len(down)
        plan = selected(client.call('export_plan', target_service_epochs=[rank0['service_epoch']],
                         candidate_ids=[c['candidate_id'] for c in candidates],
                         runtime_fingerprint=rank0['runtime_fingerprint'], frozen_arm=owner.arm))[0]
        save(output / 'draft_plan.json', plan)
        prior_config = json.loads(Path('/home/haoran/distlive_20260909_v1/normal_v2/fault_configuration.json').read_text())
        assert set(prior_config['actual_module_names']) == set(plan['selected_modules']), 'not equivalent to frozen R1'
        validation_args = {'session_dir': str(args.session), 'suite': suite, 'probe': probe,
                           'fault_configuration': prior_config, 'plan': plan,
                           'service_epoch': plan['service_epoch'], 'runtime_fingerprint': plan['runtime_fingerprint']}
        save(output / 'validate_args.json', validation_args)
        executor = factory(validation_args)
        result = validate_plan(plan, epoch=plan['service_epoch'], fingerprint=plan['runtime_fingerprint'], executor=executor)
        save(output / 'identity_result.json', result.to_dict())
        assert result.lossless_passed and result.restore_exact
        final.update(inspect=True, observed=True, observer_clean=True, binding=True, identity=True,
                     quantization_audited=sum(d['quantization']['resolution'] == 'AUDITED_IMPLEMENTATION' for d in descriptions),
                     quantization_described=len(descriptions), qmm_live='NOT_SUPPORTED', module_to_native='UNKNOWN')
    except BaseException as e:
        final.update(error=repr(e), traceback=traceback.format_exc())
        print(final['traceback'], flush=True)
        raise
    finally:
        try:
            if preflight_passed and owned_lease(args.session):
                final['observer_stop'] = client.call('observe_stop')
                owner.disarm_and_remove()
                final['clean'] = cleanup_gate(owner.clean_query(), owner.service_healthy(), 2)
            else:
                final['cleanup_skipped'] = 'NO_VERIFIED_OWN_LEASE; never remove another session hooks'
        except BaseException as e:
            final['cleanup_error'] = repr(e)
        save(output / (args.stage + '_final.json'), final)


if __name__ == '__main__':
    main()
