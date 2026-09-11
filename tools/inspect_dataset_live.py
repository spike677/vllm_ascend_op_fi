"""Dataset stage for run_interactive_inspect_live; reuses the existing R1 owner."""
import copy
import hashlib
import json
from pathlib import Path
import time
import traceback

from ascend_fi.inspect.dataset import SCOPE, QUESTION_IDS, as_probe, load_samples
from ascend_fi.inspect.dataset_effect import evaluate_dataset
from ascend_fi.inspect.client import InspectClient
from ascend_fi.inspect.qwen_r1_adapter import create_owner, factory
from ascend_fi.inspect.plans import validate_plan
from ascend_fi.evaluation.distortion.contracts import content_hash
from ascend_fi.evaluation.distortion.evaluator import SCORER_VERSION
from ascend_fi.evaluation.distortion.mechanism import cleanup_gate
from ascend_fi.evaluation.distortion.report import write_report
from run_interactive_inspect_live import save, selected, paged, owned_lease


def run_dataset(args):
    manifest, samples = load_samples(args.dataset_file, args.dataset_manifest)
    session, output = args.session, args.session / 'dataset_acceptance'
    output.mkdir(exist_ok=False)
    save(output / 'dataset_manifest.json', manifest)
    with (output / 'selected_samples.jsonl').open('x', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    frozen = json.loads(Path('/home/haoran/distlive_20260909_v1/normal_v2/suite.json').read_text())
    suite = copy.deepcopy(frozen)
    suite.update(version=SCOPE, dataset=manifest, probe_count=len(samples))
    # Keep the already validated chat template, explicitly declare payload settings.
    suite['request_protocol'].update(endpoint='/v1/chat/completions', chat_template_kwargs={'enable_thinking': False},
                                     logprobs=True, top_logprobs=1)
    owner = create_owner(session, suite)
    owner.rawroot = output / 'raw'
    owner.attempt = 'dataset_v1'
    client = InspectClient('http://127.0.0.1:18031')
    final, preflight_passed = {}, False
    try:
        preflight = owner.preflight()
        preflight_passed = True
        save(output / 'preflight.json', preflight)
        save(output / 'frozen_r1_manifest.json', owner.manifest)
        status = selected(client.call('status'))
        assert len(status) == 2 and {s['ranks']['global']['value'] for s in status} == {0, 1}
        assert all(s['reference_backend_available'] and not s['active'] and not s['installed_handles'] for s in status)
        save(output / 'initial_status.json', status)
        # Hash the exact same V2 scorer before observing outputs; never retune rules.
        import ascend_fi.evaluation.distortion.evaluator as evaluator
        scorer_files = {n: hashlib.sha256(Path(evaluator.__file__).with_name(n).read_bytes()).hexdigest()
                        for n in ('evaluator.py', 'detectors.py', 'contracts.py', 'mechanism.py')}
        save(output / 'pre_run_contract.json', {'scorer': SCORER_VERSION, 'files': scorer_files,
             'scorer_sha256': content_hash(scorer_files), 'thresholds': frozen['thresholds'],
             'budget_calibration_sequence': [1024, 1536, 2048], 'fault_strength_changes': 'FORBIDDEN',
             'sample_order': [149] + [i for i in QUESTION_IDS if i != 149],
             'complete_cycles': 1, 'robust_16_probe_gate': 'NOT_APPLICABLE',
             'task_validator': 'NONE; SEMANTIC_REVIEW_REQUIRED', 'recovery_text_tolerance': 'EXACT'})
        # Tokenize the exact chat template. This is not a generated padding request.
        lengths = {}
        for sample in samples:
            payload = {'model': owner.model, 'messages': [{'role': 'user', 'content': sample['prompt']}],
                       'chat_template_kwargs': {'enable_thinking': False}, 'add_generation_prompt': True}
            code, raw, error = owner.client._post('http://127.0.0.1:18031/tokenize', payload, 60)
            save(output / f"tokenize_{sample['question_id']}.json", {'payload': payload, 'http_status': code, 'raw': raw, 'error': error})
            assert code == 200 and type(raw.get('count')) is int
            lengths[sample['question_id']] = raw['count']
        # Every calibration is Fault OFF, before ANY lossy request. Globally refreeze.
        for attempt, budget in enumerate((1024, 1536, 2048), 1):
            assert max(lengths.values()) + budget <= 4096
            probes = [as_probe(s, budget) for s in samples]
            calibration = [owner.request(p, 'calibration', attempt) for p in probes]
            save(output / f'calibration_{attempt}.json', {'max_tokens': budget, 'rows': calibration})
            assert all(r['http_status'] == 200 for r in calibration)
            if all(r['finish_reason'] == 'stop' and r.get('response_text') for r in calibration):
                break
        else:
            raise RuntimeError('CLEAN_CALIBRATION_UNFINISHED_AT_MAX_BUDGET; no lossy request allowed')
        suite.update(probes=probes)
        suite['hash'] = content_hash(suite)
        save(output / 'suite.json', suite)
        save(output / 'request_protocol.json', {'version': 'MT_BENCH_INSPECT_CHAT_V1',
             'protocol': suite['request_protocol'], 'generation': probes[0]['generation_config'],
             'prompt_tokens': lengths, 'max_model_len': 4096, 'frozen_before_lossy': True,
             'source_sha256': manifest['sha256'], 'suite_hash': suite['hash']})
        smoke = next(p for p in probes if p['dataset_sample']['question_id'] == 149)
        rank0 = next(s for s in status if s['ranks']['global']['value'] == 0)
        epochs = [s['service_epoch'] for s in status]
        modules = {e: paged(client, 'list_modules', e) for e in epochs}
        save(output / 'model_modules.json', modules)
        down = [r['module_name'] for r in modules[rank0['service_epoch']] if r['logical_role'] == 'MLP_DOWN']
        assert len(down) == 64
        descriptions = []
        for e in epochs:
            descriptions.extend(selected(client.call('describe_target', target_service_epochs=[e], module_name=down[0])))
        save(output / 'target_quantization_contract.json', descriptions)
        assert all(d['quantization']['resolution'] == 'AUDITED_IMPLEMENTATION' for d in descriptions)
        active = selected(client.call('observe_start', target_service_epochs=epochs, module_names=down,
                          request_window='official_149_target_binding', exclusive=True, ttl_seconds=300,
                          max_events=512, max_bytes=2097152, samples_per_module=2))
        sessions = {s['service_epoch']: s['session'] for s in active}
        try:
            observed = owner.request(smoke, 'target_observation', 0)
            events = {e: paged(client, 'observe_poll', e, session=sessions[e]) for e in epochs}
        finally:
            stopped = selected(client.call('observe_stop', target_service_epochs=epochs, session_by_epoch=sessions))
        save(output / 'observations.json', events)
        save(output / 'observer_stop.json', stopped)
        assert all(not s['active'] and not s['installed_handles'] for s in stopped)
        assert all({e['module_name'] for e in ev} == set(down) for ev in events.values())
        candidates = paged(client, 'list_candidates', rank0['service_epoch'], filters={'logical_role': 'MLP_DOWN'})
        assert len(candidates) == 64
        save(output / 'candidates.json', candidates)
        plan = selected(client.call('export_plan', target_service_epochs=[rank0['service_epoch']],
                        candidate_ids=[c['candidate_id'] for c in candidates],
                        runtime_fingerprint=rank0['runtime_fingerprint'], frozen_arm=owner.arm))[0]
        save(output / 'draft_plan.json', plan)
        prior = json.loads(Path('/home/haoran/distlive_20260909_v1/normal_v2/fault_configuration.json').read_text())
        assert set(prior['actual_module_names']) == set(plan['selected_modules'])
        validation_args = {'session_dir': str(session), 'suite': suite, 'probe': smoke,
                           'fault_configuration': prior, 'plan': plan, 'validation_run': 'dataset_149_identity'}
        result = validate_plan(plan, epoch=plan['service_epoch'], fingerprint=plan['runtime_fingerprint'], executor=factory(validation_args))
        save(output / 'ValidationResult.json', result.to_dict())
        assert result.lossless_passed and result.restore_exact
        raw_identity = result.to_dict()['raw']
        assert observed['response_text'] == raw_identity['baseline']['response_text']
        before = owner.clean_query()
        assert cleanup_gate(before, owner.service_healthy(), 2)['status'] == 'PASS'
        cycle = {'cycle_id': 'cycle_01', 'before_baseline': before, 'service_healthy_before': True, 'pairs': {}}
        order = [smoke] + [p for p in probes if p is not smoke]
        config = None
        for p in order:
            current = selected(client.call('status'))
            assert all(not s['active'] and not s['installed_handles'] for s in current)
            assert any(s['service_epoch'] == plan['service_epoch'] and s['runtime_fingerprint'] == plan['runtime_fingerprint'] for s in current)
            clean_before = owner.clean_query()
            assert cleanup_gate(clean_before, owner.service_healthy(), 2)['status'] == 'PASS'
            baseline = raw_identity['baseline'] if p is smoke else owner.request(p, 'baseline', 1)
            try:
                actual = owner.install()
                assert set(actual.actual_module_names) == set(plan['selected_modules'])
                if config is not None:
                    assert config.to_dict() == actual.to_dict()
                config = actual
                fault = owner.fault_request(p, config, 1)
            finally:
                owner.disarm_and_remove()
            clean_after = owner.clean_query()
            assert cleanup_gate(clean_after, owner.service_healthy(), 2)['status'] == 'PASS'
            assert fault['mechanism_summary']['status'] == 'PASS', 'INVALID_FAULT_RUN'
            recovery = owner.request(p, 'recovery', 1)
            cycle['pairs'][p['probe_id']] = {'baseline': baseline, 'fault': fault, 'recovery': recovery}
            cycle.update(after_remove=owner.clean_query(), service_healthy=owner.service_healthy())
            save(output / ('pair_' + p['probe_id'] + '.json'), {'probe': p, 'before': clean_before,
                 'after': clean_after, 'pair': cycle['pairs'][p['probe_id']]})
            subset = {**suite, 'probes': [q for q in probes if q['probe_id'] in cycle['pairs']], 'probe_count': len(cycle['pairs'])}
            subset['hash'] = content_hash(subset)
            report = evaluate_dataset(subset, config, [cycle], evaluation_id='inspect-mtbench-' + str(len(cycle['pairs'])))
            write_report(output / ('effect_' + str(len(cycle['pairs']))), report,
                         {'source': manifest, 'complete_cycles': 1, 'suite_hash': subset['hash'], 'live': True})
            delivery = report.to_dict()['dataset_delivery']
            print(json.dumps({'dataset_progress': len(cycle['pairs']), 'sample': p['probe_id'], 'delivery': delivery['DATASET_BACKED_LOSSY_INTEGRATION'],
                              'visible': delivery['visible_sample_ids'], 'recovery_exact': delivery['exact_recovery']}), flush=True)
            assert delivery['DATASET_BACKED_LOSSY_INTEGRATION'] == 'PASS', 'integration/recovery gate before next sample'
        save(output / 'fault_configuration.json', config.to_dict())
        save(output / 'live_raw.json', {'suite': suite, 'config': config.to_dict(), 'cycles': [cycle]})
        final.update(complete=True, sample_count=8, delivery=delivery)
    except BaseException as exc:
        final.update(error=repr(exc), traceback=traceback.format_exc())
        print(final['traceback'], flush=True)
        raise
    finally:
        if preflight_passed and owned_lease(session):
            try:
                client.call('observe_stop')
                owner.disarm_and_remove()
                final['clean'] = cleanup_gate(owner.clean_query(), owner.service_healthy(), 2)
            except BaseException as exc:
                final['cleanup_error'] = repr(exc)
        save(output / 'dataset_final.json', final)
