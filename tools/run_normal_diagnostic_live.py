"""Use the verified E1 RPC/completion client with the existing Distortion owner interface.

Run only inside a session acquired by distortion_service_session.py. No new
controller, fault implementation, model launch arguments, or padding requests.
"""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ascend_fi.evaluation.distortion.contracts import load_suite, content_hash, FaultConfigurationRecord
from ascend_fi.evaluation.distortion.normal_cases import catalog, calibrate
from ascend_fi.evaluation.distortion.mechanism import cleanup_gate, validate_mechanism
from ascend_fi.evaluation.distortion.replay import configuration_from_raw
from ascend_fi.evaluation.distortion.live import collect_cycles
from ascend_fi.evaluation.distortion.evaluator import evaluate_cycles
from ascend_fi.evaluation.distortion.report import write_report

ROOT = Path('/home/haoran/demoncat_overnight_20260907')
OLD = ROOT / 'evaluation/fake_requant_effect_evaluation_v1_complete_20260908T074102Z'
HELPER = OLD / 'scripts/run_fake_requant_effect_complete.py'
BASE = 'http://127.0.0.1:18031'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False); handle.write('\n'); handle.flush(); os.fsync(handle.fileno())


def small(query):
    result = copy.deepcopy(query)
    for worker in result.get('results', []):
        worker.pop('tensor_effect_history', None)
    return result


class VerifiedOwner:
    def __init__(self, session, suite):
        self.session, self.suite = session, suite
        spec = importlib.util.spec_from_file_location('verified_e1_client', HELPER)
        self.client = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.client)
        self.manifest = json.loads((OLD / 'live_validation/manifest.json').read_text())
        self.arm = self.manifest['r1_arm_config']
        assert self.manifest['r1_arm_config_sha256'] == 'sha256:' + content_hash(self.arm)
        self.model = self.manifest['model']
        self.generation = 10000
        self.events = 0
        self.attempt = 'attempt_02_chat'
        self.rawroot = session / 'normal_v2'

    def guard(self):
        owner = json.loads((ROOT / 'coordination/NPU_LOCK/owner.json').read_text())
        assert owner.get('result_dir') == str(self.session) and owner.get('owner') == 'EVALUATION'
        if shutil.disk_usage('/home').free < 12*1024**3:
            raise RuntimeError('DISK_GUARD: below inherited 12 GiB floor')
        if self.events >= 20000:
            raise RuntimeError('EVENT_QUOTA')
        if sum(p.stat().st_size for p in self.session.rglob('*') if p.is_file()) > 4*1024**3:
            raise RuntimeError('SESSION_QUOTA_4_GiB')
        audit = json.loads((self.session / 'service_startup_audit.json').read_text())
        extension = ROOT / 'fake_requant/reference_experiment/fake_requant_reference_extension.py'
        if hashlib.sha256(extension.read_bytes()).hexdigest() != audit['extension_sha256']:
            raise RuntimeError('frozen extension changed')

    def rpc(self, method, kwargs=None):
        return self.client._rpc(BASE, method, kwargs or {}, 180)

    def clean_query(self):
        return small(self.rpc('fake_requant_query'))

    def service_healthy(self):
        return self.client._health(BASE)

    def __enter__(self):
        self.guard(); return self

    def __exit__(self, *args):
        self.disarm_and_remove()

    def preflight(self):
        self.guard()
        with urllib.request.urlopen(BASE + '/v1/models', timeout=10) as r:
            models = json.load(r)
        assert self.model in [m['id'] for m in models['data']]
        clean = cleanup_gate(self.clean_query(), self.service_healthy(), 2)
        assert clean['status'] == 'PASS', clean
        return {'status': 'PASS', 'exclusive_lock_owned': True, 'models': models,
                'clean': clean, 'verified_client_path': str(HELPER),
                'verified_client_sha256': hashlib.sha256(HELPER.read_bytes()).hexdigest(),
                'response_cache': 'DISABLED', 'request_mode': self.suite.get('request_protocol'),
                'prefix_cache': 'launcher unchanged: enabled', 'graph': 'enforce-eager'}

    def request(self, probe, phase, index):
        self.guard()
        protocol = self.suite['request_protocol']
        payload = {'model': self.model, 'messages': [{'role': 'user', 'content': probe['prompt']}],
                   **probe['generation_config'], 'chat_template_kwargs': protocol['chat_template_kwargs'],
                   'logprobs': True, 'top_logprobs': 1}
        status, raw, error = self.client._post(BASE + protocol['endpoint'], payload, 600)
        # Preserve the original chat response; adapt only the existing row decoder's view.
        view = copy.deepcopy(raw)
        if view and view.get('choices'):
            c = view['choices'][0]
            c['text'] = c.get('message', {}).get('content')
            lp = c.get('logprobs') or {}
            c['logprobs'] = {**lp, 'tokens': [t['token'] for t in lp.get('content', [])]}
        row = self.client._choice_row({'id': probe['probe_id'], 'category': probe['category']}, payload, status, view, error)
        row['raw_response'] = raw
        row.update(case_id=probe['probe_id'], phase=phase, cycle_id=f'cycle_{index:02d}',
                   attempt_id=self.attempt, request_generation=None,
                   service_healthy=self.service_healthy(), recorded_at=dt.datetime.now(dt.timezone.utc).isoformat())
        self.events += 1
        # Fault rows are saved by fault_request's finally after disarm evidence.
        if phase != 'fault':
            self.persist(row, phase, index)
        return row

    def persist(self, row, phase, index):
        relative = f'cycles/cycle_{index:02d}/{phase}/{row["sample_id"]}.{self.attempt}.json'
        row['source_ref'] = str(self.rawroot / relative)
        save(self.rawroot / relative, row)
        print(json.dumps({'event': 'request_saved', 'phase': phase, 'cycle': index,
                          'case': row['sample_id'], 'http': row.get('http_status'),
                          'tokens': row.get('generation_length')}), flush=True)

    def install(self, config=None):
        result = self.rpc('fake_requant_install', {'module_regex': self.arm['module_regex'], 'layer_filter': self.arm['layer_filter']})
        assert all(w.get('ok') for w in result['results']), result
        query = self.rpc('fake_requant_query')
        config = configuration_from_raw(self.manifest, query['results'][0]).to_dict()
        config['hit_policy'].update(budget_semantics='AT_MOST_NO_PADDING', short_request_padding=None)
        config['persistence'] = {'window': 'exactly one measured request; finally disarm; no padding'}
        config['runtime_hash'] = hashlib.sha256((ROOT / 'fake_requant/reference_experiment/fake_requant_reference_extension.py').read_bytes()).hexdigest()
        config['topology']['device_ids'] = [2, 5]
        config['unknown_fields'] = ['weights_hash']
        assert len(config['actual_module_names']) == 64 and config['targeted_ranks'] == [0]
        assert config['strength']['scale_multiplier'] == .25 and config['hit_policy']['max_hits'] == 4096
        assert config['strategy'] == 'REFERENCE_REQUANT' and config['module_families'] == ['MLP_DOWN']
        return FaultConfigurationRecord.from_dict(config)

    def disarm_and_remove(self):
        failures = []
        for method in ('fake_requant_disarm', 'fake_requant_remove'):
            try:
                self.rpc(method)
            except Exception as exc:
                failures.append(repr(exc))
        if failures:
            raise RuntimeError('cleanup failed: ' + repr(failures))

    def fault_request(self, probe, config, index, identity=False):
        self.guard()
        assert self.service_healthy()
        self.generation += 1
        arm = {**self.arm, 'request_generation': self.generation}
        if identity:
            arm['mode'] = 'IDENTITY'  # existing no-loss control, not formal R1
        row = {'sample_id': probe['probe_id'], 'case_id': probe['probe_id'], 'http_status': None}
        transaction = {}
        try:
            transaction['arm'] = self.rpc('fake_requant_arm', arm)
            assert all(w.get('ok') for w in transaction['arm']['results'])
            row = self.request(probe, 'fault', index)
            transaction['measurement_query'] = self.rpc('fake_requant_query')
            transaction['transaction_query'] = small(transaction['measurement_query'])
        except Exception as exc:
            row['error'] = repr(exc)
        finally:
            try:
                row['fault_disarm'] = self.rpc('fake_requant_disarm')
            except Exception as exc:
                row['fault_disarm'] = {'error': repr(exc)}
            row.update(fault_transaction=transaction, request_generation=self.generation,
                       attempt_id=self.attempt, cycle_id=f'cycle_{index:02d}')
            if identity:
                self.persist(row, 'identity', index)
            else:
                row['mechanism_summary'] = validate_mechanism(row, config, self.service_healthy())
                self.persist(row, 'fault', index)
        # Retain invalid fault records; only unsafe service/disarm failure stops further requests.
        if row['fault_disarm'].get('http_status') != 200 or not self.service_healthy():
            raise RuntimeError('fault request cleanup/service failure; record preserved')
        return row

    def evidence_refs(self):
        return [str(self.rawroot)]

    def save_cycle(self, cycle):
        cycle['calibration'] = json.loads((self.rawroot / 'calibration.json').read_text())
        save(self.rawroot / 'cycles' / cycle['cycle_id'] / 'cycle.json', cycle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--stage', choices=['calibrate', 'cycles'], required=True)
    parser.add_argument('--suite', type=Path, required=True)
    args = parser.parse_args()
    suite = load_suite(args.suite)
    owner = VerifiedOwner(args.session, suite)
    output = owner.rawroot
    final = {'stage': args.stage}
    try:
        preflight = owner.preflight()
        save(output / (args.stage + '_preflight.json'), preflight)
        if args.stage == 'cycles':
            old_preflight = json.loads((output / 'calibrate_preflight.json').read_text())
            assert old_preflight['verified_client_sha256'] == preflight['verified_client_sha256']
        if args.stage == 'calibrate':
            save(output / 'normal_diagnostic_cases.json', catalog(suite))
            save(output / 'suite.json', suite)
            save(output / 'frozen_r1_source_manifest.json', owner.manifest)
            runs = [{p['probe_id']: owner.request(p, 'calibration', i) for p in suite['probes']} for i in (1, 2)]
            eligibility = calibrate(suite, runs)
            save(output / 'calibration.json', eligibility)
            print(json.dumps({'calibration': eligibility}), flush=True)
        else:
            calibration = json.loads((output / 'calibration.json').read_text())
            assert calibration['suite_hash'] == suite['hash']
            assert calibration['hash'] == content_hash(calibration)
            # Existing IDENTITY and disabled-hook transparency gate before R1.
            p = suite['probes'][0]
            baseline = owner.request(p, 'identity_baseline', 0)
            assert baseline['http_status'] == 200
            config = owner.install()
            disabled = owner.request(p, 'identity_disabled', 0)
            identity = owner.fault_request(p, config, 0, identity=True)
            owner.disarm_and_remove()
            recovery = owner.request(p, 'identity_recovery', 0)
            ws = identity.get('fault_transaction', {}).get('measurement_query', {}).get('results', [])
            gate = {'response_exact': all(r.get('response_text') == baseline['response_text'] for r in (disabled, identity, recovery)),
                    'two_workers': len(ws) == 2, 'identity_hits': sum(w.get('hit_count', 0) for w in ws),
                    'changed_elements': sum(h.get('changed_elements', 0) for w in ws for h in w.get('tensor_effect_history', [])),
                    'clean': cleanup_gate(owner.clean_query(), owner.service_healthy(), 2)}
            save(output / 'identity_gate.json', gate)
            assert gate['response_exact'] and gate['two_workers'] and gate['identity_hits'] > 0 and gate['changed_elements'] == 0 and gate['clean']['status'] == 'PASS', gate
            save(output / 'fault_configuration.json', config.to_dict())
            # Existing scheduler/owner interface. Raw files survive any later exception.
            returned = collect_cycles(owner, suite, config, 3)
            cycles = returned[2]
            for c in cycles:
                c['calibration'] = calibration
            payload = {'schema': 'NORMAL_DIAGNOSTIC_LIVE_V1', 'suite': suite,
                       'config': config.to_dict(), 'cycles': cycles, 'calibration': calibration,
                       'provenance': returned[4]}
            save(output / 'live_raw.json', payload)
            result = evaluate_cycles(suite, config, cycles, evaluation_id='normal_diagnostic_live_v1')
            write_report(output / 'online_report', result, returned[4])
            # Re-open persisted raw data, using the SAME evaluator, not live memory.
            restored = json.loads((output / 'live_raw.json').read_text())
            replay = evaluate_cycles(restored['suite'], FaultConfigurationRecord.from_dict(restored['config']),
                                     restored['cycles'], evaluation_id='normal_diagnostic_live_v1')
            parity = {'equal': result.to_dict() == replay.to_dict(),
                      'online_hash': content_hash(result.to_dict()), 'replay_hash': content_hash(replay.to_dict())}
            save(output / 'online_replay_parity.json', parity)
            assert parity['equal']
            print(json.dumps({'complete_cycles': 3, 'result': result.status}), flush=True)
    except Exception as exc:
        final['error'] = repr(exc); final['traceback'] = traceback.format_exc()
        print(final['traceback'], flush=True)
        raise
    finally:
        try:
            owner.disarm_and_remove()
            final['clean_query'] = owner.clean_query()
            final['health'] = owner.service_healthy()
        except Exception as exc:
            final['cleanup_error'] = repr(exc)
        save(output / (args.stage + '_final.json'), final)


if __name__ == '__main__':
    main()
