import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from ascend_fi.inspect.dataset import load_samples, as_probe, SCOPE, SOURCE_FILE
from ascend_fi.inspect.dataset_effect import evaluate_dataset
from ascend_fi.evaluation.distortion.contracts import content_hash
from ascend_fi.evaluation.distortion.report import render_dashboard, write_report
from ascend_fi.contracts import EvaluationResult
from evaluation.test_distortion import SUITE, config, clean, fault_row, response


class DatasetTests(unittest.TestCase):
    def local(self, path):
        raw = b'{"question_id":149,"category":"fixture","turns":["Unmodified fixture?\\n"]}\n'
        (path / 'q.jsonl').write_bytes(raw)
        m = {'repository': 'lm-sys/FastChat', 'source_file': SOURCE_FILE, 'revision': 'a'*40,
             'sha256': hashlib.sha256(raw).hexdigest()}
        (path / 'm.json').write_text(json.dumps(m), encoding='utf-8')
        return load_samples(path/'q.jsonl', path/'m.json', [149])

    def test_exact_original_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            m, s = self.local(Path(tmp))
            self.assertEqual(s[0]['prompt'], 'Unmodified fixture?\n')
            self.assertEqual(as_probe(s[0], 1024)['prompt'], s[0]['prompt'])

    def test_corrupt_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); self.local(path)
            with (path/'q.jsonl').open('ab') as f:
                f.write(b' ')
            with self.assertRaisesRegex(ValueError, 'HASH_MISMATCH'):
                load_samples(path/'q.jsonl', path/'m.json', [149])

    def test_missing_id_no_substitution(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); self.local(path)
            with self.assertRaisesRegex(ValueError, 'SAMPLE_MISSING'):
                load_samples(path/'q.jsonl', path/'m.json', [82])

    def fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, samples = self.local(Path(tmp))
        probe = as_probe(samples[0], 1024)
        suite = {**copy.deepcopy(SUITE), 'version': SCOPE, 'dataset': manifest,
                 'probes': [probe], 'probe_count': 1}
        suite['hash'] = content_hash(suite)
        base = response('A fixture response with no pathology.')
        base.update(sample_id=probe['probe_id'], request={'messages': [{'role':'user','content':probe['prompt']}],
            **probe['generation_config'], **{k: suite['request_protocol'][k] for k in ('chat_template_kwargs','logprobs','top_logprobs')}})
        cycle = {'cycle_id':'one', 'pairs': {probe['probe_id']: {'baseline':base, 'fault':fault_row(base), 'recovery':copy.deepcopy(base)}},
                 'before_baseline':clean(), 'after_remove':clean(), 'service_healthy_before':True, 'service_healthy':True}
        return suite, cycle

    def test_no_pathology_integration_can_pass(self):
        s,c = self.fixture()
        r = evaluate_dataset(s, config(), [c], evaluation_id='fixture').to_dict()
        self.assertEqual(r['status'], 'PASS')
        self.assertEqual(r['distortion']['robust_gate'], 'NOT_APPLICABLE')
        self.assertEqual(r['dataset_delivery']['VISIBLE_DISTORTION_OBSERVATION'], 'NOT_OBSERVED_BY_COVERED_RULES')
        self.assertEqual(r['dataset_delivery']['repeatability'], 'NOT_PROVEN')
        self.assertEqual(r['distortion']['cycles'][0]['matrix'][0]['mechanism_state'], 'CONFIRMED')

    def test_all_three_rewritten_payloads_still_rejected(self):
        s,c = self.fixture()
        for r in next(iter(c['pairs'].values())).values():
            r['request']['messages'][0]['content'] = 'rewritten'
        with self.assertRaisesRegex(ValueError, 'REQUEST_MISMATCH'):
            evaluate_dataset(s, config(), [c], evaluation_id='fixture')

    def test_no_change_not_pass(self):
        s,c = self.fixture()
        f = next(iter(c['pairs'].values()))['fault']
        f['fault_transaction']['measurement_query']['results'][0]['tensor_effect_history'][0]['changed_elements'] = 0
        r = evaluate_dataset(s, config(), [c], evaluation_id='fixture')
        self.assertEqual(r.status, 'FAIL')
        self.assertEqual(r.to_dict()['dataset_delivery']['DATASET_BACKED_LOSSY_INTEGRATION'], 'PARTIAL')
        self.assertEqual(r.to_dict()['distortion']['cycles'][0]['matrix'][0]['mechanism_state'], 'NO_CHANGE')

    def test_no_repeatability_count_relabel(self):
        s,c = self.fixture()
        with self.assertRaisesRegex(ValueError, 'ONE_CYCLE'):
            evaluate_dataset(s, config(), [c,c], evaluation_id='fixture')

    def test_dataset_dashboard_names_scope_and_full_text(self):
        s,c = self.fixture()
        r = evaluate_dataset(s, config(), [c], evaluation_id='fixture')
        html = render_dashboard(r)
        self.assertIn('DATASET_BACKED_LOSSY_INTEGRATION', html)
        self.assertIn('SEMANTIC_REVIEW_REQUIRED', html)
        self.assertNotIn('stable distorted probes', html)
        self.assertIn('Unmodified fixture?', html)

    def test_report_configuration_is_not_hardcoded_to_r1(self):
        s,c = self.fixture()
        data = evaluate_dataset(s, config(), [c], evaluation_id='fixture').to_dict()
        data['fault_configuration'].update(module_families=['ATTN_QKV', 'MLP_DOWN'],
            targeted_layers=[0, 1], available_layers=2, actual_module_names=['qkv', 'down'],
            strength={'scale_provider': 'tensor_absmax', 'scale_multiplier': 20},
            hit_policy={'max_hits': 567808})
        with tempfile.TemporaryDirectory() as tmp:
            write_report(tmp, EvaluationResult.from_dict(data), {'origin': 'TEST_FIXTURE'})
            for name in ('effect_card.md', 'effect_dashboard.html'):
                text = (Path(tmp)/name).read_text(encoding='utf-8')
                self.assertIn('ATTN_QKV, MLP_DOWN', text)
                self.assertIn('567808', text)
                self.assertIn('2/2', text)
                self.assertNotIn('4096', text)
                self.assertNotIn('0.25', text)

    def test_report_exposes_unproven_full_window(self):
        s,c = self.fixture()
        data = evaluate_dataset(s, config(), [c], evaluation_id='fixture').to_dict()
        data['model_targeting'] = {'status': 'PARTIAL', 'plan_sha256': 'p',
            'recovery_exact': True, 'window': {'mechanism': 'PASS',
                'target_call_window': 'PASS', 'full_request_window': 'UNPROVEN'}}
        with tempfile.TemporaryDirectory() as tmp:
            write_report(tmp, EvaluationResult.from_dict(data), {'origin': 'TEST_FIXTURE'})
            for name in ('effect_card.md', 'effect_dashboard.html', 'effect_card.json'):
                text = (Path(tmp)/name).read_text(encoding='utf-8')
                self.assertIn('UNPROVEN', text)
                self.assertIn('PARTIAL', text)
