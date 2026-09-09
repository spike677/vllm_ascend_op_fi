"""Scoring regressions, not synthetic live fault evidence."""
import unittest
import json
import tempfile
from pathlib import Path
from tests.evaluation.test_distortion import SUITE, response, probe, cycles, config, fault_row
from ascend_fi.evaluation.distortion.evaluator import compare, evaluate_cycles, excerpt
from ascend_fi.evaluation.distortion.detectors import task_fact
from ascend_fi.evaluation.distortion.report import render_dashboard
from ascend_fi.evaluation.distortion.replay import load_live_snapshot


class ScoringAuditTests(unittest.TestCase):
    def test_completed_cycle_snapshot_excludes_partial_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'suite.json').write_text(json.dumps(SUITE))
            (root/'fault_configuration.json').write_text(config().to_json())
            partial = root/'cycles/cycle_02/baseline'
            partial.mkdir(parents=True)
            (partial/'partial.json').write_text('{}')
            with self.assertRaises(ValueError):
                load_live_snapshot(root, SUITE)
            complete = root/'cycles/cycle_01'
            complete.mkdir()
            (complete/'cycle.json').write_text(json.dumps(cycles(n=1)[0]))
            loaded = load_live_snapshot(root, SUITE)
            self.assertEqual(len(loaded[2]), 1)
            self.assertTrue(loaded[4]['no_npu_used'])
    def test_budget_without_answer_does_not_prove_pathology(self):
        p = probe(expected={'answer': '70000', 'answer_type': 'number'}, detectors=['task_answer', 'garbage', 'crash'])
        b = response('The final answer is 70000.')
        f = response('The house originally cost 80000. Its improved value is 1.5 times 80000, which equals 120000. Next we account for renovation costs.', finish='length')
        r = compare(p, b, f, SUITE['thresholds'])
        self.assertEqual(r['task_outcome']['status'], 'ANSWER_UNDELIVERED_WITHIN_BUDGET')
        self.assertFalse(r['output_pathology']['confirmed'])
        self.assertNotIn('UNRESOLVED', r['families'])
        self.assertNotEqual(r['verdict'], 'TASK_OR_STRUCTURE_FAILURE')

    def test_correct_answer_must_not_hide_repeated_answer_marker(self):
        p = probe(expected={'answer': '3', 'answer_type': 'number'}, detectors=['task_answer', 'garbage', 'crash'])
        r = compare(p, response('The answer is 3.\n#### 3'), response('The answer is 3.\n' + '\n#### 3\n'*30, finish='length'), SUITE['thresholds'])
        self.assertTrue(r['task_outcome']['correct'])
        self.assertIn('repetition', r['detector_routing']['added_by_actual_output'])
        self.assertIn('REPETITION', r['output_pathology']['families'])
        evidence = next(d['evidence'] for d in r['detectors'] if d['detector'] == 'repetition')
        self.assertGreater(evidence['fault_fragment']['occurrence_count'], 2)
        self.assertTrue(evidence['fault_fragment']['positions'])

    def test_legal_repeated_json_fields_not_pathology(self):
        p = probe(expected={'structure': 'json', 'schema': {'type': 'array'}}, detectors=['structure', 'garbage', 'crash'])
        r = compare(p, response('[{"a":1}]'), response('[' + ','.join('{"a":1}' for _ in range(30)) + ']'), SUITE['thresholds'])
        self.assertFalse(r['output_pathology']['confirmed'])

    def test_answer_span_beyond_excerpt_is_available(self):
        p = probe(expected={'answer': '70000', 'answer_type': 'number'})
        raw = 'Reasoning. '*70 + '\n#### 70000'
        row = response(raw, finish='length')
        e, answer = excerpt(row), task_fact(p, row)
        self.assertTrue(e['truncated_excerpt'])
        self.assertEqual(e['full_text'], raw)
        self.assertGreater(answer['answer_span'][0], 600)
        self.assertEqual(raw[slice(*answer['answer_span'])], '70000')

    def test_local_phenotype_recovery_cannot_hide_other_task_failure(self):
        c = cycles(distort=('exact_math',))
        row = c[0]['pairs']['exact_fact']['recovery']
        row['response_text'] = 'not resolved'
        r = evaluate_cycles(SUITE, config(), c).extra['distortion']
        self.assertEqual(r['recovery']['phenotype_recovery_gate'], 'PASS')
        self.assertEqual(r['recovery']['gate'], 'FAIL')
        self.assertIn('exact_fact', r['cycles'][0]['recovery_scopes']['all_task_validator']['mismatch_ids'])

    def test_dashboard_explains_excerpt_and_scoped_recovery(self):
        html = render_dashboard(evaluate_cycles(SUITE, config(), cycles()))
        for phrase in ('展开完整原始回答', '展示截短', '原生成', '恢复范围', '答案匹配'):
            self.assertIn(phrase, html)
        self.assertNotIn('<span class="badge">HEALTHY', html)
        self.assertNotIn('<span class="badge">DIVERGED_VALID', html)


if __name__ == '__main__':
    unittest.main()
