import copy
import json
from pathlib import Path
import unittest
from ascend_fi.evaluation.distortion.contracts import FaultConfigurationRecord, content_hash
from ascend_fi.evaluation.distortion.normal_cases import catalog, calibrate
from ascend_fi.evaluation.distortion.evaluator import compare
from ascend_fi.evaluation.distortion.mechanism import validate_mechanism
from tests.evaluation.test_distortion import SUITE, response, probe, config, fault_row, base_for


class NormalCasesTests(unittest.TestCase):
    def test_fixture_library(self):
        fixtures = json.loads(Path(__file__).with_name('normal_detector_fixtures.json').read_text())
        for f in fixtures['cases']:
            with self.subTest(f=f['id']):
                p = probe(expected=f.get('expected'), detectors=f['detectors'])
                r = compare(p, response(f['baseline'], count=f.get('baseline_count')), response(f['response']), SUITE['thresholds'])
                self.assertEqual(any(d['triggered'] is True for d in r['detectors']), f['expected_trigger'])
                if f.get('semantic_review'):
                    self.assertEqual(r['semantic_status'], 'SEMANTIC_REVIEW_REQUIRED')

    def test_one_source_catalog(self):
        result = catalog(SUITE)
        self.assertEqual(result['hash'], content_hash(result))
        for p, c in zip(SUITE['probes'], result['cases']):
            self.assertEqual(p['prompt'], c['prompt'])
            self.assertEqual(c['hash'], content_hash(c))
            self.assertEqual(c['case_id'], p['probe_id'])

    def test_calibration_preserves_failed_and_unstable(self):
        one = {p['probe_id']: response(base_for(p)) for p in SUITE['probes']}
        two = copy.deepcopy(one)
        one['exact_fact'] = response('wrong'); two['exact_fact'] = response('wrong')
        two['exact_math'] = response('wrong')
        c = calibrate(SUITE, [one, two])
        self.assertEqual(len(c['matrix']), 16)
        self.assertEqual(c['matrix']['exact_fact']['task_answer']['status'], 'BASELINE_FAILED')
        self.assertEqual(c['matrix']['exact_math']['task_answer']['status'], 'BASELINE_UNSTABLE')
        self.assertEqual(c['matrix']['exact_math']['structure']['status'], 'NOT_APPLICABLE')

    def test_budget_is_upper_bound_no_padding(self):
        row = fault_row(response('42'))
        conf = config().to_dict()
        conf['hit_policy'] = {'max_hits': 4096, 'budget_semantics': 'AT_MOST_NO_PADDING'}
        conf['frozen_arm_config']['max_hits'] = 4096
        for envelope in row['fault_transaction'].values():
            for w in envelope.get('results', []):
                if 'armed_config' in w:
                    w['armed_config']['max_hits'] = 4096
        result = validate_mechanism(row, FaultConfigurationRecord.from_dict(conf), True)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['measurement_hit_count'], 1)
        self.assertEqual(result['measurement_classification'], 'MUTATION_CONFIRMED')

    def test_metadata_never_changes_prompt(self):
        # V2 changes transport/template only, not prompt content or thresholds.
        self.assertEqual(SUITE['supersedes']['hash'], 'abfc3c9a26597e3eefc7204306553dd381353c45dfe5b3d135632bbd8856d502')
        restored = copy.deepcopy(SUITE)
        restored['version'] = restored.pop('supersedes')['version']
        restored.pop('request_protocol')
        self.assertEqual(content_hash(restored), 'abfc3c9a26597e3eefc7204306553dd381353c45dfe5b3d135632bbd8856d502')


if __name__ == '__main__':
    unittest.main()
