import json
from pathlib import Path
import tempfile
import unittest

from compare_model_capture import compare_groups


class ModelCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.baseline, self.fault = self.root / 'baseline', self.root / 'fault'
        for group in (self.baseline, self.fault):
            (group / 'observation').mkdir(parents=True)
            (group / 'config.json').write_text(json.dumps(dict(task='statistics', level='mix',
                dump_path=str(group / 'observation'), statistics={'summary_mode': 'statistics'})))
            (group / 'collector_identity.json').write_text(json.dumps(dict(package='/collector', files={'a.py': 'hash'})))

    def step(self, group, step, maximum=2., stack='same'):
        path = group / 'observation' / f'step{step}' / 'rank0'
        path.mkdir(parents=True)
        tensor = dict(type='torch.Tensor', shape=[1, 4], dtype='torch.bfloat16', Max=maximum, Min=-1., Mean=None, Norm=3.)
        (path / 'dump.json').write_text(json.dumps(dict(task='statistics', level='mix', framework='pytorch',
            dump_data_dir=None, data={'Torch.matmul.0.forward': {'output': [tensor]}})))
        (path / 'stack.json').write_text(json.dumps({'Torch.matmul.0.forward': [stack]}))
        (path / 'construct.json').write_text(json.dumps({'model': ['MatmulProbe']}))

    def test_numeric_step_order_and_all_reports_retained(self):
        for step in (10, 0, 2):
            self.step(self.baseline, step)
            self.step(self.fault, step, 4. if step else 2.)
        result = compare_groups(self.baseline, self.fault)
        self.assertEqual([s['step'] for s in result['steps']], [0, 2, 10])
        self.assertEqual(result['first_difference']['step'], 2)
        self.assertEqual(result['first_difference']['difference']['fault'], 4.)
        self.assertEqual(result['request_mapping'], 'UNPROVEN_REQUEST_MAPPING')
        self.assertTrue(result['steps'][0]['report']['comparison']['unobservable_fields'])

    def test_missing_step_refuses_global_first_difference(self):
        self.step(self.baseline, 0); self.step(self.fault, 0, 3.)
        self.step(self.baseline, 1)
        result = compare_groups(self.baseline, self.fault)
        self.assertEqual(result['status'], 'STEP_SET_MISMATCH')
        self.assertEqual(result['missing_from_fault'], [1])
        self.assertIsNone(result['first_difference'])
        self.assertEqual(len(result['steps']), 2)

    def test_earlier_alignment_failure_blocks_later_first_claim(self):
        self.step(self.baseline, 0); self.step(self.fault, 0, stack='different')
        self.step(self.baseline, 1); self.step(self.fault, 1, 5.)
        result = compare_groups(self.baseline, self.fault)
        self.assertEqual(result['status'], 'EARLIEST_DIFFERENCE_UNPROVEN')
        self.assertIsNone(result['first_difference'])
        self.assertEqual(result['later_observed_difference']['step'], 1)
        self.assertEqual(result['blocking_steps'], [0])


if __name__ == '__main__':
    unittest.main()
