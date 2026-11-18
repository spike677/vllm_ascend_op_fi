"""Synthetic JSON controls, never substituted for actual msprobe captures."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.worker_core_v11_20261118 import review_core_tensor_statistics as review


class RawStatisticsStepsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        config = dict(task='statistics', level='mix', statistics=dict(scope=[], list=[], tensor_list=[],
            data_mode=['all'], summary_mode='statistics'))
        self.write(self.root/'config.json', config)
        self.write(self.root/'collector_identity.json', dict(files={'pinned.py': 'a'*64}))
        tensor = dict(type='torch.Tensor', dtype='torch.float16', shape=[2], Max=2, Min=-1, Mean=None, Norm=3)
        for step in (0, 1, 2):
            path = self.root/'observation'/('step'+str(step))/'rank0';path.mkdir(parents=True)
            output = deepcopy(tensor);output['Max'] = 4 if step == 2 else 2
            self.write(path/'dump.json', dict(task='statistics', level='mix', framework='pytorch', dump_data_dir=None,
                data={'Torch.add.0.forward': dict(output=[deepcopy(tensor)]),
                      'Torch.mm.0.forward': dict(output=[output])}))
            self.write(path/'stack.json', {'0': [['Torch.add.0.forward', 'Torch.mm.0.forward'], ['same stack']]})
            self.write(path/'construct.json', {})

    def write(self, path, value):
        path.write_text(json.dumps(value), encoding='utf-8')

    def change(self, step, name, fn):
        path = self.root/'observation'/('step'+str(step))/'rank0'/name
        value = json.loads(path.read_text());fn(value);self.write(path, value)

    def test_raw_first_does_not_read_runtime_or_prescribe_operator(self):
        self.write(self.root/'collector_sidecar.json', {'must_not_be_read': 'pretend first is sigmoid'})
        summary, _ = review.review(self.root)
        self.assertTrue(summary['observable_difference_supported'])
        self.assertEqual(summary['pairs']['bf']['first_exceedance']['identity'], 'Torch.mm.0.forward/output/0')
        self.assertFalse(summary['full_numeric_prefix_observed'])
        self.assertFalse(summary['e5'] or summary['sidecar_receipts_read'])

    def test_earlier_operator_changes_win_without_expected_location(self):
        self.change(2, 'dump.json', lambda d: d['data']['Torch.add.0.forward']['output'][0].update(Max=8))
        summary, _ = review.review(self.root)
        self.assertEqual(summary['pairs']['bf']['first_exceedance']['identity'], 'Torch.add.0.forward/output/0')

    def test_repeat_noise_blocks_controlled_difference(self):
        self.change(1, 'dump.json', lambda d: d['data']['Torch.add.0.forward']['output'][0].update(Max=9))
        summary, _ = review.review(self.root)
        self.assertFalse(summary['baseline_controls_pass'] or summary['observable_difference_supported'])

    def test_later_nan_and_structure_drift_are_not_ignored(self):
        self.change(2, 'dump.json', lambda d: d['data']['Torch.mm.0.forward']['output'][0].update(Norm='NaN'))
        summary, _ = review.review(self.root)
        self.assertFalse(summary['observable_difference_supported'])
        self.assertIsNotNone(summary['pairs']['bf']['first_exceedance'])
        self.change(2, 'construct.json', lambda d: d.update({'Torch.mm.0.forward': 'foreign'}))
        summary, _ = review.review(self.root)
        self.assertEqual(summary['pairs']['bf']['status'], 'ALIGNMENT_MISMATCH')

    def test_extra_step_is_not_silently_hidden(self):
        path = self.root/'observation/step3/rank0';path.mkdir(parents=True)
        for name in ('dump.json', 'stack.json', 'construct.json'):
            (path/name).write_bytes((self.root/'observation/step0/rank0'/name).read_bytes())
        with self.assertRaisesRegex(ValueError, 'EXACT_THREE_ACTUAL_STEP_RANK_ORIGINALS_REQUIRED'):
            review.review(self.root)


if __name__ == '__main__':
    unittest.main()
