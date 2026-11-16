import copy
import json
from pathlib import Path
import tempfile
import unittest

from compare_msprobe import compare, ComparisonPolicy


class RawStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        tensor = dict(type='torch.Tensor', shape=[1, 8], dtype='torch.bfloat16', Max=2., Min=-1., Mean=0., Norm=3.)
        self.base = dict(task='statistics', level='mix', framework='pytorch', dump_data_dir=None,
            data={'Torch.matmul.0.forward': dict(input_args=[tensor, copy.deepcopy(tensor)], input_kwargs={}, output=[copy.deepcopy(tensor)])})
        for group in ('baseline', 'fault'):
            folder = self.root / group; folder.mkdir()
            for name, document in [('dump.json', self.base), ('stack.json', {'Torch.matmul.0.forward': ['same stack']}),
                ('construct.json', {'model': ['MatmulProbe']}), ('config.json', dict(task='statistics', level='mix', dump_path=str(folder), statistics={'summary_mode': 'statistics'})),
                ('collector.json', {'package': '/same/package', 'files': {'collector.py': 'sha'}})]:
                (folder / name).write_text(json.dumps(document))
    def result(self):
        b, f = self.root/'baseline', self.root/'fault'
        return compare(b, f, ComparisonPolicy(1e-8, 1e-5), baseline_config=b/'config.json', fault_config=f/'config.json',
            baseline_collector=b/'collector.json', fault_collector=f/'collector.json')
    def fault(self, doc):
        (self.root/'fault'/'dump.json').write_text(json.dumps(doc))
    def test_self_comparison(self):
        self.assertEqual(self.result()['comparison']['status'], 'NO_STATISTICS_DIFFERENCE')
    def test_raw_output_infinity_is_first_difference(self):
        f = copy.deepcopy(self.base); f['data']['Torch.matmul.0.forward']['output'][0]['Max'] = float('inf'); self.fault(f)
        r = self.result()['comparison']
        self.assertEqual(r['identity'], 'Torch.matmul.0.forward/output/0')
        self.assertEqual(r['difference']['fault'], '+Inf')
    def test_null_cannot_be_claimed_equal_or_first_numeric_difference(self):
        f = copy.deepcopy(self.base); f['data']['Torch.matmul.0.forward']['input_args'][0]['Mean'] = None; self.fault(f)
        r = self.result()['comparison']
        self.assertEqual(r['status'], 'NO_OBSERVABLE_STATISTICS_DIFFERENCE')
        self.assertTrue(r['unobservable_fields'])
    def test_input_difference_precedes_output(self):
        f = copy.deepcopy(self.base); f['data']['Torch.matmul.0.forward']['input_args'][1]['Norm'] = 4.; self.fault(f)
        self.assertEqual(self.result()['comparison']['identity'], 'Torch.matmul.0.forward/input_args/1')
    def test_module_relation_mismatch_refuses_alignment(self):
        (self.root/'fault'/'construct.json').write_text('{}')
        self.assertEqual(self.result()['status'], 'ALIGNMENT_MISMATCH')
    def test_dtype_mismatch_refuses_alignment(self):
        f = copy.deepcopy(self.base); f['data']['Torch.matmul.0.forward']['output'][0]['dtype'] = 'torch.float16'; self.fault(f)
        self.assertEqual(self.result()['status'], 'ALIGNMENT_MISMATCH')


if __name__ == '__main__':
    unittest.main()
