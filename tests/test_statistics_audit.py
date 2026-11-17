import copy
import json
from pathlib import Path
import tempfile
import unittest

from fault_injection.statistics_audit import compare_raw_statistics, compact_summary


def tensor(**values):
    return dict(type='torch.Tensor', dtype='torch.float16', shape=[8, 16],
                **dict(dict(Max=1, Min=-1, Mean=0, Norm=4), **values))


class StatisticsAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.b, self.f = self.root/'baseline', self.root/'fault'

    def capture(self, path, tensors):
        path.mkdir()
        doc = dict(task='statistics', level='mix', framework='pytorch', dump_data_dir=None,
                   data={name: dict(output=[value]) for name, value in tensors})
        for name, value in (('dump.json', doc), ('stack.json', {}), ('construct.json', {}),
                ('config.json', dict(task='statistics', level='mix', dump_path=str(path),
                                     statistics=dict(summary_mode='statistics'))),
                ('collector.json', dict(package=str(path/'msprobe'), files={'actual.py':'a'*64}))):
            (path/name).write_text(json.dumps(value), encoding='utf-8')

    def pair(self):
        return compare_raw_statistics(baseline=self.b, fault=self.f,
            baseline_config=self.b/'config.json', fault_config=self.f/'config.json',
            baseline_collector=self.b/'collector.json', fault_collector=self.f/'collector.json')

    def test_fixed_tolerance_rejects_old_fia_change_and_reports_later_operator(self):
        self.capture(self.b, [('NPU.fia.0.forward',tensor(Norm=660.572120626524)), ('Functional.up_proj.0.forward',tensor(Max=25.75))])
        self.capture(self.f, [('NPU.fia.0.forward',tensor(Norm=660.5722374)), ('Functional.up_proj.0.forward',tensor(Max=15.6875))])
        originals = {p:p.read_bytes() for root in (self.b,self.f) for p in root.iterdir()}
        result = self.pair()
        self.assertEqual(result['first_exact_change']['identity'], 'NPU.fia.0.forward/output/0')
        self.assertEqual(result['first_exceedance']['identity'], 'Functional.up_proj.0.forward/output/0')
        self.assertEqual(result['first_exceedance']['baseline'], '25.75')
        self.assertEqual((result['rtol'],result['atol']), ('0.00001','0.00000001'))
        self.assertTrue(result['e5_statistics_comparable'])
        self.assertFalse(result['root_cause_identified'])
        self.assertEqual(originals, {p:p.read_bytes() for p in originals})

    def test_earlier_operator_wins_without_expected_location_hint(self):
        self.capture(self.b,[('Torch.add.0.forward',tensor()),('NPU.fia.0.forward',tensor())])
        self.capture(self.f,[('Torch.add.0.forward',tensor(Max=2)),('NPU.fia.0.forward',tensor(Max=3))])
        self.assertEqual(self.pair()['first_exceedance']['identity'],'Torch.add.0.forward/output/0')

    def test_null_is_unobservable_not_equal(self):
        self.capture(self.b,[('NPU.fia.0.forward',tensor(Mean=None)),('Torch.mm.0.forward',tensor())])
        self.capture(self.f,[('NPU.fia.0.forward',tensor(Mean=None)),('Torch.mm.0.forward',tensor(Max=2))])
        result=self.pair()
        self.assertEqual(len(result['unobservable_fields']),1)
        self.assertEqual(result['first_exceedance']['preceding_unobservable_fields'],1)
        self.assertTrue(result['first_location_is_observable_only'])
        summary=compact_summary(result)
        self.assertNotIn('unobservable_fields',summary)
        self.assertEqual(summary['unobservable_fields_count'],1)
        self.assertEqual(summary['full_dump_inventory']['baseline']['counts']['null'],1)

    def test_all_null_does_not_pass_e5(self):
        row=tensor(Max=None,Min=None,Mean=None,Norm=None)
        self.capture(self.b,[('NPU.fia.0.forward',row)])
        self.capture(self.f,[('NPU.fia.0.forward',row)])
        result=self.pair()
        self.assertFalse(result['whole_dump_comparable'])
        self.assertFalse(result['baseline_self_comparable'])
        self.assertFalse(result['e5_statistics_comparable'])
        self.assertIsNone(result['first_exceedance'])

    def test_boolean_tensor_extrema_compare_without_changing_raw_bytes(self):
        left = dict(tensor(Max=False, Min=False, Mean=None, Norm=None), dtype='torch.bool')
        right = dict(left, Max=True)
        self.capture(self.b, [('Torch.where.0.forward', left)])
        self.capture(self.f, [('Torch.where.0.forward', right)])
        original = (self.f/'dump.json').read_bytes()
        report = self.pair()
        self.assertTrue(report['baseline_self_comparable'])
        self.assertEqual(report['first_exceedance']['baseline'], '0')
        self.assertEqual(report['first_exceedance']['fault'], '1')
        self.assertEqual(report['first_exceedance']['tolerance'], '1E-8')
        self.assertTrue(report['boolean_extrema']['fault'][0]['raw_json_value'])
        self.assertEqual((self.f/'dump.json').read_bytes(), original)

    def test_boolean_value_in_float_tensor_remains_invalid(self):
        self.capture(self.b, [('Torch.where.0.forward', tensor(Max=True))])
        self.capture(self.f, [('Torch.where.0.forward', tensor(Max=True))])
        report = self.pair()
        self.assertFalse(report['baseline_self_comparable'])
        self.assertEqual(report['full_dump_inventory']['baseline']['counts']['unsupported'], 1)
        self.assertEqual(report['boolean_extrema']['baseline'], [])

    def test_nan_after_difference_blocks_whole_dump_qualification(self):
        self.capture(self.b,[('NPU.fia.0.forward',tensor()),('Torch.mm.0.forward',tensor(Mean='NaN'))])
        self.capture(self.f,[('NPU.fia.0.forward',tensor(Norm=3)),('Torch.mm.0.forward',tensor(Mean='NaN'))])
        result=self.pair()
        self.assertEqual(result['status'],'INCOMPARABLE_STATISTICS')
        self.assertEqual(result['first_exceedance']['identity'],'NPU.fia.0.forward/output/0')
        self.assertFalse(result['baseline_self_comparable'])
        self.assertFalse(result['e5_statistics_comparable'])
        self.assertEqual(result['full_dump_inventory']['fault']['counts']['nan'],1)

    def test_nan_before_finite_difference_is_preserved(self):
        self.capture(self.b,[('Torch.add.0.forward',tensor(Mean='NaN')),('NPU.fia.0.forward',tensor())])
        self.capture(self.f,[('Torch.add.0.forward',tensor(Mean='NaN')),('NPU.fia.0.forward',tensor(Norm=3))])
        result=self.pair()
        self.assertEqual(result['first_exceedance']['preceding_incomparable_fields'],1)
        self.assertFalse(result['e5_statistics_comparable'])

    def test_alignment_is_not_guessed(self):
        self.capture(self.b,[('NPU.fia.0.forward',tensor())])
        self.capture(self.f,[('NPU.fia.0.forward',tensor(Max=2))])
        for file, mutation, reason in (('stack.json',{'stack':['different']},'stack'),
                                     ('construct.json',{'module':'other'},'construct'),
                                     ('collector.json',{'files':{'actual.py':'b'*64}},'collector')):
            target=self.f/file; original=target.read_bytes()
            target.write_text(json.dumps(mutation))
            result=self.pair()
            self.assertEqual(result['status'],'ALIGNMENT_MISMATCH')
            self.assertIn(reason,result['mismatched'])
            self.assertIsNone(result['first_exceedance'])
            self.assertFalse(result['e5_statistics_comparable'])
            target.write_bytes(original)
        changed=json.loads((self.f/'dump.json').read_text())
        changed['data']['NPU.fia.0.forward']['output'][0]['shape']=[1,16]
        (self.f/'dump.json').write_text(json.dumps(changed))
        self.assertIn('structure',self.pair()['mismatched'])

    def test_zero_baseline_and_signed_infinity_have_no_relative_value(self):
        self.capture(self.b,[('NPU.fia.0.forward',tensor(Max=0))])
        self.capture(self.f,[('NPU.fia.0.forward',tensor(Max=0.01))])
        self.assertIsNone(self.pair()['first_exceedance']['relative_change'])
        doc=json.loads((self.f/'dump.json').read_text())
        doc['data']['NPU.fia.0.forward']['output'][0]['Max']='Inf'
        (self.f/'dump.json').write_text(json.dumps(doc))
        result=self.pair()
        self.assertEqual(result['first_exceedance']['reason'],'NONFINITE_TRANSITION')
        self.assertIsNone(result['first_exceedance']['relative_change'])

    def test_decimal_precision_and_baseline_reference_at_threshold(self):
        self.capture(self.b,[('NPU.fia.0.forward',tensor())])
        self.capture(self.f,[('NPU.fia.0.forward',tensor())])
        path=self.f/'dump.json'
        raw=path.read_text().replace('"Max": 1,','"Max": 1.00001001,')
        path.write_text(raw)
        self.assertIsNone(self.pair()['first_exceedance'])
        path.write_text(raw.replace('1.00001001,','1.000010010000000000000000001,'))
        self.assertEqual(self.pair()['first_exceedance']['fault'],'1.000010010000000000000000001')

    def test_duplicate_keys_and_nonstatistics_config_are_rejected(self):
        self.capture(self.b,[('NPU.fia.0.forward',tensor())])
        self.capture(self.f,[('NPU.fia.0.forward',tensor())])
        path=self.f/'dump.json'; original=path.read_text()
        path.write_text(original.replace('"Max": 1,','"Max": 1, "Max": 2,'))
        with self.assertRaisesRegex(ValueError,'DUPLICATE_JSON_KEY'):self.pair()
        path.write_text(original)
        (self.f/'config.json').write_text('{"task":"tensor","level":"mix"}')
        with self.assertRaisesRegex(ValueError,'CONFIGURATION_NOT_STATISTICS'):self.pair()


if __name__=='__main__':unittest.main()
