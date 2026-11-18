"""Readiness controls use synthetic CPU JSON, not claimed msprobe captures."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.worker_core_v11_20261118 import review_statistics_readiness as r


class StatisticsReadinessTests(unittest.TestCase):
    def setUp(self):
        self.folder=Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.roots=[self.folder/name for name in ('baseline','repeat','fault')]
        self.tensor=dict(type='torch.Tensor',dtype='torch.float16',shape=[2],Max=2,Min=-1,Mean=.5,Norm=3)
        for index,root in enumerate(self.roots):self.capture(root,fault=index==2)

    def capture(self,root,*,fault=False,step=0):
        path=root/'observation'/('step'+str(step))/'rank0';path.mkdir(parents=True)
        value=deepcopy(self.tensor);value['Max']=4 if fault else 2
        docs={'dump.json':dict(task='statistics',level='mix',framework='pytorch',dump_data_dir=None,
            data={'Torch.add.0.forward':dict(output=[deepcopy(self.tensor)]),'Torch.mm.0.forward':dict(output=[value])}),
            'stack.json':{'0':[['Torch.add.0.forward','Torch.mm.0.forward'],['File /model/example.py, line 1, in forward, \n return x']]},
            'construct.json':{'Torch.mm.0.forward':'Module.actual.forward.0'}}
        for name,document in docs.items():self.write(path/name,document)
        self.write(root/'config.json',dict(task='statistics',level='mix',dump_path=str(path),
            statistics=dict(summary_mode='statistics',scope=[],list=[],tensor_list=[],data_mode=['all'])))
        self.write(root/'collector_identity.json',dict(package='/collector',files={'actual.py':'a'*64}))

    def write(self,path,value):path.write_text(json.dumps(value),encoding='utf-8')

    def change(self,root,name,fn):
        path=root/'observation/step0/rank0'/name;value=json.loads(path.read_text());fn(value);self.write(path,value)

    def inspect(self):return r.review(*self.roots)[0]

    def test_whole_raw_controlled_difference_without_qualification(self):
        originals={p:p.read_bytes() for root in self.roots for p in root.rglob('*.json')}
        value=self.inspect();self.assertTrue(value['all_supplied_steps_have_controlled_observable_difference'])
        self.assertTrue(value['full_numeric_prefix_observed'])
        self.assertEqual(value['pairs'][0]['pairs']['bf']['first_exceedance']['identity'],'Torch.mm.0.forward/output/0')
        self.assertFalse(value['e5'] or value['root_cause_identified'] or value['current_public_scope_qualified'])
        self.assertEqual(originals,{p:p.read_bytes() for p in originals})

    def test_missing_step_is_retained_and_not_promoted(self):
        self.capture(self.roots[0],step=1);value=self.inspect()
        self.assertFalse(value['actual_step_rank_sets_equal'] or value['all_supplied_steps_have_controlled_observable_difference'])
        self.assertEqual(len(value['pairs']),1)

    def test_baseline_repeat_change_confounds_pair_but_does_not_hide_first_difference(self):
        self.change(self.roots[1],'dump.json',lambda d:d['data']['Torch.add.0.forward']['output'][0].update(Max=9))
        value=self.inspect();self.assertFalse(value['all_supplied_steps_have_controlled_observable_difference'])
        self.assertIsNotNone(value['pairs'][0]['pairs']['bf']['first_exceedance'])

    def test_preceding_null_is_not_equal_or_reconstructed(self):
        for root in self.roots:self.change(root,'dump.json',lambda d:d['data']['Torch.add.0.forward']['output'][0].update(Mean=None))
        value=self.inspect();self.assertTrue(value['all_supplied_steps_have_controlled_observable_difference'])
        self.assertFalse(value['full_numeric_prefix_observed'])
        self.assertEqual(value['pairs'][0]['pairs']['bf']['first_exceedance']['preceding_unobservable_fields'],1)

    def test_later_nan_keeps_original_first_but_blocks_whole_capture(self):
        self.change(self.roots[2],'dump.json',lambda d:d['data']['Torch.mm.0.forward']['output'][0].update(Norm='NaN'))
        value=self.inspect();self.assertFalse(value['all_supplied_steps_have_controlled_observable_difference'])
        self.assertIsNotNone(value['pairs'][0]['pairs']['bf']['first_exceedance'])

    def test_earlier_difference_not_expected_operator_wins(self):
        self.change(self.roots[2],'dump.json',lambda d:d['data']['Torch.add.0.forward']['output'][0].update(Max=3))
        self.assertEqual(self.inspect()['pairs'][0]['pairs']['bf']['first_exceedance']['identity'],'Torch.add.0.forward/output/0')

    def test_matching_filters_are_disclosed_and_not_full_capture(self):
        for root in self.roots:
            path=root/'config.json';value=json.loads(path.read_text());value['statistics']['list']=['Torch.mm'];self.write(path,value)
        result=self.inspect();self.assertFalse(result['all_supplied_steps_have_controlled_observable_difference'])
        self.assertFalse(any(result['unfiltered_configuration'].values()))
        self.assertIsNotNone(result['pairs'][0]['pairs']['bf']['first_exceedance'])

    def test_diagnostic_body_flags_capture_without_deleting_record(self):
        for root in self.roots:
            self.change(root,'stack.json',lambda d:d['0'][1].insert(0,'File /source/fault_injection/rmsnorm_return.py, line 1, in _proxy_return, \n changed = x * .5'))
        value=self.inspect();row=value['pairs'][0]
        self.assertFalse(value['all_supplied_steps_have_controlled_observable_difference'])
        self.assertTrue(row['stack_context']['suspected_diagnostic_records'])
        self.assertFalse(row['stack_context']['any_record_excluded'])
        self.assertEqual(row['pairs']['bf']['records_compared'],2)

    def test_legitimate_dispatch_wrapper_alone_not_diagnostic_body(self):
        for root in self.roots:
            self.change(root,'stack.json',lambda d:d['0'][1].insert(0,'File /source/fault_injection/rmsnorm_return.py, line 132, in _read_call, \n result = func(*args, **kwargs)'))
        value=self.inspect();self.assertTrue(value['all_supplied_steps_have_controlled_observable_difference'])
        self.assertFalse(value['pairs'][0]['stack_context']['source_membership_proves_no_contamination'])

    def test_collector_or_call_chain_drift_refuses_pair(self):
        self.change(self.roots[2],'construct.json',lambda d:d.update({'Torch.mm.0.forward':'Module.foreign.forward.0'}))
        value=self.inspect();self.assertFalse(value['all_supplied_steps_have_controlled_observable_difference'])
        self.assertEqual(value['pairs'][0]['pairs']['bf']['status'],'ALIGNMENT_MISMATCH')


if __name__=='__main__':unittest.main()
