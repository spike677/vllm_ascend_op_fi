import copy
import json
import tempfile
from pathlib import Path
import unittest

from fault_injection.evidence import EvidenceManager, assess_execution, digest, independent_statistics_pair


ROOT = Path(__file__).resolve().parents[1]


class EvidenceTests(unittest.TestCase):
    def test_append_only_bytes_and_scope(self):
        scope = dict(mechanism='18.api_half', provider_version='1', model_instance_id='model',
                     worker_start_identity='worker', process_start_identity='boot-pid-start',
                     execution_boundary='python_hook', operator='RMSNorm', owner='owner', pid=123,
                     rank=0, logical_device=0, generation=1, source_sha256='a'*64, environment_sha256='b'*64)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root/'original.json'; raw.write_bytes(b'{ "Mean" : 0.00000000001 }\n')
            before = raw.read_bytes()
            manager = EvidenceManager(root/'attempt', scope)
            manager.reference('baseline', raw)
            with self.assertRaises(FileExistsError): manager.reference('baseline', raw)
            manifest = json.loads(manager.finalize().read_text())
            self.assertEqual(raw.read_bytes(), before)
            self.assertEqual(manifest['payload']['baseline.json'], digest(root/'attempt/baseline.json'))
            with self.assertRaises(ValueError): manager.write('fault', {})
            with self.assertRaises(FileExistsError): EvidenceManager(root/'attempt', scope)
            bad = dict(scope, execution_boundary='op_exec_cache_run')
            with self.assertRaisesRegex(ValueError, 'NATIVE_BUILD'): EvidenceManager(root/'bad', bad)

    def fixture(self, root, value, *, extra_nan=False):
        root.mkdir()
        t = dict(type='torch.Tensor', dtype='torch.float16', shape=[8, 16], Max=value, Min=-1., Mean=0., Norm=4.)
        doc = dict(task='statistics', level='mix', framework='pytorch', dump_data_dir=None,
                   data={'Torch.test.0.forward': {'output':[t]}})
        if extra_nan:
            doc['data']['Torch.test.1.forward']={'output':[dict(t, Mean='NaN')]}
        for name, data in [('dump.json', doc),('stack.json',{}),('construct.json',{})]:
            (root/name).write_text(json.dumps(data))
        (root/'config.json').write_text(json.dumps(dict(task='statistics',level='mix',statistics={'summary_mode':'statistics'})))
        (root/'collector.json').write_text(json.dumps(dict(files={'actual.py':'abc'})))

    def pair(self, root):
        b, f = root/'baseline', root/'fault'
        return independent_statistics_pair(comparator=ROOT/'fault39/tools/compare_msprobe.py',
            baseline=b,fault=f,baseline_config=b/'config.json',fault_config=f/'config.json',
            baseline_collector=b/'collector.json',fault_collector=f/'collector.json')

    def test_fixed_tolerance_raw_first_diff_and_post_diff_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.fixture(root/'baseline',1.,extra_nan=True)
            self.fixture(root/'fault',1.01,extra_nan=True)
            result=self.pair(root)
            self.assertEqual(result['pair']['comparison']['identity'],'Torch.test.0.forward/output/0')
            self.assertEqual(result['pair']['comparison']['difference']['field'],'/Max')
            self.assertFalse(result['whole_dump_comparable'])
            self.assertEqual(len(result['full_dump_inventory']['fault']['nonfinite_fields']),1)
            self.assertEqual((result['rtol'],result['atol']),(1e-5,1e-8))

    def test_subtolerance_and_alignment_never_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.fixture(root/'baseline',1.)
            self.fixture(root/'fault',1.+1e-7)
            result=self.pair(root)
            self.assertTrue(result['whole_dump_comparable'])
            self.assertEqual(result['first_difference_status'],'NO_STATISTICS_DIFFERENCE')
            (root/'fault/stack.json').write_text('{"other":1}')
            result=self.pair(root)
            self.assertEqual(result['first_difference_status'],'ALIGNMENT_MISMATCH')
            self.assertFalse(result['whole_dump_comparable'])

    def test_execution_model_recovery_statistics_are_independent(self):
        observations = dict(original_calls=3,original_completed=True,actual_interventions=1,
                            mechanism_verified=True,complete_model_output_compared=True,
                            off_no_new_hits=True,drained=True,device_completed=True,state_restored=True,
                            leases_released=True,off_exact=True,fault_effect='NO_TOKEN_CHANGE')
        record=dict(execution_kind='npu',pretrained_model=True,observations=observations)
        result=assess_execution(record)
        self.assertTrue(all(result[k] for k in ('E1_execution','E2_intervention','E3_model_comparison','E4_recovery')))
        self.assertFalse(result['E5_independent_statistics'])
        self.assertFalse(result['every_match'])
        record['observations']['off_exact']=False
        self.assertFalse(assess_execution(record)['E4_recovery'])
        record['execution_kind']='cpu'
        self.assertFalse(assess_execution(record)['E2_intervention'])
        self.assertFalse(assess_execution(record)['E3_model_comparison'])


if __name__=='__main__': unittest.main()
