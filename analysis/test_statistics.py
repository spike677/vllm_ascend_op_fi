"""CPU regressions of the independent statistical contract, not NPU evidence."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from compare_statistics import compare
from statistics_values import compare_values
from accept_statistics import accept

FIXTURE=None
SCRATCH=None


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='statistics-test-',dir=SCRATCH)
        self.root=Path(self.temp.name)
        for arm in ('baseline','fault'):
            src=FIXTURE/arm;dst=self.root/arm
            (dst/'observation/step0/rank0').mkdir(parents=True)
            for f in ('config.json','collector_provenance.json'):shutil.copyfile(src/f,dst/f)
            for f in ('dump.json','stack.json','construct.json'):
                shutil.copyfile(src/'observation/step0/rank0'/f,dst/'observation/step0/rank0'/f)
        self.b=self.root/'baseline/observation/step0/rank0/dump.json'
        self.f=self.root/'fault/observation/step0/rank0/dump.json'

    def tearDown(self):self.temp.cleanup()

    def change(self,path,fn):
        data=json.loads(path.read_bytes());fn(data)
        path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n')

    def test_old_difference_is_masked_without_changing_tolerance(self):
        r=compare(self.b,self.f)
        self.assertEqual(r['first_exact_api_difference']['B_node'],'NPU.npu_fused_infer_attention_score.0.forward')
        self.assertEqual(r['first_observed_api_difference']['B_node'],'Tensor.__mul__.20.forward')
        self.assertFalse(accept(r)['accepted'])

    def test_self_has_no_difference(self):
        r=compare(self.b,self.b)
        self.assertEqual(r['status'],'ALIGNED_NO_ABOVE_TOLERANCE_API_DIFFERENCE')
        self.assertIsNone(r['first_exact_api_difference'])

    def test_earlier_operator_wins(self):
        self.change(self.f,lambda d:d['data']['Functional.embedding.0.forward']['output'][0].update(Mean=0.1))
        r=compare(self.b,self.f)
        self.assertEqual(r['first_observed_api_difference']['B_node'],'Functional.embedding.0.forward')
        self.assertFalse(accept(r)['accepted'])

    def test_null_is_not_equal(self):
        prior=compare(self.b,self.f)['unobservable_fields']
        for path in (self.b,self.f):
            self.change(path,lambda d:d['data']['Functional.embedding.0.forward']['output'][0].update(Mean=None))
        self.assertGreater(compare(self.b,self.f)['unobservable_fields'],prior)

    def test_metadata_mismatch_is_inconclusive(self):
        self.change(self.f,lambda d:d['data']['Functional.embedding.0.forward']['output'][0]['shape'].__setitem__(0,99))
        self.assertEqual(compare(self.b,self.f)['status'],'INCONCLUSIVE_NODE_METADATA')

    def test_trace_mismatch_is_inconclusive(self):
        self.change(self.f,lambda d:d['data'].pop('Functional.embedding.0.forward'))
        self.assertEqual(compare(self.b,self.f)['status'],'INCONCLUSIVE_TRACE_OR_CONFIG')

    def test_synthetic_large_fia_statistic_passes_only_gate(self):
        self.change(self.f,lambda d:d['data']['NPU.npu_fused_infer_attention_score.0.forward']['output'][0].update(Norm=662.0))
        self.assertTrue(accept(compare(self.b,self.f))['accepted'])

    def test_zero_baseline_has_no_epsilon(self):
        a={'type':'torch.Tensor','dtype':'torch.float16','shape':[1],
           'Max':0.,'Min':0.,'Mean':0.,'Norm':0.}
        b=dict(a,Max=1.)
        field=compare_values(a,b)['tolerance_differences'][0]
        self.assertIsNone(field['relative_delta_B'])
        self.assertEqual(field['allowed_delta'],1e-8)

    def test_mean_alone_never_satisfies_effect_gate(self):
        self.change(self.f,lambda d:d['data']['NPU.npu_fused_infer_attention_score.0.forward']['output'][0].update(Mean=0.1))
        r=accept(compare(self.b,self.f));self.assertFalse(r['accepted'])
        self.assertIn('FIRST_FIA_RELIABLE_STATISTIC_BELOW_0_1_PERCENT',r['reasons'])

    def test_invalid_tolerance_is_rejected(self):
        with self.assertRaises(ValueError):compare_values(None,None,rtol=-1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True)
    a=p.parse_args();FIXTURE=a.fixture;SCRATCH=a.scratch;SCRATCH.mkdir(parents=True,exist_ok=True)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({'schema':'fault28.visible.cpu_tests.v1','tests':result.testsRun,
          'failures':len(result.failures),'errors':len(result.errors),'npu_calls':0,
          'synthetic_mutations_are_not_NPU_evidence':True}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
