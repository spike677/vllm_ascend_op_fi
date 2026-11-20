import unittest
import torch
from experiments.worker_core_v11_20261118.quant_health_observer import HealthHashes

class HealthHashTests(unittest.TestCase):
    def test_original_values_and_complete_hashes(self):
        model=torch.nn.Identity();processor=torch.nn.Identity();observer=HealthHashes(model,processor);observer.install()
        try:
            for count in (16,8):
                values=torch.arange(count*4,dtype=torch.float32).reshape(count,4)
                self.assertIs(model(values),values);self.assertIs(processor(values),values)
        finally:observer.close()
        report=observer.report();self.assertEqual(len(report['rows']),4);self.assertEqual(len(model._forward_hooks),0)
        self.assertEqual(report['rows'][0]['sha256'],report['rows'][1]['sha256'])
    def test_partial_decode_refused(self):
        model=torch.nn.Identity();observer=HealthHashes(model,torch.nn.Identity());observer.install()
        model(torch.ones(8,4));observer.close()
        with self.assertRaisesRegex(RuntimeError,'COMPLETE_PREFILL_DECODE'):observer.report()
    def test_nonfinite_refused_and_cleanup(self):
        model=torch.nn.Identity();observer=HealthHashes(model,torch.nn.Identity());observer.install()
        try:
            with self.assertRaisesRegex(RuntimeError,'NONFINITE'):model(torch.full((8,4),float('inf')))
        finally:observer.close()
        self.assertEqual(len(model._forward_hooks),0)

class HealthReviewTests(unittest.TestCase):
    def test_legacy_is_unobserved_not_pass(self):
        from experiments.worker_core_v11_20261118.review_batch_two_scopes import review_health
        self.assertFalse(review_health({}))
        with self.assertRaisesRegex(ValueError,'FULL_HEALTH_REQUIRED'):review_health({},True)
    def test_changed_single_output_digest_refused(self):
        from copy import deepcopy
        from experiments.worker_core_v11_20261118.review_batch_two_scopes import review_health
        before=dict(complete=True,owned_hooks=0,rows=[dict(kind=k,finite=True,sha256='a'*64,shape=[8,4]) for k in ('hidden','logits','hidden','logits')])
        data=dict(health_hashes_before=before,health_hashes_after=deepcopy(before))
        self.assertTrue(review_health(data,True))
        data['health_hashes_after']['rows'][3]['sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'FULL_HEALTH_CHANGED'):review_health(data,True)

if __name__=='__main__':unittest.main()
