import unittest
import torch
from unittest.mock import patch
from fault_injection.engines.region_lease import RegionLeases
from demoncat.runtime.vllm_ascend.worker_extension.external_persistent_regions import PersistentRegions,PersistentRegion


class PersistentTests(unittest.TestCase):
    def setUp(self):
        self.bound=dict(dedicated_instance=True,accepting=False,inflight=0,device_readers_complete=True,
            prefix_cache=False,kv_connector=False,offload=False,shared_pages=False,
            ended_request_generations=[],before_page_release=False)
        self.generation=1;self.request=7;self.tensor=torch.arange(8,dtype=torch.float32)
        self.leases=RegionLeases('owner',1,stream=lambda:0,complete=lambda s:None)
        self.regions=PersistentRegions(self.leases,boundary=lambda:self.bound,maximum_bytes=128)
    def region(self,kind='persistent_weight',**kw):
        d=dict(name='weight',tensor=self.tensor,region_kind=kind,allocation_generation=1,
            observe_generation=lambda:self.generation,consumers=('real_qmm',),derived_copies=())
        if kind=='kv_cache':d.update(request_generation=7,observe_request_generation=lambda:self.request)
        d.update(kw);return PersistentRegion(**d)
    def test_weight_consumer_and_restore_same_generation(self):
        original=self.tensor.clone();self.regions.prepare([self.region()]);self.regions.write('weight',self.tensor*2)
        self.regions.consumed('weight','real_qmm',request_generation=3)
        self.regions.restore('weight');self.assertTrue(torch.equal(original,self.tensor));self.assertFalse(self.leases.active)
        self.assertEqual(self.regions.events[0]['changed_elements'],7)
    def test_ingress_and_inflight_block_write(self):
        for key,value in [('accepting',True),('inflight',1),('device_readers_complete',False)]:
            old=self.bound[key];self.bound[key]=value
            with self.assertRaisesRegex(RuntimeError,'DRAINED'):self.regions.prepare([self.region()])
            self.bound[key]=old
        self.assertFalse(self.leases.active)
    def test_unknown_derivative_and_budget_refuse(self):
        for region in [self.region(derived_copies=None),self.region(derived_copies=('missing',))]:
            with self.assertRaisesRegex(RuntimeError,'DERIVED'):self.regions.prepare([region])
        self.regions.maximum_bytes=1
        with self.assertRaisesRegex(RuntimeError,'BUDGET'):self.regions.prepare([self.region()])
    def test_overlap_refused_before_backup(self):
        with self.assertRaisesRegex(RuntimeError,'OVERLAPPING'):
            self.regions.prepare([self.region(),self.region(name='alias',tensor=self.tensor[1:])])
        self.assertFalse(self.leases.active)
    def test_generation_change_keeps_recovery_owner(self):
        self.regions.prepare([self.region()]);self.regions.write('weight',self.tensor+1);self.generation=2
        with self.assertRaisesRegex(RuntimeError,'GENERATION_CHANGED'):self.regions.restore('weight')
        self.assertEqual(len(self.leases.active),1)
    def test_unobserved_legal_weight_update_never_overwritten(self):
        self.regions.prepare([self.region()]);self.regions.write('weight',self.tensor+1)
        self.regions.consumed('weight','real_qmm',request_generation=3);self.tensor.add_(2);now=self.tensor.clone()
        with self.assertRaisesRegex(RuntimeError,'LEGAL_UPDATE'):self.regions.restore('weight')
        self.assertTrue(torch.equal(now,self.tensor));self.assertEqual(self.regions.events[0]['failed'],1)
    def test_kv_end_before_recycle_only(self):
        original=self.tensor.clone();self.regions.prepare([self.region('kv_cache')]);self.regions.write('weight',self.tensor+1)
        self.regions.consumed('weight','real_qmm',request_generation=7)
        with self.assertRaisesRegex(RuntimeError,'SAFE_END'):self.regions.restore('weight')
        self.tensor.add_(1);self.regions.observe_kv_legal_write('weight',request_generation=7)
        self.bound.update(ended_request_generations=[7],before_page_release=True)
        self.regions.restore('weight');self.assertTrue(torch.equal(original,self.tensor))
    def test_kv_page_reuse_or_shared_rejected(self):
        self.bound['shared_pages']=True
        with self.assertRaisesRegex(RuntimeError,'PRIVATE_KV'):self.regions.prepare([self.region('kv_cache')])
        self.bound['shared_pages']=False;self.regions.prepare([self.region('kv_cache')]);self.request=8
        with self.assertRaisesRegex(RuntimeError,'PAGE_OWNER_CHANGED'):self.regions.write('weight',self.tensor+1)
    def test_unwritten_partial_prepare_cleanup(self):
        self.regions.prepare([self.region()]);self.regions.release_unwritten('weight');self.assertFalse(self.leases.active)
    def test_zero_new_value_change_is_applied(self):
        self.regions.prepare([self.region()]);self.regions.write('weight',self.tensor.clone())
        self.assertEqual((self.regions.events[0]['applied'],self.regions.events[0]['changed']),(1,0))

    def test_multiple_writes_retain_first_backup_and_count_actual_consumers(self):
        original=self.tensor.clone()
        self.regions.prepare([self.region(consumers=('real_qmm','derived_qmm'))])
        for i in range(3):
            self.regions.write('weight',self.tensor+1)
            self.regions.consumed('weight','real_qmm',request_generation=i)
            self.assertEqual(self.regions.events[-1]['consumed'],0)
            self.regions.consumed('weight','derived_qmm',request_generation=i)
            self.assertEqual(self.regions.events[-1]['consumed'],1)
        self.regions.restore('weight')
        self.assertTrue(torch.equal(original,self.tensor))
        self.assertEqual(sum(e['restored'] for e in self.regions.events),3)
        self.assertEqual(self.regions.reader_observations,6)

    def test_rewrite_before_consumption_fails_and_keeps_lease(self):
        self.regions.prepare([self.region()]);self.regions.write('weight',self.tensor+1)
        previous=self.tensor.clone()
        with self.assertRaisesRegex(RuntimeError,'PRIOR_WRITE_READERS_INCOMPLETE'):
            self.regions.write('weight',self.tensor+1)
        self.assertTrue(torch.equal(self.tensor,previous))
        self.assertEqual(self.regions.events[-1]['failed'],1)
        self.assertEqual(len(self.leases.active),1)

    def test_bad_generation_counts_failed_eligible_call(self):
        self.regions.prepare([self.region()]);self.generation=2
        with self.assertRaisesRegex(RuntimeError,'GENERATION_CHANGED'):
            self.regions.write('weight',self.tensor+1)
        e=self.regions.events[-1]
        self.assertEqual((e['eligible'],e['attempted'],e['applied'],e['failed']),(1,1,0,1))

    def test_parameter_write_does_not_build_autograd_graph(self):
        self.tensor=torch.nn.Parameter(self.tensor)
        original=self.tensor.detach().clone();self.regions.prepare([self.region()])
        self.regions.write('weight',self.tensor*2)
        self.regions.consumed('weight','real_qmm',request_generation=3)
        self.regions.restore('weight')
        self.assertTrue(torch.equal(self.tensor,original));self.assertIsNone(self.tensor.grad)

    def test_partial_backup_failure_releases_only_owned_unwritten_regions(self):
        acquire=self.leases.acquire
        def limited(t):
            if self.leases.active:raise RuntimeError('backup allocation failed')
            return acquire(t)
        other=self.region(name='derived',tensor=self.tensor.clone())
        with patch.object(self.leases,'acquire',side_effect=limited):
            with self.assertRaisesRegex(RuntimeError,'backup allocation'):
                self.regions.prepare([self.region(),other])
        self.assertFalse(self.leases.active);self.assertFalse(self.regions.entries)

    def test_partial_enable_rollback_does_not_invent_consumer(self):
        original=self.tensor.clone();self.regions.prepare([self.region()])
        self.regions.write('weight',self.tensor+1)
        self.regions.rollback_unconsumed_weight('weight')
        self.assertTrue(torch.equal(original,self.tensor));self.assertFalse(self.leases.active)
        self.assertEqual(self.regions.reader_observations,0)
        self.assertEqual(self.regions.events[0]['consumed'],0)
        self.assertEqual(self.regions.events[0]['restored'],1)
        self.assertIn('FAILED_ENABLE',self.regions.events[0]['restore_kind'])

    def test_rollback_refuses_intervening_mutation_and_kv(self):
        self.regions.prepare([self.region()]);self.regions.write('weight',self.tensor+1)
        self.tensor.add_(3);current=self.tensor.clone()
        with self.assertRaisesRegex(RuntimeError,'INTERVENING'):
            self.regions.rollback_unconsumed_weight('weight')
        self.assertTrue(torch.equal(current,self.tensor));self.assertTrue(self.leases.active)
        self.regions.entries['weight']['region']=self.region('kv_cache')
        with self.assertRaisesRegex(RuntimeError,'WEIGHT_ROLLBACK_ONLY'):
            self.regions.rollback_unconsumed_weight('weight')
