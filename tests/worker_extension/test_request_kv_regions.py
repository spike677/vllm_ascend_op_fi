import unittest,torch
from fault_injection.engines.region_lease import RegionLeases
from demoncat.runtime.vllm_ascend.worker_extension.persistent_kv_regions import RequestKVRegions


class RegionTests(unittest.TestCase):
    def setUp(self):
        self.context=dict(request_id='r',request_generation='g',pages=[[0,1]],before_free=False)
        self.base=torch.arange(8,dtype=torch.float32).reshape(2,4);self.original=self.base.clone()
        self.leases=RegionLeases('owner',1,stream=lambda:0,complete=lambda s:None)
        self.regions=RequestKVRegions(self.leases,owner_observation=lambda:self.context,maximum_bytes=32)
    def write(self):self.regions.write_after_producer('v',self.base[0,:1],torch.tensor([42.]),page=[0,1])
    def test_disjoint_legal_write_survives_byte_restore(self):
        self.write();self.base[1,0]=99;self.regions.acknowledge_disjoint_writer('v')
        self.regions.consumed('v');self.context['before_free']=True;self.regions.restore_before_free()
        self.assertEqual(self.base[0,0],self.original[0,0]);self.assertEqual(self.base[1,0],99)
        self.assertEqual(self.leases.count(),0);self.assertEqual(self.regions.events[0]['restored'],1)
    def test_later_write_to_same_region_blocks_restore(self):
        self.write();self.base[0,0]=99
        with self.assertRaisesRegex(RuntimeError,'LEGAL_UPDATE'):self.regions.acknowledge_disjoint_writer('v')
        self.assertEqual(self.leases.count(),1)
    def test_reused_page_or_request_generation_never_written_back(self):
        self.write();self.regions.consumed('v');self.context.update(before_free=True,request_generation='new')
        with self.assertRaisesRegex(RuntimeError,'OWNER_CHANGED'):self.regions.restore_before_free()
        self.assertEqual(self.base[0,0],42)
    def test_return_without_real_consumer_is_not_enough(self):
        self.write();self.context['before_free']=True
        with self.assertRaisesRegex(RuntimeError,'CONSUMER_NOT'):self.regions.restore_before_free()
    def test_backup_budget_refuses_before_write(self):
        self.regions.maximum_bytes=1
        with self.assertRaisesRegex(RuntimeError,'BUDGET'):self.write()
        self.assertTrue(torch.equal(self.base,self.original))
        self.assertEqual(self.regions.events[0]['failed'],1)
    def test_actual_abort_restores_without_fabricating_consumer(self):
        self.write();self.context.update(before_free=True,request_status='FINISHED_ABORTED')
        self.regions.restore_before_free();e=self.regions.events[0]
        self.assertTrue(torch.equal(self.base,self.original));self.assertEqual(self.leases.count(),0)
        self.assertEqual(e['consumed'],0);self.assertEqual(e['restore_kind'],'ABORT_BEFORE_CONSUMER_BYTE_ROLLBACK')


if __name__=='__main__':unittest.main()
