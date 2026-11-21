import unittest
import torch
from fault_injection.engines.region_lease import RegionLeases


class LeaseTests(unittest.TestCase):
    def manager(self):
        self.events=[]
        return RegionLeases('scope',1,stream=lambda:7,complete=lambda s:self.events.append(s))
    def test_backup_write_consume_restore_and_overlap(self):
        m=self.manager();x=torch.arange(8,dtype=torch.float32);b=x.clone()
        token=m.acquire(x[1:5])
        with self.assertRaisesRegex(ValueError,'OVERLAPPING'):m.acquire(x[4:7])
        m.write(token,torch.zeros(4))
        with self.assertRaisesRegex(RuntimeError,'CONSUMER'):m.restore(token)
        m.consumed(token);m.restore(token)
        self.assertTrue(torch.equal(x,b));self.assertEqual(m.count(),0)
        self.assertEqual(self.events,[7]*4)
    def test_storage_rebind_never_restores_old_snapshot_to_new_object(self):
        m=self.manager();x=torch.ones(4);token=m.acquire(x)
        x.data=torch.zeros(4)
        with self.assertRaisesRegex(RuntimeError,'STORAGE_CHANGED'):m.consumed(token)
        self.assertEqual(m.count(),1);self.assertTrue(torch.equal(x,torch.zeros(4)))
    def test_normal_inplace_update_is_not_overwritten(self):
        m=self.manager();x=torch.ones(4);token=m.acquire(x)
        m.write(token,torch.zeros(4));x.add_(2);m.consumed(token)
        with self.assertRaisesRegex(RuntimeError,'LEGAL_UPDATE'):m.restore(token)
        self.assertTrue(torch.equal(x,torch.full((4,),2.)))
    def test_inference_mode_without_version_counter(self):
        with torch.inference_mode():
            m=self.manager();x=torch.ones(4);token=m.acquire(x)
            m.write(token,torch.zeros(4));m.consumed(token);m.restore(token)
            self.assertTrue(torch.equal(x,torch.ones(4)))
    def test_producer_copy_requires_exact_known_output(self):
        m=self.manager();x=torch.ones(4);token=m.acquire(x)
        with self.assertRaisesRegex(RuntimeError,'NOT_PROVEN'):
            m.observe_output_copy(token,torch.zeros(4))
        x.copy_(torch.zeros(4));m.observe_output_copy(token,torch.zeros(4))
        m.consumed(token);m.restore(token)
        self.assertTrue(torch.equal(x,torch.ones(4)))
