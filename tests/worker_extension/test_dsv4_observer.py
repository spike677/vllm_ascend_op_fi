import unittest,torch
from fault_injection.sites.dsv4_observer import empty_report,Dsv4ObserveMode,export_report,reason
class ObserveTests(unittest.TestCase):
 def test_real_matmul_unchanged_and_metadata_only(self):
  a=torch.arange(8,dtype=torch.bfloat16).reshape(2,4);b=torch.ones(4,3,dtype=torch.bfloat16);normal=a@b
  r=empty_report()
  with Dsv4ObserveMode(r,requests=['actual-1'],phase='prefill',call_path=False):out=a@b
  self.assertTrue(torch.equal(out,normal));row=export_report(r,0)
  self.assertEqual(row['mechanisms']['39op']['observed'],1);self.assertEqual(row['mechanisms']['39op']['eligible'],1)
  self.assertFalse(row['qualification_granted']);self.assertEqual(row['injected_calls'],0)
 def test_unmapped_or_unreviewed_layout_not_eligible(self):
  out=torch.ones(2,2,dtype=torch.bfloat16)
  self.assertEqual(reason('39op',(out,),out,[]),'ACTUAL_SCHEDULER_REQUEST_JOIN_MISSING')
  self.assertEqual(reason('28',(out,),out,['r']),'SPARSE_COMPRESSED_LAYOUT_ADAPTER_REQUIRES_RUNTIME_JOIN')
 def test_no_call_rank_never_fabricated(self):
  r=export_report(empty_report(),7)
  self.assertTrue(all(v['observed']==0 for v in r['mechanisms'].values()))
