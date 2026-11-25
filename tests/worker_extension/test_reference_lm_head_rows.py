import dataclasses
import unittest
import torch
from reference_adapters.lm_head_rows import LmHeadRows,RowAdmission,VARIANTS,is_weight_reader
from fault_injection.engines.region_lease import RegionLeases,exact


class RowTests(unittest.TestCase):
    def test_reader_accepts_actual_linear_and_mm_but_not_copies_or_slices(self):
        w=torch.ones(16,8,dtype=torch.bfloat16);x=torch.ones(1,8,dtype=w.dtype)
        self.assertTrue(is_weight_reader('aten::mm',(x,w.t()),w))
        for name in ('aten::linear','npu::npu_linear','vllm::unquantized_gemm'):
            self.assertTrue(is_weight_reader(name,(x,w),w))
            self.assertFalse(is_weight_reader(name,(x,w.clone()),w))
            self.assertFalse(is_weight_reader(name,(x,w[:8]),w))
        self.assertFalse(is_weight_reader('aten::t',(w,),w))
    def make(self,variant,start=0,end=16,**changes):
        weight=torch.arange(start*8,end*8,dtype=torch.float32).reshape(end-start,8).to(torch.bfloat16)
        admission=RowAdmission(32,start,end,start,end,(),False,'a'*64,'aten::mm:weight')
        admission=dataclasses.replace(admission,**changes)
        leases=RegionLeases('owner',1,stream=lambda:0,complete=lambda s:None)
        boundary=dict(dedicated_instance=True,accepting=False,inflight=0,device_readers_complete=True)
        holder=[weight]
        site=LmHeadRows(weight,admission,variant=variant,leases=leases,
            boundary=lambda:boundary,observe_weight=lambda:holder[0])
        return site,weight,holder,boundary

    def test_owner_rules_actual_linear_and_exact_restore_two_scopes(self):
        for variant in VARIANTS:
            for owner in range(2):
                site,weight,_,_=self.make(variant);before=weight.clone()
                h=torch.arange(8,dtype=torch.bfloat16).reshape(1,8)/1024
                base=torch.nn.functional.linear(h,weight)
                site.enable();fault=torch.nn.functional.linear(h,weight)
                if variant.endswith('collapse'):
                    self.assertTrue(torch.equal(fault[:,10:13],base[:,10:11].expand(-1,3)))
                else:
                    self.assertTrue(torch.equal(fault[:,10:12],base[:,[11,10]]))
                self.assertFalse(torch.equal(base.float().softmax(-1),fault.float().softmax(-1)))
                site.consumed(weight,consumer='aten::mm:weight',request_id='request')
                site.close();self.assertTrue(exact(weight,before))
                self.assertEqual(site.regions.leases.count(),0)
                self.assertTrue(site.snapshot()['events'][0]['restored'])

    def test_nonowner_guard_never_writes(self):
        site,w,_,_=self.make(VARIANTS[0],16,32);before=w.clone();site.enable()
        site.consumed(w,consumer='aten::mm:weight',request_id='request');site.close()
        self.assertTrue(exact(w,before));self.assertEqual(site.snapshot()['events'],[])
        self.assertEqual(site.snapshot()['nonowner_guard_reads'],1)

    def test_unknown_copies_and_tied_embeddings_refused(self):
        for changes in ({'derived_copies':None},{'derived_copies':('nz',)},{'tied_embedding':True}):
            with self.assertRaises(Exception):self.make(VARIANTS[0],**changes)
    def test_split_group_refused(self):
        with self.assertRaises(Exception):self.make(VARIANTS[0],0,11)
    def test_reader_mismatch_not_counted(self):
        site,w,_,_=self.make(VARIANTS[0]);site.enable()
        with self.assertRaises(Exception):site.consumed(w.clone(),consumer='aten::mm:weight',request_id='r')
        self.assertEqual(site.snapshot()['reader_observations'],0)
        site.close();self.assertEqual(site.snapshot()['events'][0]['restore_kind'],'FAILED_ENABLE_ROLLBACK_NO_CONSUMER')
    def test_reallocation_prevents_old_snapshot_writeback(self):
        site,w,holder,_=self.make(VARIANTS[0]);site.enable();holder[0]=w.clone()
        replacement=holder[0].clone()
        with self.assertRaises(Exception):site.close()
        self.assertTrue(exact(holder[0],replacement));self.assertEqual(site.regions.leases.count(),1)
    def test_drain_and_intervening_write_are_required(self):
        site,w,_,boundary=self.make(VARIANTS[0]);boundary['accepting']=True
        with self.assertRaises(Exception):site.enable()
        self.assertEqual(site.regions.leases.count(),0)
        boundary['accepting']=False;site.enable()
        site.consumed(w,consumer='aten::mm:weight',request_id='r')
        w[10,0]=123
        with self.assertRaises(Exception):site.close()
        self.assertEqual(w[10,0],123);self.assertEqual(site.regions.leases.count(),1)
