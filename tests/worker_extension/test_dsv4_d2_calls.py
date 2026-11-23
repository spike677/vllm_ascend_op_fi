import unittest
import torch
from fault_injection.sites.dsv4_d2_calls import D2FixtureCalls,create_live_adapter
from fault_injection.sites.sparse_mla_v41 import from_operator_call
import test_sparse_mla_v41 as sparse_fixture

class D2CallTests(unittest.TestCase):
    def test_public_gate_closed_even_with_claimed_evidence(self):
        with self.assertRaisesRegex(RuntimeError,'GATE_CLOSED'):create_live_adapter(d1_passed=True)
    def test_cast_pair_two_owners_consumption_restore(self):
        for owner in ('one','two'):
            c=D2FixtureCalls(owner);a=torch.ones(2,8,dtype=torch.bfloat16);b=torch.ones(2,8)
            c.norm_output((a,b));seen=c.consume(lambda x,y:(x.clone(),y.clone()),a,b)
            self.assertEqual(float(seen[0][-1,0]),.5);self.assertEqual(float(seen[1][-1,0]),.5)
            self.assertTrue(torch.equal(a,torch.ones_like(a)));self.assertTrue(torch.equal(b,torch.ones_like(b)))
            self.assertTrue(all(e['consumed'] and e['restored'] for e in c.events));c.close()
    def test_wrong_rotary_rows_keep_q_write_restore_lookup(self):
        c=D2FixtureCalls('scope');q=torch.ones(2,1,1,8);pos=torch.tensor([1,2]);ct=torch.arange(20.).reshape(5,4);st=ct+1
        cos=ct[pos].reshape(2,1,1,4).clone();sin=st[pos].reshape_as(cos).clone();before=cos.clone();calls=[]
        def original(q,c,s,**kw):
            calls.append(kw);q[...,4:]=c+s
        c.rotary_call(original,q,cos,sin,positions=pos,cos_table=ct,sin_table=st,partial_slice=[4,8])
        self.assertTrue(torch.equal(q[...,:4],torch.ones_like(q[...,:4])))
        self.assertTrue(torch.equal(q[...,4:],(ct[pos+1]+st[pos+1]).reshape(2,1,1,4)))
        self.assertTrue(torch.equal(cos,before));self.assertEqual(len(calls),1);c.close()
    def test_projection_inf_consumed_and_restored(self):
        c=D2FixtureCalls('s');out=torch.ones(3,4,dtype=torch.bfloat16)
        c.projection_output(out,mechanism='39op');seen=c.consume(lambda x:x+1,out)
        self.assertTrue(torch.isposinf(seen[-1,0]));self.assertTrue(torch.equal(out,torch.ones_like(out)));c.close()
    def test_qmm_full_k_control_and_wrong_k_effect(self):
        c=D2FixtureCalls('s');q=torch.ones(2,8,dtype=torch.int8);w=torch.ones(8,3,dtype=torch.int8);s=torch.ones(3);ps=torch.ones(2)
        out=(q.float()@w.float()).bfloat16();c.quant_projection_output(out,q,w,s,ps)
        seen=c.consume(lambda x:x.clone(),out);self.assertEqual(float(seen[-1,0]),6);self.assertTrue(torch.equal(out,torch.full_like(out,8)))
        self.assertTrue(c.events[0]['full_range_control_passed']);c.close()
        with self.assertRaisesRegex(ValueError,'LOGICAL_LAYOUT'):c.quant_projection_output(out,q,w.reshape(2,4,3),s,ps)
    def test_sparse_sinks_causal_keys_consumer_restore(self):
        c=D2FixtureCalls('s');q,a=sparse_fixture.SparseTests().call();ids=['a','b']
        original=torch.stack([from_operator_call(q,a,row=i,request_ids=ids).recompute(block_size=2) for i in range(2)])
        out=original.clone();c.sparse_output(out,q,a,row=1,request_ids=ids,block_size=2)
        seen=c.consume(lambda x:x.clone(),out);self.assertTrue(torch.equal(seen[0],original[0]));self.assertFalse(torch.equal(seen[1],original[1]))
        self.assertTrue(torch.equal(out,original));self.assertFalse(c.events[0]['sink_replayed']);c.close()
    def test_actual_wrapper_preserves_rms_auxiliary_output(self):
        c=D2FixtureCalls('s');y=torch.ones(2,8,dtype=torch.bfloat16);rstd=torch.ones(2,1);calls=[]
        def original(*args,**kwargs):calls.append(args);return (y,rstd)
        result=c.invoke('18','npu::npu_rms_norm',original,(torch.zeros_like(y),),{},{});
        self.assertIs(result[1],rstd);self.assertEqual(len(calls),1)
        seen=c.consume(lambda x:x.clone(),result[0]);self.assertEqual(float(seen[-1,0]),.5);c.close()
    def test_schema_rejected_before_original_executes(self):
        c=D2FixtureCalls('s');calls=[]
        with self.assertRaisesRegex(ValueError,'OPERATOR_UNSUPPORTED'):
            c.invoke('28','unknown',lambda:calls.append(1),(),{}, {})
        self.assertEqual(calls,[])
    def test_sparse_actual_namespace_and_output_tuple(self):
        c=D2FixtureCalls('s');q,a=sparse_fixture.SparseTests().call();ids=['a','b']
        out=torch.stack([from_operator_call(q,a,row=i,request_ids=ids).recompute(block_size=2) for i in range(2)])
        before=out.clone();aux=torch.ones(1);calls=[]
        def original(*args,**kwargs):calls.append((args,kwargs));return out,aux
        result=c.invoke('28','cann_ops_transformer::sparse_flash_mla',original,(q,),a,
            dict(row=1,request_ids=ids,block_size=2,replay_blocks=(0,)))
        self.assertIs(result[1],aux);self.assertEqual(len(calls),1)
        seen=c.consume(lambda x:x.clone(),out)
        self.assertTrue(torch.equal(seen[0],before[0]));self.assertFalse(torch.equal(seen[1],before[1]));c.close()
        with self.assertRaisesRegex(ValueError,'OPERATOR_UNSUPPORTED'):
            c.invoke('28','_C_ascend::sparse_flash_mla',original,(q,),a,{})
    def test_wrong_consumer_never_completes_lease(self):
        c=D2FixtureCalls('s');out=torch.ones(2,3,dtype=torch.bfloat16);c.projection_output(out,mechanism='39op')
        with self.assertRaisesRegex(ValueError,'NOT_COVERED'):c.consume(lambda x:x.clone(),out.clone())
        with self.assertRaisesRegex(ValueError,'OUTSTANDING'):c.close()
        c.consume(lambda x:x.clone(),out);c.close()
    def test_off_no_region_modification(self):
        c=D2FixtureCalls('s',fault=False);out=torch.ones(2,3,dtype=torch.bfloat16)
        c.projection_output(out,mechanism='39op');self.assertTrue(torch.equal(c.consume(lambda x:x.clone(),out),out));c.close()
if __name__=='__main__':unittest.main()
