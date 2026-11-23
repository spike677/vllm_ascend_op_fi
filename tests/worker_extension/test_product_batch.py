import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
import torch
from test_batch_quant_site import BatchSiteTests,Model
from demoncat.runtime.vllm_ascend.worker_extension.product_batch import AutomaticBatches,classify_batch,summarize_events

class ProductModel(Model):
    def forward(self,input_ids,inputs_embeds,positions,intermediate_tensors):
        x=torch.stack((input_ids.float()+1,input_ids.float()*.13+1),dim=1)
        return self.quant_method.quant_method.apply(self,x)

class Runner:
    def __init__(self,model):
        self.model=model;self.requests={};self.vllm_config=NS(scheduler_config=NS(max_num_seqs=8))
        self.input_batch=NS(req_ids=[],req_id_to_index={},num_computed_tokens_cpu=[]);self.steps={}
    def _preprocess(self,s,total,intermediate=None):
        ids=self.input_batch.req_ids;values=[];positions=[]
        for rid in ids:
            p=self.requests[rid].prompt_token_ids;c=s.num_scheduled_tokens[rid];computed=self.input_batch.num_computed_tokens_cpu[self.input_batch.req_id_to_index[rid]]
            values.extend(p if computed==0 else [p[-1]+computed]);positions.extend(range(computed,computed+c))
        return torch.tensor(values),None,torch.tensor(positions),None,{},None
    def execute_model(self,s,intermediate=None):
        for n in s.scheduled_new_reqs:
            self.requests[n.req_id]=NS(prompt_token_ids=n.prompt_token_ids,sampling_params=NS(max_tokens=n.max_tokens),mm_features=[],lora_request=None,prompt_embeds=None,external_req_id='http-'+n.req_id)
            self.steps[n.req_id]=0
        ids=list(s.num_scheduled_tokens);self.current=s
        self.input_batch=NS(req_ids=ids,req_id_to_index={r:i for i,r in enumerate(ids)},num_computed_tokens_cpu=[0 if self.steps[r]==0 else len(self.requests[r].prompt_token_ids)+self.steps[r]-1 for r in ids])
        values=self._preprocess(s,s.total_num_scheduled_tokens,intermediate)
        out=self.model(**dict(zip(('input_ids','inputs_embeds','positions','intermediate_tensors'),values[:4])))
        ends=[];off=0
        for r in ids:off+=s.num_scheduled_tokens[r];ends.append(off-1)
        self.execute_model_state=NS(scheduler_output=s,hidden_states=out,logits=out[ends])
    def sample_tokens(self):
        ids=list(self.input_batch.req_ids)
        for r in ids:self.steps[r]+=1
        self.execute_model_state=None
        return NS(req_ids=ids)

class ProductBatchTests(BatchSiteTests):
    def make(self,canary=False):
        model=ProductModel();r=Runner(model)
        binding=NS(worker=NS(model_runner=r),model=model,mechanism='quant.batch_shared_scale',check=Mock(),prompts=((1,2),(10,20)),identity={'rank':0},group={'owner':'cpu','group_generation':1},arithmetic_device='cpu',reference_mode='immediate',warmups=[])
        self.enterContext(patch('fault_injection.sites.batch_quant.BatchQuantSite.actual_rows',lambda site:site.runner.current.total_num_scheduled_tokens))
        m=AutomaticBatches(binding,canary=canary);m.install();return m,r,binding
    def step(self,r,ids,*,new=None):
        new=new or {};rows=[NS(req_id=k,prompt_token_ids=v,max_tokens=4) for k,v in new.items()]
        counts={k:len(new[k]) if k in new else 1 for k in ids}
        s=NS(num_scheduled_tokens=counts,total_num_scheduled_tokens=sum(counts.values()),scheduled_new_reqs=rows,finished_req_ids=set(),scheduled_spec_decode_tokens={},scheduled_encoder_inputs={})
        r.execute_model(s);r.sample_tokens()
    def test_staggered_real_batches_only_decode_b2_injects(self):
        m,r,b=self.make()
        self.step(r,['a'],new={'a':[1,2]});self.step(r,['a'])
        self.step(r,['a','b'],new={'b':[10,20]})
        self.step(r,['a','b']);self.step(r,['b']);self.step(r,['b'])
        self.assertEqual(len(m.records),2)
        for row in m.records:
            self.assertEqual(row['coverage']['eligible'],1);self.assertEqual(row['coverage']['restored'],1)
            self.assertEqual(row['observed_model_steps'],4)
            self.assertEqual([f['request_ids'] for f in row['frames'] if f['eligible']],[['a','b']])
        self.assertEqual(sum(e['eligible'] for e in m.site.events),1) # participation != unique calls
        m.close();m.close();self.assertEqual(m.snapshot()['owned_hooks'],0)
    def test_single_request_completes_without_injection(self):
        m,r,b=self.make()
        for i in range(4):self.step(r,['solo'],new={'solo':[1,2]} if i==0 else None)
        row=m.records[0];self.assertEqual(row['status'],'NOT_INJECTED');self.assertIn('DECODE_BATCH_SIZE_ONE',row['reasons'])
        self.assertEqual(row['coverage']['applied'],0);m.close()
    def test_canary_observes_batch_without_fault(self):
        m,r,b=self.make(canary=True)
        for i in range(4):self.step(r,['a','b'],new={'a':[1,2],'b':[10,20]} if i==0 else None)
        self.assertTrue(m.completed_canary);self.assertTrue(b.warmups[0]['all_finite'])
        self.assertEqual(len(b.warmups[0]['output']),2)
        self.assertTrue(all(not e['applied'] for e in m.site.events));m.close()
    def test_scope_disable_is_not_user_completion(self):
        m,r,b=self.make();self.step(r,['a','b'],new={'a':[1,2],'b':[10,20]});self.step(r,['a','b'])
        m.close();self.assertTrue(all(not row['completed'] and row['bytes_restored'] for row in m.records))
    def test_mapping_mixed_and_unsupported_policy(self):
        rows=[dict(id='a',computed=10,prompt_length=10,reasons=[]),dict(id='b',computed=20,prompt_length=20,reasons=[])]
        self.assertTrue(classify_batch(rows,{'a':1,'b':1})[0])
        self.assertFalse(classify_batch(rows,{'a':1})[0])
        rows[1]['computed']=0;self.assertEqual(classify_batch(rows,{'a':1,'b':20})[1],'PREFILL_OR_MIXED_BATCH_NOT_INJECTED')
        rows[1]['reasons']=['unsupported'];self.assertFalse(classify_batch(rows,{'a':1,'b':1})[0])
    def test_no_summary_can_hide_eligible_restore_failure(self):
        with self.assertRaises((RuntimeError,KeyError)):summarize_events([dict(site='q',eligible=True,applied=True,consumed=True,restored=False,failed=False)])


    def test_independent_review_deduplicates_actual_batch_participation(self):
        from copy import deepcopy
        from tools.review_product_mode import review_batch_requests
        m,r,b=self.make()
        for i in range(4):self.step(r,['a','b'],new={'a':[1,2],'b':[10,20]} if i==0 else None)
        for rid in ['solo','solo2']:
            for i in range(4):self.step(r,[rid],new={rid:[4,5]} if i==0 else None)
        rows=[]
        for rec in m.records:
            peer=deepcopy(rec);peer['rank']=1
            rows.append(dict(result=dict(request_id=rec['request_id'],ranks=[rec,peer]),expected_injection=rec['request_id'] in ('a','b'),scenario='concurrent' if rec['request_id'] in ('a','b') else 'single'))
        def need(ok,why):
            if not ok:raise ValueError(why)
        out=review_batch_requests(rows,need)
        self.assertEqual(out['unique_batch_calls_by_rank'][0]['eligible'],3)
        self.assertEqual(sum(x['coverage']['eligible'] for x in out['request_participation'] if x['rank']==0),6)
        altered=deepcopy(rows);altered[0]['result']['ranks'][0]['frames'][1]['scheduled_counts']['b']=2
        with self.assertRaisesRegex(ValueError,'REAL_MULTI_REQUEST_DECODE'):review_batch_requests(altered,need)
        altered=deepcopy(rows);altered[0]['result']['ranks'][0]['sites'][0]['reference_passed']=0
        with self.assertRaisesRegex(ValueError,'QMM_REFERENCE'):review_batch_requests(altered,need)
        m.close()
