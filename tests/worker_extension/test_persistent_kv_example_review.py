import copy,unittest
from test_persistent_example_review import receipt
from tools.review_persistent_kv_example import review


def kv_receipt():
    d=receipt();d['status']='TWO_OWNER_KV_VALUE_CELL_EXAMPLE_COMPLETE_NOT_RELEASE_QUALIFICATION'
    for scope in d['scopes']:
        scope['fault_hidden_changed']=True
        for row in scope['disable']['cleanup']['disable']:
            s=row['response']['provider_state'];s['persistent_leases']=0;s['selected_sites']=[str(n) for n in range(16)]
            event=s['events'][0];s['events']=[dict(event,name=n,restore_kind='BYTE_EXACT_SAME_REQUEST_GENERATION_BEFORE_PAGE_FREE') for n in s['selected_sites']]
            s['readers']=[dict(site=n,request_id='real',consumer='npu::npu_fused_infer_attention_score',changed_cache_value_read=True,device_read_complete=True) for n in s['selected_sites'] for _ in range(3)]
            s['reader_observations']=48
    return d


class KVReviewTests(unittest.TestCase):
    def test_complete_does_not_grant(self):
        d=kv_receipt();before=copy.deepcopy(d);r=review(d)
        self.assertTrue(r['passed'],r);self.assertFalse(r['qualification_granted']);self.assertEqual(d,before)
    def test_missing_cache_consumer_or_page_restore_refused(self):
        for key,value in [('readers',[]),('persistent_leases',1),('selected_sites',[])]:
            d=kv_receipt();d['scopes'][0]['disable']['cleanup']['disable'][0]['response']['provider_state'][key]=value
            self.assertFalse(review(d)['passed'])
    def test_readonly_or_same_output_is_not_injection_acceptance(self):
        d=kv_receipt();d['scopes'][1]['fault_hidden_changed']=False;self.assertFalse(review(d)['passed'])
        d=kv_receipt();d['status']='CENSUS_COMPLETE_NOT_PRODUCT_ACCEPTANCE';self.assertFalse(review(d)['passed'])
    def negative_receipt(self):
        d=kv_receipt()
        for s in d['scopes']:
            s['range_negative']={'exit_code':4};s['page_audit']=[dict(rank=r,tickets=0,leases=0,events=[{'sequence':1}]) for r in (0,1)]
            for r in s['disable']['cleanup']['disable']:
                r['response']['ticket']=dict(owner=s['owner'],worker=r['worker'])
                r['response']['provider_state'].update(owner=s['owner']+str(r['worker']['rank']),generation=1)
            s['enable']=dict(ranks=copy.deepcopy(s['disable']['cleanup']['disable']))
        payload=dict(request_id='http-0',request_status='FINISHED_ABORTED',sequence=2,request_generation='g')
        audit=[dict(rank=r,tickets=0,leases=0,events=[dict(payload,action='before_free')],
                    restore_proofs=[dict(payload=copy.deepcopy(payload),result=dict(bytes_restored=True,request_leases=0,read_only_no_injection=False))]) for r in (0,1)]
        d['cancel_scope']=dict(http_id='http',actual_abort_proven=True,audit=audit,
            disable=copy.deepcopy(d['scopes'][0]['disable']),detach={'status':'PRODUCT_DETACHED'})
        return d
    def test_abort_requires_actual_scheduler_proof(self):
        d=self.negative_receipt();self.assertTrue(review(d,require_negatives=True)['passed'])
        d['cancel_scope']['audit'][1]['events'][0]['request_status']='FINISHED_LENGTH_CAPPED'
        self.assertFalse(review(d,require_negatives=True)['passed'])
    def test_abort_flag_without_restore_and_live_rank_fails(self):
        d=self.negative_receipt();d['cancel_scope']['audit'][1]['restore_proofs']=[]
        self.assertFalse(review(d,require_negatives=True)['passed'])
        d=self.negative_receipt();d['scopes'][1]['page_audit'][1]['leases']=1
        self.assertFalse(review(d,require_negatives=True)['passed'])
    def test_stale_rank_owner_reply_refused(self):
        d=self.negative_receipt();d['scopes'][1]['enable']['ranks'][1]['response']['provider_state']['owner']='stale'
        self.assertIn('KV_OWNER_RANK_CHAIN_MISMATCH',review(d,require_negatives=True)['issues'])
