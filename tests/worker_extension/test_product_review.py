import unittest
from copy import deepcopy
from tools.review_product_mode import review


def receipt():
    workers=[dict(rank=i, pid=100+i) for i in range(2)]
    output=[dict(hidden='h'+str(i),logits='l'+str(i),finite=True) for i in range(4)]
    observation=dict(output=output,all_finite=True)
    scopes=[]
    for scope in range(2):
        requests=[]
        for index in range(4):
            rid=f'{scope}-{index}';ranks=[]
            for rank in range(2):
                active=index<3
                counts=dict(observed=int(active),eligible=int(active),attempted=int(active),applied=int(active),
                            changed=int(active),consumed=int(active),restored=int(active),failed=0)
                site=dict(eligible=True,consumer_operator='aten::add',consumed=True,restored=True,
                          before_sha256='b',restored_sha256='b')
                ranks.append(dict(rank=rank,request_id=rid,completed=True,coverage=counts,bytes_restored=True,
                    resources=dict(leases=0,hooks=0,inflight=0),status='OBSERVED' if active else 'NOT_INJECTED',
                    reasons=[] if active else ['GENERATION_OUTSIDE_VALIDATED_1_TO_4'],sites=[site] if active else [],
                    prompt_sha256=str(index),observed_model_steps=4,changed_elements=int(active)))
            requests.append(dict(result=dict(request_id=rid,ranks=ranks)))
        cleanup={a:[dict(status='DETACHED' if a=='detach' else 'ACTION_COMPLETED',worker=w) for w in workers]
                 for a in ('disable','unregister','detach')}
        for row in cleanup['unregister']:
            row['response']=dict(provider_state=dict(leases=0,owned_hooks=0,inflight=0,registered=False,
                replay_pending=False,recovery=dict(resources_zero=True),owner='owner-'+str(scope)+'-'+str(row['worker']['rank'])))
        for row in cleanup['detach']:row['ownership_retained']=False
        canary=dict(before=[dict(rank=r,observations=[deepcopy(observation),deepcopy(observation)]) for r in range(2)],
                    after=[dict(rank=r,observations=[deepcopy(observation)]) for r in range(2)])
        scopes.append(dict(scope=scope,requests=requests,disable=dict(canary=canary,cleanup=cleanup,
            recovery=dict(bytes_restored=True,resources_zero=True,canary_exact=True,restored=True)),
            detach=dict(status='PRODUCT_DETACHED'),cleanup=[dict(leases=0,owners=0)]*2))
    return dict(entry='VLLM_HTTP_API',host_per_request_hooks=False,status='HTTP_PRODUCT_TWO_SCOPES_PENDING_REVIEW',mechanism='18.api_half',
                source_manifest_sha256='test-source',scopes=scopes)


class ProductReviewTests(unittest.TestCase):
    def test_complete_contract_does_not_grant_yellow_or_product_closure(self):
        result=review(receipt())
        self.assertFalse(result['product_closed']);self.assertFalse(result['qualification_granted'])

    def test_canary_summary_true_cannot_hide_tensor_mismatch(self):
        data=receipt();data['scopes'][0]['disable']['canary']['after'][0]['observations'][0]['output'][2]['logits']='different'
        with self.assertRaisesRegex(ValueError,'CANARY_FULL_OUTPUT_NOT_EXACT'):review(data)

    def test_missing_consumer_and_nonrestored_region_fail(self):
        for field,value in (('consumer_operator',None),('restored_sha256','wrong')):
            data=receipt();data['scopes'][0]['requests'][0]['result']['ranks'][0]['sites'][0][field]=value
            with self.assertRaisesRegex(ValueError,'ACTUAL_CONSUMER_OR_BYTE_PROOF_MISSING'):review(data)

    def test_same_prompt_does_not_meet_product_acceptance(self):
        data=receipt()
        for row in data['scopes'][0]['requests'][:3]:
            for rank in row['result']['ranks']:rank['prompt_sha256']='same'
        with self.assertRaisesRegex(ValueError,'DISTINCT_USER_PROMPTS_REQUIRED'):review(data)

    def test_resource_summary_true_does_not_hide_runtime_leak(self):
        data=receipt()
        data['scopes'][0]['disable']['cleanup']['unregister'][0]['response']['provider_state']['owned_hooks']=1
        with self.assertRaisesRegex(ValueError,'RUNTIME_RESOURCE_COUNTERS_NOT_ZERO'):review(data)

    def test_two_receipts_cannot_reuse_one_scope_owner(self):
        data=receipt()
        data['scopes'][1]['disable']['cleanup']['unregister'][0]['response']['provider_state']['owner']='owner-0-0'
        with self.assertRaisesRegex(ValueError,'SCOPES_REUSE_RUNTIME_OWNER'):review(data)

    def test_host_marked_or_non_http_receipt_not_current_acceptance(self):
        for edit in (dict(entry='LLM'),dict(host_per_request_hooks=True)):
            with self.assertRaisesRegex(ValueError,'WORKER_AUTOMATIC_HTTP'):review(dict(receipt(),**edit))

    def test_missing_rank_and_worker_restart_fail(self):
        data=receipt();data['scopes'][0]['requests'][0]['result']['ranks'].pop()
        with self.assertRaisesRegex(ValueError,'EXACT_TP2_RANK_SET_REQUIRED'):review(data)
        data=receipt();data['scopes'][1]['disable']['cleanup']['unregister'][0]['worker']={'rank':0,'pid':999}
        with self.assertRaisesRegex(ValueError,'WORKERS_CHANGED_BETWEEN_SCOPES'):review(data)


    def test_matmul_or_paged_cannot_use_generic_consumer_as_control(self):
        for mechanism,code in [('39.matmul_range','PRODUCT_MATMUL_CONTROL_MISSING'),('33.paged_decode','PRODUCT_PAGED_CONTROL_MISSING')]:
            d=receipt();d['mechanism']=mechanism
            with self.assertRaisesRegex(ValueError,code):review(d)

    def test_paged_control_does_not_replace_coverage_counters(self):
        d=receipt();d['mechanism']='33.paged_decode'
        for scope in d['scopes']:
            for request in scope['requests'][:3]:
                for rank in request['result']['ranks']:
                    rank['sites'][0]['partition_reference']=dict(control_agreement=dict(elements=100,bitwise_equal=99),
                        same_legal_key_set=True,denominator_replay=False,replacement_policy='WHOLE_TARGET_ROW_NO_VALUE_SELECTION')
        result=review(d)
        self.assertEqual(result['scopes'][0]['effects'][0]['coverage'],d['scopes'][0]['requests'][0]['result']['ranks'][0]['coverage'])
        d['scopes'][0]['requests'][0]['result']['ranks'][0]['sites'][0]['partition_reference']['control_agreement']['bitwise_equal']=98
        with self.assertRaisesRegex(ValueError,'PRODUCT_PAGED_CONTROL_MISSING'):review(d)


class ExplicitWorldSizeReviewTests(unittest.TestCase):
    def test_tp1_is_explicit_and_cannot_borrow_tp2_record(self):
        data=receipt()
        for scope in data['scopes']:
            for req in scope['requests']:req['result']['ranks']=req['result']['ranks'][:1]
            for phase in ('before','after'):scope['disable']['canary'][phase]=scope['disable']['canary'][phase][:1]
            for action in ('disable','unregister','detach'):scope['disable']['cleanup'][action]=scope['disable']['cleanup'][action][:1]
            scope['cleanup']=scope['cleanup'][:1]
        self.assertEqual(review(data,expected_tp=1)['status'],'PRODUCT_BLUE_REVIEW_PASSED')
        with self.assertRaisesRegex(ValueError,'RANK_SET_REQUIRED'):review(data,expected_tp=2)
        with self.assertRaisesRegex(ValueError,'TP_UNSUPPORTED'):review(data,expected_tp=3)
