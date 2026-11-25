import copy,unittest
from reference_adapters.review_lm_head import review

def fixture():
    base=[dict(rank=r,all_finite=True,output=dict(logits_observations=[dict(
        hidden_sha256='a'*64,logits_sha256='b'*64,finite=True,tokens=[10,11,12],
        target_logits=[[1.,2.,3.]],target_probabilities=[[.1,.2,.3]],sampler_consumed=True,
        weight_readers=['npu::npu_linear']) for _ in range(4)])) for r in range(2)]
    fault=copy.deepcopy(base)
    for rank in fault:
        for row in rank['output']['logits_observations']:
            row.update(target_logits=[[1.,1.,1.]],target_probabilities=[[.2,.2,.2]])
    scopes=[]
    for n in range(2):
        owner='o'+str(n);states=[]
        for rank in range(2):
            events=[dict(observed=1,eligible=1,attempted=1,applied=1,consumed=1,restored=1,
                changed=1,changed_elements=10,failed=0,restore_kind='BYTE_EXACT_SAME_ALLOCATION_GENERATION')] if rank==0 else []
            states.append(dict(worker=dict(rank=rank,pid=100+rank),response=dict(provider_state=dict(owner=owner+str(rank),generation=n+1,
                row_adapter=dict(owns_rows=rank==0,nonowner_guard_reads=0 if rank==0 else 4),
                events=events,reader_observations=4 if rank==0 else 0,readers=[dict(request_id='cmpl-test'+str(n)+'-0-acde',weight_read=True) for _ in range(4)]))))
        for state in states:
            state['ticket']=dict(owner=owner,worker=state['worker'],generation=n+1)
        scopes.append(dict(owner=owner,enable=dict(ranks=copy.deepcopy(states)),status=dict(ranks=copy.deepcopy(states)),fault_http=dict(id='cmpl-test'+str(n),usage=dict(completion_tokens=4),choices=[dict(finish_reason='length')]),fault_hidden=copy.deepcopy(fault),range_negative=dict(exit_code=4),
            duplicate_enable=dict(exit_code=4),cleanup=[dict(rank=r,leases=0,owners=0) for r in range(2)],
            disable=dict(recovery=dict(bytes_restored=True,resources_zero=True,canary_exact=True,restored=True),
                canary=dict(before=copy.deepcopy(base),after=copy.deepcopy(base)),cleanup=dict(disable=states))))
    return dict(mode='persistent_row12_lm_head_collapse_example',
        status='TWO_OWNER_ROW12_LM_HEAD_COLLAPSE_EXAMPLE_COMPLETE_NOT_RELEASE_QUALIFICATION',
        missing_permit=dict(exit_code=4),scopes=scopes,observer_cleanup=[dict(rank=r,status='EXAMPLE_OBSERVER_REMOVED') for r in (0,1)])

class ReviewTests(unittest.TestCase):
    def test_complete(self):self.assertTrue(review(fixture())['passed'])
    def test_sampler_missing_rejected(self):
        d=fixture();d['scopes'][0]['fault_hidden'][0]['output']['logits_observations'][0]['sampler_consumed']=False
        self.assertIn('ACTUAL_SAMPLER_MISSING',review(d)['issues'])
    def test_changed_logits_without_named_rule_rejected(self):
        d=fixture();d['scopes'][0]['fault_hidden'][0]['output']['logits_observations'][0]['target_logits']=[[1.,1.,2.]]
        self.assertIn('TARGET_LOGITS_NOT_EXACT_NAMED_RULE',review(d)['issues'])
    def test_nonowner_write_rejected(self):
        d=fixture();states=d['scopes'][0]['disable']['cleanup']['disable']
        states[1]['response']['provider_state']['events']=states[0]['response']['provider_state']['events']
        self.assertIn('NONOWNER_GUARD',review(d)['issues'])
    def test_byte_recovery_is_not_a_boolean_only(self):
        d=fixture();d['scopes'][1]['disable']['canary']['after'][0]['output']['logits_observations'][0]['logits_sha256']='c'*64
        self.assertIn('CANARY_NOT_BYTE_EXACT',review(d)['issues'])

    def test_other_request_or_rank_is_rejected(self):
        d=fixture();d['scopes'][0]['disable']['cleanup']['disable'][1]['response']['provider_state']['readers'][0]['request_id']='another'
        self.assertIn('FOUR_ACTUAL_READERS_REQUIRED',review(d)['issues'])
        d=fixture();d['scopes'][0]['fault_http']['id']='cmpl-unrelated'
        self.assertIn('HTTP_INTERNAL_REQUEST_ASSOCIATION',review(d)['issues'])
    def test_observer_not_removed_is_rejected(self):
        d=fixture();d['observer_cleanup']=[]
        self.assertIn('OBSERVER_CLEANUP_REQUIRED',review(d)['issues'])

    def test_group_owner_and_rank_owner_are_distinct_but_chain_must_match(self):
        d=fixture();self.assertTrue(review(d)['passed'])
        d['scopes'][0]['status']['ranks'][1]['response']['provider_state']['owner']='wrong-rank-owner'
        self.assertIn('OWNER_CHAIN_MISMATCH',review(d)['issues'])
        d=fixture();d['scopes'][0]['disable']['cleanup']['disable'][0]['ticket']['owner']='other-group'
        self.assertIn('OWNER_MISMATCH',review(d)['issues'])
