from copy import deepcopy
import unittest
from tools.review_continuous_public import review


def fixture():
    report=dict(acceptance_scope='PUBLIC_WORKER_RUNTIME',mechanism='28.fia.denominator_replay',scopes=[],warmup=[])
    for index in range(2):
        group=dict(owner='owner'+str(index),tp=2,members=[dict(rank=r,pid=r+10,model_instance_id='model'+str(r)) for r in range(2)],
            injection_policy='ALL_RANK_EVERY_ELIGIBLE_CALL')
        scope=dict(owner=group['owner'],prepare=dict(status='ACTION_COMPLETED',group=group),
            register=dict(status='ACTION_COMPLETED'),check=dict(status='ACTION_COMPLETED'),enable=dict(status='ACTION_COMPLETED'),
            finish=dict(status='ACTION_COMPLETED'),detach=dict(status='DETACHED'),snapshot=dict(ownership_retained=False),
            negative_extra_advance='GROUP_PHASES_EXHAUSTED',negative_missing_permit=[dict(can_prepare=False)]*2,
            fault_time_ratio=2,requests=[])
        rows=[]
        for phase in ('normal','fault','off','health'):
            rid=str(index)+phase;changed=phase=='fault'
            rows.append(dict(phase=phase,request_ids=[rid],output=[dict(hidden='f' if changed else 'b',logits='b',finite=True)]*4,
                all_finite=True,resources=dict(leases=0,hooks=0,inflight=0),
                coverage=dict(observed=4,eligible=4,attempted=4,applied=4,changed=4,consumed=4,restored=4,failed=0),
                site_count=1,quant_reference={},events_sha256='a'*64,same_stream_verified=True,bytes_restored=True,
                observation_complete=True,uncovered_sites=[]))
            state=dict(deployment_acceptance_use=dict(scope_valid=True,registration_source='BLUE_COMBINATION',actual={}),
                complete=phase=='health',next_phase='complete' if phase=='health' else phase,
                evidence=dict(actual_requests=deepcopy(rows)))
            scope['requests'].append(dict(phase=phase,mapping=[dict(internal_request_id=rid)],result=dict(status='QUERY_OK',
                response=dict(provider_state=dict(rank_states={str(r):deepcopy(state) for r in range(2)},request_plan={})))))
        clean={k:0 for k in ('leases','inflight','owned_hooks','registered','native_registered','enabled')}
        scope['unregister']=dict(status='ACTION_COMPLETED',response=dict(provider_state=dict(rank_states={str(r):deepcopy(clean) for r in range(2)})))
        report['scopes'].append(scope)
        for attempt in range(2):report['warmup'].append(dict(scope=index,ranks=[dict(warmup_valid=True,observations=[dict(all_finite=True,output=['b'])]*2)]*2))
    return report


class ReviewTests(unittest.TestCase):
    def test_release_review_requires_actual_creation_refusal_and_cleanup(self):
        r=fixture()
        with self.assertRaisesRegex(ValueError,'NEGATIVE_CREATION'):
            review(r,require_negative_creation=True)
        for s in r['scopes']:
            s['negative_missing_permit_creation']=[dict(rank=i,prepare=dict(status='PROVIDER_UNAVAILABLE'),
                snapshot=dict(ticket=None,ownership_retained=False),detach=dict(status='DETACHED')) for i in range(2)]
            s['cleanup']=[dict(rank=i,permit_remaining=False,runtime_leases=0,runtime_tenants=0) for i in range(2)]
        self.assertTrue(review(r,require_negative_creation=True)['negative_creation_verified'])
        r['scopes'][1]['cleanup'][0]['permit_remaining']=True
        with self.assertRaisesRegex(ValueError,'RESOURCES_REMAIN'):review(r,require_negative_creation=True)
    def test_pass_is_not_deployment_grant(self):
        r=review(fixture());self.assertTrue(r['passed']);self.assertFalse(r['qualification_granted'])
    def test_coverage_restore_mapping_and_warmup_are_independently_checked(self):
        for what in ('coverage','restore','mapping','warmup','members','scope'):
            r=fixture();s=r['scopes'][1]
            row=s['requests'][-1]['result']['response']['provider_state']['rank_states']['1']['evidence']['actual_requests'][1]
            if what=='coverage':row['coverage']['consumed']=3
            elif what=='restore':row['bytes_restored']=False
            elif what=='mapping':row['request_ids']=['foreign']
            elif what=='warmup':r['warmup'][-1]['ranks'][0]['observations'][-1]['all_finite']=False
            elif what=='members':s['prepare']['group']['members'][1]['pid']=999
            else:s['owner']='same-as-other'
            with self.subTest(what=what),self.assertRaises(ValueError):review(r)
