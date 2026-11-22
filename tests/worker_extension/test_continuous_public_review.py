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
    def test_native41_requires_build_bound_regions_and_exact_restoration(self):
        r=fixture();r['mechanism']='41.tail_column'
        for scope in r['scopes']:
            for host in scope['requests']:
                for state in host['result']['response']['provider_state']['rank_states'].values():
                    state['deployment_acceptance_use']['actual']['native_library_sha256']='a'*64
                    for row in state['evidence']['actual_requests']:
                        row['native_memory_effect']=dict(calls=4,library_sha256='a'*64,
                            writer_operators=['npu_scatter_pa_kv_cache'],consumer_operators=['npu_fused_infer_attention_score'],
                            regions=[dict(consumed=True,restored=True,slots=[1],page_mapping=[[0]],region_elements=128,
                                before_sha256='b'*64,restored_sha256='b'*64)]*4)
        self.assertTrue(review(r)['passed'])
        for key,val in [('calls',3),('library_sha256','c'*64),('writer_operators',[]),('regions',[])]:
            bad=deepcopy(r)
            effect=bad['scopes'][1]['requests'][-1]['result']['response']['provider_state']['rank_states']['1']['evidence']['actual_requests'][1]['native_memory_effect']
            effect[key]=val
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'KV41_'):review(bad)
        effect=r['scopes'][1]['requests'][-1]['result']['response']['provider_state']['rank_states']['1']['evidence']['actual_requests'][1]['native_memory_effect']
        effect['regions'][0]['restored_sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'KV41_REGION'):review(r)

    def test_vision_requires_real_uncached_encoder_each_phase_and_padding(self):
        r=fixture();r['mechanism']='17.spatial_padding'
        for scope in r['scopes']:
            for host in scope['requests']:
                for state in host['result']['response']['provider_state']['rank_states'].values():
                    for row in state['evidence']['actual_requests']:
                        fault=row['phase']=='fault'
                        row.pop('quant_reference')
                        row['coverage']={k:(0 if k=='failed' else 1) for k in row['coverage']}
                        row['restore_kind']='CALL_ARGUMENT_ONLY_NO_STORAGE_OVERWRITE'
                        row['vision_effect']=dict(encoder_calls=1,embedding_consumers=1,changed_outputs=int(fault),
                            actual_padding=[[0,7,7] if fault else [0,0,0]],consumer_operators=['aten.add.Tensor'])
        self.assertTrue(review(r)['passed'])
        for key,value in [('encoder_calls',0),('embedding_consumers',0),('actual_padding',[[0,0,0]]),
                          ('consumer_operators',[]),('changed_outputs',0)]:
            bad=deepcopy(r)
            row=bad['scopes'][1]['requests'][-1]['result']['response']['provider_state']['rank_states']['1']['evidence']['actual_requests'][1]
            row['vision_effect'][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'VISION17_'):review(bad)
        row=r['scopes'][1]['requests'][-1]['result']['response']['provider_state']['rank_states']['1']['evidence']['actual_requests'][2]
        row['vision_effect']['encoder_calls']=0
        with self.assertRaisesRegex(ValueError,'EACH_PHASE'):review(r)

    def test_range_and_paged_controls_cannot_be_replaced_by_model_effect(self):
        for mechanism in ('39.matmul_range','33.paged_decode'):
            r=fixture();r['mechanism']=mechanism
            for scope in r['scopes']:
                for host in scope['requests']:
                    for state in host['result']['response']['provider_state']['rank_states'].values():
                        for row in state['evidence']['actual_requests']:
                            row['actual_forward_calls']=[{'site':1} for _ in range(4)]
                            row['matmul_effect']=dict(reference_controls=4,rules=['omit_final_quarter'])
                            row['partition_effect']=dict(calls=4,selected_changed_elements=1,pairs=[dict(
                                same_legal_key_set=True,denominator_replay=False,nonselected_exact=True,
                                key_visits=259,page_mapping=dict(logical_keys=259))])
            self.assertTrue(review(r)['passed'])
            row=r['scopes'][1]['requests'][-1]['result']['response']['provider_state']['rank_states']['0']['evidence']['actual_requests'][1]
            if mechanism=='39.matmul_range':row['matmul_effect']['reference_controls']=3
            else:row['partition_effect']['pairs'][0]['denominator_replay']=True
            with self.subTest(mechanism=mechanism),self.assertRaises(ValueError):review(r)

    def test_moe_requires_current_routing_join_and_native_consumer(self):
        r=fixture();r['mechanism']='8.restore_offset'
        for scope in r['scopes']:
            for host in scope['requests']:
                for state in host['result']['response']['provider_state']['rank_states'].values():
                    for row in state['evidence']['actual_requests']:
                        row['actual_forward_calls']=[{'moe':1} for _ in range(4)]
                        row['moe_effect']=dict(actual_producer_joins=4,native_combine_calls=4,request_rows_verified=7)
        self.assertTrue(review(r)['passed'])
        for key in ('actual_producer_joins','native_combine_calls','request_rows_verified'):
            bad=deepcopy(r)
            bad['scopes'][1]['requests'][-1]['result']['response']['provider_state']['rank_states']['0']['evidence']['actual_requests'][1]['moe_effect'][key]=0
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'MOE_CURRENT_'):review(bad)

    def test_rotary_requires_changed_rows_and_aligned_qk_not_only_logits(self):
        r=fixture();r['mechanism']='22.wrong_row'
        for scope in r['scopes']:
            for host in scope['requests']:
                for state in host['result']['response']['provider_state']['rank_states'].values():
                    for row in state['evidence']['actual_requests']:
                        fault=row['phase']=='fault'
                        row['actual_forward_calls']=[{'rope':1} for _ in range(4)]
                        row['rotary_effect']=dict(rows_changed=4 if fault else 0,position_rows_changed=4 if fault else 0,
                            qk_output_fingerprints=[dict(site='rope',q='fault' if fault else 'normal',k='same') for _ in range(4)])
        self.assertTrue(review(r)['passed'])
        for field in ('rows','qk','site'):
            bad=deepcopy(r)
            effect=bad['scopes'][1]['requests'][-1]['result']['response']['provider_state']['rank_states']['0']['evidence']['actual_requests'][1]['rotary_effect']
            if field=='rows':effect['rows_changed']=0
            elif field=='qk':
                for item in effect['qk_output_fingerprints']:item['q']='normal'
            else:effect['qk_output_fingerprints'][0]['site']='different'
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'ROTARY_'):review(bad)

    def test_projection_nonfinite_fault_allowed_but_actual_calls_and_healthy_outputs_required(self):
        r=fixture();r['mechanism']='39.output_proxy'
        for scope in r['scopes']:
            for host in scope['requests']:
                for state in host['result']['response']['provider_state']['rank_states'].values():
                    for row in state['evidence']['actual_requests']:
                        if row['phase']=='fault':
                            row['actual_forward_calls']=[{'projection':1} for _ in range(4)]
                            row['all_finite']=False
                            for out in row['output']:out['finite']=False
        self.assertTrue(review(r)['passed'])
        row=r['scopes'][1]['requests'][-1]['result']['response']['provider_state']['rank_states']['0']['evidence']['actual_requests'][1]
        row['actual_forward_calls'][-1]={}
        with self.assertRaisesRegex(ValueError,'ACTUAL_CALL_COVERAGE'):review(r)
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
