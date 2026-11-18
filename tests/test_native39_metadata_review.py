"""Saved-byte adversarial contracts; no native loader, Torch or NPU imports."""
import copy
import tempfile
import unittest
from pathlib import Path

from experiments.c2c3_closure_20261117 import review_native39_metadata_arrays as audit


def counter(instance, captures=0):
    result = {key: 0 for key in audit.COUNTERS}
    result.update(version=1, size=120, instance=instance, capture_calls=captures,
                  getter_calls=captures*3, acquired_arrays=captures*3, released_arrays=captures*3,
                  peak_live_arrays=3 if captures else 0, revision=captures*14)
    return result


def fixture(trigger='once'):
    keys = [('f39v1:aclnnMatmul:fixtureA', 100, 7), ('f39v1:aclnnMatmul:fixtureB', 200, 8)]
    witnesses = [dict(cache_key=list(key), registration_live=True,
                      metadata_arrays=dict(source='actual_typed_gws_capture',
                        before=counter(22, i*3), after=counter(22, (i+1)*3))) for i,key in enumerate(keys)]
    images = [dict(name=name, path='/native/'+name, sha256=str(i+1)*64,
                   symbol='fault39_metadata_arrays_v1', mapped_paths=['/native/'+name],
                   symbol_providers=['/native/'+name], state=counter(instance, captures))
              for i,(name,instance,captures) in enumerate((('libopapi.so',11,0),('libopapi_nn.so',22,6)))]
    arrays = dict(schema='demoncat.metadata_arrays.v1', source='ACTUAL_LOADED_PROXY_COUNTERS', pid=101,
                  build_profile_sha256='a'*64, build_manifest_sha256='b'*64, images=images,
                  scope='SDK_GETTER_ARRAYS_ONLY', reset_supported=False)
    activity = dict(schema='demoncat.metadata_array_activity.v1', pid=101, build_profile_sha256='a'*64,
                    group_observed_delta={k:0 for k in audit.DELTAS}, cached_registration_witnesses=witnesses,
                    process_lifetime_nonzero_alone_accepted=False)
    def native(key=None):
        result = dict(metadata_arrays=copy.deepcopy(arrays), metadata_array_activity=copy.deepcopy(activity),
                      inflight=0, leases=0, last_error=0, owner='', enabled=False,
                      independent_candidates=[dict(signature=w['cache_key'][0],cache_identities=[dict(
                        address=w['cache_key'][1],epoch=w['cache_key'][2],registration_live=True,
                        metadata_arrays=copy.deepcopy(w['metadata_arrays']))]) for w in witnesses])
        if key:
            result.update(target_signature=key[0], selected_cache_address=key[1], selected_cache_epoch=key[2])
        return result
    names = ['discovery_1','discovery_2','reference','lossless_self','lossless_mat2','lossless_dual','fault']
    if trigger=='every_match':
        names.append('fault_repeat')
    names += ['off_observed','off']
    phases = []
    for index,name in enumerate(names):
        calls=[]
        for i,key in enumerate((keys[0],keys[1],keys[0])):
            row = dict(site=['actual-object-site',i],native_delta=[dict(key=list(key),count=1)],original_calls=1)
            if name not in ('discovery_1','discovery_2','off'):
                row['native']=native(key)
            calls.append(row)
        calls.append(dict(site=['non-target'],native_delta=[],original_calls=1))
        phases.append(dict(phase=name,request_id='request'+str(index),calls=calls,native_denominator=3,
                           independent_denominator_complete=True,metadata_arrays=copy.deepcopy(arrays),
                           model_output_sha256='c'*64))
    receipt = dict(native_phases=phases,requests=[dict(snapshot=dict(internal_request_id=p['request_id'])) for p in phases],
                   qualification=dict(trigger=trigger,completed_phases=names,actual_request_ids=[p['request_id'] for p in phases],
                    metadata_array_activity=copy.deepcopy(activity),failure=None,recovery_verified=True))
    health = dict(native=native(),baseline='c'*64,off='c'*64,owned_hooks=0,worker=[101,'actual-start','worker',0,0])
    return dict(initial=copy.deepcopy(arrays),bound=copy.deepcopy(activity),health=health,receipt=receipt)


class Native39ArrayReviewTests(unittest.TestCase):
    def assert_refuses(self, evidence, reason):
        result=audit.review(**evidence)
        self.assertFalse(result['checks_passed'],result)
        self.assertFalse(result['qualifications_granted'])
        self.assertIn(reason, ' '.join(result['errors']))

    def test_cached_reuse_proves_each_key_without_group_counter_increment(self):
        result=audit.review(**fixture())
        self.assertTrue(result['checks_passed'],result)
        self.assertEqual(result['distinct_cache_keys'],2)
        self.assertEqual(result['actual_sites'],3)
        self.assertEqual(result['per_call_native_snapshots'],18)
        self.assertEqual(result['counter_snapshots'],29)
        self.assertTrue(all(n==0 for n in result['group_observed_delta'].values()))
        self.assertFalse(result['qualifications_granted'])

    def test_complete_every_match_phase_is_required(self):
        evidence=fixture('every_match')
        self.assertTrue(audit.review(**evidence)['checks_passed'])
        evidence['receipt']['native_phases'].pop(7)
        self.assert_refuses(evidence,'INCOMPLETE_OR_REORDERED_PHASES')

    def test_nonzero_process_total_without_registration_witness_is_insufficient(self):
        evidence=fixture();evidence['bound']['cached_registration_witnesses']=[]
        self.assert_refuses(evidence,'CACHE_WITNESSES_MISSING')

    def test_missing_actual_cache_cannot_shrink_witness_universe(self):
        evidence=fixture();evidence['bound']['cached_registration_witnesses'].pop()
        self.assert_refuses(evidence,'ACTUAL_KEY_WITNESS_COVERAGE_MISMATCH')

    def test_dead_registration_is_not_a_live_key(self):
        evidence=fixture();evidence['bound']['cached_registration_witnesses'][0]['registration_live']=False
        self.assert_refuses(evidence,'DUPLICATE_OR_DEAD_CACHE')

    def test_epoch_cannot_be_borrowed_from_same_cache_address(self):
        evidence=fixture();evidence['bound']['cached_registration_witnesses'][0]['cache_key'][2]+=1
        self.assert_refuses(evidence,'ACTUAL_KEY_WITNESS_COVERAGE_MISMATCH')

    def test_before_and_after_must_each_be_quiescent(self):
        evidence=fixture();evidence['bound']['cached_registration_witnesses'][0]['metadata_arrays']['before']['live_arrays']=1
        self.assert_refuses(evidence,'UNRELEASED_OR_ERROR')

    def test_positive_lifetime_totals_do_not_replace_exact_capture_delta(self):
        evidence=fixture();w=evidence['bound']['cached_registration_witnesses'][0]['metadata_arrays']
        w['before']=copy.deepcopy(w['after'])
        self.assert_refuses(evidence,'EXACT_THREE_TENSOR_CAPTURE_REQUIRED')

    def test_witness_must_join_actual_loaded_counter_instance(self):
        evidence=fixture();w=evidence['bound']['cached_registration_witnesses'][0]['metadata_arrays']
        w['before']['instance']=w['after']['instance']=999
        self.assert_refuses(evidence,'EXACT_THREE_TENSOR_CAPTURE_REQUIRED')

    def test_one_capture_cannot_be_copied_to_another_cache_key(self):
        evidence=fixture();rows=evidence['bound']['cached_registration_witnesses']
        rows[1]['metadata_arrays']=copy.deepcopy(rows[0]['metadata_arrays'])
        self.assert_refuses(evidence,'COPIED_OR_OVERLAPPING_CAPTURE_WITNESS')

    def test_counter_reset_in_an_intermediate_phase_is_detected(self):
        evidence=fixture();value=evidence['receipt']['native_phases'][3]['metadata_arrays']['images'][1]['state']
        value['revision']-=1
        self.assert_refuses(evidence,'COUNTER_RESET_OR_REORDER')

    def test_loaded_image_instance_swap_is_detected_even_when_totals_match(self):
        evidence=fixture();evidence['receipt']['native_phases'][2]['calls'][0]['native']['metadata_arrays']['images'][1]['state']['instance']+=1
        self.assert_refuses(evidence,'LOADED_INSTANCE_OR_BUILD_CHANGED')

    def test_missing_per_call_counters_is_not_filled_from_final_health(self):
        evidence=fixture();del evidence['receipt']['native_phases'][2]['calls'][0]['native']['metadata_arrays']
        self.assert_refuses(evidence,'MALFORMED_EVIDENCE')

    def test_actual_selected_cache_key_matches_independent_native_delta(self):
        evidence=fixture();evidence['receipt']['native_phases'][2]['calls'][0]['native']['selected_cache_epoch']+=1
        self.assert_refuses(evidence,'ACTUAL_SELECTED_CACHE_MISMATCH')

    def test_final_registry_cannot_hide_stale_epoch_behind_activity_flag(self):
        evidence=fixture();evidence['health']['native']['independent_candidates'][0]['cache_identities'][0]['epoch']+=1
        self.assert_refuses(evidence,'LIVE_CACHE_REGISTRY_OR_CAPTURE_CHANGED')

    def test_group_delta_is_recomputed_from_recorded_snapshots(self):
        evidence=fixture();evidence['bound']['group_observed_delta']['acquired_arrays']=1
        self.assert_refuses(evidence,'GROUP_DELTA_MISMATCH')

    def test_final_health_requires_no_owner_leases_errors_or_hooks(self):
        for field,value in (('owner','still-owned'),('leases',1),('last_error',7)):
            with self.subTest(field=field):
                evidence=fixture();evidence['health']['native'][field]=value
                self.assert_refuses(evidence,'FINAL_NATIVE_RESOURCE_OR_ERROR')

    def test_mid_phase_error_is_not_hidden_by_clean_final_health(self):
        evidence=fixture();evidence['receipt']['native_phases'][6]['calls'][1]['native']['metadata_arrays']['images'][1]['state']['ownership_errors']=1
        self.assert_refuses(evidence,'UNRELEASED_OR_ERROR')

    def test_request_mapping_is_not_inferred_from_phase_count(self):
        evidence=fixture();evidence['receipt']['requests'][1]['snapshot']['internal_request_id']='foreign'
        self.assert_refuses(evidence,'REQUEST_MAPPING_CHANGED')

    def test_unfinished_stage_has_no_completed_health_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(audit.Refusal,'ONE_COMPLETED_ARTIFACT_REQUIRED'):
                audit.review_stage(Path(folder))

    def test_c2_c3_link_requires_same_actual_loaded_worker_and_cache(self):
        before=audit.review(**fixture());after=audit.review(**fixture('every_match'))
        self.assertTrue(audit.link_reviews(before,after)['checks_passed'])
        after['worker_identity'][1]='new-process-start'
        self.assertIn('CROSS_GROUP_WORKER_OR_LOADED_INSTANCE_CHANGED',audit.link_reviews(before,after)['errors'])

    def test_c2_c3_link_catches_counter_reset_between_completed_groups(self):
        before=audit.review(**fixture());after=audit.review(**fixture('every_match'))
        after['initial_images']['libopapi_nn.so']['revision']-=1
        result=audit.link_reviews(before,after)
        self.assertFalse(result['checks_passed'])
        self.assertIn('COUNTER_RESET_OR_REORDER',' '.join(result['errors']))

    def test_c2_c3_link_cannot_replace_same_address_registration_epoch(self):
        before=audit.review(**fixture());after=audit.review(**fixture('every_match'))
        after['caches'][0]['cache_key'][2]+=1
        self.assertIn('CROSS_GROUP_CACHE_EPOCH_OR_WITNESS_CHANGED',audit.link_reviews(before,after)['errors'])

    def test_uncompleted_c2_cannot_be_used_for_recorded_pair_review(self):
        before=audit.review(**fixture());after=audit.review(**fixture('every_match'))
        before['checks_passed']=False
        self.assertIn('BOTH_COMPLETED_STAGES_REQUIRED',audit.link_reviews(before,after)['errors'])


if __name__=='__main__':
    unittest.main()
