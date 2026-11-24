import copy,unittest
from tools.review_persistent_example import review,COUNTERS


def receipt():
    scopes=[]
    for owner in ('one','two'):
        stopped=[]
        for rank in (0,1):
            event={k:1 for k in COUNTERS};event.update(failed=0,changed_elements=48,
                restore_kind='BYTE_EXACT_SAME_ALLOCATION_GENERATION')
            stopped.append({'worker':{'rank':rank,'pid':20+rank},'response':{'provider_state':{
                'derived_copy':'original_object','events':[event],'reader_observations':4,
                'readers':[{'request_id':'actual','consumed_changed_scale':True,'device_read_complete':True} for _ in range(4)],
                'selected_site':'actual.scale'}}})
        scopes.append(dict(owner=owner,disable=dict(cleanup={'disable':stopped},
            recovery={k:True for k in ('bytes_restored','resources_zero','canary_exact','restored')},
            canary={'before':[{'rank':0,'all_finite':True,'output':{'hidden':'a'}},{'rank':1,'all_finite':True,'output':{'hidden':'b'}}],
                    'after':[{'rank':0,'all_finite':True,'output':{'hidden':'a'}},{'rank':1,'all_finite':True,'output':{'hidden':'b'}}]}),
            cleanup=[dict(rank=r,leases=0,owners=0) for r in (0,1)],detach={'status':'PRODUCT_DETACHED'}))
    return dict(status='TWO_OWNER_WEIGHT_EXAMPLE_COMPLETE_NOT_RELEASE_QUALIFICATION',missing_permit={'exit_code':4},scopes=scopes)


class PersistentReviewTests(unittest.TestCase):
    def test_valid_persistent_counts_do_not_count_readers_as_writes(self):
        r=review(receipt());self.assertTrue(r['passed']);self.assertFalse(r['qualification_granted'])
        self.assertEqual(r['ranks'][0]['reader_observations'],4)
        self.assertEqual(r['ranks'][0]['counters']['consumed'],1)
    def test_same_value_write_is_not_failure(self):
        d=receipt();s=d['scopes'][0]['disable']['cleanup']['disable'][0]['response']['provider_state']
        s['events'][0].update(changed=0,changed_elements=0)
        self.assertTrue(review(d)['passed'])
    def test_failed_rollback_cannot_pass_as_consumed_and_restored(self):
        d=receipt();e=d['scopes'][0]['disable']['cleanup']['disable'][0]['response']['provider_state']['events'][0]
        e.update(consumed=0,restore_kind='FAILED_ENABLE_ROLLBACK_NO_CONSUMER')
        self.assertIn('ROLLBACK_IS_NOT_ACCEPTANCE',review(d)['issues'])
    def test_changed_worker_missing_reader_and_canary_mismatch_fail(self):
        for change in ('worker','reader','canary','owner'):
            d=receipt();s=d['scopes'][1]
            if change=='worker':s['disable']['cleanup']['disable'][0]['worker']['pid']=99
            if change=='reader':s['disable']['cleanup']['disable'][0]['response']['provider_state']['readers']=[]
            if change=='canary':s['disable']['canary']['after']=[]
            if change=='owner':s['owner']='one'
            self.assertFalse(review(d)['passed'],change)


if __name__=='__main__':unittest.main()
