import unittest
from test_continuous_public_review import fixture
from tools.review_continuous_public import review
class ReferenceReviewTests(unittest.TestCase):
    def test_reference_cannot_masquerade_as_public(self):
        r=fixture();r.update(acceptance_scope='DIRECTION_A_EXPERIMENT_PENDING_LAUNCH_REVIEW',reference_variant='A2_BF16_SPLITFUSE_512_TO_256',mechanism='33.paged_decode')
        with self.assertRaisesRegex(ValueError,'PUBLIC_RUNTIME'):review(r)
        with self.assertRaisesRegex(ValueError,'REFERENCE_RECEIPT'):review(r,reference_variant='allgather_output_row_proxy')
    def test_splitfuse_control_threshold_checked_from_counts(self):
        r=fixture();r.update(acceptance_scope='DIRECTION_A_EXPERIMENT_PENDING_LAUNCH_REVIEW',reference_variant='A2_BF16_SPLITFUSE_512_TO_256',mechanism='33.paged_decode')
        for scope in r['scopes']:
            for request in scope['requests']:
                for state in request['result']['response']['provider_state']['rank_states'].values():
                    for row in state['evidence']['actual_requests']:
                        pair=dict(same_legal_key_set=True,denominator_replay=False,nonselected_exact=True,key_visits=2305,page_mapping={'logical_keys':2305},control_agreement={'elements':3072,'bitwise_equal':3068},replacement_policy='WHOLE_TARGET_ROW_NO_VALUE_SELECTION',control_block=512,fault_block=256,matmul_mode='bf16_out_fp32',native_control_residual_elements=4,partition_changed_elements=2)
                        row['partition_effect']=dict(calls=4,selected_changed_elements=8,pairs=[pair]*4)
        good=review(r,reference_variant=r['reference_variant']);self.assertTrue(good['passed']);self.assertFalse(good['qualification_granted'])
        f=r['scopes'][1]['requests'][-1]['result']['response']['provider_state']['rank_states']['1']['evidence']['actual_requests'][1]
        pair=f['partition_effect']['pairs'].pop()
        with self.assertRaisesRegex(ValueError,'PAGED_PARTITION_EFFECT'):review(r,reference_variant=r['reference_variant'])
        f['partition_effect']['pairs'].append(pair)
        pair['partition_changed_elements']=0
        with self.assertRaisesRegex(ValueError,'SPLITFUSE_GROUPING_EFFECT'):review(r,reference_variant=r['reference_variant'])
        pair['partition_changed_elements']=2
        f['partition_effect']['pairs'][0]['control_agreement']['bitwise_equal']=3000
        with self.assertRaisesRegex(ValueError,'SPLITFUSE_CONTROL'):review(r,reference_variant=r['reference_variant'])
if __name__=='__main__':unittest.main()
