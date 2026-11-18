"""Offline refusal contracts for complete live-KV range evidence."""
import copy
import unittest
from experiments.c2c3_postwrite_20261117.review41_group_receipts import valid_prefix


def fixture():
    geometry=dict(shape=[4,16,8,128],stride=[16384,1024,128,1],dtype='torch.bfloat16',
                  storage_identity=12,storage_offset=0)
    logical=dict(valid_tokens=17,elements_per_token=1024,
        **{k:'a'*64 for k in ('slot_sha256','query_sha256','active_key_sha256','active_value_sha256')},
        **{k:0 for k in ('slot_nonfinite','query_nonfinite','active_key_nonfinite','active_value_nonfinite')})
    return dict(input_before=logical,input_after=copy.deepcopy(logical),
        boundary=dict(slot=0,cache=copy.deepcopy(geometry),observational_metadata=dict(sequence_lengths=[17],used_pages=[[2,0]],
                      key=copy.deepcopy(geometry),value=copy.deepcopy(geometry))),
        valid_kv_range_proof=dict(read_only=True,selected_slot=0,bytes_per_cache=34816,
            storage=dict(key=copy.deepcopy(geometry),value=copy.deepcopy(geometry)),ranges=[
                dict(logical_token_start=0,physical_page=2,token_count=16,element_offset=32768,elements=16384),
                dict(logical_token_start=16,physical_page=0,token_count=1,element_offset=0,elements=1024)]))


class CompletePrefixReview(unittest.TestCase):
    def test_noncontiguous_physical_pages_cover_exact_logical_prefix(self):
        self.assertTrue(valid_prefix(fixture()))

    def test_actual_fia_merged_head_view_preserves_original_cache_storage(self):
        row=fixture()
        for name in ('key','value'):
            for item in (row['boundary']['observational_metadata'][name],row['valid_kv_range_proof']['storage'][name]):
                item.update(shape=[4,16,1024],stride=[16384,1024,1])
        self.assertTrue(valid_prefix(row))
        row['valid_kv_range_proof']['storage']['value']['stride']=[16384,1,16]
        self.assertFalse(valid_prefix(row))

    def test_missing_prefix_tail_is_not_full_restore(self):
        row=fixture();row['valid_kv_range_proof']['ranges'].pop()
        self.assertFalse(valid_prefix(row))

    def test_unused_tail_cannot_substitute_for_used_token(self):
        row=fixture();row['valid_kv_range_proof']['ranges'][1]['element_offset']=1024
        self.assertFalse(valid_prefix(row))

    def test_repeated_or_outside_pages_are_rejected(self):
        for pages in ([2,2],[2,4]):
            row=fixture();row['boundary']['observational_metadata']['used_pages']=[pages]
            self.assertFalse(valid_prefix(row))

    def test_same_hash_different_storage_does_not_prove_binding(self):
        row=fixture();row['valid_kv_range_proof']['storage']['value']['storage_identity']=13
        self.assertFalse(valid_prefix(row))

    def test_query_or_valid_value_update_not_restored_is_rejected(self):
        for key in ('query_sha256','active_value_sha256'):
            row=fixture();row['input_after'][key]='b'*64
            self.assertFalse(valid_prefix(row))

    def test_null_nonfinite_and_malformed_hash_are_not_equal(self):
        for key,value in (('active_key_sha256',None),('query_sha256','nan'),('slot_nonfinite',1)):
            row=fixture();row['input_before'][key]=row['input_after'][key]=value
            self.assertFalse(valid_prefix(row))

    def test_copy_budget_and_shape_remain_bounded(self):
        for field,value in (('bytes_per_cache',1048577),('read_only',False)):
            row=fixture();row['valid_kv_range_proof'][field]=value
            self.assertFalse(valid_prefix(row))


if __name__=='__main__':unittest.main()
