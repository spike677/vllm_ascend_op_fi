"""CPU metadata contracts only; these do not qualify a native Fault33 route."""
from types import SimpleNamespace as NS
import unittest

import torch

from demoncat.runtime.vllm_ascend.worker_extension.paged33_audit import audit_paged_call, _meta


def fixture():
    # Real vLLM can keep K and V in separate disjoint views of one allocation.
    packed = torch.arange(2 * 6 * 4 * 2 * 4, dtype=torch.float16).reshape(2, 6, 4, 2, 4)
    impl = NS(key_cache=packed[0], value_cache=packed[1], num_kv_heads=2,
              num_heads=4, head_size=4, sliding_window=None, sinks=None)
    table = torch.tensor([[0, 1], [3, 4]], dtype=torch.int32)
    slots = torch.tensor([5, 14], dtype=torch.int32)
    metadata = NS(block_tables=table, slot_mapping=slots, attn_state=NS(name='DecodeOnly'),
                  num_actual_tokens=2, seq_lens_list=[6, 3], actual_seq_lengths_q=[1, 2])
    actual = dict(query=torch.ones(2, 4, 4, dtype=torch.float16),
                  key=impl.key_cache.view(6, 4, -1), value=impl.value_cache.view(6, 4, -1),
                  block_table=table, input_layout='TND', block_size=4, sparse_mode=3,
                  actual_seq_lengths=[1, 2], actual_seq_lengths_kv=[6, 3])
    scheduler = dict(request_count=2, counts=[1, 1], token_row_count=2, row_token_positions=[5, 2],
                     request_ids=['actual-A', 'actual-B'])
    writer = dict(returned=True, key_cache=_meta(impl.key_cache), value_cache=_meta(impl.value_cache),
                  slots=[5, 14])
    return dict(actual=actual, impl=impl, metadata=metadata, scheduler=scheduler, writer=writer)


class PagedAuditTests(unittest.TestCase):
    def audit(self, fixture):
        return audit_paged_call('npu_fused_infer_attention_score', allow_cpu_test=True, **fixture)

    def test_actual_shared_allocation_disjoint_pages_is_read_only_and_unqualified(self):
        f = fixture()
        before_k, before_v = f['impl'].key_cache.clone(), f['impl'].value_cache.clone()
        row = self.audit(f)
        self.assertEqual(row['slots'], [5, 14])
        self.assertEqual(row['used_pages'], [[0, 1], [3]])
        self.assertEqual(row['metadata_binding'], 'VERIFIED')
        self.assertEqual(row['status'], 'OBSERVED_UNADAPTED_API')
        self.assertEqual(row['refusal'], 'OBSERVED_TND_FIA_REQUIRES_OWN_NATIVE_ADAPTER')
        self.assertFalse(row['public_worker_qualified'])
        self.assertFalse(row['cache_contents_verified'])
        self.assertTrue(torch.equal(before_k, f['impl'].key_cache))
        self.assertTrue(torch.equal(before_v, f['impl'].value_cache))

    def test_repacked_equal_cache_does_not_establish_live_alias(self):
        f = fixture();f['actual']['key'] = f['actual']['key'].clone()
        with self.assertRaisesRegex(RuntimeError, 'LIVE_CACHE_ALIAS_MISMATCH'):self.audit(f)

    def test_wrong_slot_page_and_duplicate_pages_refuse(self):
        for changed, reason in (('slot', 'CURRENT_TOKEN_SLOT'), ('range', 'PAGE_TABLE_OUT_OF_RANGE'),
                                ('duplicate', 'DUPLICATE_PAGE_WITHIN_SEQUENCE')):
            with self.subTest(changed=changed):
                f = fixture()
                if changed == 'slot':f['metadata'].slot_mapping[0] = 6
                if changed == 'range':f['metadata'].block_tables[0, 1] = 99
                if changed == 'duplicate':f['metadata'].block_tables[0, 1] = 0
                with self.assertRaisesRegex(RuntimeError, reason):self.audit(f)

    def test_writer_is_required_and_cannot_swap_key_value_offsets(self):
        f = fixture();f['writer'] = None
        with self.assertRaisesRegex(RuntimeError, 'ACTUAL_CACHE_WRITER_NOT_OBSERVED'):self.audit(f)
        f = fixture();f['writer']['value_cache'] = f['writer']['key_cache']
        with self.assertRaisesRegex(RuntimeError, 'WRITER_CONSUMER'):self.audit(f)

    def test_actual_api_lengths_and_table_identity_required(self):
        f = fixture();f['actual']['actual_seq_lengths_kv'] = [6, 4]
        with self.assertRaisesRegex(RuntimeError, 'ACTUAL_API_LENGTHS'):self.audit(f)
        f = fixture();f['actual']['block_table'] = f['actual']['block_table'].clone()
        with self.assertRaisesRegex(RuntimeError, 'ACTUAL_BLOCK_TABLE_NOT_METADATA'):self.audit(f)


if __name__ == '__main__':unittest.main()
