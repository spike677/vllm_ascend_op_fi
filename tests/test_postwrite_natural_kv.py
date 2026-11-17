"""CPU-only contracts for live storage, actual slot geometry and consumer recovery."""
from types import SimpleNamespace as NS
import unittest

import torch

from fault_injection.deployments.postwrite import PostwriteRefused, _validate_span
from fault_injection.deployments.postwrite_kv import live_value_tail_plan, LiveKVSlotLease
from demoncat.runtime.vllm_ascend.worker_extension.paged33_audit import _meta
from tests.test_postwrite_providers import CpuLease


class CpuSpanTransport:
    transport_kind = 'cpu_test'
    def open_span(self, cache, offset, elements, owner, generation):
        _validate_span(cache, offset, elements)
        return CpuLease(cache.view(-1).narrow(0, offset, elements))


def fixture():
    # Mirrors disjoint K/V views of one allocation; not a natural-model proof.
    packed = torch.arange(2 * 6 * 4 * 2 * 4, dtype=torch.bfloat16).reshape(2, 6, 4, 2, 4)
    impl = NS(key_cache=packed[0], value_cache=packed[1], num_kv_heads=2,
              num_heads=4, head_size=4, sliding_window=None, sinks=None)
    table = torch.tensor([[0, 1]], dtype=torch.int32)
    slots = torch.tensor([5], dtype=torch.int32)
    metadata = NS(block_tables=table, slot_mapping=slots, attn_state=NS(name='DecodeOnly'),
                  num_actual_tokens=1, seq_lens_list=[6], actual_seq_lengths_q=[1])
    actual = dict(query=torch.ones(1, 4, 4, dtype=torch.bfloat16),
        key=impl.key_cache.view(6, 4, -1), value=impl.value_cache.view(6, 4, -1),
        block_table=table, input_layout='TND', block_size=4, sparse_mode=3,
        actual_seq_lengths=[1], actual_seq_lengths_kv=[6])
    scheduler = dict(request_count=1, counts=[1], token_row_count=1, row_token_positions=[5],
                     request_ids=['actual-A'])
    writer = dict(returned=True, key_cache=_meta(impl.key_cache), value_cache=_meta(impl.value_cache), slots=[5])
    return dict(operator='npu_fused_infer_attention_score', actual=actual, impl=impl,
                metadata=metadata, scheduler=scheduler, writer=writer, allow_cpu_test=True)


class NaturalKVSlotTests(unittest.TestCase):
    def test_live_view_offset_tail_columns_consumer_and_exact_restoration(self):
        f = fixture(); cache = f['impl'].value_cache
        before_k, before_v = f['impl'].key_cache.clone(), cache.clone()
        plan = live_value_tail_plan(**f, tail_elements=2)
        self.assertIs(plan.cache, cache)
        self.assertEqual(plan.element_offset, 40)
        self.assertEqual(plan.indices, (2, 3, 6, 7))
        self.assertEqual(plan.proof['allocation_element_indices'], [234, 235, 238, 239])
        lease = LiveKVSlotLease(CpuSpanTransport(), plan, 12, 1)
        lease.arm(fault=True)
        seen = []
        def actual_consumer():
            seen.append(cache.clone())
            return cache[1, 1].clone() + 1
        changed = lease.consume(actual_consumer, (), {})
        self.assertEqual(len(seen), 1)
        expected = before_v.clone()
        expected[1, 1, :, -2:] = -expected[1, 1, :, -2:]
        self.assertTrue(torch.equal(seen[0], expected))
        self.assertTrue(torch.equal(changed, expected[1, 1] + 1))
        self.assertTrue(torch.equal(f['impl'].key_cache, before_k))
        self.assertTrue(torch.equal(cache, before_v))
        self.assertTrue(lease.report['consumer_returned'])
        self.assertFalse(lease.live)
        with self.assertRaisesRegex(PostwriteRefused, 'ONCE_OWNER'):
            lease.consume(actual_consumer, (), {})

    def test_real_consumer_exception_restores_and_releases(self):
        f = fixture(); before = f['impl'].value_cache.clone()
        plan = live_value_tail_plan(**f)
        lease = LiveKVSlotLease(CpuSpanTransport(), plan, 12, 1)
        lease.arm(fault=True)
        def failing():raise ValueError('actual consumer failed')
        with self.assertRaisesRegex(ValueError, 'actual consumer'):
            lease.consume(failing, (), {})
        self.assertEqual(lease.report['original_calls'], 1)
        self.assertFalse(lease.report['consumer_returned'])
        self.assertFalse(lease.live)
        self.assertTrue(torch.equal(f['impl'].value_cache, before))

    def test_external_writer_keeps_lease_until_exact_bytes_restored(self):
        f = fixture(); plan = live_value_tail_plan(**f)
        lease = LiveKVSlotLease(CpuSpanTransport(), plan, 12, 1)
        lease.arm(fault=True)
        def overwrite():plan.cache.view(-1)[plan.element_offset] += 1
        with self.assertRaisesRegex(PostwriteRefused, 'EXTERNAL_WRITE'):
            lease.consume(overwrite, (), {})
        self.assertTrue(lease.live)
        self.assertFalse(lease.report['released'])

    def test_consumer_failure_is_preserved_when_recovery_also_fails(self):
        f=fixture();plan=live_value_tail_plan(**f)
        lease=LiveKVSlotLease(CpuSpanTransport(),plan,12,1);lease.arm(fault=True)
        def failing_consumer():
            plan.cache.view(-1)[plan.element_offset]+=1
            raise ValueError('primary actual consumer failure')
        with self.assertRaisesRegex(ValueError,'primary actual consumer'):
            lease.consume(failing_consumer,(),{})
        self.assertTrue(lease.live)
        self.assertIn('EXTERNAL_WRITE',lease.report['recovery_error'])
        self.assertIn('primary actual consumer',lease.report['consumer_error'])

    def test_unwritten_or_repacked_cache_and_wrong_slot_refused(self):
        for change, reason in (('writer', 'ACTUAL_CACHE_WRITER'), ('copy', 'LIVE_CACHE_ALIAS'),
                               ('slot', 'CURRENT_TOKEN_SLOT')):
            with self.subTest(change=change):
                f = fixture()
                if change == 'writer':f['writer'] = None
                elif change == 'copy':f['actual']['value'] = f['actual']['value'].clone()
                else:f['metadata'].slot_mapping[0] = 6
                with self.assertRaisesRegex(RuntimeError, reason):live_value_tail_plan(**f)

    def test_span_bounds_and_unreviewed_dtype_refused(self):
        f = fixture(); cache = f['impl'].value_cache
        for offset, count in ((-1, 4), (cache.numel() - 1, 2), (0, 0)):
            with self.assertRaisesRegex(PostwriteRefused, 'BOUNDED_SPAN'):
                _validate_span(cache, offset, count)
        with self.assertRaisesRegex(PostwriteRefused, 'OUTSIDE_ACTUAL_HEAD'):
            live_value_tail_plan(**f, tail_elements=5)

    def test_query_aliasing_mutated_cache_is_refused(self):
        f=fixture()
        f['actual']['query']=f['impl'].value_cache.view(-1)[:16].view(1,4,4)
        with self.assertRaisesRegex(PostwriteRefused,'QUERY_ALIASES_MUTATED_CACHE'):
            live_value_tail_plan(**f)


if __name__ == '__main__':unittest.main()
