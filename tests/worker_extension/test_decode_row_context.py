"""Scheduler fixtures test mapping semantics; they are not serving acceptance."""
import unittest

from worker_extension.test_batch_context import batch_fixture, MetadataOnlyTensor
from demoncat.runtime.vllm_ascend.worker_extension.batch_context import (
    snapshot_first_decode, snapshot_first_decode_rows)
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import ContextRefused


def active_rows(count):
    runner,model,actual,context,expected=batch_fixture()
    ids=runner.input_batch.req_ids[:count]
    runner.input_batch.req_ids=ids
    runner.input_batch.num_reqs=count
    runner.input_batch.req_id_to_index={key:i for i,key in enumerate(ids)}
    runner.requests={key:runner.requests[key] for key in ids}
    expected={key:expected[key] for key in ids}
    runner.query_lens=[1]*count
    runner.query_start_loc.cpu=list(range(count+1))
    context.num_tokens=count
    actual['input_ids']=MetadataOnlyTensor(1,[count])
    actual['positions']=MetadataOnlyTensor(2,[count],'torch.int64')
    return runner,model,actual,context,expected


class DecodeRowContextTests(unittest.TestCase):
    def test_actual_active_count_is_not_scheduler_capacity(self):
        for n in (1,3,8):
            with self.subTest(active=n):
                result=snapshot_first_decode_rows(*active_rows(n))
                self.assertEqual((result['request_count'],result['token_row_count'],result['scheduler_capacity']),(n,n,8))
                self.assertEqual(result['row_request_indices'],list(range(n)))
        args=active_rows(1);args[0].vllm_config.scheduler_config.max_num_seqs=1
        self.assertEqual(snapshot_first_decode_rows(*args,expected_max_seqs=1)['request_count'],1)

    def test_original_quantization_reader_still_requires_b8(self):
        with self.assertRaisesRegex(ContextRefused,'ACTUAL_REQUEST_SET_MISMATCH'):
            snapshot_first_decode(*active_rows(3))

    def test_shape_rows_do_not_override_actual_scheduler(self):
        args=active_rows(3)
        args[2]['input_ids']=MetadataOnlyTensor(1,[8])
        with self.assertRaisesRegex(ContextRefused,'GEOMETRY_MISMATCH'):
            snapshot_first_decode_rows(*args)
        args=active_rows(3);args[0].query_lens=[1,2,1]
        with self.assertRaisesRegex(ContextRefused,'FIRST_PURE_DECODE'):
            snapshot_first_decode_rows(*args)
        args=active_rows(3);args[0].input_batch.req_id_to_index={k:2-i for i,k in enumerate(args[0].input_batch.req_ids)}
        with self.assertRaisesRegex(ContextRefused,'REQUEST_INDEX_MISMATCH'):
            snapshot_first_decode_rows(*args)

    def test_unreviewed_capacity_and_boolean_counts_refuse(self):
        args=active_rows(3)
        with self.assertRaisesRegex(ContextRefused,'UNREVIEWED_SEQUENCE_CAPACITY'):
            snapshot_first_decode_rows(*args,expected_max_seqs=16)
        args=active_rows(1);args[0].input_batch.num_reqs=True
        with self.assertRaises(ContextRefused):snapshot_first_decode_rows(*args)
        args=active_rows(1);args[0].query_lens=[True]
        with self.assertRaises(ContextRefused):snapshot_first_decode_rows(*args)


if __name__=='__main__':unittest.main()
