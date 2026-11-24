import unittest,torch
from tools.paged33_control_checkpoint import reviewed_partitions,control_per_head
from fault_injection.engines.paged33_native_control import split_fuse_control


class PerHeadTests(unittest.TestCase):
    def tensors(self):
        torch.manual_seed(33)
        return [torch.randn(*shape).bfloat16() for shape in [(1,4,8),(17,2,8),(17,2,8)]]
    def test_same_spans_preserve_existing_control_bits(self):
        q,k,v=self.tensors();spans=[[0,8],[8,17]]
        a=split_fuse_control(q,k,v,scale=.3,split_spans=spans)
        b=control_per_head(q,k,v,scale=.3,partitions=[spans,spans],matmul_mode='fp32')
        self.assertTrue(torch.equal(a.view(torch.int16),b.view(torch.int16)))
    def test_different_partitions_keep_correct_gqa_head_group(self):
        q,k,v=self.tensors();parts=[[[0,17]],[[0,5],[5,17]]]
        actual=control_per_head(q,k,v,scale=.3,partitions=parts,matmul_mode='fp32')
        for h in range(2):
            expected=split_fuse_control(q[:,h*2:h*2+2],k[:,h:h+1],v[:,h:h+1],scale=.3,split_spans=parts[h])
            self.assertTrue(torch.equal(actual[:,h*2:h*2+2].view(torch.int16),expected.view(torch.int16)))
    def test_missing_overlap_or_ambiguous_mapping_refused(self):
        for review in [{'per_kv_head_spans':[[[0,17]]]},
                       {'per_kv_head_spans':[[[0,17]],[[0,10],[9,17]]]},
                       {'spans':[[0,17]],'per_kv_head_spans':[[[0,17]],[[0,17]]]}]:
            with self.assertRaises(ValueError):reviewed_partitions(review,2,17)
