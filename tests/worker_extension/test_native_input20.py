import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from fault_injection.sites.native_input20 import quantized_read
import test_native_regions


class InputTests(unittest.TestCase):
    def test_actual_quant_reads_overwritten_input_and_input_is_restored_before_qmm(self):
        native=test_native_regions.NativeRegionsTests().manager();x=torch.ones(2,128,dtype=torch.bfloat16)
        source=torch.full_like(x,4);before=x.clone();seen=[];event={}
        def quant(x):
            seen.append(x.clone());return x.to(torch.int8),torch.ones(x.shape[0])
        class Event:
            def record(self,s):pass
            def synchronize(self):pass
        with patch.object(torch,'npu',NS(Event=Event),create=True):
            q,scale=quantized_read(native,source,x,quant,(),{},event,fault=True)
        self.assertEqual(seen[0][-1,0].item(),4);self.assertEqual(seen[0][0,0].item(),1)
        self.assertTrue(torch.equal(x,before));self.assertTrue(torch.equal(source,torch.full_like(source,4)))
        self.assertEqual((q.float()@torch.ones(128,1))[-1,0].item(),320)
        self.assertEqual(native.count(),0);self.assertEqual(event['before_sha256'],event['restored_sha256'])

    def test_same_storage_and_different_rows_refuse_without_writing(self):
        native=test_native_regions.NativeRegionsTests().manager();x=torch.ones(2,128,dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError,'DISTINCT_LIVE_SOURCE'):
            quantized_read(native,x,x,lambda x:None,(),{},{},fault=True)
        with self.assertRaisesRegex(RuntimeError,'ROW_GEOMETRY'):
            quantized_read(native,x[:1],x,lambda x:None,(),{},{},fault=True)
        self.assertEqual(native.count(),0);self.assertEqual(x.sum().item(),256)
