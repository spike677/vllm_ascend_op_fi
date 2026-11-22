import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from fault_injection.sites.native_kv41 import KVMode
from test_native_regions import NativeRegionsTests
from demoncat.runtime.vllm_ascend.worker_extension.paged33_audit import audit_paged_call


class Op:
    def __init__(self,name,fn,fields):
        self._schema=NS(name='npu::'+name,arguments=[NS(name=k,default_value=None) for k in fields]);self.fn=fn
    def __call__(self,*args,**kwargs):return self.fn(*args,**kwargs)


class NativeKVTests(unittest.TestCase):
    def test_real_writer_mapping_native_span_consumer_and_exact_restore(self):
        native=NativeRegionsTests().manager();stream=native._stream()
        k=torch.zeros(3,4,1,8,dtype=torch.bfloat16);v=k.clone()
        impl=NS(key_cache=k,value_cache=v,num_kv_heads=1,head_size=8,num_heads=1,sliding_window=None,sinks=None)
        site=NS(fault=True,runner=NS(input_batch=NS(req_ids=['actual'])),writers={},native=native)
        meta=NS(causal=True,seq_lens_list=[3],actual_seq_lengths_q=[1],num_actual_tokens=1,
            slot_mapping=torch.tensor([6]),block_tables=torch.tensor([[1]]),attn_state=NS(name='DecodeOnly'))
        def writer(**a):
            k.view(-1,1,8)[6].copy_(a['key'][0]);v.view(-1,1,8)[6].copy_(a['value'][0])
        w=dict(key=torch.ones(1,1,8,dtype=torch.bfloat16),value=torch.full((1,1,8),2,dtype=torch.bfloat16),
            key_cache=k,value_cache=v,slot_mapping=meta.slot_mapping,cache_mode='Norm')
        a=dict(query=torch.ones(1,1,8,dtype=torch.bfloat16),key=k,value=v,block_table=meta.block_tables,
            block_size=4,sparse_mode=3,input_layout='TND',actual_seq_lengths=[1],actual_seq_lengths_kv=[3])
        e=dict(request_ids=['actual'],eligible=False,applied=False,changed=0,consumed=False,restored=False)
        class Event:
            def record(self,s):pass
            def synchronize(self):pass
        cpu_audit=lambda op,actual,**kw:audit_paged_call(op,actual,allow_cpu_test=True,**kw)
        with patch.object(torch,'npu',NS(current_stream=lambda:stream,Event=Event),create=True),patch(
                'demoncat.runtime.vllm_ascend.worker_extension.paged33_audit.audit_paged_call',cpu_audit):
            wm=KVMode(site,'layer',impl,'writer');wm.__torch_dispatch__(Op('npu_scatter_pa_kv_cache',writer,w),(),(),w)
            before=v.clone();cm=KVMode(site,'layer',impl,'consumer',meta,e)
            out=cm.__torch_dispatch__(Op('npu_fused_infer_attention_score',lambda **a:a['value'].sum(),a),(),(),a)
        self.assertEqual(out.item(),12);self.assertEqual(before.sum().item(),16)
        self.assertEqual(e['changed'],2);self.assertTrue(e['consumed'] and e['restored'])
        self.assertTrue(torch.equal(v,before));self.assertEqual(native.count(),0)
        self.assertEqual(e['before_sha256'],e['restored_sha256'])
