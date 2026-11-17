"""Actual tensor/storage CPU fixtures for the separate TND native adapter."""
from types import SimpleNamespace as NS
import unittest

import torch
from demoncat.runtime.vllm_ascend.worker_extension.paged33_audit import audit_paged_call, _meta
from demoncat.runtime.vllm_ascend.worker_extension.paged33_tnd import validate_tnd_call, validate_fai_launch, Paged33TNDProvider


def fixture():
    packed=torch.zeros((2,70,16,8,128),dtype=torch.float16)
    impl=NS(key_cache=packed[0],value_cache=packed[1],num_kv_heads=8,num_heads=16,head_size=128,
            sliding_window=None,sinks=None)
    table=torch.arange(65,dtype=torch.int32).reshape(1,65)
    metadata=NS(block_tables=table,slot_mapping=torch.tensor([1024],dtype=torch.int32),
                attn_state=NS(name='DecodeOnly'),num_actual_tokens=1,seq_lens_list=[1025],actual_seq_lengths_q=[1])
    actual=dict(query=torch.ones(1,16,128,dtype=torch.float16),key=packed[0].view(70,16,1024),
        value=packed[1].view(70,16,1024),block_table=table,input_layout='TND',block_size=16,sparse_mode=3,
        actual_seq_lengths=[1],actual_seq_lengths_kv=[1025],num_heads=16,num_key_value_heads=8,scale=128**-.5)
    scheduler=dict(request_count=1,counts=[1],token_row_count=1,row_token_positions=[1024],request_ids=['real-A'])
    writer=dict(returned=True,key_cache=_meta(impl.key_cache),value_cache=_meta(impl.value_cache),slots=[1024])
    boundary=audit_paged_call('npu_fused_infer_attention_score',actual,impl=impl,metadata=metadata,
                              scheduler=scheduler,writer=writer,allow_cpu_test=True)
    return actual,boundary


class TNDNativeContracts(unittest.TestCase):
    def test_final_fai_workspace_and_compiled_entry_must_agree(self):
        for arm in ('baseline','fault','off'):
            for mask in (0,1):
                tile=256 if arm=='fault' else 512;unit=16*128*tile*3
                typed=dict(numHeads=16,kvHeads=8,batch=1,embeddingSize=128,embeddingSizeV=128,
                    blockSize=16,maxQSeqlen=1,maxKvSeqlen=1025,sparseMode=3,firstBatchTaskNum=8,
                    totalTaskNum=8,splitLseTotalSize=0,splitOTotalSize=0,totalSplitNodeNum=0,maskType=mask)
                typed.update({name:unit*width for name,width in dict(mm1OutSize=4,smOnlineOutSize=2,
                    mm2OutSize=4,UpdateSize=4,workSpaceSize=14).items()})
                launch=dict(typed=typed,tiling_key=5000000000010200100+(3 if mask else 0)+
                    (900000000000000000 if arm=='fault' else 0),block_dim=16,schedule_mode=1)
                host=dict(route='FAInferTiling',aic=16,native_route_or_decode_flags_changed=False,
                    workspace=16*1024**2+16*128*tile*4*3*4)
                self.assertEqual(validate_fai_launch(launch,host,arm=arm,kv_length=1025)['selected_s2'],tile)
                typed['mm2OutSize']+=1
                with self.assertRaisesRegex(RuntimeError,'WORKSPACE_DIFFERS'):
                    validate_fai_launch(launch,host,arm=arm,kv_length=1025)
                typed['mm2OutSize']-=1
                launch['tiling_key']=103000000010200003
                with self.assertRaisesRegex(RuntimeError,'ENTRY_OR_LAUNCH_DIFFERS'):
                    validate_fai_launch(launch,host,arm=arm,kv_length=1025)

    def test_live_flat_page_view_is_retained_and_not_repacked(self):
        actual,boundary=fixture()
        before=actual['key'].untyped_storage()._cdata
        report=validate_tnd_call(actual,boundary,allow_cpu_test=True)
        self.assertEqual(report['page_size'],16)
        self.assertEqual(report['fault_s2'],256)
        self.assertEqual(actual['key'].untyped_storage()._cdata,before)

    def test_equal_copy_does_not_bind_to_natural_cache(self):
        actual,boundary=fixture();actual['key']=actual['key'].clone()
        with self.assertRaisesRegex(RuntimeError,'STORAGE_CHANGED'):
            validate_tnd_call(actual,boundary,allow_cpu_test=True)

    def test_old_bnsd_page128_and_short_audit_do_not_qualify(self):
        for field,value in (('input_layout','BNSD'),('block_size',128),('actual_seq_lengths_kv',[15]),
                            ('inner_precise',1),('scale',1),('num_heads',8),('antiquant_scale',torch.ones(1))):
            with self.subTest(field=field):
                actual,boundary=fixture();actual[field]=value
                with self.assertRaises(RuntimeError):validate_tnd_call(actual,boundary,allow_cpu_test=True)

    def test_cpu_metadata_cannot_authorize_real_native_execution(self):
        actual,boundary=fixture()
        with self.assertRaisesRegex(RuntimeError,'AUDIT_SCOPE_DIFFERS'):
            validate_tnd_call(actual,boundary)

    def test_provider_preserves_actual_arguments_and_return_identity(self):
        result=object();query,key,value=object(),object(),object();table=object()
        seen=[]
        def original(*args,**kwargs):
            seen.append((args,kwargs));return result
        class Adapter:
            def execute(self,actual,boundary,invoke):
                self.actual=actual
                return invoke()
        provider=object.__new__(Paged33TNDProvider);provider.adapter=Adapter()
        args=(query,key,value);kwargs=dict(block_table=table,input_layout='TND')
        returned=provider._invoke_api('npu_fused_infer_attention_score',original,args,kwargs,
                                      dict(metadata_binding='VERIFIED'))
        self.assertIs(returned,result)
        self.assertIs(seen[0][0][0],query)
        self.assertIs(seen[0][1]['block_table'],table)
        self.assertIs(provider.adapter.actual['key'],key)
        self.assertEqual(len(seen),1)


if __name__=='__main__':unittest.main()
